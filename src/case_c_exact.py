"""
src/case_c_exact.py
Exact PRZZ Case C kernel implementation.

PRZZ TeX References:
- 2301-2310: omega definition + rescaling note at 2309
- 2360-2362: Case C polynomial-rescaling rewrite (key!)
- 2371-2374: Full Case C definition of F_d
- 2366-2368: Where it combines in main I_{1,d} sum
- 2382-2384: Product-side version with (log N)^omega and b-integral
- 2387-2388: "Only 6 cases" sanity check for K=3
- 624-635: N = T^theta normalization

KEY INSIGHT (from GPT):
The integrand is NOT just exp(R*k*a). It is:
    integral_0^1 P((1-a)*u) * a^{omega-1} * (N/n)^{-alpha*a} da

With (N/n)^{-alpha*a} = exp(R*theta*u*a) after substituting alpha=-R/L, N=T^theta.

The Case C kernel for polynomial P at quadrature point u is:
    K_omega(u; R) = integral_0^1 P((1-a)*u) * a^{omega-1} * exp(R*theta*u*a) da

Then:
- B x C: Replace Case C side with u^omega * K_omega(u; R)
- C x C: Compute K_omega_L(u; R) and K_omega_R(u; R) and multiply
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Callable, Any
import math

from src.polynomials import load_przz_polynomials, Polynomial
from src.quadrature import gauss_legendre_01


# Constants
THETA = 4/7
R_BENCHMARK1 = 1.3036
R_BENCHMARK2 = 1.1167
C_TARGET = 2.13745440613217263636
KAPPA_TARGET = 0.417293962


def compute_case_c_kernel(
    P: Polynomial,
    u: np.ndarray,
    omega: int,
    R: float,
    theta: float,
    n_quad_a: int = 30
) -> np.ndarray:
    """
    Compute exact Case C kernel K_omega(u; R) for polynomial P.

    PRZZ TeX 2360-2362, 2371-2374:
    K_omega(u; R) = integral_0^1 P((1-a)*u) * a^{omega-1} * exp(R*theta*u*a) da

    This is the key transformation that rescales the polynomial argument.

    Args:
        P: Polynomial object with eval() method
        u: Array of u-quadrature points
        omega: Case C parameter (omega = k-2, so P_2 -> omega=1, P_3 -> omega=2)
        R: Shift parameter
        theta: Mollifier exponent (typically 4/7)
        n_quad_a: Number of quadrature points for a-integral

    Returns:
        Array K_omega(u; R) for each u point
    """
    # Get quadrature nodes/weights for a-integral
    a_nodes, a_weights = gauss_legendre_01(n_quad_a)

    # K will have shape of u
    K = np.zeros_like(u)

    # For each u point, compute the 1D integral in a
    for i, u_val in enumerate(u.flat):
        # Polynomial argument: (1-a)*u for each a
        poly_arg = (1 - a_nodes) * u_val

        # P((1-a)*u)
        P_vals = P.eval(poly_arg)

        # a^{omega-1} weight
        if omega == 1:
            a_weight = np.ones_like(a_nodes)  # a^0 = 1
        elif omega == 2:
            a_weight = a_nodes  # a^1 = a
        else:
            a_weight = a_nodes ** (omega - 1)

        # exp(R*theta*u*a) - note the u factor!
        exp_factor = np.exp(R * theta * u_val * a_nodes)

        # Integrand: P((1-a)*u) * a^{omega-1} * exp(R*theta*u*a)
        integrand = P_vals * a_weight * exp_factor

        # Integrate
        K.flat[i] = np.sum(a_weights * integrand)

    return K


def compute_case_c_kernel_vectorized(
    P: Polynomial,
    u: np.ndarray,
    omega: int,
    R: float,
    theta: float,
    n_quad_a: int = 30
) -> np.ndarray:
    """
    Vectorized version of Case C kernel computation.

    For efficiency, computes all u points at once using broadcasting.

    Args:
        P: Polynomial object
        u: Array of u-quadrature points (shape: (n_u,) or (n_u, n_t))
        omega: Case C parameter
        R: Shift parameter
        theta: Mollifier exponent
        n_quad_a: Quadrature points for a

    Returns:
        K_omega(u; R) array with same shape as u
    """
    a_nodes, a_weights = gauss_legendre_01(n_quad_a)

    # Reshape for broadcasting: u has shape (n_u,) or (n_u, n_t), a has shape (n_a,)
    # We want to compute for all (u, a) pairs
    original_shape = u.shape
    u_flat = u.flatten()
    n_u = len(u_flat)
    n_a = len(a_nodes)

    # Create meshgrid: u_grid[i,j] = u_flat[i], a_grid[i,j] = a_nodes[j]
    u_grid = np.tile(u_flat[:, np.newaxis], (1, n_a))  # shape (n_u, n_a)
    a_grid = np.tile(a_nodes[np.newaxis, :], (n_u, 1))  # shape (n_u, n_a)

    # Polynomial argument: (1-a)*u
    poly_arg = (1 - a_grid) * u_grid  # shape (n_u, n_a)

    # Evaluate P at all arguments - need to handle 2D input
    poly_arg_flat = poly_arg.flatten()
    P_vals_flat = P.eval(poly_arg_flat)
    P_vals = P_vals_flat.reshape(n_u, n_a)

    # a^{omega-1} weight
    if omega == 1:
        a_weight = np.ones(n_a)
    elif omega == 2:
        a_weight = a_nodes
    else:
        a_weight = a_nodes ** (omega - 1)

    # exp(R*theta*u*a)
    exp_factor = np.exp(R * theta * u_grid * a_grid)

    # Integrand
    integrand = P_vals * a_weight[np.newaxis, :] * exp_factor

    # Integrate over a (axis 1)
    K_flat = np.sum(a_weights[np.newaxis, :] * integrand, axis=1)

    return K_flat.reshape(original_shape)


def test_kernel_implementation(verbose: bool = True) -> Dict[str, Any]:
    """
    Test Case C kernel implementation against known properties.

    Properties to verify:
    1. When R=0, exp factor is 1, so K_omega should equal integral of P((1-a)*u)*a^{omega-1}
    2. For P=1 (constant), K_omega should equal 1/omega (the Beta function value)
    3. Vectorized and non-vectorized should match
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = THETA
    R = R_BENCHMARK1

    # Test points
    u_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    results = {}

    # Test 1: Vectorized vs non-vectorized match
    K1_vec = compute_case_c_kernel_vectorized(P1, u_test, omega=0, R=R, theta=theta)
    K1_loop = compute_case_c_kernel(P1, u_test, omega=1, R=R, theta=theta)

    # Note: omega=0 is Case B (shouldn't use this function), omega=1 is P_2's case
    K2_vec = compute_case_c_kernel_vectorized(P2, u_test, omega=1, R=R, theta=theta)
    K2_loop = compute_case_c_kernel(P2, u_test, omega=1, R=R, theta=theta)

    match_2 = np.allclose(K2_vec, K2_loop, rtol=1e-10)
    results["vectorized_matches_loop"] = match_2

    # Test 2: For constant P=1, K_omega(u; R=0) = 1/omega
    # When R=0, exp=1, so integral_0^1 1 * a^{omega-1} da = 1/omega
    from src.polynomials import Polynomial
    P_const = Polynomial([1.0])  # P(x) = 1

    K_const_omega1 = compute_case_c_kernel_vectorized(P_const, u_test, omega=1, R=0, theta=theta)
    K_const_omega2 = compute_case_c_kernel_vectorized(P_const, u_test, omega=2, R=0, theta=theta)

    expected_omega1 = 1.0 / 1  # = 1
    expected_omega2 = 1.0 / 2  # = 0.5

    match_const_1 = np.allclose(K_const_omega1, expected_omega1, rtol=1e-10)
    match_const_2 = np.allclose(K_const_omega2, expected_omega2, rtol=1e-10)
    results["constant_P_omega1"] = match_const_1
    results["constant_P_omega2"] = match_const_2

    if verbose:
        print("\n" + "=" * 60)
        print("CASE C KERNEL IMPLEMENTATION TEST")
        print("=" * 60)

        print("\n--- Test 1: Vectorized vs Loop Match ---")
        print(f"  K_2 (omega=1) vectorized matches loop: {match_2}")
        if not match_2:
            print(f"    Max diff: {np.max(np.abs(K2_vec - K2_loop))}")

        print("\n--- Test 2: Constant P, R=0 ---")
        print(f"  K_omega=1 should be 1.0: {K_const_omega1[0]:.10f} (match: {match_const_1})")
        print(f"  K_omega=2 should be 0.5: {K_const_omega2[0]:.10f} (match: {match_const_2})")

        print("\n--- Kernel Values for PRZZ Polynomials ---")
        print(f"  u = {u_test}")
        print(f"  K_2(u; R={R}): {K2_vec}")

        K3_vec = compute_case_c_kernel_vectorized(P3, u_test, omega=2, R=R, theta=theta)
        print(f"  K_3(u; R={R}): {K3_vec}")

        print("=" * 60)

    return results


def compute_i1_with_exact_case_c(
    ell1: int,
    ell2: int,
    R: float,
    theta: float = THETA,
    n_quad: int = 60,
    n_quad_a: int = 30
) -> float:
    """
    Compute I_1 for pair (ell1, ell2) with exact Case C treatment.

    PRZZ structure (TeX 2366-2368):
    - For B x B: No Case C modification
    - For B x C: Replace P_right with u^omega * K_omega(u; R)
    - For C x C: Replace both P factors with their kernels and multiply

    For simplicity, this computes the CHANGE from the raw formula,
    not the full I_1 (which requires all the derivative/series machinery).

    Actually, to properly incorporate Case C, we need to modify
    how the polynomial factors are evaluated in the main integrand.

    Args:
        ell1, ell2: Pair indices (1, 2, or 3)
        R: Shift parameter
        theta: Mollifier exponent
        n_quad: Main quadrature points
        n_quad_a: Case C a-integral quadrature points

    Returns:
        I_1 contribution (placeholder - full implementation needs more work)
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    # Determine cases
    def get_omega(ell):
        # Our P_1 -> k=2 -> omega = 0 (Case B)
        # Our P_2 -> k=3 -> omega = 1 (Case C)
        # Our P_3 -> k=4 -> omega = 2 (Case C)
        return ell - 1

    omega1 = get_omega(ell1)
    omega2 = get_omega(ell2)

    case1 = "B" if omega1 == 0 else "C"
    case2 = "B" if omega2 == 0 else "C"

    # Get polynomials
    polys = {1: P1, 2: P2, 3: P3}
    P_left = polys[ell1]
    P_right = polys[ell2]

    # For now, compute a diagnostic: the ratio of "with Case C" to "without"
    # This will help validate the structure before full implementation

    U_nodes, U_weights = gauss_legendre_01(n_quad)

    if case1 == "B" and case2 == "B":
        # No Case C: just P(u) * P(u)
        integrand_raw = P_left.eval(U_nodes) * P_right.eval(U_nodes)
        integrand_case_c = integrand_raw  # Same
        ratio = 1.0

    elif case1 == "B" and case2 == "C":
        # B x C: Replace P_right(u) with u^omega * K_omega(u; R)
        # Per PRZZ 2382-2384, there's a u^omega factor

        P_left_vals = P_left.eval(U_nodes)

        # Raw: P_left(u) * P_right(u)
        P_right_vals_raw = P_right.eval(U_nodes)
        integrand_raw = P_left_vals * P_right_vals_raw

        # Case C: P_left(u) * [u^omega * K_omega(u; R)]
        K_right = compute_case_c_kernel_vectorized(P_right, U_nodes, omega2, R, theta, n_quad_a)
        u_power = U_nodes ** omega2
        integrand_case_c = P_left_vals * u_power * K_right

        ratio = np.sum(U_weights * integrand_case_c) / np.sum(U_weights * integrand_raw)

    elif case1 == "C" and case2 == "C":
        # C x C: Both get replaced
        # The a and b integrals factor (PRZZ 2382-2384)

        # Raw: P_left(u) * P_right(u)
        P_left_vals_raw = P_left.eval(U_nodes)
        P_right_vals_raw = P_right.eval(U_nodes)
        integrand_raw = P_left_vals_raw * P_right_vals_raw

        # Case C: [u^omega1 * K_omega1(u; R)] * [u^omega2 * K_omega2(u; R)]
        K_left = compute_case_c_kernel_vectorized(P_left, U_nodes, omega1, R, theta, n_quad_a)
        K_right = compute_case_c_kernel_vectorized(P_right, U_nodes, omega2, R, theta, n_quad_a)
        u_power_left = U_nodes ** omega1
        u_power_right = U_nodes ** omega2

        integrand_case_c = (u_power_left * K_left) * (u_power_right * K_right)

        ratio = np.sum(U_weights * integrand_case_c) / np.sum(U_weights * integrand_raw)

    else:
        raise ValueError(f"Invalid case combination: {case1} x {case2}")

    return ratio


def diagnose_case_c_ratios(verbose: bool = True) -> Dict[str, Any]:
    """
    Compute Case C correction ratios for all pairs.

    This shows the multiplicative correction factor from implementing
    exact Case C kernel vs raw polynomial evaluation.
    """
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    R_values = [R_BENCHMARK1, R_BENCHMARK2]

    results = {}

    for R in R_values:
        results[f"R={R}"] = {}
        for ell1, ell2 in pairs:
            ratio = compute_i1_with_exact_case_c(ell1, ell2, R, THETA)
            results[f"R={R}"][f"{ell1}{ell2}"] = ratio

    if verbose:
        print("\n" + "=" * 70)
        print("EXACT CASE C CORRECTION RATIOS")
        print("=" * 70)
        print("\nThese show: integral(Case C treatment) / integral(raw P(u))")
        print("For B x B, ratio = 1 (no change)")
        print("For B x C and C x C, ratio shows the effect of Case C kernel")
        print()

        print(f"{'Pair':<8} {'Cases':<8} {'R=1.3036':>12} {'R=1.1167':>12} {'Delta':>10}")
        print("-" * 60)

        for ell1, ell2 in pairs:
            omega1 = ell1 - 1
            omega2 = ell2 - 1
            case1 = "B" if omega1 == 0 else "C"
            case2 = "B" if omega2 == 0 else "C"

            r1 = results[f"R={R_BENCHMARK1}"][f"{ell1}{ell2}"]
            r2 = results[f"R={R_BENCHMARK2}"][f"{ell1}{ell2}"]
            delta = (r1 - r2) / r2 * 100 if r2 != 0 else 0

            print(f"({ell1},{ell2})    {case1}x{case2:<6} {r1:>12.6f} {r2:>12.6f} {delta:>+9.2f}%")

        print("=" * 70)

    return results


if __name__ == "__main__":
    # Test implementation
    test_kernel_implementation(verbose=True)

    # Compute correction ratios
    diagnose_case_c_ratios(verbose=True)
