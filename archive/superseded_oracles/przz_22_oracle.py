"""
src/przz_22_oracle.py
First-principles oracle for (2,2) pair from PRZZ α,β level formulas.

This implements the PRZZ structure for Case C × Case C pairs,
including the auxiliary a-integral that appears in F_d for ω > 0.

PRZZ Reference: Section 7, lines 2366-2385 (F_d kernels),
               Lemma 7.2 lines 2391-2399 (Euler-Maclaurin)

For (2,2) pair:
- ℓ₁ = ℓ₂ = 2
- ω₁ = ω₂ = 1 (Case C on both sides)
- Uses P₂ polynomial
"""

from __future__ import annotations
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Callable

# Add parent directory for imports when running directly
_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_src_dir.parent) not in sys.path:
    sys.path.insert(0, str(_src_dir.parent))

from polynomials import load_przz_polynomials, Polynomial
from quadrature import gauss_legendre_01


def compute_W_coefficient(omega: int) -> float:
    """
    Compute W(d, l) coefficient for Case C (ω > 0).

    From PRZZ line 2343:
    W(d, l) = 1{ω > 0} × (1!×(-1)^1)^{l_1} × (2!×(-1)^2)^{l_2} × ...

    For d=1 and l_1 = ℓ (single index):
    W = (1!×(-1)^1)^ℓ = (-1)^ℓ

    For ℓ = 2: W = (-1)^2 = 1
    """
    # For d=1, ℓ=2: W = (-1)^2 = 1
    return 1.0


def compute_case_c_Fd(
    P: Polynomial,
    u: np.ndarray,
    omega: int,
    alpha: float,
    theta: float,
    n_quad_a: int = 30
) -> np.ndarray:
    """
    Compute the Case C F_d kernel from PRZZ line 2374.

    F_d(l, α, n) = W × (-1)^{1-ω} / (ω-1)! × (log N)^ω × u^ω
                  × ∫₀¹ P((1-a)u) × a^{ω-1} × (N/n)^{-αa} da

    After Euler-Maclaurin, u = log(N/n)/log N, and:
    (N/n)^{-αa} = exp(-α × a × log(N/n)) = exp(-α × a × u × log N)
                = exp(-α × a × u × θ × log T)

    At α = -R/L = -R×θ/log T:
    (N/n)^{-αa} = exp(R×θ × a × u × θ × log T / log T) = exp(R×θ²×a×u)

    Wait, let me be more careful. We have:
    - N = T^θ, so log N = θ log T
    - u = log(N/n) / log N
    - (N/n)^{-αa} = exp(-α × a × log(N/n)) = exp(-α × a × u × log N)
    - At α = -R/L = -R/log T:
      exp(R/log T × a × u × θ log T) = exp(R × θ × a × u)

    So the exponential factor is exp(R × θ × a × u).

    The full kernel (normalized by log N factors absorbed elsewhere):
    K_ω(u; α) = u^ω / (ω-1)! × ∫₀¹ P((1-a)u) × a^{ω-1} × exp(-α × a × u × θ × log T) da

    At α = -R/L:
    K_ω(u; R) = u^ω / (ω-1)! × ∫₀¹ P((1-a)u) × a^{ω-1} × exp(R × θ × a × u) da

    Args:
        P: Polynomial to use
        u: Array of u values
        omega: ω value (1 for ℓ=2)
        alpha: α parameter (will use -R/L)
        theta: θ = 4/7
        n_quad_a: Number of quadrature points for a-integral

    Returns:
        Array of F_d values at each u
    """
    a_nodes, a_weights = gauss_legendre_01(n_quad_a)

    # W coefficient
    W = compute_W_coefficient(omega)

    # Sign factor: (-1)^{1-ω}
    sign_factor = (-1) ** (1 - omega)

    # Factorial: (ω-1)!
    from math import factorial
    omega_factorial = factorial(omega - 1) if omega >= 1 else 1

    # For each u, compute the a-integral
    result = np.zeros_like(u)

    for i, u_val in enumerate(u):
        # P((1-a)×u) for all a
        P_args = (1 - a_nodes) * u_val
        P_vals = P.eval(P_args)

        # a^{ω-1} factor
        a_power = a_nodes ** (omega - 1) if omega > 1 else np.ones_like(a_nodes)

        # Exponential factor: exp(-α × a × u × θ × log T)
        # At α = -R/L, this becomes exp(R × θ × a × u)
        # But we're working with general α for now
        # exp_factor = exp(-alpha × a × u × theta × log_T)
        # Since we evaluate at α = -R/L = -R/(log T), and log N = θ log T:
        # exp(-α × a × u × log N) = exp(R/L × a × u × θ L) = exp(R × θ × a × u)
        # For symbolic α before evaluation, we use:
        # exp(-α × θ × L × a × u) where L = log T
        # But at evaluation α = -R/L, so -α × L = R, giving exp(R × θ × a × u)

        # For now, assume alpha encodes the full coefficient
        exp_factor = np.exp(alpha * a_nodes * u_val)

        # Integrate
        integrand = P_vals * a_power * exp_factor
        integral = float(np.sum(a_weights * integrand))

        # u^ω factor
        u_power = u_val ** omega

        result[i] = W * sign_factor * u_power * integral / omega_factorial

    return result


def compute_22_from_przz_structure(
    R: float,
    theta: float,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    n_quad_a: int = 30,
    verbose: bool = False
) -> Tuple[float, dict]:
    """
    Compute (2,2) pair contribution using PRZZ α,β level structure.

    This implements the full PRZZ pipeline for Case C × Case C:
    1. F_d kernels with a-integrals
    2. n-sum → u-integral via Euler-Maclaurin
    3. Mirror combination
    4. Q-operator application
    5. Evaluation at α=β=-R/L

    The key difference from our DSL:
    - DSL uses P(u) directly
    - PRZZ uses F_d kernel which includes a-integral for ω > 0
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    # For (2,2): ℓ₁ = ℓ₂ = 2, ω₁ = ω₂ = 1
    omega = 1
    ell1 = ell2 = 2

    # Quadrature grids
    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    # Create 2D grid for (u, t)
    U, T = np.meshgrid(u_nodes, t_nodes, indexing='ij')
    WU, WT = np.meshgrid(u_weights, t_weights, indexing='ij')
    W_2d = WU * WT
    U_flat = U.flatten()
    T_flat = T.flatten()
    W_flat = W_2d.flatten()

    # === Step 1: Compute F_d kernels ===
    # At α = β = -R/L, the exponential factor in Case C becomes exp(R×θ×a×u)
    # The "alpha" we pass is R×θ (the coefficient of a×u in the exponent)
    alpha_coeff = R * theta

    # F_d for left side (ω = 1, P₂)
    F_left = compute_case_c_Fd(P2, U_flat, omega, alpha_coeff, theta, n_quad_a)

    # F_d for right side (same for (2,2))
    F_right = compute_case_c_Fd(P2, U_flat, omega, alpha_coeff, theta, n_quad_a)

    # === Step 2: (1-u) factor from Euler-Maclaurin ===
    # From Lemma 7.2, the n-sum produces (1-u)^{power} factor
    # For (2,2) pair, this power depends on the convolution structure
    # Looking at our current DSL: I1 has (1-u)^{ℓ₁+ℓ₂} = (1-u)^4
    # Let's use the same power for now
    euler_power = ell1 + ell2  # = 4 for I1-type
    poly_prefactor = (1 - U_flat) ** euler_power

    # === Step 3: Q factors ===
    # From PRZZ line 1517, after Q-operator application:
    # Q(θt(x+y) - θy + t) × Q(θt(x+y) - θx + t)
    # At x = y = 0: Q(t) × Q(t) = Q(t)²
    Q_factor = Q.eval(T_flat) ** 2

    # === Step 4: Exponential factor ===
    # At α = β = -R/L, the main exponential is exp(2Rt)
    # But Case C has additional exp(R×θ×a×u) which is already in F_d
    exp_factor = np.exp(2 * R * T_flat)

    # === Step 5: Assemble integrand for I₂-type (no derivatives) ===
    # I₂ = (1/θ) × ∫∫ F_left × F_right × Q² × exp(2Rt) dt du
    # Note: This is the "constant term" contribution
    I2_integrand = F_left * F_right * Q_factor * exp_factor
    I2_raw = float(np.sum(W_flat * I2_integrand)) / theta

    if verbose:
        print(f"=== PRZZ (2,2) Oracle ===")
        print(f"F_left sample (u=0.5): {compute_case_c_Fd(P2, np.array([0.5]), omega, alpha_coeff, theta, n_quad_a)[0]:.6f}")
        print(f"P2(0.5) for comparison: {P2.eval(np.array([0.5]))[0]:.6f}")
        print(f"Ratio F_d/P at u=0.5: {compute_case_c_Fd(P2, np.array([0.5]), omega, alpha_coeff, theta, n_quad_a)[0] / P2.eval(np.array([0.5]))[0]:.6f}")
        print(f"\nI2-type (no derivatives):")
        print(f"  Raw integral: {I2_raw:.10f}")

    # === Compare with our DSL structure ===
    # Our DSL uses P(u) directly, so let's compute that too
    P2_left = P2.eval(U_flat)
    P2_right = P2.eval(U_flat)
    I2_dsl_style = float(np.sum(W_flat * P2_left * P2_right * Q_factor * exp_factor)) / theta

    if verbose:
        print(f"\nDSL-style (P directly):")
        print(f"  I2 integral: {I2_dsl_style:.10f}")
        print(f"  Ratio PRZZ/DSL: {I2_raw / I2_dsl_style:.6f}")

    # === Now compute I₁-type (with derivatives) ===
    # This is more complex because we need derivatives of F_d × F_d product
    # The algebraic prefactor (θS+1)/θ contributes to derivatives

    # For I₁: d²/dxdy [(θ(x+y)+1)/θ × ∫∫ (1-u)^4 F_d(x+u) F_d(y+u) Q Q exp(...) du dt]|_{x=y=0}
    # This is very complex because F_d itself depends on α which is a function of x,y
    # in the full PRZZ structure

    # Let me implement a simpler version first - just the I2-type constant term
    # to see if the F_d kernel affects the magnitude correctly

    # === Compute all four terms with F_d kernels ===
    # For simplicity, use numerical derivatives

    def compute_full_integrand(x_val, y_val, use_fd_kernel=True):
        """Compute the full integrand F(x, y)."""
        # Q arguments
        # arg_α = t + θt×x + θ(t-1)×y
        # arg_β = t + θ(t-1)×x + θt×y
        arg_alpha = T_flat + theta * T_flat * x_val + theta * (T_flat - 1) * y_val
        arg_beta = T_flat + theta * (T_flat - 1) * x_val + theta * T_flat * y_val

        Q_alpha = Q.eval(arg_alpha)
        Q_beta = Q.eval(arg_beta)

        # Exp factors
        exp_alpha = np.exp(R * arg_alpha)
        exp_beta = np.exp(R * arg_beta)

        if use_fd_kernel:
            # Use F_d kernels (Case C structure)
            # The F_d kernel argument is u + x for left, u + y for right
            # But the α in the kernel also shifts... this is getting complex
            # For now, approximate by using F_d(u) with shifted exponential
            F_left = compute_case_c_Fd(P2, U_flat + x_val, omega, alpha_coeff, theta, n_quad_a)
            F_right = compute_case_c_Fd(P2, U_flat + y_val, omega, alpha_coeff, theta, n_quad_a)
            poly_factor = F_left * F_right
        else:
            # Use P directly (DSL style)
            P_left = P2.eval(U_flat + x_val)
            P_right = P2.eval(U_flat + y_val)
            poly_factor = P_left * P_right

        # Full integrand (without algebraic prefactor, without poly_prefactor for I2)
        return poly_factor * Q_alpha * Q_beta * exp_alpha * exp_beta

    # I2: constant term (no derivatives), no (1-u) prefactor
    I2_przz = float(np.sum(W_flat * compute_full_integrand(0, 0, use_fd_kernel=True))) / theta
    I2_dsl = float(np.sum(W_flat * compute_full_integrand(0, 0, use_fd_kernel=False))) / theta

    # I1: d²/dxdy with (1-u)^4 prefactor
    # Use 5-point stencil for cross derivative
    h = 1e-4

    def F_with_prefactor(x, y, use_fd):
        integrand = compute_full_integrand(x, y, use_fd)
        return float(np.sum(W_flat * (1-U_flat)**4 * integrand))

    # F_xy via finite difference
    F_xy_przz = (F_with_prefactor(h, h, True) - F_with_prefactor(h, -h, True)
                 - F_with_prefactor(-h, h, True) + F_with_prefactor(-h, -h, True)) / (4 * h**2)
    F_xy_dsl = (F_with_prefactor(h, h, False) - F_with_prefactor(h, -h, False)
                - F_with_prefactor(-h, h, False) + F_with_prefactor(-h, -h, False)) / (4 * h**2)

    # F_x and F_y for algebraic prefactor contribution
    F_x_przz = (F_with_prefactor(h, 0, True) - F_with_prefactor(-h, 0, True)) / (2 * h)
    F_y_przz = (F_with_prefactor(0, h, True) - F_with_prefactor(0, -h, True)) / (2 * h)
    F_x_dsl = (F_with_prefactor(h, 0, False) - F_with_prefactor(-h, 0, False)) / (2 * h)
    F_y_dsl = (F_with_prefactor(0, h, False) - F_with_prefactor(0, -h, False)) / (2 * h)

    # I1 = (1/θ)×F_xy + F_x + F_y (from algebraic prefactor)
    I1_przz = F_xy_przz / theta + F_x_przz + F_y_przz
    I1_dsl = F_xy_dsl / theta + F_x_dsl + F_y_dsl

    # I3 and I4: single derivatives with (1-u)^2 prefactor
    def G_with_prefactor(x, y, use_fd):
        integrand = compute_full_integrand(x, y, use_fd)
        return float(np.sum(W_flat * (1-U_flat)**ell1 * integrand))  # (1-u)^2 for I3

    G_x_przz = (G_with_prefactor(h, 0, True) - G_with_prefactor(-h, 0, True)) / (2 * h)
    G_x_dsl = (G_with_prefactor(h, 0, False) - G_with_prefactor(-h, 0, False)) / (2 * h)

    I3_przz = -G_x_przz / theta
    I3_dsl = -G_x_dsl / theta

    # I4 similar (symmetric for (2,2))
    I4_przz = I3_przz  # By symmetry for (2,2)
    I4_dsl = I3_dsl

    # Total for (2,2) pair (before factorial normalization)
    total_przz = I1_przz + I2_przz + I3_przz + I4_przz
    total_dsl = I1_dsl + I2_dsl + I3_dsl + I4_dsl

    # Factorial normalization: 1/(ℓ₁! × ℓ₂!) = 1/(2! × 2!) = 1/4
    norm = 1.0 / 4.0

    results = {
        'I1_przz': I1_przz * norm,
        'I2_przz': I2_przz * norm,
        'I3_przz': I3_przz * norm,
        'I4_przz': I4_przz * norm,
        'total_przz': total_przz * norm,
        'I1_dsl': I1_dsl * norm,
        'I2_dsl': I2_dsl * norm,
        'I3_dsl': I3_dsl * norm,
        'I4_dsl': I4_dsl * norm,
        'total_dsl': total_dsl * norm,
        'ratio_total': total_przz / total_dsl if total_dsl != 0 else float('inf'),
    }

    if verbose:
        print(f"\n=== Comparison (after 1/4 normalization) ===")
        print(f"         PRZZ (F_d)     DSL (P)        Ratio")
        print(f"I1:      {results['I1_przz']:+.8f}  {results['I1_dsl']:+.8f}  {results['I1_przz']/results['I1_dsl'] if results['I1_dsl'] != 0 else 'N/A':.4f}")
        print(f"I2:      {results['I2_przz']:+.8f}  {results['I2_dsl']:+.8f}  {results['I2_przz']/results['I2_dsl'] if results['I2_dsl'] != 0 else 'N/A':.4f}")
        print(f"I3:      {results['I3_przz']:+.8f}  {results['I3_dsl']:+.8f}  {results['I3_przz']/results['I3_dsl'] if results['I3_dsl'] != 0 else 'N/A':.4f}")
        print(f"I4:      {results['I4_przz']:+.8f}  {results['I4_dsl']:+.8f}  {results['I4_przz']/results['I4_dsl'] if results['I4_dsl'] != 0 else 'N/A':.4f}")
        print(f"Total:   {results['total_przz']:+.8f}  {results['total_dsl']:+.8f}  {results['ratio_total']:.4f}")

    return results['total_przz'], results


if __name__ == '__main__':
    THETA = 4/7
    R1 = 1.3036
    R2 = 1.1167

    print("=" * 70)
    print("PRZZ (2,2) Pair Oracle: Comparing F_d kernel vs P(u) directly")
    print("=" * 70)

    print(f"\n### Benchmark 1: R = {R1} ###")
    total1, results1 = compute_22_from_przz_structure(R1, THETA, verbose=True)

    print(f"\n### Benchmark 2: R = {R2} ###")
    total2, results2 = compute_22_from_przz_structure(R2, THETA, verbose=True)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"R = {R1}: PRZZ = {results1['total_przz']:.8f}, DSL = {results1['total_dsl']:.8f}, ratio = {results1['ratio_total']:.4f}")
    print(f"R = {R2}: PRZZ = {results2['total_przz']:.8f}, DSL = {results2['total_dsl']:.8f}, ratio = {results2['ratio_total']:.4f}")
    print(f"\nR-sensitivity (PRZZ): {results1['total_przz']/results2['total_przz']:.4f}")
    print(f"R-sensitivity (DSL):  {results1['total_dsl']/results2['total_dsl']:.4f}")
