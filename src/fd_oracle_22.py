"""
src/fd_oracle_22.py
Finite-Difference Oracle for (2,2) Pair Validation

This module validates the (2,2) term structure by comparing:
1. DSL-computed I1_22 value
2. FD-computed fourth derivative of the integrand

Key structural question: Does PRZZ use SUMMED arguments P₂(x1+x2+u)
or SEPARATE arguments for each variable?

If summed: ∂²/∂x1∂x2[P₂(x1+x2+u)] = P₂''(u)
           ∂⁴/∂x1∂x2∂y1∂y2[P₂(...)P₂(...)] = P₂''(u)² at x=y=0

PRZZ Reference: Lines 1636+, 1714-1727
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Callable, Tuple

from src.quadrature import tensor_grid_2d
from src.polynomials import load_przz_polynomials


def compute_F_I1_22_summed(
    x1: float, x2: float, y1: float, y2: float,
    P2, Q,
    theta: float, R: float,
    n_quad: int
) -> float:
    """
    Compute I1_22 integrand with SUMMED arguments.

    F(x1,x2,y1,y2) = ∫∫ (1-u)^4 × P₂(x1+x2+u) × P₂(y1+y2+u)
                     × Q(Arg_α) × Q(Arg_β) × exp(R·Arg_α) × exp(R·Arg_β) du dt

    Without derivative extraction or algebraic prefactor.

    Args:
        x1, x2, y1, y2: Formal variable values
        P2: P₂ polynomial
        Q: Q polynomial
        theta: θ parameter
        R: R parameter
        n_quad: Quadrature points per dimension

    Returns:
        Value of F(x1, x2, y1, y2)
    """
    U, T, W = tensor_grid_2d(n_quad)

    # Total sum for Q/exp arguments
    S = x1 + x2 + y1 + y2
    X = x1 + x2
    Y = y1 + y2

    # Q arguments (general form)
    # Arg_α = t + θtS - θY = t(1 + θS) - θY
    # Arg_β = t + θtS - θX = t(1 + θS) - θX
    Arg_alpha = T * (1 + theta * S) - theta * Y
    Arg_beta = T * (1 + theta * S) - theta * X

    # Polynomial factors (SUMMED interpretation)
    P2_left = P2.eval(X + U)   # P₂(x1+x2+u)
    P2_right = P2.eval(Y + U)  # P₂(y1+y2+u)

    Q_alpha = Q.eval(Arg_alpha)
    Q_beta = Q.eval(Arg_beta)

    # Exponential factors
    exp_alpha = np.exp(R * Arg_alpha)
    exp_beta = np.exp(R * Arg_beta)

    # Polynomial prefactor (1-u)^4 for (2,2)
    poly_pref = (1 - U) ** 4

    # Full integrand
    integrand = poly_pref * P2_left * P2_right * Q_alpha * Q_beta * exp_alpha * exp_beta

    return float(np.sum(W * integrand))


def compute_fourth_derivative_fd(
    F: Callable[[float, float, float, float], float],
    h: float = 1e-3
) -> float:
    """
    Compute ∂⁴/∂x1∂x2∂y1∂y2 F(0,0,0,0) using finite differences.

    Uses a 16-point stencil for mixed fourth derivative.

    ∂⁴F/∂x1∂x2∂y1∂y2 ≈ (1/16h⁴) × sum of F(±h,±h,±h,±h) with appropriate signs
    """
    total = 0.0
    for sx1 in [-1, 1]:
        for sx2 in [-1, 1]:
            for sy1 in [-1, 1]:
                for sy2 in [-1, 1]:
                    sign = sx1 * sx2 * sy1 * sy2
                    total += sign * F(sx1*h, sx2*h, sy1*h, sy2*h)

    return total / (16 * h**4)


def richardson_fourth_derivative(
    F: Callable[[float, float, float, float], float],
    h0: float = 1e-2,
    order: int = 4
) -> Tuple[float, np.ndarray]:
    """
    Compute fourth derivative using Richardson extrapolation.
    """
    estimates = []

    for i in range(order):
        h = h0 / (2 ** i)
        fd = compute_fourth_derivative_fd(F, h)
        estimates.append(fd)

    estimates = np.array(estimates)

    # Richardson extrapolation (error is O(h²) for our stencil)
    R_table = np.zeros((order, order))
    R_table[:, 0] = estimates

    for j in range(1, order):
        for i in range(j, order):
            R_table[i, j] = R_table[i, j-1] + (R_table[i, j-1] - R_table[i-1, j-1]) / (4**j - 1)

    return R_table[order-1, order-1], estimates


def validate_I1_22_structure(
    theta: float = 4/7,
    R: float = 1.3036,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict:
    """
    Validate I1_22 structure by comparing DSL to FD.

    Tests whether the SUMMED argument interpretation P₂(x1+x2+u) is correct.
    """
    from src.terms_k3_d1 import make_I1_22
    from src.evaluate import evaluate_term

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Get DSL value
    term = make_I1_22(theta, R)
    dsl_result = evaluate_term(term, polys, n_quad)
    dsl_value = dsl_result.value

    # Build F function for FD
    def F(x1, x2, y1, y2):
        return compute_F_I1_22_summed(x1, x2, y1, y2, P2, Q, theta, R, n_quad)

    # Compute FD derivative
    fd_deriv, raw_estimates = richardson_fourth_derivative(F)

    # The DSL includes algebraic prefactor (θS+1)/θ evaluated at S=0 → 1/θ
    # So DSL_value = (1/θ) × ∂⁴F/∂x1∂x2∂y1∂y2
    fd_with_prefactor = (1.0 / theta) * fd_deriv

    # Compute relative error
    if abs(dsl_value) > 1e-15:
        rel_err = abs(dsl_value - fd_with_prefactor) / abs(dsl_value)
    else:
        rel_err = abs(dsl_value - fd_with_prefactor)

    results = {
        "dsl_value": dsl_value,
        "fd_derivative": fd_deriv,
        "fd_with_prefactor": fd_with_prefactor,
        "rel_error": rel_err,
        "raw_estimates": raw_estimates,
        "theta": theta,
        "R": R,
        "n_quad": n_quad,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("FD ORACLE: I1_22 STRUCTURE VALIDATION")
        print("=" * 70)
        print(f"\nConfig: θ = {theta:.10f}, R = {R}, n = {n_quad}")
        print(f"\nStructure being tested: SUMMED arguments P₂(x1+x2+u)")

        print(f"\n--- Derivative Estimates ---")
        print(f"  Raw FD estimates (h halving):")
        for i, est in enumerate(raw_estimates):
            h = 1e-2 / (2**i)
            print(f"    h = {h:.0e}: {est:+.12f}")
        print(f"  Richardson extrapolated: {fd_deriv:+.12f}")

        print(f"\n--- DSL vs FD Comparison ---")
        print(f"  DSL value:              {dsl_value:+.12f}")
        print(f"  FD × (1/θ):             {fd_with_prefactor:+.12f}")

        print(f"\n--- Relative Error ---")
        print(f"  |DSL - FD×(1/θ)| / |DSL| = {rel_err:.6e}")

        print(f"\n--- Verdict ---")
        if rel_err < 1e-3:
            print(f"  ✓ SUMMED argument structure VALIDATED")
            print(f"    DSL matches FD with algebraic prefactor 1/θ")
            results["verdict"] = "summed_validated"
        elif rel_err < 0.1:
            print(f"  ⚠️  Moderate agreement (rel_err = {rel_err:.2%})")
            print(f"    Structure may be correct but with numerical issues")
            results["verdict"] = "moderate_agreement"
        else:
            print(f"  ❌ SUMMED argument structure NOT validated")
            print(f"    DSL does not match FD - structure may be wrong")
            results["verdict"] = "not_validated"

        print("\n" + "=" * 70)

    return results


def compare_summed_vs_P2_double_prime(
    theta: float = 4/7,
    R: float = 1.3036,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict:
    """
    Test the key prediction: with summed arguments,
    ∂²/∂x1∂x2[P₂(x1+x2+u)] = P₂''(u)

    This verifies that the polynomial contribution factorizes.
    """
    P1, P2, P3, Q = load_przz_polynomials()

    # Compute P₂''(x) analytically
    # If P₂(x) = Σ aᵢ xⁱ, then P₂''(x) = Σ i(i-1)aᵢ xⁱ⁻²
    # Need to convert to monomial form first
    P2_mono = P2.to_monomial()
    coeffs = P2_mono.coeffs
    P2_dd_coeffs = []
    for i, c in enumerate(coeffs):
        if i >= 2:
            P2_dd_coeffs.append(i * (i-1) * c)

    def eval_P2_dd(u):
        """Evaluate P₂''(u)"""
        result = 0.0
        for i, c in enumerate(P2_dd_coeffs):
            result += c * (u ** i)
        return result

    def P2_summed(x1, x2, u):
        """P₂(x1 + x2 + u)"""
        return P2.eval(x1 + x2 + u)

    def numerical_d2_dx1dx2(x1, x2, u, h=1e-5):
        """Numerical ∂²/∂x1∂x2 using central differences"""
        return (P2_summed(x1+h, x2+h, u) - P2_summed(x1+h, x2-h, u)
                - P2_summed(x1-h, x2+h, u) + P2_summed(x1-h, x2-h, u)) / (4*h**2)

    test_u = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {"test_points": [], "P2_dd_coeffs": P2_dd_coeffs}

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMED ARGUMENT TEST: ∂²/∂x1∂x2[P₂(x1+x2+u)] vs P₂''(u)")
        print("=" * 70)
        print(f"\nP₂ coefficients: {coeffs}")
        print(f"P₂'' coefficients: {P2_dd_coeffs}")
        P2_label = 'P₂"(u)'
        print(f"\n{'u':>6} | {'FD':>15} | {P2_label:>15} | {'diff':>12} | {'match':>5}")
        print("-" * 60)

    all_match = True
    for u in test_u:
        fd_val = numerical_d2_dx1dx2(0, 0, u)
        analytic_val = eval_P2_dd(u)
        diff = abs(fd_val - analytic_val)
        match = diff < 1e-6
        if not match:
            all_match = False

        results["test_points"].append({
            "u": u,
            "fd": fd_val,
            "analytic": analytic_val,
            "diff": diff,
            "match": match
        })

        if verbose:
            match_str = "✓" if match else "✗"
            print(f"{u:>6.2f} | {fd_val:>+15.10f} | {analytic_val:>+15.10f} | {diff:>12.2e} | {match_str:>5}")

    results["all_match"] = all_match

    if verbose:
        print()
        if all_match:
            print("✓ CONFIRMED: ∂²/∂x1∂x2[P₂(x1+x2+u)] = P₂''(u)")
            print("  The summed argument interpretation is mathematically correct.")
        else:
            print("✗ FAILED: Some test points don't match")
        print("=" * 70)

    return results


if __name__ == "__main__":
    # Test polynomial structure first
    compare_summed_vs_P2_double_prime(verbose=True)

    # Then validate full I1_22
    validate_I1_22_structure(verbose=True)
