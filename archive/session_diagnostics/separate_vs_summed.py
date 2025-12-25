"""
src/separate_vs_summed.py
Diagnostic: Summed vs Separate Variable Interpretations

PRZZ TeX References:
- Lines 1233-1236: Shows separate contour measures dz_i/z_i^2
- Lines 1245-1249: Shows products over i=1..ell_1, j=1..ell_2

Key Mathematical Question:
For (2,2) pair, does the polynomial structure give P_2''(u)^2 (summed)
or something involving P_2'(u)^4 (separate)?

Summed interpretation:
    d^4/dx1 dx2 dy1 dy2 [P_2(x1+x2+u) * P_2(y1+y2+u)]|_{x=y=0}
    = P_2''(u) * P_2''(u)

Separate interpretation:
    If each Mellin variable contributes independently via residues,
    we might have product structure like [P_2'(u)]^4 instead.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

from src.polynomials import load_przz_polynomials
from src.quadrature import tensor_grid_2d


def compute_summed_interpretation(
    theta: float = 4/7,
    R: float = 1.3036,
    n_quad: int = 60
) -> float:
    """
    Compute (2,2) I1 using SUMMED interpretation: P_2''(u)^2.

    This is our current implementation.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    # Get P_2 second derivative coefficients
    P2_mono = P2.to_monomial()
    coeffs = P2_mono.coeffs

    P2_dd_coeffs = []
    for i, c in enumerate(coeffs):
        if i >= 2:
            P2_dd_coeffs.append(i * (i-1) * c)

    def P2_second_deriv(u):
        """P_2''(u)"""
        result = 0.0
        for i, c in enumerate(P2_dd_coeffs):
            result += c * (u ** i)
        return result

    # Quadrature
    U, T, W = tensor_grid_2d(n_quad)

    # Q arguments at x=y=0
    Q_alpha = Q.eval(T)
    Q_beta = Q.eval(T)

    # Exp factors at x=y=0
    exp_factor = np.exp(2 * R * T)

    # Polynomial: P_2''(u)^2
    P2_dd_u = np.array([P2_second_deriv(u) for u in U.flatten()]).reshape(U.shape)
    poly_factor = P2_dd_u ** 2

    # (1-u)^4 prefactor
    poly_pref = (1 - U) ** 4

    # Q^2 factor
    Q_factor = Q_alpha * Q_beta

    # Algebraic prefactor at x=y=0: (theta*0 + 1)/theta = 1/theta
    alg_pref = 1.0 / theta

    # Full integrand (without alg_pref cross-terms since x=y=0)
    integrand = poly_pref * poly_factor * Q_factor * exp_factor

    return alg_pref * float(np.sum(W * integrand))


def compute_separate_interpretation(
    theta: float = 4/7,
    R: float = 1.3036,
    n_quad: int = 60
) -> float:
    """
    Compute (2,2) I1 using SEPARATE interpretation: [P_2'(u)]^4.

    In this interpretation, each of the 4 Mellin variables contributes
    one factor of P_2'(u) rather than combining into P_2''(u)^2.

    This tests the hypothesis from PRZZ lines 1233-1249.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    # Get P_2 first derivative coefficients
    P2_mono = P2.to_monomial()
    coeffs = P2_mono.coeffs

    P2_d_coeffs = []
    for i, c in enumerate(coeffs):
        if i >= 1:
            P2_d_coeffs.append(i * c)

    def P2_first_deriv(u):
        """P_2'(u)"""
        result = 0.0
        for i, c in enumerate(P2_d_coeffs):
            result += c * (u ** i)
        return result

    # Quadrature
    U, T, W = tensor_grid_2d(n_quad)

    # Q arguments at x=y=0
    Q_alpha = Q.eval(T)
    Q_beta = Q.eval(T)

    # Exp factors at x=y=0
    exp_factor = np.exp(2 * R * T)

    # Polynomial: [P_2'(u)]^4
    P2_d_u = np.array([P2_first_deriv(u) for u in U.flatten()]).reshape(U.shape)
    poly_factor = P2_d_u ** 4

    # (1-u)^4 prefactor
    poly_pref = (1 - U) ** 4

    # Q^2 factor
    Q_factor = Q_alpha * Q_beta

    # Algebraic prefactor at x=y=0: 1/theta
    alg_pref = 1.0 / theta

    # Full integrand
    integrand = poly_pref * poly_factor * Q_factor * exp_factor

    return alg_pref * float(np.sum(W * integrand))


def compute_hybrid_interpretation(
    theta: float = 4/7,
    R: float = 1.3036,
    n_quad: int = 60
) -> float:
    """
    Hybrid: [P_2'(u)]^2 * [P_2'(u)]^2 = [P_2'(u)]^4.

    This is equivalent to separate, but tests if the structure might be
    [P_2'(u)]^2 on each "side" (left from ζ, right from ζ̄).
    """
    # Same as separate for (2,2)
    return compute_separate_interpretation(theta, R, n_quad)


def compare_interpretations(
    theta: float = 4/7,
    R: float = 1.3036,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict:
    """
    Compare summed vs separate interpretations for (2,2).

    Returns dict with both values and their ratio.
    """
    summed = compute_summed_interpretation(theta, R, n_quad)
    separate = compute_separate_interpretation(theta, R, n_quad)

    # Get actual DSL value for comparison
    from src.terms_k3_d1 import make_I1_22
    from src.evaluate import evaluate_term

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    term = make_I1_22(theta, R)
    dsl_result = evaluate_term(term, polys, n_quad)
    dsl_value = dsl_result.value

    results = {
        "summed": summed,
        "separate": separate,
        "dsl": dsl_value,
        "ratio_summed_to_separate": summed / separate if abs(separate) > 1e-15 else float('inf'),
        "ratio_dsl_to_summed": dsl_value / summed if abs(summed) > 1e-15 else float('inf'),
        "ratio_dsl_to_separate": dsl_value / separate if abs(separate) > 1e-15 else float('inf'),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMED vs SEPARATE INTERPRETATION COMPARISON: (2,2) I1")
        print("=" * 70)
        print(f"\nConfig: theta = {theta:.10f}, R = {R}, n = {n_quad}")

        print("\n--- Interpretation Values ---")
        print(f"  Summed [P_2''(u)]^2:     {summed:+.12f}")
        print(f"  Separate [P_2'(u)]^4:    {separate:+.12f}")
        print(f"  DSL (current impl):      {dsl_value:+.12f}")

        print("\n--- Ratios ---")
        print(f"  Summed / Separate:       {results['ratio_summed_to_separate']:.6f}")
        print(f"  DSL / Summed:            {results['ratio_dsl_to_summed']:.6f}")
        print(f"  DSL / Separate:          {results['ratio_dsl_to_separate']:.6f}")

        print("\n--- Analysis ---")
        if abs(results['ratio_dsl_to_summed'] - 1.0) < 0.01:
            print("  DSL matches SUMMED interpretation")
            if results['ratio_summed_to_separate'] > 1.05:
                print(f"  Summed is {(results['ratio_summed_to_separate']-1)*100:.1f}% larger than separate")
            elif results['ratio_summed_to_separate'] < 0.95:
                print(f"  Summed is {(1-results['ratio_summed_to_separate'])*100:.1f}% smaller than separate")
        elif abs(results['ratio_dsl_to_separate'] - 1.0) < 0.01:
            print("  DSL matches SEPARATE interpretation")
        else:
            print("  DSL matches NEITHER interpretation directly")
            print("  Need to check for additional factors")

        print("=" * 70)

    return results


def theoretical_ratio_P2dd_to_P2d4():
    """
    Compute the theoretical ratio of integral of P_2''(u)^2 to [P_2'(u)]^4.

    This helps understand the structural difference.
    """
    from src.quadrature import gauss_legendre_01

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    # Get derivative coefficients
    P2_mono = P2.to_monomial()
    coeffs = P2_mono.coeffs

    # P_2'
    P2_d_coeffs = []
    for i, c in enumerate(coeffs):
        if i >= 1:
            P2_d_coeffs.append(i * c)

    # P_2''
    P2_dd_coeffs = []
    for i, c in enumerate(coeffs):
        if i >= 2:
            P2_dd_coeffs.append(i * (i-1) * c)

    def P2_d(u):
        result = 0.0
        for i, c in enumerate(P2_d_coeffs):
            result += c * (u ** i)
        return result

    def P2_dd(u):
        result = 0.0
        for i, c in enumerate(P2_dd_coeffs):
            result += c * (u ** i)
        return result

    # Integrate over [0,1] with (1-u)^4 weight using Gauss-Legendre
    x, w = gauss_legendre_01(100)

    def integrand_summed(u):
        return (1-u)**4 * P2_dd(u)**2

    def integrand_separate(u):
        return (1-u)**4 * P2_d(u)**4

    val_summed = sum(w[i] * integrand_summed(x[i]) for i in range(len(x)))
    val_separate = sum(w[i] * integrand_separate(x[i]) for i in range(len(x)))

    print("\n" + "=" * 70)
    print("THEORETICAL RATIO ANALYSIS (polynomial part only)")
    print("=" * 70)
    print(f"\nIntegral of (1-u)^4 * [P_2''(u)]^2:  {val_summed:.10f}")
    print(f"Integral of (1-u)^4 * [P_2'(u)]^4:   {val_separate:.10f}")
    print(f"Ratio summed/separate: {val_summed/val_separate:.6f}")

    # Show polynomial structure
    print(f"\nP_2 coefficients: {coeffs}")
    print(f"P_2' coefficients: {P2_d_coeffs}")
    print(f"P_2'' coefficients: {P2_dd_coeffs}")

    # Sample values
    test_u = [0.0, 0.25, 0.5, 0.75]
    print("\nSample values:")
    for u in test_u:
        print(f"  u={u}: P_2'({u})={P2_d(u):.6f}, P_2''({u})={P2_dd(u):.6f}")

    print("=" * 70)

    return val_summed / val_separate


if __name__ == "__main__":
    # First show polynomial structure
    theoretical_ratio_P2dd_to_P2d4()

    # Then compare full integrals
    compare_interpretations(verbose=True)
