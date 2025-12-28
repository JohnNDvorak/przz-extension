#!/usr/bin/env python3
"""
scripts/derive_g_i1_epsilon.py
Derive epsilon_I1 from Q-derivative analysis

The θ(2-θ) formula gives g_I2 = 1.01943635 (within 0.0015% of calibrated).
But g_I1 = 1.0 still has a 0.09% gap from calibrated g_I1 = 1.00091.

This script investigates whether the Q-derivative contribution can explain
the epsilon_I1 correction.

KEY INSIGHT from GPT:
- I1 has Q evaluated at affine arguments A_alpha(x,y,t), A_beta(x,y,t)
- The [x] and [y] coefficients of Q(A)Q(A) involve Q'
- These affect the ratio (F_x + F_y) / F_xy which determines g_internal

The formula for [x] coefficient of Q(A_alpha)Q(A_beta):
  [x] = Q(t)Q'(t) × (a_alpha + a_beta) = Q(t)Q'(t) × θ(2t-1)

Similarly [y] = Q(t)Q'(t) × θ(2t-1)

The ratio ([x] + [y]) / [xy] from Q-derivatives might give epsilon_I1.

Created: 2025-12-27
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star, Polynomial
from src.quadrature import gauss_legendre_01


def compute_q_coefficient_structure(Q: Polynomial, t: float, theta: float):
    """
    Compute the nilpotent series coefficients of Q(A_alpha)Q(A_beta).

    For post-identity eigenvalues:
        A_alpha = t + θ(t-1)x + θt·y
        A_beta  = t + θt·x + θ(t-1)y

    Returns:
        Dict with [1], [x], [y], [xy] coefficients from Q
    """
    t_arr = np.array([t])

    Q_val = Q.eval(t_arr)[0]
    Q_prime_val = Q.eval_deriv(t_arr, 1)[0]
    Q_double_prime_val = Q.eval_deriv(t_arr, 2)[0]

    # Eigenvalue coefficients
    a_alpha = theta * (t - 1)
    b_alpha = theta * t
    a_beta = theta * t
    b_beta = theta * (t - 1)

    # [1] coefficient: Q(t)²
    coeff_1 = Q_val ** 2

    # [x] coefficient: Q(t)Q'(t)(a_alpha + a_beta)
    # a_alpha + a_beta = θ(t-1) + θt = θ(2t-1)
    coeff_x = Q_val * Q_prime_val * theta * (2*t - 1)

    # [y] coefficient: Q(t)Q'(t)(b_alpha + b_beta)
    # b_alpha + b_beta = θt + θ(t-1) = θ(2t-1)
    coeff_y = Q_val * Q_prime_val * theta * (2*t - 1)

    # [xy] coefficient: the full formula
    # Q(t)Q''(t)(a_alpha*b_alpha + a_beta*b_beta) + (Q'(t))²(a_alpha*b_beta + a_beta*b_alpha)
    # = Q(t)Q''(t) × 2θ²t(t-1) + (Q')² × θ²((t-1)² + t²)
    coeff_xy = (
        Q_val * Q_double_prime_val * 2 * theta**2 * t * (t - 1) +
        Q_prime_val**2 * theta**2 * ((t-1)**2 + t**2)
    )

    return {
        "coeff_1": coeff_1,
        "coeff_x": coeff_x,
        "coeff_y": coeff_y,
        "coeff_xy": coeff_xy,
        "Q_val": Q_val,
        "Q_prime": Q_prime_val,
        "Q_double_prime": Q_double_prime_val,
    }


def compute_q_ratio_analysis(Q: Polynomial, theta: float, R: float, n_quad: int = 80):
    """
    Compute the Q-derivative contribution to the correction ratio.

    The log factor correction is:
        g_internal = 1 + θ × (F_x + F_y) / F_xy

    The Q-derivative affects F_x, F_y, and F_xy differently.
    This function computes the Q-only contribution to this ratio.

    For frozen Q (Q(t) only):
        - F_x, F_y come only from non-Q parts
        - F_xy comes only from non-Q parts

    For full Q:
        - F_x, F_y get additional terms from Q'(t)
        - F_xy gets additional terms from Q' and Q''

    The epsilon is determined by how this ratio changes.
    """
    t_pts, t_wts = gauss_legendre_01(n_quad)
    weights_minus = np.exp(-2 * R * t_pts)

    # Compute integrals of each Q coefficient
    sum_coeff_1 = 0.0
    sum_coeff_x = 0.0
    sum_coeff_y = 0.0
    sum_coeff_xy = 0.0

    for i, t in enumerate(t_pts):
        coeffs = compute_q_coefficient_structure(Q, t, theta)
        w = t_wts[i] * weights_minus[i]

        sum_coeff_1 += w * coeffs["coeff_1"]
        sum_coeff_x += w * coeffs["coeff_x"]
        sum_coeff_y += w * coeffs["coeff_y"]
        sum_coeff_xy += w * coeffs["coeff_xy"]

    return {
        "integral_1": sum_coeff_1,
        "integral_x": sum_coeff_x,
        "integral_y": sum_coeff_y,
        "integral_xy": sum_coeff_xy,
        "ratio_xy_over_1": sum_coeff_xy / sum_coeff_1 if abs(sum_coeff_1) > 1e-15 else float('nan'),
        "ratio_x_over_1": sum_coeff_x / sum_coeff_1 if abs(sum_coeff_1) > 1e-15 else float('nan'),
        "ratio_x_plus_y_over_xy": (sum_coeff_x + sum_coeff_y) / sum_coeff_xy if abs(sum_coeff_xy) > 1e-15 else float('nan'),
    }


def derive_epsilon_i1_from_q(Q: Polynomial, theta: float, R: float, K: int = 3, n_quad: int = 80):
    """
    Derive epsilon_I1 from the Q-derivative structure.

    HYPOTHESIS:
    The log factor correction for I1 is modified by Q-derivatives.

    With frozen Q:
        g_internal_frozen = 1 + θ × (cross_frozen / main_frozen) = g_baseline

    With full Q, the [x] and [y] from Q add to cross, [xy] adds to main:
        g_internal_full = 1 + θ × (cross_frozen + Q_cross) / (main_frozen + Q_main)

    If Q_cross is small and Q_main is significant:
        g_internal_full ≈ g_baseline / (1 + Q_main/main_frozen)
                        ≈ g_baseline × (1 - Q_main/main_frozen)

    This would make g_I1 slightly BELOW g_baseline, not 1.0...

    ALTERNATIVE HYPOTHESIS:
    The Q-derivative in [x] and [y] provides additional correction:
        epsilon_I1 = θ × (Q_cross / main_frozen)
    """
    result = compute_q_ratio_analysis(Q, theta, R, n_quad)

    # The [x] + [y] from Q provides additional cross-terms
    Q_cross = result["integral_x"] + result["integral_y"]

    # The [xy] from Q adds to the main term
    Q_main = result["integral_xy"]

    # The frozen Q² is the baseline main term
    frozen_main = result["integral_1"]

    print(f"\nQ-derivative analysis at R={R}:")
    print(f"  frozen_main (∫Q²): {frozen_main:.6f}")
    print(f"  Q_main (∫[xy]):    {Q_main:.6f}")
    print(f"  Q_cross (∫[x+y]):  {Q_cross:.6f}")
    print(f"  ratio Q_main/frozen: {Q_main/frozen_main:.6f}")
    print(f"  ratio Q_cross/frozen: {Q_cross/frozen_main:.6f}")

    # The g_baseline from Beta moment
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    beta_moment = theta / (2 * K * (2 * K + 1))

    # HYPOTHESIS 1: epsilon from Q_cross / Q_main ratio
    # The internal correction formula involves (F_x + F_y) / F_xy
    # Q adds to both, changing the ratio
    ratio_cross_over_main = Q_cross / Q_main if abs(Q_main) > 1e-15 else float('nan')
    epsilon_h1 = theta * ratio_cross_over_main

    print(f"\n  HYPOTHESIS 1: epsilon = θ × (Q_cross/Q_main)")
    print(f"    ratio (Q_cross/Q_main): {ratio_cross_over_main:.6f}")
    print(f"    epsilon_I1: {epsilon_h1:.8f}")
    print(f"    g_I1 = 1 + epsilon: {1 + epsilon_h1:.8f}")

    # HYPOTHESIS 2: epsilon from Q_cross / frozen_main
    # If the baseline main is frozen_main
    epsilon_h2 = theta * Q_cross / frozen_main if abs(frozen_main) > 1e-15 else float('nan')

    print(f"\n  HYPOTHESIS 2: epsilon = θ × (Q_cross/frozen_main)")
    print(f"    epsilon_I1: {epsilon_h2:.8f}")
    print(f"    g_I1 = 1 + epsilon: {1 + epsilon_h2:.8f}")

    # HYPOTHESIS 3: epsilon from ratio modification
    # g_full = g_baseline × frozen_main / (frozen_main + Q_main)
    # epsilon = g_baseline × (1 - 1/(1 + Q_main/frozen_main)) ≈ -g_baseline × Q_main/frozen_main
    ratio_Q_main_over_frozen = Q_main / frozen_main if abs(frozen_main) > 1e-15 else 0
    epsilon_h3 = -beta_moment * ratio_Q_main_over_frozen

    print(f"\n  HYPOTHESIS 3: epsilon = -β × (Q_main/frozen)")
    print(f"    epsilon_I1: {epsilon_h3:.8f}")
    print(f"    g_I1 = 1 + epsilon: {1 + epsilon_h3:.8f}")

    # HYPOTHESIS 4: Combined effect
    # The Q_cross adds to numerator, Q_main adds to denominator
    # Net effect: epsilon ≈ θ × Q_cross/frozen - β × Q_main/frozen
    epsilon_h4 = theta * Q_cross / frozen_main - beta_moment * Q_main / frozen_main

    print(f"\n  HYPOTHESIS 4: epsilon = θ × Q_cross/frozen - β × Q_main/frozen")
    print(f"    epsilon_I1: {epsilon_h4:.8f}")
    print(f"    g_I1 = 1 + epsilon: {1 + epsilon_h4:.8f}")

    # Calibrated values for comparison
    g_I1_calibrated = 1.00091428
    epsilon_calibrated = g_I1_calibrated - 1.0

    print(f"\n  CALIBRATED VALUE:")
    print(f"    epsilon_I1 calibrated: {epsilon_calibrated:.8f}")
    print(f"    g_I1 calibrated: {g_I1_calibrated:.8f}")

    return {
        "epsilon_h1": epsilon_h1,
        "epsilon_h2": epsilon_h2,
        "epsilon_h3": epsilon_h3,
        "epsilon_h4": epsilon_h4,
        "epsilon_calibrated": epsilon_calibrated,
        "Q_cross": Q_cross,
        "Q_main": Q_main,
        "frozen_main": frozen_main,
    }


def main():
    print("=" * 70)
    print("DERIVING EPSILON_I1 FROM Q-DERIVATIVE ANALYSIS")
    print("=" * 70)

    theta = 4/7
    K = 3

    # Kappa benchmark
    print("\n" + "=" * 70)
    print("KAPPA BENCHMARK (R = 1.3036)")
    print("=" * 70)

    _, _, _, Q_kappa = load_przz_polynomials(enforce_Q0=False)
    Q_mono = Q_kappa.to_monomial()

    result_kappa = derive_epsilon_i1_from_q(Q_mono, theta, R=1.3036, K=K)

    # Kappa* benchmark
    print("\n" + "=" * 70)
    print("KAPPA* BENCHMARK (R = 1.1167)")
    print("=" * 70)

    _, _, _, Q_star = load_przz_polynomials_kappa_star(enforce_Q0=False)
    Q_star_mono = Q_star.to_monomial()

    result_star = derive_epsilon_i1_from_q(Q_star_mono, theta, R=1.1167, K=K)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nThe calibrated epsilon_I1 = 0.00091 is:")
    print(f"  - About 6.7% of g_baseline - 1 = {theta/(2*K*(2*K+1)):.6f}")
    print(f"  - About {0.00091/0.0136*100:.1f}% of the Beta moment correction")

    print("\nHypothesis comparison:")
    print(f"  H1 (θ×Q_cross/Q_main):    kappa={result_kappa['epsilon_h1']:.6f}, kappa*={result_star['epsilon_h1']:.6f}")
    print(f"  H2 (θ×Q_cross/frozen):    kappa={result_kappa['epsilon_h2']:.6f}, kappa*={result_star['epsilon_h2']:.6f}")
    print(f"  H3 (-β×Q_main/frozen):    kappa={result_kappa['epsilon_h3']:.6f}, kappa*={result_star['epsilon_h3']:.6f}")
    print(f"  H4 (combined):            kappa={result_kappa['epsilon_h4']:.6f}, kappa*={result_star['epsilon_h4']:.6f}")
    print(f"  Calibrated:               {result_kappa['epsilon_calibrated']:.6f}")


if __name__ == "__main__":
    main()
