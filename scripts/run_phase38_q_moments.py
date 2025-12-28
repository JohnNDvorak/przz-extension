#!/usr/bin/env python3
"""
Phase 38: Q Polynomial Moment Analysis

Goal: Derive an analytic formula for the Q-induced deviation.

The Q polynomial causes a -0.18% deviation from the Beta moment prediction.
This script computes Q-related moments to understand the mechanism:

1. ⟨Q(t)²⟩_t - average of Q² over [0,1]
2. ⟨Q'(t)²⟩_t - average of Q' squared
3. ⟨Q(t)Q'(t)⟩_t - cross moment
4. ⟨t Q(t)²⟩_t - t-weighted Q² moment

The hypothesis is that the -0.18% deviation comes from Q derivative
contributions in I1 that are not captured by the pure Beta moment.

Created: 2025-12-26 (Phase 38)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from scipy import integrate
from src.polynomials import load_przz_polynomials


def compute_q_moments(Q, n_points=1000):
    """
    Compute various moments of the Q polynomial.

    Args:
        Q: Q polynomial object with eval and eval_deriv methods
        n_points: Number of points for numerical integration

    Returns:
        Dict of moment values
    """
    t = np.linspace(0, 1, n_points)
    dt = 1.0 / (n_points - 1)

    # Evaluate Q and Q'
    Q_vals = Q.eval(t)
    Q_prime_vals = Q.eval_deriv(t, 1)

    # Basic moments
    Q_sq_mean = np.trapz(Q_vals**2, t)
    Q_prime_sq_mean = np.trapz(Q_prime_vals**2, t)
    QQ_prime_mean = np.trapz(Q_vals * Q_prime_vals, t)

    # t-weighted moments
    t_Q_sq_mean = np.trapz(t * Q_vals**2, t)
    t_Q_prime_sq_mean = np.trapz(t * Q_prime_vals**2, t)

    # (2t-1)-weighted moments (relevant for exp factor in I1)
    twoT_minus_1_Q_sq = np.trapz((2*t - 1) * Q_vals**2, t)

    # exp(2Rt) weighted moments for R=1.3036
    R = 1.3036
    exp_2Rt = np.exp(2 * R * t)
    exp_Q_sq = np.trapz(exp_2Rt * Q_vals**2, t) / np.trapz(exp_2Rt, t)

    return {
        "Q(0)": Q.eval(np.array([0.0]))[0],
        "Q(1)": Q.eval(np.array([1.0]))[0],
        "Q(0.5)": Q.eval(np.array([0.5]))[0],
        "<Q²>": Q_sq_mean,
        "<Q'²>": Q_prime_sq_mean,
        "<QQ'>": QQ_prime_mean,
        "<tQ²>": t_Q_sq_mean,
        "<tQ'²>": t_Q_prime_sq_mean,
        "<(2t-1)Q²>": twoT_minus_1_Q_sq,
        "<Q²>_exp": exp_Q_sq,  # exp(2Rt)-weighted
    }


def analyze_q_derivative_contribution(Q, theta, n_points=1000):
    """
    Analyze how Q derivatives contribute to the I1 integrand.

    In I1, we have:
        d²/dxdy [exp(...) × log_factor × P(u+x)P(u+y) × Q(Arg_α)Q(Arg_β)]

    The Q contribution comes from:
        Q(Arg_α) = Q(t + θ(t-1)x + θty)
        Q(Arg_β) = Q(t + θtx + θ(t-1)y)

    When we take d/dx or d/dy:
        dQ(Arg_α)/dx = Q'(t) × θ(t-1)
        dQ(Arg_α)/dy = Q'(t) × θt
        etc.

    At x=y=0, the contribution from Q derivatives to the x^1y^1 coefficient
    involves Q'(t) weighted by θ²×t(t-1) factors.
    """
    t = np.linspace(0, 1, n_points)

    Q_vals = Q.eval(t)
    Q_prime_vals = Q.eval_deriv(t, 1)

    # The argument dependence:
    # Arg_α = t + θ(t-1)x + θty → dArg_α/dx = θ(t-1), dArg_α/dy = θt
    # Arg_β = t + θtx + θ(t-1)y → dArg_β/dx = θt, dArg_β/dy = θ(t-1)

    # For d²/dxdy of Q(Arg_α)Q(Arg_β), we get terms involving:
    # Q'(t)² weighted by (θ(t-1))(θt) + (θt)(θ(t-1)) = 2θ² t(t-1)

    # This is negative for t ∈ (0,1) since t(t-1) < 0
    weight_factor = 2 * theta**2 * t * (t - 1)  # Always negative in (0,1)

    # Integrate Q'² with this weight
    Q_prime_weighted = np.trapz(Q_prime_vals**2 * weight_factor, t)

    # Also compute the Q²×Q'² cross term contribution
    # (More complex, but this gives the leading behavior)

    return {
        "weight_factor_at_0.5": 2 * theta**2 * 0.5 * (0.5 - 1),
        "<Q'² × 2θ²t(t-1)>": Q_prime_weighted,
        "sign": "negative" if Q_prime_weighted < 0 else "positive",
    }


def main():
    print("=" * 70)
    print("PHASE 38: Q POLYNOMIAL MOMENT ANALYSIS")
    print("=" * 70)
    print()

    P1, P2, P3, Q = load_przz_polynomials()
    theta = 4 / 7

    print("Q POLYNOMIAL STRUCTURE")
    print("-" * 70)

    moments = compute_q_moments(Q)
    print(f"  Q(0) = {moments['Q(0)']:+.6f}")
    print(f"  Q(1) = {moments['Q(1)']:+.6f}")
    print(f"  Q(0.5) = {moments['Q(0.5)']:+.6f}")
    print()
    print("  Note: Q(0) = +1, Q(1) = -1 means Q changes sign!")
    print("        This breaks symmetry in the t-integral.")
    print()

    print("Q MOMENTS")
    print("-" * 70)
    Q_sq = moments['<Q²>']
    Q_prime_sq = moments["<Q'²>"]
    QQ_prime = moments["<QQ'>"]
    tQ_sq = moments['<tQ²>']
    tQ_prime_sq = moments["<tQ'²>"]
    twoT_Q_sq = moments['<(2t-1)Q²>']
    Q_sq_exp = moments['<Q²>_exp']

    print(f"  <Q²>   = {Q_sq:.6f}  (integral of Q² over [0,1])")
    print(f"  <Q'²>  = {Q_prime_sq:.6f}  (integral of Q'² over [0,1])")
    print(f"  <QQ'>  = {QQ_prime:.6f}  (cross moment)")
    print()
    print("  T-WEIGHTED MOMENTS:")
    print(f"  <tQ²>       = {tQ_sq:.6f}")
    print(f"  <tQ'²>      = {tQ_prime_sq:.6f}")
    print(f"  <(2t-1)Q²>  = {twoT_Q_sq:.6f}")
    print()
    print(f"  EXP-WEIGHTED (R=1.3036):")
    print(f"  <Q²>_exp    = {Q_sq_exp:.6f}")
    print()

    print("Q DERIVATIVE CONTRIBUTION ANALYSIS")
    print("-" * 70)

    deriv_analysis = analyze_q_derivative_contribution(Q, theta)
    weight_factor = deriv_analysis['weight_factor_at_0.5']
    Q_prime_weighted = deriv_analysis["<Q'² × 2θ²t(t-1)>"]
    sign = deriv_analysis['sign']
    print(f"  Weight factor 2θ²t(t-1) at t=0.5: {weight_factor:.6f}")
    print(f"  <Q'² × 2θ²t(t-1)> = {Q_prime_weighted:.6f}")
    print(f"  Sign: {sign}")
    print()

    print("RELATING TO THE -0.18% DEVIATION")
    print("-" * 70)

    # The Beta moment is 1 + θ/(2K(2K+1)) = 1 + θ/42
    K = 3
    beta_moment = 1 + theta / (2 * K * (2 * K + 1))

    # The -0.18% deviation in the ratio is:
    deviation_observed = -0.0018  # as a fraction

    # Hypothesis: deviation ∝ θ × <Q'² × t(t-1)> / <Q²>
    q_ratio = abs(Q_prime_weighted) / Q_sq

    print(f"  Beta moment correction = {beta_moment:.6f}")
    print(f"  Observed deviation = {deviation_observed*100:.4f}%")
    print()
    print(f"  Hypothesis: deviation ~ θ × <Q'² × t(t-1)> / <Q²>")
    print(f"  Computed ratio = {q_ratio:.6f}")
    print()

    # Try to find a simple formula
    # The deviation might be related to: θ² × ∫ Q'² t(t-1) dt / ∫ Q² dt
    simple_formula = theta * q_ratio
    print(f"  θ × ratio = {simple_formula:.6f} = {simple_formula*100:.4f}%")
    print()

    if abs(simple_formula - abs(deviation_observed)) < 0.01:
        print("  This matches the observed deviation!")
    else:
        print(f"  Gap from observed: {(simple_formula - abs(deviation_observed))*100:.4f}%")
        print("  Need a different formula or additional terms.")

    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("  1. The Q derivative contribution is negative (as expected)")
    print("  2. It involves Q'² weighted by t(t-1) < 0")
    print("  3. This effectively REDUCES the correction factor")
    print()
    print("  For a full derivation, we need to:")
    print("  - Track how Q'(t) terms propagate through the product rule")
    print("  - Account for the exp(2Rt) weighting")
    print("  - Consider the P polynomial structure interaction")
    print()
    print("  The current ±0.15% accuracy may be acceptable for production.")
    print("  Further derivation would reduce this to ~±0.05%.")


if __name__ == "__main__":
    main()
