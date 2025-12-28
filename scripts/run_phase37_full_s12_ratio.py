#!/usr/bin/env python3
"""
Phase 37C: Full S12 (I1+I2) Ratio Analysis with Q Modes

Now that we know:
- I1 is only ~10% of S12
- I2 always uses Q(t)² (frozen)
- Q derivative effects in I1 are diluted by I2

This script computes the full S12 = I1 + I2 ratio for:
- P=real, Q=1 (baseline: no Q effects)
- P=real, Q=frozen (I1 uses frozen-Q, I2 unchanged)
- P=real, Q=real (full computation)

The key question: Does the S12(+R)/S12(-R) ratio change?
If so, by how much, and does this explain the ±0.13% residual?

Created: 2025-12-26 (Phase 37)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from src.polynomials import load_przz_polynomials
from src.unified_i1_paper import compute_I1_unified_paper
from src.unified_i2_paper import compute_I2_unified_paper
from src.unified_s12.frozen_q_experiment import compute_I1_with_Q_mode


def compute_I2_with_Q_mode(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: dict,
    q_mode: str,
    n_quad: int = 60,
) -> float:
    """
    Compute I2 with specified Q mode.

    Note: I2 ALWAYS uses Q(t)² in the actual formula.
    But for comparison, we can compute it with Q=1 as well.
    """
    from src.quadrature import gauss_legendre_01
    from src.unified_i2_paper import eval_P_paper, omega_for_ell, _extract_poly_coeffs

    import numpy as np

    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    Q_coeffs = _extract_poly_coeffs(Q) if Q is not None else None

    theta_recip = 1.0 / theta

    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        P1_val = eval_P_paper(P_ell1, u, omega1, R, theta, n_quad_a=40)
        P2_val = eval_P_paper(P_ell2, u, omega2, R, theta, n_quad_a=40)
        PP_val = P1_val * P2_val

        for t, t_w in zip(t_nodes, t_weights):
            exp_factor = math.exp(2 * R * t)

            if q_mode == "none" or Q_coeffs is None:
                Q_factor = 1.0
            else:
                # Both frozen and normal use Q(t)² for I2
                Q_at_t = sum(c * t**i for i, c in enumerate(Q_coeffs))
                Q_factor = Q_at_t * Q_at_t

            integrand = theta_recip * exp_factor * PP_val * Q_factor
            total += integrand * u_w * t_w

    return total


def compute_full_S12(
    R: float,
    theta: float,
    polynomials: dict,
    i1_q_mode: str,  # "none", "frozen", or "normal"
    i2_q_mode: str,  # "none" or "frozen/normal" (same for I2)
    n_quad: int = 60,
) -> float:
    """Compute S12 = I1 + I2 with specified Q modes."""
    pairs = ["11", "22", "33", "12", "13", "23"]
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    total = 0.0
    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        I1 = compute_I1_with_Q_mode(
            R, theta, ell1, ell2, polynomials,
            q_mode=i1_q_mode, n_quad_u=n_quad,
        )
        I2 = compute_I2_with_Q_mode(
            R, theta, ell1, ell2, polynomials,
            q_mode=i2_q_mode, n_quad=n_quad,
        )

        norm = f_norm[pair_key] * symmetry[pair_key]
        total += (I1 + I2) * norm

    return total


def main():
    print("=" * 70)
    print("PHASE 37C: FULL S12 RATIO ANALYSIS WITH Q MODES")
    print("=" * 70)
    print()

    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_full = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Also test with Q=1
    from src.polynomials import Polynomial
    Q_one = Polynomial(np.array([1.0]))
    polynomials_Qone = {"P1": P1, "P2": P2, "P3": P3, "Q": Q_one}

    theta = 4 / 7
    R = 1.3036
    K = 3
    n_quad = 60

    corr_beta = 1 + theta / (2 * K * (2 * K + 1))
    m_base = math.exp(R) + (2 * K - 1)

    print(f"Parameters: θ={theta:.6f}, R={R}, K={K}")
    print(f"Beta moment correction = 1 + θ/(2K(2K+1)) = {corr_beta:.8f}")
    print(f"m_base = exp(R) + 5 = {m_base:.4f}")
    print()

    # Scenario 1: P=real, Q=1 (no Q at all)
    # Scenario 2: P=real, Q=frozen (I1 uses frozen, I2 uses frozen)
    # Scenario 3: P=real, Q=real (I1 uses full Q, I2 uses Q(t)²)

    scenarios = [
        ("P=real, Q=1", polynomials_Qone, "none", "none"),
        ("P=real, Q=frozen", polynomials_full, "frozen", "frozen"),
        ("P=real, Q=real", polynomials_full, "normal", "frozen"),  # I2 is always frozen
    ]

    print("S12 VALUES AT +R AND -R")
    print("-" * 70)
    print(f"{'Scenario':<20} | {'S12(+R)':<14} | {'S12(-R)':<14} | {'Ratio +/-':<12}")
    print("-" * 70)

    results = {}
    for name, polys, i1_mode, i2_mode in scenarios:
        S12_plus = compute_full_S12(R, theta, polys, i1_mode, i2_mode, n_quad)
        S12_minus = compute_full_S12(-R, theta, polys, i1_mode, i2_mode, n_quad)
        ratio = S12_plus / S12_minus if abs(S12_minus) > 1e-15 else float('inf')

        results[name] = {
            "S12_plus": S12_plus,
            "S12_minus": S12_minus,
            "ratio": ratio,
        }

        print(f"{name:<20} | {S12_plus:+.8f}   | {S12_minus:+.8f}   | {ratio:.6f}")

    print()

    # Compute m_needed for each scenario using c_target
    c_target = 2.137454406132173  # PRZZ κ benchmark

    # For now, just compute the ratio changes
    ratio_Qone = results["P=real, Q=1"]["ratio"]
    ratio_Qfrozen = results["P=real, Q=frozen"]["ratio"]
    ratio_Qreal = results["P=real, Q=real"]["ratio"]

    print("RATIO CHANGES (relative to P=real, Q=1)")
    print("-" * 70)
    delta_frozen = (ratio_Qfrozen / ratio_Qone - 1) * 100
    delta_real = (ratio_Qreal / ratio_Qone - 1) * 100
    delta_deriv = (ratio_Qreal / ratio_Qfrozen - 1) * 100

    print(f"  Frozen-Q vs Q=1:    {delta_frozen:+.4f}%   (Q t-reweighting effect)")
    print(f"  Real-Q vs Q=1:      {delta_real:+.4f}%   (total Q effect)")
    print(f"  Real-Q vs Frozen-Q: {delta_deriv:+.4f}%   (Q derivative effect)")
    print()

    # Now compute m_derived and c_derived for the real-Q case
    print("CORRECTION ANALYSIS FOR REAL-Q CASE")
    print("-" * 70)

    S12_plus = results["P=real, Q=real"]["S12_plus"]
    S12_minus = results["P=real, Q=real"]["S12_minus"]

    m_derived = m_base * corr_beta
    c_with_derived = S12_plus + m_derived * S12_minus

    m_empirical = m_base
    c_with_empirical = S12_plus + m_empirical * S12_minus

    ratio_c = c_with_derived / c_with_empirical

    print(f"  m_base = {m_base:.6f}")
    print(f"  m_derived = m_base × corr_beta = {m_derived:.6f}")
    print()
    print(f"  S12(+R) = {S12_plus:.8f}")
    print(f"  S12(-R) = {S12_minus:.8f}")
    print()
    print(f"  c_empirical = S12(+R) + m_base × S12(-R) = {c_with_empirical:.6f}")
    print(f"  c_derived   = S12(+R) + m_derived × S12(-R) = {c_with_derived:.6f}")
    print(f"  Ratio = {ratio_c:.8f} (gap = {(ratio_c-1)*100:+.4f}%)")
    print()

    # Compare to benchmark
    print("COMPARISON TO BENCHMARK")
    print("-" * 70)
    print(f"  c_target (PRZZ) = {c_target:.6f}")
    print(f"  c_derived       = {c_with_derived:.6f}")
    print(f"  Gap from target = {(c_with_derived/c_target - 1)*100:+.4f}%")
    print()

    # Note: This is S12 only, missing S34
    print("NOTE: This analysis uses S12 only (I1+I2). S34 (I3+I4) is not included.")
    print("      The full c computation would add S34 ≈ -0.6 to both values.")


if __name__ == "__main__":
    main()
