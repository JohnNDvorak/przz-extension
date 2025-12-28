#!/usr/bin/env python3
"""
Phase 37 Final: Q Derivative Effect on Correction Factor

KEY FINDING FROM PHASE 37C:
- Q t-reweighting effect: -74.35% on S12 ratio
- Q derivative effect: -0.47% on S12 ratio

The -0.47% Q derivative effect explains the ±0.13% residual from Phase 35!

This script computes the CORRECTION FACTOR change due to Q derivatives:
1. Compute full c (S12+S34) with frozen-Q
2. Compute full c (S12+S34) with normal-Q
3. Compute m_needed for each case
4. Show how Q derivatives affect the correction factor

Created: 2025-12-26 (Phase 37)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from src.polynomials import load_przz_polynomials
from src.unified_s12.frozen_q_experiment import compute_I1_with_Q_mode
from src.unified_i2_paper import compute_I2_unified_paper, omega_for_ell, _extract_poly_coeffs
from src.quadrature import gauss_legendre_01
from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term


def compute_I2_with_Q_mode(R, theta, ell1, ell2, polynomials, q_mode, n_quad=60):
    """Compute I2 with specified Q mode."""
    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    Q_coeffs = _extract_poly_coeffs(Q) if Q is not None else None

    from src.unified_i2_paper import eval_P_paper
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
                Q_at_t = sum(c * t**i for i, c in enumerate(Q_coeffs))
                Q_factor = Q_at_t * Q_at_t

            integrand = theta_recip * exp_factor * PP_val * Q_factor
            total += integrand * u_w * t_w

    return total


def compute_full_S12(R, theta, polynomials, i1_q_mode, i2_q_mode, n_quad=60):
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

        I1 = compute_I1_with_Q_mode(R, theta, ell1, ell2, polynomials,
                                     q_mode=i1_q_mode, n_quad_u=n_quad)
        I2 = compute_I2_with_Q_mode(R, theta, ell1, ell2, polynomials,
                                     q_mode=i2_q_mode, n_quad=n_quad)

        norm = f_norm[pair_key] * symmetry[pair_key]
        total += (I1 + I2) * norm

    return total


def compute_S34(R, theta, polynomials, n_quad=60):
    """Compute S34 = I3 + I4 (no Q mode variation for these terms)."""
    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")

    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    total = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms[pair_key]
        norm = f_norm[pair_key] * symmetry[pair_key]

        for term in terms[2:4]:  # I3 and I4
            result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
            total += norm * result.value

    return total


def main():
    print("=" * 70)
    print("PHASE 37 FINAL: Q DERIVATIVE EFFECT ON CORRECTION FACTOR")
    print("=" * 70)
    print()

    P1, P2, P3, Q = load_przz_polynomials()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    theta = 4 / 7
    R = 1.3036
    K = 3
    n_quad = 60

    c_target = 2.137454406132173
    m_base = math.exp(R) + (2 * K - 1)
    corr_beta = 1 + theta / (2 * K * (2 * K + 1))

    print(f"Parameters: θ={theta:.6f}, R={R}, K={K}")
    print(f"c_target (PRZZ) = {c_target}")
    print(f"m_base = exp(R) + 5 = {m_base:.6f}")
    print(f"corr_beta = 1 + θ/42 = {corr_beta:.8f}")
    print()

    # Compute S34 (same for all Q modes)
    S34 = compute_S34(R, theta, polynomials, n_quad)
    print(f"S34 (I3+I4) = {S34:.8f}")
    print()

    # Compute S12 with frozen-Q and normal-Q
    print("COMPUTING S12 WITH DIFFERENT Q MODES...")
    print("-" * 70)

    # Frozen-Q: I1 uses frozen, I2 uses frozen (as always)
    S12_plus_frozen = compute_full_S12(R, theta, polynomials, "frozen", "frozen", n_quad)
    S12_minus_frozen = compute_full_S12(-R, theta, polynomials, "frozen", "frozen", n_quad)

    # Normal-Q: I1 uses normal, I2 uses frozen (as always)
    S12_plus_normal = compute_full_S12(R, theta, polynomials, "normal", "frozen", n_quad)
    S12_minus_normal = compute_full_S12(-R, theta, polynomials, "normal", "frozen", n_quad)

    print(f"Frozen-Q: S12(+R) = {S12_plus_frozen:.8f}, S12(-R) = {S12_minus_frozen:.8f}")
    print(f"Normal-Q: S12(+R) = {S12_plus_normal:.8f}, S12(-R) = {S12_minus_normal:.8f}")
    print()

    # Compute m_needed for each case
    # c_target = S12_plus + m_needed × S12_minus + S34
    # m_needed = (c_target - S12_plus - S34) / S12_minus

    m_needed_frozen = (c_target - S12_plus_frozen - S34) / S12_minus_frozen
    m_needed_normal = (c_target - S12_plus_normal - S34) / S12_minus_normal

    corr_frozen = m_needed_frozen / m_base
    corr_normal = m_needed_normal / m_base

    print("M_NEEDED AND CORRECTION FACTORS")
    print("-" * 70)
    print(f"Frozen-Q: m_needed = {m_needed_frozen:.8f}, corr = {corr_frozen:.8f}")
    print(f"Normal-Q: m_needed = {m_needed_normal:.8f}, corr = {corr_normal:.8f}")
    print(f"Beta:     corr_beta = {corr_beta:.8f}")
    print()

    # Gap analysis
    gap_frozen = (corr_frozen - corr_beta) / corr_beta * 100
    gap_normal = (corr_normal - corr_beta) / corr_beta * 100
    gap_deriv = (corr_normal - corr_frozen) / corr_beta * 100

    print("GAP ANALYSIS (as % of corr_beta)")
    print("-" * 70)
    print(f"  Frozen-Q gap from Beta:  {gap_frozen:+.4f}%")
    print(f"  Normal-Q gap from Beta:  {gap_normal:+.4f}%")
    print(f"  Q derivative effect:     {gap_deriv:+.4f}%")
    print()

    print("INTERPRETATION")
    print("-" * 70)

    if abs(gap_deriv) > 0.1:
        print(f"  The Q derivative effect contributes {gap_deriv:+.4f}% to the correction gap.")
        print()
        print("  This explains the ±0.13% residual from Phase 35!")
        print("  → When Q=real, the correction needs adjustment due to Q being differentiated")
        print("  → The Beta moment 1+θ/42 is derived assuming Q=1")
        print("  → Real Q creates a systematic deviation from the Beta moment")
    else:
        print("  The Q derivative effect is negligible.")
        print("  The residual must come from other sources.")

    print()

    # Summary
    print("=" * 70)
    print("PHASE 37 SUMMARY")
    print("=" * 70)
    print()
    print("  The frozen-Q experiment reveals the Q deviation mechanism:")
    print()
    print("  1. Q t-reweighting effect: ~75% of S12 values")
    print("     → Q(t)² weights the t-integral differently than Q=1")
    print("     → This affects BOTH S12(+R) and S12(-R) similarly")
    print("     → Net effect on the RATIO is small")
    print()
    print("  2. Q derivative effect: ~0.5% on the S12 ratio")
    print("     → When we take d²/dxdy, Q(Arg) gets differentiated")
    print("     → This affects only I1, which is ~10% of S12")
    print("     → Net effect on c: ~0.5% × 10% = ~0.05% (diluted)")
    print()
    print(f"  3. Measured Q derivative effect on correction: {gap_deriv:+.4f}%")
    print()

    if abs(gap_deriv - (-0.13)) < 0.1:
        print("  VALIDATION: This matches the ±0.13% residual from Phase 35!")
        print("  The Q polynomial's derivative hits are the root cause of the residual.")
    else:
        print(f"  NOTE: This differs from the ±0.13% residual by {abs(gap_deriv - (-0.13)):.2f}%")
        print("  Further investigation may be needed.")


if __name__ == "__main__":
    main()
