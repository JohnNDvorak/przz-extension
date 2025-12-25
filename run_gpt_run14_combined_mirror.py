#!/usr/bin/env python3
"""
GPT Run 14: TeX-Combined Mirror Evaluator (K=3 only)

This script implements the COMBINED +R/-R integral directly from TeX
and computes EXACT mirror weights without any amplitude calibration.

Key insight from Run 12-13:
- Current tex_mirror separates +R and -R, combining with calibrated m1, m2
- TeX (lines 1502-1511) shows I(α,β) + T^{-α-β}·I(-β,-α) combined structure
- At α=β=-R/L, this becomes a single combined integral

Goal: Compute m_exact(R) and test if it's pair-independent.

Formula:
    m1_exact = (I1_TeX_combined - I1_plus) / I1_minus_base
    m2_exact = (I2_TeX_combined - I2_plus) / I2_minus_base

If m1_exact, m2_exact are pair-independent, we have a universal mirror formula.
If they vary by pair, the structure is more complex.

Usage:
    python run_gpt_run14_combined_mirror.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
    Polynomial,
)
from src.quadrature import gauss_legendre_01
from src.evaluate import compute_c_paper_tex_mirror


THETA = 4.0 / 7.0

# PRZZ Targets
TARGETS = {
    "kappa": {
        "name": "κ",
        "R": 1.3036,
        "c_target": 2.13745440613217263636,
    },
    "kappa_star": {
        "name": "κ*",
        "R": 1.1167,
        "c_target": 1.93801,
    }
}

# K=3 pairs (ℓ₁, ℓ₂) with multiplicities
# 6 unique pairs for ordered: (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
K3_PAIRS = [
    (1, 1), (1, 2), (1, 3),
    (2, 2), (2, 3),
    (3, 3),
]


def eval_poly(P, u: float) -> float:
    """Evaluate polynomial at point u."""
    return float(P.eval(np.array([u]))[0])


@dataclass
class ExactMirrorResult:
    """Result of exact mirror weight computation for one pair."""
    ell1: int
    ell2: int
    I1_combined: float  # I1 with +R/-R combined (TeX formula)
    I1_plus: float      # I1 at +R only
    I1_minus_base: float  # I1 at -R (base, no operator)
    I2_combined: float
    I2_plus: float
    I2_minus_base: float
    m1_exact: float     # Extracted exact weight
    m2_exact: float


def compute_I1_combined_for_pair(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    n_quad: int = 60,
) -> float:
    """
    Compute I₁ for pair (ℓ₁, ℓ₂) using COMBINED +R/-R formula from TeX.

    The TeX mirror structure (lines 1502-1511) shows:
        I(α,β) + T^{-α-β}·I(-β,-α)

    At α=β=-R/L (where L=log N, and we normalize so L≈1/θ), this becomes
    a combined integral where both +R and -R contributions are present.

    For I₁, the combined structure at x=y=0 involves:
    - The +R branch: contributes exp(2Rt) × Q(t)²
    - The -R branch: contributes T^{2R/L} × exp(-2Rt) × Q(-t)² reflected terms

    Simplified combined form (at x=y=0):
        I₁_combined = ∫∫ (1-u)^power P_{ℓ₁}(u) P_{ℓ₂}(u) [exp(2Rt) + exp(2R/θ)×exp(-2Rt)] Q(t)² du dt
    """
    nodes, weights = gauss_legendre_01(n_quad)

    # (1-u) power follows OLD convention: ℓ₁ + ℓ₂
    power = ell1 + ell2

    # The mirror factor T^{-α-β} at α=β=-R/L becomes exp(2R/θ)
    # because T = N^θ and -α-β = 2R/L ≈ 2Rθ (with L≈1/θ)
    mirror_factor = np.exp(2 * R / theta)

    result = 0.0
    for i, (u, wu) in enumerate(zip(nodes, weights)):
        for j, (t, wt) in enumerate(zip(nodes, weights)):
            # (1-u)^power
            omu_power = (1 - u) ** power

            # P factors
            P1_val = eval_poly(P_ell1, u)
            P2_val = eval_poly(P_ell2, u)

            # Q factor (at t for +R branch, at t for -R reflected)
            Q_val = eval_poly(Q, t)
            Q_sq = Q_val ** 2

            # Combined exponential: exp(2Rt) + mirror_factor × exp(-2Rt)
            # This captures both +R and -R branches
            exp_combined = np.exp(2 * R * t) + mirror_factor * np.exp(-2 * R * t)

            integrand = (1.0 / theta) * omu_power * P1_val * P2_val * Q_sq * exp_combined
            result += wu * wt * integrand

    return result


def compute_I1_plus_for_pair(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    n_quad: int = 60,
) -> float:
    """Compute I₁ at +R only (the direct branch)."""
    nodes, weights = gauss_legendre_01(n_quad)

    power = ell1 + ell2

    result = 0.0
    for i, (u, wu) in enumerate(zip(nodes, weights)):
        for j, (t, wt) in enumerate(zip(nodes, weights)):
            omu_power = (1 - u) ** power
            P1_val = eval_poly(P_ell1, u)
            P2_val = eval_poly(P_ell2, u)
            Q_val = eval_poly(Q, t)
            Q_sq = Q_val ** 2
            exp_factor = np.exp(2 * R * t)

            integrand = (1.0 / theta) * omu_power * P1_val * P2_val * Q_sq * exp_factor
            result += wu * wt * integrand

    return result


def compute_I1_minus_base_for_pair(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    n_quad: int = 60,
) -> float:
    """
    Compute I₁ at -R (the mirror base).

    This is the "raw" -R evaluation without mirror factor.
    The mirror factor exp(2R/θ) is what converts -R base to mirror contribution.
    """
    nodes, weights = gauss_legendre_01(n_quad)

    power = ell1 + ell2

    result = 0.0
    for i, (u, wu) in enumerate(zip(nodes, weights)):
        for j, (t, wt) in enumerate(zip(nodes, weights)):
            omu_power = (1 - u) ** power
            P1_val = eval_poly(P_ell1, u)
            P2_val = eval_poly(P_ell2, u)
            Q_val = eval_poly(Q, t)
            Q_sq = Q_val ** 2
            # At -R, the exponential is exp(-2Rt)
            exp_factor = np.exp(-2 * R * t)

            integrand = (1.0 / theta) * omu_power * P1_val * P2_val * Q_sq * exp_factor
            result += wu * wt * integrand

    return result


def compute_I2_combined_for_pair(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    n_quad: int = 60,
) -> float:
    """
    Compute I₂ with combined +R/-R formula.

    I₂ from TeX line 1548:
        I₂ = T Φ̂(0)/θ ∫∫ Q(t)² exp(2Rt) P_{ℓ₁}(u) P_{ℓ₂}(u) dt du

    Combined with mirror:
        I₂_combined = ∫∫ P_{ℓ₁}(u) P_{ℓ₂}(u) Q(t)² [exp(2Rt) + exp(2R/θ)×exp(-2Rt)] du dt
    """
    nodes, weights = gauss_legendre_01(n_quad)

    mirror_factor = np.exp(2 * R / theta)

    result = 0.0
    for i, (u, wu) in enumerate(zip(nodes, weights)):
        for j, (t, wt) in enumerate(zip(nodes, weights)):
            P1_val = eval_poly(P_ell1, u)
            P2_val = eval_poly(P_ell2, u)
            Q_val = eval_poly(Q, t)
            Q_sq = Q_val ** 2

            exp_combined = np.exp(2 * R * t) + mirror_factor * np.exp(-2 * R * t)

            integrand = (1.0 / theta) * P1_val * P2_val * Q_sq * exp_combined
            result += wu * wt * integrand

    return result


def compute_I2_plus_for_pair(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    n_quad: int = 60,
) -> float:
    """Compute I₂ at +R only."""
    nodes, weights = gauss_legendre_01(n_quad)

    result = 0.0
    for i, (u, wu) in enumerate(zip(nodes, weights)):
        for j, (t, wt) in enumerate(zip(nodes, weights)):
            P1_val = eval_poly(P_ell1, u)
            P2_val = eval_poly(P_ell2, u)
            Q_val = eval_poly(Q, t)
            Q_sq = Q_val ** 2
            exp_factor = np.exp(2 * R * t)

            integrand = (1.0 / theta) * P1_val * P2_val * Q_sq * exp_factor
            result += wu * wt * integrand

    return result


def compute_I2_minus_base_for_pair(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    n_quad: int = 60,
) -> float:
    """Compute I₂ at -R (base, no mirror factor)."""
    nodes, weights = gauss_legendre_01(n_quad)

    result = 0.0
    for i, (u, wu) in enumerate(zip(nodes, weights)):
        for j, (t, wt) in enumerate(zip(nodes, weights)):
            P1_val = eval_poly(P_ell1, u)
            P2_val = eval_poly(P_ell2, u)
            Q_val = eval_poly(Q, t)
            Q_sq = Q_val ** 2
            exp_factor = np.exp(-2 * R * t)

            integrand = (1.0 / theta) * P1_val * P2_val * Q_sq * exp_factor
            result += wu * wt * integrand

    return result


def compute_exact_mirror_for_pair(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    n_quad: int = 60,
) -> ExactMirrorResult:
    """
    Compute exact mirror weights for a single pair.

    m1_exact = (I1_combined - I1_plus) / I1_minus_base
    m2_exact = (I2_combined - I2_plus) / I2_minus_base
    """
    I1_combined = compute_I1_combined_for_pair(theta, R, ell1, ell2, P_ell1, P_ell2, Q, n_quad)
    I1_plus = compute_I1_plus_for_pair(theta, R, ell1, ell2, P_ell1, P_ell2, Q, n_quad)
    I1_minus_base = compute_I1_minus_base_for_pair(theta, R, ell1, ell2, P_ell1, P_ell2, Q, n_quad)

    I2_combined = compute_I2_combined_for_pair(theta, R, ell1, ell2, P_ell1, P_ell2, Q, n_quad)
    I2_plus = compute_I2_plus_for_pair(theta, R, ell1, ell2, P_ell1, P_ell2, Q, n_quad)
    I2_minus_base = compute_I2_minus_base_for_pair(theta, R, ell1, ell2, P_ell1, P_ell2, Q, n_quad)

    # Extract exact weights
    # m_exact = (I_combined - I_plus) / I_minus_base
    if abs(I1_minus_base) > 1e-12:
        m1_exact = (I1_combined - I1_plus) / I1_minus_base
    else:
        m1_exact = float('inf')

    if abs(I2_minus_base) > 1e-12:
        m2_exact = (I2_combined - I2_plus) / I2_minus_base
    else:
        m2_exact = float('inf')

    return ExactMirrorResult(
        ell1=ell1,
        ell2=ell2,
        I1_combined=I1_combined,
        I1_plus=I1_plus,
        I1_minus_base=I1_minus_base,
        I2_combined=I2_combined,
        I2_plus=I2_plus,
        I2_minus_base=I2_minus_base,
        m1_exact=m1_exact,
        m2_exact=m2_exact,
    )


def get_poly_for_ell(ell: int, polynomials: Dict):
    """Get polynomial P_ℓ from dictionary."""
    if ell == 1:
        return polynomials["P1"]
    elif ell == 2:
        return polynomials["P2"]
    elif ell == 3:
        return polynomials["P3"]
    else:
        raise ValueError(f"Unknown ell={ell}")


def main():
    print("=" * 70)
    print("GPT Run 14: TeX-Combined Mirror Evaluator (K=3)")
    print("=" * 70)
    print()
    print("Computing EXACT mirror weights from combined +R/-R integrals.")
    print()
    print("Formula:")
    print("  m1_exact = (I1_combined - I1_plus) / I1_minus_base")
    print("  m2_exact = (I2_combined - I2_plus) / I2_minus_base")
    print()
    print("The combined integral uses the TeX mirror factor exp(2R/θ).")
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    benchmarks = [
        ("κ", polys_kappa, TARGETS["kappa"]),
        ("κ*", polys_kappa_star, TARGETS["kappa_star"]),
    ]

    for bench_name, polys, target in benchmarks:
        R = target["R"]
        c_target = target["c_target"]

        # Get tex_mirror reference for comparison
        tex_result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            terms_version="old",
            tex_exp_component="exp_R_ref",
        )

        print("=" * 70)
        print(f"BENCHMARK: {bench_name} (R={R})")
        print("=" * 70)
        print()

        # tex_mirror reference
        print("tex_mirror reference (exp_R_ref mode):")
        print(f"  m1 = {tex_result.m1:.4f}")
        print(f"  m2 = {tex_result.m2:.4f}")
        print(f"  c  = {tex_result.c:.6f} (gap {100*(tex_result.c - c_target)/c_target:+.2f}%)")
        print()

        # Theoretical mirror factor
        mirror_factor = np.exp(2 * R / THETA)
        print(f"Theoretical mirror factor exp(2R/θ) = {mirror_factor:.4f}")
        print()

        # Compute exact weights for each pair
        print("Pair-by-pair exact mirror weights:")
        print("-" * 70)
        print(f"{'Pair':<8} {'m1_exact':>12} {'m2_exact':>12} {'I1_combined':>14} {'I2_combined':>14}")
        print("-" * 70)

        results = []
        for ell1, ell2 in K3_PAIRS:
            P_ell1 = get_poly_for_ell(ell1, polys)
            P_ell2 = get_poly_for_ell(ell2, polys)
            Q = polys["Q"]

            result = compute_exact_mirror_for_pair(
                THETA, R, ell1, ell2, P_ell1, P_ell2, Q, n_quad=60
            )
            results.append(result)

            print(
                f"({ell1},{ell2}){' ' * (5 - len(str(ell1)+str(ell2)))} "
                f"{result.m1_exact:12.4f} {result.m2_exact:12.4f} "
                f"{result.I1_combined:14.6f} {result.I2_combined:14.6f}"
            )

        # Analyze pair independence
        m1_values = [r.m1_exact for r in results]
        m2_values = [r.m2_exact for r in results]

        m1_mean = np.mean(m1_values)
        m1_std = np.std(m1_values)
        m2_mean = np.mean(m2_values)
        m2_std = np.std(m2_values)

        print("-" * 70)
        print()
        print("PAIR INDEPENDENCE ANALYSIS:")
        print(f"  m1_exact: mean = {m1_mean:.4f}, std = {m1_std:.4f}, CV = {100*m1_std/abs(m1_mean):.1f}%")
        print(f"  m2_exact: mean = {m2_mean:.4f}, std = {m2_std:.4f}, CV = {100*m2_std/abs(m2_mean):.1f}%")
        print()

        # Comparison with tex_mirror
        print("COMPARISON WITH tex_mirror:")
        print(f"  tex_mirror m1 = {tex_result.m1:.4f}, exact mean = {m1_mean:.4f}, delta = {m1_mean - tex_result.m1:+.4f}")
        print(f"  tex_mirror m2 = {tex_result.m2:.4f}, exact mean = {m2_mean:.4f}, delta = {m2_mean - tex_result.m2:+.4f}")
        print()

    # Final summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
KEY FINDINGS:

1. The exact mirror weights are computed from:
   m_exact = (I_combined - I_plus) / I_minus_base

2. The combined integral uses the TeX mirror factor exp(2R/θ).

3. If m1_exact and m2_exact have low coefficient of variation (CV < 10%),
   they are approximately pair-independent.

4. Compare exact mean to tex_mirror's fitted values to assess calibration quality.

INTERPRETATION:

- If exact weights ≈ tex_mirror: The amplitude model is a good approximation
- If exact weights differ: The amplitude model needs revision
- If CV is high: Weights are pair-dependent, structure is more complex
""")


if __name__ == "__main__":
    main()
