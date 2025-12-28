#!/usr/bin/env python3
"""
scripts/run_phase28_paper_regime_mirror.py
Phase 28: Mirror Transform in Paper Regime

Computes the effective mirror multiplier m_eff within the PAPER regime,
where the empirical m = exp(R) + 5 was originally calibrated.

Key insight from Phase 28:
- unified_general = raw regime
- compute_c_paper_with_mirror = paper regime
- Empirical m = exp(R) + 5 was calibrated for paper regime
- Phase 27's m_eff ≈ 3.94 was computed in raw regime (wrong comparison)

This script answers: What is m_eff when computed entirely within paper regime?

Created: 2025-12-26 (Phase 28)
"""

import sys
import math

sys.path.insert(0, ".")

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.terms_k3_d1 import (
    make_I1_11, make_I1_22, make_I1_33,
    make_I1_12, make_I1_13, make_I1_23,
    make_I2_11, make_I2_22, make_I2_33,
    make_I2_12, make_I2_13, make_I2_23,
    make_all_terms_k3,
)
from src.evaluate import evaluate_term


def compute_I1_paper_regime(pair_key, theta, R, polynomials, n_quad=60):
    """Compute I1 using term DSL with paper regime."""
    make_fns = {
        "11": make_I1_11,
        "22": make_I1_22,
        "33": make_I1_33,
        "12": make_I1_12,
        "13": make_I1_13,
        "23": make_I1_23,
    }
    make_fn = make_fns[pair_key]
    term = make_fn(theta, R, kernel_regime="paper")
    result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
    return result.value


def compute_I2_paper_regime(pair_key, theta, R, polynomials, n_quad=60):
    """Compute I2 using term DSL with paper regime."""
    make_fns = {
        "11": make_I2_11,
        "22": make_I2_22,
        "33": make_I2_33,
        "12": make_I2_12,
        "13": make_I2_13,
        "23": make_I2_23,
    }
    make_fn = make_fns[pair_key]
    term = make_fn(theta, R, kernel_regime="paper")
    result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
    return result.value


def compute_S12_paper_regime(R, theta, polynomials, n_quad=60, use_factorial_norm=True):
    """
    Compute total S12 (I1 + I2 summed over all pairs) using paper regime.

    Uses diagonal + 2×off-diagonal convention, matching compute_c_paper_with_mirror.

    Args:
        use_factorial_norm: If True, applies 1/(ℓ₁!×ℓ₂!) normalization (default True)
    """
    pairs_diag = ["11", "22", "33"]
    pairs_offdiag = ["12", "13", "23"]

    # Factorial normalization factors (matching evaluate.py)
    f_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 0.25
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 0.5
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    }

    total = 0.0
    breakdown = {}

    for pair in pairs_diag:
        I1 = compute_I1_paper_regime(pair, theta, R, polynomials, n_quad)
        I2 = compute_I2_paper_regime(pair, theta, R, polynomials, n_quad)

        norm = f_norm[pair] if use_factorial_norm else 1.0
        sym = 1.0  # diagonal
        full_norm = sym * norm

        pair_contrib = full_norm * (I1 + I2)
        total += pair_contrib
        breakdown[pair] = {"I1": I1, "I2": I2, "I1_norm": norm * I1, "I2_norm": norm * I2,
                          "contrib": pair_contrib, "mult": 1, "norm": norm}

    for pair in pairs_offdiag:
        I1 = compute_I1_paper_regime(pair, theta, R, polynomials, n_quad)
        I2 = compute_I2_paper_regime(pair, theta, R, polynomials, n_quad)

        norm = f_norm[pair] if use_factorial_norm else 1.0
        sym = 2.0  # off-diagonal
        full_norm = sym * norm

        pair_contrib = full_norm * (I1 + I2)
        total += pair_contrib
        breakdown[pair] = {"I1": I1, "I2": I2, "I1_norm": norm * I1, "I2_norm": norm * I2,
                          "contrib": pair_contrib, "mult": 2, "norm": norm}

    return total, breakdown


def run_mirror_analysis(benchmark_name, R, theta, polynomials):
    """Run mirror analysis for one benchmark."""

    print(f"\n{'='*80}")
    print(f"BENCHMARK: {benchmark_name} (R={R}, theta={theta:.6f})")
    print(f"{'='*80}")

    # Compute S12(+R) and S12(-R) both in paper regime WITH factorial normalization
    S12_plusR, breakdown_plus = compute_S12_paper_regime(R, theta, polynomials, n_quad=60, use_factorial_norm=True)
    S12_minusR, breakdown_minus = compute_S12_paper_regime(-R, theta, polynomials, n_quad=60, use_factorial_norm=True)

    print(f"\nS12 at +R = {R} (with factorial normalization):")
    print(f"  S12_total(+R) = {S12_plusR:.6f}")

    print(f"\nS12 at -R = {-R} (with factorial normalization):")
    print(f"  S12_total(-R) = {S12_minusR:.6f}")

    # Per-pair breakdown (normalized values)
    print(f"\nPer-pair breakdown (normalized I1 values, with sym×norm):")
    print(f"{'Pair':<8} {'I1_norm(+R)':<14} {'I1_norm(-R)':<14} {'Contrib(+R)':<14}")
    print("-" * 60)

    for pair in ["11", "22", "33", "12", "13", "23"]:
        I1_norm_plus = breakdown_plus[pair]["I1_norm"] * breakdown_plus[pair]["mult"]
        I1_norm_minus = breakdown_minus[pair]["I1_norm"] * breakdown_minus[pair]["mult"]
        contrib_plus = breakdown_plus[pair]["contrib"]

        print(f"{pair:<8} {I1_norm_plus:>12.6e}  {I1_norm_minus:>12.6e}  {contrib_plus:>12.6e}")

    # Compute effective m
    print(f"\n{'='*60}")
    print("MIRROR MULTIPLIER ANALYSIS")
    print(f"{'='*60}")

    m_empirical = math.exp(R) + 5

    # The formula is: c = S12(+R) + m × S12(-R) + S34(+R)
    # We're checking if S12(+R) / S12(-R) relates to the empirical m

    if abs(S12_minusR) > 1e-15:
        ratio_S12 = S12_plusR / S12_minusR
        print(f"\nS12(+R) / S12(-R) = {ratio_S12:.4f}")
        print(f"\nEmpirical m = exp(R) + 5 = {m_empirical:.4f}")
        print(f"Ratio / m_empirical = {ratio_S12 / m_empirical:.4f}")

    # Also compute m × S12(-R) and compare to expected mirror contribution
    mirror_contrib = m_empirical * S12_minusR
    print(f"\nm × S12(-R) = {m_empirical:.4f} × {S12_minusR:.6f} = {mirror_contrib:.6f}")

    # What would c be using these values (need S34 too)
    print(f"\nPartial c = S12(+R) + m×S12(-R) = {S12_plusR + mirror_contrib:.6f}")
    print("  (add S34(+R) ≈ -0.6 to get full c)")
    print(f"  Estimated c ≈ {S12_plusR + mirror_contrib - 0.6:.4f} (target: 2.138)")

    # Per-pair comparison of normalized I1 values
    print(f"\nPer-pair I1 ratio (normalized, +R/-R):")
    print(f"{'Pair':<8} {'ratio':<12} {'vs emp m':<12}")
    print("-" * 40)

    for pair in ["11", "22", "33", "12", "13", "23"]:
        I1_norm_plus = breakdown_plus[pair]["I1_norm"] * breakdown_plus[pair]["mult"]
        I1_norm_minus = breakdown_minus[pair]["I1_norm"] * breakdown_minus[pair]["mult"]

        if abs(I1_norm_minus) > 1e-15:
            ratio = I1_norm_plus / I1_norm_minus
            ratio_to_emp = ratio / m_empirical
            print(f"{pair:<8} {ratio:>10.4f}   {ratio_to_emp:>10.2%}")
        else:
            print(f"{pair:<8} {'N/A':<10}   {'N/A':<10}")

    return {
        "S12_plusR": S12_plusR,
        "S12_minusR": S12_minusR,
        "m_empirical": m_empirical,
        "breakdown_plus": breakdown_plus,
        "breakdown_minus": breakdown_minus,
    }


def main():
    """Main entry point."""

    print("="*80)
    print("PHASE 28: MIRROR TRANSFORM IN PAPER REGIME")
    print("="*80)
    print("""
Purpose: Compute m_eff entirely within paper regime, where m = exp(R) + 5
was originally calibrated.

The key question: When we compute S12(+R) and S12(-R) both using paper regime
(with Case C kernel attenuation), what is the relationship?

If m = exp(R) + 5 is correct, we should find:
    S12_mirror = m × S12_proxy(-R)
    where S12_mirror is what appears in the c formula.
""")

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    kappa_polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    theta = 4 / 7

    # Run analysis for kappa benchmark
    results_kappa = run_mirror_analysis("KAPPA", R=1.3036, theta=theta, polynomials=kappa_polys)

    # Summary
    print(f"\n{'#'*80}")
    print("SUMMARY")
    print(f"{'#'*80}")

    print("""
KEY FINDING:
The ratio S12(+R) / S12(-R) in paper regime reveals the implicit mirror
relationship when computed consistently within the same regime.

Compare this to:
- Phase 27's m_eff ≈ 3.94 (computed in raw regime - WRONG regime)
- Empirical m = exp(R) + 5 ≈ 8.68 (calibrated in paper regime)

If the ratio here matches the empirical m, the mirror derivation is validated.
""")


if __name__ == "__main__":
    main()
