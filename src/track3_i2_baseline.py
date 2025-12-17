"""
src/track3_i2_baseline.py
Track 3: I2-Only Baseline Diagnostic

This script tests whether the two-benchmark instability originates from:
A) Derivative extraction (I1, I3, I4) - if I2-only ratio is stable (~1.1)
B) Base integrals (I2) - if I2-only ratio is unstable (>>1.1)

I2 terms have NO derivative extraction - they evaluate P(u)^2 Q(t)^2 e^{2Rt} directly.
If they're stable, the problem is in series/derivative machinery.

PRZZ Reference: Section 6.2.1
"""

from __future__ import annotations
import math
from typing import Dict

from src.evaluate import evaluate_term
from src.terms_k3_d1 import make_all_terms_k3
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.composition import PolyLike


def evaluate_i2_only(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    use_factorial_normalization: bool = True
) -> Dict[str, float]:
    """
    Evaluate ONLY I2 terms (no derivatives) and return per-pair breakdown.

    Returns dict with:
    - "I2_11", "I2_22", etc.: raw I2 values
    - "c_11", "c_22", etc.: normalized pair contributions (factorial + symmetry)
    - "total": sum of all normalized contributions
    """
    all_terms = make_all_terms_k3(theta, R)

    # Normalization factors
    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),
    }

    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0
    }

    results = {}
    total = 0.0

    for pair_key, terms in all_terms.items():
        # I2 is at index 1 in each pair's term list
        # (I1 at 0, I2 at 1, I3 at 2, I4 at 3)
        i2_term = terms[1]

        # Verify it's actually I2
        assert i2_term.name.startswith("I2_"), f"Expected I2 term, got {i2_term.name}"

        # Evaluate
        result = evaluate_term(i2_term, polynomials, n, R=R, theta=theta)
        results[i2_term.name] = result.value

        # Apply normalization
        if use_factorial_normalization:
            norm = factorial_norm[pair_key]
        else:
            norm = 1.0

        sym = symmetry_factor[pair_key]
        normalized_contrib = sym * norm * result.value
        results[f"c_{pair_key}"] = normalized_contrib
        total += normalized_contrib

    results["total"] = total
    return results


def compute_c_from_kappa(kappa: float, R: float) -> float:
    """Compute c from kappa using c = exp(R*(1-kappa))."""
    return math.exp(R * (1 - kappa))


def run_i2_baseline_test(n: int = 80):
    """
    Run the I2-only baseline test on both benchmarks.

    This is the key diagnostic for Track 3:
    - If I2-only ratio is ~1.1: derivative extraction is the problem
    - If I2-only ratio is ~2.1: base integral is the problem
    """
    theta = 4.0 / 7.0

    # Target values
    R_kappa = 1.3036
    kappa_target = 0.417293962
    c_kappa_target = compute_c_from_kappa(kappa_target, R_kappa)

    R_kappa_star = 1.1167
    kappa_star_target = 0.407511457
    c_kappa_star_target = compute_c_from_kappa(kappa_star_target, R_kappa_star)

    # PRZZ c ratio (target stability)
    przz_c_ratio = c_kappa_target / c_kappa_star_target

    print("=" * 70)
    print("TRACK 3: I2-ONLY BASELINE TEST")
    print("=" * 70)
    print(f"\nQuadrature: n = {n}")
    print(f"PRZZ c_kappa / c_kappa* ratio (target): {przz_c_ratio:.4f}")

    # Load polynomials
    print("\n" + "-" * 40)
    print("Loading polynomials...")
    polys_kappa = load_przz_polynomials(enforce_Q0=True)
    polys_kappa_star = load_przz_polynomials_kappa_star(enforce_Q0=True)

    # Convert to dict format
    P1_k, P2_k, P3_k, Q_k = polys_kappa
    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = polys_kappa_star
    polys_ks = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    # Evaluate I2-only for kappa benchmark
    print("\n" + "-" * 40)
    print("BENCHMARK 1: kappa polynomials @ R = 1.3036")
    print("-" * 40)

    results_k = evaluate_i2_only(theta, R_kappa, n, polys_k)

    print("\nPer-pair I2 values (raw):")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        print(f"  I2_{pair}: {results_k[f'I2_{pair}']:+12.6f}")

    print("\nPer-pair contributions (normalized):")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        print(f"  c_{pair}:  {results_k[f'c_{pair}']:+12.6f}")

    print(f"\nI2-only total: {results_k['total']:.10f}")

    # Evaluate I2-only for kappa* benchmark
    print("\n" + "-" * 40)
    print("BENCHMARK 2: kappa* polynomials @ R = 1.1167")
    print("-" * 40)

    results_ks = evaluate_i2_only(theta, R_kappa_star, n, polys_ks)

    print("\nPer-pair I2 values (raw):")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        print(f"  I2_{pair}: {results_ks[f'I2_{pair}']:+12.6f}")

    print("\nPer-pair contributions (normalized):")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        print(f"  c_{pair}:  {results_ks[f'c_{pair}']:+12.6f}")

    print(f"\nI2-only total: {results_ks['total']:.10f}")

    # Compute ratios
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    i2_ratio = results_k['total'] / results_ks['total']
    print(f"\nI2-only ratio (kappa/kappa*): {i2_ratio:.4f}")
    print(f"PRZZ c ratio (target):         {przz_c_ratio:.4f}")

    # Per-pair ratio comparison
    print("\nPer-pair ratio comparison:")
    print(f"{'Pair':<6} {'kappa I2':>12} {'kappa* I2':>12} {'Ratio':>8}")
    print("-" * 40)
    for pair in ["11", "22", "33", "12", "13", "23"]:
        val_k = results_k[f'I2_{pair}']
        val_ks = results_ks[f'I2_{pair}']
        if abs(val_ks) > 1e-12:
            ratio = val_k / val_ks
            print(f"  {pair:<4} {val_k:+12.6f} {val_ks:+12.6f} {ratio:8.2f}")
        else:
            print(f"  {pair:<4} {val_k:+12.6f} {val_ks:+12.6f}     N/A")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Full DSL ratio for comparison (known from HANDOFF_SUMMARY)
    full_dsl_ratio = 1.960 / 0.937  # Approximately 2.09

    if abs(i2_ratio - przz_c_ratio) < 0.2:
        print(f"\nRESULT: I2-only ratio ({i2_ratio:.2f}) is STABLE (close to PRZZ {przz_c_ratio:.2f})")
        print("\n  -> The instability is in DERIVATIVE EXTRACTION (I1, I3, I4)")
        print("  -> Next step: Build (1,2) oracle to debug derivative terms")
        stability = "stable"
    elif abs(i2_ratio - full_dsl_ratio) < 0.3:
        print(f"\nRESULT: I2-only ratio ({i2_ratio:.2f}) matches FULL DSL ({full_dsl_ratio:.2f})")
        print("\n  -> I2 terms DOMINATE the instability")
        print("  -> The base P^2 Q^2 e^{2Rt} integral is polynomial-sensitive")
        print("  -> Next step: Investigate polynomial evaluation or numerical precision")
        stability = "matches_full"
    else:
        print(f"\nRESULT: I2-only ratio ({i2_ratio:.2f}) is UNSTABLE")
        print(f"  - PRZZ target ratio: {przz_c_ratio:.2f}")
        print(f"  - Full DSL ratio:    {full_dsl_ratio:.2f}")
        print("\n  -> Base integral has intermediate sensitivity")
        print("  -> Both I2 and derivative terms may contribute")
        stability = "intermediate"

    print("\n" + "=" * 70)

    return {
        "results_kappa": results_k,
        "results_kappa_star": results_ks,
        "i2_ratio": i2_ratio,
        "przz_c_ratio": przz_c_ratio,
        "stability": stability
    }


if __name__ == "__main__":
    run_i2_baseline_test(n=80)
