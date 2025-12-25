#!/usr/bin/env python3
"""
run_phase9_derived_mirror_check.py
Phase 9.3A: Diagnostic runner for derived mirror term investigation.

This script compares:
1. DSL "minus basis" approach: S12(-R) with unchanged Q
2. TeX derived mirror: exp(2R) × I_shifted_Q(+R) with Q(1+D)

Outputs for each benchmark (κ, κ*):
- S12_plus, S12_minus_basis, S12_mirror_derived, S34
- m1_empirical vs m1_derived (implied from derived/minus ratio)
- c_empirical vs c_derived
- Gap vs target

Key insight from Phase 8:
The DSL "minus basis" is NOT the TeX mirror term.
This diagnostic reveals their relationship.
"""

import sys
import math
import numpy as np

# Add src to path
sys.path.insert(0, "src")

from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from mirror_exact import (
    compute_S12_mirror_derived,
    compute_S12_minus_basis,
    compute_I1_mirror_derived,
    compute_I2_mirror_derived,
)
from evaluate import compute_c_paper_ordered


# Benchmark definitions
BENCHMARKS = {
    "kappa": {
        "name": "κ benchmark",
        "R": 1.3036,
        "theta": 4 / 7,
        "c_target": 2.13745440613217263636,
        "kappa_target": 0.417293962,
        "load_fn": load_przz_polynomials,
    },
    "kappa_star": {
        "name": "κ* benchmark",
        "R": 1.1167,
        "theta": 4 / 7,
        "c_target": 1.93795,  # Approximate target
        "kappa_target": 0.41,  # Approximate target
        "load_fn": load_przz_polynomials_kappa_star,
    },
}


def format_value(v: float, width: int = 12) -> str:
    """Format a value for aligned output."""
    if abs(v) < 0.01 or abs(v) > 1000:
        return f"{v:>{width}.6e}"
    return f"{v:>{width}.6f}"


def compute_benchmark_diagnostics(benchmark_key: str, n: int = 60, verbose: bool = False):
    """
    Compute full diagnostics for a single benchmark.

    Returns dict with all computed values.
    """
    config = BENCHMARKS[benchmark_key]
    R = config["R"]
    theta = config["theta"]
    c_target = config["c_target"]
    kappa_target = config["kappa_target"]

    # Load polynomials
    P1, P2, P3, Q = config["load_fn"]()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Compute using existing evaluator (empirical m1)
    result_empirical = compute_c_paper_ordered(
        theta=theta, R=R, n=n, polynomials=polynomials,
        s12_pair_mode="triangle"
    )

    # Extract channel values from per_term dict (underscore-prefixed keys)
    S12_plus = result_empirical.per_term.get("_S12_plus_total", 0.0)
    S34 = result_empirical.per_term.get("_S34_ordered_total", 0.0)

    # DSL minus basis
    S12_minus_basis = compute_S12_minus_basis(
        theta=theta, R=R, n=n, polynomials=polynomials
    )

    # TeX derived mirror
    S12_mirror_derived = compute_S12_mirror_derived(
        theta=theta, R=R, n=n, polynomials=polynomials
    )

    # Empirical m1
    m1_empirical = math.exp(R) + 5

    # Implied m1 from derived mirror
    m1_derived = S12_mirror_derived / S12_minus_basis if abs(S12_minus_basis) > 1e-15 else float('inf')

    # Compute c values
    c_empirical = S12_plus + m1_empirical * S12_minus_basis + S34
    c_derived = S12_plus + S12_mirror_derived + S34

    # Alternative: c using derived m1 on minus basis
    c_derived_via_m1 = S12_plus + m1_derived * S12_minus_basis + S34

    # Kappa values
    kappa_empirical = 1.0 - math.log(c_empirical) / R
    kappa_derived = 1.0 - math.log(c_derived) / R

    # Gaps vs target
    gap_empirical = (c_empirical - c_target) / c_target * 100
    gap_derived = (c_derived - c_target) / c_target * 100

    # Ratio analysis
    ratio_plus_minus = S12_plus / S12_minus_basis if abs(S12_minus_basis) > 1e-15 else float('inf')
    ratio_derived_minus = S12_mirror_derived / S12_minus_basis if abs(S12_minus_basis) > 1e-15 else float('inf')

    # T^{-(α+β)} weight
    T_weight = math.exp(2 * R)

    # a coefficient back-solve: m1_derived = a * exp(R) + 5
    a_coefficient = (m1_derived - 5) / math.exp(R) if math.isfinite(m1_derived) else float('nan')

    return {
        "benchmark": benchmark_key,
        "name": config["name"],
        "R": R,
        "theta": theta,
        "n": n,
        "c_target": c_target,
        "kappa_target": kappa_target,

        # Raw channel values
        "S12_plus": S12_plus,
        "S12_minus_basis": S12_minus_basis,
        "S12_mirror_derived": S12_mirror_derived,
        "S34": S34,

        # m1 values
        "m1_empirical": m1_empirical,
        "m1_derived": m1_derived,
        "m1_ratio": m1_derived / m1_empirical if math.isfinite(m1_derived) else float('inf'),
        "a_coefficient": a_coefficient,

        # c values
        "c_empirical": c_empirical,
        "c_derived": c_derived,
        "c_derived_via_m1": c_derived_via_m1,

        # kappa values
        "kappa_empirical": kappa_empirical,
        "kappa_derived": kappa_derived,

        # Gaps
        "gap_empirical_pct": gap_empirical,
        "gap_derived_pct": gap_derived,

        # Ratios
        "ratio_plus_minus": ratio_plus_minus,
        "ratio_derived_minus": ratio_derived_minus,
        "T_weight": T_weight,

        # Per-pair breakdown (optional)
        "per_pair": None,  # Can be filled in if needed
    }


def print_benchmark_report(diag: dict):
    """Print formatted report for a benchmark."""
    print("\n" + "=" * 70)
    print(f"  {diag['name']} (R = {diag['R']}, θ = {diag['theta']:.6f})")
    print("=" * 70)

    print("\n--- Channel Values ---")
    print(f"  S12_plus:           {format_value(diag['S12_plus'])}")
    print(f"  S12_minus_basis:    {format_value(diag['S12_minus_basis'])}  (DSL -R branch)")
    print(f"  S12_mirror_derived: {format_value(diag['S12_mirror_derived'])}  (TeX T^{{-s}} × Q(1+D))")
    print(f"  S34:                {format_value(diag['S34'])}")

    print("\n--- Key Ratios ---")
    print(f"  S12_plus / S12_minus:   {format_value(diag['ratio_plus_minus'])}")
    print(f"  S12_derived / S12_minus:{format_value(diag['ratio_derived_minus'])}  (= m1_derived)")
    print(f"  T^{{-(α+β)}} = exp(2R):   {format_value(diag['T_weight'])}")

    print("\n--- m₁ Analysis ---")
    print(f"  m1_empirical:       {format_value(diag['m1_empirical'])}  (exp(R) + 5)")
    print(f"  m1_derived:         {format_value(diag['m1_derived'])}  (S12_derived / S12_minus)")
    print(f"  Ratio:              {format_value(diag['m1_ratio'])}  (derived / empirical)")
    print(f"  a coefficient:      {format_value(diag['a_coefficient'])}  (m1_derived = a×exp(R) + 5)")

    print("\n--- c Values ---")
    print(f"  c_target:           {format_value(diag['c_target'])}")
    print(f"  c_empirical:        {format_value(diag['c_empirical'])}  (gap: {diag['gap_empirical_pct']:+.4f}%)")
    print(f"  c_derived:          {format_value(diag['c_derived'])}  (gap: {diag['gap_derived_pct']:+.4f}%)")

    print("\n--- κ Values ---")
    print(f"  κ_target:           {format_value(diag['kappa_target'])}")
    print(f"  κ_empirical:        {format_value(diag['kappa_empirical'])}")
    print(f"  κ_derived:          {format_value(diag['kappa_derived'])}")


def compute_per_pair_breakdown(benchmark_key: str, n: int = 60):
    """Compute breakdown by (ℓ₁, ℓ₂) pair."""
    config = BENCHMARKS[benchmark_key]
    R = config["R"]
    theta = config["theta"]

    P1, P2, P3, Q = config["load_fn"]()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    pairs = [
        (1, 1), (2, 2), (3, 3),  # diagonal
        (1, 2), (1, 3), (2, 3),  # off-diagonal
    ]

    results = []
    for ell1, ell2 in pairs:
        # I1 and I2 derived mirror
        I1_result = compute_I1_mirror_derived(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2
        )
        I2_result = compute_I2_mirror_derived(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2
        )

        I_total_derived = I1_result.value + I2_result.value

        results.append({
            "ell1": ell1,
            "ell2": ell2,
            "I1_derived": I1_result.value,
            "I2_derived": I2_result.value,
            "I_total_derived": I_total_derived,
            "T_weight": I1_result.T_weight,
            "I1_shifted_Q": I1_result.I_shifted_Q_plus_R,
        })

    return results


def compute_shift_effect_diagnostic(benchmark_key: str, n: int = 60):
    """Analyze how the Q(1+·) shift affects the integral values."""
    from mirror_exact import compute_I1_with_shifted_Q

    config = BENCHMARKS[benchmark_key]
    R = config["R"]
    theta = config["theta"]

    P1, P2, P3, Q = config["load_fn"]()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    print(f"\n--- Q Shift Effect Analysis ({config['name']}) ---")
    print(f"{'Pair':^8} {'I1_std(+R)':>12} {'I1_shift(+R)':>14} {'Ratio':>10}")
    print("-" * 50)

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    for ell1, ell2 in pairs:
        # Standard Q (shift=0)
        I1_standard = compute_I1_with_shifted_Q(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2, shift=0.0
        )

        # Shifted Q (shift=1)
        I1_shifted = compute_I1_with_shifted_Q(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2, shift=1.0
        )

        ratio = I1_shifted / I1_standard if abs(I1_standard) > 1e-15 else float('inf')

        pair_str = f"({ell1},{ell2})"
        print(f"{pair_str:^8} {I1_standard:>12.4e} {I1_shifted:>14.4e} {ratio:>10.4f}")


def compute_mirror_variants_comparison(benchmark_key: str, n: int = 60):
    """Compare different mirror formulations."""
    from mirror_exact import (
        compute_I1_with_shifted_Q,
        _compute_I2_with_shifted_Q,
        compute_S12_minus_basis,
    )

    config = BENCHMARKS[benchmark_key]
    R = config["R"]
    theta = config["theta"]

    P1, P2, P3, Q = config["load_fn"]()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0/36.0,
        "12": 0.5, "13": 1.0/6.0, "23": 1.0/12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    pairs = ["11", "22", "33", "12", "13", "23"]

    # Compute three variants
    S12_plus_std = 0.0       # Standard I(+R) with standard Q
    S12_plus_shifted = 0.0   # I(+R) with Q(1+·)
    exp_2R = math.exp(2 * R)

    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])
        full_norm = symmetry[pair_key] * factorial_norm[pair_key]

        # I1 and I2 at +R with standard Q
        I1_std = compute_I1_with_shifted_Q(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2, shift=0.0
        )
        I2_std = _compute_I2_with_shifted_Q(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2, shift=0.0
        )
        S12_plus_std += full_norm * (I1_std + I2_std)

        # I1 and I2 at +R with shifted Q
        I1_shift = compute_I1_with_shifted_Q(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2, shift=1.0
        )
        I2_shift = _compute_I2_with_shifted_Q(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2, shift=1.0
        )
        S12_plus_shifted += full_norm * (I1_shift + I2_shift)

    # DSL minus basis
    S12_minus_basis = compute_S12_minus_basis(
        theta=theta, R=R, n=n, polynomials=polynomials
    )

    # Mirror variants
    mirror_std = exp_2R * S12_plus_std       # exp(2R) × I(+R, std Q)
    mirror_shifted = exp_2R * S12_plus_shifted  # exp(2R) × I(+R, shifted Q)

    m1_empirical = math.exp(R) + 5

    # Implied m1 from each variant
    m1_from_std = mirror_std / S12_minus_basis if abs(S12_minus_basis) > 1e-15 else float('inf')
    m1_from_shifted = mirror_shifted / S12_minus_basis if abs(S12_minus_basis) > 1e-15 else float('inf')

    print(f"\n--- Mirror Variant Comparison ({config['name']}) ---")
    print(f"  exp(2R):              {exp_2R:>12.4f}")
    print(f"  S12(+R, std Q):       {S12_plus_std:>12.6f}")
    print(f"  S12(+R, shifted Q):   {S12_plus_shifted:>12.6f}")
    print(f"  S12(-R, std Q):       {S12_minus_basis:>12.6f}  (DSL minus basis)")
    print()
    print(f"  Mirror (std Q):       {mirror_std:>12.6f} = exp(2R) × S12(+R, std)")
    print(f"  Mirror (shifted Q):   {mirror_shifted:>12.6f} = exp(2R) × S12(+R, shifted)")
    print()
    print(f"  m1_empirical:         {m1_empirical:>12.4f}")
    print(f"  m1 from std mirror:   {m1_from_std:>12.4f} (ratio {m1_from_std/m1_empirical:.4f})")
    print(f"  m1 from shifted:      {m1_from_shifted:>12.4f} (ratio {m1_from_shifted/m1_empirical:.4f})")

    # What if we use S12(+R) / S12(-R) directly?
    ratio_plus_minus = S12_plus_std / S12_minus_basis if abs(S12_minus_basis) > 1e-15 else float('inf')
    print()
    print(f"  S12(+R) / S12(-R):    {ratio_plus_minus:>12.4f}")
    print(f"  exp(R):               {math.exp(R):>12.4f}")
    print(f"  exp(2R)/exp(R):       {exp_2R/math.exp(R):>12.4f} = exp(R)")


def print_per_pair_table(breakdown: list, benchmark_name: str):
    """Print per-pair breakdown as a table."""
    print(f"\n--- Per-Pair Breakdown ({benchmark_name}) ---")
    print(f"{'Pair':^8} {'I1_derived':>14} {'I2_derived':>14} {'I_total':>14} {'I1_shifted':>14}")
    print("-" * 70)

    for row in breakdown:
        pair_str = f"({row['ell1']},{row['ell2']})"
        print(f"{pair_str:^8} {row['I1_derived']:>14.6e} {row['I2_derived']:>14.6e} "
              f"{row['I_total_derived']:>14.6e} {row['I1_shifted_Q']:>14.6e}")


def main():
    """Run full diagnostics."""
    print("=" * 70)
    print("  PHASE 9.3A: Derived Mirror Term Diagnostics")
    print("  Comparing DSL 'minus basis' vs TeX mirror T^{-(α+β)}×Q(1+D)")
    print("=" * 70)

    n = 60  # Quadrature points
    print(f"\nUsing n = {n} quadrature points")

    all_diags = []

    for benchmark_key in ["kappa", "kappa_star"]:
        print(f"\nComputing {benchmark_key}...")
        diag = compute_benchmark_diagnostics(benchmark_key, n=n)
        all_diags.append(diag)
        print_benchmark_report(diag)

    # Summary comparison
    print("\n" + "=" * 70)
    print("  SUMMARY COMPARISON")
    print("=" * 70)

    print("\n{:15} {:>12} {:>12} {:>12} {:>12}".format(
        "Benchmark", "m1_emp", "m1_der", "a_coeff", "m1_ratio"))
    print("-" * 55)
    for diag in all_diags:
        print("{:15} {:>12.4f} {:>12.4f} {:>12.6f} {:>12.6f}".format(
            diag['benchmark'],
            diag['m1_empirical'],
            diag['m1_derived'],
            diag['a_coefficient'],
            diag['m1_ratio']))

    print("\n{:15} {:>12} {:>12} {:>12} {:>12}".format(
        "Benchmark", "c_target", "c_emp", "c_der", "gap_der%"))
    print("-" * 55)
    for diag in all_diags:
        print("{:15} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.4f}".format(
            diag['benchmark'],
            diag['c_target'],
            diag['c_empirical'],
            diag['c_derived'],
            diag['gap_derived_pct']))

    # Key insight
    print("\n" + "-" * 70)
    print("KEY INSIGHT:")
    print("-" * 70)
    kappa_diag = all_diags[0]
    print(f"""
  DSL 'minus basis' S12(-R) is NOT the TeX mirror term.

  The TeX mirror is: T^{{-(α+β)}} × I(-β,-α) with Q(1+D) operators
                   = exp(2R) × I_shifted_Q(+R)

  For κ benchmark:
    exp(2R) = {kappa_diag['T_weight']:.4f}
    S12_derived / S12_minus = {kappa_diag['m1_derived']:.4f}

  If m1 = a × exp(R) + 5, then a = {kappa_diag['a_coefficient']:.6f}
  (cf. fitted a ≈ 1.037 from Phase 8)
""")

    # Mirror variant comparison (key diagnostic)
    for benchmark_key in ["kappa", "kappa_star"]:
        compute_mirror_variants_comparison(benchmark_key, n=n)

    # Q shift effect analysis (always show - it's diagnostic)
    for benchmark_key in ["kappa", "kappa_star"]:
        compute_shift_effect_diagnostic(benchmark_key, n=n)

    # Per-pair breakdown (optional, verbose)
    if "--verbose" in sys.argv or "-v" in sys.argv:
        for benchmark_key in ["kappa", "kappa_star"]:
            breakdown = compute_per_pair_breakdown(benchmark_key, n=n)
            print_per_pair_table(breakdown, BENCHMARKS[benchmark_key]["name"])

    print("\nDone.")
    return all_diags


if __name__ == "__main__":
    main()
