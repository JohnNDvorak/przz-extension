#!/usr/bin/env python3
"""
GPT Run 17C: Mirror Residual Truth Table

This script aggregates findings from 17A0, 17A, and 17B to build a comprehensive
table showing where the tex_mirror residual concentrates.

KEY FINDINGS FROM PREVIOUS RUNS:
- 17A0: Correct prefactor is exp(2R), NOT exp(2R/θ)
- 17A: I1_combined = I1_plus + exp(2R)×I1_minus exactly (m_implied = exp(2R))
       But I1_combined ≈ 13.3, way off from target c ≈ 2.14
- 17B: S34_combined = S34_plus + exp(2R)×S34_minus gives 8.96
       Also way off from target

CRITICAL INSIGHT:
The naive formula I = I_plus + exp(2R)×I_minus gives values 5-10x larger than target.
This means the TeX formula is NOT a simple additive mirror structure.

The TeX formula (lines 1503-1522) shows a DIFFERENCE, not a sum:
    (N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)

The integral in line 1510 integrates over the mirror combination:
    = N^{αx+βy} log(N^{x+y}T) ∫_0^1 (N^{x+y}T)^{-t(α+β)}dt

This is fundamentally different from I_plus + prefactor×I_minus.

Usage:
    python run_gpt_run17c_residual_table.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
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


@dataclass
class ResidualSummary:
    """Summary of residual analysis for one benchmark."""
    name: str
    R: float
    c_target: float
    c_tex_mirror: float
    c_gap_pct: float
    # 17A0 findings
    exp_2R: float
    exp_2R_theta: float
    # 17A findings
    I1_combined_naive: float
    I1_tex_mirror: float
    I1_delta_pct: float
    # 17B findings
    S34_combined_naive: float
    S34_tex_mirror: float
    S34_delta_pct: float
    # Total naive c
    c_naive_total: float
    c_naive_gap_pct: float


def compute_residual_summary(
    bench_name: str,
    R: float,
    c_target: float,
    polynomials: Dict,
) -> ResidualSummary:
    """Compute residual summary for one benchmark."""

    # Get tex_mirror results
    tex_result = compute_c_paper_tex_mirror(
        theta=THETA,
        R=R,
        n=60,
        polynomials=polynomials,
        terms_version="old",
        tex_exp_component="exp_R_ref",
    )

    c_tex_mirror = tex_result.c
    c_gap_pct = 100 * (c_tex_mirror - c_target) / c_target

    # Prefactors from 17A0
    exp_2R = np.exp(2 * R)
    exp_2R_theta = np.exp(2 * R / THETA)

    # 17A findings: I1 combined vs tex_mirror
    # From 17A run, I1_combined with exp(2R) prefactor
    # κ: ~13.32, κ*: ~5.91
    I1_combined_naive = 13.318 if bench_name == "κ" else 5.907
    I1_tex_mirror = tex_result.I1_plus + tex_result.m1 * tex_result.I1_minus_base
    I1_delta_pct = 100 * (I1_combined_naive - I1_tex_mirror) / c_target

    # 17B findings: S34 combined vs tex_mirror
    # From 17B run, S34_combined with exp(2R) prefactor
    # κ: ~8.96, κ*: ~4.39
    S34_combined_naive = 8.956 if bench_name == "κ" else 4.389
    S34_tex_mirror = tex_result.S34_plus
    S34_delta_pct = 100 * (S34_combined_naive - S34_tex_mirror) / c_target

    # Total naive c (I1 + I2 + S34 with naive exp(2R) assembly)
    # I2 doesn't have derivatives so I2_combined ≈ I2_plus + exp(2R)×I2_minus_base
    I2_plus = tex_result.I2_plus
    I2_minus = tex_result.I2_minus_base
    I2_combined_naive = I2_plus + exp_2R * I2_minus
    c_naive_total = I1_combined_naive + I2_combined_naive + S34_combined_naive
    c_naive_gap_pct = 100 * (c_naive_total - c_target) / c_target

    return ResidualSummary(
        name=bench_name,
        R=R,
        c_target=c_target,
        c_tex_mirror=c_tex_mirror,
        c_gap_pct=c_gap_pct,
        exp_2R=exp_2R,
        exp_2R_theta=exp_2R_theta,
        I1_combined_naive=I1_combined_naive,
        I1_tex_mirror=I1_tex_mirror,
        I1_delta_pct=I1_delta_pct,
        S34_combined_naive=S34_combined_naive,
        S34_tex_mirror=S34_tex_mirror,
        S34_delta_pct=S34_delta_pct,
        c_naive_total=c_naive_total,
        c_naive_gap_pct=c_naive_gap_pct,
    )


def main():
    print("=" * 90)
    print("GPT Run 17C: Mirror Residual Truth Table")
    print("=" * 90)
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

    results = []
    for bench_name, polys, target in benchmarks:
        summary = compute_residual_summary(
            bench_name, target["R"], target["c_target"], polys
        )
        results.append(summary)

    # Print truth table
    print("MIRROR RESIDUAL TRUTH TABLE")
    print("=" * 90)
    print()

    for s in results:
        print(f"Benchmark: {s.name} (R={s.R})")
        print("-" * 60)
        print(f"  c_target:        {s.c_target:.4f}")
        print(f"  c_tex_mirror:    {s.c_tex_mirror:.4f} ({s.c_gap_pct:+.2f}%)")
        print()
        print("  Prefactors:")
        print(f"    exp(2R):       {s.exp_2R:.2f}")
        print(f"    exp(2R/θ):     {s.exp_2R_theta:.2f} (Run 14's wrong value)")
        print()
        print("  I1 Analysis (17A):")
        print(f"    I1 combined (naive):  {s.I1_combined_naive:.4f}")
        print(f"    I1 tex_mirror:        {s.I1_tex_mirror:.4f}")
        print(f"    Delta:                {s.I1_delta_pct:+.2f}% of c_target")
        print()
        print("  S34 Analysis (17B):")
        print(f"    S34 combined (naive): {s.S34_combined_naive:.4f}")
        print(f"    S34 tex_mirror:       {s.S34_tex_mirror:.4f}")
        print(f"    Delta:                {s.S34_delta_pct:+.2f}% of c_target")
        print()
        print("  Total c with naive exp(2R) assembly:")
        print(f"    c_naive:              {s.c_naive_total:.4f}")
        print(f"    Gap:                  {s.c_naive_gap_pct:+.2f}%")
        print()

    # Print critical analysis
    print("=" * 90)
    print("CRITICAL ANALYSIS")
    print("=" * 90)
    print("""
1. PREFACTOR VERIFICATION (17A0):
   ✓ Correct prefactor is exp(2R), NOT exp(2R/θ)
   ✓ Run 14's exp(2R/θ) was 6-7x too large

2. I1 COMBINED MIRROR (17A):
   ✗ Naive assembly I1_plus + exp(2R)×I1_minus gives ~13.3 (κ) or ~5.9 (κ*)
   ✗ This is 5-6x larger than target c ≈ 2.14
   ✓ m_implied = exp(2R) exactly (derivative doesn't change weight)

3. S34 COMBINED MIRROR (17B):
   ✗ Naive assembly S34_plus + exp(2R)×S34_minus gives ~8.96 (κ)
   ✗ This is 4x larger than target c ≈ 2.14

4. TOTAL NAIVE c:
   ✗ c_naive = I1_naive + I2_naive + S34_naive ≈ 25-30
   ✗ This is 10-15x larger than target c ≈ 2.14

CONCLUSION:
===========
The naive formula "I = I(+R) + exp(2R)×I(-R)" is WRONG for this application.

Looking at TeX lines 1503-1510, the actual structure is a DIFFERENCE not a SUM:
    (N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)

This gets converted to an integral (line 1510):
    = N^{αx+βy} log(N^{x+y}T) ∫_0^1 (N^{x+y}T)^{-t(α+β)}dt

The Q operators are applied to this COMBINED expression, not to separate +R/-R.

tex_mirror's shape×amplitude factorization is an APPROXIMATION that works by:
1. Calibrating amplitudes A1, A2 to match specific reference R values
2. Using shape factors m_implied ≈ 1 to account for derivative structure
3. Achieving <1% accuracy through this heuristic

The ~1% structural gap CANNOT be closed by using exp(2R) directly because:
- Direct use gives 10-15x too large values
- The gap comes from the factorization approximation, not wrong prefactor

RECOMMENDED PATH FORWARD:
========================
1. Keep tex_mirror with exp_R_ref as production model (<1% gap is acceptable)
2. To improve further, need to implement TeX's actual combined integral structure
3. The integral form ∫_0^1 (N^{x+y}T)^{-t(α+β)}dt before extracting derivatives
4. This is a fundamentally different computation, not just a mirror weight change
""")


if __name__ == "__main__":
    main()
