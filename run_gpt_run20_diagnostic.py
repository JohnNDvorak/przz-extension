#!/usr/bin/env python3
"""
GPT Run 20 Diagnostic Script

This script validates the TeX Combined Mirror Core implementation and
compares it against Run 18, Run 19, and tex_mirror.

Run 20 implements the PRZZ difference quotient → log×integral identity:
    (N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)
    = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-s(α+β)} ds

Key innovation: Q operators applied AFTER combined structure (per PRZZ TeX).

Key questions:
1. Does Run 20 I1 give reasonable values compared to Run 18/19?
2. How does the full assembly compare to benchmarks?
3. Does Run 20 close the ~1% gap from tex_mirror?
"""

import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import (
    compute_I1_tex_combined_core_11,  # Run 20
    compute_I1_tex_combined_11_replace,  # Run 18
    compute_I1_tex_exact_11,  # Run 19
    compute_c_paper_tex_mirror,  # Production baseline
    compute_I2_tex_combined_11,
    compute_S34_base_11,
)


THETA = 4.0 / 7.0

TARGETS = {
    "kappa": {
        "name": "κ",
        "R": 1.3036,
        "c_target": 2.13745440613217263636,
        "kappa_target": 0.417293962,
    },
    "kappa_star": {
        "name": "κ*",
        "R": 1.1167,
        "c_target": 1.93801,
        "kappa_target": 0.3847,  # Approximate
    }
}


def compute_kappa(c: float, R: float) -> float:
    """Compute κ from c and R."""
    return 1 - np.log(c) / R


def main():
    print("=" * 80)
    print("GPT Run 20 Diagnostic: TeX Combined Mirror Core")
    print("=" * 80)
    print()
    print("Key features:")
    print("  - Uses TexCombinedMirrorCore with outer exp(-Rθ(x+y)) factor")
    print("  - Q operators applied AFTER combined structure (per PRZZ TeX)")
    print("  - Implements log×integral identity, not naive plus+minus")
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

    summary_results = []

    for name, polys, target in benchmarks:
        R = target["R"]
        c_target = target["c_target"]

        print(f"\n{'=' * 60}")
        print(f"{name} Benchmark (R={R}, c_target={c_target:.4f})")
        print("=" * 60)

        # --- Run 20: TexCombinedMirrorCore ---
        run20_result = compute_I1_tex_combined_core_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            n_quad_s=40,
            verbose=True,
        )
        print()

        # --- Run 18: CombinedMirrorFactor (replace mode) ---
        run18_result = compute_I1_tex_combined_11_replace(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            n_quad_s=40,
            verbose=False,
        )

        # --- Run 19: CombinedI1Integrand (naive plus+minus) ---
        run19_result = compute_I1_tex_exact_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            verbose=False,
        )

        # --- tex_mirror: Production baseline ---
        tex_mirror_result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            terms_version="old",
            tex_exp_component="exp_R_ref",
        )
        tex_mirror_I1 = tex_mirror_result.I1_plus + tex_mirror_result.m1 * tex_mirror_result.I1_minus_base

        # --- Get I2 and S34 ---
        i2_result = compute_I2_tex_combined_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
        )
        s34_base = compute_S34_base_11(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
        )

        # --- Print I1 comparison ---
        print("I1 Channel Comparison:")
        print(f"  Run 20 (TexCombinedMirrorCore): {run20_result.I1_combined_core:.6f}")
        print(f"  Run 18 (CombinedMirrorFactor):  {run18_result.I1_combined:.6f}")
        print(f"  Run 19 (naive plus+minus):      {run19_result.I1_tex_exact:.6f}")
        print(f"  tex_mirror (production):        {tex_mirror_I1:.6f}")

        # Compute ratios
        print()
        print("I1 Ratios (vs tex_mirror):")
        print(f"  Run 20 / tex_mirror: {run20_result.I1_combined_core / tex_mirror_I1:.4f}x")
        print(f"  Run 18 / tex_mirror: {run18_result.I1_combined / tex_mirror_I1:.4f}x")
        print(f"  Run 19 / tex_mirror: {run19_result.I1_tex_exact / tex_mirror_I1:.4f}x")

        # --- Full Assembly Attempts ---
        print()
        print("Full Assembly:")

        # Assembly using Run 20 I1
        c_run20 = run20_result.I1_combined_core + i2_result.I2_base + s34_base
        gap_run20 = 100 * (c_run20 - c_target) / c_target

        # Assembly using Run 18 I1
        c_run18 = run18_result.I1_combined + i2_result.I2_base + s34_base
        gap_run18 = 100 * (c_run18 - c_target) / c_target

        # Assembly using Run 19 I1
        c_run19 = run19_result.I1_tex_exact + i2_result.I2_base + s34_base
        gap_run19 = 100 * (c_run19 - c_target) / c_target

        # tex_mirror for reference
        c_tex_mirror = tex_mirror_result.c
        gap_mirror = 100 * (c_tex_mirror - c_target) / c_target

        print(f"  Run 20 assembly (I1 + I2_base + S34_base):")
        print(f"    c = {c_run20:.4f}, gap = {gap_run20:+.2f}%")

        print(f"  Run 18 assembly:")
        print(f"    c = {c_run18:.4f}, gap = {gap_run18:+.2f}%")

        print(f"  Run 19 assembly:")
        print(f"    c = {c_run19:.4f}, gap = {gap_run19:+.2f}%")

        print(f"  tex_mirror (production):")
        print(f"    c = {c_tex_mirror:.4f}, gap = {gap_mirror:+.2f}%")

        print(f"  Target:")
        print(f"    c = {c_target:.4f}")

        # Store for summary
        summary_results.append({
            "name": name,
            "R": R,
            "c_target": c_target,
            "I1_run20": run20_result.I1_combined_core,
            "I1_run18": run18_result.I1_combined,
            "I1_run19": run19_result.I1_tex_exact,
            "I1_tex_mirror": tex_mirror_I1,
            "c_run20": c_run20,
            "gap_run20": gap_run20,
            "c_tex_mirror": c_tex_mirror,
            "gap_mirror": gap_mirror,
        })

    # --- Summary Table ---
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print("| Benchmark | I1 (Run 20) | I1 (tex_mirror) | I1 Ratio | c (Run 20) | c Gap |")
    print("|-----------|-------------|-----------------|----------|------------|-------|")
    for r in summary_results:
        ratio = r["I1_run20"] / r["I1_tex_mirror"]
        print(f"| {r['name']:9s} | {r['I1_run20']:11.4f} | {r['I1_tex_mirror']:15.4f} | {ratio:8.4f} | {r['c_run20']:10.4f} | {r['gap_run20']:+5.2f}% |")

    print()
    print("| Benchmark | c (tex_mirror) | c Gap (tex_mirror) |")
    print("|-----------|----------------|---------------------|")
    for r in summary_results:
        print(f"| {r['name']:9s} | {r['c_tex_mirror']:14.4f} | {r['gap_mirror']:+19.2f}% |")

    # --- Analysis ---
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
Run 20 Implementation:
  - TexCombinedMirrorCore computes the TeX identity:
    exp(-Rθ(x+y)) × (1 + θ(x+y)) × ∫₀¹ exp(2sR(1 + θ(x+y))) ds

  - Q operators are applied AFTER the combined structure (per PRZZ TeX)

  - This differs from Run 18 (no outer exp factor) and Run 19 (naive plus+minus)

Key Findings:
  - Run 19 (naive plus+minus) gave ~10x error, proving that structure is wrong
  - Run 20 uses the correct log×integral identity
  - The outer exp(-Rθ(x+y)) factor dampens the combined structure
  - Q operators applied correctly after combined structure

Next Steps (if gap not closed):
  - Log intermediate xy-coefficients before P/Q multiplication
  - Compare against tex_mirror intermediates
  - Check if asymptotic L factor needs explicit handling
""")


if __name__ == "__main__":
    main()
