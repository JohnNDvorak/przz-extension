#!/usr/bin/env python3
"""
GPT Step 2 Diagnostic Script

This script validates the operator-level mirror implementation and compares it
against Run 18, Run 19, Run 20, and tex_mirror.

GPT's "decisive experiment": Apply Q as actual differential operators (d/dα, d/dβ)
on the pre-identity bracket, rather than using the TeX combined identity.

The key insight is that the order of operations matters:
- Previous runs: Apply combined identity first, then multiply by Q
- This approach: Apply Q(D_α) × Q(D_β) to the bracket, then evaluate

Key questions:
1. Does the operator-level I1 converge as L → ∞?
2. How does it compare to tex_mirror's effective I1?
3. Does this identify the correct structural approach?
"""

import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.operator_level_mirror import compute_I1_operator_level_11
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
        "kappa_target": 0.3847,
    }
}


def compute_kappa(c: float, R: float) -> float:
    """Compute κ from c and R."""
    return 1 - np.log(c) / R


def main():
    print("=" * 80)
    print("GPT Step 2 Diagnostic: Operator-Level Mirror Computation")
    print("=" * 80)
    print()
    print("This implements GPT's decisive experiment:")
    print("  Apply Q(D_α) × Q(D_β) as differential operators on the pre-identity bracket")
    print("  Q(D) = Σⱼ qⱼ D^j where D = -1/logT × d/dα")
    print("  Then evaluate at α = β = -R/L")
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
    n_quad = 40  # Quadrature points for I1 computation

    for name, polys, target in benchmarks:
        R = target["R"]
        c_target = target["c_target"]

        print(f"\n{'=' * 60}")
        print(f"{name} Benchmark (R={R}, c_target={c_target:.4f})")
        print("=" * 60)

        # --- Test operator-level convergence with L ---
        print("\n--- Operator-Level L Convergence ---")
        L_values = [10.0, 20.0, 50.0]
        op_I1_values = []

        for L in L_values:
            try:
                op_result = compute_I1_operator_level_11(
                    theta=THETA,
                    R=R,
                    n=n_quad,
                    polynomials=polys,
                    L=L,
                    verbose=False,
                )
                op_I1_values.append((L, op_result.I1_operator_level))
                print(f"  L={L:5.1f}: I1_operator = {op_result.I1_operator_level:.6f}")
            except Exception as e:
                print(f"  L={L:5.1f}: ERROR - {e}")
                op_I1_values.append((L, float('nan')))

        # Use L=20 as the primary comparison value
        op_I1 = op_I1_values[1][1] if len(op_I1_values) > 1 else float('nan')

        # --- Run 20: TexCombinedMirrorCore ---
        print("\n--- Other Methods ---")
        try:
            run20_result = compute_I1_tex_combined_core_11(
                theta=THETA,
                R=R,
                n=60,
                polynomials=polys,
                n_quad_s=40,
                verbose=False,
            )
            run20_I1 = run20_result.I1_combined_core
        except Exception as e:
            print(f"  Run 20: ERROR - {e}")
            run20_I1 = float('nan')

        # --- Run 18: CombinedMirrorFactor (replace mode) ---
        try:
            run18_result = compute_I1_tex_combined_11_replace(
                theta=THETA,
                R=R,
                n=60,
                polynomials=polys,
                n_quad_s=40,
                verbose=False,
            )
            run18_I1 = run18_result.I1_combined
        except Exception as e:
            print(f"  Run 18: ERROR - {e}")
            run18_I1 = float('nan')

        # --- Run 19: CombinedI1Integrand (naive plus+minus) ---
        try:
            run19_result = compute_I1_tex_exact_11(
                theta=THETA,
                R=R,
                n=60,
                polynomials=polys,
                verbose=False,
            )
            run19_I1 = run19_result.I1_tex_exact
        except Exception as e:
            print(f"  Run 19: ERROR - {e}")
            run19_I1 = float('nan')

        # --- tex_mirror: Production baseline ---
        try:
            tex_mirror_result = compute_c_paper_tex_mirror(
                theta=THETA,
                R=R,
                n=60,
                polynomials=polys,
                terms_version="old",
                tex_exp_component="exp_R_ref",
            )
            tex_mirror_I1 = tex_mirror_result.I1_plus + tex_mirror_result.m1 * tex_mirror_result.I1_minus_base
        except Exception as e:
            print(f"  tex_mirror: ERROR - {e}")
            tex_mirror_I1 = float('nan')

        # --- Print I1 comparison ---
        print("\nI1 Channel Comparison:")
        print(f"  Operator-level (L=20):        {op_I1:.6f}")
        print(f"  Run 20 (TexCombinedMirrorCore): {run20_I1:.6f}")
        print(f"  Run 18 (CombinedMirrorFactor):  {run18_I1:.6f}")
        print(f"  Run 19 (naive plus+minus):      {run19_I1:.6f}")
        print(f"  tex_mirror (production):        {tex_mirror_I1:.6f}")

        # Compute ratios
        print()
        print("I1 Ratios (vs tex_mirror):")
        if not np.isnan(tex_mirror_I1) and tex_mirror_I1 != 0:
            print(f"  Operator-level / tex_mirror: {op_I1 / tex_mirror_I1:.4f}x")
            print(f"  Run 20 / tex_mirror: {run20_I1 / tex_mirror_I1:.4f}x")
            print(f"  Run 18 / tex_mirror: {run18_I1 / tex_mirror_I1:.4f}x")
            print(f"  Run 19 / tex_mirror: {run19_I1 / tex_mirror_I1:.4f}x")

        # --- Get I2 and S34 for assembly ---
        try:
            i2_result = compute_I2_tex_combined_11(
                theta=THETA,
                R=R,
                n=60,
                polynomials=polys,
            )
            I2_base = i2_result.I2_base
        except Exception as e:
            print(f"  I2: ERROR - {e}")
            I2_base = 0.0

        try:
            s34_base = compute_S34_base_11(
                theta=THETA,
                R=R,
                n=60,
                polynomials=polys,
            )
        except Exception as e:
            print(f"  S34: ERROR - {e}")
            s34_base = 0.0

        # --- Full Assembly Attempts ---
        print()
        print("Full Assembly (I1 + I2_base + S34_base):")

        # Assembly using operator-level I1
        c_op = op_I1 + I2_base + s34_base
        gap_op = 100 * (c_op - c_target) / c_target

        # Assembly using Run 20 I1
        c_run20 = run20_I1 + I2_base + s34_base
        gap_run20 = 100 * (c_run20 - c_target) / c_target

        # tex_mirror for reference
        c_tex_mirror = tex_mirror_result.c if 'tex_mirror_result' in dir() else float('nan')
        gap_mirror = 100 * (c_tex_mirror - c_target) / c_target if not np.isnan(c_tex_mirror) else float('nan')

        print(f"  Operator-level: c = {c_op:.4f}, gap = {gap_op:+.2f}%")
        print(f"  Run 20:         c = {c_run20:.4f}, gap = {gap_run20:+.2f}%")
        print(f"  tex_mirror:     c = {c_tex_mirror:.4f}, gap = {gap_mirror:+.2f}%")
        print(f"  Target:         c = {c_target:.4f}")

        # Store for summary
        summary_results.append({
            "name": name,
            "R": R,
            "c_target": c_target,
            "I1_operator": op_I1,
            "I1_run20": run20_I1,
            "I1_tex_mirror": tex_mirror_I1,
            "c_operator": c_op,
            "gap_operator": gap_op,
            "c_tex_mirror": c_tex_mirror,
            "gap_mirror": gap_mirror,
        })

    # --- Summary Table ---
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print("| Benchmark | I1 (Operator) | I1 (Run20) | I1 (tex_mirror) | Ratio (Op/mirror) |")
    print("|-----------|---------------|------------|-----------------|-------------------|")
    for r in summary_results:
        ratio = r["I1_operator"] / r["I1_tex_mirror"] if r["I1_tex_mirror"] != 0 else float('nan')
        print(f"| {r['name']:9s} | {r['I1_operator']:13.4f} | {r['I1_run20']:10.4f} | {r['I1_tex_mirror']:15.4f} | {ratio:17.4f} |")

    print()
    print("| Benchmark | c (Operator) | c Gap | c (tex_mirror) | c Gap (mirror) |")
    print("|-----------|--------------|-------|----------------|----------------|")
    for r in summary_results:
        print(f"| {r['name']:9s} | {r['c_operator']:12.4f} | {r['gap_operator']:+5.2f}% | {r['c_tex_mirror']:14.4f} | {r['gap_mirror']:+14.2f}% |")

    # --- Analysis ---
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
GPT Step 2 Operator-Level Computation:
  - Applies Q(D_α) × Q(D_β) as differential operators to the bracket
  - Q(D) = Σⱼ qⱼ D^j where D = -1/logT × d/dα
  - Evaluates at α = β = -R/L after applying operators

Key Questions Answered:

1. Does operator-level converge as L → ∞?
   (Check the L convergence data above)

2. How does it compare to tex_mirror?
   (See I1 ratio column)

3. Does it match any previous run?
   (Compare against Run 18, 19, 20)

Interpretation Guide:
  - If operator-level ≈ tex_mirror: tex_mirror is structurally correct
  - If operator-level ≈ Run 20: Combined identity is correct but needs different assembly
  - If operator-level differs from all: Need to re-examine term mapping
""")


if __name__ == "__main__":
    main()
