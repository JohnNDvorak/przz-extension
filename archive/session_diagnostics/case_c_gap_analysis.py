"""
src/case_c_gap_analysis.py
Analyze the gap from non-(1,1) pairs to identify Case C integral needs.
"""

import math

def analyze_case_c_gap():
    """Analyze what non-(1,1) pairs need to contribute."""

    # Results from previous analysis
    # Full computed c
    c_full_k = 1.950064
    c_full_ks = 0.823073

    # (1,1) pair contributions
    c_11_k = 0.441931
    c_11_ks = 0.374153

    # Target c values
    c_target_k = 2.13745440613217263636
    c_target_ks = 1.9379524124677437

    # Non-(1,1) contributions
    c_other_k = c_full_k - c_11_k
    c_other_ks = c_full_ks - c_11_ks

    # What non-(1,1) SHOULD contribute to hit target
    c_other_target_k = c_target_k - c_11_k
    c_other_target_ks = c_target_ks - c_11_ks

    print("\n" + "=" * 70)
    print("CASE C GAP ANALYSIS")
    print("=" * 70)

    print("\n--- (1,1) Pair (B×B - No Case C) ---")
    print(f"  κ:  c(1,1) = {c_11_k:.6f}")
    print(f"  κ*: c(1,1) = {c_11_ks:.6f}")

    print("\n--- Non-(1,1) Pairs (Need Case C Integrals) ---")
    print(f"  κ:  c(other) computed = {c_other_k:.6f}")
    print(f"  κ:  c(other) needed   = {c_other_target_k:.6f}")
    print(f"  κ:  Gap = {c_other_target_k - c_other_k:.6f} ({(c_other_target_k / c_other_k - 1)*100:+.1f}% missing)")

    print(f"\n  κ*: c(other) computed = {c_other_ks:.6f}")
    print(f"  κ*: c(other) needed   = {c_other_target_ks:.6f}")
    print(f"  κ*: Gap = {c_other_target_ks - c_other_ks:.6f} ({(c_other_target_ks / c_other_ks - 1)*100:+.1f}% missing)")

    print("\n--- Multiplicative Factors Needed ---")
    factor_other_k = c_other_target_k / c_other_k if abs(c_other_k) > 1e-10 else float('inf')
    factor_other_ks = c_other_target_ks / c_other_ks if abs(c_other_ks) > 1e-10 else float('inf')

    print(f"  κ:  non-(1,1) factor = {factor_other_k:.4f}")
    print(f"  κ*: non-(1,1) factor = {factor_other_ks:.4f}")
    print(f"  Factor ratio (κ*/κ) = {factor_other_ks / factor_other_k:.4f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print("""
For κ benchmark:
  - Non-(1,1) pairs need to contribute ~12% more
  - This is plausible for missing Case C integrals

For κ* benchmark:
  - Non-(1,1) pairs need to contribute ~248% more (3.48x!)
  - This is TOO LARGE to be explained by Case C integrals alone

Key observation:
  The κ* polynomials have MUCH smaller P₂/P₃ coefficients.
  If Case C integrals scale with polynomial norms, κ* Case C
  contributions would also be smaller - making the gap WORSE.

This suggests either:
1. PRZZ's c definition includes terms beyond I₁-I₄
2. There's a polynomial normalization factor we're missing
3. The κ* polynomials are transcribed incorrectly
4. PRZZ computes κ and κ* using different methodologies
""")

    print("\n--- Decomposition Check ---")
    print(f"  κ: (1,1) = {c_11_k/c_target_k*100:.1f}%, other = {c_other_k/c_target_k*100:.1f}% → total = {c_full_k/c_target_k*100:.1f}%")
    print(f"  κ*: (1,1) = {c_11_ks/c_target_ks*100:.1f}%, other = {c_other_ks/c_target_ks*100:.1f}% → total = {c_full_ks/c_target_ks*100:.1f}%")

    # What if (1,1) were scaled to hit target?
    scale_to_11_k = c_target_k / c_11_k
    scale_to_11_ks = c_target_ks / c_11_ks

    print("\n--- What If (1,1) Were Scaled to Full c? ---")
    print(f"  κ:  Scale factor = {scale_to_11_k:.4f}")
    print(f"  κ*: Scale factor = {scale_to_11_ks:.4f}")
    print(f"  Ratio = {scale_to_11_ks / scale_to_11_k:.4f}")

    # The ratio being close to 1 would suggest a simple scaling
    if abs(scale_to_11_ks / scale_to_11_k - 1) < 0.2:
        print("  → Ratios are similar! Simple scaling might work.")
    else:
        print("  → Ratios differ significantly - structure is different.")


if __name__ == "__main__":
    analyze_case_c_gap()
