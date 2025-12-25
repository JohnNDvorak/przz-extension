"""
run_ordered_pair_diagnostic.py
Diagnostic to test the ordered pair / mirror swap hypothesis.

Key Insight from Codex Analysis
-------------------------------
The audit showed that:
1. m_needed for Block12 only ≈ 8.61 (very close to exp(R)+5 ≈ 8.68)
2. Off-diagonal (1,2) pair explodes under naive exp(2R) mirroring
3. The mirror transform may require polynomial role swapping (12 ↔ 21)

PRZZ Mirror Transform Hypothesis
--------------------------------
For the mirror of I(α,β) → I(-β,-α), off-diagonal pairs swap roles:
- (1,2)_mirror ↔ (2,1)
- (1,3)_mirror ↔ (3,1)
- (2,3)_mirror ↔ (3,2)

This diagnostic compares:
- direct: upper-triangle pairs (11, 22, 33, 12, 13, 23) with symmetry factors
- swapped: ordered pairs where mirror uses swapped indices

If the hypothesis is correct, the swapped configuration should fix the
(1,2) explosion seen in the naive mirror audit.
"""

from __future__ import annotations

import math
from typing import Dict, List

from src.evaluate import evaluate_term
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.terms_k3_d1 import (
    make_all_terms_k3_ordered_v2,
)
from src.mirror_transform import transform_terms_exp_factors

# Constants
THETA = 4.0 / 7.0
N_QUAD = 60
N_QUAD_A = 40

# Normalization factors
FACTORIAL_NORM: Dict[str, float] = {
    "11": 1.0,
    "22": 1.0 / 4.0,
    "33": 1.0 / 36.0,
    "12": 1.0 / 2.0,
    "21": 1.0 / 2.0,
    "13": 1.0 / 6.0,
    "31": 1.0 / 6.0,
    "23": 1.0 / 12.0,
    "32": 1.0 / 12.0,
}


def evaluate_pair(
    pair_key: str,
    terms: List,
    polynomials: Dict,
    R: float,
) -> Dict[str, float]:
    """Evaluate a single pair and return I1..I4 contributions."""
    vals = [
        evaluate_term(term, polynomials, N_QUAD, R=R, theta=THETA, n_quad_a=N_QUAD_A).value
        for term in terms
    ]
    i1, i2, i3, i4 = (vals + [0.0, 0.0, 0.0, 0.0])[:4]

    return {
        "I1": i1,
        "I2": i2,
        "I3": i3,
        "I4": i4,
        "I12": i1 + i2,
        "I34": i3 + i4,
        "total": i1 + i2 + i3 + i4,
    }


def run_benchmark(name: str, R: float, c_target: float, polynomials: Dict):
    """Run the ordered pair diagnostic for one benchmark."""
    print()
    print("=" * 78)
    print(f"ORDERED PAIR DIAGNOSTIC: {name} (R={R})")
    print("=" * 78)

    # Generate all ordered terms
    all_terms_ordered = make_all_terms_k3_ordered_v2(THETA, R, kernel_regime="paper")

    # PART 1: Compare direct (12) vs swapped (21) for each off-diagonal
    print()
    print("Part 1: Direct vs Swapped Pair Comparison")
    print("-" * 78)
    print(f"{'Pair':>6}  {'direct I12':>14}  {'direct I34':>14}  {'swapped I12':>14}  {'swapped I34':>14}")

    direct_pairs = {}
    swapped_pairs = {}

    for base in ["12", "13", "23"]:
        swap = base[1] + base[0]  # "21", "31", "32"

        # Direct pair
        terms_direct = all_terms_ordered[base]
        r_direct = evaluate_pair(base, terms_direct, polynomials, R)
        direct_pairs[base] = r_direct

        # Swapped pair
        terms_swapped = all_terms_ordered[swap]
        r_swapped = evaluate_pair(swap, terms_swapped, polynomials, R)
        swapped_pairs[swap] = r_swapped

        norm = FACTORIAL_NORM[base]
        print(f"  {base}:  {norm * r_direct['I12']:+14.8f}  {norm * r_direct['I34']:+14.8f}  "
              f"{norm * r_swapped['I12']:+14.8f}  {norm * r_swapped['I34']:+14.8f}")

    # Also evaluate diagonal pairs (unchanged)
    print(f"\n{'Pair':>6}  {'I12':>14}  {'I34':>14}")
    for diag in ["11", "22", "33"]:
        terms = all_terms_ordered[diag]
        r = evaluate_pair(diag, terms, polynomials, R)
        direct_pairs[diag] = r
        norm = FACTORIAL_NORM[diag]
        print(f"  {diag}:  {norm * r['I12']:+14.8f}  {norm * r['I34']:+14.8f}")

    # PART 2: Build mirror using exp-sign flip on SWAPPED pairs
    print()
    print("Part 2: Mirror Transform with Swapped Pairs")
    print("-" * 78)

    # For diagonal pairs: mirror = exp-sign flip on same pair
    # For off-diagonal: mirror = exp-sign flip on SWAPPED pair
    direct_I12_total = 0.0
    direct_I34_total = 0.0
    mirror_I12_total = 0.0
    mirror_I34_total = 0.0

    print(f"{'Pair':>6}  {'direct I12':>14}  {'direct I34':>14}  {'mirror I12':>14}  {'mirror I34':>14}")

    # Diagonal pairs (11, 22, 33): mirror = exp-flip same pair
    for diag in ["11", "22", "33"]:
        terms = all_terms_ordered[diag]
        r_direct = evaluate_pair(diag, terms, polynomials, R)

        # Create exp-sign flipped terms
        terms_mirror = transform_terms_exp_factors(terms, scale_multiplier=-1.0)
        r_mirror = evaluate_pair(diag, terms_mirror, polynomials, R)

        norm = FACTORIAL_NORM[diag]
        direct_I12_total += norm * r_direct["I12"]
        direct_I34_total += norm * r_direct["I34"]
        mirror_I12_total += norm * r_mirror["I12"]
        mirror_I34_total += norm * r_mirror["I34"]

        print(f"  {diag}:  {norm * r_direct['I12']:+14.8f}  {norm * r_direct['I34']:+14.8f}  "
              f"{norm * r_mirror['I12']:+14.8f}  {norm * r_mirror['I34']:+14.8f}")

    # Off-diagonal pairs: direct = base pair, mirror = exp-flip SWAPPED pair
    for base in ["12", "13", "23"]:
        swap = base[1] + base[0]

        # Direct uses base pair with symmetry factor of 2
        terms_direct = all_terms_ordered[base]
        r_direct = evaluate_pair(base, terms_direct, polynomials, R)

        # Mirror uses SWAPPED pair with exp-sign flip
        terms_swap = all_terms_ordered[swap]
        terms_mirror = transform_terms_exp_factors(terms_swap, scale_multiplier=-1.0)
        r_mirror = evaluate_pair(swap, terms_mirror, polynomials, R)

        # Symmetry factor = 2 for off-diagonals
        norm = 2.0 * FACTORIAL_NORM[base]
        direct_I12_total += norm * r_direct["I12"]
        direct_I34_total += norm * r_direct["I34"]
        # For swapped mirror, we average contributions from both orderings
        # Actually the proper way: direct(12) + mirror(21) should both be counted
        # But the "symmetry factor 2" already accounts for 12+21 in direct...
        # For mirror, we use the swapped pair contribution
        mirror_I12_total += norm * r_mirror["I12"]
        mirror_I34_total += norm * r_mirror["I34"]

        print(f"  {base}/{swap}:  {norm * r_direct['I12']:+14.8f}  {norm * r_direct['I34']:+14.8f}  "
              f"{norm * r_mirror['I12']:+14.8f}  {norm * r_mirror['I34']:+14.8f}")

    direct_c = direct_I12_total + direct_I34_total
    mirror_c = mirror_I12_total + mirror_I34_total

    print()
    print("Totals (normalized):")
    print(f"  direct I12:  {direct_I12_total:+14.8f}")
    print(f"  direct I34:  {direct_I34_total:+14.8f}")
    print(f"  direct c:    {direct_c:+14.8f}")
    print()
    print(f"  mirror I12 (swapped): {mirror_I12_total:+14.8f}")
    print(f"  mirror I34 (swapped): {mirror_I34_total:+14.8f}")
    print(f"  mirror c (swapped):   {mirror_c:+14.8f}")

    # PART 3: Recombination tests
    print()
    print("Part 3: Recombination Analysis")
    print("-" * 78)

    exp_2R = math.exp(2.0 * R)
    exp_R_plus_5 = math.exp(R) + 5.0

    print(f"  c_target:   {c_target:+14.8f}")
    print(f"  exp(2R):    {exp_2R:+14.8f}")
    print(f"  exp(R)+5:   {exp_R_plus_5:+14.8f}")
    print()

    # Model A: direct + exp(2R) * mirror (both groups)
    c_model_a = direct_c + exp_2R * mirror_c
    print(f"Model A: direct + exp(2R)*mirror_all")
    print(f"  c = {direct_c:.6f} + {exp_2R:.6f} * {mirror_c:.6f} = {c_model_a:+14.8f}")
    print(f"  gap: {(c_model_a - c_target) / c_target * 100:+.2f}%")
    print()

    # Model B: direct + m * mirror_I12 only (no mirror for I34)
    # Solve for m: c_target = direct_c + m * mirror_I12
    if mirror_I12_total != 0:
        m_needed_B = (c_target - direct_c) / mirror_I12_total
    else:
        m_needed_B = float("inf")
    c_model_b = direct_c + exp_R_plus_5 * mirror_I12_total
    print(f"Model B: direct + m * mirror_I12_only")
    print(f"  m_needed to hit target: {m_needed_B:+14.8f}")
    print(f"  Using m = exp(R)+5 = {exp_R_plus_5:.6f}:")
    print(f"  c = {direct_c:.6f} + {exp_R_plus_5:.6f} * {mirror_I12_total:.6f} = {c_model_b:+14.8f}")
    print(f"  gap: {(c_model_b - c_target) / c_target * 100:+.2f}%")
    print()

    # Model C: Two-weight model (separate multipliers for I12 and I34)
    # c_target = direct_c + a * mirror_I12 + b * mirror_I34
    print(f"Model C: Two-weight model c = direct + a*mirror_I12 + b*mirror_I34")
    if mirror_I12_total != 0 and mirror_I34_total != 0:
        # We have 1 equation and 2 unknowns, so let's try b=0 (Model B) as baseline
        # Then try other combinations
        gap_from_direct = c_target - direct_c
        print(f"  gap_from_direct: {gap_from_direct:+14.8f}")
        print(f"  If b=0:  a = {gap_from_direct / mirror_I12_total:+14.8f}")
        print(f"  If a=0:  b = {gap_from_direct / mirror_I34_total:+14.8f}")

        # Try a = exp(2R), solve for b
        a_val = exp_2R
        b_val = (c_target - direct_c - a_val * mirror_I12_total) / mirror_I34_total
        print(f"  If a=exp(2R)={a_val:.4f}:  b = {b_val:+14.8f}")

        # Try a = exp(R)+5, solve for b
        a_val = exp_R_plus_5
        b_val = (c_target - direct_c - a_val * mirror_I12_total) / mirror_I34_total
        print(f"  If a=exp(R)+5={a_val:.4f}:  b = {b_val:+14.8f}")

    return {
        "direct_c": direct_c,
        "mirror_c": mirror_c,
        "direct_I12": direct_I12_total,
        "direct_I34": direct_I34_total,
        "mirror_I12": mirror_I12_total,
        "mirror_I34": mirror_I34_total,
    }


def main():
    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Run diagnostics
    result_k = run_benchmark("κ", R=1.3036, c_target=2.137, polynomials=polys_k)
    result_s = run_benchmark("κ*", R=1.1167, c_target=1.938, polynomials=polys_s)

    # Summary
    print()
    print("=" * 78)
    print("SUMMARY: Cross-Benchmark Comparison")
    print("=" * 78)
    print()

    ratio_direct = result_k["direct_c"] / result_s["direct_c"]
    ratio_mirror_I12 = result_k["mirror_I12"] / result_s["mirror_I12"]
    print(f"Ratio direct c:         {ratio_direct:.6f}  (target: 1.103)")
    print(f"Ratio mirror I12:       {ratio_mirror_I12:.6f}")

    # Key diagnostic: m_needed consistency
    m_k = (2.137 - result_k["direct_c"]) / result_k["mirror_I12"]
    m_s = (1.938 - result_s["direct_c"]) / result_s["mirror_I12"]
    print()
    print("m_needed for Model B (I12 mirror only):")
    print(f"  κ:  {m_k:.6f}  (exp(R)+5 = {math.exp(1.3036)+5:.6f})")
    print(f"  κ*: {m_s:.6f}  (exp(R)+5 = {math.exp(1.1167)+5:.6f})")
    print(f"  Difference: {abs(m_k - m_s):.6f} ({abs(m_k - m_s) / m_k * 100:.2f}%)")


if __name__ == "__main__":
    main()
