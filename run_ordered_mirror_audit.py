"""
run_ordered_mirror_audit.py
Test: Does ordered-pair swapping fix the 12-pair mirror catastrophe?

GPT Analysis (2025-12-19)
-------------------------
The 12 pair dominates failure under naive mirroring because:
- TeX mirror maps (1,2) → (2,1) (swaps left/right roles)
- Current code only builds upper-triangle with symmetry factor ×2
- This cannot represent the actual TeX swap

Test: Evaluate ordered pairs (12 AND 21) separately for mirror terms.
If the 12 catastrophe disappears, the ordered-pair swap is the fix.
"""

from __future__ import annotations

import math
from typing import Dict, List

from src.evaluate import evaluate_term
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.terms_k3_d1 import make_all_terms_k3_ordered_v2

# Constants
THETA = 4.0 / 7.0
N_QUAD = 60
N_QUAD_A = 40

# Benchmarks
KAPPA_R = 1.3036
KAPPA_TARGET = 2.137

KAPPA_STAR_R = 1.1167
KAPPA_STAR_TARGET = 1.938

# Factorial normalization (same for ij and ji)
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


def evaluate_pair_i12(
    terms: List,
    polynomials: Dict,
    R: float,
) -> float:
    """Evaluate I1+I2 for a list of terms."""
    total = 0.0
    for term in terms[:2]:  # I1, I2
        val = evaluate_term(
            term, polynomials, N_QUAD, R=R, theta=THETA, n_quad_a=N_QUAD_A
        ).value
        total += val
    return total


def run_benchmark(name: str, R: float, c_target: float, polynomials: Dict):
    """Run ordered-pair mirror audit for one benchmark."""
    print()
    print("=" * 78)
    print(f"ORDERED-PAIR MIRROR AUDIT: {name} (R={R})")
    print("=" * 78)

    # Build all ordered terms at +R and -R
    all_terms_plus = make_all_terms_k3_ordered_v2(THETA, R, kernel_regime="paper")
    all_terms_minus = make_all_terms_k3_ordered_v2(THETA, -R, kernel_regime="paper")

    mirror_mult = math.exp(R) + 5

    print()
    print("Part 1: Diagonal pairs (unchanged - no swap needed)")
    print("-" * 78)
    print(f"{'Pair':>6}  {'direct I12':>14}  {'mirror(-R) I12':>14}  {'recombined':>14}")

    direct_diag = 0.0
    mirror_diag = 0.0

    for pair in ["11", "22", "33"]:
        terms_plus = all_terms_plus[pair]
        terms_minus = all_terms_minus[pair]

        direct = evaluate_pair_i12(terms_plus, polynomials, R)
        mirror = evaluate_pair_i12(terms_minus, polynomials, -R)

        norm = FACTORIAL_NORM[pair]
        direct_norm = norm * direct
        mirror_norm = norm * mirror
        recomb = direct_norm + mirror_mult * mirror_norm

        direct_diag += direct_norm
        mirror_diag += mirror_norm

        print(f"  {pair}:  {direct_norm:+14.8f}  {mirror_norm:+14.8f}  {recomb:+14.8f}")

    print()
    print("Part 2: Off-diagonal pairs WITH SWAP")
    print("-" * 78)
    print("For (ij), direct uses ij terms, mirror uses SWAPPED ji terms at -R")
    print()
    print(f"{'Pair':>6}  {'direct(ij) I12':>16}  {'mirror(ji@-R)':>16}  {'recombined':>14}")

    direct_offdiag = 0.0
    mirror_offdiag = 0.0

    for base, swap in [("12", "21"), ("13", "31"), ("23", "32")]:
        # Direct: use base pair at +R
        terms_direct = all_terms_plus[base]
        direct = evaluate_pair_i12(terms_direct, polynomials, R)

        # Mirror: use SWAPPED pair at -R (this is the key!)
        terms_mirror = all_terms_minus[swap]
        mirror = evaluate_pair_i12(terms_mirror, polynomials, -R)

        # Note: we use ×2 total because we're combining ij + ji contributions
        # but each is evaluated once (direct ij, mirror ji)
        norm = 2.0 * FACTORIAL_NORM[base]
        direct_norm = norm * direct
        mirror_norm = norm * mirror
        recomb = direct_norm + mirror_mult * mirror_norm

        direct_offdiag += direct_norm
        mirror_offdiag += mirror_norm

        print(f"  {base}/{swap}:  {direct_norm:+14.8f}  {mirror_norm:+14.8f}  {recomb:+14.8f}")

    print()
    print("Part 3: Comparison - swap vs no-swap for 12")
    print("-" * 78)

    # Without swap: both direct and mirror use "12" terms
    terms_12_plus = all_terms_plus["12"]
    terms_12_minus = all_terms_minus["12"]

    direct_12_no_swap = 2.0 * FACTORIAL_NORM["12"] * evaluate_pair_i12(terms_12_plus, polynomials, R)
    mirror_12_no_swap = 2.0 * FACTORIAL_NORM["12"] * evaluate_pair_i12(terms_12_minus, polynomials, -R)

    # With swap: direct uses "12", mirror uses "21"
    terms_21_minus = all_terms_minus["21"]
    direct_12_with_swap = 2.0 * FACTORIAL_NORM["12"] * evaluate_pair_i12(terms_12_plus, polynomials, R)
    mirror_21_with_swap = 2.0 * FACTORIAL_NORM["21"] * evaluate_pair_i12(terms_21_minus, polynomials, -R)

    print(f"  12 NO SWAP:   direct={direct_12_no_swap:+.6f}  mirror(12@-R)={mirror_12_no_swap:+.6f}")
    print(f"  12 WITH SWAP: direct={direct_12_with_swap:+.6f}  mirror(21@-R)={mirror_21_with_swap:+.6f}")
    print()

    recomb_no_swap = direct_12_no_swap + mirror_mult * mirror_12_no_swap
    recomb_with_swap = direct_12_with_swap + mirror_mult * mirror_21_with_swap

    print(f"  Recombined (no swap):   {recomb_no_swap:+.6f}")
    print(f"  Recombined (with swap): {recomb_with_swap:+.6f}")

    # Part 4: Full totals
    print()
    print("Part 4: Full I12 totals")
    print("-" * 78)

    total_direct = direct_diag + direct_offdiag
    total_mirror_swap = mirror_diag + mirror_offdiag

    # For comparison: mirror without swap
    mirror_no_swap_offdiag = 0.0
    for base in ["12", "13", "23"]:
        terms_minus = all_terms_minus[base]  # Same pair, not swapped
        mirror = evaluate_pair_i12(terms_minus, polynomials, -R)
        norm = 2.0 * FACTORIAL_NORM[base]
        mirror_no_swap_offdiag += norm * mirror

    total_mirror_no_swap = mirror_diag + mirror_no_swap_offdiag

    print(f"  Direct I12 total:        {total_direct:+14.8f}")
    print(f"  Mirror I12 (WITH swap):  {total_mirror_swap:+14.8f}")
    print(f"  Mirror I12 (NO swap):    {total_mirror_no_swap:+14.8f}")

    # Recombination
    c_with_swap = total_direct + mirror_mult * total_mirror_swap
    c_no_swap = total_direct + mirror_mult * total_mirror_no_swap

    # Note: we're only computing I12 here, need to add I34 for full c
    # Get I34 from direct evaluation
    i34_total = 0.0
    for pair in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms_plus[pair]
        sym = 1.0 if pair in ["11", "22", "33"] else 2.0
        norm = sym * FACTORIAL_NORM[pair]
        for term in terms[2:4]:  # I3, I4
            val = evaluate_term(
                term, polynomials, N_QUAD, R=R, theta=THETA, n_quad_a=N_QUAD_A
            ).value
            i34_total += norm * val

    print(f"  I34 total (direct):      {i34_total:+14.8f}")
    print()

    full_c_with_swap = c_with_swap + i34_total
    full_c_no_swap = c_no_swap + i34_total

    gap_with_swap = (full_c_with_swap - c_target) / c_target * 100
    gap_no_swap = (full_c_no_swap - c_target) / c_target * 100

    print(f"Full c target:           {c_target:+14.8f}")
    print(f"Full c (WITH swap):      {full_c_with_swap:+14.8f}  (gap: {gap_with_swap:+.2f}%)")
    print(f"Full c (NO swap):        {full_c_no_swap:+14.8f}  (gap: {gap_no_swap:+.2f}%)")

    return {
        "c_with_swap": full_c_with_swap,
        "c_no_swap": full_c_no_swap,
        "gap_with_swap": gap_with_swap,
        "gap_no_swap": gap_no_swap,
    }


def main():
    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    result_k = run_benchmark("κ", KAPPA_R, KAPPA_TARGET, polys_k)
    result_s = run_benchmark("κ*", KAPPA_STAR_R, KAPPA_STAR_TARGET, polys_s)

    # Summary
    print()
    print("=" * 78)
    print("SUMMARY: Does ordered-pair swapping matter?")
    print("=" * 78)
    print()

    print(f"κ benchmark:")
    print(f"  WITH swap: {result_k['gap_with_swap']:+.2f}%")
    print(f"  NO swap:   {result_k['gap_no_swap']:+.2f}%")
    print()
    print(f"κ* benchmark:")
    print(f"  WITH swap: {result_s['gap_with_swap']:+.2f}%")
    print(f"  NO swap:   {result_s['gap_no_swap']:+.2f}%")
    print()

    # Ratio analysis
    if result_s["c_with_swap"] != 0 and result_s["c_no_swap"] != 0:
        ratio_with = result_k["c_with_swap"] / result_s["c_with_swap"]
        ratio_no = result_k["c_no_swap"] / result_s["c_no_swap"]
        target_ratio = KAPPA_TARGET / KAPPA_STAR_TARGET

        print(f"Ratio (target: {target_ratio:.6f}):")
        print(f"  WITH swap: {ratio_with:.6f}  (gap: {(ratio_with - target_ratio) / target_ratio * 100:+.2f}%)")
        print(f"  NO swap:   {ratio_no:.6f}  (gap: {(ratio_no - target_ratio) / target_ratio * 100:+.2f}%)")

    # Conclusion
    print()
    if abs(result_k["gap_with_swap"]) < abs(result_k["gap_no_swap"]) and \
       abs(result_s["gap_with_swap"]) < abs(result_s["gap_no_swap"]):
        print("CONCLUSION: Ordered-pair swapping IMPROVES accuracy")
        print("  → The 12 catastrophe is partially due to missing swap")
    else:
        print("CONCLUSION: Ordered-pair swapping does NOT significantly help")
        print("  → The issue is elsewhere (operator ordering?)")


if __name__ == "__main__":
    main()
