"""
run_raw_mirror_test.py
Test: Does raw regime + mirror formula work?

GPT Recommendation #5 (2025-12-19)
----------------------------------
Run `kernel_regime="raw"` with the same mirror recombination.

If raw+mirror also gets near the ratio target:
  → Case C might be unnecessary
If raw+mirror gives a bad ratio but paper+mirror gives a good one:
  → Case C is indeed part of truth

This is critical because earlier "paper fixes ratio" conclusions were drawn
*before mirror existed*. Mirror may completely reorder that story.
"""

from __future__ import annotations

import math
from typing import Dict

from src.evaluate import evaluate_term
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.terms_k3_d1 import make_all_terms_k3

# Constants
THETA = 4.0 / 7.0
N_QUAD = 60
N_QUAD_A = 40

# Benchmarks
KAPPA_R = 1.3036
KAPPA_TARGET = 2.137

KAPPA_STAR_R = 1.1167
KAPPA_STAR_TARGET = 1.938

TARGET_RATIO = KAPPA_TARGET / KAPPA_STAR_TARGET  # ~1.103

# Normalization
FACTORIAL_NORM: Dict[str, float] = {
    "11": 1.0,
    "22": 1.0 / 4.0,
    "33": 1.0 / 36.0,
    "12": 1.0 / 2.0,
    "13": 1.0 / 6.0,
    "23": 1.0 / 12.0,
}

SYMMETRY_FACTOR: Dict[str, float] = {
    "11": 1.0, "22": 1.0, "33": 1.0,
    "12": 2.0, "13": 2.0, "23": 2.0
}


def compute_c_with_mirror(
    theta: float,
    R: float,
    polynomials: Dict,
    kernel_regime: str,
    K: int = 3,
) -> Dict[str, float]:
    """Compute c using mirror formula for specified kernel regime."""

    # Build terms
    all_terms_plus = make_all_terms_k3(theta, R, kernel_regime=kernel_regime)
    all_terms_minus = make_all_terms_k3(theta, -R, kernel_regime=kernel_regime)

    # Mirror multiplier: m = exp(R) + (2K - 1)
    mirror_mult = math.exp(R) + (2 * K - 1)

    i12_plus = 0.0
    i12_minus = 0.0
    i34_plus = 0.0

    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms_plus = all_terms_plus[pair_key]
        terms_minus = all_terms_minus[pair_key]

        norm = FACTORIAL_NORM[pair_key] * SYMMETRY_FACTOR[pair_key]

        # I₁ and I₂ (indices 0, 1) - need mirror
        for i in [0, 1]:  # I₁, I₂
            val_plus = evaluate_term(
                terms_plus[i], polynomials, N_QUAD, R=R, theta=theta, n_quad_a=N_QUAD_A
            ).value
            val_minus = evaluate_term(
                terms_minus[i], polynomials, N_QUAD, R=-R, theta=theta, n_quad_a=N_QUAD_A
            ).value

            i12_plus += norm * val_plus
            i12_minus += norm * val_minus

        # I₃ and I₄ (indices 2, 3) - NO mirror
        for i in [2, 3]:  # I₃, I₄
            val_plus = evaluate_term(
                terms_plus[i], polynomials, N_QUAD, R=R, theta=theta, n_quad_a=N_QUAD_A
            ).value
            i34_plus += norm * val_plus

    # Mirror assembly
    c = i12_plus + mirror_mult * i12_minus + i34_plus

    return {
        "c": c,
        "i12_plus": i12_plus,
        "i12_minus": i12_minus,
        "i34_plus": i34_plus,
        "mirror_mult": mirror_mult,
    }


def run_test(name: str, R: float, c_target: float, polynomials: Dict):
    """Run test for one benchmark."""
    print()
    print(f"=== {name} (R={R}) ===")

    # Raw regime + mirror
    raw_result = compute_c_with_mirror(THETA, R, polynomials, "raw")
    raw_c = raw_result["c"]
    raw_gap = (raw_c - c_target) / c_target * 100

    # Paper regime + mirror
    paper_result = compute_c_with_mirror(THETA, R, polynomials, "paper")
    paper_c = paper_result["c"]
    paper_gap = (paper_c - c_target) / c_target * 100

    print(f"  Target c:     {c_target:.6f}")
    print()
    print(f"  RAW + mirror:")
    print(f"    I12+: {raw_result['i12_plus']:+.6f}")
    print(f"    I12-: {raw_result['i12_minus']:+.6f}")
    print(f"    I34+: {raw_result['i34_plus']:+.6f}")
    print(f"    m:    {raw_result['mirror_mult']:.6f}")
    print(f"    c:    {raw_c:.6f}  (gap: {raw_gap:+.2f}%)")
    print()
    print(f"  PAPER + mirror:")
    print(f"    I12+: {paper_result['i12_plus']:+.6f}")
    print(f"    I12-: {paper_result['i12_minus']:+.6f}")
    print(f"    I34+: {paper_result['i34_plus']:+.6f}")
    print(f"    m:    {paper_result['mirror_mult']:.6f}")
    print(f"    c:    {paper_c:.6f}  (gap: {paper_gap:+.2f}%)")

    return {
        "raw_c": raw_c,
        "paper_c": paper_c,
        "raw_gap": raw_gap,
        "paper_gap": paper_gap,
    }


def main():
    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    print("=" * 78)
    print("RAW vs PAPER + MIRROR TEST")
    print("=" * 78)
    print()
    print("Testing GPT recommendation #5: Does raw+mirror work?")
    print("If raw+mirror gets good ratio → Case C unnecessary")
    print("If paper+mirror gets good ratio but raw doesn't → Case C is truth")

    result_k = run_test("κ", KAPPA_R, KAPPA_TARGET, polys_k)
    result_s = run_test("κ*", KAPPA_STAR_R, KAPPA_STAR_TARGET, polys_s)

    # Ratio analysis
    print()
    print("=" * 78)
    print("RATIO ANALYSIS")
    print("=" * 78)
    print()
    print(f"  Target ratio:     {TARGET_RATIO:.6f}")
    print()

    raw_ratio = result_k["raw_c"] / result_s["raw_c"]
    paper_ratio = result_k["paper_c"] / result_s["paper_c"]

    raw_ratio_gap = (raw_ratio - TARGET_RATIO) / TARGET_RATIO * 100
    paper_ratio_gap = (paper_ratio - TARGET_RATIO) / TARGET_RATIO * 100

    print(f"  RAW + mirror:")
    print(f"    κ/κ* ratio: {raw_ratio:.6f}  (gap: {raw_ratio_gap:+.2f}%)")
    print()
    print(f"  PAPER + mirror:")
    print(f"    κ/κ* ratio: {paper_ratio:.6f}  (gap: {paper_ratio_gap:+.2f}%)")

    print()
    print("=" * 78)
    print("CONCLUSION")
    print("=" * 78)

    if abs(paper_ratio_gap) < abs(raw_ratio_gap):
        print("  → PAPER + mirror gives better ratio")
        print("  → Case C (paper regime) IS part of truth")
    else:
        print("  → RAW + mirror gives better ratio")
        print("  → Case C might be unnecessary?")

    if abs(paper_ratio_gap) < 5:
        print(f"  → Paper regime ratio within 5%: GOOD")
    else:
        print(f"  → Paper regime ratio outside 5%: needs investigation")


if __name__ == "__main__":
    main()
