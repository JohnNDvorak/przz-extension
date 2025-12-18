"""
analyze_psi_signs.py

Analyze sign patterns in Ψ combinatorial formulas to investigate ratio reversal.

Context:
- We need const ratio (κ/κ*) ≈ 0.94
- Our naive formula gives ratio 1.71 (WRONG DIRECTION!)

Questions to answer:
1. What fraction of monomials have negative coefficients for each pair?
2. Do NEGATIVE monomials contribute MORE for κ (higher degree polynomials)?
3. Could sign pattern interactions with polynomial degree reverse the ratio?
"""

from __future__ import annotations
from typing import Dict, Tuple
from src.psi_combinatorial import psi_d1_configs


def analyze_sign_patterns(ell: int, ellbar: int) -> Dict[str, any]:
    """
    Analyze sign patterns for a given (ℓ, ℓ̄) pair.

    Returns dict with:
    - total_monomials: int
    - positive_count: int
    - negative_count: int
    - positive_sum: int (sum of positive coefficients)
    - negative_sum: int (sum of absolute values of negative coefficients)
    - positive_fraction: float
    - monomials: list of (k1,k2,l1,m1,coeff) tuples sorted by coefficient
    """
    configs = psi_d1_configs(ell, ellbar)

    positive = []
    negative = []

    for (k1, k2, l1, m1), coeff in configs.items():
        if coeff > 0:
            positive.append(((k1, k2, l1, m1), coeff))
        elif coeff < 0:
            negative.append(((k1, k2, l1, m1), coeff))

    total = len(configs)
    pos_count = len(positive)
    neg_count = len(negative)

    pos_sum = sum(c for _, c in positive)
    neg_sum = sum(abs(c) for _, c in negative)

    return {
        'total_monomials': total,
        'positive_count': pos_count,
        'negative_count': neg_count,
        'positive_sum': pos_sum,
        'negative_sum': neg_sum,
        'positive_fraction': pos_count / total if total > 0 else 0,
        'negative_fraction': neg_count / total if total > 0 else 0,
        'coefficient_balance': pos_sum - neg_sum,
        'positive_monomials': sorted(positive, key=lambda x: -x[1]),
        'negative_monomials': sorted(negative, key=lambda x: x[1]),
    }


def print_sign_analysis():
    """Print sign pattern analysis for all K=3 pairs."""

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    print("=" * 80)
    print("Ψ SIGN PATTERN ANALYSIS")
    print("=" * 80)
    print()
    print("Question: Could sign patterns explain the ratio reversal?")
    print("  Target ratio (κ/κ*) ≈ 0.94")
    print("  Naive formula gives 1.71 (WRONG DIRECTION!)")
    print()

    for (ell, ellbar) in pairs:
        analysis = analyze_sign_patterns(ell, ellbar)

        print(f"({ell},{ellbar}) PAIR:")
        print(f"  Total monomials: {analysis['total_monomials']}")
        print(f"  Positive: {analysis['positive_count']} ({analysis['positive_fraction']:.1%})")
        print(f"  Negative: {analysis['negative_count']} ({analysis['negative_fraction']:.1%})")
        print(f"  Sum of positive coeffs: {analysis['positive_sum']:+d}")
        print(f"  Sum of |negative| coeffs: {analysis['negative_sum']:+d}")
        print(f"  Net coefficient balance: {analysis['coefficient_balance']:+d}")
        print()

        # Show top 3 positive and top 3 negative
        print("  Top positive monomials:")
        for (k1, k2, l1, m1), coeff in analysis['positive_monomials'][:3]:
            parts = []
            if k1 > 0: parts.append(f"C^{k1}" if k1 > 1 else "C")
            if k2 > 0: parts.append(f"D^{k2}" if k2 > 1 else "D")
            if l1 > 0: parts.append(f"A^{l1}" if l1 > 1 else "A")
            if m1 > 0: parts.append(f"B^{m1}" if m1 > 1 else "B")
            mono_str = "×".join(parts) if parts else "1"
            print(f"    {coeff:+4d} × {mono_str}")

        if analysis['negative_monomials']:
            print("  Top negative monomials:")
            for (k1, k2, l1, m1), coeff in analysis['negative_monomials'][:3]:
                parts = []
                if k1 > 0: parts.append(f"C^{k1}" if k1 > 1 else "C")
                if k2 > 0: parts.append(f"D^{k2}" if k2 > 1 else "D")
                if l1 > 0: parts.append(f"A^{l1}" if l1 > 1 else "A")
                if m1 > 0: parts.append(f"B^{m1}" if m1 > 1 else "B")
                mono_str = "×".join(parts) if parts else "1"
                print(f"    {coeff:+4d} × {mono_str}")

        print()

    print("=" * 80)
    print()


def analyze_cross_integrals():
    """
    Analyze how cross-integrals (1,3) and (2,3) might contribute.

    Key finding from handoff: P₃ changes sign on [0,1], so cross-integrals
    can be NEGATIVE.
    """
    print("=" * 80)
    print("CROSS-INTEGRAL ANALYSIS")
    print("=" * 80)
    print()
    print("Key insight: P₃ changes sign on [0,1]")
    print("This means ∫P_i·P_j can be NEGATIVE for cross-terms (i,j) = (1,3), (2,3)")
    print()
    print("For κ (higher degree): More terms involve P₃")
    print("For κ* (lower degree): Fewer terms involve P₃")
    print()
    print("Could this flip the relative contribution of negative terms?")
    print()

    # Analyze (1,3) and (2,3) sign patterns
    for (ell, ellbar) in [(1, 3), (2, 3)]:
        analysis = analyze_sign_patterns(ell, ellbar)
        print(f"({ell},{ellbar}) CROSS-INTEGRAL:")
        print(f"  Negative fraction: {analysis['negative_fraction']:.1%}")
        print(f"  Coefficient balance: {analysis['coefficient_balance']:+d}")
        print()


def compare_22_detail():
    """Detailed analysis of (2,2) pair for the 12 monomials."""
    print("=" * 80)
    print("DETAILED (2,2) ANALYSIS")
    print("=" * 80)
    print()

    analysis = analyze_sign_patterns(2, 2)
    configs = psi_d1_configs(2, 2)

    print("All 12 monomials with coefficients:")
    print()

    # Group by sign
    for (k1, k2, l1, m1), coeff in sorted(configs.items()):
        parts = []
        if k1 > 0: parts.append(f"C^{k1}" if k1 > 1 else "C")
        if k2 > 0: parts.append(f"D^{k2}" if k2 > 1 else "D")
        if l1 > 0: parts.append(f"A^{l1}" if l1 > 1 else "A")
        if m1 > 0: parts.append(f"B^{m1}" if m1 > 1 else "B")
        mono_str = " × ".join(parts) if parts else "1"

        sign_marker = "POS" if coeff > 0 else "NEG"
        print(f"  {coeff:+4d} × {mono_str:<20}  [{sign_marker}]  (C^{k1} D^{k2} A^{l1} B^{m1})")

    print()
    print(f"Sum of positive: {analysis['positive_sum']}")
    print(f"Sum of |negative|: {analysis['negative_sum']}")
    print(f"Net balance: {analysis['coefficient_balance']}")
    print()


if __name__ == "__main__":
    print_sign_analysis()
    print()
    analyze_cross_integrals()
    print()
    compare_22_detail()
