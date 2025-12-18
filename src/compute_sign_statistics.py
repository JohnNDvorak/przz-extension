"""
compute_sign_statistics.py

Compute detailed sign pattern statistics for Ψ monomials.
This investigates whether negative monomials could explain the ratio reversal.
"""

from __future__ import annotations
from src.psi_combinatorial import psi_d1_configs


def compute_detailed_statistics():
    """Compute and print detailed sign statistics for all K=3 pairs."""

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    print("=" * 80)
    print("Ψ SIGN PATTERN STATISTICS")
    print("=" * 80)
    print()
    print("Context: Investigating ratio reversal problem")
    print("  Required: const_κ / const_κ* ≈ 0.94 (κ < κ*)")
    print("  Our naive: ratio ≈ 1.71 (κ > κ*) — WRONG DIRECTION!")
    print()
    print("Hypothesis: Negative Ψ monomials reduce κ more than κ*")
    print("=" * 80)
    print()

    all_stats = {}

    for (ell, ellbar) in pairs:
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

        stats = {
            'total': total,
            'positive_count': pos_count,
            'negative_count': neg_count,
            'positive_sum': pos_sum,
            'negative_sum': neg_sum,
            'net_balance': pos_sum - neg_sum,
            'positive_monomials': sorted(positive, key=lambda x: -x[1]),
            'negative_monomials': sorted(negative, key=lambda x: x[1]),
        }

        all_stats[(ell, ellbar)] = stats

        print(f"PAIR ({ell},{ellbar}):")
        print(f"  Total monomials:      {total}")
        print(f"  Positive count:       {pos_count:2d}  ({pos_count/total*100:5.1f}%)")
        print(f"  Negative count:       {neg_count:2d}  ({neg_count/total*100:5.1f}%)")
        print(f"  Sum(positive coeffs): {pos_sum:+6d}")
        print(f"  Sum(|negative| coeffs): {neg_sum:+6d}")
        print(f"  Net balance:          {pos_sum - neg_sum:+6d}")
        print()

        if neg_sum > 0:
            print(f"  Ratio |negative|/|positive|: {neg_sum/pos_sum:.3f}")
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print("Pair | Total | Pos | Neg | Pos% | Sum(+) | Sum(|−|) | Balance | |−|/|+|")
    print("-----|-------|-----|-----|------|--------|----------|---------|--------")

    for (ell, ellbar) in pairs:
        s = all_stats[(ell, ellbar)]
        pos_pct = s['positive_count'] / s['total'] * 100 if s['total'] > 0 else 0
        ratio = s['negative_sum'] / s['positive_sum'] if s['positive_sum'] > 0 else 0
        print(f"({ell},{ellbar}) | {s['total']:5d} | {s['positive_count']:3d} | {s['negative_count']:3d} | "
              f"{pos_pct:4.0f}% | {s['positive_sum']:6d} | {s['negative_sum']:8d} | "
              f"{s['net_balance']:7d} | {ratio:6.3f}")

    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Check if negative fraction increases with pair degree
    print("1. Does negative fraction increase with pair degree?")
    print()
    for (ell, ellbar) in pairs:
        s = all_stats[(ell, ellbar)]
        neg_frac = s['negative_count'] / s['total'] if s['total'] > 0 else 0
        print(f"   ({ell},{ellbar}): {neg_frac*100:5.1f}% negative")
    print()

    # Check if |negative| > |positive| for higher pairs
    print("2. Does |negative| dominate |positive| for higher pairs?")
    print()
    for (ell, ellbar) in pairs:
        s = all_stats[(ell, ellbar)]
        if s['positive_sum'] > 0:
            ratio = s['negative_sum'] / s['positive_sum']
            dominance = "DOMINATES" if ratio > 1.0 else "balanced" if ratio > 0.8 else "minor"
            print(f"   ({ell},{ellbar}): |−|/|+| = {ratio:.3f}  [{dominance}]")
    print()

    # Check cross-pairs
    print("3. Are cross-pairs (1,3), (2,3) particularly negative?")
    print()
    for (ell, ellbar) in [(1, 3), (2, 3)]:
        s = all_stats[(ell, ellbar)]
        if s['positive_sum'] > 0:
            ratio = s['negative_sum'] / s['positive_sum']
            print(f"   ({ell},{ellbar}): |−|/|+| = {ratio:.3f}")
            print(f"            Net balance = {s['net_balance']:+d}")
    print()

    return all_stats


def analyze_monomial_structure(ell: int, ellbar: int):
    """
    Analyze which types of monomials are positive vs negative.
    Focus on A, B exponents (derivative order).
    """
    configs = psi_d1_configs(ell, ellbar)

    print("=" * 80)
    print(f"MONOMIAL STRUCTURE ANALYSIS: ({ell},{ellbar})")
    print("=" * 80)
    print()

    # Group by derivative structure
    high_deriv_pos = []  # a+b >= 2, coeff > 0
    high_deriv_neg = []  # a+b >= 2, coeff < 0
    low_deriv_pos = []   # a+b < 2, coeff > 0
    low_deriv_neg = []   # a+b < 2, coeff < 0

    for (k1, k2, l1, m1), coeff in configs.items():
        deriv_order = l1 + m1  # a + b (A and B exponents)

        if deriv_order >= 2:
            if coeff > 0:
                high_deriv_pos.append(((k1, k2, l1, m1), coeff))
            else:
                high_deriv_neg.append(((k1, k2, l1, m1), coeff))
        else:
            if coeff > 0:
                low_deriv_pos.append(((k1, k2, l1, m1), coeff))
            else:
                low_deriv_neg.append(((k1, k2, l1, m1), coeff))

    print("Monomials grouped by derivative order (a+b):")
    print()
    print(f"  High derivative (a+b ≥ 2):")
    print(f"    Positive: {len(high_deriv_pos)} monomials, sum = {sum(c for _,c in high_deriv_pos):+d}")
    print(f"    Negative: {len(high_deriv_neg)} monomials, sum = {sum(c for _,c in high_deriv_neg):+d}")
    print()
    print(f"  Low derivative (a+b < 2):")
    print(f"    Positive: {len(low_deriv_pos)} monomials, sum = {sum(c for _,c in low_deriv_pos):+d}")
    print(f"    Negative: {len(low_deriv_neg)} monomials, sum = {sum(c for _,c in low_deriv_neg):+d}")
    print()

    # Show detailed breakdown
    if high_deriv_neg:
        print("High-derivative negative monomials (these reduce const):")
        for (k1, k2, l1, m1), coeff in sorted(high_deriv_neg, key=lambda x: x[1]):
            parts = []
            if k1 > 0: parts.append(f"C^{k1}" if k1 > 1 else "C")
            if k2 > 0: parts.append(f"D^{k2}" if k2 > 1 else "D")
            if l1 > 0: parts.append(f"A^{l1}" if l1 > 1 else "A")
            if m1 > 0: parts.append(f"B^{m1}" if m1 > 1 else "B")
            mono_str = "×".join(parts) if parts else "1"
            print(f"  {coeff:+4d} × {mono_str:<20}  (deriv order = {l1}+{m1} = {l1+m1})")
    print()


def detailed_22_breakdown():
    """Complete breakdown of (2,2) monomials."""
    print("=" * 80)
    print("DETAILED (2,2) BREAKDOWN — ALL 12 MONOMIALS")
    print("=" * 80)
    print()

    configs = psi_d1_configs(2, 2)

    print("Monomial | Coeff | Sign | C | D | A | B | Deriv")
    print("---------|-------|------|---|---|---|---|------")

    for (k1, k2, l1, m1), coeff in sorted(configs.items(), key=lambda x: -x[1]):
        parts = []
        if k1 > 0: parts.append(f"C^{k1}" if k1 > 1 else "C")
        if k2 > 0: parts.append(f"D^{k2}" if k2 > 1 else "D")
        if l1 > 0: parts.append(f"A^{l1}" if l1 > 1 else "A")
        if m1 > 0: parts.append(f"B^{m1}" if m1 > 1 else "B")
        mono_str = "×".join(parts) if parts else "1"

        sign = "+" if coeff > 0 else "−"
        deriv_order = l1 + m1

        print(f"{mono_str:<15} | {coeff:+4d} | {sign:4} | {k1} | {k2} | {l1} | {m1} | {deriv_order}")

    print()

    # Compute statistics
    pos_sum = sum(c for c in configs.values() if c > 0)
    neg_sum = sum(abs(c) for c in configs.values() if c < 0)

    print(f"Sum of positive coefficients: {pos_sum:+d}")
    print(f"Sum of |negative| coefficients: {neg_sum:+d}")
    print(f"Net balance: {pos_sum - neg_sum:+d}")
    print(f"Ratio |negative|/|positive|: {neg_sum/pos_sum:.3f}")
    print()


if __name__ == "__main__":
    # Main sign statistics
    all_stats = compute_detailed_statistics()

    print()

    # Structure analysis for (2,2)
    analyze_monomial_structure(2, 2)

    print()

    # Detailed (2,2) breakdown
    detailed_22_breakdown()

    print()

    # Cross-pair analysis
    print("=" * 80)
    print("CROSS-PAIR STRUCTURE")
    print("=" * 80)
    print()

    for (ell, ellbar) in [(1, 3), (2, 3)]:
        analyze_monomial_structure(ell, ellbar)
        print()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("If hypothesis is correct, we should see:")
    print("  1. Higher pairs have |negative| > |positive|")
    print("  2. Negative monomials have high A, B exponents (high derivatives)")
    print("  3. These scale with polynomial degree, reducing κ more than κ*")
    print()
    print("Check the ratios above to see if this pattern holds.")
    print()
