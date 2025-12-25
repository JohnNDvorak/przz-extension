"""
Analyze Case A/B/C distribution for κ vs κ* polynomial configurations.

This script investigates whether higher polynomial degrees lead to more Case C
monomials, which could explain the negative correlation between polynomial
magnitude and contribution.

Key Questions:
1. Do κ polynomials (higher degree) have more Case C monomials than κ*?
2. Do Case C kernels attenuate contributions more strongly?
3. Is there polynomial-degree-dependent normalization in F_d structure?
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List

from src.psi_fd_mapping import (
    FdCase, map_pair_monomials, group_by_case_pair,
    group_by_triple
)


def analyze_case_distribution(config_name: str) -> Dict:
    """
    Analyze Case A/B/C distribution for all K=3 pairs.

    Returns:
        Dict with per-pair and total statistics
    """
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    total_by_case = defaultdict(int)
    total_by_case_pair = defaultdict(int)
    per_pair_data = {}

    for ell, ellbar in pairs:
        mappings = map_pair_monomials(ell, ellbar)
        by_case = group_by_case_pair(mappings)
        by_triple = group_by_triple(mappings)

        # Count by case
        case_counts = defaultdict(int)
        for (case_l, case_r), group in by_case.items():
            case_counts[(case_l, case_r)] = len(group)
            total_by_case_pair[(case_l, case_r)] += len(group)

        # Count individual Case occurrences (left and right)
        for mapping in mappings:
            total_by_case[mapping.triple.case_left] += 1
            total_by_case[mapping.triple.case_right] += 1

        # Store per-pair data
        per_pair_data[(ell, ellbar)] = {
            'n_monomials': len(mappings),
            'n_triples': len(by_triple),
            'case_counts': dict(case_counts),
            'omega_distribution': _analyze_omega_dist(mappings)
        }

    return {
        'config': config_name,
        'per_pair': per_pair_data,
        'total_by_case': dict(total_by_case),
        'total_by_case_pair': dict(total_by_case_pair)
    }


def _analyze_omega_dist(mappings: List) -> Dict:
    """Analyze distribution of ω values (left and right)."""
    omega_left_counts = defaultdict(int)
    omega_right_counts = defaultdict(int)

    for m in mappings:
        omega_left_counts[m.triple.omega_left] += 1
        omega_right_counts[m.triple.omega_right] += 1

    return {
        'omega_left': dict(omega_left_counts),
        'omega_right': dict(omega_right_counts)
    }


def print_comparison_table():
    """Print detailed comparison of κ vs κ* Case distributions."""

    print("=" * 100)
    print("CASE A/B/C DISTRIBUTION: κ vs κ* POLYNOMIAL CONFIGURATIONS")
    print("=" * 100)
    print()

    # Analyze both configurations
    # Note: The mapping depends only on (a,b,c,d) structure, NOT on polynomial coefficients
    # So both κ and κ* have the SAME Case distribution for K=3, d=1
    kappa_dist = analyze_case_distribution("κ (R=1.3036)")
    kappa_star_dist = analyze_case_distribution("κ* (R=1.1167)")

    print("KEY FINDING: Case distribution depends on Ψ structure (K, d), NOT polynomial degrees!")
    print("Both κ and κ* use K=3, d=1, so they have IDENTICAL Case A/B/C distributions.")
    print()

    # Print per-pair breakdown
    print("=" * 100)
    print("PER-PAIR CASE DISTRIBUTION (identical for both benchmarks)")
    print("=" * 100)
    print()

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    print(f"{'Pair':^8} | {'Monomials':^10} | {'Triples':^8} | Case Pair Counts")
    print("-" * 100)

    for pair in pairs:
        data = kappa_dist['per_pair'][pair]
        case_str = " | ".join(
            f"{cl.value},{cr.value}:{cnt}"
            for (cl, cr), cnt in sorted(data['case_counts'].items())
        )
        print(f"{str(pair):^8} | {data['n_monomials']:^10} | {data['n_triples']:^8} | {case_str}")

    print()

    # Print total counts
    print("=" * 100)
    print("TOTAL CASE COUNTS (sum over all pairs)")
    print("=" * 100)
    print()

    total_cases = kappa_dist['total_by_case']
    total_A = total_cases.get(FdCase.A, 0)
    total_B = total_cases.get(FdCase.B, 0)
    total_C = total_cases.get(FdCase.C, 0)
    total = total_A + total_B + total_C

    print(f"Case A (ω=-1, l=0):  {total_A:3d} occurrences ({100*total_A/total:5.1f}%)")
    print(f"Case B (ω=0,  l=1):  {total_B:3d} occurrences ({100*total_B/total:5.1f}%)")
    print(f"Case C (ω>0,  l>1):  {total_C:3d} occurrences ({100*total_C/total:5.1f}%)")
    print(f"{'Total:':20s} {total:3d} occurrences")
    print()

    # Print case pair distribution
    print("=" * 100)
    print("CASE PAIR DISTRIBUTION")
    print("=" * 100)
    print()

    case_pair_counts = kappa_dist['total_by_case_pair']
    total_pairs = sum(case_pair_counts.values())

    for (cl, cr), count in sorted(case_pair_counts.items(),
                                   key=lambda x: (x[0][0].value, x[0][1].value)):
        pct = 100 * count / total_pairs
        print(f"({cl.value},{cr.value}): {count:3d} monomials ({pct:5.1f}%)")

    print()

    # Analyze polynomial structure differences
    print("=" * 100)
    print("POLYNOMIAL STRUCTURE DIFFERENCES (κ vs κ*)")
    print("=" * 100)
    print()

    print("κ polynomials (R=1.3036):")
    print("  P₁: degree 4 (constrained, x + x(1-x)·P̃)")
    print("  P₂: degree 3 (monomial)")
    print("  P₃: degree 3 (monomial)")
    print("  Q:  degree 5 (Chebyshev basis)")
    print()

    print("κ* polynomials (R=1.1167):")
    print("  P₁: degree 4 (constrained, x + x(1-x)·P̃)")
    print("  P₂: degree 2 (monomial) ← LOWER than κ")
    print("  P₃: degree 2 (monomial) ← LOWER than κ")
    print("  Q:  degree 1 (linear)   ← MUCH LOWER than κ")
    print()

    # Analyze impact
    print("=" * 100)
    print("ANALYSIS: IMPACT ON F_d EVALUATION")
    print("=" * 100)
    print()

    print("KEY INSIGHT: Case distribution is IDENTICAL, but F_d values differ due to:")
    print()
    print("1. Case B (ω=0, l=1): F_d = V(d,l) × P(u)")
    print("   - Evaluates polynomial directly")
    print("   - Higher-degree polynomials → larger values (potentially)")
    print()
    print("2. Case C (ω>0, l>1): F_d = W × (logN)^ω × u^ω × ∫ P((1-a)u) × ... da")
    print("   - Integral kernel attenuates contribution")
    print("   - logN^ω factor can explode (known issue in section7_fd_evaluator.py)")
    print("   - Higher-degree polynomials integrate differently")
    print()
    print("3. Polynomial magnitude effect:")
    print("   - κ has P₂(u=0.5) ≈ 0.758, κ* has P₂(u=0.5) ≈ 0.500")
    print("   - But κ contributes LESS overall (c=1.950 vs target=2.137)")
    print("   - This suggests F_d kernel ATTENUATES higher polynomials")
    print()

    # Print detailed omega distribution for key pairs
    print("=" * 100)
    print("ω DISTRIBUTION FOR DIAGONAL PAIRS")
    print("=" * 100)
    print()

    for pair in [(1, 1), (2, 2), (3, 3)]:
        data = kappa_dist['per_pair'][pair]
        omega_dist = data['omega_distribution']
        print(f"Pair {pair}:")
        print(f"  ω_left:  {omega_dist['omega_left']}")
        print(f"  ω_right: {omega_dist['omega_right']}")
        print()

    print("=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    print()
    print("1. Case A/B/C distribution is IDENTICAL for κ and κ* (both use K=3, d=1)")
    print()
    print("2. The negative correlation between ||P|| and contribution arises from:")
    print("   a) F_d kernel structure (Case B vs Case C evaluation)")
    print("   b) Polynomial degree affecting integral values")
    print("   c) Possible missing normalization in our F_d evaluator")
    print()
    print("3. Higher-degree polynomials (κ) have:")
    print("   - Same number of Case C terms as κ*")
    print("   - But different integral magnitudes due to polynomial shape")
    print("   - Case C kernel may attenuate high-degree contributions more")
    print()
    print("4. The ~26% F_d/P ratio at u=0.5 (from SESSION_SUMMARY) suggests:")
    print("   - F_d kernel significantly reduces polynomial contribution")
    print("   - This reduction may be polynomial-degree-dependent")
    print("   - Could explain why higher ||P|| → smaller contribution")
    print()


def analyze_fd_ratio_by_case():
    """Analyze how F_d/P ratio varies by Case."""
    print("=" * 100)
    print("F_d/P RATIO ANALYSIS BY CASE")
    print("=" * 100)
    print()

    print("From SESSION_SUMMARY_2025_12_17.md:")
    print("  - F_d/P ratio at u=0.5 is approximately 26%")
    print("  - This is measured for a specific polynomial configuration")
    print()

    print("Case-by-case expected behavior:")
    print()
    print("Case A (ω=-1, l=0):")
    print("  F_d = U(d,l) × [α·P(u) + P'(u)/logN]")
    print("  Ratio ≈ |α| + |P'/P|/logN")
    print("  With α = -R/logT ≈ -0.013, logN ≈ 57:")
    print("  If P'/P ~ 1, ratio ≈ 0.013 + 0.018 = 0.031 (3%)")
    print()
    print("Case B (ω=0, l=1):")
    print("  F_d = V(d,l) × P(u)")
    print("  With V(1,(1,)) = -1:")
    print("  Ratio = |V| = 1.0 (100%)")
    print()
    print("Case C (ω>0, l>1):")
    print("  F_d = W × (logN)^ω × u^ω × ∫ P((1-a)u) × a^(ω-1) × exp(...) da")
    print("  This is complex and depends heavily on:")
    print("    - ω value (higher ω → stronger suppression from u^ω)")
    print("    - Polynomial shape (integral may be smaller than P(u))")
    print("    - logN^ω factor (known to cause explosions in current code)")
    print()
    print("HYPOTHESIS: The 26% overall ratio suggests Case C dominance with:")
    print("  - Case C integral < P(u) due to kernel attenuation")
    print("  - Possible missing normalization in logN^ω term")
    print("  - Higher-degree polynomials attenuated more strongly")
    print()


if __name__ == "__main__":
    print_comparison_table()
    print()
    analyze_fd_ratio_by_case()
