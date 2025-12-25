"""
run_s34_assembly_test.py
Test three S34 assembly candidates (GPT instructions 2025-12-19)

The question: How should off-diagonal I₃/I₄ terms be assembled?

Candidate 1: ORDERED
  S34 = sum over all 9 ordered pairs: I₃(pq) + I₄(pq) for each (p,q)
  No symmetry factor applied.

Candidate 2: TRIANGLE×2 (control - known wrong)
  S34 = 2 × [I₃(12) + I₄(12)] + 2 × [I₃(13) + I₄(13)] + 2 × [I₃(23) + I₄(23)] + diag
  This assumes S34(pq) = S34(qp), which we proved FALSE.

Candidate 3: CROSS-RECOMBINED
  Based on TeX observation that I₃ uses d/dx, I₄ uses d/dy.
  Hypothesis: I₃(pq) ≈ I₄(qp) due to x↔y symmetry with polynomial swap.
  If true: pair I₃(pq) with I₄(qp) for off-diagonals.

This script:
1. Computes I₃, I₄ separately for all 9 pairs
2. Tests cross-symmetry: I₃(pq) vs I₄(qp)
3. Computes c using all three assemblies
4. Reports which assembly eliminates 12/21 pathology
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from src.terms_k3_d1 import make_all_terms_k3_ordered
from src.evaluate import evaluate_term
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

# Constants
THETA = 4.0 / 7.0
N_QUAD = 60
N_QUAD_A = 40

# Factorial weights
FACTORIAL_WEIGHTS: Dict[str, float] = {
    "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1.0
    "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 0.25
    "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
    "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 0.5
    "21": 1.0 / (math.factorial(2) * math.factorial(1)),  # 0.5
    "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
    "31": 1.0 / (math.factorial(3) * math.factorial(1)),  # 1/6
    "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    "32": 1.0 / (math.factorial(3) * math.factorial(2)),  # 1/12
}

ORDERED_PAIRS = ["11", "22", "33", "12", "21", "13", "31", "23", "32"]
DIAGONAL_PAIRS = ["11", "22", "33"]
OFF_DIAGONAL_PAIRS = [("12", "21"), ("13", "31"), ("23", "32")]


@dataclass
class PairITerms:
    """I-term values for a single ordered pair."""
    pair: str
    I1: float
    I2: float
    I3: float
    I4: float

    @property
    def S12(self) -> float:
        """I1 + I2"""
        return self.I1 + self.I2

    @property
    def S34(self) -> float:
        """I3 + I4"""
        return self.I3 + self.I4

    @property
    def total(self) -> float:
        """I1 + I2 + I3 + I4"""
        return self.I1 + self.I2 + self.I3 + self.I4

    @property
    def weight(self) -> float:
        return FACTORIAL_WEIGHTS[self.pair]

    @property
    def S12_weighted(self) -> float:
        return self.weight * self.S12

    @property
    def S34_weighted(self) -> float:
        return self.weight * self.S34

    @property
    def total_weighted(self) -> float:
        return self.weight * self.total


def compute_pair_iterms(
    pair: str,
    terms: List,
    polynomials: Dict,
    R: float,
    n: int = N_QUAD,
    n_quad_a: int = N_QUAD_A,
) -> PairITerms:
    """Compute I1, I2, I3, I4 for a single pair."""
    # Terms are ordered: I1, I2, I3, I4 (indices 0, 1, 2, 3)
    I1 = evaluate_term(terms[0], polynomials, n, R=R, theta=THETA, n_quad_a=n_quad_a).value
    I2 = evaluate_term(terms[1], polynomials, n, R=R, theta=THETA, n_quad_a=n_quad_a).value
    I3 = evaluate_term(terms[2], polynomials, n, R=R, theta=THETA, n_quad_a=n_quad_a).value
    I4 = evaluate_term(terms[3], polynomials, n, R=R, theta=THETA, n_quad_a=n_quad_a).value
    return PairITerms(pair=pair, I1=I1, I2=I2, I3=I3, I4=I4)


def run_s34_assembly_test(
    name: str,
    R: float,
    c_target: float,
    polynomials: Dict,
    n: int = N_QUAD,
    n_quad_a: int = N_QUAD_A,
):
    """Run complete S34 assembly test for a benchmark."""
    print()
    print("=" * 100)
    print(f"S34 ASSEMBLY TEST: {name} (R={R}, c_target={c_target})")
    print("=" * 100)

    # Get all terms
    all_terms = make_all_terms_k3_ordered(THETA, R, kernel_regime="paper")

    # Compute I-terms for all pairs
    pair_data: Dict[str, PairITerms] = {}
    for pair in ORDERED_PAIRS:
        pair_data[pair] = compute_pair_iterms(
            pair, all_terms[pair], polynomials, R, n, n_quad_a
        )

    # =========================================================================
    # Section 1: Raw I₃, I₄ values per pair
    # =========================================================================
    print()
    print("Section 1: Individual I₃, I₄ values (RAW, not weighted)")
    print("-" * 100)
    print(f"{'Pair':>6}  {'I3':>14}  {'I4':>14}  {'S34=I3+I4':>14}")
    print("-" * 100)
    for pair in ORDERED_PAIRS:
        d = pair_data[pair]
        print(f"  {pair:>4}  {d.I3:+14.8f}  {d.I4:+14.8f}  {d.S34:+14.8f}")

    # =========================================================================
    # Section 2: Cross-symmetry test: I₃(pq) vs I₄(qp)
    # =========================================================================
    print()
    print("Section 2: Cross-symmetry test: I₃(pq) vs I₄(qp)")
    print("-" * 100)
    print("Hypothesis: Due to x↔y symmetry in TeX, I₃(pq) ≈ I₄(qp)")
    print("-" * 100)
    print(f"{'Pair':>8}  {'I3(pq)':>14}  {'I4(qp)':>14}  {'Delta':>14}  {'Rel%':>10}")
    print("-" * 100)

    cross_sym_holds = True
    for pq, qp in OFF_DIAGONAL_PAIRS:
        I3_pq = pair_data[pq].I3
        I4_qp = pair_data[qp].I4
        delta = I3_pq - I4_qp
        rel = 100 * delta / abs(I3_pq) if abs(I3_pq) > 1e-15 else float('inf')
        sym_status = "YES" if abs(rel) < 1.0 else "NO"
        if abs(rel) >= 1.0:
            cross_sym_holds = False
        print(f"  {pq}/{qp}  {I3_pq:+14.8f}  {I4_qp:+14.8f}  {delta:+14.8f}  {rel:+10.2f}%  {sym_status}")

        # Also check I₃(qp) vs I₄(pq)
        I3_qp = pair_data[qp].I3
        I4_pq = pair_data[pq].I4
        delta2 = I3_qp - I4_pq
        rel2 = 100 * delta2 / abs(I3_qp) if abs(I3_qp) > 1e-15 else float('inf')
        sym_status2 = "YES" if abs(rel2) < 1.0 else "NO"
        if abs(rel2) >= 1.0:
            cross_sym_holds = False
        print(f"  {qp}/{pq}  {I3_qp:+14.8f}  {I4_pq:+14.8f}  {delta2:+14.8f}  {rel2:+10.2f}%  {sym_status2}")

    print("-" * 100)
    if cross_sym_holds:
        print("RESULT: Cross-symmetry I₃(pq) ≈ I₄(qp) HOLDS within 1%")
    else:
        print("RESULT: Cross-symmetry I₃(pq) ≈ I₄(qp) does NOT hold")

    # =========================================================================
    # Section 3: Compute S12 using triangle×2 (known to be correct)
    # =========================================================================
    s12_total = 0.0
    for pair in DIAGONAL_PAIRS:
        s12_total += pair_data[pair].S12_weighted
    for pq, qp in OFF_DIAGONAL_PAIRS:
        # Triangle×2: use pq term × 2 (equiv to pq + qp since S12 is symmetric)
        s12_total += 2 * pair_data[pq].S12_weighted

    # =========================================================================
    # Section 4: Compute S34 using three candidate assemblies
    # =========================================================================
    print()
    print("Section 3: S34 Assembly Candidates")
    print("-" * 100)

    # Candidate 1: ORDERED (sum all 9 pairs)
    s34_ordered = 0.0
    for pair in ORDERED_PAIRS:
        s34_ordered += pair_data[pair].S34_weighted

    # Candidate 2: TRIANGLE×2 (pq × 2 for off-diagonals)
    s34_triangle = 0.0
    for pair in DIAGONAL_PAIRS:
        s34_triangle += pair_data[pair].S34_weighted
    for pq, qp in OFF_DIAGONAL_PAIRS:
        s34_triangle += 2 * pair_data[pq].S34_weighted

    # Candidate 3: CROSS-RECOMBINED
    # For off-diagonals: pair I₃(pq) with I₄(qp) and I₃(qp) with I₄(pq)
    # So: [I₃(pq) + I₄(qp)] + [I₃(qp) + I₄(pq)] with weight
    s34_cross = 0.0
    for pair in DIAGONAL_PAIRS:
        s34_cross += pair_data[pair].S34_weighted
    for pq, qp in OFF_DIAGONAL_PAIRS:
        # Weight is the same for pq and qp
        w = FACTORIAL_WEIGHTS[pq]
        I3_pq = pair_data[pq].I3
        I4_qp = pair_data[qp].I4
        I3_qp = pair_data[qp].I3
        I4_pq = pair_data[pq].I4
        # Cross-recombined: [I₃(pq) + I₄(qp)] + [I₃(qp) + I₄(pq)]
        s34_cross += w * (I3_pq + I4_qp) + w * (I3_qp + I4_pq)

    print(f"  S12 (triangle×2, verified symmetric):  {s12_total:+14.8f}")
    print()
    print(f"  S34 Candidate 1 (ORDERED):             {s34_ordered:+14.8f}")
    print(f"  S34 Candidate 2 (TRIANGLE×2):          {s34_triangle:+14.8f}")
    print(f"  S34 Candidate 3 (CROSS-RECOMBINED):    {s34_cross:+14.8f}")

    # =========================================================================
    # Section 5: Total c for each assembly
    # =========================================================================
    print()
    print("Section 4: Total c = S12 + S34 for each assembly")
    print("-" * 100)

    c_ordered = s12_total + s34_ordered
    c_triangle = s12_total + s34_triangle
    c_cross = s12_total + s34_cross

    gap_ordered = 100 * (c_ordered - c_target) / c_target
    gap_triangle = 100 * (c_triangle - c_target) / c_target
    gap_cross = 100 * (c_cross - c_target) / c_target

    print(f"{'Assembly':>25}  {'c':>12}  {'Target':>12}  {'Gap':>10}")
    print("-" * 100)
    print(f"  {'Candidate 1 (ORDERED)':>23}  {c_ordered:12.6f}  {c_target:12.6f}  {gap_ordered:+10.2f}%")
    print(f"  {'Candidate 2 (TRIANGLE×2)':>23}  {c_triangle:12.6f}  {c_target:12.6f}  {gap_triangle:+10.2f}%")
    print(f"  {'Candidate 3 (CROSS)':>23}  {c_cross:12.6f}  {c_target:12.6f}  {gap_cross:+10.2f}%")

    # =========================================================================
    # Section 6: 12/21 Sign Analysis
    # =========================================================================
    print()
    print("Section 5: 12/21 Sign Inversion Analysis")
    print("-" * 100)

    d12 = pair_data["12"]
    d21 = pair_data["21"]

    print(f"  Pair 12: S34 = {d12.S34:+14.8f}  (weighted: {d12.S34_weighted:+14.8f})")
    print(f"  Pair 21: S34 = {d21.S34:+14.8f}  (weighted: {d21.S34_weighted:+14.8f})")
    print(f"  Delta S34(12) - S34(21) = {d12.S34 - d21.S34:+14.8f}")

    # Cross-recombined analysis
    I3_12 = d12.I3
    I4_21 = d21.I4
    I3_21 = d21.I3
    I4_12 = d12.I4

    print()
    print("  Cross-recombined breakdown:")
    print(f"    I₃(12) = {I3_12:+14.8f}")
    print(f"    I₄(21) = {I4_21:+14.8f}")
    print(f"    I₃(12) + I₄(21) = {I3_12 + I4_21:+14.8f}")
    print()
    print(f"    I₃(21) = {I3_21:+14.8f}")
    print(f"    I₄(12) = {I4_12:+14.8f}")
    print(f"    I₃(21) + I₄(12) = {I3_21 + I4_12:+14.8f}")

    # Check if signs are aligned in cross-recombined
    cross_12_21 = I3_12 + I4_21
    cross_21_12 = I3_21 + I4_12
    if cross_12_21 * cross_21_12 > 0:
        print()
        print("  CROSS-RECOMBINED: Signs ALIGNED (both same sign)")
    else:
        print()
        print("  CROSS-RECOMBINED: Signs STILL INVERTED")

    # =========================================================================
    # Section 7: Best assembly recommendation
    # =========================================================================
    print()
    print("=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)

    gaps = [
        ("ORDERED", abs(gap_ordered)),
        ("TRIANGLE×2", abs(gap_triangle)),
        ("CROSS-RECOMBINED", abs(gap_cross)),
    ]
    best = min(gaps, key=lambda x: x[1])
    print(f"  Best assembly: {best[0]} with |gap| = {best[1]:.2f}%")

    return {
        "c_ordered": c_ordered,
        "c_triangle": c_triangle,
        "c_cross": c_cross,
        "c_target": c_target,
        "s12_total": s12_total,
        "s34_ordered": s34_ordered,
        "s34_triangle": s34_triangle,
        "s34_cross": s34_cross,
        "cross_symmetry_holds": cross_sym_holds,
    }


def main():
    print("=" * 100)
    print("S34 ASSEMBLY CANDIDATE TEST")
    print("Testing: ORDERED vs TRIANGLE×2 vs CROSS-RECOMBINED")
    print("=" * 100)

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Run tests
    results_k = run_s34_assembly_test(
        "κ", R=1.3036, c_target=2.137, polynomials=polys_k
    )

    results_s = run_s34_assembly_test(
        "κ*", R=1.1167, c_target=1.938, polynomials=polys_s
    )

    # Summary
    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print()
    print(f"{'Benchmark':>10}  {'Assembly':>15}  {'c':>12}  {'Target':>12}  {'Gap%':>10}")
    print("-" * 100)

    for name, r, R in [("κ", results_k, 1.3036), ("κ*", results_s, 1.1167)]:
        for asm, c_val in [
            ("ORDERED", r["c_ordered"]),
            ("TRIANGLE×2", r["c_triangle"]),
            ("CROSS", r["c_cross"]),
        ]:
            gap = 100 * (c_val - r["c_target"]) / r["c_target"]
            print(f"  {name:>8}  {asm:>15}  {c_val:12.6f}  {r['c_target']:12.6f}  {gap:+10.2f}%")
        print()

    # Cross-symmetry summary
    print("-" * 100)
    print("Cross-symmetry I₃(pq) ≈ I₄(qp) status:")
    print(f"  κ:  {'HOLDS' if results_k['cross_symmetry_holds'] else 'BROKEN'}")
    print(f"  κ*: {'HOLDS' if results_s['cross_symmetry_holds'] else 'BROKEN'}")


if __name__ == "__main__":
    main()
