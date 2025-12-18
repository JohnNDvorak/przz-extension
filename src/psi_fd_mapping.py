"""
src/psi_fd_mapping.py
Map Ψ monomials to (k, l, m) triples for F_d evaluation.

This module provides the critical mapping between:
  - Ψ monomials: A^a × B^b × C^c × D^d (from psi_monomial_expansion.py)
  - F_d triples: (k₁, l₁, m₁) for PRZZ Section 7 evaluation

Key formulas:
  - l₁ = a + d (left derivative count: A's plus D's)
  - m₁ = b + d (right derivative count: B's plus D's)
  - k₁ = c (convolution index: C's)
  - ω_left = l₁ - 1 (determines Case A/B/C for left F_d)
  - ω_right = m₁ - 1 (determines Case A/B/C for right F_d)

Case classification (d=1):
  - Case A: ω = -1 (l=0) → derivative form, no kernel integral
  - Case B: ω = 0 (l=1) → direct polynomial evaluation
  - Case C: ω > 0 (l>1) → kernel integral with a-variable

PRZZ Reference: arXiv:1802.10521, Section 7
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, NamedTuple
from enum import Enum

from src.psi_monomial_expansion import expand_pair_to_monomials


class FdCase(Enum):
    """F_d evaluation case based on ω value."""
    A = "A"  # ω = -1 (l=0)
    B = "B"  # ω = 0 (l=1)
    C = "C"  # ω > 0 (l>1)


@dataclass
class FdTriple:
    """
    (k₁, l₁, m₁) triple for F_d evaluation.

    Attributes:
        k1: Convolution index (from C factors)
        l1: Left derivative count (from A and D factors)
        m1: Right derivative count (from B and D factors)
        omega_left: ω for left F_d = l₁ - 1
        omega_right: ω for right F_d = m₁ - 1
        case_left: Case A/B/C for left F_d
        case_right: Case A/B/C for right F_d
    """
    k1: int
    l1: int
    m1: int
    omega_left: int
    omega_right: int
    case_left: FdCase
    case_right: FdCase

    def key(self) -> Tuple[int, int, int]:
        """Return (k₁, l₁, m₁) tuple for grouping."""
        return (self.k1, self.l1, self.m1)

    def case_pair(self) -> Tuple[FdCase, FdCase]:
        """Return (Case_left, Case_right) tuple."""
        return (self.case_left, self.case_right)

    def __repr__(self) -> str:
        return f"FdTriple(k1={self.k1}, l1={self.l1}, m1={self.m1}, cases={self.case_left.value},{self.case_right.value})"


@dataclass
class MonomialMapping:
    """
    Complete mapping for a single Ψ monomial.

    Combines the (a,b,c,d) exponents with the derived (k,l,m) triple.
    """
    a: int  # A exponent
    b: int  # B exponent
    c: int  # C exponent
    d: int  # D exponent
    coeff: int  # Ψ coefficient
    triple: FdTriple

    def abcd(self) -> Tuple[int, int, int, int]:
        return (self.a, self.b, self.c, self.d)

    def __repr__(self) -> str:
        return f"MonomialMapping(A^{self.a}B^{self.b}C^{self.c}D^{self.d}, coeff={self.coeff:+d}, {self.triple})"


def get_case(omega: int) -> FdCase:
    """Determine F_d Case from ω value."""
    if omega == -1:
        return FdCase.A
    elif omega == 0:
        return FdCase.B
    else:
        return FdCase.C


def monomial_to_triple(a: int, b: int, c: int, d: int) -> FdTriple:
    """
    Map (a,b,c,d) exponents to (k₁, l₁, m₁) triple.

    Args:
        a: Power of A (ζ'/ζ with α)
        b: Power of B (ζ'/ζ with β)
        c: Power of C (common ζ'/ζ)
        d: Power of D (common (ζ'/ζ)')

    Returns:
        FdTriple with computed (k₁, l₁, m₁) and Cases
    """
    # Core mapping formulas
    l1 = a + d  # Left derivative count
    m1 = b + d  # Right derivative count
    k1 = c      # Convolution index

    # ω values determine Case
    omega_left = l1 - 1
    omega_right = m1 - 1

    case_left = get_case(omega_left)
    case_right = get_case(omega_right)

    return FdTriple(
        k1=k1,
        l1=l1,
        m1=m1,
        omega_left=omega_left,
        omega_right=omega_right,
        case_left=case_left,
        case_right=case_right
    )


def map_pair_monomials(ell: int, ellbar: int) -> List[MonomialMapping]:
    """
    Map all Ψ monomials for pair (ℓ, ℓ̄) to their F_d triples.

    Args:
        ell: Left piece index (1, 2, or 3 for K=3)
        ellbar: Right piece index

    Returns:
        List of MonomialMapping objects
    """
    monomials = expand_pair_to_monomials(ell, ellbar)

    mappings = []
    for (a, b, c, d), coeff in monomials.items():
        triple = monomial_to_triple(a, b, c, d)
        mappings.append(MonomialMapping(
            a=a, b=b, c=c, d=d,
            coeff=coeff,
            triple=triple
        ))

    return mappings


def group_by_triple(mappings: List[MonomialMapping]) -> Dict[Tuple[int,int,int], List[MonomialMapping]]:
    """
    Group monomials by their (k₁, l₁, m₁) triple.

    Monomials with the same triple use the same F_d × F_d structure
    and can be evaluated together.
    """
    groups = defaultdict(list)
    for m in mappings:
        groups[m.triple.key()].append(m)
    return dict(groups)


def group_by_case_pair(mappings: List[MonomialMapping]) -> Dict[Tuple[FdCase, FdCase], List[MonomialMapping]]:
    """
    Group monomials by their Case pair (Case_left, Case_right).

    This is useful for understanding the integral structure needed.
    """
    groups = defaultdict(list)
    for m in mappings:
        groups[m.triple.case_pair()].append(m)
    return dict(groups)


def print_pair_analysis(ell: int, ellbar: int) -> None:
    """Print detailed analysis of (ℓ, ℓ̄) pair mapping."""
    mappings = map_pair_monomials(ell, ellbar)

    print(f"\n{'='*80}")
    print(f"Ψ → F_d MAPPING FOR ({ell},{ellbar})")
    print(f"{'='*80}")
    print(f"Total monomials: {len(mappings)}")

    # Group by triple
    by_triple = group_by_triple(mappings)
    print(f"Unique (k₁,l₁,m₁) triples: {len(by_triple)}")

    # Group by case pair
    by_case = group_by_case_pair(mappings)
    print(f"Unique Case pairs: {len(by_case)}")

    print(f"\n--- By (k₁,l₁,m₁) Triple ---")
    for (k1, l1, m1), group in sorted(by_triple.items()):
        total_coeff = sum(m.coeff for m in group)
        case_str = f"{group[0].triple.case_left.value},{group[0].triple.case_right.value}"
        print(f"  ({k1},{l1},{m1}) Case {case_str}: {len(group)} monomials, net coeff = {total_coeff:+d}")
        for m in group:
            print(f"    {m.coeff:+2d} × A^{m.a}B^{m.b}C^{m.c}D^{m.d}")

    print(f"\n--- By Case Pair ---")
    for (case_l, case_r), group in sorted(by_case.items(), key=lambda x: (x[0][0].value, x[0][1].value)):
        total_coeff = sum(m.coeff for m in group)
        print(f"  ({case_l.value},{case_r.value}): {len(group)} monomials, net coeff = {total_coeff:+d}")


def summary_table_k3() -> None:
    """Print summary table for all K=3 pairs."""
    pairs = [(1,1), (2,2), (3,3), (1,2), (1,3), (2,3)]

    print("\n" + "="*90)
    print("K=3 Ψ → F_d MAPPING SUMMARY")
    print("="*90)
    print(f"{'Pair':^8} | {'Monomials':^10} | {'Triples':^8} | Case Pairs Distribution")
    print("-"*90)

    for ell, ellbar in pairs:
        mappings = map_pair_monomials(ell, ellbar)
        by_triple = group_by_triple(mappings)
        by_case = group_by_case_pair(mappings)

        case_dist = []
        for (cl, cr), group in sorted(by_case.items(), key=lambda x: (x[0][0].value, x[0][1].value)):
            case_dist.append(f"{cl.value},{cr.value}:{len(group)}")

        print(f"({ell},{ellbar})     |     {len(mappings):2d}     |    {len(by_triple):2d}    | {' | '.join(case_dist)}")

    print("-"*90)

    # Show I-term correspondence for (1,1)
    print("\n--- (1,1) I-term Correspondence ---")
    mappings_11 = map_pair_monomials(1, 1)
    by_triple_11 = group_by_triple(mappings_11)

    print("(k₁,l₁,m₁) = (0,1,1) Case B,B → I₁ (AB) + I₂ (D) structure")
    print("(k₁,l₁,m₁) = (1,1,0) Case B,A → I₃ (AC) structure")
    print("(k₁,l₁,m₁) = (1,0,1) Case A,B → I₄ (BC) structure")

    for (k1, l1, m1), group in sorted(by_triple_11.items()):
        monomials_str = " + ".join(f"{m.coeff:+d}×A^{m.a}B^{m.b}C^{m.c}D^{m.d}" for m in group)
        print(f"  ({k1},{l1},{m1}): {monomials_str}")


def get_eval_structure(ell: int, ellbar: int) -> Dict:
    """
    Get the evaluation structure for pair (ℓ, ℓ̄).

    Returns a dict suitable for the Section 7 evaluator:
    {
        (k1, l1, m1): {
            'case_left': FdCase,
            'case_right': FdCase,
            'omega_left': int,
            'omega_right': int,
            'psi_coeff': int,  # Sum of Ψ coefficients for this triple
            'monomials': [(a,b,c,d,coeff), ...]
        },
        ...
    }
    """
    mappings = map_pair_monomials(ell, ellbar)
    by_triple = group_by_triple(mappings)

    result = {}
    for (k1, l1, m1), group in by_triple.items():
        result[(k1, l1, m1)] = {
            'case_left': group[0].triple.case_left,
            'case_right': group[0].triple.case_right,
            'omega_left': group[0].triple.omega_left,
            'omega_right': group[0].triple.omega_right,
            'psi_coeff': sum(m.coeff for m in group),
            'monomials': [(m.a, m.b, m.c, m.d, m.coeff) for m in group]
        }

    return result


if __name__ == "__main__":
    # Run analysis
    summary_table_k3()

    print("\n")
    print_pair_analysis(1, 1)
    print_pair_analysis(2, 2)
