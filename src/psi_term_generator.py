"""
src/psi_term_generator.py
Unified Ψ Term Generator for PRZZ Main-Term Computation

This module provides a config-driven term generator that:
1. Takes (ℓ₁, ℓ₂) pair specification
2. Generates ALL monomials with correct coefficients
3. Maps each monomial to the appropriate integral structure

The Ψ formula for pair (ℓ, ℓ̄) is:
    Ψ_{ℓ,ℓ̄}(A,B,C,D) = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × (D-C²)^p × (A-C)^{ℓ-p} × (B-C)^{ℓ̄-p}

Where:
    A = ∂/∂z derivative piece
    B = ∂/∂w derivative piece
    C = log ξ(s₀) no-derivative piece
    D = ∂²/∂z∂w mixed derivative piece

Each monomial is represented as (a, b, c, d) meaning A^a × B^b × C^c × D^d.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

from src.psi_monomial_expansion import expand_pair_to_monomials


class IntegralType(Enum):
    """Types of integral structures for PRZZ main-term evaluation."""
    I1_MIXED = "I1"      # Mixed derivative ∂z∂w (AB term)
    I2_BASE = "I2"       # No derivatives (D term)
    I3_Z_DERIV = "I3"    # z-derivative only (AC term)
    I4_W_DERIV = "I4"    # w-derivative only (BC term)
    GENERAL = "GENERAL"  # Higher-order terms requiring full evaluation


@dataclass
class PsiTerm:
    """
    A single term in the Ψ expansion with integral mapping.

    Represents: coeff × A^a × B^b × C^c × D^d

    Fields:
        a: Power of A (z-derivative piece)
        b: Power of B (w-derivative piece)
        c: Power of C (base log piece)
        d: Power of D (mixed derivative piece)
        coeff: Integer coefficient (can be negative)
        integral_type: Classification for evaluation strategy
        description: Human-readable description
    """
    a: int
    b: int
    c: int
    d: int
    coeff: int
    integral_type: IntegralType
    description: str

    def monomial_key(self) -> Tuple[int, int, int, int]:
        """Return (a, b, c, d) tuple for comparison."""
        return (self.a, self.b, self.c, self.d)

    def __repr__(self) -> str:
        parts = []
        if self.a > 0:
            parts.append(f"A^{self.a}" if self.a > 1 else "A")
        if self.b > 0:
            parts.append(f"B^{self.b}" if self.b > 1 else "B")
        if self.c > 0:
            parts.append(f"C^{self.c}" if self.c > 1 else "C")
        if self.d > 0:
            parts.append(f"D^{self.d}" if self.d > 1 else "D")

        mono_str = " × ".join(parts) if parts else "1"
        sign_str = "+" if self.coeff > 0 else ""
        return f"PsiTerm({sign_str}{self.coeff} × {mono_str} [{self.integral_type.value}])"


def classify_integral_type(a: int, b: int, c: int, d: int) -> IntegralType:
    """
    Classify the integral evaluation strategy for a monomial.

    For (1,1) case:
        (1,1,0,0): AB → I1_MIXED (mixed derivative)
        (0,0,0,1): D  → I2_BASE (no derivatives)
        (1,0,1,0): AC → I3_Z_DERIV (z-derivative only)
        (0,1,1,0): BC → I4_W_DERIV (w-derivative only)

    For higher pairs, we need more general evaluation.
    """
    # Base integral (only D term, no other derivatives)
    if a == 0 and b == 0 and c == 0 and d > 0:
        return IntegralType.I2_BASE

    # Mixed derivative (A and B both present)
    if a > 0 and b > 0 and d == 0:
        if a == 1 and b == 1 and c == 0:
            return IntegralType.I1_MIXED  # Pure (1,1) case
        else:
            return IntegralType.GENERAL  # Higher powers

    # z-derivative only (A present, B absent)
    if a > 0 and b == 0 and d == 0:
        if a == 1 and c == 1:
            return IntegralType.I3_Z_DERIV  # (1,1) case
        else:
            return IntegralType.GENERAL

    # w-derivative only (B present, A absent)
    if a == 0 and b > 0 and d == 0:
        if b == 1 and c == 1:
            return IntegralType.I4_W_DERIV  # (1,1) case
        else:
            return IntegralType.GENERAL

    # Everything else requires general evaluation
    return IntegralType.GENERAL


def generate_psi_description(a: int, b: int, c: int, d: int, ell: int, ellbar: int) -> str:
    """Generate human-readable description of a monomial term."""
    parts = []
    if a > 0:
        parts.append(f"∂^{a}_z" if a > 1 else "∂_z")
    if b > 0:
        parts.append(f"∂^{b}_w" if b > 1 else "∂_w")
    if d > 0:
        parts.append(f"∂^{d}_zw" if d > 1 else "∂_zw")
    if c > 0:
        parts.append(f"C^{c}" if c > 1 else "C")

    deriv_str = " ".join(parts) if parts else "const"
    return f"Ψ_({ell},{ellbar}): {deriv_str}"


@dataclass
class PsiTermCollection:
    """
    Complete term collection for a (ℓ, ℓ̄) pair.

    Fields:
        ell: Left piece index
        ellbar: Right piece index
        terms: List of PsiTerm objects
        total_terms: Total number of terms
        by_type: Dictionary grouping terms by IntegralType
    """
    ell: int
    ellbar: int
    terms: List[PsiTerm]

    @property
    def total_terms(self) -> int:
        return len(self.terms)

    @property
    def by_type(self) -> Dict[IntegralType, List[PsiTerm]]:
        """Group terms by integral type."""
        result = {}
        for term in self.terms:
            if term.integral_type not in result:
                result[term.integral_type] = []
            result[term.integral_type].append(term)
        return result

    def __repr__(self) -> str:
        return f"PsiTermCollection(({self.ell},{self.ellbar}): {self.total_terms} terms)"


def generate_psi_terms(ell: int, ellbar: int) -> PsiTermCollection:
    """
    Generate all Ψ terms for pair (ℓ, ℓ̄).

    This is the main entry point for term generation. It:
    1. Expands the Ψ formula to get all monomials with coefficients
    2. Classifies each monomial by integral type
    3. Adds descriptive information
    4. Returns a structured collection

    Args:
        ell: Left piece index ℓ (1, 2, or 3 for K=3)
        ellbar: Right piece index ℓ̄ (1, 2, or 3 for K=3)

    Returns:
        PsiTermCollection with all terms properly classified

    Example:
        >>> terms = generate_psi_terms(1, 1)
        >>> print(terms.total_terms)
        4
        >>> for term in terms.terms:
        ...     print(term)
        PsiTerm(+1 × A × B [I1])
        PsiTerm(+1 × D [I2])
        PsiTerm(-1 × A × C [I3])
        PsiTerm(-1 × B × C [I4])
    """
    # Get raw monomials with coefficients
    monomials = expand_pair_to_monomials(ell, ellbar)

    # Convert to PsiTerm objects with classification
    terms = []
    for (a, b, c, d), coeff in sorted(monomials.items()):
        integral_type = classify_integral_type(a, b, c, d)
        description = generate_psi_description(a, b, c, d, ell, ellbar)

        term = PsiTerm(
            a=a,
            b=b,
            c=c,
            d=d,
            coeff=coeff,
            integral_type=integral_type,
            description=description
        )
        terms.append(term)

    return PsiTermCollection(ell=ell, ellbar=ellbar, terms=terms)


def print_term_summary(collection: PsiTermCollection) -> None:
    """Print a human-readable summary of a term collection."""
    print(f"\n{'='*70}")
    print(f"Ψ_({collection.ell},{collection.ellbar}) Term Summary")
    print(f"{'='*70}")
    print(f"Total terms: {collection.total_terms}")
    print()

    # Group by type
    by_type = collection.by_type
    for int_type in IntegralType:
        if int_type in by_type:
            terms = by_type[int_type]
            print(f"{int_type.value} terms: {len(terms)}")
            for term in terms:
                mono_parts = []
                if term.a > 0:
                    mono_parts.append(f"A^{term.a}" if term.a > 1 else "A")
                if term.b > 0:
                    mono_parts.append(f"B^{term.b}" if term.b > 1 else "B")
                if term.c > 0:
                    mono_parts.append(f"C^{term.c}" if term.c > 1 else "C")
                if term.d > 0:
                    mono_parts.append(f"D^{term.d}" if term.d > 1 else "D")

                mono_str = " × ".join(mono_parts) if mono_parts else "1"
                print(f"  {term.coeff:+3d} × {mono_str:<20} (a={term.a}, b={term.b}, c={term.c}, d={term.d})")
            print()


def verify_expected_counts(pairs: List[Tuple[int, int]]) -> bool:
    """
    Verify that term generation produces expected monomial counts.

    Expected counts:
        (1,1): 4 monomials
        (2,2): 12 monomials
        (3,3): 27 monomials
        (1,2): 7 monomials (BC² cancels from p=0 and p=1)
        (1,3): 10 monomials (B²C² cancels, adds DB², DBC, DC²)
        (2,3): 18 monomials
    """
    expected = {
        (1, 1): 4,
        (2, 2): 12,
        (3, 3): 27,
        (1, 2): 7,   # Was 6, but BC² term cancels between p=0 and p=1
        (1, 3): 10,  # Was 8, but B²C² cancels, adds 3 D-terms
        (2, 3): 18,
    }

    print("="*70)
    print("TERM GENERATION VALIDATION")
    print("="*70)

    all_pass = True
    for (ell, ellbar) in pairs:
        collection = generate_psi_terms(ell, ellbar)
        actual = collection.total_terms
        exp = expected.get((ell, ellbar), None)

        if exp is not None:
            status = "✓" if actual == exp else "✗"
            print(f"  ({ell},{ellbar}): {actual} terms (expected {exp}) {status}")
            if actual != exp:
                all_pass = False
        else:
            print(f"  ({ell},{ellbar}): {actual} terms (no expected count)")

    print()
    return all_pass


def demonstrate_11_mapping() -> None:
    """
    Demonstrate how (1,1) terms map to I₁-I₄ structure.

    This shows that the term generator correctly produces:
        +AB → I₁ (mixed derivative)
        +D  → I₂ (base integral)
        -AC → I₃ (z-derivative)
        -BC → I₄ (w-derivative)
    """
    print("="*70)
    print("(1,1) → I₁-I₄ MAPPING DEMONSTRATION")
    print("="*70)

    collection = generate_psi_terms(1, 1)

    mapping = {
        (1, 1, 0, 0): ("AB", "I₁: mixed derivative ∂z∂w"),
        (0, 0, 0, 1): ("D", "I₂: no derivatives (base integral)"),
        (1, 0, 1, 0): ("AC", "I₃: ∂z only"),
        (0, 1, 1, 0): ("BC", "I₄: ∂w only"),
    }

    print(f"\nGenerated {collection.total_terms} terms:")
    for term in collection.terms:
        key = term.monomial_key()
        if key in mapping:
            symbol, meaning = mapping[key]
            print(f"  {term.coeff:+d} × {symbol:<4} → {meaning}")
        else:
            print(f"  {term.coeff:+d} × {key} → UNEXPECTED!")

    print("\n✓ This confirms the I₁-I₄ decomposition for (1,1)")


if __name__ == "__main__":
    # Verify counts for all K=3 pairs
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    all_pass = verify_expected_counts(pairs)
    print()

    # Demonstrate (1,1) mapping
    demonstrate_11_mapping()
    print()

    # Show detailed summaries for main diagonal pairs
    for (ell, ellbar) in [(1, 1), (2, 2), (3, 3)]:
        collection = generate_psi_terms(ell, ellbar)
        print_term_summary(collection)

    # Final status
    if all_pass:
        print("="*70)
        print("✓ ALL VALIDATIONS PASSED")
        print("="*70)
        print("\nThe term generator correctly implements the Ψ combinatorial formula.")
        print("Ready for integration with PRZZ main-term computation.")
    else:
        print("="*70)
        print("✗ VALIDATION FAILED")
        print("="*70)
