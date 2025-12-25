"""
src/psi_expansion.py
CANONICAL Psi Expansion Module for PRZZ Section 7

This is the SINGLE SOURCE OF TRUTH for Ψ expansion.
All downstream modules (fd_evaluation.py, evaluators, tests) must import from here.

CRITICAL INVARIANTS (HARD GATES):
    Monomial counts after combining like terms:
    - (1,1) = 4 monomials
    - (2,2) = 12 monomials
    - (3,3) = 27 monomials

STRUCTURE:
    X = A - C_beta    (alpha-side block minus beta-pole contribution)
    Y = B - C_alpha   (beta-side block minus alpha-pole contribution)
    Z = D - C_alpha * C_beta   (mixed block minus pole product)

    Ψ_{ℓ,ℓ̄} = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}

BLOCK MEANINGS:
    A = zeta'/zeta at (1+s+u) with z-derivative (singleton x-block)
    B = zeta'/zeta at (1+s+u) with w-derivative (singleton y-block)
    C_alpha = contribution from 1/zeta(1+alpha+s) pole
    C_beta = contribution from 1/zeta(1+beta+u) pole
    D = (zeta'/zeta)' at (1+s+u) mixed z,w derivative (paired block)

WEIGHT RULE (Euler-Maclaurin):
    Each monomial A^a B^b C_α^{c_α} C_β^{c_β} D^d has weight (1-u)^{a+b}
    - A blocks (singleton x): contribute (1-u)
    - B blocks (singleton y): contribute (1-u)
    - D blocks (paired): contribute NO weight factor
    - C blocks: are factors, not integration blocks

DERIVATIVE STRUCTURE:
    For F_d evaluation:
    - l₁ = a (x-derivative count from A singletons)
    - m₁ = b (y-derivative count from B singletons)
    - D blocks contribute to the paired integral structure (I₂-type), not polynomial derivatives
"""

from __future__ import annotations
from dataclasses import dataclass
from math import comb, factorial
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# ============================================================================
# HARD GATES: Expected monomial counts (must pass or implementation is wrong)
# ============================================================================

EXPECTED_MONOMIAL_COUNTS = {
    (1, 1): 4,
    (1, 2): 7,
    (1, 3): 10,
    (2, 2): 12,
    (2, 3): 18,
    (3, 3): 27,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MonomialTwoC:
    """
    A monomial A^a × B^b × C_α^{c_α} × C_β^{c_β} × D^d with integer coefficient.

    This is the canonical monomial type for all PRZZ evaluators.

    Attributes:
        a: Power of A (singleton x-block, contributes x-derivative)
        b: Power of B (singleton y-block, contributes y-derivative)
        c_alpha: Power of C_α (from 1+α+s pole)
        c_beta: Power of C_β (from 1+β+u pole)
        d: Power of D (paired block, no additional derivatives)
        coeff: Integer coefficient (can be negative from Ψ expansion)

    Derivative structure:
        l₁ = a (x-derivatives)
        m₁ = b (y-derivatives)

    Weight:
        (1-u)^{a+b}
    """
    a: int          # Power of A (singleton x-block)
    b: int          # Power of B (singleton y-block)
    c_alpha: int    # Power of C_α (α-pole contribution)
    c_beta: int     # Power of C_β (β-pole contribution)
    d: int          # Power of D (paired block)
    coeff: int      # Integer coefficient

    def key(self) -> Tuple[int, int, int, int, int]:
        """Unique key for combining like terms."""
        return (self.a, self.b, self.c_alpha, self.c_beta, self.d)

    @property
    def l1(self) -> int:
        """X-derivative count (from A singletons)."""
        return self.a

    @property
    def m1(self) -> int:
        """Y-derivative count (from B singletons)."""
        return self.b

    @property
    def weight_exponent(self) -> int:
        """Euler-Maclaurin weight exponent: (1-u)^{a+b}."""
        return self.a + self.b

    def __repr__(self) -> str:
        parts = []
        if self.coeff != 1 or (self.a == 0 and self.b == 0 and
                               self.c_alpha == 0 and self.c_beta == 0 and self.d == 0):
            parts.append(f"{self.coeff:+d}")
        if self.a > 0:
            parts.append(f"A^{self.a}" if self.a > 1 else "A")
        if self.b > 0:
            parts.append(f"B^{self.b}" if self.b > 1 else "B")
        if self.c_alpha > 0:
            parts.append(f"C_α^{self.c_alpha}" if self.c_alpha > 1 else "C_α")
        if self.c_beta > 0:
            parts.append(f"C_β^{self.c_beta}" if self.c_beta > 1 else "C_β")
        if self.d > 0:
            parts.append(f"D^{self.d}" if self.d > 1 else "D")
        return "×".join(parts) if parts else "1"


# ============================================================================
# PSI EXPANSION
# ============================================================================

def expand_psi(ell: int, ellbar: int) -> List[MonomialTwoC]:
    """
    Generate all monomials for Ψ_{ℓ,ℓ̄} with combined like terms.

    This is the CANONICAL expansion function. All downstream code should use this.

    Args:
        ell: Left piece index (1, 2, or 3 for K=3)
        ellbar: Right piece index (1, 2, or 3 for K=3)

    Returns:
        List of MonomialTwoC with combined coefficients

    Raises:
        AssertionError: If monomial count doesn't match expected (HARD GATE)
    """
    monomial_dict: Dict[Tuple[int, int, int, int, int], int] = defaultdict(int)

    # Sum over p-configurations
    for p in range(0, min(ell, ellbar) + 1):
        p_coeff = comb(ell, p) * comb(ellbar, p) * factorial(p)
        x_exp = ell - p      # X = (A - C_β)
        y_exp = ellbar - p   # Y = (B - C_α)
        z_exp = p            # Z = (D - C_α×C_β)

        # Expand X^{x_exp} × Y^{y_exp} × Z^{z_exp}
        for i in range(x_exp + 1):  # A^i × (-C_β)^{x_exp-i}
            coeff_x = comb(x_exp, i) * ((-1) ** (x_exp - i))
            c_beta_from_x = x_exp - i

            for j in range(y_exp + 1):  # B^j × (-C_α)^{y_exp-j}
                coeff_y = comb(y_exp, j) * ((-1) ** (y_exp - j))
                c_alpha_from_y = y_exp - j

                for r in range(z_exp + 1):  # D^r × (-C_α×C_β)^{z_exp-r}
                    coeff_z = comb(z_exp, r) * ((-1) ** (z_exp - r))

                    # Final exponents
                    a = i
                    b = j
                    d = r
                    c_alpha = c_alpha_from_y + (z_exp - r)
                    c_beta = c_beta_from_x + (z_exp - r)

                    # Accumulate coefficient
                    total_coeff = p_coeff * coeff_x * coeff_y * coeff_z
                    if total_coeff != 0:
                        key = (a, b, c_alpha, c_beta, d)
                        monomial_dict[key] += total_coeff

    # Convert to list, excluding zero coefficients
    result = []
    for (a, b, c_alpha, c_beta, d), coeff in monomial_dict.items():
        if coeff != 0:
            result.append(MonomialTwoC(
                a=a, b=b, c_alpha=c_alpha, c_beta=c_beta, d=d, coeff=coeff
            ))

    # Sort for consistent ordering
    result.sort(key=lambda m: m.key())

    # HARD GATE: Verify monomial count
    pair = (ell, ellbar)
    if pair in EXPECTED_MONOMIAL_COUNTS:
        expected = EXPECTED_MONOMIAL_COUNTS[pair]
        actual = len(result)
        assert actual == expected, (
            f"HARD GATE FAILURE: ({ell},{ellbar}) has {actual} monomials, expected {expected}"
        )

    return result


def get_monomial_count(ell: int, ellbar: int) -> int:
    """Get the expected monomial count for a pair."""
    return len(expand_psi(ell, ellbar))


# ============================================================================
# VALIDATION
# ============================================================================

def validate_expansion(ell: int, ellbar: int, verbose: bool = False) -> bool:
    """
    Validate the Ψ expansion for a pair.

    Args:
        ell: Left piece index
        ellbar: Right piece index
        verbose: Print detailed output

    Returns:
        True if validation passes
    """
    try:
        monomials = expand_psi(ell, ellbar)

        if verbose:
            print(f"\nΨ_{{{ell},{ellbar}}} = {len(monomials)} monomials:")
            for m in monomials:
                print(f"  {m} | l₁={m.l1}, m₁={m.m1}, weight=(1-u)^{m.weight_exponent}")

        # Check specific invariants for (1,1)
        if ell == 1 and ellbar == 1:
            expected = {
                (1, 1, 0, 0, 0): 1,   # AB
                (1, 0, 1, 0, 0): -1,  # -AC_α
                (0, 1, 0, 1, 0): -1,  # -BC_β
                (0, 0, 0, 0, 1): 1,   # D
            }
            actual = {m.key(): m.coeff for m in monomials}
            if actual != expected:
                if verbose:
                    print(f"  ERROR: (1,1) monomials don't match expected!")
                    print(f"  Expected: {expected}")
                    print(f"  Actual: {actual}")
                return False

        return True

    except AssertionError as e:
        if verbose:
            print(f"  Validation failed: {e}")
        return False


def validate_all_k3_pairs(verbose: bool = False) -> bool:
    """Validate all K=3 pairs."""
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    all_pass = True

    for ell, ellbar in pairs:
        if not validate_expansion(ell, ellbar, verbose):
            all_pass = False

    return all_pass


# ============================================================================
# I-TERM MAPPING (for diagnostic purposes)
# ============================================================================

def classify_monomial(m: MonomialTwoC) -> str:
    """
    Classify a monomial by its I-term type (for diagnostics only).

    For (1,1), this gives the exact I₁-I₄ mapping.
    For higher pairs, this is approximate.
    """
    a, b, c_alpha, c_beta, d = m.a, m.b, m.c_alpha, m.c_beta, m.d

    # I₂-type: no singleton derivatives (a=b=0), paired block present (d>0) or pure C's
    if a == 0 and b == 0:
        if d > 0:
            return "I₂-type (D block)"
        else:
            return "I₂-type (C-only)"

    # I₃-type: x-derivative only (a>0, b=0)
    if a > 0 and b == 0:
        return "I₃-type (x-deriv)"

    # I₄-type: y-derivative only (a=0, b>0)
    if a == 0 and b > 0:
        return "I₄-type (y-deriv)"

    # I₁-type: mixed derivatives (a>0 and b>0)
    if a > 0 and b > 0:
        return "I₁-type (mixed)"

    return "Unknown"


def print_expansion_summary(ell: int, ellbar: int) -> None:
    """Print a detailed summary of the Ψ expansion for a pair."""
    monomials = expand_psi(ell, ellbar)

    print(f"\n{'='*70}")
    print(f"Ψ_{{{ell},{ellbar}}} Expansion: {len(monomials)} monomials")
    print(f"{'='*70}")

    # Group by derivative order (l₁, m₁)
    by_deriv_order: Dict[Tuple[int, int], List[MonomialTwoC]] = defaultdict(list)
    for m in monomials:
        by_deriv_order[(m.l1, m.m1)].append(m)

    for (l1, m1), monos in sorted(by_deriv_order.items()):
        print(f"\n(l₁={l1}, m₁={m1}): {len(monos)} monomials")
        for m in monos:
            iterm_type = classify_monomial(m)
            print(f"  {m} | weight=(1-u)^{m.weight_exponent} | {iterm_type}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CANONICAL PSI EXPANSION MODULE - Validation")
    print("=" * 70)

    # Validate all pairs
    print("\nValidating all K=3 pairs...")
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    print(f"\n{'Pair':<10} {'Expected':<10} {'Actual':<10} {'Status'}")
    print("-" * 45)

    all_pass = True
    for ell, ellbar in pairs:
        expected = EXPECTED_MONOMIAL_COUNTS.get((ell, ellbar), "?")
        try:
            monomials = expand_psi(ell, ellbar)
            actual = len(monomials)
            status = "✓ PASS" if actual == expected else "✗ FAIL"
        except AssertionError as e:
            actual = "ERR"
            status = "✗ FAIL"
            all_pass = False

        print(f"({ell},{ellbar})      {expected:<10} {actual:<10} {status}")

    print()
    if all_pass:
        print("All HARD GATES passed!")
    else:
        print("HARD GATE FAILURES detected!")

    # Detailed (1,1) validation
    print("\n" + "-" * 70)
    print("(1,1) Detailed Check:")
    validate_expansion(1, 1, verbose=True)

    # Show (2,2) structure
    print_expansion_summary(2, 2)
