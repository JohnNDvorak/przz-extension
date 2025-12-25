"""
src/psi_monomial_evaluator.py
Monomial Evaluator for Ψ-Based Main Term Computation

This evaluates monomials A^a × B^b × C^c × D^d where:
  A = ζ'/ζ(1+α+s) → z-derivative structure
  B = ζ'/ζ(1+β+u) → w-derivative structure
  C = ζ'/ζ(1+s+u) → base value structure
  D = (ζ'/ζ)'(1+s+u) → mixed derivative structure

For (1,1), the mapping to I-terms is:
  AB → I₁ (mixed derivative)
  D  → I₂ (base integral)
  AC → |I₃| (z-derivative base value)
  BC → |I₄| (w-derivative base value)

The Ψ coefficients (+1, +1, -1, -1) handle the signs.

For higher pairs, the monomial evaluator must generalize to handle
A², B², D², and mixed products, using PRZZ Section 7 machinery.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Dict, Tuple
from math import exp
from dataclasses import dataclass

from src.psi_monomial_expansion import expand_pair_to_monomials


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


@dataclass
class MonomialEvalResult:
    """Result of evaluating a monomial."""
    a: int
    b: int
    c: int
    d: int
    value: float
    description: str


def eval_monomial_11_basis(
    a: int, b: int, c: int, d: int,
    P, Q, theta: float, R: float,
    n_quad: int = 60
) -> float:
    """
    Evaluate a monomial for the (1,1) case using the I-term basis.

    This is a reference implementation that maps monomials to
    existing I₁, I₂, I₃, I₄ oracle computations.

    For (1,1), the only valid monomials are:
      (1,1,0,0): AB → I₁
      (0,0,0,1): D  → I₂
      (1,0,1,0): AC → base value for I₃
      (0,1,1,0): BC → base value for I₄

    Returns the base (unsigned) value; sign comes from Ψ coefficient.
    """
    # Import oracle for comparison
    from src.przz_22_exact_oracle import przz_oracle_22

    # Use oracle to get I-term values
    # Note: oracle is named for (2,2) but uses P for both sides,
    # so for (1,1) we pass P1
    oracle = przz_oracle_22(P, Q, theta, R, n_quad)

    # Map monomial to I-term
    if (a, b, c, d) == (1, 1, 0, 0):  # AB
        return oracle.I1  # Already positive
    elif (a, b, c, d) == (0, 0, 0, 1):  # D
        return oracle.I2  # Already positive
    elif (a, b, c, d) == (1, 0, 1, 0):  # AC
        # I₃ is negative, but this is the base value
        return abs(oracle.I3)
    elif (a, b, c, d) == (0, 1, 1, 0):  # BC
        # I₄ is negative, but this is the base value
        return abs(oracle.I4)
    else:
        raise ValueError(f"Monomial ({a},{b},{c},{d}) not valid for (1,1)")


def validate_11_monomial_sum(P1, Q, theta: float, R: float, n_quad: int = 60) -> bool:
    """
    Validate that Ψ_{1,1} monomial sum equals oracle total.

    Ψ_{1,1} = +1×AB + 1×D - 1×AC - 1×BC
            = I₁ + I₂ + I₃ + I₄  (using I₃, I₄ already negative)
    """
    from src.przz_22_exact_oracle import przz_oracle_22

    print("=" * 60)
    print("VALIDATION: Ψ_{1,1} Monomial Sum vs Oracle")
    print("=" * 60)

    # Get oracle values
    oracle = przz_oracle_22(P1, Q, theta, R, n_quad)
    print(f"\nOracle I-terms:")
    print(f"  I₁ = {oracle.I1:+.6f}")
    print(f"  I₂ = {oracle.I2:+.6f}")
    print(f"  I₃ = {oracle.I3:+.6f}")
    print(f"  I₄ = {oracle.I4:+.6f}")
    print(f"  Total = {oracle.total:.6f}")

    # Get Ψ monomials
    monomials = expand_pair_to_monomials(1, 1)
    print(f"\nΨ monomials (4 total):")

    psi_sum = 0.0
    for (a, b, c, d), coeff in sorted(monomials.items()):
        # Get base monomial value
        base_val = eval_monomial_11_basis(a, b, c, d, P1, Q, theta, R, n_quad)
        contribution = coeff * base_val
        psi_sum += contribution

        mono_str = f"A^{a}B^{b}C^{c}D^{d}"
        print(f"  {coeff:+d} × {mono_str:<12} = {coeff:+d} × {base_val:.4f} = {contribution:+.4f}")

    print(f"\nΨ sum = {psi_sum:.6f}")
    print(f"Oracle total = {oracle.total:.6f}")
    print(f"Difference = {abs(psi_sum - oracle.total):.2e}")

    match = abs(psi_sum - oracle.total) < 1e-6
    print(f"\n{'✓ MATCH' if match else '✗ MISMATCH'}")

    return match


def analyze_monomial_structure() -> None:
    """
    Analyze the monomial structure for all K=3 pairs.

    This helps understand what monomial types appear and
    what evaluators are needed.
    """
    print("=" * 70)
    print("MONOMIAL STRUCTURE ANALYSIS FOR K=3")
    print("=" * 70)

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    # Collect all unique monomial structures
    all_structures = set()

    for ell, ellbar in pairs:
        monomials = expand_pair_to_monomials(ell, ellbar)
        print(f"\n({ell},{ellbar}): {len(monomials)} monomials")

        # Analyze structure
        max_a = max(a for (a, b, c, d) in monomials.keys())
        max_b = max(b for (a, b, c, d) in monomials.keys())
        max_c = max(c for (a, b, c, d) in monomials.keys())
        max_d = max(d for (a, b, c, d) in monomials.keys())

        print(f"  Max powers: A^{max_a}, B^{max_b}, C^{max_c}, D^{max_d}")

        for key in monomials.keys():
            all_structures.add(key)

    print("\n" + "-" * 70)
    print(f"Total unique monomial structures: {len(all_structures)}")
    print()

    # Group by (a+d, b+d) to understand derivative order
    print("Grouped by derivative order (a+d = z-derivs, b+d = w-derivs):")
    by_deriv_order = {}
    for (a, b, c, d) in sorted(all_structures):
        z_order = a + d
        w_order = b + d
        key = (z_order, w_order)
        if key not in by_deriv_order:
            by_deriv_order[key] = []
        by_deriv_order[key].append((a, b, c, d))

    for (z_ord, w_ord), monos in sorted(by_deriv_order.items()):
        print(f"  ∂^{z_ord}_z ∂^{w_ord}_w: {len(monos)} monomials")
        for (a, b, c, d) in monos[:3]:
            print(f"    A^{a}B^{b}C^{c}D^{d}")
        if len(monos) > 3:
            print(f"    ... and {len(monos)-3} more")


if __name__ == "__main__":
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036

    # Validate (1,1) mapping
    validate_11_monomial_sum(P1, Q, theta, R)

    # Analyze structure
    print("\n")
    analyze_monomial_structure()
