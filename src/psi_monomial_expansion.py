"""
src/psi_monomial_expansion.py
Ψ Monomial Expansion - Convert p-configs to (a,b,c,d) monomial vectors

GPT's key insight: X, Y, Z are NOT scalar weights on the integrand.
They are COMBINATORIAL BRANCHING OPERATORS that produce monomial vectors:
  (a, b, c, d) = (#A, #B, #C, #D)

Where the raw blocks are zeta-function values:
  A = ζ'/ζ(1+α+s)
  B = ζ'/ζ(1+β+u)
  C = ζ'/ζ(1+s+u)
  D = ζ''/ζ(1+s+u) = (ζ'/ζ)'(1+s+u)

The connected blocks:
  X = (A - C)  →  "take A" or "take -C"
  Y = (B - C)  →  "take B" or "take -C"
  Z = (D - C²) →  "take D" or "take -C²"

For each p-config term: coeff × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}
Expand using binomial theorem to get monomials in (A, B, C, D).
"""

from __future__ import annotations
from collections import defaultdict
from math import comb
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass

from src.psi_block_configs import psi_p_configs, BlockConfig


@dataclass
class Monomial:
    """
    A monomial A^a × B^b × C^c × D^d with coefficient.

    These are the atomic units that the Section 7 machinery evaluates.
    """
    a: int      # Power of A = ζ'/ζ(1+α+s)
    b: int      # Power of B = ζ'/ζ(1+β+u)
    c: int      # Power of C = ζ'/ζ(1+s+u)
    d: int      # Power of D = (ζ'/ζ)'(1+s+u)
    coeff: int  # Integer coefficient (can be negative)

    def key(self) -> Tuple[int, int, int, int]:
        return (self.a, self.b, self.c, self.d)

    def __repr__(self) -> str:
        parts = []
        if self.coeff != 1:
            parts.append(f"{self.coeff:+d}")
        else:
            parts.append("+1")
        if self.a > 0:
            parts.append(f"A^{self.a}" if self.a > 1 else "A")
        if self.b > 0:
            parts.append(f"B^{self.b}" if self.b > 1 else "B")
        if self.c > 0:
            parts.append(f"C^{self.c}" if self.c > 1 else "C")
        if self.d > 0:
            parts.append(f"D^{self.d}" if self.d > 1 else "D")
        return f"Monomial({' × '.join(parts)})"


def expand_p_config_to_monomials(cfg: BlockConfig) -> List[Monomial]:
    """
    Expand a single p-config into monomials.

    Given: coeff × Z^{z_exp} × X^{x_exp} × Y^{y_exp}

    Expand:
      X^{x_exp} = (A-C)^{x_exp} = Σ_{i=0}^{x_exp} C(x_exp,i) A^i (-C)^{x_exp-i}
      Y^{y_exp} = (B-C)^{y_exp} = Σ_{j=0}^{y_exp} C(y_exp,j) B^j (-C)^{y_exp-j}
      Z^{z_exp} = (D-C²)^{z_exp} = Σ_{r=0}^{z_exp} C(z_exp,r) D^r (-C²)^{z_exp-r}

    Combined:
      a = i (power of A)
      b = j (power of B)
      d = r (power of D)
      c = (x_exp - i) + (y_exp - j) + 2*(z_exp - r) (power of C)
    """
    monomials = []

    x_exp = cfg.x_exp  # ℓ - p
    y_exp = cfg.y_exp  # ℓ̄ - p
    z_exp = cfg.z_exp  # p

    # Expand X^{x_exp} = (A - C)^{x_exp}
    for i in range(x_exp + 1):
        coeff_x = comb(x_exp, i) * ((-1) ** (x_exp - i))
        # Contributes A^i and (-C)^{x_exp-i} = (-1)^{x_exp-i} C^{x_exp-i}

        # Expand Y^{y_exp} = (B - C)^{y_exp}
        for j in range(y_exp + 1):
            coeff_y = comb(y_exp, j) * ((-1) ** (y_exp - j))

            # Expand Z^{z_exp} = (D - C²)^{z_exp}
            for r in range(z_exp + 1):
                coeff_z = comb(z_exp, r) * ((-1) ** (z_exp - r))

                # Exponents
                a = i
                b = j
                d = r
                c = (x_exp - i) + (y_exp - j) + 2 * (z_exp - r)

                # Total coefficient
                total_coeff = cfg.coeff * coeff_x * coeff_y * coeff_z

                if total_coeff != 0:
                    monomials.append(Monomial(a=a, b=b, c=c, d=d, coeff=total_coeff))

    return monomials


def expand_pair_to_monomials(ell: int, ellbar: int) -> Dict[Tuple[int,int,int,int], int]:
    """
    Expand all p-configs for pair (ℓ, ℓ̄) into combined monomials.

    Returns dict: (a, b, c, d) -> total coefficient
    """
    configs = psi_p_configs(ell, ellbar)

    combined = defaultdict(int)
    for cfg in configs:
        monomials = expand_p_config_to_monomials(cfg)
        for mono in monomials:
            combined[mono.key()] += mono.coeff

    # Remove zeros
    return {k: v for k, v in combined.items() if v != 0}


def print_pair_monomials(ell: int, ellbar: int) -> None:
    """Print all monomials for a (ℓ, ℓ̄) pair."""
    monomials = expand_pair_to_monomials(ell, ellbar)

    print(f"\nΨ_({ell},{ellbar}) = {len(monomials)} monomials:")
    print("-" * 50)

    # Sort by (a, b, c, d)
    for (a, b, c, d), coeff in sorted(monomials.items()):
        parts = []
        if a > 0:
            parts.append(f"A^{a}" if a > 1 else "A")
        if b > 0:
            parts.append(f"B^{b}" if b > 1 else "B")
        if c > 0:
            parts.append(f"C^{c}" if c > 1 else "C")
        if d > 0:
            parts.append(f"D^{d}" if d > 1 else "D")
        mono_str = " × ".join(parts) if parts else "1"
        print(f"  {coeff:+3d} × {mono_str:<20}  (a={a}, b={b}, c={c}, d={d})")


def verify_11_structure() -> None:
    """
    Verify that (1,1) expands to exactly: AB - AC - BC + D

    Expected monomials:
      +1 × A × B     (a=1, b=1, c=0, d=0)  → I₁ structure
      -1 × A × C     (a=1, b=0, c=1, d=0)  → I₃ structure
      -1 × B × C     (a=0, b=1, c=1, d=0)  → I₄ structure
      +1 × D         (a=0, b=0, c=0, d=1)  → I₂ structure
    """
    print("=" * 60)
    print("VERIFICATION: (1,1) → AB - AC - BC + D")
    print("=" * 60)

    monomials = expand_pair_to_monomials(1, 1)

    expected = {
        (1, 1, 0, 0): +1,  # AB
        (1, 0, 1, 0): -1,  # -AC
        (0, 1, 1, 0): -1,  # -BC
        (0, 0, 0, 1): +1,  # D
    }

    print(f"\nExpected: {len(expected)} monomials")
    print(f"Got:      {len(monomials)} monomials")

    # Check each expected monomial
    all_match = True
    for key, exp_coeff in expected.items():
        got_coeff = monomials.get(key, 0)
        status = "✓" if got_coeff == exp_coeff else "✗"
        a, b, c, d = key
        print(f"  A^{a}B^{b}C^{c}D^{d}: expected {exp_coeff:+d}, got {got_coeff:+d} {status}")
        if got_coeff != exp_coeff:
            all_match = False

    # Check for unexpected monomials
    for key in monomials:
        if key not in expected:
            print(f"  UNEXPECTED: {key} with coeff {monomials[key]}")
            all_match = False

    print(f"\n{'PASSED' if all_match else 'FAILED'}")

    # Show mapping to I-terms
    if all_match:
        print("\nMapping to I-terms:")
        print("  +AB → I₁ (mixed derivative ∂z∂w)")
        print("  +D  → I₂ (no derivative, base term)")
        print("  -AC → I₃ (z-derivative with C subtraction)")
        print("  -BC → I₄ (w-derivative with C subtraction)")


def verify_22_structure() -> None:
    """
    Verify (2,2) produces 12 monomials.

    Expected from p-config expansion:
      p=0: 1 × X² × Y² = (A-C)²(B-C)² → 9 terms
      p=1: 4 × Z × X × Y = 4(D-C²)(A-C)(B-C) → 8 terms
      p=2: 2 × Z² = 2(D-C²)² → 3 terms
    But many combine, giving 12 total.
    """
    print("\n" + "=" * 60)
    print("VERIFICATION: (2,2) → 12 monomials")
    print("=" * 60)

    monomials = expand_pair_to_monomials(2, 2)

    print(f"\nGot {len(monomials)} monomials (expected 12)")

    # Print all monomials
    print_pair_monomials(2, 2)

    # Verify count
    if len(monomials) == 12:
        print("\n✓ Count matches!")
    else:
        print(f"\n✗ Count mismatch: got {len(monomials)}, expected 12")


def print_all_k3_monomials() -> None:
    """Print monomial expansions for all K=3 pairs."""
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    print("=" * 70)
    print("ALL K=3 MONOMIALS")
    print("=" * 70)

    total_monomials = 0
    for ell, ellbar in pairs:
        monomials = expand_pair_to_monomials(ell, ellbar)
        print(f"\n({ell},{ellbar}): {len(monomials)} monomials")
        total_monomials += len(monomials)

        # Show first few
        for i, ((a, b, c, d), coeff) in enumerate(sorted(monomials.items())):
            if i >= 5:
                print(f"  ... and {len(monomials) - 5} more")
                break
            parts = []
            if a > 0: parts.append(f"A^{a}" if a > 1 else "A")
            if b > 0: parts.append(f"B^{b}" if b > 1 else "B")
            if c > 0: parts.append(f"C^{c}" if c > 1 else "C")
            if d > 0: parts.append(f"D^{d}" if d > 1 else "D")
            print(f"  {coeff:+3d} × {' '.join(parts) or '1'}")

    print(f"\n{'=' * 70}")
    print(f"Total monomials for K=3: {total_monomials}")
    print(f"(This should be 78 = 4 + 12 + 27 + 7 + 10 + 18)")


if __name__ == "__main__":
    verify_11_structure()
    verify_22_structure()
    print("\n")
    print_all_k3_monomials()
