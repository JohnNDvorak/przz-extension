"""
src/psi_separated_c.py
Psi expansion with SEPARATED C_alpha and C_beta.

This is the CORRECTED version per GPT's diagnosis.

CURRENT (WRONG):
    X = A - C
    Y = B - C
    Z = D - C^2

CORRECTED (THIS FILE):
    X = A - C_beta    (alpha-side block minus beta-pole contribution)
    Y = B - C_alpha   (beta-side block minus alpha-pole contribution)
    Z = D - C_alpha * C_beta   (mixed block minus pole product)

Where:
    A = zeta'/zeta at (1+s+u) with z-derivative
    B = zeta'/zeta at (1+s+u) with w-derivative
    C_alpha = contribution from 1/zeta(1+alpha+s) pole
    C_beta = contribution from 1/zeta(1+beta+u) pole
    D = (zeta'/zeta)' at (1+s+u) mixed z,w derivative

The Psi formula remains:
    Psi_{ell,ellbar} = sum_{p=0}^{min(ell,ellbar)} C(ell,p)C(ellbar,p)p! * Z^p * X^{ell-p} * Y^{ellbar-p}

KEY INSIGHT: For the (1,1) case:
    Psi_{1,1} = X * Y + Z
             = (A - C_beta)(B - C_alpha) + (D - C_alpha * C_beta)
             = AB - A*C_alpha - B*C_beta + C_alpha*C_beta + D - C_alpha*C_beta
             = AB - A*C_alpha - B*C_beta + D

The C_alpha*C_beta terms CANCEL, giving exactly 4 terms matching I1-I4.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import comb, factorial
from typing import List, Dict, Tuple
from collections import defaultdict


@dataclass
class MonomialSeparatedC:
    """
    A monomial A^a * B^b * C_alpha^c_alpha * C_beta^c_beta * D^d with coefficient.

    Key difference from old Monomial: separate c_alpha and c_beta powers
    instead of single c power.

    The source_p field tracks which p-config this monomial came from,
    which determines the (1-u)^{ℓ+ℓ̄-2p} Euler-Maclaurin weight.
    """
    a: int          # Power of A (z-derivative at s+u)
    b: int          # Power of B (w-derivative at s+u)
    c_alpha: int    # Power of C_alpha (from 1+alpha+s denominator)
    c_beta: int     # Power of C_beta (from 1+beta+u denominator)
    d: int          # Power of D (mixed derivative at s+u)
    coeff: int      # Integer coefficient (can be negative)
    source_p: int = 0  # Which p-config this came from (for (1-u) weight)

    def key(self) -> Tuple[int, int, int, int, int]:
        """Unique key for combining like terms."""
        return (self.a, self.b, self.c_alpha, self.c_beta, self.d)

    def total_derivative_order(self) -> Tuple[int, int]:
        """Return (z-derivative-order, w-derivative-order)."""
        # A contributes to z-side, B to w-side, D to both
        z_order = self.a + self.d
        w_order = self.b + self.d
        return (z_order, w_order)

    def __repr__(self) -> str:
        parts = []
        if self.coeff != 1:
            parts.append(str(self.coeff))
        if self.a > 0:
            parts.append(f"A^{self.a}" if self.a > 1 else "A")
        if self.b > 0:
            parts.append(f"B^{self.b}" if self.b > 1 else "B")
        if self.c_alpha > 0:
            parts.append(f"C_a^{self.c_alpha}" if self.c_alpha > 1 else "C_a")
        if self.c_beta > 0:
            parts.append(f"C_b^{self.c_beta}" if self.c_beta > 1 else "C_b")
        if self.d > 0:
            parts.append(f"D^{self.d}" if self.d > 1 else "D")
        return " * ".join(parts) if parts else "1"


@dataclass
class BlockConfigSeparatedC:
    """
    A p-configuration using separated C_alpha and C_beta.

    Represents: coeff * Z^p * X^{ell-p} * Y^{ellbar-p}

    where:
        X = (A - C_beta)     # z-block minus beta-pole
        Y = (B - C_alpha)    # w-block minus alpha-pole
        Z = (D - C_alpha*C_beta)  # mixed block minus pole product
    """
    ell: int
    ellbar: int
    p: int
    coeff: int      # C(ell,p) * C(ellbar,p) * p!
    x_exp: int      # ell - p
    y_exp: int      # ellbar - p
    z_exp: int      # p

    def __repr__(self) -> str:
        parts = [f"{self.coeff}"]
        if self.z_exp > 0:
            parts.append(f"Z^{self.z_exp}" if self.z_exp > 1 else "Z")
        if self.x_exp > 0:
            parts.append(f"X^{self.x_exp}" if self.x_exp > 1 else "X")
        if self.y_exp > 0:
            parts.append(f"Y^{self.y_exp}" if self.y_exp > 1 else "Y")
        return f"BlockConfigSeparatedC(({self.ell},{self.ellbar}), p={self.p}: {' * '.join(parts)})"


def psi_p_configs_separated(ell: int, ellbar: int) -> List[BlockConfigSeparatedC]:
    """
    Generate p-sum configurations for Psi_{ell,ellbar} with separated C's.

    Same structure as psi_p_configs() but uses BlockConfigSeparatedC
    to indicate the TWO-C interpretation.

    Args:
        ell: Left piece index (1, 2, or 3 for K=3)
        ellbar: Right piece index (1, 2, or 3 for K=3)

    Returns:
        List of BlockConfigSeparatedC, one per p-value
    """
    configs = []

    for p in range(0, min(ell, ellbar) + 1):
        coeff = comb(ell, p) * comb(ellbar, p) * factorial(p)

        config = BlockConfigSeparatedC(
            ell=ell,
            ellbar=ellbar,
            p=p,
            coeff=coeff,
            x_exp=ell - p,      # singleton z-blocks
            y_exp=ellbar - p,   # singleton w-blocks
            z_exp=p             # paired blocks
        )
        configs.append(config)

    return configs


def expand_p_config_to_monomials_separated(cfg: BlockConfigSeparatedC) -> List[MonomialSeparatedC]:
    """
    Expand a single p-config into monomials with separated C_alpha, C_beta.

    Given: coeff * Z^p * X^{x_exp} * Y^{y_exp}

    Expand using:
      X^{x_exp} = (A - C_beta)^{x_exp}
                = sum_{i=0}^{x_exp} C(x_exp,i) * A^i * (-C_beta)^{x_exp-i}

      Y^{y_exp} = (B - C_alpha)^{y_exp}
                = sum_{j=0}^{y_exp} C(y_exp,j) * B^j * (-C_alpha)^{y_exp-j}

      Z^{z_exp} = (D - C_alpha*C_beta)^{z_exp}
                = sum_{r=0}^{z_exp} C(z_exp,r) * D^r * (-C_alpha*C_beta)^{z_exp-r}

    Combined monomial exponents:
      a = i (power of A)
      b = j (power of B)
      c_alpha = (y_exp - j) + (z_exp - r)  # from Y and Z
      c_beta = (x_exp - i) + (z_exp - r)   # from X and Z
      d = r (power of D)
    """
    monomials = []
    x_exp, y_exp, z_exp = cfg.x_exp, cfg.y_exp, cfg.z_exp

    # Expand X^{x_exp} = (A - C_beta)^{x_exp}
    for i in range(x_exp + 1):
        coeff_x = comb(x_exp, i) * ((-1) ** (x_exp - i))
        c_beta_from_x = x_exp - i  # Power of C_beta from X term

        # Expand Y^{y_exp} = (B - C_alpha)^{y_exp}
        for j in range(y_exp + 1):
            coeff_y = comb(y_exp, j) * ((-1) ** (y_exp - j))
            c_alpha_from_y = y_exp - j  # Power of C_alpha from Y term

            # Expand Z^{z_exp} = (D - C_alpha*C_beta)^{z_exp}
            for r in range(z_exp + 1):
                coeff_z = comb(z_exp, r) * ((-1) ** (z_exp - r))

                # Exponents
                a = i
                b = j
                d = r
                c_alpha = c_alpha_from_y + (z_exp - r)  # from Y and Z
                c_beta = c_beta_from_x + (z_exp - r)   # from X and Z

                # Total coefficient
                total_coeff = cfg.coeff * coeff_x * coeff_y * coeff_z

                if total_coeff != 0:
                    monomials.append(MonomialSeparatedC(
                        a=a, b=b, c_alpha=c_alpha, c_beta=c_beta, d=d,
                        coeff=total_coeff, source_p=cfg.p
                    ))

    return monomials


def expand_pair_to_monomials_separated(ell: int, ellbar: int,
                                       combine_across_p: bool = True) -> List[MonomialSeparatedC]:
    """
    Expand Psi_{ell,ellbar} to full monomial list with separated C's.

    Combines like terms within the same p-config, and optionally across p-configs.

    Args:
        ell: Left piece index
        ellbar: Right piece index
        combine_across_p: If True (default), combine like terms across different p-values.
                         This is correct because the (1-u)^{a+b} weight depends on
                         the monomial structure (a, b), not the p-config.
                         If False, keep monomials from different p-configs separate
                         (only useful for debugging the expansion structure).

    Returns:
        List of MonomialSeparatedC with combined coefficients
    """
    configs = psi_p_configs_separated(ell, ellbar)

    # Collect all monomials, combining like terms
    # Key includes source_p to keep different p-configs separate
    if combine_across_p:
        # Old behavior: combine by (a, b, c_alpha, c_beta, d) only
        monomial_dict: Dict[Tuple[int, int, int, int, int], int] = defaultdict(int)
        for cfg in configs:
            monomials = expand_p_config_to_monomials_separated(cfg)
            for mono in monomials:
                monomial_dict[mono.key()] += mono.coeff

        # Convert to list, excluding zero coefficients
        result = []
        for (a, b, c_alpha, c_beta, d), coeff in monomial_dict.items():
            if coeff != 0:
                result.append(MonomialSeparatedC(
                    a=a, b=b, c_alpha=c_alpha, c_beta=c_beta, d=d, coeff=coeff, source_p=0
                ))
    else:
        # New behavior: keep different p-configs separate for correct (1-u) weights
        # Key is (a, b, c_alpha, c_beta, d, source_p)
        monomial_dict: Dict[Tuple[int, int, int, int, int, int], int] = defaultdict(int)
        for cfg in configs:
            monomials = expand_p_config_to_monomials_separated(cfg)
            for mono in monomials:
                key_with_p = (mono.a, mono.b, mono.c_alpha, mono.c_beta, mono.d, mono.source_p)
                monomial_dict[key_with_p] += mono.coeff

        # Convert to list, excluding zero coefficients
        result = []
        for (a, b, c_alpha, c_beta, d, source_p), coeff in monomial_dict.items():
            if coeff != 0:
                result.append(MonomialSeparatedC(
                    a=a, b=b, c_alpha=c_alpha, c_beta=c_beta, d=d, coeff=coeff, source_p=source_p
                ))

    # Sort for consistent ordering (by key, then by source_p)
    result.sort(key=lambda m: (m.key(), m.source_p))

    return result


def compute_euler_maclaurin_weight(ell: int, ellbar: int, p: int) -> int:
    """
    Compute the (1-u) exponent for a p-configuration.

    From PRZZ Lemma 7.2:
        Weight = (1-u)^{ell + ellbar - 2p}

    - Singleton blocks (from X and Y): each contributes (1-u)
    - Paired blocks (from Z): contribute NO (1-u)

    Args:
        ell: Left piece index
        ellbar: Right piece index
        p: Pairing count (number of Z blocks)

    Returns:
        Exponent for (1-u) weight
    """
    return ell + ellbar - 2 * p


def verify_11_expansion() -> bool:
    """
    Verify that (1,1) expansion gives exactly 4 terms after combining.

    Expected:
        Psi_{1,1} = AB - A*C_alpha - B*C_beta + D

    The C_alpha*C_beta terms should cancel when combining across p-configs.
    """
    # Use combine_across_p=True to verify the algebraic cancellation
    monomials = expand_pair_to_monomials_separated(1, 1, combine_across_p=True)

    # Expected monomials (sorted by key)
    expected = {
        (1, 1, 0, 0, 0): 1,   # AB (coeff +1)
        (1, 0, 1, 0, 0): -1,  # -A*C_alpha (coeff -1)
        (0, 1, 0, 1, 0): -1,  # -B*C_beta (coeff -1)
        (0, 0, 0, 0, 1): 1,   # D (coeff +1)
    }

    actual = {m.key(): m.coeff for m in monomials}

    return actual == expected


def verify_11_expansion_with_weights() -> bool:
    """
    Verify that (1,1) expansion with source_p tracking gives correct structure.

    When NOT combining across p-configs, we should have 6 monomials:
    - From p=0: AB, -AC_α, -BC_β, +C_α×C_β
    - From p=1: D, -C_α×C_β

    The C_α×C_β terms have opposite signs and cancel in the sum, but they
    have different (1-u) weights so we keep them separate for evaluation.
    """
    monomials = expand_pair_to_monomials_separated(1, 1, combine_across_p=False)

    # Should have 6 monomials with source_p tracking
    if len(monomials) != 6:
        return False

    # Check the weight structure
    p0_count = sum(1 for m in monomials if m.source_p == 0)
    p1_count = sum(1 for m in monomials if m.source_p == 1)

    return p0_count == 4 and p1_count == 2


def print_pair_expansion(ell: int, ellbar: int) -> None:
    """Print detailed expansion for a pair."""
    print(f"\n{'='*60}")
    print(f"Psi_({ell},{ellbar}) Expansion with Separated C's")
    print(f"{'='*60}")

    # p-configs
    configs = psi_p_configs_separated(ell, ellbar)
    print(f"\nStep 1: p-configurations ({len(configs)} configs)")
    for cfg in configs:
        weight = compute_euler_maclaurin_weight(ell, ellbar, cfg.p)
        print(f"  {cfg}  --> (1-u)^{weight} weight")

    # Monomials
    monomials = expand_pair_to_monomials_separated(ell, ellbar)
    print(f"\nStep 2: Monomial expansion ({len(monomials)} monomials)")
    for mono in monomials:
        print(f"  {mono}")

    # Group by I-term structure
    print(f"\nStep 3: I-term mapping")
    for mono in monomials:
        z_ord, w_ord = mono.total_derivative_order()
        if z_ord == 1 and w_ord == 1 and mono.c_alpha == 0 and mono.c_beta == 0:
            print(f"  {mono} --> I1-like (mixed deriv, no C)")
        elif z_ord == 1 and w_ord == 1 and mono.d > 0:
            print(f"  {mono} --> I2-like (base integral)")
        elif mono.c_alpha > 0 and mono.c_beta == 0:
            print(f"  {mono} --> I3-like (alpha-pole)")
        elif mono.c_beta > 0 and mono.c_alpha == 0:
            print(f"  {mono} --> I4-like (beta-pole)")
        else:
            print(f"  {mono} --> Mixed C_alpha/C_beta term")


def print_summary() -> None:
    """Print summary for all K=3 pairs."""
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    print("=" * 70)
    print("PSI EXPANSION SUMMARY (Separated C_alpha and C_beta)")
    print("=" * 70)
    print()
    print("X = (A - C_beta), Y = (B - C_alpha), Z = (D - C_alpha*C_beta)")
    print()

    print(f"{'Pair':<10} {'p-configs':<12} {'Monomials':<12} {'(1-u) exponents'}")
    print("-" * 60)

    for (ell, ellbar) in pairs:
        configs = psi_p_configs_separated(ell, ellbar)
        monomials = expand_pair_to_monomials_separated(ell, ellbar)
        weights = [compute_euler_maclaurin_weight(ell, ellbar, c.p) for c in configs]

        print(f"({ell},{ellbar})      {len(configs):<12} {len(monomials):<12} {weights}")

    print()

    # Verify (1,1) - algebraic cancellation
    if verify_11_expansion():
        print("(1,1) algebraic verification: PASSED (4 terms, C_alpha*C_beta cancelled)")
    else:
        print("(1,1) algebraic verification: FAILED!")

    # Verify (1,1) with source_p tracking
    if verify_11_expansion_with_weights():
        print("(1,1) weight structure verification: PASSED (6 monomials with correct p-tracking)")
    else:
        print("(1,1) weight structure verification: FAILED!")


if __name__ == "__main__":
    print_summary()
    print()
    print_pair_expansion(1, 1)
    print()
    print_pair_expansion(2, 2)
