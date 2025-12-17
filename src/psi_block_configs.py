"""
src/psi_block_configs.py
Ψ Block Configuration Generator (p-sum representation)

This implements GPT's recommended approach: work with factorized blocks
(A-C), (B-C), (D-C²) instead of expanding to C^k monomials.

The Ψ formula in block form:
    Ψ_{ℓ,ℓ̄} = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}

Where:
    X = (A - C) = "connected singleton z-block"
    Y = (B - C) = "connected singleton w-block"
    Z = (D - C²) = "connected paired block"

Key insight: We never need to ask "what is C?" in isolation.
The C factors only appear through the connected combinations.

For K=3, this gives only 16 p-config terms instead of 78 monomials:
    (1,1): 2 p-values (p=0,1)
    (2,2): 3 p-values (p=0,1,2)
    (3,3): 4 p-values (p=0,1,2,3)
    (1,2): 2 p-values (p=0,1)
    (1,3): 2 p-values (p=0,1)
    (2,3): 3 p-values (p=0,1,2)
"""

from __future__ import annotations
from math import comb, factorial
from typing import List, NamedTuple
from dataclasses import dataclass


@dataclass
class BlockConfig:
    """
    A single p-configuration for the Ψ block expansion.

    Represents: coeff × Z^z_exp × X^x_exp × Y^y_exp

    where:
        X = (A - C) = connected singleton z-block
        Y = (B - C) = connected singleton w-block
        Z = (D - C²) = connected paired block
    """
    ell: int         # Left piece index ℓ
    ellbar: int      # Right piece index ℓ̄
    p: int           # Partition parameter (number of paired blocks)
    coeff: int       # C(ℓ,p) × C(ℓ̄,p) × p!
    x_exp: int       # Power of X = (A-C), equals ℓ-p
    y_exp: int       # Power of Y = (B-C), equals ℓ̄-p
    z_exp: int       # Power of Z = (D-C²), equals p

    def __repr__(self) -> str:
        parts = [f"{self.coeff}"]
        if self.z_exp > 0:
            parts.append(f"Z^{self.z_exp}" if self.z_exp > 1 else "Z")
        if self.x_exp > 0:
            parts.append(f"X^{self.x_exp}" if self.x_exp > 1 else "X")
        if self.y_exp > 0:
            parts.append(f"Y^{self.y_exp}" if self.y_exp > 1 else "Y")
        return f"BlockConfig(({self.ell},{self.ellbar}), p={self.p}: {' × '.join(parts)})"


def psi_p_configs(ell: int, ellbar: int) -> List[BlockConfig]:
    """
    Generate p-sum configurations for the Ψ_{ℓ,ℓ̄} block expansion.

    Instead of returning 4/12/27 monomials, returns 2/3/4 p-configs:

    Ψ_{ℓ,ℓ̄} = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}

    Args:
        ell: Left piece index ℓ (1, 2, or 3 for K=3)
        ellbar: Right piece index ℓ̄ (1, 2, or 3 for K=3)

    Returns:
        List of BlockConfig, one per p-value from 0 to min(ℓ,ℓ̄)
    """
    configs = []

    for p in range(0, min(ell, ellbar) + 1):
        coeff = comb(ell, p) * comb(ellbar, p) * factorial(p)

        config = BlockConfig(
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


def count_monomials_from_p_configs(configs: List[BlockConfig]) -> int:
    """
    Count how many monomials result from expanding the p-configs.

    This verifies that the p-sum representation gives the same count
    as the direct monomial expansion.

    Each config Z^p × X^x × Y^y expands to:
        (D-C²)^p × (A-C)^x × (B-C)^y

    Using binomial expansion:
        (D-C²)^p has (p+1) terms
        (A-C)^x has (x+1) terms
        (B-C)^y has (y+1) terms

    But some terms combine, so this is an upper bound.

    For exact count, we need to track distinct (k1,k2,l1,m1) tuples.
    """
    from collections import defaultdict

    # Track all resulting monomials
    monomials = defaultdict(int)

    for cfg in configs:
        # Expand (D-C²)^p
        # (D-C²)^p = Σ_{j=0}^p C(p,j) D^j (-C²)^{p-j}
        #          = Σ_{j=0}^p C(p,j) (-1)^{p-j} D^j C^{2(p-j)}
        for j in range(cfg.z_exp + 1):
            d_pow = j
            c_from_z = 2 * (cfg.z_exp - j)
            coeff_z = comb(cfg.z_exp, j) * ((-1) ** (cfg.z_exp - j))

            # Expand (A-C)^x
            for a in range(cfg.x_exp + 1):
                a_pow = a
                c_from_x = cfg.x_exp - a
                coeff_x = comb(cfg.x_exp, a) * ((-1) ** (cfg.x_exp - a))

                # Expand (B-C)^y
                for b in range(cfg.y_exp + 1):
                    b_pow = b
                    c_from_y = cfg.y_exp - b
                    coeff_y = comb(cfg.y_exp, b) * ((-1) ** (cfg.y_exp - b))

                    # Total C power
                    c_pow = c_from_z + c_from_x + c_from_y

                    # Monomial key: (c_pow, d_pow, a_pow, b_pow)
                    key = (c_pow, d_pow, a_pow, b_pow)
                    monomials[key] += cfg.coeff * coeff_z * coeff_x * coeff_y

    # Count non-zero monomials
    return sum(1 for v in monomials.values() if v != 0)


def print_p_configs_summary() -> None:
    """Print summary of p-configs for all K=3 pairs."""
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    print("=" * 70)
    print("Ψ BLOCK CONFIGURATION SUMMARY (p-sum representation)")
    print("=" * 70)
    print()
    print("Formula: Ψ_{ℓ,ℓ̄} = Σ_p C(ℓ,p)C(ℓ̄,p)p! × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}")
    print("Where: X=(A-C), Y=(B-C), Z=(D-C²)")
    print()

    total_configs = 0
    total_monomials = 0

    for (ell, ellbar) in pairs:
        configs = psi_p_configs(ell, ellbar)
        n_monomials = count_monomials_from_p_configs(configs)

        print(f"({ell},{ellbar}): {len(configs)} p-configs → {n_monomials} monomials")
        for cfg in configs:
            print(f"    {cfg}")
        print()

        total_configs += len(configs)
        total_monomials += n_monomials

    print("-" * 70)
    print(f"Total for K=3: {total_configs} p-configs → {total_monomials} monomials")
    print()
    print("This is much simpler than the 78-monomial approach!")


def verify_monomial_counts() -> bool:
    """
    Verify that p-config expansion matches expected monomial counts.

    Expected from direct Ψ expansion:
        (1,1): 4 monomials
        (2,2): 12 monomials
        (3,3): 27 monomials
    """
    expected = {
        (1, 1): 4,
        (2, 2): 12,
        (3, 3): 27,
    }

    print("=" * 60)
    print("VERIFICATION: p-config expansion → monomial counts")
    print("=" * 60)

    all_pass = True

    for (ell, ellbar), expected_count in expected.items():
        configs = psi_p_configs(ell, ellbar)
        actual_count = count_monomials_from_p_configs(configs)

        status = "✓" if actual_count == expected_count else "✗"
        print(f"  ({ell},{ellbar}): {len(configs)} p-configs → {actual_count} monomials (expected {expected_count}) {status}")

        if actual_count != expected_count:
            all_pass = False

    print()
    return all_pass


def demonstrate_11_expansion() -> None:
    """
    Show how (1,1) p-configs expand to I₁-I₄ structure.

    Ψ_{1,1} = Z^1·X^0·Y^0 + Z^0·X^1·Y^1
            = (D-C²) + (A-C)(B-C)
            = D - C² + AB - AC - BC + C²
            = AB - AC - BC + D

    Which maps to:
        AB  → I₁ (mixed derivative)
        D   → I₂ (no derivatives)
        -AC → I₃ (z-derivative only)
        -BC → I₄ (w-derivative only)
    """
    print("=" * 60)
    print("(1,1) EXPANSION: p-configs → I₁-I₄")
    print("=" * 60)

    configs = psi_p_configs(1, 1)

    print("\nStep 1: p-configs")
    for cfg in configs:
        print(f"  {cfg}")

    print("\nStep 2: Symbolic expansion")
    print("  p=0: 1 × X¹ × Y¹ = (A-C)(B-C)")
    print("  p=1: 1 × Z¹     = (D-C²)")
    print()
    print("  Sum: (A-C)(B-C) + (D-C²)")
    print("     = AB - AC - BC + C² + D - C²")
    print("     = AB - AC - BC + D")

    print("\nStep 3: Map to I-terms")
    print("  +AB  → I₁ (mixed derivative ∂z∂w)")
    print("  +D   → I₂ (no derivatives, base integral)")
    print("  -AC  → I₃ (z-derivative only, with -C factor)")
    print("  -BC  → I₄ (w-derivative only, with -C factor)")

    print("\nConclusion: The 4-term I₁-I₄ structure emerges naturally")
    print("from the 2 p-configs when ℓ=ℓ̄=1.")
    print()


if __name__ == "__main__":
    # Verify monomial counts
    verify_monomial_counts()
    print()

    # Print full summary
    print_p_configs_summary()

    # Show (1,1) expansion detail
    demonstrate_11_expansion()
