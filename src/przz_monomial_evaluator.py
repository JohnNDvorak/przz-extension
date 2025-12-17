"""
src/przz_monomial_evaluator.py
PRZZ Monomial-Based Evaluator

Key insight: Each p-config (X^a Y^b Z^c) expands to MULTIPLE monomials.
Each monomial A^i B^j C^k D^m requires its own derivative extraction.

For example:
- XY = (A-C)(B-C) = AB - AC - BC + C²  (4 monomials)
- X²Y² = (A-C)²(B-C)² = 8 monomials
- XYZ = (A-C)(B-C)(D-C²) = ...

The monomial exponents (i, j, k, m) determine:
- i: x-derivative order (number of A blocks)
- j: y-derivative order (number of B blocks)
- k: C power (base block, no derivatives)
- m: D power (paired block, mixed xy structure)

The F_d case (A/B/C) depends on these orders:
- ω_left = i + m - 1 (total left derivatives)
- ω_right = j + m - 1 (total right derivatives)

The (1-u) weight exponent comes from the Euler-Maclaurin structure.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, List, Dict, NamedTuple
from math import exp, log, comb, factorial
from dataclasses import dataclass


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@dataclass
class Monomial:
    """
    A single monomial A^i B^j C^k D^m with coefficient.

    The exponents determine the derivative structure:
    - i: number of singleton x-blocks (A factors)
    - j: number of singleton y-blocks (B factors)
    - k: number of base blocks (C factors)
    - m: number of paired blocks (D factors)
    """
    i: int  # A power (x-derivative order)
    j: int  # B power (y-derivative order)
    k: int  # C power (base, no deriv)
    m: int  # D power (mixed xy)
    coeff: int  # coefficient (can be negative)

    def __str__(self):
        parts = []
        if self.i > 0:
            parts.append(f"A^{self.i}" if self.i > 1 else "A")
        if self.j > 0:
            parts.append(f"B^{self.j}" if self.j > 1 else "B")
        if self.k > 0:
            parts.append(f"C^{self.k}" if self.k > 1 else "C")
        if self.m > 0:
            parts.append(f"D^{self.m}" if self.m > 1 else "D")
        if not parts:
            return str(self.coeff)
        return f"{self.coeff:+d}×" + "".join(parts)

    @property
    def deriv_order_x(self) -> int:
        """Total x-derivative order: from A and D blocks."""
        return self.i + self.m

    @property
    def deriv_order_y(self) -> int:
        """Total y-derivative order: from B and D blocks."""
        return self.j + self.m


def expand_pconfig_to_monomials(ell: int, ellbar: int, p: int) -> List[Monomial]:
    """
    Expand a p-config to its constituent monomials.

    p-config: C(ℓ,p)C(ℓ̄,p)p! × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}

    where X = A-C, Y = B-C, Z = D-C².

    Returns list of Monomials with their coefficients.
    """
    x_power = ell - p
    y_power = ellbar - p
    z_power = p

    # Coefficient from Ψ combinatorics
    psi_coeff = comb(ell, p) * comb(ellbar, p) * factorial(p)

    # Expand X^{x_power} = (A-C)^{x_power}
    # = Σ_{a=0}^{x_power} C(x_power, a) × A^a × (-C)^{x_power-a}
    # = Σ_{a=0}^{x_power} C(x_power, a) × (-1)^{x_power-a} × A^a × C^{x_power-a}

    # Expand Y^{y_power} = (B-C)^{y_power}
    # = Σ_{b=0}^{y_power} C(y_power, b) × (-1)^{y_power-b} × B^b × C^{y_power-b}

    # Expand Z^{z_power} = (D-C²)^{z_power}
    # = Σ_{d=0}^{z_power} C(z_power, d) × (-1)^{z_power-d} × D^d × C^{2(z_power-d)}

    monomials = []

    for a in range(x_power + 1):
        x_coeff = comb(x_power, a) * ((-1) ** (x_power - a))
        c_from_x = x_power - a

        for b in range(y_power + 1):
            y_coeff = comb(y_power, b) * ((-1) ** (y_power - b))
            c_from_y = y_power - b

            for d in range(z_power + 1):
                z_coeff = comb(z_power, d) * ((-1) ** (z_power - d))
                c_from_z = 2 * (z_power - d)

                total_coeff = psi_coeff * x_coeff * y_coeff * z_coeff
                total_c = c_from_x + c_from_y + c_from_z

                if total_coeff != 0:
                    monomials.append(Monomial(
                        i=a,      # A power
                        j=b,      # B power
                        k=total_c,  # C power
                        m=d,      # D power
                        coeff=int(total_coeff)
                    ))

    return monomials


def expand_pair_to_monomials(ell: int, ellbar: int) -> List[Monomial]:
    """
    Expand all p-configs for pair (ℓ, ℓ̄) to monomials.

    Returns combined list of monomials (may have duplicates with same exponents).
    """
    all_monomials = []
    max_p = min(ell, ellbar)

    for p in range(max_p + 1):
        monomials = expand_pconfig_to_monomials(ell, ellbar, p)
        all_monomials.extend(monomials)

    return all_monomials


def combine_like_monomials(monomials: List[Monomial]) -> Dict[Tuple[int,int,int,int], int]:
    """
    Combine monomials with same (i,j,k,m) by summing coefficients.

    Returns dict mapping (i,j,k,m) -> total coefficient.
    """
    combined = {}
    for m in monomials:
        key = (m.i, m.j, m.k, m.m)
        combined[key] = combined.get(key, 0) + m.coeff
    return combined


class MonomialEvaluator:
    """
    Evaluates pair contribution by summing over all monomials.

    Each monomial A^i B^j C^k D^m is evaluated via:
    1. Derivative extraction based on (i, j, m)
    2. Base factor from C^k
    3. Euler-Maclaurin weight from derivative structure
    """

    def __init__(self, P_left, P_right, Q, theta: float, R: float, n_quad: int = 60):
        self.P_left = P_left
        self.P_right = P_right
        self.Q = Q
        self.theta = theta
        self.R = R
        self.n_quad = n_quad

        # Set up quadrature
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.t_nodes, self.t_weights = gauss_legendre_01(n_quad)

        # Precompute polynomial values
        self._precompute()

    def _precompute(self):
        """Precompute polynomial values and ratios at quadrature nodes."""
        # Left polynomial
        self.P_L = self.P_left.eval(self.u_nodes)
        self.Pp_L = self.P_left.eval_deriv(self.u_nodes, 1)

        # Right polynomial
        self.P_R = self.P_right.eval(self.u_nodes)
        self.Pp_R = self.P_right.eval_deriv(self.u_nodes, 1)

        # Q polynomial
        self.Q_t = self.Q.eval(self.t_nodes)
        self.Qp_t = self.Q.eval_deriv(self.t_nodes, 1)
        self.Qpp_t = self.Q.eval_deriv(self.t_nodes, 2)

    def eval_monomial(self, mono: Monomial, ell: int, ellbar: int) -> float:
        """
        Evaluate a single monomial contribution.

        The monomial A^i B^j C^k D^m determines:
        - i: x-derivative order for left P
        - j: y-derivative order for right P
        - m: mixed derivative (affects both)
        - k: no derivatives, just base C factor

        The derivative structure matches the oracle's I-terms:
        - (i=1, j=1, k=0, m=0) → AB → like I₁ d²/dxdy
        - (i=0, j=0, k=0, m=1) → D → like I₂ base
        - (i=1, j=0, k=1, m=0) → AC → like I₃ d/dx
        - (i=0, j=1, k=1, m=0) → BC → like I₄ d/dy
        """
        i, j, k, m = mono.i, mono.j, mono.k, mono.m

        # Determine the (1-u) weight exponent
        # From Euler-Maclaurin: base weight is (1-u)^{ℓ+ℓ̄}
        # Each derivative reduces the weight
        # TODO: Verify this formula from PRZZ
        total_derivs = i + j + m  # Total derivative order
        base_weight = ell + ellbar
        weight_exp = max(0, base_weight - total_derivs)

        # Determine what kind of integral this is
        # Based on (i, j, k, m), we compute different things

        total = 0.0

        for iu in range(self.n_quad):
            u = self.u_nodes[iu]
            wu = self.u_weights[iu]

            P_L_val = self.P_L[iu]
            Pp_L_val = self.Pp_L[iu]
            P_R_val = self.P_R[iu]
            Pp_R_val = self.Pp_R[iu]

            # Skip if polynomial is zero
            if abs(P_L_val) < 1e-15 or abs(P_R_val) < 1e-15:
                continue

            # Compute the "block" values at this u
            # A block = P'/P (log-derivative from x)
            # B block = P'/P (log-derivative from y)
            # C block = some base value (related to integration constant)
            # D block = second derivative structure

            A_block = Pp_L_val / P_L_val
            B_block = Pp_R_val / P_R_val

            # For C block: in the Ψ formula, C = ζ'/ζ at base point
            # In polynomial approx, this is related to P'/P but needs careful interpretation
            # For now, use 0 as placeholder (the C terms should cancel in XY + Z)
            C_block = 0.0

            # For D block: paired structure, related to (Q'/Q)² or (P''/P - (P'/P)²)
            # For now, compute as P''/P - (P'/P)² for left side
            Ppp_L_val = self.P_left.eval_deriv(np.array([u]), 2)[0]
            D_block_raw = Ppp_L_val / P_L_val - (Pp_L_val / P_L_val) ** 2 if abs(P_L_val) > 1e-15 else 0.0

            for it in range(self.n_quad):
                t = self.t_nodes[it]
                wt = self.t_weights[it]

                Qt = self.Q_t[it]
                Qp = self.Qp_t[it]
                E2 = exp(2 * self.R * t)

                if abs(Qt) < 1e-15:
                    continue

                # Add Q contributions to blocks
                Q_ratio = Qp / Qt
                darg_dx = self.theta * t  # dα/dx at x=0
                darg_dy = self.theta * (t - 1)  # dα/dy at x=0

                # Full block values including Q and exp
                A_full = A_block + (Q_ratio + self.R) * (darg_dx + darg_dy * 0.5)  # Simplified
                B_full = B_block + (Q_ratio + self.R) * (darg_dy + darg_dx * 0.5)  # Simplified

                # D block from Q structure
                d2_logQ = self.Qpp_t[it] / Qt - (Qp / Qt) ** 2 if abs(Qt) > 1e-15 else 0.0
                D_block = D_block_raw + d2_logQ * darg_dx * darg_dy

                # Base integrand
                base = P_L_val * P_R_val * Qt * Qt * E2

                # Compute monomial value: A^i × B^j × C^k × D^m × Base
                mono_val = (A_full ** i) * (B_full ** j) * (C_block ** k) * (D_block ** m) * base

                # Apply weight
                weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

                total += wu * wt * mono.coeff * mono_val * weight

        # Apply 1/θ prefactor
        return total / self.theta

    def eval_pair(self, ell: int, ellbar: int, verbose: bool = False) -> float:
        """
        Evaluate full contribution for pair (ℓ, ℓ̄).

        Expands to monomials and evaluates each.
        """
        all_monomials = expand_pair_to_monomials(ell, ellbar)
        combined = combine_like_monomials(all_monomials)

        if verbose:
            print(f"\nPair ({ell},{ellbar}): {len(combined)} unique monomials")
            for (i, j, k, m), coeff in sorted(combined.items()):
                print(f"  {coeff:+d} × A^{i}B^{j}C^{k}D^{m}")

        total = 0.0
        for (i, j, k, m), coeff in combined.items():
            if coeff == 0:
                continue

            mono = Monomial(i=i, j=j, k=k, m=m, coeff=coeff)
            contrib = self.eval_monomial(mono, ell, ellbar)
            total += contrib

            if verbose:
                print(f"    A^{i}B^{j}C^{k}D^{m}: {contrib:.6f}")

        if verbose:
            print(f"  Total ({ell},{ellbar}) = {total:.6f}")

        return total


def test_monomial_expansion():
    """Test the monomial expansion for (1,1) and (2,2)."""
    print("=" * 70)
    print("MONOMIAL EXPANSION TEST")
    print("=" * 70)

    for (ell, ellbar) in [(1, 1), (2, 2)]:
        print(f"\n--- Pair ({ell},{ellbar}) ---")

        all_monomials = expand_pair_to_monomials(ell, ellbar)
        combined = combine_like_monomials(all_monomials)

        print(f"Total monomials: {len(all_monomials)}")
        print(f"Unique monomials: {len(combined)}")
        print(f"Non-zero monomials: {sum(1 for c in combined.values() if c != 0)}")

        print("\nExpanded form:")
        for (i, j, k, m), coeff in sorted(combined.items()):
            if coeff != 0:
                mono_str = ""
                if i > 0:
                    mono_str += f"A^{i}" if i > 1 else "A"
                if j > 0:
                    mono_str += f"B^{j}" if j > 1 else "B"
                if k > 0:
                    mono_str += f"C^{k}" if k > 1 else "C"
                if m > 0:
                    mono_str += f"D^{m}" if m > 1 else "D"
                if not mono_str:
                    mono_str = "1"
                print(f"  {coeff:+d} × {mono_str}")


def validate_11_monomial():
    """Validate monomial evaluator on (1,1)."""
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("\n" + "=" * 70)
    print("MONOMIAL EVALUATOR VALIDATION: (1,1)")
    print("=" * 70)

    # Oracle reference
    oracle = przz_oracle_22(P1, Q, theta, R, n_quad)
    print(f"\nOracle (1,1): {oracle.total:.6f}")
    print(f"  I₁={oracle.I1:.6f}, I₂={oracle.I2:.6f}")
    print(f"  I₃={oracle.I3:.6f}, I₄={oracle.I4:.6f}")

    # Monomial evaluator
    evaluator = MonomialEvaluator(P1, P1, Q, theta, R, n_quad)
    mono_total = evaluator.eval_pair(1, 1, verbose=True)

    print(f"\nMonomial total: {mono_total:.6f}")
    print(f"Oracle total:   {oracle.total:.6f}")
    print(f"Ratio:          {mono_total/oracle.total:.6f}")


if __name__ == "__main__":
    test_monomial_expansion()
    validate_11_monomial()
