"""
src/przz_section7_evaluator.py
PRZZ Section 7 Config-Based Evaluator

Key insight from GPT:
- Ψ gives CONFIG STRUCTURE (coefficients, combinatorics)
- F_d + Euler-Maclaurin give VALUES
- Each config has its own weight and Case A/B/C behavior based on ω

The I₁-I₄ structure is the (1,1) truncation of the general config sum.
For higher ℓ, we must use the full config enumeration.

For d=1:
  ω(1, l) = l₁ - 1
  Case A (ω = -1, l₁ = 0): derivative form
  Case B (ω = 0, l₁ = 1): evaluation form
  Case C (ω > 0, l₁ > 1): kernel integral form
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
class PConfig:
    """
    A p-config for pair (ℓ, ℓ̄).

    Represents one term in the Ψ expansion:
    coeff × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}

    where X = A-C, Y = B-C, Z = D-C².

    The monomial structure determines which F_d case to use:
    - Powers of A determine left-side ω
    - Powers of B determine right-side ω
    """
    ell: int      # ℓ (left piece index)
    ellbar: int   # ℓ̄ (right piece index)
    p: int        # p-index in the sum
    coeff: int    # C(ℓ,p) × C(ℓ̄,p) × p!

    # Derived quantities
    x_power: int  # ℓ - p (power of X = A-C)
    y_power: int  # ℓ̄ - p (power of Y = B-C)
    z_power: int  # p (power of Z = D-C²)

    # The left/right derivative orders for F_d evaluation
    # For d=1: l_left = x_power + z_power (contributions involving A or D)
    # For d=1: l_right = y_power + z_power (contributions involving B or D)

    @property
    def omega_left(self) -> int:
        """ω for left F_d: l₁ - 1 where l₁ = effective derivative order on x-side."""
        # The X^k = (A-C)^k term contributes k "A-blocks"
        # The Z^p = (D-C²)^p term contributes p "D-blocks" which involve mixed xy
        # For the left side: effective l₁ = x_power (number of A factors)
        return self.x_power - 1

    @property
    def omega_right(self) -> int:
        """ω for right F_d: l₁ - 1 where l₁ = effective derivative order on y-side."""
        return self.y_power - 1

    @property
    def euler_maclaurin_weight(self) -> int:
        """
        The (1-u)^k power for Euler-Maclaurin.

        From PRZZ: the convolution index determines this.
        For pair (ℓ, ℓ̄), the base weight is (1-u)^{ℓ+ℓ̄}.
        But derivatives reduce this: each derivative loses one power.

        For X^{ℓ-p} Y^{ℓ̄-p} Z^p:
        - X contributes (ℓ-p) derivatives on left
        - Y contributes (ℓ̄-p) derivatives on right
        - Z contributes p "paired" derivatives

        The weight exponent should be related to the remaining undifferentiated terms.

        TODO: Verify this formula from PRZZ Section 7.
        """
        # For now, use the base weight (ℓ+ℓ̄) minus derivative order
        # This is a placeholder - needs verification
        total_deriv_order = self.x_power + self.y_power + 2 * self.z_power
        return max(0, self.ell + self.ellbar - total_deriv_order)


def generate_pconfigs(ell: int, ellbar: int) -> List[PConfig]:
    """
    Generate all p-configs for pair (ℓ, ℓ̄).

    Ψ_{ℓ,ℓ̄} = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}
    """
    configs = []
    max_p = min(ell, ellbar)

    for p in range(max_p + 1):
        coeff = comb(ell, p) * comb(ellbar, p) * factorial(p)
        configs.append(PConfig(
            ell=ell,
            ellbar=ellbar,
            p=p,
            coeff=coeff,
            x_power=ell - p,
            y_power=ellbar - p,
            z_power=p
        ))

    return configs


class Section7Evaluator:
    """
    Evaluates main term contributions using PRZZ Section 7 structure.

    Each config is evaluated via:
    1. Euler-Maclaurin weight from config
    2. F_d case A/B/C based on ω
    3. Q/exp factors
    """

    def __init__(self, P_left, P_right, Q, theta: float, R: float, n_quad: int = 60):
        """
        Initialize evaluator for a pair.

        Args:
            P_left: Left polynomial (P_ℓ)
            P_right: Right polynomial (P_ℓ̄)
            Q: Q polynomial
            theta: θ = 4/7
            R: R parameter
            n_quad: Quadrature points
        """
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
        """Precompute polynomial values at quadrature nodes."""
        # Left polynomial and derivatives
        self.P_left_u = self.P_left.eval(self.u_nodes)
        self.P_left_prime_u = self.P_left.eval_deriv(self.u_nodes, 1)
        self.P_left_double_prime_u = self.P_left.eval_deriv(self.u_nodes, 2)

        # Right polynomial and derivatives
        self.P_right_u = self.P_right.eval(self.u_nodes)
        self.P_right_prime_u = self.P_right.eval_deriv(self.u_nodes, 1)
        self.P_right_double_prime_u = self.P_right.eval_deriv(self.u_nodes, 2)

        # Q polynomial and derivatives
        self.Q_t = self.Q.eval(self.t_nodes)
        self.Q_prime_t = self.Q.eval_deriv(self.t_nodes, 1)
        self.Q_double_prime_t = self.Q.eval_deriv(self.t_nodes, 2)

    def compute_F_d_left(self, iu: int, omega: int) -> float:
        """
        Compute left F_d factor based on ω case.

        For d=1:
        - Case A (ω = -1): P'/P type (derivative at point)
        - Case B (ω = 0): P type (evaluation at point)
        - Case C (ω > 0): kernel integral type

        Args:
            iu: u-node index
            omega: ω value for this config

        Returns:
            F_d contribution for left side
        """
        P_val = self.P_left_u[iu]
        Pp_val = self.P_left_prime_u[iu]

        if omega == -1:
            # Case A: derivative form
            # F_d ~ P'(u) contribution
            return Pp_val

        elif omega == 0:
            # Case B: evaluation form
            # F_d ~ P(u) contribution
            return P_val

        else:  # omega > 0
            # Case C: kernel integral form
            # F_d involves ∫ P(u) × kernel du
            # For now, use P(u) as placeholder - needs kernel implementation
            # TODO: Implement proper Case C kernel from PRZZ
            return P_val

    def compute_F_d_right(self, iu: int, omega: int) -> float:
        """Compute right F_d factor based on ω case."""
        P_val = self.P_right_u[iu]
        Pp_val = self.P_right_prime_u[iu]

        if omega == -1:
            # Case A: derivative form
            return Pp_val

        elif omega == 0:
            # Case B: evaluation form
            return P_val

        else:  # omega > 0
            # Case C: kernel integral form
            # TODO: Implement proper Case C kernel
            return P_val

    def eval_config(self, config: PConfig, verbose: bool = False) -> float:
        """
        Evaluate a single p-config contribution.

        Uses:
        1. Euler-Maclaurin weight (1-u)^k
        2. Left F_d based on omega_left
        3. Right F_d based on omega_right
        4. Q and exp factors
        """
        omega_left = config.omega_left
        omega_right = config.omega_right
        weight_exp = config.euler_maclaurin_weight

        if verbose:
            print(f"  Config p={config.p}: ω_L={omega_left}, ω_R={omega_right}, weight=(1-u)^{weight_exp}")

        total = 0.0

        for iu in range(self.n_quad):
            u = self.u_nodes[iu]
            wu = self.u_weights[iu]
            weight = (1.0 - u) ** weight_exp if weight_exp >= 0 else 1.0

            # Compute F_d factors
            F_left = self.compute_F_d_left(iu, omega_left)
            F_right = self.compute_F_d_right(iu, omega_right)

            for it in range(self.n_quad):
                t = self.t_nodes[it]
                wt = self.t_weights[it]

                # Q and exp factors
                Q_val = self.Q_t[it]
                Q_val_sq = Q_val * Q_val
                exp_factor = exp(2 * self.R * t)

                # Combine: coeff × F_left × F_right × Q² × exp × weight
                integrand = F_left * F_right * Q_val_sq * exp_factor

                total += wu * wt * config.coeff * integrand * weight

        # Apply 1/θ prefactor
        total /= self.theta

        return total

    def eval_pair(self, ell: int, ellbar: int, verbose: bool = False) -> float:
        """
        Evaluate full contribution for pair (ℓ, ℓ̄).

        Sums over all p-configs.
        """
        configs = generate_pconfigs(ell, ellbar)

        if verbose:
            print(f"\nEvaluating pair ({ell},{ellbar}) with {len(configs)} configs:")

        total = 0.0
        for config in configs:
            contrib = self.eval_config(config, verbose=verbose)
            total += contrib

            if verbose:
                print(f"    p={config.p}: coeff={config.coeff}, contrib={contrib:.6f}")

        if verbose:
            print(f"  Total ({ell},{ellbar}) = {total:.6f}")

        return total


def validate_11_section7():
    """
    Validate that Section 7 evaluator reproduces (1,1) correctly.

    For (1,1), there are 2 p-configs:
    - p=0: coeff=1, X^1 Y^1 Z^0
    - p=1: coeff=1, X^0 Y^0 Z^1
    """
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("=" * 70)
    print("SECTION 7 VALIDATION: (1,1) pair")
    print("=" * 70)

    # Oracle reference
    oracle = przz_oracle_22(P1, Q, theta, R, n_quad)
    print(f"\nOracle (1,1): I₁={oracle.I1:.6f}, I₂={oracle.I2:.6f}, "
          f"I₃={oracle.I3:.6f}, I₄={oracle.I4:.6f}")
    print(f"Oracle total: {oracle.total:.6f}")

    # Section 7 evaluator
    evaluator = Section7Evaluator(P1, P1, Q, theta, R, n_quad)
    section7_total = evaluator.eval_pair(1, 1, verbose=True)

    print(f"\nSection 7 total: {section7_total:.6f}")
    print(f"Oracle total:    {oracle.total:.6f}")
    print(f"Ratio:           {section7_total/oracle.total:.6f}")

    if abs(section7_total - oracle.total) < 0.01 * abs(oracle.total):
        print("\n✓ (1,1) validation PASSED")
        return True
    else:
        print("\n✗ (1,1) validation FAILED")
        return False


def evaluate_22_section7():
    """
    Evaluate (2,2) using Section 7 config-based approach.

    For (2,2), there are 3 p-configs:
    - p=0: coeff=1, X^2 Y^2 Z^0
    - p=1: coeff=4, X^1 Y^1 Z^1
    - p=2: coeff=2, X^0 Y^0 Z^2
    """
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("\n" + "=" * 70)
    print("SECTION 7 EVALUATION: (2,2) pair")
    print("=" * 70)

    # 4-term oracle (for comparison, but it's NOT the correct oracle for (2,2))
    oracle = przz_oracle_22(P2, Q, theta, R, n_quad)
    print(f"\n4-term 'oracle' (INCOMPLETE for (2,2)): {oracle.total:.6f}")
    print(f"  (This uses I₁-I₄ structure which is the (1,1) truncation)")

    # Section 7 evaluator
    evaluator = Section7Evaluator(P2, P2, Q, theta, R, n_quad)
    section7_total = evaluator.eval_pair(2, 2, verbose=True)

    print(f"\nSection 7 total (3 p-configs): {section7_total:.6f}")
    print(f"4-term oracle:                  {oracle.total:.6f}")
    print(f"Ratio (Section7 / 4-term):      {section7_total/oracle.total:.6f}")


def evaluate_full_k3_section7():
    """Evaluate all K=3 pairs using Section 7 approach."""
    from src.polynomials import load_przz_polynomials
    from math import log

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {1: P1, 2: P2, 3: P3}
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("\n" + "=" * 70)
    print("FULL K=3 EVALUATION: Section 7 Config-Based")
    print("=" * 70)

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    total_c = 0.0

    for (ell, ellbar) in pairs:
        P_left = polys[ell]
        P_right = polys[ellbar]

        evaluator = Section7Evaluator(P_left, P_right, Q, theta, R, n_quad)
        contrib = evaluator.eval_pair(ell, ellbar, verbose=False)

        # Symmetry factor
        sym_factor = 1 if ell == ellbar else 2
        total_contrib = sym_factor * contrib
        total_c += total_contrib

        n_configs = min(ell, ellbar) + 1
        print(f"({ell},{ellbar}): {n_configs} configs, contrib={contrib:.6f}, "
              f"sym×={sym_factor}, total={total_contrib:.6f}")

    kappa = 1 - log(total_c) / R if total_c > 0 else float('nan')

    print()
    print(f"Total c = {total_c:.6f}")
    print(f"Target c = 2.137")
    print(f"Ratio: {total_c/2.137:.4f}")
    print()
    print(f"κ = 1 - log(c)/R = {kappa:.6f}")
    print(f"Target κ = 0.417")


if __name__ == "__main__":
    validate_11_section7()
    evaluate_22_section7()
    evaluate_full_k3_section7()
