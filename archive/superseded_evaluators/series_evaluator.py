"""
src/series_evaluator.py
Series-Based Per-Monomial Evaluator for PRZZ

This evaluator uses the TruncatedSeries engine to automatically extract
derivatives instead of hard-coding I-term integrands. This makes (2,2)+
pairs tractable without writing eval_l1m1_XY for every combination.

KEY ARCHITECTURE (from GPT guidance, Session 9):

1. The Ψ expansion gives monomials A^a × B^b × C_α^{c_α} × C_β^{c_β} × D^d
2. A = singleton x-block (contributes x-derivative)
3. B = singleton y-block (contributes y-derivative)
4. D = paired block (no additional singleton derivatives)
5. C_α, C_β = pole contributions (factors, not integration blocks)

DERIVATIVE EXTRACTION:
- The coefficient of x^a y^b in the series gives the contribution
- Weight is (1-u)^{a+b} (per-monomial, not per-I-term)

STAGE SEPARATION:
1. Pre-mirror, pre-Q: This evaluator computes the F_d × F_d' product structure
2. Q-operators: Applied based on total derivative orders
3. Numerical integration: Over u, t, and auxiliary variables

VALIDATED AGAINST:
- (1,1) oracle: 0.359159
- GenEval for (1,1) only (GenEval is NOT a universal basis for higher pairs)
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

from src.series import TruncatedSeries
from src.composition import compose_polynomial_on_affine, compose_exp_on_affine
from src.psi_expansion import expand_psi, MonomialTwoC


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@dataclass
class MonomialContribution:
    """Result of evaluating a single Ψ monomial."""
    monomial: MonomialTwoC
    deriv_x: int          # x-derivative order (= a from monomial)
    deriv_y: int          # y-derivative order (= b from monomial)
    weight_exp: int       # (1-u)^{a+b} exponent
    integral_value: float # The evaluated integral
    contribution: float   # psi_coeff × integral


class SeriesEvaluator:
    """
    Evaluator that uses TruncatedSeries for automatic derivative extraction.

    For a pair (ℓ, ℓ̄), computes:
        c_{ℓ,ℓ̄} = Σ_{monomials} coeff × ∫∫ [x^a y^b term] × (1-u)^{a+b} × Q-ops × du dt

    The key insight is that we represent the integrand as a bivariate series
    in (x, y) and extract the appropriate coefficient for each monomial.
    """

    def __init__(self, P_ell, P_ellbar, Q, R: float, theta: float, n_quad: int = 60):
        """
        Initialize the series evaluator.

        Args:
            P_ell: Left polynomial (P_ℓ)
            P_ellbar: Right polynomial (P_ℓ̄)
            Q: Q polynomial
            R: PRZZ R parameter
            theta: PRZZ θ parameter (typically 4/7)
            n_quad: Quadrature points
        """
        self.P_ell = P_ell
        self.P_ellbar = P_ellbar
        self.Q = Q
        self.R = R
        self.theta = theta
        self.n_quad = n_quad

        # Variable names for the series
        self.var_names = ("x", "y")

        # Quadrature nodes
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.t_nodes, self.t_weights = gauss_legendre_01(n_quad)

        # Cache for repeated values
        self._setup_cache()

    def _setup_cache(self):
        """Precompute commonly used values."""
        # Q values on t grid
        self.Q_t = self.Q.eval(self.t_nodes)
        self.Q_t_deriv1 = self.Q.eval_deriv(self.t_nodes, 1)
        self.Q_t_deriv2 = self.Q.eval_deriv(self.t_nodes, 2)

        # Exponential factor
        self.exp_2Rt = np.exp(2 * self.R * self.t_nodes)

    def _build_integrand_series(self, u_idx: int, t_idx: int) -> TruncatedSeries:
        """
        Build the integrand as a TruncatedSeries in (x, y).

        The PRZZ integrand structure (matching GenEval/PerMonomialEvaluator):
        - P_ℓ(u - x)  ← note: minus x, not plus
        - P_ℓ̄(u - y)  ← note: minus y, not plus
        - Q(t) × Q(t)  ← Q is evaluated at t, with chain rule for x,y dependence
        - exp(2Rt)
        - Prefactor (1 + θ(x+y))/θ

        The Q arguments at x=y=0 are:
        - arg_α = θt  (but Q is evaluated at t)
        - arg_β = θ(1-t)  (but Q is evaluated at t)

        Chain rule derivatives:
        - darg_α/dx = θt, darg_α/dy = θ(t-1)
        - darg_β/dx = θ(t-1), darg_β/dy = θt

        Returns a TruncatedSeries that we can extract coefficients from.
        """
        u = self.u_nodes[u_idx]
        t = self.t_nodes[t_idx]

        # Chain rule argument derivatives (from GenEval structure)
        darg_alpha_dx = self.theta * t
        darg_alpha_dy = self.theta * (t - 1)
        darg_beta_dx = self.theta * (t - 1)
        darg_beta_dy = self.theta * t

        # P_ℓ(u - x): composed with -x coefficient
        P_ell_series = compose_polynomial_on_affine(
            self.P_ell, u, {"x": -1.0}, self.var_names
        )

        # P_ℓ̄(u - y): composed with -y coefficient
        P_ellbar_series = compose_polynomial_on_affine(
            self.P_ellbar, u, {"y": -1.0}, self.var_names
        )

        # Q(arg_α) where arg_α at x=y=0 is t (or θt depending on convention)
        # The GenEval uses Q.eval(t), so base point is t
        # Chain rule gives Q'(t) × darg_α/dx for x-derivative
        Q_alpha_series = compose_polynomial_on_affine(
            self.Q, t, {"x": darg_alpha_dx, "y": darg_alpha_dy}, self.var_names
        )

        # Q(arg_β) where arg_β at x=y=0 is also t in the code
        Q_beta_series = compose_polynomial_on_affine(
            self.Q, t, {"x": darg_beta_dx, "y": darg_beta_dy}, self.var_names
        )

        # exp(R(α + β))
        # At x=y=0: exp(2Rt)
        # d(α+β)/dx = darg_alpha_dx + darg_beta_dx = θt + θ(t-1) = θ(2t-1)
        # d(α+β)/dy = darg_alpha_dy + darg_beta_dy = θ(t-1) + θt = θ(2t-1)
        sum_dx = darg_alpha_dx + darg_beta_dx
        sum_dy = darg_alpha_dy + darg_beta_dy
        exp_series = compose_exp_on_affine(
            self.R, 2 * t, {"x": sum_dx, "y": sum_dy}, self.var_names
        )

        # Prefactor (1 + θ(x+y))/θ = 1/θ + x + y
        prefactor = TruncatedSeries.from_scalar(1.0 / self.theta, self.var_names)
        prefactor = prefactor + TruncatedSeries.variable("x", self.var_names)
        prefactor = prefactor + TruncatedSeries.variable("y", self.var_names)

        # Full integrand: prefactor × P_ℓ × P_ℓ̄ × Q_α × Q_β × exp(R(α+β))
        integrand = prefactor * P_ell_series * P_ellbar_series * Q_alpha_series * Q_beta_series * exp_series

        return integrand

    def _extract_derivative_contribution(
        self,
        deriv_x: int,
        deriv_y: int,
        weight_exp: int
    ) -> float:
        """
        Extract the contribution for derivative order (deriv_x, deriv_y).

        This integrates over (u, t) with weight (1-u)^{weight_exp}.

        Args:
            deriv_x: Number of x-derivatives (= a from monomial)
            deriv_y: Number of y-derivatives (= b from monomial)
            weight_exp: Exponent for (1-u) weight (= a + b)

        Returns:
            The integrated contribution.
        """
        # Build the derivative extraction key
        deriv_vars = tuple(["x"] * deriv_x + ["y"] * deriv_y)

        # Double integral over (u, t)
        result = 0.0

        for u_idx in range(self.n_quad):
            u = self.u_nodes[u_idx]
            weight = (1.0 - u) ** weight_exp if weight_exp > 0 else 1.0

            t_integral = 0.0
            for t_idx in range(self.n_quad):
                # Build integrand series at this (u, t) point
                integrand = self._build_integrand_series(u_idx, t_idx)

                # Extract the x^{deriv_x} y^{deriv_y} coefficient
                coeff = integrand.extract(deriv_vars)
                if isinstance(coeff, np.ndarray):
                    coeff = float(coeff)

                t_integral += self.t_weights[t_idx] * coeff

            result += self.u_weights[u_idx] * weight * t_integral

        # The prefactor (1/θ + x + y) already includes the 1/θ factor,
        # so no additional division is needed here.
        return result

    def eval_monomial(self, mono: MonomialTwoC) -> MonomialContribution:
        """
        Evaluate a single Ψ monomial contribution.

        Args:
            mono: MonomialTwoC from the Ψ expansion

        Returns:
            MonomialContribution with detailed breakdown
        """
        # Derivative orders from A, B counts
        deriv_x = mono.a  # A contributes x-derivatives
        deriv_y = mono.b  # B contributes y-derivatives

        # Weight exponent
        weight_exp = mono.a + mono.b

        # Extract the derivative contribution
        integral_value = self._extract_derivative_contribution(
            deriv_x, deriv_y, weight_exp
        )

        # Full contribution with Ψ coefficient
        contribution = mono.coeff * integral_value

        return MonomialContribution(
            monomial=mono,
            deriv_x=deriv_x,
            deriv_y=deriv_y,
            weight_exp=weight_exp,
            integral_value=integral_value,
            contribution=contribution
        )

    def eval_pair(self, ell: int, ellbar: int, verbose: bool = False) -> float:
        """
        Evaluate the full contribution from pair (ℓ, ℓ̄).

        Args:
            ell: Left piece index
            ellbar: Right piece index
            verbose: Print per-monomial breakdown

        Returns:
            Total pair contribution c_{ℓ,ℓ̄}
        """
        monomials = expand_psi(ell, ellbar)

        if verbose:
            print(f"\n=== SeriesEvaluator ({ell},{ellbar}): {len(monomials)} monomials ===")

        total = 0.0
        for mono in monomials:
            result = self.eval_monomial(mono)
            total += result.contribution

            if verbose:
                print(f"  {mono}:")
                print(f"    deriv=({result.deriv_x},{result.deriv_y}), "
                      f"weight=(1-u)^{result.weight_exp}")
                print(f"    integral={result.integral_value:.6f}, "
                      f"contrib={result.contribution:.6f}")

        if verbose:
            print(f"  TOTAL = {total:.6f}")

        return total


def compute_c_series(P1, P2, P3, Q, R: float, theta: float = 4/7,
                     n_quad: int = 60, verbose: bool = False) -> float:
    """
    Compute total c using the SeriesEvaluator.

    Args:
        P1, P2, P3: The three PRZZ polynomials
        Q: The Q polynomial
        R: PRZZ R parameter
        theta: PRZZ θ parameter
        n_quad: Quadrature points
        verbose: Print per-pair breakdown

    Returns:
        Total c value
    """
    poly_map = {1: P1, 2: P2, 3: P3}
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    total = 0.0
    for ell, ellbar in pairs:
        P_ell = poly_map[ell]
        P_ellbar = poly_map[ellbar]

        evaluator = SeriesEvaluator(P_ell, P_ellbar, Q, R, theta, n_quad)
        contrib = evaluator.eval_pair(ell, ellbar, verbose=verbose)

        # Symmetry factor
        sym = 1 if ell == ellbar else 2
        total += sym * contrib

        if verbose:
            print(f"  ({ell},{ellbar}) × {sym} = {sym * contrib:.6f}")

    if verbose:
        print(f"\nTotal c = {total:.6f}")

    return total


# =============================================================================
# TESTING
# =============================================================================

def test_11_oracle():
    """Test (1,1) against the known oracle."""
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    evaluator = SeriesEvaluator(P1, P1, Q, R=1.3036, theta=4/7, n_quad=60)
    result = evaluator.eval_pair(1, 1, verbose=True)

    oracle = 0.359159
    print(f"\n(1,1) Result: {result:.6f}")
    print(f"Oracle: {oracle:.6f}")
    print(f"Error: {abs(result - oracle) / oracle * 100:.2f}%")


def test_all_pairs():
    """Test all K=3 pairs."""
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    c = compute_c_series(P1, P2, P3, Q, R=1.3036, theta=4/7, n_quad=60, verbose=True)

    target = 2.137
    print(f"\nTotal c = {c:.6f}")
    print(f"Target = {target:.3f}")
    print(f"Error = {abs(c - target) / target * 100:.1f}%")


if __name__ == "__main__":
    print("=" * 70)
    print("SERIES EVALUATOR TEST")
    print("=" * 70)

    test_11_oracle()
    print("\n" + "=" * 70)
    test_all_pairs()
