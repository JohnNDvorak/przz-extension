"""
src/difference_quotient.py
PRZZ Difference Quotient Identity Implementation

Phase 21: Implement exact difference quotient for D = 0

PURPOSE:
========
Implement the PRZZ difference quotient identity (TeX Lines 1502-1511) to achieve
D = 0 analytically by pre-combining direct and mirror terms before operator
application.

MATHEMATICAL STRUCTURE:
=======================
The PRZZ difference quotient identity transforms:

    [N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
    = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt

This identity:
1. Removes the 1/(α+β) singularity analytically via t-integral
2. Pre-combines direct and mirror terms BEFORE operator application
3. Uses operator shift Q(D) → Q(1+D) for mirror eigenvalues

KEY INSIGHT:
============
Current implementation applies mirror AFTER evaluating integrals separately.
This module combines direct and mirror WITHIN the integral, producing D = 0.

OPERATOR SHIFT IDENTITY:
========================
For any polynomial Q and the T-weight factor T^{-s}:
    Q(D_α)(T^{-s}F) = T^{-s} × Q(1 + D_α)F

This means mirror terms use Q(1 + eigenvalue) instead of Q(eigenvalue).

REFERENCES:
===========
- PRZZ TeX Lines 1499-1501: Bracket definition
- PRZZ TeX Lines 1502-1511: Difference quotient identity
- docs/TEX_MIRROR_OPERATOR_SHIFT.md: Operator shift derivation

USAGE:
======
>>> from src.difference_quotient import DifferenceQuotientBracket
>>> bracket = DifferenceQuotientBracket(theta=4/7, R=1.3036)
>>> bracket.verify_scalar_identity()  # Gate test
>>> series = bracket.evaluate_as_series(t=0.5, var_names=("x", "y"))
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
import numpy as np

from src.series import TruncatedSeries
from src.composition import compose_exp_on_affine, compose_polynomial_on_affine
from src.quadrature import gauss_legendre_01


# =============================================================================
# SCALAR DIFFERENCE QUOTIENT IDENTITY
# =============================================================================


def scalar_difference_quotient_lhs(z: float, s: float) -> float:
    """
    Compute (1 - z^{-s}) / s for the difference quotient LHS.

    This is the direct difference quotient form.

    Args:
        z: Base (z > 0)
        s: Exponent (s != 0)

    Returns:
        (1 - z^{-s}) / s

    Raises:
        ValueError: If s is too close to zero
    """
    if abs(s) < 1e-15:
        raise ValueError("s cannot be zero (singularity)")
    return (1.0 - z**(-s)) / s


def scalar_difference_quotient_rhs(z: float, s: float, n_quad: int = 50) -> float:
    """
    Compute log(z) × ∫₀¹ z^{-ts} dt for the difference quotient RHS.

    This is the regularized t-integral form that avoids the 1/s singularity.

    Args:
        z: Base (z > 0)
        s: Exponent
        n_quad: Number of quadrature points

    Returns:
        log(z) × ∫₀¹ z^{-ts} dt
    """
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    integral = 0.0
    for t, w in zip(t_nodes, t_weights):
        integral += z**(-t * s) * w

    return np.log(z) * integral


def verify_scalar_difference_quotient(
    z: float,
    s: float,
    n_quad: int = 50
) -> Tuple[float, float, float]:
    """
    Verify the scalar difference quotient identity:
        (1 - z^{-s})/s == log(z) × ∫₀¹ z^{-ts} dt

    Args:
        z: Base (z > 0)
        s: Exponent (s != 0)
        n_quad: Number of quadrature points

    Returns:
        (lhs, rhs, relative_error)
    """
    lhs = scalar_difference_quotient_lhs(z, s)
    rhs = scalar_difference_quotient_rhs(z, s, n_quad)

    if abs(lhs) > 1e-15:
        rel_error = abs(lhs - rhs) / abs(lhs)
    else:
        rel_error = abs(lhs - rhs)

    return lhs, rhs, rel_error


# =============================================================================
# PRZZ-SPECIFIC SCALAR LIMITS
# =============================================================================


def przz_scalar_limit(R: float) -> float:
    """
    Compute the scalar limit of the PRZZ bracket at x=y=0.

    At α = β = -R/L and x = y = 0:
        Bracket → (1 - exp(2R)) / (-2R/L) × L
                = (exp(2R) - 1) / (2R)

    This is the expected value for gate testing.

    Args:
        R: PRZZ R parameter

    Returns:
        (exp(2R) - 1) / (2R)
    """
    if abs(R) < 1e-15:
        return 1.0  # L'Hôpital limit
    return (np.exp(2 * R) - 1) / (2 * R)


def przz_scalar_limit_via_t_integral(R: float, n_quad: int = 50) -> float:
    """
    Compute the scalar limit using the t-integral representation.

    At x = y = 0:
        ∫₀¹ exp(2Rt) dt = (exp(2R) - 1) / (2R)

    Args:
        R: PRZZ R parameter
        n_quad: Number of quadrature points

    Returns:
        ∫₀¹ exp(2Rt) dt
    """
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    integral = 0.0
    for t, w in zip(t_nodes, t_weights):
        integral += np.exp(2 * R * t) * w

    return integral


def verify_przz_scalar_limit(R: float, n_quad: int = 50) -> Tuple[float, float, float]:
    """
    Verify the PRZZ scalar limit matches the t-integral representation.

    Args:
        R: PRZZ R parameter
        n_quad: Number of quadrature points

    Returns:
        (analytic, quadrature, relative_error)
    """
    analytic = przz_scalar_limit(R)
    quadrature = przz_scalar_limit_via_t_integral(R, n_quad)

    if abs(analytic) > 1e-15:
        rel_error = abs(analytic - quadrature) / abs(analytic)
    else:
        rel_error = abs(analytic - quadrature)

    return analytic, quadrature, rel_error


# =============================================================================
# EIGENVALUE COMPUTATION
# =============================================================================


def get_direct_eigenvalues(
    t: float,
    theta: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Get the DIRECT eigenvalues A_α and A_β for the post-combined-identity structure.

    After the combined identity transformation, the Q operators act on
    exponentials with these affine eigenvalues.

    Eigenvalue structure: A = u0 + x_coeff * x + y_coeff * y

    For the DIRECT contribution:
        A_α = t + θ(t-1)x + θt·y
        A_β = t + θt·x + θ(t-1)·y

    Args:
        t: Integration variable in [0, 1]
        theta: PRZZ θ parameter

    Returns:
        ((u0_α, x_α, y_α), (u0_β, x_β, y_β))
    """
    u0 = t
    x_alpha = theta * (t - 1)
    y_alpha = theta * t
    x_beta = theta * t
    y_beta = theta * (t - 1)

    return (u0, x_alpha, y_alpha), (u0, x_beta, y_beta)


def get_mirror_eigenvalues(theta: float) -> Tuple[float, float]:
    """
    Get the MIRROR eigenvalues for the operator shift computation.

    For the mirror term T^{-s}N^{-βx-αy}, the eigenvalues are:
        A_α^{mirror} = θy  (note: swapped and flipped!)
        A_β^{mirror} = θx

    These are CONSTANT in t (no t-dependence for pure mirror eigenvalues).

    The operator shift identity then gives:
        Q(D_α)(T^{-s}F) = T^{-s} × Q(1 + A_α^{mirror})F

    Args:
        theta: PRZZ θ parameter

    Returns:
        (α_mirror_coeff, β_mirror_coeff) where eigenvalue = coeff * (x or y)
    """
    # A_α^{mirror} = θy → coefficient is θ for y
    # A_β^{mirror} = θx → coefficient is θ for x
    return (theta, theta)  # (y_coeff for α, x_coeff for β)


def get_unified_bracket_eigenvalues(
    t: float,
    theta: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Get the UNIFIED eigenvalues that combine direct and mirror contributions.

    The difference quotient identity combines both terms into a single t-integral.
    The t parameter interpolates between:
        - t=0: Mirror-dominated (T^{-s} contribution)
        - t=1: Direct-dominated (N^{αx+βy} contribution)

    For the UNIFIED structure after the combined identity:
        A_α = t + θ(t-1)x + θt·y
        A_β = t + θt·x + θ(t-1)·y

    These are exactly the post-identity eigenvalues.

    Args:
        t: Integration variable in [0, 1]
        theta: PRZZ θ parameter

    Returns:
        ((u0_α, x_α, y_α), (u0_β, x_β, y_β))
    """
    # The unified eigenvalues are the same as direct eigenvalues
    # because the combined identity absorbs the mirror structure
    return get_direct_eigenvalues(t, theta)


# =============================================================================
# SERIES CONSTRUCTION
# =============================================================================


def build_bracket_exp_series(
    t: float,
    theta: float,
    R: float,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Build the exponential factor for the difference quotient bracket.

    After the combined identity transformation and asymptotic simplification:
        exp(-Rθ(x+y)) × exp(2Rt(1+θ(x+y)))
        = exp(-Rθ(x+y) + 2Rt + 2Rtθ(x+y))
        = exp(2Rt + Rθ(2t-1)(x+y))

    This is the exponential core that enters the t-integral.

    Args:
        t: Integration variable in [0, 1]
        theta: PRZZ θ parameter
        R: PRZZ R parameter
        var_names: Variable names tuple

    Returns:
        TruncatedSeries for exp(2Rt + Rθ(2t-1)(x+y))
    """
    u0 = 2 * R * t
    lin_coeff = R * theta * (2 * t - 1)
    lin = {var_names[0]: lin_coeff, var_names[1]: lin_coeff}

    return compose_exp_on_affine(1.0, u0, lin, var_names)


def build_log_factor_series(
    theta: float,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Build the log factor (1 + θ(x+y)) from log(N^{x+y}T).

    In the combined identity:
        log(N^{x+y}T) = L × (1 + θ(x+y))

    The L factor is absorbed asymptotically. This returns just (1 + θ(x+y)).

    Args:
        theta: PRZZ θ parameter
        var_names: Variable names tuple

    Returns:
        TruncatedSeries for (1 + θ(x+y))
    """
    series = TruncatedSeries.from_scalar(1.0, var_names)
    series = series + TruncatedSeries.variable(var_names[0], var_names) * theta
    series = series + TruncatedSeries.variable(var_names[1], var_names) * theta
    return series


def build_q_factor_series(
    Q_poly,
    t: float,
    theta: float,
    var_names: Tuple[str, ...] = ("x", "y"),
    use_shifted: bool = False,
    shift_amount: float = 1.0
) -> TruncatedSeries:
    """
    Build the Q(A_α) × Q(A_β) factor for the bracket evaluation.

    For direct terms:
        Q(A_α)Q(A_β) where A_α, A_β are the post-identity eigenvalues

    For mirror terms (with operator shift):
        Q(shift + A_α^{mirror}) × Q(shift + A_β^{mirror})

    Args:
        Q_poly: PRZZ Q polynomial
        t: Integration variable in [0, 1]
        theta: PRZZ θ parameter
        var_names: Variable names tuple
        use_shifted: If True, use shifted eigenvalues (for mirror)
        shift_amount: Amount to shift (typically 1.0 for operator shift identity)

    Returns:
        TruncatedSeries for Q(A_α) × Q(A_β)
    """
    (u0_alpha, x_alpha, y_alpha), (u0_beta, x_beta, y_beta) = get_unified_bracket_eigenvalues(t, theta)

    if use_shifted:
        # Apply operator shift: Q(D) → Q(shift + D)
        u0_alpha += shift_amount
        u0_beta += shift_amount

    lin_alpha = {var_names[0]: x_alpha, var_names[1]: y_alpha}
    lin_beta = {var_names[0]: x_beta, var_names[1]: y_beta}

    Q_alpha = compose_polynomial_on_affine(Q_poly, u0_alpha, lin_alpha, var_names)
    Q_beta = compose_polynomial_on_affine(Q_poly, u0_beta, lin_beta, var_names)

    return Q_alpha * Q_beta


# =============================================================================
# DIFFERENCE QUOTIENT BRACKET CLASS
# =============================================================================


@dataclass
class BracketEvaluationResult:
    """Result of evaluating the difference quotient bracket."""

    # Scalar limit at x=y=0
    scalar_limit: float

    # Per-t integrand values (for debugging)
    t_values: np.ndarray
    integrand_values: np.ndarray

    # Final integrated value
    integrated_value: float

    # Comparison with analytic expectation
    analytic_expectation: float
    relative_error: float


class DifferenceQuotientBracket:
    """
    PRZZ difference quotient bracket evaluator.

    This class implements the difference quotient identity (TeX Lines 1502-1511):

        [N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
        = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt

    The key feature is that direct and mirror terms are COMBINED into a single
    t-integral, which avoids the 1/(α+β) singularity and produces D = 0.

    Attributes:
        theta: PRZZ θ parameter (typically 4/7)
        R: PRZZ R parameter
        n_quad_t: Number of quadrature points for t-integration
    """

    def __init__(
        self,
        theta: float = 4.0 / 7.0,
        R: float = 1.3036,
        n_quad_t: int = 40
    ):
        """
        Initialize the difference quotient bracket.

        Args:
            theta: PRZZ θ parameter
            R: PRZZ R parameter
            n_quad_t: Number of quadrature points for t-integration
        """
        self.theta = theta
        self.R = R
        self.n_quad_t = n_quad_t

        # Precompute quadrature nodes and weights
        self._t_nodes, self._t_weights = gauss_legendre_01(n_quad_t)

    def verify_scalar_identity(self, tol: float = 1e-10) -> Tuple[bool, float, float, float]:
        """
        Verify the scalar difference quotient identity as a gate test.

        This checks that:
            (exp(2R) - 1) / (2R) == ∫₀¹ exp(2Rt) dt

        Args:
            tol: Tolerance for identity verification

        Returns:
            (passed, analytic, quadrature, relative_error)
        """
        analytic, quadrature, rel_error = verify_przz_scalar_limit(
            self.R, self.n_quad_t
        )
        passed = rel_error < tol
        return passed, analytic, quadrature, rel_error

    def evaluate_integrand_at_t(
        self,
        t: float,
        var_names: Tuple[str, ...] = ("x", "y"),
        Q_poly=None,
        include_log_factor: bool = True
    ) -> TruncatedSeries:
        """
        Evaluate the bracket integrand at a single t value.

        The integrand is:
            exp(2Rt + Rθ(2t-1)(x+y)) × [log factor] × [Q factors]

        Args:
            t: Integration variable value
            var_names: Variable names tuple
            Q_poly: Optional Q polynomial (if None, uses Q=1)
            include_log_factor: Whether to include (1+θ(x+y)) factor

        Returns:
            TruncatedSeries representing the integrand at this t
        """
        # Build exponential core
        exp_series = build_bracket_exp_series(t, self.theta, self.R, var_names)

        # Optionally include log factor
        if include_log_factor:
            log_series = build_log_factor_series(self.theta, var_names)
            exp_series = exp_series * log_series

        # Apply Q factors if provided
        if Q_poly is not None:
            q_series = build_q_factor_series(
                Q_poly, t, self.theta, var_names,
                use_shifted=False
            )
            exp_series = exp_series * q_series

        return exp_series

    def evaluate_scalar_integral(self) -> float:
        """
        Evaluate the scalar (x=y=0) limit of the bracket integral.

        Returns:
            ∫₀¹ exp(2Rt) dt = (exp(2R) - 1) / (2R)
        """
        integral = 0.0
        for t, w in zip(self._t_nodes, self._t_weights):
            integral += np.exp(2 * self.R * t) * w
        return integral

    def evaluate_xy_coefficient_integral(
        self,
        Q_poly=None,
        include_log_factor: bool = True,
        var_names: Tuple[str, ...] = ("x", "y")
    ) -> float:
        """
        Evaluate the xy coefficient of the bracket integral.

        This is the key quantity for I₁ evaluation:
            [d²/dxdy ∫₀¹ integrand(t,x,y) dt]_{x=y=0}

        Args:
            Q_poly: Optional Q polynomial
            include_log_factor: Whether to include log factor
            var_names: Variable names tuple

        Returns:
            The xy coefficient of the integrated bracket
        """
        # Get bitmask for xy term
        x_bit = 1 << var_names.index(var_names[0])
        y_bit = 1 << var_names.index(var_names[1])
        xy_mask = x_bit | y_bit

        # Integrate the xy coefficient over t
        xy_coeff_integral = 0.0
        for t, w in zip(self._t_nodes, self._t_weights):
            integrand = self.evaluate_integrand_at_t(
                t, var_names, Q_poly, include_log_factor
            )
            # Extract xy coefficient
            xy_coeff = integrand.coeffs.get(xy_mask, 0.0)
            if isinstance(xy_coeff, np.ndarray):
                xy_coeff = float(xy_coeff)
            xy_coeff_integral += xy_coeff * w

        return xy_coeff_integral

    def compute_bracket_result(
        self,
        Q_poly=None,
        include_log_factor: bool = True,
        var_names: Tuple[str, ...] = ("x", "y")
    ) -> BracketEvaluationResult:
        """
        Compute the full bracket evaluation result with diagnostics.

        Args:
            Q_poly: Optional Q polynomial
            include_log_factor: Whether to include log factor
            var_names: Variable names tuple

        Returns:
            BracketEvaluationResult with diagnostics
        """
        # Evaluate scalar limit
        scalar_limit = self.evaluate_scalar_integral()
        analytic = przz_scalar_limit(self.R)

        # Evaluate xy coefficient at each t
        xy_mask = (1 << var_names.index(var_names[0])) | (1 << var_names.index(var_names[1]))

        t_values = np.array(self._t_nodes)
        integrand_values = np.zeros(len(self._t_nodes))

        for i, t in enumerate(self._t_nodes):
            integrand = self.evaluate_integrand_at_t(
                t, var_names, Q_poly, include_log_factor
            )
            xy_coeff = integrand.coeffs.get(xy_mask, 0.0)
            if isinstance(xy_coeff, np.ndarray):
                xy_coeff = float(xy_coeff)
            integrand_values[i] = xy_coeff

        # Integrate
        integrated_value = np.sum(integrand_values * self._t_weights)

        # Compute relative error on scalar limit
        if abs(analytic) > 1e-15:
            rel_error = abs(scalar_limit - analytic) / abs(analytic)
        else:
            rel_error = abs(scalar_limit - analytic)

        return BracketEvaluationResult(
            scalar_limit=scalar_limit,
            t_values=t_values,
            integrand_values=integrand_values,
            integrated_value=integrated_value,
            analytic_expectation=analytic,
            relative_error=rel_error,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_difference_quotient_evaluator(
    benchmark: str = "kappa"
) -> DifferenceQuotientBracket:
    """
    Create a DifferenceQuotientBracket for a specific benchmark.

    Args:
        benchmark: "kappa" or "kappa_star"

    Returns:
        Configured DifferenceQuotientBracket
    """
    if benchmark == "kappa":
        R = 1.3036
    elif benchmark == "kappa_star":
        R = 1.1167
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return DifferenceQuotientBracket(theta=4.0 / 7.0, R=R)


def run_scalar_gate_test(
    R: float = 1.3036,
    n_quad: int = 50,
    tol: float = 1e-10
) -> bool:
    """
    Run the scalar gate test for the difference quotient identity.

    This verifies:
        (exp(2R) - 1) / (2R) == ∫₀¹ exp(2Rt) dt

    Args:
        R: PRZZ R parameter
        n_quad: Number of quadrature points
        tol: Tolerance for identity verification

    Returns:
        True if the identity holds within tolerance
    """
    analytic, quadrature, rel_error = verify_przz_scalar_limit(R, n_quad)
    return rel_error < tol


# =============================================================================
# MAIN EXECUTION
# =============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("DIFFERENCE QUOTIENT BRACKET VERIFICATION")
    print("=" * 70)
    print()

    # Test both benchmarks
    for benchmark in ["kappa", "kappa_star"]:
        print(f"\n--- {benchmark.upper()} ---")
        bracket = create_difference_quotient_evaluator(benchmark)

        # Scalar gate test
        passed, analytic, quadrature, rel_error = bracket.verify_scalar_identity()
        status = "PASS" if passed else "FAIL"
        print(f"Scalar identity: {status}")
        print(f"  Analytic:    {analytic:.10f}")
        print(f"  Quadrature:  {quadrature:.10f}")
        print(f"  Rel error:   {rel_error:.2e}")

        # xy coefficient test (Q=1)
        result = bracket.compute_bracket_result(Q_poly=None, include_log_factor=True)
        print(f"\nBracket evaluation (Q=1):")
        print(f"  Scalar limit:     {result.scalar_limit:.10f}")
        print(f"  Analytic expect:  {result.analytic_expectation:.10f}")
        print(f"  xy coefficient:   {result.integrated_value:.10f}")
