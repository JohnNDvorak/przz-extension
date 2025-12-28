"""
src/unified_s12/logfactor_split.py
Phase 35A: Instrument the unified bracket to split main vs cross contributions.

Goal: Compute the correction factor as:
  corr(K, R) = [(1/θ)×F_xy(0,0) + F_x(0,0) + F_y(0,0)] / [(1/θ)×F_xy(0,0)]
             = 1 + θ × (F_x + F_y) / F_xy

This is the exact product-rule split from PRZZ I₁ formula (line 1530):
  d²/dxdy [(1/θ + x + y) × F(x,y)] = (1/θ)×F_xy + F_x + F_y

The predicted correction is 1 + θ/(2K(2K+1)) = 1 + θ×Beta(2, 2K).

This module provides functions to:
1. Extract F_xy, F_x, F_y from the unified S12 integrand
2. Compute the correction factor directly from these
3. Compare to the theoretical prediction

Created: 2025-12-26 (Phase 35A)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class LogFactorSplit:
    """Result of splitting the log factor contribution.

    The log factor (1/θ + x + y) multiplies F(x,y). Under d²/dxdy:

        d²/dxdy [(1/θ + x + y) × F] = (1/θ)×F_xy + F_y + F_x

    Where:
        - (1/θ)×F comes from the constant term, contributing (1/θ)×F_xy
        - x×F contributes d²/dxdy(x×F) = d/dy(F + x×F_x) = F_y + x×F_xy → F_y at (0,0)
        - y×F contributes d²/dxdy(y×F) = d/dx(F + y×F_y) = F_x + y×F_xy → F_x at (0,0)

    So:
        cross_from_x_term = F_y  (the derivative of the x-dependent factor)
        cross_from_y_term = F_x  (the derivative of the y-dependent factor)
    """

    # The three coefficient components at x=y=0
    main_coeff: float            # (1/θ) × F_xy(0,0) - from constant term in log factor
    cross_from_x_term: float     # F_y(0,0) - from x term in log factor
    cross_from_y_term: float     # F_x(0,0) - from y term in log factor

    # Parameters
    theta: float
    R: float
    K: int
    pair_key: str          # e.g., "11", "22", "12"

    @property
    def total_coeff(self) -> float:
        """Main + cross terms."""
        return self.main_coeff + self.cross_from_x_term + self.cross_from_y_term

    @property
    def correction_factor(self) -> float:
        """
        Correction factor = total / main.

        This should equal 1 + θ × (F_x + F_y) / F_xy
        and should match 1 + θ/(2K(2K+1)) if the Beta moment derivation is correct.
        """
        if abs(self.main_coeff) < 1e-15:
            return float('nan')
        return self.total_coeff / self.main_coeff

    @property
    def predicted_correction(self) -> float:
        """Theoretical prediction: 1 + θ/(2K(2K+1))."""
        return 1 + self.theta / (2 * self.K * (2 * self.K + 1))

    @property
    def correction_gap(self) -> float:
        """Gap between measured and predicted correction, as percentage."""
        return (self.correction_factor / self.predicted_correction - 1) * 100


def split_logfactor_coeffs_from_series(
    F_series: Dict[Tuple[int, ...], float],
    var_names: List[str],
    theta: float,
) -> Tuple[float, float, float]:
    """
    Extract main and cross coefficients from a bivariate series F(x,y).

    The log factor split uses the identity:
        coeff of (x^a × y^b) in [(1/θ + x + y) × F(x,y)]
        = (1/θ) × F_{a,b} + F_{a-1,b} + F_{a,b-1}

    For the I₁ contribution at x=y=0 (a=b=1 for d²/dxdy):
        main_coeff = (1/θ) × F_{1,1}
        cross_from_x = F_{0,1} = F_y  (from the x term)
        cross_from_y = F_{1,0} = F_x  (from the y term)

    Args:
        F_series: Dictionary mapping variable exponent tuples to coefficients
        var_names: List of variable names ["x", "y"]
        theta: The θ parameter

    Returns:
        (main_coeff, cross_from_x_term, cross_from_y_term)
    """
    assert len(var_names) == 2, "Expected bivariate series"

    # For d²/dxdy extraction, we need:
    # - F_{1,1}: coefficient of x¹y¹
    # - F_{0,1}: coefficient of x⁰y¹ = y → this is F_y, comes from x term
    # - F_{1,0}: coefficient of x¹y⁰ = x → this is F_x, comes from y term

    F_11 = F_series.get((1, 1), 0.0)
    F_01 = F_series.get((0, 1), 0.0)  # F_y
    F_10 = F_series.get((1, 0), 0.0)  # F_x

    main_coeff = F_11 / theta
    cross_from_x = F_01  # F_y, from the x term in log factor
    cross_from_y = F_10  # F_x, from the y term in log factor

    return main_coeff, cross_from_x, cross_from_y


def compute_correction_from_split(
    main_coeff: float,
    cross_from_x: float,
    cross_from_y: float,
    theta: float,
    K: int,
) -> Dict[str, float]:
    """
    Compute correction factor and compare to theory.

    Args:
        main_coeff: (1/θ) × F_xy(0,0)
        cross_from_x: F_y(0,0) - from the x term in log factor
        cross_from_y: F_x(0,0) - from the y term in log factor
        theta: θ parameter
        K: Number of mollifier pieces

    Returns:
        Dictionary with correction metrics
    """
    total = main_coeff + cross_from_x + cross_from_y
    measured_correction = total / main_coeff if abs(main_coeff) > 1e-15 else float('nan')

    predicted = 1 + theta / (2 * K * (2 * K + 1))
    gap_pct = (measured_correction / predicted - 1) * 100 if not np.isnan(measured_correction) else float('nan')

    # What the ratio (F_x + F_y) / F_xy should be for the Beta moment to hold
    # Note: cross_from_x = F_y, cross_from_y = F_x
    cross_total = cross_from_x + cross_from_y  # = F_x + F_y
    # main_coeff = (1/θ) × F_xy, so F_xy = theta × main_coeff
    F_xy = theta * main_coeff
    ratio_FxFy_over_Fxy = cross_total / F_xy if abs(F_xy) > 1e-15 else float('nan')

    # This ratio should equal Beta(2, 2K) = 1/(2K(2K+1))
    predicted_ratio = 1 / (2 * K * (2 * K + 1))
    ratio_gap_pct = (ratio_FxFy_over_Fxy / predicted_ratio - 1) * 100 if not np.isnan(ratio_FxFy_over_Fxy) else float('nan')

    return {
        "main_coeff": main_coeff,
        "cross_from_x": cross_from_x,  # F_y
        "cross_from_y": cross_from_y,  # F_x
        "total": total,
        "measured_correction": measured_correction,
        "predicted_correction": predicted,
        "gap_pct": gap_pct,
        "ratio_FxFy_over_Fxy": ratio_FxFy_over_Fxy,
        "predicted_ratio_beta": predicted_ratio,
        "ratio_gap_pct": ratio_gap_pct,
    }


def evaluate_term_with_split(
    term,
    polynomials: Dict,
    n_quad: int,
    theta: float,
    R: float,
) -> Dict[str, float]:
    r"""
    Evaluate a term and extract coefficient breakdown for log factor split.

    CRITICAL: We extract coefficients BEFORE applying the algebraic_prefactor
    (which contains the log factor), so we get the pure F series.

    For a general (ℓ₁, ℓ₂) pair with variables x₁,...,xₗ₁, y₁,...,yₗ₂:
    - Log factor: (1/θ + Σxᵢ + Σyⱼ)
    - Derivative: ∂^{ℓ₁+ℓ₂}/∂x₁...∂xₗ₁∂y₁...∂yₗ₂

    Product rule gives:
    - main = (1/θ) × F_{all_vars}
    - cross_from_x = Σᵢ F_{all_vars \ xᵢ}  (each x term contributes one derivative)
    - cross_from_y = Σⱼ F_{all_vars \ yⱼ}  (each y term contributes one derivative)

    The correction factor is:
        corr = [main + cross] / main = 1 + θ × (cross_x + cross_y) / F_{all}

    Args:
        term: The Term to evaluate
        polynomials: Dict mapping poly_name to polynomial object
        n_quad: Number of quadrature points per dimension
        theta: θ parameter
        R: R parameter

    Returns:
        Dict with coefficient values integrated over the quadrature grid
    """
    from src.quadrature import tensor_grid_2d
    from src.series import TruncatedSeries

    # Build quadrature grid
    U, T, W = tensor_grid_2d(n_quad)

    # Create series context
    ctx = term.create_context()

    # Build integrand from formal-var-dependent factors
    if len(term.vars) > 0:
        integrand = ctx.scalar_series(np.ones_like(U))
    else:
        integrand = ctx.scalar_series(np.ones_like(U))

    # Multiply in poly factors
    for factor in term.poly_factors:
        poly = polynomials.get(factor.poly_name)
        if poly is None:
            raise ValueError(f"Polynomial '{factor.poly_name}' not found")
        factor_series = factor.evaluate(poly, U, T, ctx, R=R, theta=theta)
        integrand = integrand * factor_series

    # Multiply in exp factors
    for factor in term.exp_factors:
        factor_series = factor.evaluate(U, T, ctx)
        integrand = integrand * factor_series

    # DO NOT multiply by algebraic_prefactor here!
    # We want the pure F series, not (log_factor) × F

    # Extract ALL coefficients from the PURE F series (before log factor)
    var_names = list(term.vars)
    n_vars = len(var_names)

    results = {}

    # Split variables into x-group and y-group
    x_vars = [v for v in var_names if v.startswith('x')]
    y_vars = [v for v in var_names if v.startswith('y')]

    # Extract F_{all_vars} - the full derivative coefficient
    F_all = np.asarray(integrand.extract(tuple(var_names)))
    if F_all.shape == ():
        F_all = np.full_like(W, float(F_all))

    # Extract F_{all_vars \ xᵢ} for each x variable
    cross_from_x_total = np.zeros_like(W)
    for x_var in x_vars:
        remaining = tuple(v for v in var_names if v != x_var)
        F_missing_x = np.asarray(integrand.extract(remaining))
        if F_missing_x.shape == ():
            F_missing_x = np.full_like(W, float(F_missing_x))
        cross_from_x_total = cross_from_x_total + F_missing_x

    # Extract F_{all_vars \ yⱼ} for each y variable
    cross_from_y_total = np.zeros_like(W)
    for y_var in y_vars:
        remaining = tuple(v for v in var_names if v != y_var)
        F_missing_y = np.asarray(integrand.extract(remaining))
        if F_missing_y.shape == ():
            F_missing_y = np.full_like(W, float(F_missing_y))
        cross_from_y_total = cross_from_y_total + F_missing_y

    # Apply numeric prefactor
    prefactor = term.numeric_prefactor

    # Apply poly_prefactors (e.g., (1-u)²)
    for prefactor_func in term.poly_prefactors:
        prefactor_vals = prefactor_func(U, T)
        F_all = F_all * prefactor_vals
        cross_from_x_total = cross_from_x_total * prefactor_vals
        cross_from_y_total = cross_from_y_total * prefactor_vals

    # Integrate to get scalar values
    results["F_xy"] = float(np.sum(W * F_all)) * prefactor
    results["F_x"] = float(np.sum(W * cross_from_y_total)) * prefactor  # F_x comes from y-term
    results["F_y"] = float(np.sum(W * cross_from_x_total)) * prefactor  # F_y comes from x-term

    # Store additional info
    results["n_vars"] = n_vars
    results["x_vars"] = x_vars
    results["y_vars"] = y_vars

    return results


def split_logfactor_for_pair(
    pair_key: str,
    theta: float,
    R: float,
    K: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> LogFactorSplit:
    """
    Compute the log factor split for a specific (ℓ₁, ℓ₂) pair.

    This integrates the I₁ integrand numerically and extracts F_xy, F_x, F_y.

    Args:
        pair_key: Pair identifier like "11", "22", "12"
        theta: θ parameter
        R: R parameter
        K: Number of mollifier pieces
        polynomials: Dict with P1, P2, P3, Q
        n_quad: Quadrature points

    Returns:
        LogFactorSplit with coefficient breakdown
    """
    from src.terms_k3_d1 import make_all_terms_k3

    # Get the I₁ term for this pair (index 0 in the term list)
    terms = make_all_terms_k3(theta, R, kernel_regime="paper")
    I1_term = terms[pair_key][0]

    # Evaluate with coefficient extraction
    coeffs = evaluate_term_with_split(I1_term, polynomials, n_quad, theta, R)

    # Compute main and cross coefficients
    # The log factor is (1/θ + x + y) multiplying F
    #
    # Product rule for d²/dxdy [(1/θ + x + y) × F]:
    #   = (1/θ) × F_xy          [from (1/θ)×F]
    #   + F_y                   [from x×F: d²/dxdy(x×F) = F_y at x=y=0]
    #   + F_x                   [from y×F: d²/dxdy(y×F) = F_x at x=y=0]
    #
    # So: main = (1/θ) × F_xy, cross = F_x + F_y
    F_xy = coeffs.get("F_xy", 0.0)
    F_x = coeffs.get("F_x", 0.0)
    F_y = coeffs.get("F_y", 0.0)

    main_coeff = F_xy / theta
    # Cross terms: x×F contributes F_y, y×F contributes F_x
    cross_from_x = F_y  # From the x term in log factor
    cross_from_y = F_x  # From the y term in log factor

    return LogFactorSplit(
        main_coeff=main_coeff,
        cross_from_x_term=cross_from_x,
        cross_from_y_term=cross_from_y,
        theta=theta,
        R=R,
        K=K,
        pair_key=pair_key,
    )


def compute_aggregate_correction_k3(
    theta: float,
    R: float,
    polynomials: Dict,
    n_quad: int = 60,
) -> Dict[str, float]:
    """
    Compute the aggregate correction factor across all K=3 pairs.

    This sums up the main and cross contributions from all (ℓ₁, ℓ₂) pairs
    weighted by their factorial and symmetry factors.

    NOTE: This function is K=3-specific. The pair weights are hardcoded for K=3.
    For K=4+, use compute_aggregate_correction_general().

    Args:
        theta: θ parameter
        R: R parameter
        polynomials: Polynomial dict
        n_quad: Quadrature points

    Returns:
        Dict with:
        - total_main: Sum of weighted main coefficients
        - total_cross: Sum of weighted cross coefficients
        - measured_correction: (total_main + total_cross) / total_main
        - predicted_correction: 1 + θ/(2K(2K+1))
        - gap_pct: Percentage gap between measured and predicted
    """
    K = 3  # Hardcoded - this function is K=3 specific

    # Pair configuration for K=3
    # Weight = symmetry_factor / (ℓ₁! × ℓ₂!)
    # symmetry_factor = 2 for off-diagonal, 1 for diagonal
    factorial_norm = {
        "11": 1.0,       # 1/(1!×1!) = 1
        "22": 0.25,      # 1/(2!×2!) = 1/4
        "33": 1.0/36.0,  # 1/(3!×3!) = 1/36
        "12": 0.5,       # 1/(1!×2!) = 1/2
        "13": 1.0/6.0,   # 1/(1!×3!) = 1/6
        "23": 1.0/12.0,  # 1/(2!×3!) = 1/12
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    total_main = 0.0
    total_cross_from_x = 0.0
    total_cross_from_y = 0.0

    pair_results = {}

    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        split = split_logfactor_for_pair(pair_key, theta, R, K, polynomials, n_quad)

        weight = factorial_norm[pair_key] * symmetry_factor[pair_key]
        total_main += weight * split.main_coeff
        total_cross_from_x += weight * split.cross_from_x_term
        total_cross_from_y += weight * split.cross_from_y_term

        pair_results[pair_key] = {
            "main": split.main_coeff,
            "cross_from_x": split.cross_from_x_term,  # F_y
            "cross_from_y": split.cross_from_y_term,  # F_x
            "correction": split.correction_factor,
            "weight": weight,
        }

    total_cross = total_cross_from_x + total_cross_from_y
    measured_correction = (total_main + total_cross) / total_main if abs(total_main) > 1e-15 else float('nan')
    predicted_correction = 1 + theta / (2 * K * (2 * K + 1))
    gap_pct = (measured_correction / predicted_correction - 1) * 100

    return {
        "K": K,
        "total_main": total_main,
        "total_cross_from_x": total_cross_from_x,
        "total_cross_from_y": total_cross_from_y,
        "total_cross": total_cross,
        "measured_correction": measured_correction,
        "predicted_correction": predicted_correction,
        "gap_pct": gap_pct,
        "pair_results": pair_results,
    }


def compute_aggregate_correction(
    theta: float,
    R: float,
    K: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> Dict[str, float]:
    """
    Compute aggregate correction - dispatches to K-specific function.

    Currently only K=3 is supported.
    """
    if K != 3:
        raise NotImplementedError(f"compute_aggregate_correction only supports K=3, got K={K}")
    return compute_aggregate_correction_k3(theta, R, polynomials, n_quad)


def print_logfactor_split_report(
    theta: float = 4/7,
    R: float = 1.3036,
    K: int = 3,
) -> None:
    """
    Print a report comparing the log factor split to theoretical prediction.

    Args:
        theta: θ parameter
        R: R parameter
        K: Number of mollifier pieces
    """
    predicted = 1 + theta / (2 * K * (2 * K + 1))
    beta = 1 / (2 * K * (2 * K + 1))

    print("=" * 70)
    print("LOG FACTOR SPLIT ANALYSIS")
    print("=" * 70)
    print()
    print(f"Parameters: θ = {theta:.6f}, R = {R:.4f}, K = {K}")
    print()
    print("THEORETICAL PREDICTION")
    print("-" * 50)
    print(f"  Beta(2, 2K) = 1/(2K(2K+1)) = 1/{2*K*(2*K+1)} = {beta:.8f}")
    print(f"  Correction = 1 + θ × Beta(2, 2K) = {predicted:.8f}")
    print()
    print("PRODUCT RULE STRUCTURE")
    print("-" * 50)
    print("  d²/dxdy [(1/θ + x + y) × F(x,y)] = (1/θ)×F_xy + F_x + F_y")
    print()
    print("  At x=y=0:")
    print("    Main term: (1/θ) × F_xy(0,0)")
    print("    Cross terms: F_x(0,0) + F_y(0,0)")
    print()
    print("  Correction = [Main + Cross] / Main")
    print("             = 1 + θ × (F_x + F_y) / F_xy")
    print()
    print("NEXT STEP: Instrument series engine to extract F_xy, F_x, F_y")
    print("Then compare measured correction to theoretical prediction.")
    print()


if __name__ == "__main__":
    print_logfactor_split_report()
