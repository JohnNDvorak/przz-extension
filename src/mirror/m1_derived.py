"""
src/mirror/m1_derived.py
Phase 18.4: Derived m1 Computation

PURPOSE:
========
Attempt to derive m1 from paper formulas, NOT fitted to benchmarks.

CURRENT STATUS:
===============
The derivation is incomplete. The paper's T^{-α-β} factor gives values
much larger than the empirical m1 = exp(R) + 5:

- PRZZ T^{-α-β} at α=β=-R: T^{2R} → exp(2R/θ) ≈ exp(3.55) ≈ 35 for κ
- Empirical m1: exp(R) + 5 ≈ 8.7 for κ

This 4x discrepancy suggests either:
1. The T^{-α-β} interpretation is wrong for our finite-R regime
2. There's additional normalization that cancels part of the factor
3. The mirror assembly formula needs refinement

For now, we keep empirical as the default and provide infrastructure
for testing derivation hypotheses.

DESIGN PRINCIPLE:
=================
- Empirical mode remains the default (validated at K=3)
- Derived modes are EXPERIMENTAL until they pass the two-benchmark gate
- No silent fallback - if derivation fails, it should be explicit

USAGE:
======
>>> from src.mirror.m1_derived import compare_m1_modes
>>> comparison = compare_m1_modes(R=1.3036)
>>> # Shows empirical, fitted, and various derivation attempts
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional
import numpy as np

from src.m1_policy import M1_FITTED_COEFFICIENT_A, M1_FITTED_COEFFICIENT_B


class M1DerivationMode(Enum):
    """Modes for m1 computation."""

    EMPIRICAL = auto()
    """Empirical formula: exp(R) + (2K-1). Validated at K=3."""

    FITTED = auto()
    """Fitted formula: A*exp(R) + B. Achieves 0% gap but is calibration."""

    NAIVE_PAPER = auto()
    """Naive paper formula: exp(2R). From T^{-α-β} at α=β=-R, ignoring θ."""

    PRZZ_LIMIT = auto()
    """PRZZ limit formula: exp(2R/θ). Full T^{-α-β} scaling."""

    TAYLOR_CORRECTION = auto()
    """Empirical with Taylor correction: exp(R) + 5 + O(R²) terms."""

    # Phase 19.4.2 candidates - no fitting allowed
    UNIFORM_AVG_EXP_2RT = auto()
    """Simple average: ∫₀¹ exp(2Rt) dt = (exp(2R) - 1)/(2R)."""

    E_EXP_2RT_UNDER_Q2 = auto()
    """Weighted expectation: E[exp(2Rt)] under Q(t)² weight."""

    SINH_SCALED = auto()
    """Sinh-based: 2*sinh(R)/R. Symmetric in ±R."""


@dataclass(frozen=True)
class M1DerivedResult:
    """Result of derived m1 computation."""

    m1_value: float
    """Computed m1 value."""

    derivation_mode: str
    """Mode used for derivation."""

    m1_empirical: float
    """Reference: exp(R) + (2K-1)."""

    m1_fitted: float
    """Reference: A*exp(R) + B."""

    ratio_to_empirical: float
    """m1_value / m1_empirical."""

    ratio_to_fitted: float
    """m1_value / m1_fitted."""

    R: float
    K: int
    theta: float


def m1_empirical_formula(R: float, K: int) -> float:
    """
    Compute empirical m1 = exp(R) + (2K-1).

    This is the VALIDATED formula for K=3.
    """
    return np.exp(R) + (2 * K - 1)


def m1_fitted_formula(R: float) -> float:
    """
    Compute fitted m1 = A*exp(R) + B.

    This achieves 0% gap on both benchmarks but is pure calibration.
    """
    return M1_FITTED_COEFFICIENT_A * np.exp(R) + M1_FITTED_COEFFICIENT_B


def m1_naive_paper(R: float) -> float:
    """
    Compute naive paper m1 = exp(2R).

    From T^{-α-β} at α=β=-R, ignoring θ dependence.
    This is TOO LARGE by about 1.5x for κ.
    """
    return np.exp(2 * R)


def m1_przz_limit(R: float, theta: float) -> float:
    """
    Compute PRZZ limit m1 = exp(2R/θ).

    From T^{-α-β} with α=β=-R/L where L = log(T) and T = N^θ.
    This is WAY TOO LARGE (35x for κ).

    The discrepancy suggests this formula applies at asymptotic L,
    not at our finite-R evaluation.
    """
    return np.exp(2 * R / theta)


def m1_taylor_correction(R: float, K: int) -> float:
    """
    Compute empirical with Taylor correction.

    m1 = exp(R) + (2K-1) + correction(R)

    The correction is currently ZERO because we haven't derived it.
    This is a placeholder for future work.
    """
    base = m1_empirical_formula(R, K)
    correction = 0.0  # TODO: derive from paper
    return base + correction


def m1_uniform_avg_exp_2Rt(R: float) -> float:
    """
    Phase 19.4.2 candidate: Simple uniform average.

    ∫₀¹ exp(2Rt) dt = (exp(2R) - 1) / (2R)

    This is the simplest "expected exp(2Rt)" without weighting.
    No fitting involved - pure closed-form integral.
    """
    if abs(R) < 1e-10:
        return 1.0  # Taylor limit at R=0
    return (np.exp(2 * R) - 1) / (2 * R)


def m1_E_exp_2Rt_under_Q2(R: float, Q_coeffs: Optional[tuple] = None) -> float:
    """
    Phase 19.4.2 candidate: Weighted expectation under Q².

    E[exp(2Rt)] where t has weight Q(t)² on [0,1], normalized.

    For Q(t) = c₀ + c₁t + c₂t² + ..., compute:
    ∫₀¹ Q(t)² exp(2Rt) dt / ∫₀¹ Q(t)² dt

    Args:
        R: R parameter
        Q_coeffs: Tuple of Q polynomial coefficients. If None, uses Q(t)=1.

    Returns:
        Weighted expectation of exp(2Rt)
    """
    if Q_coeffs is None or len(Q_coeffs) == 0:
        # Default: Q(t) = 1, so weight is uniform
        return m1_uniform_avg_exp_2Rt(R)

    # Numerical integration for general Q
    from scipy import integrate

    def Q_squared(t):
        val = sum(c * (t ** i) for i, c in enumerate(Q_coeffs))
        return val * val

    def weighted_integrand(t):
        return Q_squared(t) * np.exp(2 * R * t)

    # Compute integrals
    numerator, _ = integrate.quad(weighted_integrand, 0, 1)
    denominator, _ = integrate.quad(Q_squared, 0, 1)

    if abs(denominator) < 1e-15:
        return m1_uniform_avg_exp_2Rt(R)  # Fall back

    return numerator / denominator


def m1_sinh_scaled(R: float) -> float:
    """
    Phase 19.4.2 candidate: Sinh-based formula.

    2 * sinh(R) / R = (exp(R) - exp(-R)) / R

    This is symmetric in ±R and arises from integrating exp(Rt)
    over symmetric intervals or from hyperbolic identities.
    """
    if abs(R) < 1e-10:
        return 2.0  # Taylor limit at R=0
    return 2 * np.sinh(R) / R


def m1_derived(
    R: float,
    K: int,
    theta: float,
    derivation_mode: M1DerivationMode = M1DerivationMode.EMPIRICAL,
    Q_coeffs: Optional[tuple] = None,
) -> float:
    """
    Compute m1 from specified derivation mode.

    Args:
        R: R parameter
        K: Number of pieces
        theta: theta parameter
        derivation_mode: Which derivation to use
        Q_coeffs: Q polynomial coefficients (for E_EXP_2RT_UNDER_Q2 mode)

    Returns:
        Computed m1 value

    Note:
        EMPIRICAL is the default and only validated mode for K=3.
        Other modes are EXPERIMENTAL.
    """
    if derivation_mode == M1DerivationMode.EMPIRICAL:
        return m1_empirical_formula(R, K)
    elif derivation_mode == M1DerivationMode.FITTED:
        return m1_fitted_formula(R)
    elif derivation_mode == M1DerivationMode.NAIVE_PAPER:
        return m1_naive_paper(R)
    elif derivation_mode == M1DerivationMode.PRZZ_LIMIT:
        return m1_przz_limit(R, theta)
    elif derivation_mode == M1DerivationMode.TAYLOR_CORRECTION:
        return m1_taylor_correction(R, K)
    elif derivation_mode == M1DerivationMode.UNIFORM_AVG_EXP_2RT:
        return m1_uniform_avg_exp_2Rt(R)
    elif derivation_mode == M1DerivationMode.E_EXP_2RT_UNDER_Q2:
        return m1_E_exp_2Rt_under_Q2(R, Q_coeffs)
    elif derivation_mode == M1DerivationMode.SINH_SCALED:
        return m1_sinh_scaled(R)
    else:
        raise ValueError(f"Unknown derivation mode: {derivation_mode}")


def m1_derived_with_breakdown(
    R: float,
    K: int = 3,
    theta: float = 4.0 / 7.0,
    derivation_mode: M1DerivationMode = M1DerivationMode.EMPIRICAL,
) -> M1DerivedResult:
    """
    Compute derived m1 with full comparison breakdown.

    Args:
        R: R parameter
        K: Number of pieces
        theta: theta parameter
        derivation_mode: Mode to use

    Returns:
        M1DerivedResult with value and comparisons
    """
    m1_value = m1_derived(R, K, theta, derivation_mode)
    m1_emp = m1_empirical_formula(R, K)
    m1_fit = m1_fitted_formula(R)

    return M1DerivedResult(
        m1_value=float(m1_value),
        derivation_mode=derivation_mode.name,
        m1_empirical=float(m1_emp),
        m1_fitted=float(m1_fit),
        ratio_to_empirical=float(m1_value / m1_emp),
        ratio_to_fitted=float(m1_value / m1_fit),
        R=float(R),
        K=K,
        theta=float(theta),
    )


def compare_m1_modes(
    R: float,
    K: int = 3,
    theta: float = 4.0 / 7.0,
    verbose: bool = True,
) -> Dict[str, M1DerivedResult]:
    """
    Compare all m1 computation modes.

    Args:
        R: R parameter
        K: Number of pieces
        theta: theta parameter
        verbose: Print comparison table

    Returns:
        Dict with results for each mode
    """
    results = {}
    for mode in M1DerivationMode:
        results[mode.name] = m1_derived_with_breakdown(R, K, theta, mode)

    if verbose:
        print_m1_comparison(results, R, K, theta)

    return results


def print_m1_comparison(
    results: Dict[str, M1DerivedResult],
    R: float,
    K: int,
    theta: float,
) -> None:
    """Print formatted m1 comparison table."""
    print()
    print("=" * 70)
    print(f"M1 MODE COMPARISON (R={R}, K={K}, theta={theta:.4f})")
    print("=" * 70)
    print()

    m1_emp = results["EMPIRICAL"].m1_empirical
    m1_fit = results["EMPIRICAL"].m1_fitted

    print(f"Reference values:")
    print(f"  EMPIRICAL (exp(R)+5): {m1_emp:.6f}")
    print(f"  FITTED (A*exp(R)+B):  {m1_fit:.6f}")
    print()

    print("-" * 70)
    print(f"{'Mode':<20} {'m1 value':<12} {'vs emp':<10} {'vs fit':<10} {'Status':<15}")
    print("-" * 70)

    for mode_name, result in results.items():
        ratio_emp = result.ratio_to_empirical
        ratio_fit = result.ratio_to_fitted

        # Status based on closeness to empirical
        if abs(ratio_emp - 1.0) < 0.01:
            status = "MATCH"
        elif abs(ratio_emp - 1.0) < 0.05:
            status = "CLOSE"
        elif ratio_emp > 2.0:
            status = "TOO LARGE"
        elif ratio_emp < 0.5:
            status = "TOO SMALL"
        else:
            status = "OFF"

        print(
            f"{mode_name:<20} {result.m1_value:<12.4f} "
            f"{ratio_emp:<10.4f} {ratio_fit:<10.4f} {status:<15}"
        )

    print("-" * 70)
    print()

    # Key finding
    przz = results["PRZZ_LIMIT"]
    print("KEY FINDING:")
    print(f"  PRZZ limit exp(2R/θ) = {przz.m1_value:.2f} is {przz.ratio_to_empirical:.1f}x empirical")
    print("  This discrepancy indicates the limit formula doesn't apply at finite R.")
    print("  The empirical formula exp(R)+5 remains the best option for production.")
    print()


def run_both_benchmarks_comparison(verbose: bool = True) -> None:
    """Run m1 comparison for both PRZZ benchmarks."""
    benchmarks = [
        ("kappa", 1.3036),
        ("kappa_star", 1.1167),
    ]

    for name, R in benchmarks:
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {name.upper()}")
        compare_m1_modes(R, K=3, theta=4.0 / 7.0, verbose=verbose)


if __name__ == "__main__":
    run_both_benchmarks_comparison(verbose=True)
