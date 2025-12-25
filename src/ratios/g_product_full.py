"""
src/ratios/g_product_full.py
Phase 15A: Full Zeta-Factor Computation at Finite α

PURPOSE:
========
Compute the ACTUAL zeta log-derivative product at the PRZZ evaluation point α=β=-R,
rather than using Laurent approximations.

KEY FINDING (Phase 15A Investigation):
=====================================
The current RAW_LOGDERIV mode uses (1/R + γ)² as an approximation to (ζ'/ζ)(1-R)².
This Laurent approximation has significant error at the PRZZ evaluation points:
- κ (R=1.3036): 22% error
- κ* (R=1.1167): 17% error

The actual numerical values are:
- (ζ'/ζ)(1-R)² actual ≈ 3.00 (κ), 3.16 (κ*)
- (1/R + γ)² Laurent ≈ 1.81 (κ), 2.17 (κ*)

THREE POSSIBLE CORRECTIONS:
==========================
1. ACTUAL_LOGDERIV_SQUARED: Use actual (ζ'/ζ)(1-R)² value
2. FULL_G_PRODUCT: Use G(-R)² = (ζ'/ζ²)(1-R)² [includes 1/ζ² factor]
3. Keep Laurent but with correction factor

The G-product is TOO large (35x for κ, 19x for κ*) - this is NOT the right fix.
The ACTUAL_LOGDERIV_SQUARED (without 1/ζ² factor) gives a ~60% correction.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math

try:
    import mpmath
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

from src.ratios.zeta_laurent import EULER_MASCHERONI


@dataclass(frozen=True)
class ZetaFactorResult:
    """Result of zeta-factor computation at finite α."""
    R: float
    evaluation_point: float  # 1 - R
    zeta_val: float
    zeta_deriv: float

    # Three different J12 factor computations
    logderiv_actual: float  # (ζ'/ζ)(1-R)
    logderiv_actual_squared: float  # (ζ'/ζ)(1-R)²
    logderiv_laurent: float  # 1/R + γ (approximation)
    logderiv_laurent_squared: float  # (1/R + γ)²
    G_value: float  # G(-R) = ζ'(1-R) / ζ(1-R)²
    G_squared: float  # G(-R)²

    # Correction factors
    actual_vs_laurent_ratio: float  # How much larger actual is than Laurent
    laurent_error_percent: float  # % error in Laurent approximation


def compute_zeta_at_point(s: float, precision: int = 50) -> Tuple[float, float]:
    """
    Compute ζ(s) and ζ'(s) at any real value using mpmath.

    Args:
        s: Real value (can be negative)
        precision: mpmath precision in decimal digits

    Returns:
        (zeta_value, zeta_derivative)
    """
    if not MPMATH_AVAILABLE:
        raise ImportError("mpmath required for high-precision zeta computation")

    with mpmath.workdps(precision):
        zeta_val = float(mpmath.zeta(s))
        zeta_deriv = float(mpmath.diff(mpmath.zeta, s))

    return zeta_val, zeta_deriv


def compute_zeta_factors(R: float, precision: int = 50) -> ZetaFactorResult:
    """
    Compute all relevant zeta-factor values at α = -R.

    Args:
        R: PRZZ R parameter
        precision: mpmath precision

    Returns:
        ZetaFactorResult with all computed values
    """
    s = 1.0 - R  # evaluation point
    zeta_val, zeta_deriv = compute_zeta_at_point(s, precision)

    # Actual (ζ'/ζ)(1-R)
    logderiv_actual = zeta_deriv / zeta_val
    logderiv_actual_squared = logderiv_actual ** 2

    # Laurent approximation: (ζ'/ζ)(1+ε) ≈ -1/ε + γ
    # At ε = -R: -1/(-R) + γ = 1/R + γ
    gamma = EULER_MASCHERONI
    logderiv_laurent = 1.0 / R + gamma
    logderiv_laurent_squared = logderiv_laurent ** 2

    # G-product: G = (1/ζ)(ζ'/ζ) = ζ'/ζ²
    G_value = zeta_deriv / (zeta_val ** 2)
    G_squared = G_value ** 2

    # Correction factors
    actual_vs_laurent_ratio = logderiv_actual_squared / logderiv_laurent_squared
    laurent_error = abs(logderiv_actual - logderiv_laurent) / abs(logderiv_actual) * 100

    return ZetaFactorResult(
        R=R,
        evaluation_point=s,
        zeta_val=zeta_val,
        zeta_deriv=zeta_deriv,
        logderiv_actual=logderiv_actual,
        logderiv_actual_squared=logderiv_actual_squared,
        logderiv_laurent=logderiv_laurent,
        logderiv_laurent_squared=logderiv_laurent_squared,
        G_value=G_value,
        G_squared=G_squared,
        actual_vs_laurent_ratio=actual_vs_laurent_ratio,
        laurent_error_percent=laurent_error,
    )


def compute_zeta_logderiv_actual(R: float, precision: int = 50) -> float:
    """
    Compute ACTUAL (ζ'/ζ)(1-R) value, NOT the Laurent approximation.

    Phase 16: This is the SINGLE factor used by J13 and J14.
    J12 uses the SQUARED factor; J13/J14 use the SINGLE factor.

    Args:
        R: PRZZ R parameter
        precision: mpmath precision

    Returns:
        Actual (ζ'/ζ)(1-R) value (single factor, not squared)

    Expected values:
        κ (R=1.3036):  ~1.73  (Laurent ~1.35, error ~29%)
        κ* (R=1.1167): ~1.78  (Laurent ~1.47, error ~21%)
    """
    result = compute_zeta_factors(R, precision)
    return result.logderiv_actual


def compute_j12_actual_logderiv_squared(R: float, precision: int = 50) -> float:
    """
    Compute J12 constant term using ACTUAL (ζ'/ζ)(1-R)² value.

    This is the numerical value of (ζ'/ζ)² at the evaluation point,
    NOT the Laurent approximation (1/R + γ)².

    Args:
        R: PRZZ R parameter
        precision: mpmath precision

    Returns:
        Actual (ζ'/ζ)(1-R)² value
    """
    result = compute_zeta_factors(R, precision)
    return result.logderiv_actual_squared


def compute_j12_full_G_product(R: float, precision: int = 50) -> float:
    """
    Compute J12 constant term using FULL G-product at α=β=-R.

    Returns G(-R)² = [(1/ζ)(ζ'/ζ)(1-R)]²

    WARNING: This gives values 10-20x larger than expected!
    The G-product (with 1/ζ² factor) is likely NOT the correct formula.

    Args:
        R: PRZZ R parameter
        precision: mpmath precision

    Returns:
        G(-R)² value
    """
    result = compute_zeta_factors(R, precision)
    return result.G_squared


def get_laurent_correction_factor(R: float, precision: int = 50) -> float:
    """
    Get the correction factor to multiply Laurent approximation by.

    actual_value = laurent_value × correction_factor

    Args:
        R: PRZZ R parameter
        precision: mpmath precision

    Returns:
        Correction factor
    """
    result = compute_zeta_factors(R, precision)
    return result.actual_vs_laurent_ratio


def print_zeta_factor_analysis(R: float = 1.3036, precision: int = 50):
    """Print detailed zeta-factor analysis for debugging."""
    result = compute_zeta_factors(R, precision)

    print("=" * 70)
    print(f"PHASE 15A: ZETA-FACTOR ANALYSIS (R={R})")
    print("=" * 70)
    print()
    print("EVALUATION POINT:")
    print(f"  s = 1 - R = {result.evaluation_point:.6f}")
    print()
    print("ZETA VALUES at s:")
    print(f"  ζ(s) = {result.zeta_val:.10f}")
    print(f"  ζ'(s) = {result.zeta_deriv:.10f}")
    print()
    print("LOG-DERIVATIVE (ζ'/ζ):")
    print(f"  Actual (ζ'/ζ)(1-R) = {result.logderiv_actual:.10f}")
    print(f"  Laurent (1/R + γ) = {result.logderiv_laurent:.10f}")
    print(f"  Laurent error: {result.laurent_error_percent:.2f}%")
    print()
    print("SQUARED VALUES (what J12 uses):")
    print(f"  Actual (ζ'/ζ)²: {result.logderiv_actual_squared:.10f}")
    print(f"  Laurent (1/R+γ)²: {result.logderiv_laurent_squared:.10f}")
    print(f"  Ratio actual/Laurent: {result.actual_vs_laurent_ratio:.6f}")
    print()
    print("G-PRODUCT (includes 1/ζ² factor):")
    print(f"  G(-R) = {result.G_value:.10f}")
    print(f"  G(-R)² = {result.G_squared:.10f}")
    print(f"  G²/Laurent ratio = {result.G_squared/result.logderiv_laurent_squared:.6f}")
    print()

    print("RECOMMENDATION:")
    correction = (result.actual_vs_laurent_ratio - 1.0) * 100
    if abs(correction) > 5:
        print(f"  Laurent approximation is off by {correction:+.1f}%")
        print(f"  Consider using ACTUAL_LOGDERIV_SQUARED mode")
    else:
        print(f"  Laurent approximation is reasonably accurate ({correction:+.1f}%)")

    G_ratio = result.G_squared / result.logderiv_laurent_squared
    print(f"  G-product is {G_ratio:.1f}x larger than Laurent - TOO LARGE!")
    print()
    print("=" * 70)
    return result


def run_both_benchmarks(precision: int = 50) -> Dict[str, ZetaFactorResult]:
    """
    Run zeta-factor analysis for both PRZZ benchmarks.

    Returns:
        Dict with 'kappa' and 'kappa_star' results
    """
    results = {}

    print("\n" + "=" * 70)
    print("PHASE 15A: ZETA-FACTOR COMPARISON - BOTH BENCHMARKS")
    print("=" * 70 + "\n")

    # κ benchmark (R = 1.3036)
    results['kappa'] = compute_zeta_factors(1.3036, precision)

    # κ* benchmark (R = 1.1167)
    results['kappa_star'] = compute_zeta_factors(1.1167, precision)

    # Summary table
    print("-" * 80)
    print(f"{'Benchmark':<12} {'R':<8} {'Actual²':<14} {'Laurent²':<14} "
          f"{'Ratio':<10} {'Error%':<10}")
    print("-" * 80)

    for name, res in results.items():
        print(f"{name:<12} {res.R:<8.4f} {res.logderiv_actual_squared:<14.6f} "
              f"{res.logderiv_laurent_squared:<14.6f} "
              f"{res.actual_vs_laurent_ratio:<10.4f} "
              f"{res.laurent_error_percent:<10.2f}")

    print("-" * 80)

    # Key question: if we use ACTUAL instead of LAURENT, what's the effect?
    print("\nPREDICTED EFFECT ON +5 GATE:")
    print("-" * 40)

    for name, res in results.items():
        # The +5 gate currently shows ~5% gap
        # If Laurent is off by 66%/46%, how does this affect δ?
        correction = res.actual_vs_laurent_ratio
        print(f"{name}: actual/Laurent = {correction:.4f} ({(correction-1)*100:+.1f}%)")

    print()
    print("NOTE: If the formula should use actual (ζ'/ζ)² instead of")
    print("      Laurent (1/R+γ)², the effect on δ would be significant.")
    print("      However, the DIRECTION of correction needs investigation.")
    print()

    return results


if __name__ == "__main__":
    # Run full analysis
    if MPMATH_AVAILABLE:
        print_zeta_factor_analysis(R=1.3036)
        print()
        print_zeta_factor_analysis(R=1.1167)
        print()
        run_both_benchmarks()
    else:
        print("ERROR: mpmath not available. Install with: pip install mpmath")
