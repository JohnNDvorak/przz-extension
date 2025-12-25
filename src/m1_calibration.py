"""
src/m1_calibration.py
M1 Calibration Harness - Tool for validating m1 formula.

This module provides utilities to:
1. Solve for the implied m1 from a reference c_target
2. Compare computed m1 against empirical formula
3. Validate m1 formula at new K values (when references exist)

The calibration harness is a HIGH-LEVERAGE safety tool:
- For K=3, it should reproduce exp(R)+5 at both benchmarks
- For K=4 (if a reference target exists), it becomes the validation mechanism

This enables "If someone reports a K=4 target later, we can validate in 10 minutes."

See: docs/K_SAFE_BASELINE_LOCKDOWN.md for detailed history.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from src.m1_policy import M1Policy, M1Mode, m1_formula


@dataclass
class CalibrationResult:
    """Result of m1 calibration."""

    m1_solved: float
    """The m1 value that achieves c = c_target."""

    m1_empirical: float
    """The empirical m1 = exp(R) + (2K-1)."""

    ratio: float
    """m1_solved / m1_empirical (should be ~1.0 if formula is correct)."""

    c_target: float
    """The target c value used for calibration."""

    c_computed: float
    """The c value computed with m1_solved."""

    residual: float
    """c_computed - c_target (should be ~0)."""

    R: float
    """R parameter used."""

    K: int
    """K value used."""


def solve_m1_from_channels(
    c_target: float,
    I12_plus: float,
    I12_minus: float,
    S34_plus: float,
    R: float,
    K: int = 3,
) -> CalibrationResult:
    """
    Solve for m1 from the channel decomposition.

    Given:
        c = I12_plus + m1 * I12_minus + S34_plus

    Solve for m1:
        m1 = (c_target - I12_plus - S34_plus) / I12_minus

    Args:
        c_target: Target c value from PRZZ
        I12_plus: Sum of I1+I2 at +R
        I12_minus: Sum of I1+I2 at -R (base, before m1 scaling)
        S34_plus: Sum of I3+I4 at +R (no mirror)
        R: R parameter
        K: Number of mollifier pieces

    Returns:
        CalibrationResult with solved m1 and validation metrics

    Raises:
        ValueError: If I12_minus is too close to zero
    """
    if abs(I12_minus) < 1e-15:
        raise ValueError(
            f"I12_minus is too small ({I12_minus:.2e}), cannot solve for m1"
        )

    # Solve for m1
    numerator = c_target - I12_plus - S34_plus
    m1_solved = numerator / I12_minus

    # Compute empirical m1 for comparison
    m1_empirical = np.exp(R) + (2 * K - 1)

    # Compute c with solved m1
    c_computed = I12_plus + m1_solved * I12_minus + S34_plus

    return CalibrationResult(
        m1_solved=float(m1_solved),
        m1_empirical=float(m1_empirical),
        ratio=float(m1_solved / m1_empirical),
        c_target=float(c_target),
        c_computed=float(c_computed),
        residual=float(c_computed - c_target),
        R=float(R),
        K=int(K),
    )


def validate_m1_formula_at_k3(
    channels_kappa: Dict[str, float],
    channels_kappa_star: Dict[str, float],
    c_target_kappa: float = 2.137454,
    c_target_kappa_star: float = 1.938,
    R_kappa: float = 1.3036,
    R_kappa_star: float = 1.1167,
    tol: float = 0.05,
) -> Dict[str, CalibrationResult]:
    """
    Validate that m1 = exp(R) + 5 reproduces targets at both benchmarks.

    This is the sanity check: for K=3, the solved m1 should match
    the empirical formula exp(R) + 5 at both kappa and kappa*.

    Args:
        channels_kappa: Dict with "I12_plus", "I12_minus", "S34_plus" for kappa
        channels_kappa_star: Dict with "I12_plus", "I12_minus", "S34_plus" for kappa*
        c_target_kappa: Target c for kappa benchmark (~2.137)
        c_target_kappa_star: Target c for kappa* benchmark (~1.938)
        R_kappa: R for kappa benchmark (1.3036)
        R_kappa_star: R for kappa* benchmark (1.1167)
        tol: Tolerance for ratio validation (default 5%)

    Returns:
        Dict with "kappa" and "kappa_star" CalibrationResult entries

    Raises:
        ValueError: If either benchmark fails validation (ratio outside 1.0 +/- tol)
    """
    # Validate kappa
    result_kappa = solve_m1_from_channels(
        c_target=c_target_kappa,
        I12_plus=channels_kappa["I12_plus"],
        I12_minus=channels_kappa["I12_minus"],
        S34_plus=channels_kappa["S34_plus"],
        R=R_kappa,
        K=3,
    )

    # Validate kappa*
    result_kappa_star = solve_m1_from_channels(
        c_target=c_target_kappa_star,
        I12_plus=channels_kappa_star["I12_plus"],
        I12_minus=channels_kappa_star["I12_minus"],
        S34_plus=channels_kappa_star["S34_plus"],
        R=R_kappa_star,
        K=3,
    )

    # Check ratios
    for name, result in [("kappa", result_kappa), ("kappa*", result_kappa_star)]:
        if abs(result.ratio - 1.0) > tol:
            raise ValueError(
                f"m1 formula validation failed for {name}: "
                f"ratio={result.ratio:.4f} (expected ~1.0 +/- {tol}). "
                f"m1_solved={result.m1_solved:.4f}, "
                f"m1_empirical={result.m1_empirical:.4f}"
            )

    return {
        "kappa": result_kappa,
        "kappa_star": result_kappa_star,
    }


def estimate_m1_for_k4(
    channels_k4: Dict[str, float],
    c_target_k4: float,
    R: float,
) -> CalibrationResult:
    """
    Estimate m1 for K=4 from a reference target.

    If a K=4 target c value is available (e.g., from PRZZ paper),
    this function solves for the implied m1 and compares it to
    the extrapolated formula exp(R) + 7.

    Args:
        channels_k4: Dict with "I12_plus", "I12_minus", "S34_plus" for K=4
        c_target_k4: Target c for K=4 benchmark
        R: R parameter

    Returns:
        CalibrationResult with K=4 validation

    Note:
        If ratio is close to 1.0, the extrapolation formula works!
        If ratio is significantly different, we need a new formula.
    """
    return solve_m1_from_channels(
        c_target=c_target_k4,
        I12_plus=channels_k4["I12_plus"],
        I12_minus=channels_k4["I12_minus"],
        S34_plus=channels_k4["S34_plus"],
        R=R,
        K=4,
    )


def print_calibration_report(result: CalibrationResult, benchmark_name: str = "") -> None:
    """
    Print a formatted calibration report.

    Args:
        result: CalibrationResult to report
        benchmark_name: Optional name (e.g., "kappa", "kappa*")
    """
    header = f"M1 Calibration Report" + (f" ({benchmark_name})" if benchmark_name else "")
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"K = {result.K}, R = {result.R:.4f}")
    print()
    print(f"Target c:     {result.c_target:.6f}")
    print(f"Computed c:   {result.c_computed:.6f}")
    print(f"Residual:     {result.residual:.2e}")
    print()
    print(f"m1 solved:    {result.m1_solved:.6f}")
    print(f"m1 empirical: {result.m1_empirical:.6f}")
    print(f"Ratio:        {result.ratio:.6f}")
    print()

    if abs(result.ratio - 1.0) < 0.05:
        print("Status: VALIDATED (ratio within 5% of 1.0)")
    elif abs(result.ratio - 1.0) < 0.1:
        print("Status: MARGINAL (ratio within 10% of 1.0)")
    else:
        print(f"Status: FAILED (ratio {result.ratio:.2f}x off)")
