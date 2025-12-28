"""
src/evaluator/correction_policy.py
Phase 45.1/46.0: Correction Policy Infrastructure with Anchoring Guard

This module provides explicit control over which correction mode is used,
preventing "quiet calibration creep" by making anchoring explicit.

CORRECTION MODES:
=================

1. DERIVED_BASELINE_ONLY (DEFAULT):
   - Uses g = 1 + θ/(2K(2K+1)) uniformly for all integrals
   - Uses base = exp(R) + (2K-1) from difference quotient (Phase 32)
   - NO benchmark-anchored constants
   - Gives ±0.15% gap on κ/κ* benchmarks
   - THIS IS THE DEFAULT - fully first-principles

2. FIRST_PRINCIPLES_I1_I2:
   - Uses g_I1 = 1.0 (log factor cross-terms self-correct)
   - Uses g_I2 = 1 + θ/(2K(2K+1)) (full Beta moment for I2)
   - NO benchmark-anchored constants
   - Gives ~0.4% gap on κ/κ* benchmarks
   - More accurate than uniform baseline, still first-principles

3. ANCHORED_TWO_BENCHMARKS:
   - Uses calibrated g_I1 = 1.00091428, g_I2 = 1.01945154
   - These were obtained by solving 2-benchmark system (κ, κ* targets)
   - Gives ~0% gap on κ/κ* benchmarks
   - **REQUIRES allow_target_anchoring=True** to use
   - This is NOT first-principles - it's curve-fitting

ANCHORING GUARD (Phase 46.0):
=============================

The `get_g_correction()` function has an `allow_target_anchoring` parameter
that defaults to False. Attempting to use ANCHORED_TWO_BENCHMARKS without
explicitly setting allow_target_anchoring=True will raise ValueError.

This prevents accidental use of calibrated constants, ensuring that production
code paths explicitly acknowledge when they're using non-derived values.

USAGE:
======

from src.evaluator.correction_policy import CorrectionMode, get_g_correction

# Default (derived baseline)
result = get_g_correction(R=1.3036, theta=4/7, K=3)

# First-principles I1/I2 (more accurate, still derived)
result = get_g_correction(
    R=1.3036, theta=4/7, K=3, f_I1=0.233,
    mode=CorrectionMode.FIRST_PRINCIPLES_I1_I2
)

# Anchored mode (requires explicit opt-in)
result = get_g_correction(
    R=1.3036, theta=4/7, K=3, f_I1=0.233,
    mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
    allow_target_anchoring=True  # REQUIRED
)

Created: 2025-12-27 (Phase 45.1)
Updated: 2025-12-27 (Phase 46.0 - Added anchoring guard)
"""

from enum import Enum
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


class CorrectionMode(Enum):
    """Correction mode for mirror multiplier computation."""

    DERIVED_BASELINE_ONLY = "derived_baseline_only"
    """
    Uses purely derived formula: g = 1 + θ/(2K(2K+1)) uniformly.
    No benchmark-anchored constants.
    Gives ±0.15% gap on κ/κ*.
    """

    FIRST_PRINCIPLES_I1_I2 = "first_principles_i1_i2"
    """
    Uses derived formula with I1/I2 differentiation:
      g_I1 = 1.0 (log factor cross-terms self-correct)
      g_I2 = 1 + θ/(2K(2K+1)) (full Beta moment for I2)
    No benchmark-anchored constants.
    Gives ~0.4% gap on κ/κ*.
    """

    ANCHORED_TWO_BENCHMARKS = "anchored_two_benchmarks"
    """
    Uses calibrated g_I1, g_I2 from 2-benchmark solve.
    REQUIRES allow_target_anchoring=True to use.
    Gives ~0% gap on κ/κ*.
    NOT first-principles - this is curve-fitting.
    """

    # Legacy alias for backwards compatibility (deprecated)
    COMPONENT_RENORM_ANCHORED = "anchored_two_benchmarks"
    """DEPRECATED: Use ANCHORED_TWO_BENCHMARKS instead."""


# Calibrated constants from Phase 45 (2-benchmark solve)
# These are NOT derived from first principles
G_I1_CALIBRATED = 1.00091428
G_I2_CALIBRATED = 1.01945154


@dataclass
class CorrectionResult:
    """Result of correction computation."""
    g: float                  # The g correction factor
    base: float               # The base term: exp(R) + (2K-1)
    m: float                  # Full multiplier: g × base
    mode: CorrectionMode      # Which mode was used
    g_baseline: float         # The derived baseline (for comparison)

    # Only set for COMPONENT_RENORM_ANCHORED mode
    g_I1: float = None        # Calibrated g for I1
    g_I2: float = None        # Calibrated g for I2
    f_I1: float = None        # I1 fraction used


def compute_g_baseline(theta: float, K: int) -> float:
    """
    Compute the derived g baseline from Beta moment.

    Formula: g = 1 + θ/(2K(2K+1))

    This is FULLY DERIVED from first principles (Phase 34C).
    """
    return 1 + theta / (2 * K * (2 * K + 1))


def compute_base(R: float, K: int) -> float:
    """
    Compute the base term from difference quotient.

    Formula: base = exp(R) + (2K-1)

    This is FULLY DERIVED from first principles (Phase 32).
    """
    return math.exp(R) + (2 * K - 1)


def compute_g_anchored(f_I1: float) -> float:
    """
    Compute g using the anchored I1/I2 weighted formula.

    Formula: g = f_I1 × g_I1 + (1 - f_I1) × g_I2

    Uses calibrated constants from 2-benchmark solve.
    """
    return f_I1 * G_I1_CALIBRATED + (1 - f_I1) * G_I2_CALIBRATED


def _is_anchored_mode(mode: CorrectionMode) -> bool:
    """Check if mode uses benchmark-anchored constants (internal helper)."""
    return mode in (
        CorrectionMode.ANCHORED_TWO_BENCHMARKS,
        CorrectionMode.COMPONENT_RENORM_ANCHORED,  # Legacy alias
    )


def get_g_correction(
    R: float,
    theta: float,
    K: int,
    f_I1: float = None,
    mode: CorrectionMode = CorrectionMode.DERIVED_BASELINE_ONLY,
    allow_target_anchoring: bool = False,
) -> CorrectionResult:
    """
    Get the g correction factor for mirror multiplier.

    Args:
        R: The R parameter
        theta: θ parameter (typically 4/7)
        K: Number of mollifier pieces (typically 3)
        f_I1: I1 fraction at -R (required for I1/I2 differentiated modes)
        mode: Which correction mode to use (default: DERIVED_BASELINE_ONLY)
        allow_target_anchoring: Must be True to use ANCHORED_TWO_BENCHMARKS mode.
            This guard prevents accidental use of calibrated constants.

    Returns:
        CorrectionResult with g, base, m, and mode information

    Raises:
        ValueError: If ANCHORED_TWO_BENCHMARKS mode is used without
            allow_target_anchoring=True, or if f_I1 is missing when required.
    """
    g_baseline = compute_g_baseline(theta, K)
    base = compute_base(R, K)

    # GUARD: Anchored mode requires explicit opt-in (Phase 46.0)
    if _is_anchored_mode(mode) and not allow_target_anchoring:
        raise ValueError(
            f"Mode {mode.name} uses benchmark-anchored constants and requires "
            f"allow_target_anchoring=True. This is NOT first-principles derivation. "
            f"If you intentionally want to use calibrated constants, set "
            f"allow_target_anchoring=True explicitly."
        )

    if mode == CorrectionMode.DERIVED_BASELINE_ONLY:
        g = g_baseline
        result = CorrectionResult(
            g=g,
            base=base,
            m=g * base,
            mode=mode,
            g_baseline=g_baseline,
        )

    elif mode == CorrectionMode.FIRST_PRINCIPLES_I1_I2:
        if f_I1 is None:
            raise ValueError(
                "FIRST_PRINCIPLES_I1_I2 mode requires f_I1 parameter. "
                "Compute f_I1 = I1(-R) / (I1(-R) + I2(-R)) first."
            )

        # First-principles derived values (no calibration)
        g_I1_derived = 1.0  # Log factor cross-terms self-correct
        g_I2_derived = g_baseline  # Full Beta moment for I2

        g = f_I1 * g_I1_derived + (1 - f_I1) * g_I2_derived

        result = CorrectionResult(
            g=g,
            base=base,
            m=g * base,
            mode=mode,
            g_baseline=g_baseline,
            g_I1=g_I1_derived,
            g_I2=g_I2_derived,
            f_I1=f_I1,
        )

    elif _is_anchored_mode(mode):
        # allow_target_anchoring was already checked above
        if f_I1 is None:
            raise ValueError(
                "ANCHORED_TWO_BENCHMARKS mode requires f_I1 parameter. "
                "Compute f_I1 = I1(-R) / (I1(-R) + I2(-R)) first."
            )

        g = compute_g_anchored(f_I1)

        # Log the anchored constants being used
        logger.warning(
            f"Using ANCHORED_TWO_BENCHMARKS mode with calibrated constants:\n"
            f"  g_I1 = {G_I1_CALIBRATED:.8f} (calibrated, NOT derived)\n"
            f"  g_I2 = {G_I2_CALIBRATED:.8f} (calibrated, NOT derived)\n"
            f"  f_I1 = {f_I1:.6f}\n"
            f"  g_total = {g:.8f}\n"
            f"  g_baseline = {g_baseline:.8f}\n"
            f"  delta = {g - g_baseline:+.8f} ({(g/g_baseline - 1)*100:+.4f}%)"
        )

        # Normalize to ANCHORED_TWO_BENCHMARKS for consistent storage
        result = CorrectionResult(
            g=g,
            base=base,
            m=g * base,
            mode=CorrectionMode.ANCHORED_TWO_BENCHMARKS,
            g_baseline=g_baseline,
            g_I1=G_I1_CALIBRATED,
            g_I2=G_I2_CALIBRATED,
            f_I1=f_I1,
        )

    else:
        raise ValueError(f"Unknown correction mode: {mode}")

    return result


def get_mirror_multiplier(
    R: float,
    theta: float = 4/7,
    K: int = 3,
    f_I1: float = None,
    mode: CorrectionMode = CorrectionMode.DERIVED_BASELINE_ONLY,
    allow_target_anchoring: bool = False,
) -> float:
    """
    Convenience function to get the mirror multiplier m = g × base.

    Returns just the multiplier value, not the full CorrectionResult.
    """
    result = get_g_correction(R, theta, K, f_I1, mode, allow_target_anchoring)
    return result.m


# Validation helpers for tests
def is_derived_mode(mode: CorrectionMode) -> bool:
    """Check if mode uses only derived (non-anchored) constants."""
    return mode in (
        CorrectionMode.DERIVED_BASELINE_ONLY,
        CorrectionMode.FIRST_PRINCIPLES_I1_I2,
    )


def is_anchored_mode(mode: CorrectionMode) -> bool:
    """Check if mode uses benchmark-anchored constants."""
    return _is_anchored_mode(mode)


def get_default_mode() -> CorrectionMode:
    """Return the default correction mode (should be DERIVED_BASELINE_ONLY)."""
    return CorrectionMode.DERIVED_BASELINE_ONLY


def get_all_derived_modes() -> list:
    """Return all correction modes that are first-principles derived."""
    return [
        CorrectionMode.DERIVED_BASELINE_ONLY,
        CorrectionMode.FIRST_PRINCIPLES_I1_I2,
    ]


def get_all_anchored_modes() -> list:
    """Return all correction modes that use benchmark anchoring."""
    return [
        CorrectionMode.ANCHORED_TWO_BENCHMARKS,
    ]
