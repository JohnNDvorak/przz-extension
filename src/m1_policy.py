"""
src/m1_policy.py
M1 Policy Module - HARD SAFETY LOCK for K>3 Extension.

The mirror weight m1 = exp(R) + (2K-1) is an EMPIRICAL formula that has been
calibrated for K=3 but is NOT derived from first principles.

This module implements a HARD SAFETY LOCK that:
1. Prevents accidental K>3 extrapolation
2. Requires explicit opt-in for extrapolation
3. Documents the empirical nature of the formula

EMPIRICAL FORMULA STATUS:
- m1 = exp(R) + 5 for K=3 (calibrated against both kappa and kappa* benchmarks)
- m1 = exp(R) + (2K-1) is the conjectured generalization
- Neither finite-L nor unified-t approaches derived m1 from first principles
- The formula is QUARANTINED as a volatile parameter

SPEC AUTHORITY (for this module):
1. TRUTH_SPEC.md - establishes I1/I2 have mirror structure
2. Empirical calibration - shows exp(R)+5 works for K=3
3. This code - enforces safety gates

See: docs/K_SAFE_BASELINE_LOCKDOWN.md for detailed history.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np
import warnings


class M1Mode(Enum):
    """
    Modes for computing the mirror weight m1.

    Each mode represents a different formula or approach for m1.
    Some modes have restrictions on which K values they support.
    """

    K3_EMPIRICAL = auto()
    """
    exp(R) + 5, only valid for K=3.

    This is the calibrated value that matches both kappa (R=1.3036)
    and kappa* (R=1.1167) benchmarks. It is NOT derived from first
    principles.

    RAISES ValueError if K != 3.
    """

    K_DEP_EMPIRICAL = auto()
    """
    exp(R) + (2K-1), extrapolative formula.

    This generalizes K3_EMPIRICAL using the pattern:
    - K=3: exp(R) + 5 (validated)
    - K=4: exp(R) + 7 (UNVALIDATED extrapolation)
    - K=5: exp(R) + 9 (UNVALIDATED extrapolation)

    RAISES ValueError if K>3 and allow_extrapolation=False.
    """

    PAPER_NAIVE = auto()
    """
    exp(2R), the naive formula from combined identity.

    This comes from evaluating T^{-alpha-beta} at alpha=beta=-R/L
    in the combined identity. It is INCORRECT for actual computation
    but useful as a reference.

    The naive formula gives m1 that is too large:
    - kappa: naive/empirical = 1.56
    - kappa*: naive/empirical = 1.16

    RAISES ValueError if K>3 and allow_extrapolation=False.
    """

    OVERRIDE = auto()
    """
    User-supplied m1 value.

    For research purposes when testing alternative formulas.
    Requires override_value to be set.

    RAISES ValueError if override_value is None.
    """

    DIAGNOSTIC_FITTED = auto()
    """
    ⚠️ DIAGNOSTIC ONLY - NOT DERIVED FROM FIRST PRINCIPLES ⚠️

    m1 = 1.0374 * exp(R) + 4.9938

    This formula was determined by fitting to c_target at both κ and κ*
    benchmarks. It achieves 0% gap but is CALIBRATION CREEP — it masks
    the underlying derivation problem rather than solving it.

    USE CASES:
    - Diagnostic scripts to characterize the gap
    - Experiment scripts investigating m₁ derivation
    - NEVER for production evaluators

    REQUIRES: allow_diagnostic=True in policy, otherwise RAISES.

    GPT WARNING (2025-12-22):
    > "Adding m1 = 1.037*exp(R)+5 is EXACTLY the kind of 'quiet calibration
    > creep' you said you don't want. It's fine as a DIAGNOSTIC ARTIFACT
    > (a clue!), but it must NOT become baseline behavior."
    """


@dataclass(frozen=True)
class M1Policy:
    """
    Policy for computing m1.

    This is a HARD SAFETY LOCK that prevents accidental K>3 extrapolation
    and prevents calibration creep from fitted m₁.

    By default, K>3 will RAISE unless explicitly enabled.
    By default, DIAGNOSTIC_FITTED will RAISE unless explicitly enabled.

    Attributes:
        mode: Which formula to use
        allow_extrapolation: If True, permit K>3 (with warning)
        allow_diagnostic: If True, permit DIAGNOSTIC_FITTED mode (with warning)
        override_value: Required if mode=OVERRIDE

    Examples:
        # Safe K=3 usage (default)
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        m1 = m1_formula(K=3, R=1.3036, policy=policy)  # Works

        # Explicit K=4 extrapolation (requires opt-in)
        policy = M1Policy(
            mode=M1Mode.K_DEP_EMPIRICAL,
            allow_extrapolation=True
        )
        m1 = m1_formula(K=4, R=1.3036, policy=policy)  # Works with warning

        # K=4 without opt-in (RAISES)
        policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL)
        m1 = m1_formula(K=4, R=1.3036, policy=policy)  # RAISES!

        # DIAGNOSTIC_FITTED requires explicit opt-in
        policy = M1Policy(mode=M1Mode.DIAGNOSTIC_FITTED, allow_diagnostic=True)
        m1 = m1_formula(K=3, R=1.3036, policy=policy)  # Works with warning
    """

    mode: M1Mode
    allow_extrapolation: bool = False
    allow_diagnostic: bool = False
    override_value: Optional[float] = None


class M1ExtrapolationError(ValueError):
    """Raised when K>3 extrapolation is attempted without opt-in."""
    pass


class M1DiagnosticError(ValueError):
    """
    Raised when DIAGNOSTIC_FITTED mode is used without allow_diagnostic=True.

    This is a HARD GUARD against calibration creep. The fitted m₁ formula
    achieves 0% gap but is NOT derived from first principles — using it
    masks the underlying derivation problem.

    If you see this error in production code, you're trying to use a
    diagnostic-only formula where an empirical or derived formula should be used.
    """
    pass


def m1_formula(K: int, R: float, policy: M1Policy) -> float:
    """
    Compute the mirror weight m1 according to policy.

    This function enforces HARD SAFETY GATES:
    - K3_EMPIRICAL: RAISES if K != 3
    - K_DEP_EMPIRICAL: RAISES if K > 3 and not allow_extrapolation
    - PAPER_NAIVE: RAISES if K > 3 and not allow_extrapolation
    - OVERRIDE: RAISES if override_value is None

    Args:
        K: Number of mollifier pieces (typically 3)
        R: R parameter (typically 1.3036 for kappa, 1.1167 for kappa*)
        policy: M1Policy specifying mode and safety options

    Returns:
        The mirror weight m1

    Raises:
        ValueError: If K3_EMPIRICAL used with K != 3
        M1ExtrapolationError: If K > 3 without allow_extrapolation=True
        ValueError: If OVERRIDE used without override_value

    Examples:
        >>> from src.m1_policy import M1Policy, M1Mode, m1_formula
        >>> policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        >>> m1 = m1_formula(K=3, R=1.3036, policy=policy)
        >>> abs(m1 - 8.683) < 0.01
        True
    """
    if policy.mode == M1Mode.K3_EMPIRICAL:
        if K != 3:
            raise ValueError(
                f"M1Mode.K3_EMPIRICAL is only valid for K=3, got K={K}. "
                f"Use M1Mode.K_DEP_EMPIRICAL with allow_extrapolation=True for K>3."
            )
        return float(np.exp(R) + 5)

    if policy.mode == M1Mode.K_DEP_EMPIRICAL:
        if K > 3 and not policy.allow_extrapolation:
            raise M1ExtrapolationError(
                f"Extrapolation to K={K} requires allow_extrapolation=True. "
                f"The formula m1 = exp(R) + (2K-1) is UNVALIDATED for K>3. "
                f"Set M1Policy(mode=M1Mode.K_DEP_EMPIRICAL, allow_extrapolation=True) "
                f"to proceed at your own risk."
            )
        if K > 3:
            warnings.warn(
                f"Using EXTRAPOLATED m1 formula for K={K}: m1 = exp(R) + {2*K-1}. "
                f"This is UNVALIDATED and may produce incorrect results.",
                UserWarning
            )
        return float(np.exp(R) + (2 * K - 1))

    if policy.mode == M1Mode.PAPER_NAIVE:
        if K > 3 and not policy.allow_extrapolation:
            raise M1ExtrapolationError(
                f"M1Mode.PAPER_NAIVE at K={K} requires allow_extrapolation=True."
            )
        if K > 3:
            warnings.warn(
                f"Using PAPER_NAIVE m1 = exp(2R) at K={K}. "
                f"This is known to be too large by ~50%.",
                UserWarning
            )
        return float(np.exp(2 * R))

    if policy.mode == M1Mode.OVERRIDE:
        if policy.override_value is None:
            raise ValueError(
                "M1Mode.OVERRIDE requires override_value to be set. "
                "Use M1Policy(mode=M1Mode.OVERRIDE, override_value=X.XX)."
            )
        return float(policy.override_value)

    if policy.mode == M1Mode.DIAGNOSTIC_FITTED:
        if not policy.allow_diagnostic:
            raise M1DiagnosticError(
                "M1Mode.DIAGNOSTIC_FITTED requires allow_diagnostic=True. "
                "This mode uses a FITTED formula (m1 = 1.037*exp(R) + 5) that "
                "achieves 0% gap but is NOT derived from first principles. "
                "Using it in production code constitutes CALIBRATION CREEP. "
                "If you need this for diagnostic purposes, use: "
                "M1Policy(mode=M1Mode.DIAGNOSTIC_FITTED, allow_diagnostic=True)"
            )
        warnings.warn(
            "Using DIAGNOSTIC_FITTED m1 = 1.037*exp(R) + 4.99. "
            "This is NOT derived from first principles — it's a fit to c_target. "
            "Do NOT use for production evaluators.",
            UserWarning
        )
        return float(M1_FITTED_COEFFICIENT_A * np.exp(R) + M1_FITTED_COEFFICIENT_B)

    raise AssertionError(f"Unhandled M1Mode: {policy.mode}")


# =============================================================================
# Reference values for validation
# =============================================================================

# Empirical m1 values for K=3 at both benchmarks
M1_EMPIRICAL_KAPPA = np.exp(1.3036) + 5  # ~8.683
M1_EMPIRICAL_KAPPA_STAR = np.exp(1.1167) + 5  # ~8.055

# Naive (paper) m1 values - these are TOO LARGE
M1_NAIVE_KAPPA = np.exp(2 * 1.3036)  # ~13.56
M1_NAIVE_KAPPA_STAR = np.exp(2 * 1.1167)  # ~9.33

# Ratio of naive to empirical (shows naive is too large)
M1_NAIVE_TO_EMPIRICAL_RATIO_KAPPA = M1_NAIVE_KAPPA / M1_EMPIRICAL_KAPPA  # ~1.56
M1_NAIVE_TO_EMPIRICAL_RATIO_KAPPA_STAR = M1_NAIVE_KAPPA_STAR / M1_EMPIRICAL_KAPPA_STAR  # ~1.16

# =============================================================================
# Fitted m1 formula (Phase 8.2 discovery, 2025-12-22)
# =============================================================================
#
# The IDEAL m1 formula was determined by fitting to achieve 0% gap on both benchmarks:
#   m1_ideal = a * exp(R) + b
# where:
#   a = 1.0374 (very close to 1.0)
#   b = 4.9938 (very close to 5.0)
#
# This is almost identical to the empirical formula m1 = exp(R) + 5, but with a
# ~3.7% correction to the exp(R) coefficient.
#
# The remaining ~1.3% gap in the empirical formula is explained by a ≠ 1.

M1_FITTED_COEFFICIENT_A = 1.037353  # Coefficient of exp(R)
M1_FITTED_COEFFICIENT_B = 4.993849  # Constant term (almost exactly 5)

M1_IDEAL_KAPPA = M1_FITTED_COEFFICIENT_A * np.exp(1.3036) + M1_FITTED_COEFFICIENT_B  # ~8.814
M1_IDEAL_KAPPA_STAR = M1_FITTED_COEFFICIENT_A * np.exp(1.1167) + M1_FITTED_COEFFICIENT_B  # ~8.163


def m1_diagnostic_fitted(R: float) -> float:
    """
    ⚠️ DIAGNOSTIC ONLY — NOT DERIVED FROM FIRST PRINCIPLES ⚠️

    Compute m1 using the fitted formula that achieves 0% gap.

    m1 = 1.0374 * exp(R) + 4.9938

    This formula was determined by fitting to c_target at both κ (R=1.3036)
    and κ* (R=1.1167) benchmarks simultaneously.

    ⚠️ WARNING: Using this formula constitutes CALIBRATION CREEP.
    It masks the underlying derivation problem rather than solving it.

    USE CASES:
    - Diagnostic scripts characterizing the gap
    - Experiment scripts investigating m₁ derivation
    - NEVER for production evaluators

    Args:
        R: R parameter

    Returns:
        The fitted m1 value

    Note:
        This is very close to the empirical m1 = exp(R) + 5 but with a ~3.7%
        correction to the exp(R) coefficient. The "a ≈ 1.037" is a CLUE
        pointing to a missing structural factor, not a solution.

    GPT WARNING (2025-12-22):
        > "Adding m1 = 1.037*exp(R)+5 is EXACTLY the kind of 'quiet calibration
        > creep' you said you don't want."
    """
    warnings.warn(
        "m1_diagnostic_fitted() is DIAGNOSTIC ONLY. "
        "Do NOT use in production evaluators.",
        UserWarning
    )
    return M1_FITTED_COEFFICIENT_A * np.exp(R) + M1_FITTED_COEFFICIENT_B


def get_m1_reference_values() -> dict:
    """
    Get reference m1 values for testing and validation.

    Returns:
        Dict with empirical, fitted, and naive m1 values at both benchmarks.
    """
    return {
        "empirical": {
            "kappa": M1_EMPIRICAL_KAPPA,
            "kappa_star": M1_EMPIRICAL_KAPPA_STAR,
        },
        "fitted": {
            "kappa": M1_IDEAL_KAPPA,
            "kappa_star": M1_IDEAL_KAPPA_STAR,
            "coefficient_a": M1_FITTED_COEFFICIENT_A,
            "coefficient_b": M1_FITTED_COEFFICIENT_B,
        },
        "naive": {
            "kappa": M1_NAIVE_KAPPA,
            "kappa_star": M1_NAIVE_KAPPA_STAR,
        },
        "naive_to_empirical_ratio": {
            "kappa": M1_NAIVE_TO_EMPIRICAL_RATIO_KAPPA,
            "kappa_star": M1_NAIVE_TO_EMPIRICAL_RATIO_KAPPA_STAR,
        },
        "fitted_to_empirical_ratio": {
            "kappa": M1_IDEAL_KAPPA / M1_EMPIRICAL_KAPPA,  # ~1.015
            "kappa_star": M1_IDEAL_KAPPA_STAR / M1_EMPIRICAL_KAPPA_STAR,  # ~1.013
        },
    }
