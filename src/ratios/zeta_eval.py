"""
src/ratios/zeta_eval.py
Phase 19.2: Unified Zeta Evaluation API

PURPOSE:
========
Provide a single, authoritative interface for evaluating ζ'/ζ with explicit
mode separation between:

1. SEMANTIC_LAURENT_NEAR_1: Paper's asymptotic regime (α ~ 1/L small)
2. NUMERIC_FUNCTIONAL_EQ: Analytic continuation at arbitrary points

This prevents the "silent mode drift" issue where diagnostic code evaluated
ζ'/ζ at s=1-R (far from 1) but production code used Laurent expansions.

KEY INSIGHT (from GPT Phase 19 Guidance):
=========================================
PRZZ works in the small-shift regime (α,β ~ 1/L) where Laurent expansion is valid.
The diagnostic pipelines evaluated ζ'/ζ at s=1-R where R≈1.3, which is NOT small.
These are different mathematical operations and must not be mixed.

PAPER CONTEXT:
=============
PRZZ TeX Lines 1502-1511: The contour integral regime uses α=β=-R/L with L=log(T).
For T large, α is small and Laurent expansion is valid.

At finite R (benchmark evaluation), we use the SAME formula structure but
evaluate numerically where needed.

DECISION 8 COMPLIANCE:
=====================
- Semantic mode (RAW_LOGDERIV): Uses Laurent (1/R + γ)
- Numeric mode (ACTUAL_LOGDERIV): Uses mpmath evaluation

See docs/DERIVE_ZETA_LOGDERIV_FACTOR.md for the full derivation.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Optional
import math
import warnings

try:
    import mpmath
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

__all__ = [
    'ZetaMode',
    'ZetaEvalResult',
    'zeta_logderiv_scaled',
    'zeta_logderiv_at_point',
    'zeta_logderiv_squared',
    'get_laurent_at_point',
    'validate_mode_for_point',
    'EULER_MASCHERONI',
    'STIELTJES_GAMMA1',
]

# Constants
EULER_MASCHERONI = 0.5772156649015329
STIELTJES_GAMMA1 = -0.0728158454836767
STIELTJES_GAMMA2 = -0.0096903631928723


class ZetaMode(Enum):
    """
    Evaluation modes for ζ'/ζ.

    SEMANTIC_LAURENT_NEAR_1:
        Paper's asymptotic expansion regime. Valid when |s-1| << 1.
        Uses: (ζ'/ζ)(1+ε) = -1/ε + γ + O(ε)

        This is the "semantic" mode that matches the paper's derivation
        structure. Use for theoretical validation and asymptotic analysis.

    NUMERIC_FUNCTIONAL_EQ:
        Direct numerical evaluation via mpmath. Valid at any point
        (except poles).

        This is the "numeric" mode for production accuracy at finite R.
        Use for benchmark matching and production κ computation.
    """
    SEMANTIC_LAURENT_NEAR_1 = auto()
    NUMERIC_FUNCTIONAL_EQ = auto()


@dataclass(frozen=True)
class ZetaEvalResult:
    """Result of ζ'/ζ evaluation with metadata."""
    value: float
    squared: float
    mode: ZetaMode
    eval_point: complex
    is_scaled: bool  # Whether this is the scaled version for PRZZ
    warning: Optional[str] = None


def _compute_laurent(eps: complex, order: int = 2) -> complex:
    """
    Laurent expansion of (ζ'/ζ)(1+ε) around ε=0.

    (ζ'/ζ)(1+ε) = -1/ε + γ + γ₁ε + O(ε²)

    Args:
        eps: The argument ε (should be small for accuracy)
        order: Expansion order (2=pole+constant, 3=adds linear term)

    Returns:
        Laurent approximation
    """
    if abs(eps) < 1e-15:
        raise ValueError("Cannot evaluate Laurent at ε=0 (pole)")

    result = -1.0 / eps  # Pole term

    if order >= 2:
        result += EULER_MASCHERONI

    if order >= 3:
        result += STIELTJES_GAMMA1 * eps

    if order >= 4:
        result += STIELTJES_GAMMA2 * eps * eps / 2.0

    return complex(result)


def _compute_mpmath(s: complex, precision: int = 50) -> complex:
    """
    Compute (ζ'/ζ)(s) using mpmath for arbitrary s.

    Args:
        s: Complex argument
        precision: mpmath working precision

    Returns:
        Numerical (ζ'/ζ)(s)
    """
    if not MPMATH_AVAILABLE:
        raise ImportError(
            "mpmath required for NUMERIC_FUNCTIONAL_EQ mode. "
            "Install with: pip install mpmath"
        )

    with mpmath.workdps(precision):
        # ζ'/ζ = d/ds log(ζ(s))
        zeta_val = mpmath.zeta(s)
        if abs(zeta_val) < 1e-100:
            raise ValueError(f"ζ({s}) is too close to zero (near a trivial zero?)")

        zeta_deriv = mpmath.diff(mpmath.zeta, s)
        return complex(zeta_deriv / zeta_val)


def validate_mode_for_point(
    s: complex,
    mode: ZetaMode,
    strict: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate that the mode is appropriate for the evaluation point.

    This is a key Phase 19 guardrail: prevent using Laurent expansion
    far from s=1 without explicit acknowledgment.

    Args:
        s: Evaluation point
        mode: Requested mode
        strict: If True, raise on invalid mode; if False, just warn

    Returns:
        (is_valid, warning_message)
    """
    eps = abs(s - 1.0)

    if mode == ZetaMode.SEMANTIC_LAURENT_NEAR_1:
        if eps > 0.5:
            msg = (
                f"Laurent expansion at s={s} (|s-1|={eps:.3f}) is inaccurate. "
                f"Laurent is valid for |s-1| << 1. For s=1-R with R≈1.3, "
                f"use NUMERIC_FUNCTIONAL_EQ mode instead."
            )
            if strict:
                raise ValueError(msg)
            return False, msg
        elif eps > 0.1:
            msg = (
                f"Laurent expansion at s={s} (|s-1|={eps:.3f}) may have "
                f"significant error (~{eps*100:.0f}%). Consider NUMERIC mode."
            )
            return True, msg

    return True, None


def zeta_logderiv_at_point(
    s: complex,
    mode: ZetaMode = ZetaMode.NUMERIC_FUNCTIONAL_EQ,
    precision: int = 50,
    validate: bool = True
) -> ZetaEvalResult:
    """
    Compute (ζ'/ζ)(s) at an arbitrary point with explicit mode.

    This is the primary API for diagnostic and production code.

    Args:
        s: Evaluation point
        mode: Which evaluation method to use
        precision: mpmath precision (for NUMERIC mode)
        validate: Whether to check mode appropriateness

    Returns:
        ZetaEvalResult with value and metadata

    Example:
        # Diagnostic at finite R (correct usage)
        result = zeta_logderiv_at_point(1 - 1.3036, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ)

        # Paper structure validation (correct usage)
        result = zeta_logderiv_at_point(1 - 0.01, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)

        # This will warn (Laurent far from 1)
        result = zeta_logderiv_at_point(1 - 1.3, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)
    """
    warning = None
    if validate:
        _, warning = validate_mode_for_point(s, mode, strict=False)
        if warning:
            warnings.warn(warning, UserWarning, stacklevel=2)

    if mode == ZetaMode.SEMANTIC_LAURENT_NEAR_1:
        eps = s - 1.0
        value = _compute_laurent(eps, order=4)
    else:
        value = _compute_mpmath(s, precision)

    return ZetaEvalResult(
        value=float(value.real) if abs(value.imag) < 1e-10 else complex(value),
        squared=float((value * value).real),
        mode=mode,
        eval_point=s,
        is_scaled=False,
        warning=warning
    )


def zeta_logderiv_scaled(
    alpha_over_L: float,
    L: float,
    mode: ZetaMode = ZetaMode.SEMANTIC_LAURENT_NEAR_1
) -> ZetaEvalResult:
    """
    Compute (ζ'/ζ) in the PRZZ asymptotic regime with explicit scaling.

    PRZZ regime: α = -R/L with L = log(T), so the evaluation point is
    s = 1 + α = 1 - R/L, which approaches 1 as T → ∞.

    The Laurent expansion (ζ'/ζ)(1+ε) ≈ -1/ε + γ is valid in this regime.

    Args:
        alpha_over_L: The ratio α/L (dimensionless)
        L: The scaling parameter log(T)
        mode: Evaluation mode (SEMANTIC recommended here)

    Returns:
        ZetaEvalResult with the scaled value

    Note:
        For production at finite R, use zeta_logderiv_at_point(1-R)
        with NUMERIC mode instead.
    """
    alpha = alpha_over_L * L
    s = 1.0 + alpha

    if mode == ZetaMode.SEMANTIC_LAURENT_NEAR_1:
        # Paper regime: use Laurent at s = 1 + α
        # (ζ'/ζ)(1 + α) = -1/α + γ + O(α)
        if abs(alpha) < 1e-10:
            raise ValueError("α too small for Laurent expansion (pole)")
        value = -1.0 / alpha + EULER_MASCHERONI
    else:
        # Numeric evaluation (unusual for scaled version but allowed)
        result = _compute_mpmath(s)
        value = float(result.real)

    return ZetaEvalResult(
        value=value,
        squared=value * value,
        mode=mode,
        eval_point=complex(s),
        is_scaled=True,
        warning=None
    )


def zeta_logderiv_squared(
    R: float,
    mode: ZetaMode = ZetaMode.NUMERIC_FUNCTIONAL_EQ,
    precision: int = 50
) -> float:
    """
    Compute (ζ'/ζ)²(1-R) for the J12 bracket term.

    This is the squared log-derivative that appears in the J12 factor.
    At PRZZ evaluation points (R≈1.3, R≈1.1), this differs significantly
    from the Laurent approximation (1/R + γ)².

    Phase 15A Finding:
        - κ (R=1.3036): actual² ≈ 3.00, Laurent² ≈ 1.81 (66% error)
        - κ* (R=1.1167): actual² ≈ 3.16, Laurent² ≈ 2.17 (46% error)

    Args:
        R: PRZZ R parameter
        mode: Evaluation mode

    Returns:
        (ζ'/ζ)²(1-R) value

    Example:
        # Production usage (ACTUAL_LOGDERIV from Decision 8)
        j12_factor = zeta_logderiv_squared(1.3036, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ)

        # Paper structure (RAW_LOGDERIV from Decision 8)
        j12_factor = zeta_logderiv_squared(1.3036, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)
    """
    s = 1.0 - R
    result = zeta_logderiv_at_point(s, mode=mode, precision=precision, validate=True)
    return result.squared


def get_laurent_at_point(R: float) -> Tuple[float, float]:
    """
    Get the Laurent approximation values at s=1-R.

    Returns both the single factor (for J13/J14) and squared (for J12).

    Args:
        R: PRZZ R parameter

    Returns:
        (single_factor, squared_factor) where:
        - single_factor = 1/R + γ ≈ (ζ'/ζ)(1-R)
        - squared_factor = (1/R + γ)² ≈ (ζ'/ζ)²(1-R)

    Note:
        This is the Laurent approximation, which has significant error
        at finite R. Use for structural validation only, not production.
    """
    single = 1.0 / R + EULER_MASCHERONI
    return single, single * single


def compare_modes(R: float, precision: int = 50) -> dict:
    """
    Compare SEMANTIC vs NUMERIC modes at a given R value.

    Useful for diagnostics and understanding the approximation error.

    Args:
        R: PRZZ R parameter
        precision: mpmath precision

    Returns:
        Dict with comparison data
    """
    s = 1.0 - R

    # Laurent (semantic)
    laurent_single, laurent_squared = get_laurent_at_point(R)

    # Actual (numeric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress validation warnings
        numeric_result = zeta_logderiv_at_point(
            s, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ, precision=precision, validate=False
        )

    actual_single = numeric_result.value
    actual_squared = numeric_result.squared

    # Compute errors
    single_error = abs(actual_single - laurent_single) / abs(actual_single) * 100
    squared_error = abs(actual_squared - laurent_squared) / abs(actual_squared) * 100

    return {
        "R": R,
        "eval_point": s,
        "laurent_single": laurent_single,
        "laurent_squared": laurent_squared,
        "actual_single": actual_single,
        "actual_squared": actual_squared,
        "single_error_percent": single_error,
        "squared_error_percent": squared_error,
        "squared_ratio": actual_squared / laurent_squared,
    }


# Convenience aliases matching Decision 8 terminology
def get_raw_logderiv_squared(R: float) -> float:
    """RAW_LOGDERIV mode: Laurent (1/R + γ)²."""
    _, squared = get_laurent_at_point(R)
    return squared


def get_actual_logderiv_squared(R: float, precision: int = 50) -> float:
    """ACTUAL_LOGDERIV mode: mpmath (ζ'/ζ)²(1-R)."""
    return zeta_logderiv_squared(R, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ, precision=precision)


def get_actual_logderiv_single(R: float, precision: int = 50) -> float:
    """ACTUAL_LOGDERIV single factor for J13/J14."""
    s = 1.0 - R
    result = zeta_logderiv_at_point(s, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ, precision=precision)
    return result.value
