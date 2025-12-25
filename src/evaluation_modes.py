"""
Evaluation modes for PRZZ c/kappa computation.

This module enforces the separation between:
- MAIN_TERM_ONLY: Computing the asymptotic main term c (I₅ and A-derivatives forbidden)
- WITH_ERROR_TERMS: Diagnostic mode that allows error-order contributions

Per TRUTH_SPEC.md Section 4, I₅ is explicitly O(T/L), which is lower order than
the main term O(T). Using I₅ to calibrate c masks structural bugs.

References:
    TRUTH_SPEC.md Lines 1621-1628: "I₅ ≪ T/L ... Hence the term associated to
    A_{α,β}^{(1,1)}(0,0;β,α) is an error term."

    TRUTH_SPEC.md Lines 1722-1727: "Any nonzero derivatives of A contribute
    only to the error term O(T/L)."
"""

from enum import Enum, auto
from typing import Optional
import warnings
from contextlib import contextmanager

__all__ = [
    'EvaluationMode',
    'get_evaluation_mode',
    'set_evaluation_mode',
    'evaluation_mode_context',
    'assert_main_term_only',
    'I5ForbiddenError',
]


class EvaluationMode(Enum):
    """
    Semantic evaluation modes per TRUTH_SPEC.md Section 4.

    MAIN_TERM_ONLY:
        - I₅ contributions are FORBIDDEN
        - A^{(m,n)} with m+n > 0 are FORBIDDEN
        - This is the correct mode for matching published κ

    WITH_ERROR_TERMS:
        - I₅ contributions are ALLOWED (with warning)
        - Useful for diagnostics and understanding error structure
        - Should NEVER be used to "fix" the main-term c
    """
    MAIN_TERM_ONLY = auto()
    WITH_ERROR_TERMS = auto()


class I5ForbiddenError(ValueError):
    """
    Raised when attempting to use I₅ or A-derivatives in MAIN_TERM_ONLY mode.

    This is a hard error, not a warning, because using I₅ to match κ
    indicates we're computing the wrong main-term object.
    """
    pass


# Global mode state
_current_mode: EvaluationMode = EvaluationMode.MAIN_TERM_ONLY


def get_evaluation_mode() -> EvaluationMode:
    """Get the current evaluation mode."""
    return _current_mode


def set_evaluation_mode(mode: EvaluationMode) -> EvaluationMode:
    """
    Set the evaluation mode.

    Args:
        mode: The mode to set

    Returns:
        The previous mode (for restoration)

    Warns:
        If setting WITH_ERROR_TERMS, emits UserWarning
    """
    global _current_mode
    previous = _current_mode
    _current_mode = mode

    if mode == EvaluationMode.WITH_ERROR_TERMS:
        warnings.warn(
            "WITH_ERROR_TERMS mode enabled. I₅ and A-derivative contributions "
            "are error-order (≪ T/L) per TRUTH_SPEC.md Lines 1621-1628. "
            "Do NOT use this mode to calibrate main-term c or match published κ.",
            UserWarning,
            stacklevel=2
        )

    return previous


@contextmanager
def evaluation_mode_context(mode: EvaluationMode):
    """
    Context manager for temporarily changing evaluation mode.

    Usage:
        with evaluation_mode_context(EvaluationMode.WITH_ERROR_TERMS):
            # I₅ allowed in this block
            result = compute_with_i5()
        # Automatically restored to previous mode
    """
    previous = set_evaluation_mode(mode)
    try:
        yield
    finally:
        set_evaluation_mode(previous)


def assert_main_term_only(operation: str) -> None:
    """
    Assert that we're in MAIN_TERM_ONLY mode.

    Call this before any operation that uses I₅ or A-derivatives.
    In MAIN_TERM_ONLY mode, this raises I5ForbiddenError.
    In WITH_ERROR_TERMS mode, this is a no-op.

    Args:
        operation: Description of the forbidden operation (for error message)

    Raises:
        I5ForbiddenError: If in MAIN_TERM_ONLY mode
    """
    if _current_mode == EvaluationMode.MAIN_TERM_ONLY:
        raise I5ForbiddenError(
            f"Operation '{operation}' is forbidden in MAIN_TERM_ONLY mode. "
            f"I₅ and A^{{(m,n)}} derivatives with m+n>0 are error-order terms "
            f"(≪ T/L) per TRUTH_SPEC.md Lines 1621-1628. "
            f"Using them to match κ masks structural bugs in the main-term computation. "
            f"If you need these terms for diagnostics, use:\n"
            f"    from src.evaluation_modes import evaluation_mode_context, EvaluationMode\n"
            f"    with evaluation_mode_context(EvaluationMode.WITH_ERROR_TERMS):\n"
            f"        # your code here"
        )


def check_no_a_derivatives(m: int, n: int, operation: str = "A-derivative") -> None:
    """
    Check that A^{(m,n)} with m+n > 0 is not being used in main mode.

    Per TRUTH_SPEC.md Lines 1714-1727, nonzero derivatives of A contribute
    only to error terms O(T/L).

    Args:
        m: First derivative order
        n: Second derivative order
        operation: Description for error message

    Raises:
        I5ForbiddenError: If m+n > 0 and in MAIN_TERM_ONLY mode
    """
    if m + n > 0:
        assert_main_term_only(f"{operation} with A^{{({m},{n})}}")


# Convenience aliases
MAIN_MODE = EvaluationMode.MAIN_TERM_ONLY
ERROR_MODE = EvaluationMode.WITH_ERROR_TERMS
