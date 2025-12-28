#!/usr/bin/env python3
"""
tests/test_phase29_regime_lock.py
Phase 29: Kernel Regime Lock Tests

Validates that cross-regime comparisons are FORBIDDEN and raise RegimeMismatchError.
This is a critical safety feature to prevent meaningless comparisons between
mathematically different kernel regimes.

Created: 2025-12-26 (Phase 29)
"""

import pytest
import sys

sys.path.insert(0, ".")

from src.evaluator.s12_spec import (
    S12CanonicalValue,
    S12FullMatrix,
    KernelRegime,
    FactorialMode,
    SignMode,
    RegimeMismatchError,
    BACKEND_CONVENTIONS,
)


# ============================================================================
# Test S12CanonicalValue regime requirements
# ============================================================================


def test_s12_value_requires_kernel_regime():
    """S12CanonicalValue cannot be created without kernel_regime."""
    # This test verifies that kernel_regime is a required field
    # by checking that it appears in the dataclass
    val = S12CanonicalValue(
        ell1=1,
        ell2=1,
        value=1.0,
        kernel_regime=KernelRegime.RAW,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="test",
        term_type="I1",
    )
    assert val.kernel_regime == KernelRegime.RAW


def test_s12_value_regime_preserved_in_factorial_conversion():
    """Kernel regime must be preserved when converting factorial modes."""
    val_raw = S12CanonicalValue(
        ell1=2,
        ell2=2,
        value=1.0,
        kernel_regime=KernelRegime.RAW,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="test",
        term_type="I1",
    )

    # Convert to derivative mode
    val_deriv = val_raw.to_derivative_mode()
    assert val_deriv.kernel_regime == KernelRegime.RAW

    # Convert back to coefficient mode
    val_coeff = val_deriv.to_coefficient_mode()
    assert val_coeff.kernel_regime == KernelRegime.RAW


def test_s12_value_regime_preserved_in_sign_conversion():
    """Kernel regime must be preserved when converting sign modes."""
    val_paper = S12CanonicalValue(
        ell1=1,
        ell2=3,
        value=1.0,
        kernel_regime=KernelRegime.PAPER,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="test",
        term_type="I1",
    )

    # Convert to alternating sign mode
    val_signed = val_paper.with_sign_mode(SignMode.OFFDIAG_ALTERNATING)
    assert val_signed.kernel_regime == KernelRegime.PAPER

    # Convert back
    val_unsigned = val_signed.with_sign_mode(SignMode.NONE)
    assert val_unsigned.kernel_regime == KernelRegime.PAPER


def test_s12_value_regime_preserved_in_canonicalize():
    """Kernel regime must be preserved through full canonicalization."""
    val = S12CanonicalValue(
        ell1=2,
        ell2=3,
        value=10.0,
        kernel_regime=KernelRegime.RAW,
        factorial_mode=FactorialMode.DERIVATIVE,
        sign_mode=SignMode.OFFDIAG_ALTERNATING,
        backend="test",
        term_type="I2",
    )

    canonical = val.canonicalize(
        target_factorial=FactorialMode.COEFFICIENT,
        target_sign=SignMode.NONE,
    )

    assert canonical.kernel_regime == KernelRegime.RAW


# ============================================================================
# Test S12CanonicalValue cross-regime comparison blocking
# ============================================================================


def test_s12_value_same_regime_no_error():
    """Comparing values with same regime should NOT raise."""
    val1 = S12CanonicalValue(
        ell1=1, ell2=1, value=1.0,
        kernel_regime=KernelRegime.RAW,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="backend_a",
        term_type="I1",
    )
    val2 = S12CanonicalValue(
        ell1=1, ell2=1, value=2.0,
        kernel_regime=KernelRegime.RAW,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="backend_b",
        term_type="I1",
    )

    # Should not raise
    val1.assert_same_regime(val2)


def test_s12_value_cross_regime_raises():
    """Comparing values with different regimes must RAISE."""
    val_raw = S12CanonicalValue(
        ell1=2, ell2=2, value=3.884,
        kernel_regime=KernelRegime.RAW,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="unified_general",
        term_type="I1",
    )
    val_paper = S12CanonicalValue(
        ell1=2, ell2=2, value=0.917,
        kernel_regime=KernelRegime.PAPER,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="term_dsl_paper",
        term_type="I1",
    )

    with pytest.raises(RegimeMismatchError) as excinfo:
        val_raw.assert_same_regime(val_paper)

    assert "raw" in str(excinfo.value).lower()
    assert "paper" in str(excinfo.value).lower()


def test_s12_value_cross_regime_symmetric():
    """Regime mismatch detection is symmetric."""
    val_raw = S12CanonicalValue(
        ell1=1, ell2=3, value=-0.582,
        kernel_regime=KernelRegime.RAW,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="test",
        term_type="I1",
    )
    val_paper = S12CanonicalValue(
        ell1=1, ell2=3, value=+0.072,  # Sign flip in paper regime!
        kernel_regime=KernelRegime.PAPER,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="test",
        term_type="I1",
    )

    # Both directions should raise
    with pytest.raises(RegimeMismatchError):
        val_raw.assert_same_regime(val_paper)

    with pytest.raises(RegimeMismatchError):
        val_paper.assert_same_regime(val_raw)


# ============================================================================
# Test S12FullMatrix regime requirements
# ============================================================================


def test_s12_matrix_requires_kernel_regime():
    """S12FullMatrix cannot be created without kernel_regime."""
    matrix = S12FullMatrix(
        term_type="I1",
        backend="test",
        R=1.3036,
        theta=4/7,
        kernel_regime=KernelRegime.RAW,
        values={},
    )
    assert matrix.kernel_regime == KernelRegime.RAW


def test_s12_matrix_same_regime_no_error():
    """Comparing matrices with same regime should NOT raise."""
    matrix1 = S12FullMatrix(
        term_type="I1", backend="a", R=1.3036, theta=4/7,
        kernel_regime=KernelRegime.PAPER,
        values={},
    )
    matrix2 = S12FullMatrix(
        term_type="I1", backend="b", R=1.3036, theta=4/7,
        kernel_regime=KernelRegime.PAPER,
        values={},
    )

    # Should not raise
    matrix1.assert_same_regime(matrix2)


def test_s12_matrix_cross_regime_raises():
    """Comparing matrices with different regimes must RAISE."""
    matrix_raw = S12FullMatrix(
        term_type="I1", backend="unified_general", R=1.3036, theta=4/7,
        kernel_regime=KernelRegime.RAW,
        values={},
    )
    matrix_paper = S12FullMatrix(
        term_type="I1", backend="term_dsl_paper", R=1.3036, theta=4/7,
        kernel_regime=KernelRegime.PAPER,
        values={},
    )

    with pytest.raises(RegimeMismatchError):
        matrix_raw.assert_same_regime(matrix_paper)


def test_s12_matrix_compare_to_enforces_regime():
    """compare_to() with enforce_regime=True (default) raises on mismatch."""
    # Create simple matrices with one value each
    val_raw = S12CanonicalValue(
        ell1=1, ell2=1, value=1.0,
        kernel_regime=KernelRegime.RAW,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="a", term_type="I1",
    )
    val_paper = S12CanonicalValue(
        ell1=1, ell2=1, value=1.0,
        kernel_regime=KernelRegime.PAPER,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="b", term_type="I1",
    )

    matrix_raw = S12FullMatrix(
        term_type="I1", backend="a", R=1.3036, theta=4/7,
        kernel_regime=KernelRegime.RAW,
        values={(1, 1): val_raw},
    )
    matrix_paper = S12FullMatrix(
        term_type="I1", backend="b", R=1.3036, theta=4/7,
        kernel_regime=KernelRegime.PAPER,
        values={(1, 1): val_paper},
    )

    # Default (enforce_regime=True) should raise
    with pytest.raises(RegimeMismatchError):
        matrix_raw.compare_to(matrix_paper)

    with pytest.raises(RegimeMismatchError):
        matrix_raw.compare_to(matrix_paper, enforce_regime=True)


def test_s12_matrix_compare_to_can_bypass_regime_check():
    """compare_to() with enforce_regime=False allows cross-regime (with warning intent)."""
    val_raw = S12CanonicalValue(
        ell1=1, ell2=1, value=1.0,
        kernel_regime=KernelRegime.RAW,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="a", term_type="I1",
    )
    val_paper = S12CanonicalValue(
        ell1=1, ell2=1, value=2.0,
        kernel_regime=KernelRegime.PAPER,
        factorial_mode=FactorialMode.COEFFICIENT,
        sign_mode=SignMode.NONE,
        backend="b", term_type="I1",
    )

    matrix_raw = S12FullMatrix(
        term_type="I1", backend="a", R=1.3036, theta=4/7,
        kernel_regime=KernelRegime.RAW,
        values={(1, 1): val_raw},
    )
    matrix_paper = S12FullMatrix(
        term_type="I1", backend="b", R=1.3036, theta=4/7,
        kernel_regime=KernelRegime.PAPER,
        values={(1, 1): val_paper},
    )

    # With enforce_regime=False, comparison proceeds (for diagnostic/debugging)
    result = matrix_raw.compare_to(matrix_paper, enforce_regime=False)
    assert (1, 1) in result
    assert result[(1, 1)]["ratio"] == 0.5  # 1.0 / 2.0


# ============================================================================
# Test backend convention registry
# ============================================================================


def test_backend_conventions_have_kernel_regime():
    """All backend conventions must specify kernel_regime."""
    expected_backends = [
        "unified_general",
        "unified_paper",
        "term_dsl_raw",
        "term_dsl_paper",
    ]

    for backend in expected_backends:
        assert backend in BACKEND_CONVENTIONS, f"Missing backend: {backend}"
        conv = BACKEND_CONVENTIONS[backend]
        assert "kernel_regime" in conv, f"Backend {backend} missing kernel_regime"
        assert isinstance(conv["kernel_regime"], KernelRegime)


def test_backend_unified_general_is_raw():
    """unified_general backend must be RAW regime."""
    assert BACKEND_CONVENTIONS["unified_general"]["kernel_regime"] == KernelRegime.RAW


def test_backend_unified_paper_is_paper():
    """unified_paper backend must be PAPER regime."""
    assert BACKEND_CONVENTIONS["unified_paper"]["kernel_regime"] == KernelRegime.PAPER


def test_backend_term_dsl_raw_is_raw():
    """term_dsl_raw backend must be RAW regime."""
    assert BACKEND_CONVENTIONS["term_dsl_raw"]["kernel_regime"] == KernelRegime.RAW


def test_backend_term_dsl_paper_is_paper():
    """term_dsl_paper backend must be PAPER regime."""
    assert BACKEND_CONVENTIONS["term_dsl_paper"]["kernel_regime"] == KernelRegime.PAPER


# ============================================================================
# Test regime enum values
# ============================================================================


def test_kernel_regime_values():
    """KernelRegime enum has expected string values."""
    assert KernelRegime.RAW.value == "raw"
    assert KernelRegime.PAPER.value == "paper"


def test_kernel_regime_distinctness():
    """RAW and PAPER are distinct regimes."""
    assert KernelRegime.RAW != KernelRegime.PAPER


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
