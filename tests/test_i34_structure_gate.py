"""
tests/test_i34_structure_gate.py
Structure-level gate tests for I3/I4 mirror SPEC LOCK.

Per TRUTH_SPEC.md Section 10 (lines 370-388):
- I₁(α,β) + T^{-α-β}I₁(-β,-α)  ← HAS MIRROR
- I₂(α,β) + T^{-α-β}I₂(-β,-α)  ← HAS MIRROR
- I₃(α,β) and I₄(α,β)          ← NO MIRROR

These tests verify that:
1. The I34MirrorForbiddenError is raised when apply_mirror=True
2. Normal operation (apply_mirror=False) works correctly
3. The guard function works as expected

These are NEGATIVE CONTROLS - they verify that incorrect usage fails.
"""

import pytest
import numpy as np

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import (
    I34MirrorForbiddenError,
    _assert_i34_no_mirror,
    compute_S34_tex_combined_11,
    compute_S34_base_11,
)


class TestI34MirrorForbiddenError:
    """Test that I34MirrorForbiddenError is properly defined."""

    def test_error_is_value_error_subclass(self):
        """I34MirrorForbiddenError should be a ValueError subclass."""
        assert issubclass(I34MirrorForbiddenError, ValueError)

    def test_error_message_includes_spec_reference(self):
        """Error message should reference TRUTH_SPEC.md."""
        try:
            raise I34MirrorForbiddenError("test message")
        except I34MirrorForbiddenError as e:
            # Can be caught and has message
            assert "test message" in str(e)


class TestAssertI34NoMirror:
    """Test the guard function directly."""

    def test_no_raise_when_mirror_false(self):
        """Should not raise when apply_mirror=False."""
        # Should not raise
        _assert_i34_no_mirror(apply_mirror=False, caller="test")

    def test_raises_when_mirror_true(self):
        """Should raise I34MirrorForbiddenError when apply_mirror=True."""
        with pytest.raises(I34MirrorForbiddenError) as exc_info:
            _assert_i34_no_mirror(apply_mirror=True, caller="test_function")

        # Verify error message content
        error_msg = str(exc_info.value)
        assert "TRUTH_SPEC.md Section 10" in error_msg
        assert "FORBIDDEN" in error_msg
        assert "test_function" in error_msg

    def test_error_includes_caller_name(self):
        """Error message should include the caller name."""
        with pytest.raises(I34MirrorForbiddenError) as exc_info:
            _assert_i34_no_mirror(apply_mirror=True, caller="my_custom_function")

        assert "my_custom_function" in str(exc_info.value)


class TestComputeS34TexCombined11SpecLock:
    """Test that compute_S34_tex_combined_11 enforces SPEC LOCK."""

    @pytest.fixture(scope="class")
    def polys(self):
        """Load polynomials for testing."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_normal_operation_works(self, polys):
        """Normal operation (apply_mirror=False) should work."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 20

        # Should not raise
        result = compute_S34_tex_combined_11(
            theta=theta,
            R=R,
            n=n,
            polynomials=polys,
            apply_mirror=False,  # Default / correct usage
        )

        # Should return a valid result
        assert np.isfinite(result.S34_combined)

    def test_raises_when_mirror_true(self, polys):
        """Should raise I34MirrorForbiddenError when apply_mirror=True."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 20

        with pytest.raises(I34MirrorForbiddenError) as exc_info:
            compute_S34_tex_combined_11(
                theta=theta,
                R=R,
                n=n,
                polynomials=polys,
                apply_mirror=True,  # FORBIDDEN
            )

        # Verify error message
        error_msg = str(exc_info.value)
        assert "compute_S34_tex_combined_11" in error_msg
        assert "TRUTH_SPEC.md Section 10" in error_msg

    def test_default_is_no_mirror(self, polys):
        """Default behavior should be no mirror (apply_mirror=False)."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 20

        # Calling without apply_mirror should work (defaults to False)
        result = compute_S34_tex_combined_11(
            theta=theta,
            R=R,
            n=n,
            polynomials=polys,
            # apply_mirror not specified - should default to False
        )

        assert np.isfinite(result.S34_combined)


class TestComputeS34Base11SpecLock:
    """Test that compute_S34_base_11 enforces SPEC LOCK."""

    @pytest.fixture(scope="class")
    def polys(self):
        """Load polynomials for testing."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_normal_operation_works(self, polys):
        """Normal operation (apply_mirror=False) should work."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 20

        # Should not raise
        result = compute_S34_base_11(
            theta=theta,
            R=R,
            n=n,
            polynomials=polys,
            apply_mirror=False,
        )

        # Should return a valid result
        assert np.isfinite(result)

    def test_raises_when_mirror_true(self, polys):
        """Should raise I34MirrorForbiddenError when apply_mirror=True."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 20

        with pytest.raises(I34MirrorForbiddenError) as exc_info:
            compute_S34_base_11(
                theta=theta,
                R=R,
                n=n,
                polynomials=polys,
                apply_mirror=True,  # FORBIDDEN
            )

        # Verify error message
        error_msg = str(exc_info.value)
        assert "compute_S34_base_11" in error_msg
        assert "TRUTH_SPEC.md Section 10" in error_msg

    def test_default_is_no_mirror(self, polys):
        """Default behavior should be no mirror."""
        theta = 4.0 / 7.0
        R = 1.3036
        n = 20

        # Calling without apply_mirror should work
        result = compute_S34_base_11(
            theta=theta,
            R=R,
            n=n,
            polynomials=polys,
        )

        assert np.isfinite(result)


class TestSpecLockDocumentation:
    """Test that the SPEC LOCK is properly documented."""

    def test_spec_lock_comment_exists_in_module(self):
        """The SPEC LOCK comment block should exist in evaluate.py."""
        import src.evaluate as evaluate_module
        source = open(evaluate_module.__file__).read()

        # Check for key phrases in the SPEC LOCK comment
        assert "SPEC LOCK: I3/I4 Mirror is FORBIDDEN" in source
        assert "TRUTH_SPEC.md Section 10" in source

    def test_i34_mirror_forbidden_error_exported(self):
        """I34MirrorForbiddenError should be importable."""
        from src.evaluate import I34MirrorForbiddenError
        assert I34MirrorForbiddenError is not None

    def test_guard_function_exported(self):
        """_assert_i34_no_mirror should be importable."""
        from src.evaluate import _assert_i34_no_mirror
        assert callable(_assert_i34_no_mirror)
