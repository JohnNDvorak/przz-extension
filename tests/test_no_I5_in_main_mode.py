"""
Tests for evaluation mode guardrails.

These tests ensure that I₅ and A-derivative contributions are properly
forbidden in MAIN_TERM_ONLY mode per TRUTH_SPEC.md Section 4.

References:
    TRUTH_SPEC.md Lines 1621-1628: I₅ is error term
    TRUTH_SPEC.md Lines 1722-1727: A-derivatives are error terms
"""

import pytest
import warnings

from src.evaluation_modes import (
    EvaluationMode,
    get_evaluation_mode,
    set_evaluation_mode,
    evaluation_mode_context,
    assert_main_term_only,
    check_no_a_derivatives,
    I5ForbiddenError,
    MAIN_MODE,
    ERROR_MODE,
)


class TestEvaluationModeBasics:
    """Basic mode get/set functionality."""

    def test_default_mode_is_main_term_only(self):
        """Default should be MAIN_TERM_ONLY for safety."""
        # Reset to default first
        set_evaluation_mode(EvaluationMode.MAIN_TERM_ONLY)
        assert get_evaluation_mode() == EvaluationMode.MAIN_TERM_ONLY

    def test_set_mode_returns_previous(self):
        """set_evaluation_mode returns the previous mode."""
        set_evaluation_mode(EvaluationMode.MAIN_TERM_ONLY)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            previous = set_evaluation_mode(EvaluationMode.WITH_ERROR_TERMS)
        assert previous == EvaluationMode.MAIN_TERM_ONLY

    def test_mode_aliases(self):
        """Convenience aliases work."""
        assert MAIN_MODE == EvaluationMode.MAIN_TERM_ONLY
        assert ERROR_MODE == EvaluationMode.WITH_ERROR_TERMS


class TestI5ForbiddenInMainMode:
    """I₅ is forbidden in MAIN_TERM_ONLY mode."""

    def setup_method(self):
        """Ensure we start in MAIN_TERM_ONLY mode."""
        set_evaluation_mode(EvaluationMode.MAIN_TERM_ONLY)

    def test_assert_main_term_only_raises_in_main_mode(self):
        """assert_main_term_only raises I5ForbiddenError in main mode."""
        with pytest.raises(I5ForbiddenError) as excinfo:
            assert_main_term_only("compute I₅ contribution")

        # Check error message references TRUTH_SPEC
        assert "TRUTH_SPEC" in str(excinfo.value)
        assert "1621" in str(excinfo.value) or "error-order" in str(excinfo.value)

    def test_assert_main_term_only_silent_in_error_mode(self):
        """assert_main_term_only is a no-op in WITH_ERROR_TERMS mode."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_evaluation_mode(EvaluationMode.WITH_ERROR_TERMS)

        # Should not raise
        assert_main_term_only("compute I₅ contribution")

    def test_i5_error_includes_operation_name(self):
        """Error message includes the operation that was forbidden."""
        with pytest.raises(I5ForbiddenError) as excinfo:
            assert_main_term_only("add J_{1,5} term to bracket")

        assert "J_{1,5}" in str(excinfo.value)


class TestADerivativesForbidden:
    """A^{(m,n)} with m+n > 0 are forbidden in main mode."""

    def setup_method(self):
        """Ensure we start in MAIN_TERM_ONLY mode."""
        set_evaluation_mode(EvaluationMode.MAIN_TERM_ONLY)

    def test_a_0_0_allowed(self):
        """A^{(0,0)} is allowed (no derivatives)."""
        # Should not raise
        check_no_a_derivatives(0, 0, "A evaluation")

    def test_a_1_0_forbidden(self):
        """A^{(1,0)} is forbidden."""
        with pytest.raises(I5ForbiddenError):
            check_no_a_derivatives(1, 0, "A derivative")

    def test_a_0_1_forbidden(self):
        """A^{(0,1)} is forbidden."""
        with pytest.raises(I5ForbiddenError):
            check_no_a_derivatives(0, 1, "A derivative")

    def test_a_1_1_forbidden(self):
        """A^{(1,1)} is forbidden (this is the J_{1,5} case)."""
        with pytest.raises(I5ForbiddenError) as excinfo:
            check_no_a_derivatives(1, 1, "J_{1,5} from A^{(1,1)}")

        assert "(1,1)" in str(excinfo.value)

    def test_a_derivatives_allowed_in_error_mode(self):
        """A-derivatives allowed in WITH_ERROR_TERMS mode."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_evaluation_mode(EvaluationMode.WITH_ERROR_TERMS)

        # All should be allowed
        check_no_a_derivatives(1, 0)
        check_no_a_derivatives(0, 1)
        check_no_a_derivatives(1, 1)
        check_no_a_derivatives(2, 2)


class TestModeContextManager:
    """Context manager for temporary mode changes."""

    def setup_method(self):
        """Ensure we start in MAIN_TERM_ONLY mode."""
        set_evaluation_mode(EvaluationMode.MAIN_TERM_ONLY)

    def test_context_manager_changes_mode(self):
        """Context manager changes mode inside block."""
        assert get_evaluation_mode() == EvaluationMode.MAIN_TERM_ONLY

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with evaluation_mode_context(EvaluationMode.WITH_ERROR_TERMS):
                assert get_evaluation_mode() == EvaluationMode.WITH_ERROR_TERMS
                # I₅ allowed here
                assert_main_term_only("test operation")  # Should not raise

    def test_context_manager_restores_mode(self):
        """Context manager restores original mode after block."""
        assert get_evaluation_mode() == EvaluationMode.MAIN_TERM_ONLY

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with evaluation_mode_context(EvaluationMode.WITH_ERROR_TERMS):
                pass

        # Should be restored
        assert get_evaluation_mode() == EvaluationMode.MAIN_TERM_ONLY

    def test_context_manager_restores_on_exception(self):
        """Context manager restores mode even if exception raised."""
        assert get_evaluation_mode() == EvaluationMode.MAIN_TERM_ONLY

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                with evaluation_mode_context(EvaluationMode.WITH_ERROR_TERMS):
                    raise RuntimeError("test error")

        # Should still be restored
        assert get_evaluation_mode() == EvaluationMode.MAIN_TERM_ONLY


class TestWarnings:
    """Warning behavior for mode switches."""

    def setup_method(self):
        """Ensure we start in MAIN_TERM_ONLY mode."""
        set_evaluation_mode(EvaluationMode.MAIN_TERM_ONLY)

    def test_setting_error_mode_warns(self):
        """Setting WITH_ERROR_TERMS mode emits a warning."""
        with pytest.warns(UserWarning) as record:
            set_evaluation_mode(EvaluationMode.WITH_ERROR_TERMS)

        assert len(record) == 1
        assert "error-order" in str(record[0].message).lower()
        assert "TRUTH_SPEC" in str(record[0].message)

    def test_setting_main_mode_no_warning(self):
        """Setting MAIN_TERM_ONLY mode does not warn."""
        # First set to error mode
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_evaluation_mode(EvaluationMode.WITH_ERROR_TERMS)

        # Now switch back - should not warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_evaluation_mode(EvaluationMode.MAIN_TERM_ONLY)
            # Filter to only UserWarnings about error-order
            relevant = [x for x in w if "error-order" in str(x.message).lower()]
            assert len(relevant) == 0


class TestIntegrationWithI5:
    """
    Integration tests simulating actual I₅ usage patterns.

    These tests don't actually compute I₅, but simulate the patterns
    that would be used in the real computation.
    """

    def setup_method(self):
        """Ensure we start in MAIN_TERM_ONLY mode."""
        set_evaluation_mode(EvaluationMode.MAIN_TERM_ONLY)

    def test_simulated_i5_computation_blocked(self):
        """Simulated I₅ computation is blocked in main mode."""

        def compute_i5_contribution():
            """Simulate I₅ computation."""
            assert_main_term_only("I₅ prime sum")
            return 0.001  # Would be actual computation

        with pytest.raises(I5ForbiddenError):
            compute_i5_contribution()

    def test_simulated_i5_computation_allowed_in_context(self):
        """Simulated I₅ computation allowed in error mode context."""

        def compute_i5_contribution():
            """Simulate I₅ computation."""
            assert_main_term_only("I₅ prime sum")
            return 0.001

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with evaluation_mode_context(EvaluationMode.WITH_ERROR_TERMS):
                result = compute_i5_contribution()
                assert result == 0.001

    def test_j15_decomposition_pattern(self):
        """Test pattern for J₁ decomposition with J_{1,5}."""

        def compute_j1_decomposition(include_j15: bool = False):
            """Simulate J₁ decomposition."""
            result = 1.0  # J_{1,1} + J_{1,2} + J_{1,3} + J_{1,4}

            if include_j15:
                check_no_a_derivatives(1, 1, "J_{1,5}")
                result += 0.01  # J_{1,5} contribution

            return result

        # Without J15 - should work
        assert compute_j1_decomposition(include_j15=False) == 1.0

        # With J15 in main mode - should fail
        with pytest.raises(I5ForbiddenError):
            compute_j1_decomposition(include_j15=True)

        # With J15 in error mode - should work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with evaluation_mode_context(EvaluationMode.WITH_ERROR_TERMS):
                result = compute_j1_decomposition(include_j15=True)
                assert result == 1.01


class TestErrorMessages:
    """Error messages are informative and actionable."""

    def setup_method(self):
        """Ensure we start in MAIN_TERM_ONLY mode."""
        set_evaluation_mode(EvaluationMode.MAIN_TERM_ONLY)

    def test_error_message_has_fix_suggestion(self):
        """Error message includes how to use error mode."""
        with pytest.raises(I5ForbiddenError) as excinfo:
            assert_main_term_only("test operation")

        msg = str(excinfo.value)
        assert "evaluation_mode_context" in msg
        assert "WITH_ERROR_TERMS" in msg

    def test_error_message_references_truth_spec(self):
        """Error message references TRUTH_SPEC lines."""
        with pytest.raises(I5ForbiddenError) as excinfo:
            assert_main_term_only("test operation")

        msg = str(excinfo.value)
        assert "TRUTH_SPEC" in msg
        assert "1621" in msg or "Lines" in msg
