"""
tests/test_no_fitted_m1_in_production_paths.py
SPEC LOCK: Fitted m₁ is quarantined and cannot be used in production.

This test file enforces the Phase 8.1 hard guard against calibration creep.
The fitted m₁ = 1.037×exp(R)+5 achieves 0% gap but is NOT derived from
first principles. Using it in production code masks the derivation problem.

BACKGROUND (2025-12-22):
GPT WARNING:
> "Adding m1 = 1.037*exp(R)+5 is EXACTLY the kind of 'quiet calibration creep'
> you said you don't want. It's fine as a DIAGNOSTIC ARTIFACT (a clue!),
> but it must NOT become baseline behavior."

Reference: Plan file Phase 8.1
"""

import math
import pytest
import warnings

from src.m1_policy import (
    M1Mode,
    M1Policy,
    m1_formula,
    M1DiagnosticError,
    m1_diagnostic_fitted,
    M1_FITTED_COEFFICIENT_A,
    M1_FITTED_COEFFICIENT_B,
)


class TestDiagnosticGuard:
    """Test that DIAGNOSTIC_FITTED mode requires explicit opt-in."""

    def test_diagnostic_without_flag_raises(self):
        """DIAGNOSTIC_FITTED without allow_diagnostic=True must RAISE."""
        policy = M1Policy(mode=M1Mode.DIAGNOSTIC_FITTED)

        with pytest.raises(M1DiagnosticError) as exc_info:
            m1_formula(K=3, R=1.3036, policy=policy)

        # Verify error message mentions calibration creep
        assert "CALIBRATION CREEP" in str(exc_info.value)
        assert "allow_diagnostic=True" in str(exc_info.value)

    def test_diagnostic_with_flag_works(self):
        """DIAGNOSTIC_FITTED with allow_diagnostic=True should work (with warning)."""
        policy = M1Policy(mode=M1Mode.DIAGNOSTIC_FITTED, allow_diagnostic=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m1 = m1_formula(K=3, R=1.3036, policy=policy)

            # Should return the fitted value
            expected = M1_FITTED_COEFFICIENT_A * math.exp(1.3036) + M1_FITTED_COEFFICIENT_B
            assert abs(m1 - expected) < 1e-10

            # Should emit a warning
            assert len(w) >= 1
            assert "DIAGNOSTIC_FITTED" in str(w[0].message)
            assert "NOT derived from first principles" in str(w[0].message)

    def test_diagnostic_helper_function_emits_warning(self):
        """m1_diagnostic_fitted() should emit a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m1 = m1_diagnostic_fitted(R=1.3036)

            # Should return the fitted value
            expected = M1_FITTED_COEFFICIENT_A * math.exp(1.3036) + M1_FITTED_COEFFICIENT_B
            assert abs(m1 - expected) < 1e-10

            # Should emit a warning
            assert len(w) >= 1
            assert "DIAGNOSTIC ONLY" in str(w[0].message)


class TestProductionEvaluatorsUseEmpirical:
    """Verify production evaluators use empirical m₁, not fitted."""

    def test_compute_c_paper_ordered_uses_empirical(self):
        """
        compute_c_paper_ordered uses empirical m₁ = exp(R) + (2K-1).

        This test verifies that the production evaluator does NOT use
        the fitted m₁ formula (which would be calibration creep).
        """
        # Import here to avoid circular imports in test collection
        from src.polynomials import load_przz_polynomials
        from src.evaluate import compute_c_paper_ordered

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        theta = 4.0 / 7.0
        R = 1.3036
        K = 3

        # Expected empirical mirror multiplier
        m_empirical = math.exp(R) + (2 * K - 1)  # = exp(R) + 5

        # Expected fitted mirror multiplier (should NOT be used)
        m_fitted = M1_FITTED_COEFFICIENT_A * math.exp(R) + M1_FITTED_COEFFICIENT_B

        # Verify they differ (this is the whole point!)
        assert abs(m_empirical - m_fitted) > 0.1, \
            "Empirical and fitted m₁ should differ by ~0.13"

        # Run the evaluator
        result = compute_c_paper_ordered(
            theta=theta,
            R=R,
            n=40,  # Low n for speed
            polynomials=polynomials,
            K=K,
            s12_pair_mode="triangle",
        )

        # The evaluator uses the empirical formula, which gives ~-1.35% gap
        # If it used fitted, it would give ~0% gap
        c_target = 2.13745440613217263636
        gap_percent = (result.total - c_target) / c_target * 100

        # With empirical m₁: gap should be around -1.3%
        # With fitted m₁: gap would be ~0%
        assert gap_percent < -0.5, \
            f"Expected negative gap with empirical m₁, got {gap_percent:+.2f}%"
        assert gap_percent > -3.0, \
            f"Gap too large, got {gap_percent:+.2f}%"


class TestK3EmpiricalStillDefault:
    """Verify K3_EMPIRICAL is still the default production mode."""

    def test_k3_empirical_works_without_flags(self):
        """K3_EMPIRICAL should work without any special flags."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        m1 = m1_formula(K=3, R=1.3036, policy=policy)

        expected = math.exp(1.3036) + 5
        assert abs(m1 - expected) < 1e-10

    def test_k3_empirical_no_warning(self):
        """K3_EMPIRICAL should NOT emit warnings (it's the safe default)."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m1 = m1_formula(K=3, R=1.3036, policy=policy)

            # Should NOT emit any warnings for K3_EMPIRICAL
            assert len(w) == 0, f"K3_EMPIRICAL should not warn, got: {w}"


class TestErrorMessageQuality:
    """Test that error messages are helpful."""

    def test_diagnostic_error_explains_calibration_creep(self):
        """Error message should explain what calibration creep means."""
        policy = M1Policy(mode=M1Mode.DIAGNOSTIC_FITTED)

        with pytest.raises(M1DiagnosticError) as exc_info:
            m1_formula(K=3, R=1.3036, policy=policy)

        msg = str(exc_info.value)
        # Should mention what's wrong
        assert "NOT derived from first principles" in msg
        # Should mention the fix
        assert "allow_diagnostic=True" in msg

    def test_m1_diagnostic_error_is_value_error(self):
        """M1DiagnosticError should be a ValueError subclass."""
        assert issubclass(M1DiagnosticError, ValueError)


class TestDocumentation:
    """Test that documentation is correct."""

    def test_diagnostic_mode_exists(self):
        """DIAGNOSTIC_FITTED mode should exist."""
        assert hasattr(M1Mode, "DIAGNOSTIC_FITTED")
        assert M1Mode.DIAGNOSTIC_FITTED is not None

    def test_diagnostic_function_docstring_warns(self):
        """m1_diagnostic_fitted docstring should contain warning."""
        docstring = m1_diagnostic_fitted.__doc__
        assert "DIAGNOSTIC ONLY" in docstring
        assert "CALIBRATION CREEP" in docstring
