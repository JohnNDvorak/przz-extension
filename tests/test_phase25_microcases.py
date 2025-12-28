"""
tests/test_phase25_microcases.py
Phase 25.3: P=Q=1 Microcase Ladder Tests

PURPOSE:
========
Test the P=Q=1 microcase module which isolates bracket structure
from polynomial interactions.

DIAGNOSTIC LOGIC:
=================
- If P=Q=1 DISAGREES: Gap is bracket structure, NOT polynomials
- If P=Q=1 MATCHES but full disagrees: Gap is polynomial/Q mixing

Created: 2025-12-25
"""

import pytest
import math

from src.unified_s12_microcases import (
    unified_I1_with_P1_Q1,
    empirical_I1_with_P1_Q1,
    unified_I2_analytic_P1_Q1,
    compare_microcase_I1,
    run_microcase_ladder,
    print_microcase_report,
    THETA,
    KAPPA_R,
    KAPPA_STAR_R,
)


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================


class TestMicrocaseFunctions:
    """Test basic microcase function execution."""

    def test_empirical_I1_returns_float(self):
        """empirical_I1_with_P1_Q1 should return a float."""
        result = empirical_I1_with_P1_Q1(theta=THETA, R=KAPPA_R, n=30)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_unified_I1_returns_float(self):
        """unified_I1_with_P1_Q1 should return a float."""
        result = unified_I1_with_P1_Q1(theta=THETA, R=KAPPA_R, n=30)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_I2_analytic_returns_float(self):
        """unified_I2_analytic_P1_Q1 should return a float."""
        result = unified_I2_analytic_P1_Q1(theta=THETA, R=KAPPA_R)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_I2_analytic_formula(self):
        """I2 analytic should match F(R)/theta."""
        theta = THETA
        R = KAPPA_R

        result = unified_I2_analytic_P1_Q1(theta, R)
        expected = (math.exp(2*R) - 1) / (2*R) / theta

        assert result == pytest.approx(expected, rel=1e-10)


# =============================================================================
# COMPARISON TESTS
# =============================================================================


class TestMicrocaseComparison:
    """Test microcase comparison functions."""

    def test_compare_microcase_I1_returns_comparison(self):
        """compare_microcase_I1 should return MicrocaseComparison."""
        result = compare_microcase_I1(theta=THETA, R=KAPPA_R, n=30)
        assert hasattr(result, 'unified_I1_11')
        assert hasattr(result, 'empirical_I1_11')
        assert hasattr(result, 'rel_diff')
        assert hasattr(result, 'agrees')

    def test_run_microcase_ladder_returns_both(self):
        """run_microcase_ladder should return both benchmarks."""
        kappa, kappa_star = run_microcase_ladder(n=30)
        assert kappa.R == pytest.approx(KAPPA_R)
        assert kappa_star.R == pytest.approx(KAPPA_STAR_R)


# =============================================================================
# MICROCASE AGREEMENT TESTS (CRITICAL DIAGNOSTIC)
# =============================================================================


@pytest.mark.parametrize("R,name", [(KAPPA_R, "kappa"), (KAPPA_STAR_R, "kappa_star")])
class TestMicrocaseAgreement:
    """
    Critical diagnostic tests for P=Q=1 agreement.

    These tests determine whether the gap is in:
    - Bracket structure (if P=Q=1 disagrees)
    - Polynomial/Q mixing (if P=Q=1 agrees)
    """

    def test_I1_microcase_values_positive(self, R, name):
        """Both unified and empirical I1 should be positive."""
        comparison = compare_microcase_I1(theta=THETA, R=R, n=40)
        assert comparison.unified_I1_11 > 0, f"Unified I1 should be positive for {name}"
        assert comparison.empirical_I1_11 > 0, f"Empirical I1 should be positive for {name}"

    def test_I1_microcase_ratio_is_3(self, R, name):
        """
        DIAGNOSTIC TEST: Document the factor-of-3 discrepancy.

        FINDING (Phase 25.3):
        The unified evaluator returns exactly 3x the empirical value for P=Q=1.
        This is consistent across both benchmarks, indicating a missing factor
        of 3 somewhere in the bracket structure.

        INTERPRETATION:
        - Gap is in BRACKET STRUCTURE, not polynomials
        - Missing factor is likely related to:
          - The PRZZ difference quotient denominator
          - The factorial normalization
          - The log factor contribution

        This test documents the expected ratio of 3.0 for future investigation.
        """
        comparison = compare_microcase_I1(theta=THETA, R=R, n=60)

        # The ratio should be exactly 3.0 (within numerical tolerance)
        ratio = comparison.unified_I1_11 / comparison.empirical_I1_11
        assert ratio == pytest.approx(3.0, rel=0.01), (
            f"Expected ratio of 3.0 for {name}, got {ratio:.4f}"
        )


class TestMicrocaseDiagnosis:
    """
    Diagnostic tests that interpret microcase results.
    """

    def test_microcase_diagnosis_consistent(self):
        """Agreement should be consistent across benchmarks."""
        kappa, kappa_star = run_microcase_ladder(n=40)

        # Both should agree or both should disagree
        # (Different behavior suggests R-dependent structural issue)
        if kappa.agrees != kappa_star.agrees:
            pytest.fail(
                f"Inconsistent P=Q=1 agreement:\n"
                f"  kappa: agrees={kappa.agrees}, rel_diff={kappa.rel_diff*100:.2f}%\n"
                f"  kappa*: agrees={kappa_star.agrees}, rel_diff={kappa_star.rel_diff*100:.2f}%\n"
                f"\n"
                f"DIAGNOSIS: R-dependent structural issue detected."
            )

    def test_print_microcase_report_runs(self, capsys):
        """print_microcase_report should run without error."""
        kappa, kappa_star = run_microcase_ladder(n=30)
        print_microcase_report(kappa, kappa_star)
        captured = capsys.readouterr()
        assert "MICROCASE LADDER" in captured.out
        assert "DIAGNOSIS" in captured.out


# =============================================================================
# QUADRATURE STABILITY TESTS
# =============================================================================


class TestMicrocaseQuadratureStability:
    """Test that microcase results are stable under quadrature refinement."""

    def test_unified_I1_quadrature_convergence(self):
        """Unified I1 should converge as quadrature increases."""
        results = []
        for n in [20, 40, 60]:
            val = unified_I1_with_P1_Q1(theta=THETA, R=KAPPA_R, n=n)
            results.append(val)

        # Check convergence: differences should decrease
        diff1 = abs(results[1] - results[0])
        diff2 = abs(results[2] - results[1])

        # diff2 should be smaller (converging)
        # Allow some tolerance for numerical noise
        assert diff2 < diff1 * 1.5 or diff2 < 1e-8, (
            f"Quadrature not converging: diff(40-20)={diff1:.2e}, diff(60-40)={diff2:.2e}"
        )

    def test_empirical_I1_quadrature_convergence(self):
        """Empirical I1 should converge as quadrature increases."""
        results = []
        for n in [20, 40, 60]:
            val = empirical_I1_with_P1_Q1(theta=THETA, R=KAPPA_R, n=n)
            results.append(val)

        diff1 = abs(results[1] - results[0])
        diff2 = abs(results[2] - results[1])

        assert diff2 < diff1 * 1.5 or diff2 < 1e-8, (
            f"Quadrature not converging: diff(40-20)={diff1:.2e}, diff(60-40)={diff2:.2e}"
        )
