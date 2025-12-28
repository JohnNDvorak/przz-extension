#!/usr/bin/env python3
"""
tests/test_polynomial_scorer.py
Phase 47: Polynomial Scorer Tests

Tests for the polynomial scoring harness.

Created: 2025-12-27 (Phase 47)
"""

import pytest
import numpy as np
from typing import Dict

from src.polynomial_scorer import (
    PolynomialScorer,
    ContractResult,
    LadderResult,
    SweepResult,
    ConfirmationResult,
    TwoBenchmarkResult,
    score_przz_polynomials,
    quick_score,
    KAPPA_TARGET,
    C_TARGET_KAPPA,
)
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
    Polynomial,
    P1Polynomial,
    PellPolynomial,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def przz_polynomials() -> Dict:
    """Load PRZZ κ polynomials."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def przz_polynomials_kappa_star() -> Dict:
    """Load PRZZ κ* polynomials."""
    P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=False)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def scorer() -> PolynomialScorer:
    """Create a standard scorer."""
    return PolynomialScorer(K=3, theta=4/7)


@pytest.fixture
def bad_polynomials() -> Dict:
    """Create polynomials that violate constraints."""
    # P1 that doesn't satisfy P1(0)=0
    bad_P1 = Polynomial(np.array([1.0, 1.0]))  # P1(0) = 1 ≠ 0
    P2 = PellPolynomial(tilde_coeffs=np.array([1.0]))
    P3 = PellPolynomial(tilde_coeffs=np.array([1.0]))
    Q = Polynomial(np.array([1.0]))
    return {"P1": bad_P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def nan_polynomials() -> Dict:
    """Create polynomials with NaN coefficients."""
    P1 = P1Polynomial(tilde_coeffs=np.array([np.nan]))
    P2 = PellPolynomial(tilde_coeffs=np.array([1.0]))
    P3 = PellPolynomial(tilde_coeffs=np.array([1.0]))
    Q = Polynomial(np.array([1.0]))
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


# =============================================================================
# STAGE A: CONTRACT VALIDATION TESTS
# =============================================================================

class TestContractValidation:
    """Tests for Stage A: Contract validation."""

    def test_przz_passes_contract(self, scorer, przz_polynomials):
        """PRZZ polynomials should pass all contract checks."""
        result = scorer.validate_contract(przz_polynomials)

        assert result.passed, f"PRZZ failed contract: {result.message}"
        assert result.message == "Contract passed"
        assert all(result.checks.values()), f"Some checks failed: {result.checks}"

    def test_bad_p1_fails_contract(self, scorer, bad_polynomials):
        """Polynomials with P1(0) ≠ 0 should fail."""
        result = scorer.validate_contract(bad_polynomials)

        assert not result.passed
        assert "P1(0)" in result.message

    def test_nan_coeffs_fail_contract(self, scorer, nan_polynomials):
        """Polynomials with NaN coefficients should fail."""
        result = scorer.validate_contract(nan_polynomials)

        assert not result.passed
        # NaN coefficients cause either direct NaN detection or boundary check failure
        assert "NaN" in result.message or "nan" in result.message.lower()

    def test_extreme_coeffs_fail_contract(self, scorer):
        """Polynomials with extreme coefficients should fail."""
        P1 = P1Polynomial(tilde_coeffs=np.array([1e7]))  # Extreme
        P2 = PellPolynomial(tilde_coeffs=np.array([1.0]))
        P3 = PellPolynomial(tilde_coeffs=np.array([1.0]))
        Q = Polynomial(np.array([1.0]))
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = scorer.validate_contract(polynomials)

        assert not result.passed
        assert "extreme" in result.message.lower()

    def test_bad_q0_fails_contract(self, scorer):
        """Polynomials with Q(0) ≠ 1 should fail."""
        P1 = P1Polynomial(tilde_coeffs=np.array([0.0]))
        P2 = PellPolynomial(tilde_coeffs=np.array([1.0]))
        P3 = PellPolynomial(tilde_coeffs=np.array([1.0]))
        Q = Polynomial(np.array([0.5]))  # Q(0) = 0.5 ≠ 1
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = scorer.validate_contract(polynomials)

        assert not result.passed
        assert "Q(0)" in result.message


# =============================================================================
# STAGE B: MICROCASE LADDER TESTS
# =============================================================================

class TestMicrocaseLadder:
    """Tests for Stage B: Microcase ladder."""

    def test_przz_passes_ladder(self, scorer, przz_polynomials):
        """PRZZ polynomials should pass microcase ladder."""
        result = scorer.run_microcase_ladder(przz_polynomials, n_quad=30)

        assert result.ladder_valid, f"Ladder failed: {result.validation_message}"
        assert len(result.cases) == 4  # P=Q=1, P=real/Q=1, P=1/Q=real, P=Q=real

    def test_ladder_all_cases_succeed(self, scorer, przz_polynomials):
        """All microcase ladder cases should succeed."""
        result = scorer.run_microcase_ladder(przz_polynomials, n_quad=30)

        for name, case in result.cases.items():
            assert case.success, f"Case {name} failed: {case.error}"
            assert np.isfinite(case.c), f"Case {name} has invalid c"
            assert case.c > 0, f"Case {name} has c <= 0"

    def test_ladder_c_progression(self, scorer, przz_polynomials):
        """c values should be in reasonable range."""
        result = scorer.run_microcase_ladder(przz_polynomials, n_quad=30)

        c_values = [case.c for case in result.cases.values()]

        # All c values should be positive and reasonable
        assert all(c > 0.01 for c in c_values), f"Some c too small: {c_values}"
        assert all(c < 100 for c in c_values), f"Some c too large: {c_values}"


# =============================================================================
# STAGE C/D: R-SWEEP TESTS
# =============================================================================

class TestRSweep:
    """Tests for Stage C/D: R-sweep."""

    def test_fast_sweep_finds_optimum(self, scorer, przz_polynomials):
        """Fast R-sweep should find an optimum."""
        result = scorer.run_fast_r_sweep(
            przz_polynomials,
            R_range=(1.0, 1.5),
            n_points=10,
            n_quad=20,
        )

        assert len(result.points) > 0
        assert np.isfinite(result.R_opt)
        assert np.isfinite(result.c_opt)
        assert np.isfinite(result.kappa_opt)

    def test_fast_sweep_r_opt_in_range(self, scorer, przz_polynomials):
        """Optimal R should be within sweep range."""
        R_min, R_max = 1.0, 1.5
        result = scorer.run_fast_r_sweep(
            przz_polynomials,
            R_range=(R_min, R_max),
            n_points=10,
            n_quad=20,
        )

        assert R_min <= result.R_opt <= R_max

    def test_slow_confirmation_convergence(self, scorer, przz_polynomials):
        """Slow confirmation should show convergence."""
        result = scorer.run_slow_confirmation(
            przz_polynomials,
            R_opt=1.3036,
            n_quad_levels=[30, 40, 50],
        )

        assert len(result.levels) == 3
        # Check max drift is reasonable
        assert result.max_drift < 0.1, f"Drift too high: {result.max_drift}"


# =============================================================================
# TWO-BENCHMARK GATE TESTS
# =============================================================================

class TestTwoBenchmarkGate:
    """Tests for two-benchmark gate."""

    def test_przz_passes_with_correct_polynomials(
        self, scorer, przz_polynomials, przz_polynomials_kappa_star
    ):
        """PRZZ polynomials should pass two-benchmark gate when using correct polys for each."""
        result = scorer.run_two_benchmark_gate(
            przz_polynomials,
            polynomials_kappa_star=przz_polynomials_kappa_star,
            n_quad=40,
        )

        assert result.gate_passed, (
            f"Gate failed:\n"
            f"  κ gap: {result.kappa['c_gap_pct']:.4f}%\n"
            f"  κ* gap: {result.kappa_star['c_gap_pct']:.4f}%\n"
            f"  ratio gap: {result.ratio_gap_pct:.4f}%"
        )

    def test_kappa_benchmark_accuracy(
        self, scorer, przz_polynomials, przz_polynomials_kappa_star
    ):
        """κ benchmark should match target within 1%."""
        result = scorer.run_two_benchmark_gate(
            przz_polynomials,
            polynomials_kappa_star=przz_polynomials_kappa_star,
            n_quad=40,
        )

        assert abs(result.kappa["c_gap_pct"]) < 1.0, (
            f"κ c gap too large: {result.kappa['c_gap_pct']:.4f}%"
        )

    def test_kappa_star_benchmark_accuracy(
        self, scorer, przz_polynomials, przz_polynomials_kappa_star
    ):
        """κ* benchmark should match target within 1%."""
        result = scorer.run_two_benchmark_gate(
            przz_polynomials,
            polynomials_kappa_star=przz_polynomials_kappa_star,
            n_quad=40,
        )

        assert abs(result.kappa_star["c_gap_pct"]) < 1.0, (
            f"κ* c gap too large: {result.kappa_star['c_gap_pct']:.4f}%"
        )

    def test_ratio_accuracy(
        self, scorer, przz_polynomials, przz_polynomials_kappa_star
    ):
        """c ratio should match target within 1%."""
        result = scorer.run_two_benchmark_gate(
            przz_polynomials,
            polynomials_kappa_star=przz_polynomials_kappa_star,
            n_quad=40,
        )

        assert abs(result.ratio_gap_pct) < 1.0, (
            f"Ratio gap too large: {result.ratio_gap_pct:.4f}%"
        )


# =============================================================================
# QUICK SCORE TESTS
# =============================================================================

class TestQuickScore:
    """Tests for the quick_score convenience function."""

    def test_quick_score_przz(self):
        """Quick score should work with PRZZ coefficients."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)

        result = quick_score(
            P1_coeffs=P1.tilde_coeffs.tolist(),
            P2_coeffs=P2.tilde_coeffs.tolist(),
            P3_coeffs=P3.tilde_coeffs.tolist(),
            Q_coeffs=Q.to_monomial().coeffs.tolist(),
            R=1.3036,
            n_quad=40,
        )

        assert np.isfinite(result.kappa)
        assert np.isfinite(result.c)
        # Should be close to target
        gap_pct = abs(result.c / C_TARGET_KAPPA - 1) * 100
        assert gap_pct < 1.0, f"c gap too large: {gap_pct:.4f}%"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full scoring pipeline."""

    @pytest.mark.slow
    def test_full_scoring_pipeline(self, przz_polynomials):
        """Full scoring pipeline should work end-to-end."""
        result = score_przz_polynomials(n_quad=40)

        # Contract should pass
        assert result.contract.passed

        # Ladder should pass
        assert result.ladder.ladder_valid

        # Should produce valid results
        assert np.isfinite(result.c_opt)
        assert np.isfinite(result.kappa_opt)
        assert result.R_opt > 0

    def test_scorer_reproducibility(self, scorer, przz_polynomials):
        """Same polynomials should give same score."""
        result1 = scorer.run_two_benchmark_gate(przz_polynomials, n_quad=30)
        result2 = scorer.run_two_benchmark_gate(przz_polynomials, n_quad=30)

        assert result1.kappa["c"] == result2.kappa["c"]
        assert result1.kappa_star["c"] == result2.kappa_star["c"]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_R(self, scorer, przz_polynomials):
        """Scorer should handle small R values."""
        result = scorer.run_fast_r_sweep(
            przz_polynomials,
            R_range=(0.5, 0.8),
            n_points=5,
            n_quad=20,
        )

        assert len(result.points) > 0
        assert all(np.isfinite(p.c) for p in result.points)

    def test_very_large_R(self, scorer, przz_polynomials):
        """Scorer should handle large R values."""
        result = scorer.run_fast_r_sweep(
            przz_polynomials,
            R_range=(1.5, 2.0),
            n_points=5,
            n_quad=20,
        )

        assert len(result.points) > 0
        assert all(np.isfinite(p.c) for p in result.points)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
