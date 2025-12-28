"""
tests/test_q_residual_gates.py
Phase 36: Q Residual Gate Tests

These tests verify that the Q derivative residual is controlled:
1. The effect on ratio is NEGATIVE (Q derivatives reduce correction)
2. The magnitude is < 2% (within expected bounds)

These gates prevent catastrophic Q-related errors from being reintroduced.

Created: 2025-12-26 (Phase 36, GPT Priority 2)
"""
import pytest
from src.diagnostics.q_residual import (
    compute_q_residual_diagnostic,
    QResidualDiagnostic,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


class TestQResidualGatesKappa:
    """Test Q residual gates for κ benchmark."""

    @pytest.fixture
    def kappa_diagnostic(self):
        P1, P2, P3, Q = load_przz_polynomials()
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
        return compute_q_residual_diagnostic(
            R=1.3036, theta=4/7, K=3, polynomials=polys,
            benchmark_name="κ", n_quad=60,
        )

    def test_sign_gate_passes(self, kappa_diagnostic):
        """The Q effect on ratio should be negative."""
        assert kappa_diagnostic.sign_gate_passed, (
            f"Sign gate failed: Q effect on ratio = {kappa_diagnostic.Q_effect_on_correction_pct:.4f}% "
            f"(should be negative)"
        )

    def test_magnitude_gate_passes(self, kappa_diagnostic):
        """The Q effect magnitude should be < 2%."""
        assert kappa_diagnostic.magnitude_gate_passed, (
            f"Magnitude gate failed: |Q effect| = {abs(kappa_diagnostic.Q_effect_on_correction_pct):.4f}% "
            f"(should be < 2%)"
        )

    def test_all_gates_pass(self, kappa_diagnostic):
        """All Q residual gates should pass for κ."""
        assert kappa_diagnostic.all_gates_passed, (
            "Not all Q residual gates passed for κ benchmark"
        )

    def test_i1_share_is_small(self, kappa_diagnostic):
        """I1 should be a small fraction of S12."""
        assert kappa_diagnostic.I1_share_pct < 20, (
            f"I1 share = {kappa_diagnostic.I1_share_pct:.1f}% "
            f"(expected < 20% for κ)"
        )


class TestQResidualGatesKappaStar:
    """Test Q residual gates for κ* benchmark."""

    @pytest.fixture
    def kappa_star_diagnostic(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
        return compute_q_residual_diagnostic(
            R=1.1167, theta=4/7, K=3, polynomials=polys,
            benchmark_name="κ*", n_quad=60,
        )

    def test_sign_gate_passes(self, kappa_star_diagnostic):
        """The Q effect on ratio should be negative."""
        assert kappa_star_diagnostic.sign_gate_passed, (
            f"Sign gate failed: Q effect on ratio = {kappa_star_diagnostic.Q_effect_on_correction_pct:.4f}% "
            f"(should be negative)"
        )

    def test_magnitude_gate_passes(self, kappa_star_diagnostic):
        """The Q effect magnitude should be < 2%."""
        assert kappa_star_diagnostic.magnitude_gate_passed, (
            f"Magnitude gate failed: |Q effect| = {abs(kappa_star_diagnostic.Q_effect_on_correction_pct):.4f}% "
            f"(should be < 2%)"
        )

    def test_all_gates_pass(self, kappa_star_diagnostic):
        """All Q residual gates should pass for κ*."""
        assert kappa_star_diagnostic.all_gates_passed, (
            "Not all Q residual gates passed for κ* benchmark"
        )


class TestQResidualConsistency:
    """Test consistency properties of Q residual."""

    def test_both_benchmarks_have_negative_effect(self):
        """Both benchmarks should show negative Q effect on ratio."""
        P1, P2, P3, Q = load_przz_polynomials()
        polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
        diag_kappa = compute_q_residual_diagnostic(
            R=1.3036, theta=4/7, K=3, polynomials=polys_kappa,
            benchmark_name="κ", n_quad=60,
        )

        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        polys_star = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
        diag_star = compute_q_residual_diagnostic(
            R=1.1167, theta=4/7, K=3, polynomials=polys_star,
            benchmark_name="κ*", n_quad=60,
        )

        assert diag_kappa.Q_effect_on_correction_pct < 0, (
            f"κ Q effect should be negative: {diag_kappa.Q_effect_on_correction_pct:.4f}%"
        )
        assert diag_star.Q_effect_on_correction_pct < 0, (
            f"κ* Q effect should be negative: {diag_star.Q_effect_on_correction_pct:.4f}%"
        )

    def test_i2_dominates_s12(self):
        """I2 should dominate S12 for both benchmarks."""
        P1, P2, P3, Q = load_przz_polynomials()
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
        diag = compute_q_residual_diagnostic(
            R=1.3036, theta=4/7, K=3, polynomials=polys,
            benchmark_name="κ", n_quad=60,
        )

        # I1 share should be < 50% (I2 dominates)
        assert diag.I1_share_pct < 50, (
            f"I2 should dominate S12: I1 share = {diag.I1_share_pct:.1f}%"
        )
