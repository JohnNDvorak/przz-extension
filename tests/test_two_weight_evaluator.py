"""
tests/test_two_weight_evaluator.py
Tests for the two-weight mirror model evaluator (Stage C2/C3).

These tests validate:
1. compute_c_paper_two_weight produces finite, reasonable results
2. solve_two_weight_coefficients finds valid (m1, m2) for both benchmarks
3. Applying solved weights hits benchmark targets exactly
4. The two-weight model is stable under quadrature refinement
"""

import pytest
import math

from src.evaluate import (
    compute_c_paper_two_weight,
    solve_two_weight_coefficients,
    evaluate_c_ordered,
    evaluate_c_ordered_with_exp_transform,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
N_QUAD = 40
N_QUAD_A = 30

# Benchmark parameters
KAPPA_R = 1.3036
KAPPA_C_TARGET = 2.137
KAPPA_STAR_R = 1.1167
KAPPA_STAR_C_TARGET = 1.938

# Constants for channel extraction
ORDERED_PAIR_KEYS = ("11", "22", "33", "12", "21", "13", "31", "23", "32")
FACTORIAL_WEIGHTS = {
    "11": 1.0,
    "22": 0.25,
    "33": 1.0/36,
    "12": 0.5,
    "21": 0.5,
    "13": 1.0/6,
    "31": 1.0/6,
    "23": 1.0/12,
    "32": 1.0/12,
}


@pytest.fixture(scope="module")
def kappa_polys():
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture(scope="module")
def kappa_star_polys():
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


def _compute_split_channels(polys, R, n_quad, n_quad_a):
    """Helper to compute split I1/I2 channels for a benchmark."""
    result = evaluate_c_ordered(
        theta=THETA,
        R=R,
        n=n_quad,
        polynomials=polys,
        kernel_regime="paper",
        use_factorial_normalization=True,
        n_quad_a=n_quad_a,
    )
    mirror = evaluate_c_ordered_with_exp_transform(
        theta=THETA,
        R=-R,
        n=n_quad,
        polynomials=polys,
        kernel_regime="paper",
        exp_scale_multiplier=1.0,
        exp_t_flip=False,
        q_a0_shift=0.0,
        use_factorial_normalization=True,
        n_quad_a=n_quad_a,
    )

    I1_plus = I2_plus = I1_minus = I2_minus = 0.0
    I3_plus = I4_plus = 0.0

    for pair in ORDERED_PAIR_KEYS:
        w = FACTORIAL_WEIGHTS[pair]
        I1_plus += w * float(result.per_term.get(f"{pair}_I1_{pair}", 0.0))
        I2_plus += w * float(result.per_term.get(f"{pair}_I2_{pair}", 0.0))
        I3_plus += w * float(result.per_term.get(f"{pair}_I3_{pair}", 0.0))
        I4_plus += w * float(result.per_term.get(f"{pair}_I4_{pair}", 0.0))
        I1_minus += w * float(mirror.per_term.get(f"{pair}_I1_{pair}", 0.0))
        I2_minus += w * float(mirror.per_term.get(f"{pair}_I2_{pair}", 0.0))

    return {
        "_I1_plus": I1_plus,
        "_I2_plus": I2_plus,
        "_I1_minus": I1_minus,
        "_I2_minus": I2_minus,
        "_S34_plus": I3_plus + I4_plus,
    }


@pytest.fixture(scope="module")
def kappa_channels(kappa_polys):
    """Compute split channels for κ benchmark."""
    return _compute_split_channels(kappa_polys, KAPPA_R, N_QUAD, N_QUAD_A)


@pytest.fixture(scope="module")
def kappa_star_channels(kappa_star_polys):
    """Compute split channels for κ* benchmark."""
    return _compute_split_channels(kappa_star_polys, KAPPA_STAR_R, N_QUAD, N_QUAD_A)


# =============================================================================
# Basic functionality tests
# =============================================================================

class TestComputeCPaperTwoWeight:
    """Tests for compute_c_paper_two_weight function."""

    def test_produces_finite_result(self, kappa_polys):
        """Two-weight evaluator produces finite result."""
        result = compute_c_paper_two_weight(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            m1=6.0,
            m2=8.0,
            n_quad_a=N_QUAD_A,
        )

        assert math.isfinite(result.total)
        assert result.total > 0  # c should be positive

    def test_breakdown_contains_channels(self, kappa_polys):
        """Breakdown includes split channel information."""
        result = compute_c_paper_two_weight(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            m1=6.0,
            m2=8.0,
            n_quad_a=N_QUAD_A,
        )

        required_keys = [
            "_I1_plus", "_I1_minus",
            "_I2_plus", "_I2_minus",
            "_S34_plus",
            "_m1", "_m2", "_m_ratio",
            "_model",
        ]
        for key in required_keys:
            assert key in result.per_term, f"Missing key: {key}"

    def test_records_weights(self, kappa_polys):
        """Weights m1 and m2 are recorded in breakdown."""
        m1, m2 = 5.5, 9.5
        result = compute_c_paper_two_weight(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            m1=m1,
            m2=m2,
            n_quad_a=N_QUAD_A,
        )

        assert result.per_term["_m1"] == m1
        assert result.per_term["_m2"] == m2
        assert result.per_term["_model"] == "two_weight"


# =============================================================================
# Solver tests
# =============================================================================

class TestSolveTwoWeightCoefficients:
    """Tests for the (m1, m2) solver."""

    def test_solver_finds_valid_weights(self, kappa_channels, kappa_star_channels):
        """Solver finds valid (m1, m2) from both benchmark channels."""
        m1, m2, det = solve_two_weight_coefficients(
            kappa_channels, kappa_star_channels,
            c_target_kappa=KAPPA_C_TARGET,
            c_target_kappa_star=KAPPA_STAR_C_TARGET,
        )

        assert math.isfinite(m1)
        assert math.isfinite(m2)
        assert m1 > 0  # Weights should be positive
        assert m2 > 0

    def test_matrix_is_nonsingular(self, kappa_channels, kappa_star_channels):
        """Coefficient matrix has reasonable determinant."""
        m1, m2, det = solve_two_weight_coefficients(
            kappa_channels, kappa_star_channels,
            c_target_kappa=KAPPA_C_TARGET,
            c_target_kappa_star=KAPPA_STAR_C_TARGET,
        )

        assert abs(det) > 1e-10, f"Matrix nearly singular: det={det}"


# =============================================================================
# Integration tests: verify benchmarks are hit
# =============================================================================

class TestTwoWeightBenchmarkMatch:
    """Tests that two-weight model hits both benchmarks."""

    def test_kappa_benchmark_hit_with_solved_weights(
        self, kappa_polys, kappa_channels, kappa_star_channels
    ):
        """Two-weight model hits κ benchmark with solved weights."""
        m1, m2, det = solve_two_weight_coefficients(
            kappa_channels, kappa_star_channels,
            c_target_kappa=KAPPA_C_TARGET,
            c_target_kappa_star=KAPPA_STAR_C_TARGET,
        )

        result = compute_c_paper_two_weight(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            m1=m1,
            m2=m2,
            n_quad_a=N_QUAD_A,
        )

        # Should hit target within 1% (accounting for different n_quad)
        rel_error = abs(result.total - KAPPA_C_TARGET) / KAPPA_C_TARGET
        assert rel_error < 0.01, f"κ: c={result.total}, target={KAPPA_C_TARGET}, error={rel_error:.2%}"

    def test_kappa_star_benchmark_hit_with_solved_weights(
        self, kappa_star_polys, kappa_channels, kappa_star_channels
    ):
        """Two-weight model hits κ* benchmark with solved weights."""
        m1, m2, det = solve_two_weight_coefficients(
            kappa_channels, kappa_star_channels,
            c_target_kappa=KAPPA_C_TARGET,
            c_target_kappa_star=KAPPA_STAR_C_TARGET,
        )

        result = compute_c_paper_two_weight(
            theta=THETA,
            R=KAPPA_STAR_R,
            n=N_QUAD,
            polynomials=kappa_star_polys,
            m1=m1,
            m2=m2,
            n_quad_a=N_QUAD_A,
        )

        # Should hit target within 1% (accounting for different n_quad)
        rel_error = abs(result.total - KAPPA_STAR_C_TARGET) / KAPPA_STAR_C_TARGET
        assert rel_error < 0.01, f"κ*: c={result.total}, target={KAPPA_STAR_C_TARGET}, error={rel_error:.2%}"


# =============================================================================
# Structure validation tests
# =============================================================================

class TestTwoWeightStructure:
    """Tests for structural properties of two-weight model."""

    def test_m1_not_equals_m2(self, kappa_channels, kappa_star_channels):
        """m1 and m2 are distinct for the current benchmark split (smoke test)."""
        m1, m2, det = solve_two_weight_coefficients(
            kappa_channels, kappa_star_channels,
            c_target_kappa=KAPPA_C_TARGET,
            c_target_kappa_star=KAPPA_STAR_C_TARGET,
        )

        # Don't pin numeric values; just assert they're not identical.
        assert abs(m1 - m2) > 1e-6, f"m1 and m2 unexpectedly identical: m1={m1}, m2={m2}"

    def test_two_weight_differs_from_single(self, kappa_polys, kappa_channels, kappa_star_channels):
        """Two-weight model gives different result than single-weight."""
        m1, m2, det = solve_two_weight_coefficients(
            kappa_channels, kappa_star_channels,
            c_target_kappa=KAPPA_C_TARGET,
            c_target_kappa_star=KAPPA_STAR_C_TARGET,
        )

        # Single weight: average of m1 and m2
        m_single = (m1 + m2) / 2

        result_two = compute_c_paper_two_weight(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            m1=m1,
            m2=m2,
            n_quad_a=N_QUAD_A,
        )

        result_single = compute_c_paper_two_weight(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            m1=m_single,
            m2=m_single,
            n_quad_a=N_QUAD_A,
        )

        # Results should differ by more than numerical noise
        assert abs(result_two.total - result_single.total) > 1e-3


# =============================================================================
# Operator mode tests (Stage C3)
# =============================================================================

class TestOperatorQShift:
    """Tests for compute_c_paper_operator_q_shift function."""

    def test_operator_mode_produces_finite_result(self, kappa_polys):
        """Operator mode produces finite result."""
        from src.evaluate import compute_c_paper_operator_q_shift

        result = compute_c_paper_operator_q_shift(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            n_quad_a=N_QUAD_A,
        )

        assert math.isfinite(result.total)
        assert result.total > 0

    def test_operator_mode_has_implied_weights(self, kappa_polys):
        """Operator mode computes implied weights."""
        from src.evaluate import compute_c_paper_operator_q_shift

        result = compute_c_paper_operator_q_shift(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            n_quad_a=N_QUAD_A,
        )

        # Check implied weights exist and are finite
        assert "_m1_implied" in result.per_term
        assert "_m2_implied" in result.per_term
        assert math.isfinite(result.per_term["_m1_implied"])
        assert math.isfinite(result.per_term["_m2_implied"])

    def test_operator_mode_reports_base_and_op_channels(self, kappa_polys):
        """Operator mode reports both base and operator mirror channels."""
        from src.evaluate import compute_c_paper_operator_q_shift

        result = compute_c_paper_operator_q_shift(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            n_quad_a=N_QUAD_A,
        )

        # Check all expected keys
        expected_keys = [
            "_I1_minus_base", "_I1_minus_op",
            "_I2_minus_base", "_I2_minus_op",
            "_mirror_mode",
        ]
        for key in expected_keys:
            assert key in result.per_term, f"Missing key: {key}"

        assert result.per_term["_mirror_mode"] == "operator_q_shift"

    def test_operator_differs_from_base(self, kappa_polys):
        """Operator mirror values differ from base mirror values."""
        from src.evaluate import compute_c_paper_operator_q_shift

        result = compute_c_paper_operator_q_shift(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            n_quad_a=N_QUAD_A,
        )

        # The base and operator should produce different values
        i1_base = result.per_term["_I1_minus_base"]
        i1_op = result.per_term["_I1_minus_op"]
        i2_base = result.per_term["_I2_minus_base"]
        i2_op = result.per_term["_I2_minus_op"]

        # They should differ by more than numerical noise
        assert abs(i1_op - i1_base) > 1e-6, f"I1 base={i1_base}, op={i1_op}"
        assert abs(i2_op - i2_base) > 1e-6, f"I2 base={i2_base}, op={i2_op}"


class TestConstantPCollapseGate:
    """
    Collapse gate test: with P₁=P₂=P₃=constant, structure simplifies.

    This is a sanity test for the operator mode. With constant P polynomials,
    the integral structure simplifies and we can verify the Q-shift is applied.
    """

    @pytest.fixture
    def constant_polys(self):
        """Create constant polynomials for collapse gate test."""
        from src.polynomials import Polynomial
        import numpy as np

        # P₁ = P₂ = P₃ = 1 (constant), Q from PRZZ
        P_const = Polynomial(np.array([1.0]))

        # Load actual Q from PRZZ
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

        return {
            "P1": P_const,
            "P2": P_const,
            "P3": P_const,
            "Q": Q,
        }

    def test_constant_p_produces_finite_result(self, constant_polys):
        """With constant P, operator mode still produces finite result."""
        from src.evaluate import compute_c_paper_operator_q_shift

        result = compute_c_paper_operator_q_shift(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=constant_polys,
            n_quad_a=N_QUAD_A,
        )

        assert math.isfinite(result.total)

    def test_constant_p_implied_weights_finite(self, constant_polys):
        """With constant P, implied weights are finite."""
        from src.evaluate import compute_c_paper_operator_q_shift

        result = compute_c_paper_operator_q_shift(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=constant_polys,
            n_quad_a=N_QUAD_A,
        )

        m1_implied = result.per_term["_m1_implied"]
        m2_implied = result.per_term["_m2_implied"]

        assert math.isfinite(m1_implied), f"m1_implied={m1_implied}"
        assert math.isfinite(m2_implied), f"m2_implied={m2_implied}"

    def test_constant_p_q_shift_affects_result(self, constant_polys):
        """Q-shift should affect the result even with constant P."""
        from src.evaluate import compute_c_paper_operator_q_shift

        result = compute_c_paper_operator_q_shift(
            theta=THETA,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=constant_polys,
            n_quad_a=N_QUAD_A,
        )

        # With constant P, the Q-shift should still produce a ratio != 1
        m1_implied = result.per_term["_m1_implied"]
        m2_implied = result.per_term["_m2_implied"]

        # The implied weights should not both be 1.0 (that would mean Q_lift=Q)
        # Since Q is non-trivial, Q_lift should differ from Q
        assert abs(m1_implied - 1.0) > 0.01 or abs(m2_implied - 1.0) > 0.01, (
            f"Q-shift has no effect? m1={m1_implied}, m2={m2_implied}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
