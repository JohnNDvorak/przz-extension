"""
tests/test_operator_residual_factorization.py

Guardrail tests for operator residual factorization utilities.

These tests are NOT benchmark-matching assertions. They only verify internal
consistency of the "shape vs amplitude" bookkeeping:

- raw operator components use NO normalization
- σ=0 behaves as an identity (m_shape == 1)
- residual amplitude report reduces to solved weights when σ=0
"""

from __future__ import annotations

import pytest

from src.evaluate import (
    compute_operator_components_raw,
    compute_operator_factorization,
    report_residual_amplitude,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167
KAPPA_C_TARGET = 2.137
KAPPA_STAR_C_TARGET = 1.938


@pytest.fixture
def polys_kappa():
    """Load κ benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def polys_kappa_star():
    """Load κ* benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


class TestOperatorComponentsRaw:
    """Tests for raw operator component computation (no normalization)."""

    def test_raw_components_are_unnormalized_and_consistent(self, polys_kappa):
        raw = compute_operator_components_raw(
            theta=THETA,
            R=KAPPA_R,
            polynomials=polys_kappa,
            sigma=0.0,
            lift_scope="both",
            n=20,
            n_quad_a=10,
        )

        assert raw["normalization"] == "none"
        assert raw["m1_implied"] == pytest.approx(1.0, abs=1e-10)
        assert raw["m2_implied"] == pytest.approx(1.0, abs=1e-10)

        # Internal consistency: implied weights equal op/base ratios.
        assert raw["m1_implied"] == pytest.approx(
            raw["I1_minus_op"] / raw["I1_minus_base"], rel=1e-12, abs=1e-12
        )
        assert raw["m2_implied"] == pytest.approx(
            raw["I2_minus_op"] / raw["I2_minus_base"], rel=1e-12, abs=1e-12
        )


class TestOperatorFactorization:
    """Tests for operator factorization bookkeeping."""

    def test_sigma_zero_shapes_equal_one(self, polys_kappa):
        fact = compute_operator_factorization(
            theta=THETA,
            R=KAPPA_R,
            polynomials=polys_kappa,
            sigma=0.0,
            lift_scope="both",
            n=20,
            n_quad_a=10,
        )

        assert fact.m1_shape == pytest.approx(1.0, abs=1e-10)
        assert fact.m2_shape == pytest.approx(1.0, abs=1e-10)


class TestResidualAmplitudeReport:
    """Tests for residual amplitude reporting."""

    def test_sigma_zero_report_reduces_to_solved_weights(self, polys_kappa, polys_kappa_star):
        report_k, report_ks = report_residual_amplitude(
            theta=THETA,
            R_kappa=KAPPA_R,
            R_kappa_star=KAPPA_STAR_R,
            polys_kappa=polys_kappa,
            polys_kappa_star=polys_kappa_star,
            c_target_kappa=KAPPA_C_TARGET,
            c_target_kappa_star=KAPPA_STAR_C_TARGET,
            sigma=0.0,
            lift_scope="both",
            n=16,
            n_quad_a=8,
        )

        assert report_k.benchmark == "kappa"
        assert report_ks.benchmark == "kappa_star"

        for rep in (report_k, report_ks):
            assert rep.m1_shape == pytest.approx(1.0, abs=1e-10)
            assert rep.m2_shape == pytest.approx(1.0, abs=1e-10)

            # When m_shape == 1, residual amplitudes equal the solved weights.
            assert rep.A1_resid == pytest.approx(rep.m1_solved, rel=1e-12, abs=1e-12)
            assert rep.A2_resid == pytest.approx(rep.m2_solved, rel=1e-12, abs=1e-12)
            assert rep.A_ratio == pytest.approx(rep.m1_solved / rep.m2_solved, rel=1e-12, abs=1e-12)

