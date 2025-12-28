"""
tests/test_phase28_regime_equivalence.py
Phase 28: Regime Equivalence Tests

Locks the key finding: unified_general ≡ raw regime term DSL.

Created: 2025-12-26 (Phase 28)
"""

import pytest
import math

from src.polynomials import load_przz_polynomials
from src.unified_i1_general import compute_I1_unified_general
from src.terms_k3_d1 import (
    make_I1_11, make_I1_22, make_I1_33,
    make_I1_12, make_I1_13, make_I1_23
)
from src.evaluate import evaluate_term


@pytest.fixture
def kappa_polynomials():
    """Load kappa benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def kappa_params():
    """Parameters for kappa benchmark."""
    return {"R": 1.3036, "theta": 4 / 7}


class TestUnifiedGeneralEqualsRawRegime:
    """
    CRITICAL FINDING: unified_general produces identical values to
    term DSL with kernel_regime="raw" (or None).

    This test LOCKS this equivalence.
    """

    @pytest.mark.parametrize("pair_key,make_fn", [
        ("11", make_I1_11),
        ("22", make_I1_22),
        ("33", make_I1_33),
        ("12", make_I1_12),
        ("13", make_I1_13),
        ("23", make_I1_23),
    ])
    def test_unified_equals_raw_regime(
        self, kappa_polynomials, kappa_params, pair_key, make_fn
    ):
        """unified_general must match raw regime term DSL for all pairs."""
        R, theta = kappa_params["R"], kappa_params["theta"]
        ell1, ell2 = int(pair_key[0]), int(pair_key[1])

        # unified_general
        result_unified = compute_I1_unified_general(
            R=R, theta=theta, ell1=ell1, ell2=ell2,
            polynomials=kappa_polynomials, n_quad_u=60, n_quad_t=60,
        )
        val_unified = result_unified.I1_value

        # Raw regime term DSL
        term_raw = make_fn(theta, R, kernel_regime="raw")
        result_raw = evaluate_term(term_raw, kappa_polynomials, 60, R=R, theta=theta)
        val_raw = result_raw.value

        # They must match within numerical precision
        if abs(val_raw) > 1e-10:
            rel_err = abs(val_unified - val_raw) / abs(val_raw)
            assert rel_err < 1e-6, (
                f"I1({ell1},{ell2}): unified={val_unified:.8e}, raw={val_raw:.8e}, "
                f"rel_err={rel_err:.2e}"
            )
        else:
            assert abs(val_unified - val_raw) < 1e-10


class TestPaperRegimeDiffers:
    """
    Verify that paper regime gives DIFFERENT values than unified_general.
    This is expected - paper regime has Case C kernel attenuation.
    """

    @pytest.mark.parametrize("pair_key,make_fn", [
        ("22", make_I1_22),
        ("13", make_I1_13),
        ("23", make_I1_23),
    ])
    def test_paper_differs_from_unified(
        self, kappa_polynomials, kappa_params, pair_key, make_fn
    ):
        """Paper regime must give different values for P2/P3 pairs."""
        R, theta = kappa_params["R"], kappa_params["theta"]
        ell1, ell2 = int(pair_key[0]), int(pair_key[1])

        # unified_general (= raw)
        result_unified = compute_I1_unified_general(
            R=R, theta=theta, ell1=ell1, ell2=ell2,
            polynomials=kappa_polynomials, n_quad_u=60, n_quad_t=60,
        )
        val_unified = result_unified.I1_value

        # Paper regime term DSL
        term_paper = make_fn(theta, R, kernel_regime="paper")
        result_paper = evaluate_term(term_paper, kappa_polynomials, 60, R=R, theta=theta)
        val_paper = result_paper.value

        # They must be DIFFERENT
        ratio = val_unified / val_paper if abs(val_paper) > 1e-15 else float('inf')
        assert abs(ratio - 1.0) > 0.1, (
            f"I1({ell1},{ell2}): unified={val_unified:.8e}, paper={val_paper:.8e}, "
            f"ratio={ratio:.3f} - expected different values!"
        )


class TestDiagonalPairConsistency:
    """
    For (1,1), both regimes should give similar results since P1 doesn't
    have Case C attenuation.
    """

    def test_11_similar_across_regimes(self, kappa_polynomials, kappa_params):
        """(1,1) should be similar in both regimes."""
        R, theta = kappa_params["R"], kappa_params["theta"]

        # unified_general
        result_unified = compute_I1_unified_general(
            R=R, theta=theta, ell1=1, ell2=1,
            polynomials=kappa_polynomials, n_quad_u=60, n_quad_t=60,
        )

        # Paper regime
        term_paper = make_I1_11(theta, R, kernel_regime="paper")
        result_paper = evaluate_term(term_paper, kappa_polynomials, 60, R=R, theta=theta)

        # Raw regime
        term_raw = make_I1_11(theta, R, kernel_regime="raw")
        result_raw = evaluate_term(term_raw, kappa_polynomials, 60, R=R, theta=theta)

        # All three should match for (1,1)
        rel_err_paper = abs(result_unified.I1_value - result_paper.value) / abs(result_paper.value)
        rel_err_raw = abs(result_unified.I1_value - result_raw.value) / abs(result_raw.value)

        assert rel_err_paper < 1e-6, f"(1,1) paper vs unified: rel_err={rel_err_paper:.2e}"
        assert rel_err_raw < 1e-6, f"(1,1) raw vs unified: rel_err={rel_err_raw:.2e}"
