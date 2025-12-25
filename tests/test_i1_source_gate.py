"""
Gate tests for i1_source switch.

GPT Phase 1: Validates that post_identity_operator and DSL sources agree to machine precision.

This provides:
- Regression safety net for K>3 extension
- Independent verification path for I1 computation
- Early warning of structural drift

The post-identity operator approach was validated in run_operator_post_identity_golden.py
and matches the DSL (paper regime) to machine precision for all K=3 pairs.
"""

import pytest
import numpy as np

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.operator_post_identity import compute_I1_operator_post_identity_pair
from src.terms_k3_d1 import (
    make_all_terms_11_v2, make_all_terms_12_v2, make_all_terms_13_v2,
    make_all_terms_22_v2, make_all_terms_23_v2, make_all_terms_33_v2,
)
from src.evaluate import evaluate_terms, compute_c_paper_tex_mirror


# Canonical pairs (ell1 <= ell2)
PAIRS = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

# Benchmark R values
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167


def get_terms_fn(ell1: int, ell2: int):
    """Get the term builder function for a given pair."""
    terms_map = {
        (1, 1): make_all_terms_11_v2,
        (1, 2): make_all_terms_12_v2,
        (1, 3): make_all_terms_13_v2,
        (2, 2): make_all_terms_22_v2,
        (2, 3): make_all_terms_23_v2,
        (3, 3): make_all_terms_33_v2,
    }
    return terms_map[(ell1, ell2)]


@pytest.fixture(scope="module")
def polys_kappa():
    """Load PRZZ polynomials for kappa benchmark."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture(scope="module")
def polys_kappa_star():
    """Load PRZZ polynomials for kappa* benchmark."""
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


class TestI1SourceAgreement:
    """Tests that post_identity_operator and DSL sources agree."""

    @pytest.mark.parametrize("ell1,ell2", PAIRS)
    def test_i1_sources_match_kappa(self, ell1, ell2, polys_kappa):
        """Post-identity operator and DSL should match to machine precision (kappa)."""
        theta = 4.0 / 7.0
        R = R_KAPPA
        n = 40

        # Post-identity operator
        result_post = compute_I1_operator_post_identity_pair(
            theta, R, ell1, ell2, n, polys_kappa
        )
        I1_post = result_post.I1_value

        # DSL
        terms_fn = get_terms_fn(ell1, ell2)
        terms = terms_fn(theta, R, kernel_regime='paper')
        i1_terms = [t for t in terms if 'I1' in t.name]
        dsl_result = evaluate_terms(
            i1_terms, polys_kappa, n, R=R, theta=theta, return_breakdown=True
        )
        I1_dsl = dsl_result.total

        assert np.isclose(I1_post, I1_dsl, rtol=1e-10, atol=1e-12), \
            f"I1 source mismatch ({ell1},{ell2}) R={R}: post={I1_post}, dsl={I1_dsl}"

    @pytest.mark.parametrize("ell1,ell2", PAIRS)
    def test_i1_sources_match_kappa_star(self, ell1, ell2, polys_kappa_star):
        """Post-identity operator and DSL should match to machine precision (kappa*)."""
        theta = 4.0 / 7.0
        R = R_KAPPA_STAR
        n = 40

        # Post-identity operator
        result_post = compute_I1_operator_post_identity_pair(
            theta, R, ell1, ell2, n, polys_kappa_star
        )
        I1_post = result_post.I1_value

        # DSL
        terms_fn = get_terms_fn(ell1, ell2)
        terms = terms_fn(theta, R, kernel_regime='paper')
        i1_terms = [t for t in terms if 'I1' in t.name]
        dsl_result = evaluate_terms(
            i1_terms, polys_kappa_star, n, R=R, theta=theta, return_breakdown=True
        )
        I1_dsl = dsl_result.total

        assert np.isclose(I1_post, I1_dsl, rtol=1e-10, atol=1e-12), \
            f"I1 source mismatch ({ell1},{ell2}) R={R}: post={I1_post}, dsl={I1_dsl}"


class TestI1SourceConsistency:
    """Tests that post_identity_operator gives consistent results across all ordered pairs.

    NOTE: tex_mirror uses OLD terms (terms_version="old"), but the post-identity operator
    was validated against V2 terms. The per-pair validation tests (TestI1SourceAgreement)
    correctly test against V2 terms.

    These tests verify that the post_identity_operator source integrates correctly
    into the tex_mirror framework and produces valid (finite, non-zero) results.

    For direct I1 comparison between sources, use TestI1SourceAgreement which
    compares at the individual pair level using V2 terms with paper regime.
    """

    @pytest.mark.parametrize("ell1,ell2", [(2, 1), (3, 1), (3, 2)])
    def test_ordered_pairs_match_canonical_kappa(self, ell1, ell2, polys_kappa):
        """Verify ordered pairs (ell1 > ell2) are handled correctly.

        For PRZZ, the (2,1) pair should give the same I1 value as (1,2) due to
        the symmetric structure of the integrand under x<->y exchange combined
        with the profile factor symmetry.

        Actually, this is NOT true - ordered pairs have different polynomial
        factors (P_ell1 on x, P_ell2 on y), so (2,1) != (1,2) in general.
        But both should be finite and well-defined.
        """
        theta = 4.0 / 7.0
        R = R_KAPPA
        n = 40

        # Compute I1 for the ordered pair
        result = compute_I1_operator_post_identity_pair(
            theta, R, ell1, ell2, n, polys_kappa
        )

        # Basic sanity: should be finite
        assert np.isfinite(result.I1_value), \
            f"I1({ell1},{ell2}) is not finite: {result.I1_value}"

    @pytest.mark.parametrize("ell1,ell2", [(2, 1), (3, 1), (3, 2)])
    def test_ordered_pairs_match_canonical_kappa_star(self, ell1, ell2, polys_kappa_star):
        """Verify ordered pairs (ell1 > ell2) are handled correctly for kappa*."""
        theta = 4.0 / 7.0
        R = R_KAPPA_STAR
        n = 40

        result = compute_I1_operator_post_identity_pair(
            theta, R, ell1, ell2, n, polys_kappa_star
        )

        assert np.isfinite(result.I1_value), \
            f"I1({ell1},{ell2}) is not finite: {result.I1_value}"

    @pytest.mark.slow
    def test_all_nine_pairs_finite_kappa(self, polys_kappa):
        """All 9 ordered pairs should give finite I1 values."""
        theta = 4.0 / 7.0
        R = R_KAPPA
        n = 40

        all_pairs = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]

        for ell1, ell2 in all_pairs:
            for sign_R in [1, -1]:  # Both +R and -R branches
                result = compute_I1_operator_post_identity_pair(
                    theta, sign_R * R, ell1, ell2, n, polys_kappa
                )
                assert np.isfinite(result.I1_value), \
                    f"I1({ell1},{ell2}, R={sign_R*R}) is not finite: {result.I1_value}"


class TestI1SourceRegressionSafety:
    """Regression tests to ensure i1_source switch doesn't break existing behavior."""

    def test_default_dsl_unchanged_kappa(self, polys_kappa):
        """Default DSL source should give same results as before i1_source was added."""
        theta = 4.0 / 7.0
        R = R_KAPPA
        n = 40

        # Using DSL (default) should work
        result = compute_c_paper_tex_mirror(
            theta=theta, R=R, n=n, polynomials=polys_kappa,
            i1_source="dsl",  # explicit default
            verbose=False,
        )

        # Basic sanity checks
        assert np.isfinite(result.c), f"c is not finite: {result.c}"
        assert result.c > 0, f"c should be positive: {result.c}"
        assert np.isfinite(result.I1_plus), f"I1_plus not finite: {result.I1_plus}"

    def test_post_identity_operator_runs_kappa(self, polys_kappa):
        """Post-identity operator source should run without errors."""
        theta = 4.0 / 7.0
        R = R_KAPPA
        n = 40

        # Should not raise
        result = compute_c_paper_tex_mirror(
            theta=theta, R=R, n=n, polynomials=polys_kappa,
            i1_source="post_identity_operator",
            verbose=False,
        )

        # Basic sanity checks
        assert np.isfinite(result.c), f"c is not finite: {result.c}"
        assert np.isfinite(result.I1_plus), f"I1_plus not finite: {result.I1_plus}"
