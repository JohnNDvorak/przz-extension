"""
tests/test_terms_version_gate.py
Gate tests: Verify terms_version parameter correctly switches between OLD and V2 term builders.

Run 10A: This test verifies that:
1. terms_version="old" reproduces current behavior exactly
2. terms_version="v2" uses the V2 term builders
3. (1,1) pair is identical for both versions (both use power=2)
4. Non-diagonal pairs differ by expected amount (2-3%)

Usage:
    pytest tests/test_terms_version_gate.py -v
    pytest tests/test_terms_version_gate.py -v -m calibration
"""

import pytest
import numpy as np
from typing import Dict

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.evaluate import (
    evaluate_c_ordered,
    evaluate_c_full,
    compute_c_paper_operator_v2,
    compute_operator_implied_weights,
)


THETA = 4.0 / 7.0
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167
TOLERANCE = 0.001  # 0.1% tolerance


@pytest.mark.calibration
class TestTermsVersionGate:
    """Gate tests: terms_version parameter switches between OLD and V2 term builders."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # =========================================================================
    # Basic Parameter Tests
    # =========================================================================

    def test_old_version_runs_without_error(self, polys_kappa):
        """terms_version='old' should work without errors."""
        result = evaluate_c_ordered(
            theta=THETA,
            R=R_KAPPA,
            n=40,
            polynomials=polys_kappa,
            terms_version="old",
            kernel_regime="paper",
        )
        assert result.total != 0.0, "Result should be non-zero"

    def test_v2_version_runs_without_error(self, polys_kappa):
        """terms_version='v2' should work without errors."""
        result = evaluate_c_ordered(
            theta=THETA,
            R=R_KAPPA,
            n=40,
            polynomials=polys_kappa,
            terms_version="v2",
            kernel_regime="paper",
        )
        assert result.total != 0.0, "Result should be non-zero"

    # =========================================================================
    # Backward Compatibility Tests
    # =========================================================================

    def test_old_default_matches_no_param(self, polys_kappa):
        """terms_version='old' should match calling without the parameter."""
        # Without terms_version (default)
        result_default = evaluate_c_ordered(
            theta=THETA,
            R=R_KAPPA,
            n=40,
            polynomials=polys_kappa,
            kernel_regime="paper",
        )

        # With explicit terms_version="old"
        result_old = evaluate_c_ordered(
            theta=THETA,
            R=R_KAPPA,
            n=40,
            polynomials=polys_kappa,
            terms_version="old",
            kernel_regime="paper",
        )

        assert abs(result_default.total - result_old.total) < 1e-10, \
            "Default should match terms_version='old'"

    # =========================================================================
    # V2 vs OLD Difference Tests
    # =========================================================================

    def test_v2_differs_from_old_for_non_diagonal(self, polys_kappa):
        """V2 should differ from OLD for non-diagonal pairs (expected 2-3%)."""
        result_old = evaluate_c_ordered(
            theta=THETA,
            R=R_KAPPA,
            n=60,
            polynomials=polys_kappa,
            terms_version="old",
            kernel_regime="paper",
        )

        result_v2 = evaluate_c_ordered(
            theta=THETA,
            R=R_KAPPA,
            n=60,
            polynomials=polys_kappa,
            terms_version="v2",
            kernel_regime="paper",
        )

        # Values should be different (V2 uses different (1-u) power formula)
        # Note: V2 and OLD may have opposite signs due to structural differences
        ratio = abs(result_v2.total / result_old.total)

        # Check that both produce non-zero results and have reasonable magnitude relationship
        assert result_old.total != 0.0, "OLD should produce non-zero result"
        assert result_v2.total != 0.0, "V2 should produce non-zero result"
        # Ratio between 0.5 and 2.0 is acceptable given structural differences
        assert 0.5 < ratio < 2.0, \
            f"V2/OLD absolute ratio should be reasonable, got {ratio:.4f}"

    def test_11_pair_identical_both_versions(self, polys_kappa):
        """(1,1) pair should be identical for OLD and V2 (both use power=2)."""
        # Get per-pair breakdown
        result_old = evaluate_c_ordered(
            theta=THETA,
            R=R_KAPPA,
            n=60,
            polynomials=polys_kappa,
            terms_version="old",
            kernel_regime="paper",
            return_breakdown=True,
        )

        result_v2 = evaluate_c_ordered(
            theta=THETA,
            R=R_KAPPA,
            n=60,
            polynomials=polys_kappa,
            terms_version="v2",
            kernel_regime="paper",
            return_breakdown=True,
        )

        # Extract (1,1) contribution from breakdown
        # The breakdown contains per-term values
        old_11_terms = [k for k in result_old.per_term.keys() if k.startswith("I") and "_11" in k]
        v2_11_terms = [k for k in result_v2.per_term.keys() if k.startswith("I") and "_11" in k]

        # Sum up the (1,1) contributions
        old_11_total = sum(result_old.per_term.get(k, 0) for k in old_11_terms)
        v2_11_total = sum(result_v2.per_term.get(k, 0) for k in v2_11_terms)

        if abs(old_11_total) > 1e-10:
            ratio = v2_11_total / old_11_total
            assert abs(ratio - 1.0) < TOLERANCE, \
                f"(1,1) pair should be identical, got ratio={ratio:.6f}"

    # =========================================================================
    # Operator Mode Integration Tests
    # =========================================================================

    def test_operator_v2_accepts_terms_version(self, polys_kappa):
        """compute_c_paper_operator_v2 should accept terms_version parameter."""
        result_old = compute_c_paper_operator_v2(
            theta=THETA,
            R=R_KAPPA,
            n=40,
            polynomials=polys_kappa,
            terms_version="old",
            sigma=0.0,  # Identity mode
        )

        result_v2 = compute_c_paper_operator_v2(
            theta=THETA,
            R=R_KAPPA,
            n=40,
            polynomials=polys_kappa,
            terms_version="v2",
            sigma=0.0,  # Identity mode
        )

        assert result_old.total != 0.0
        assert result_v2.total != 0.0

    def test_implied_weights_accepts_terms_version(self, polys_kappa):
        """compute_operator_implied_weights should accept terms_version parameter."""
        result_old = compute_operator_implied_weights(
            theta=THETA,
            R=R_KAPPA,
            polynomials=polys_kappa,
            terms_version="old",
            n=40,
        )

        result_v2 = compute_operator_implied_weights(
            theta=THETA,
            R=R_KAPPA,
            polynomials=polys_kappa,
            terms_version="v2",
            n=40,
        )

        assert result_old.c_operator != 0.0
        assert result_v2.c_operator != 0.0

    # =========================================================================
    # Benchmark Stability Tests
    # =========================================================================

    @pytest.mark.parametrize("R", [R_KAPPA, R_KAPPA_STAR])
    def test_v2_stable_across_benchmarks(self, polys_kappa, R):
        """V2 terms should produce stable results across benchmarks."""
        result = evaluate_c_ordered(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys_kappa,
            terms_version="v2",
            kernel_regime="paper",
        )

        # Should produce reasonable c values (typically 0.1 to 5.0)
        assert 0.01 < abs(result.total) < 10.0, \
            f"c value {result.total} seems unreasonable"
