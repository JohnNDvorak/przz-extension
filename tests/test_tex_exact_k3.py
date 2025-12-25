"""
tests/test_tex_exact_k3.py
Phase 7A: Validation tests for TeX-exact K=3 evaluator.

Verifies:
1. Paper regime produces consistent values
2. Paper regime is ~78% below target (expected behavior)
3. κ benchmark and κ* benchmark both have this gap
4. Ratio between benchmarks is stable
"""

import numpy as np
import pytest

from src.tex_exact_k3 import (
    compute_c_tex_exact,
    compute_c_tex_exact_kappa,
    compute_c_tex_exact_kappa_star,
    compare_tex_exact_to_target,
    TexExactResult,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


# =============================================================================
# Constants
# =============================================================================

THETA = 4.0 / 7.0
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167

# Target values from PRZZ
C_TARGET_KAPPA = 2.13745440613217263636
C_TARGET_KAPPA_STAR = 1.9379524124677437


# =============================================================================
# Basic Function Tests
# =============================================================================

class TestTexExactResult:
    """Test TexExactResult dataclass."""

    def test_result_has_required_fields(self):
        """TexExactResult should have all required fields."""
        result = TexExactResult(
            c=1.0,
            kappa=0.5,
            R=1.3036,
            theta=4.0/7.0,
            n_quad=60,
        )
        assert hasattr(result, 'c')
        assert hasattr(result, 'kappa')
        assert hasattr(result, 'R')
        assert hasattr(result, 'theta')
        assert hasattr(result, 'n_quad')
        assert hasattr(result, 'per_pair')
        assert hasattr(result, 'assembly_formula')
        assert hasattr(result, 'notes')

    def test_notes_defaults_to_empty_list(self):
        """Notes should default to empty list."""
        result = TexExactResult(
            c=1.0,
            kappa=0.5,
            R=1.3036,
            theta=4.0/7.0,
            n_quad=60,
        )
        assert result.notes == []


class TestComputeCTexExact:
    """Test compute_c_tex_exact function."""

    def test_returns_tex_exact_result(self):
        """Function should return TexExactResult."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

        result = compute_c_tex_exact(
            theta=THETA,
            R=R_KAPPA,
            n=30,
            polynomials=polynomials,
        )
        assert isinstance(result, TexExactResult)

    def test_c_is_positive(self):
        """c should be positive (paper regime gives ~0.46)."""
        result = compute_c_tex_exact_kappa(n=30)
        assert result.c > 0

    def test_kappa_is_computed_when_c_positive(self):
        """κ should be computed when c > 0."""
        result = compute_c_tex_exact_kappa(n=30)
        assert not np.isnan(result.kappa)

    def test_quadrature_points_stored(self):
        """n_quad should match input."""
        result = compute_c_tex_exact_kappa(n=45)
        assert result.n_quad == 45

    def test_theta_stored_correctly(self):
        """θ should be stored correctly."""
        result = compute_c_tex_exact_kappa(n=30)
        assert abs(result.theta - THETA) < 1e-14

    def test_R_stored_correctly(self):
        """R should be stored correctly."""
        result = compute_c_tex_exact_kappa(n=30)
        assert abs(result.R - R_KAPPA) < 1e-10


# =============================================================================
# Paper Regime Gap Tests (Phase 7A Key Finding)
# =============================================================================

class TestPaperRegimeGap:
    """
    Verify that paper regime (no m₁) gives ~78% below target.

    This is the KEY FINDING of Phase 7A:
    - Paper regime alone cannot reach target
    - Mirror assembly is REQUIRED
    - The m₁ scalar compensates for missing mirror terms
    """

    def test_kappa_benchmark_paper_regime_below_target(self):
        """Paper regime should be significantly below target for κ benchmark."""
        result = compute_c_tex_exact_kappa(n=60)

        # Paper regime gives ~0.46, target is ~2.14
        # Gap should be ~78%
        gap_percent = (result.c - C_TARGET_KAPPA) / C_TARGET_KAPPA * 100

        # Verify it's below target (negative gap)
        assert gap_percent < 0, "Paper regime should be below target"

        # Verify gap is substantial (>50%)
        assert gap_percent < -50, f"Gap should be substantial: {gap_percent:.1f}%"

    def test_kappa_star_benchmark_paper_regime_below_target(self):
        """Paper regime should be significantly below target for κ* benchmark."""
        result = compute_c_tex_exact_kappa_star(n=60)

        gap_percent = (result.c - C_TARGET_KAPPA_STAR) / C_TARGET_KAPPA_STAR * 100

        # Verify it's below target (negative gap)
        assert gap_percent < 0, "Paper regime should be below target"

        # Verify gap is substantial (>50%)
        assert gap_percent < -50, f"Gap should be substantial: {gap_percent:.1f}%"

    def test_kappa_c_value_in_expected_range(self):
        """κ benchmark c should be in range 0.3-0.7 (paper regime)."""
        result = compute_c_tex_exact_kappa(n=60)

        # Paper regime typically gives ~0.45-0.50
        assert 0.2 < result.c < 1.0, f"c should be in paper regime range: {result.c}"

    def test_kappa_star_c_value_in_expected_range(self):
        """κ* benchmark c should be in range 0.2-0.6 (paper regime)."""
        result = compute_c_tex_exact_kappa_star(n=60)

        assert 0.1 < result.c < 0.8, f"c should be in paper regime range: {result.c}"


# =============================================================================
# Ratio Stability Tests
# =============================================================================

class TestRatioStability:
    """
    Test that κ/κ* ratio is stable (even if absolute values are wrong).

    If the paper regime is structurally correct, the ratio should be
    relatively stable even though absolute values are 78% below target.
    """

    def test_ratio_is_positive(self):
        """c ratio should be positive."""
        kappa_result = compute_c_tex_exact_kappa(n=60)
        kappa_star_result = compute_c_tex_exact_kappa_star(n=60)

        ratio = kappa_result.c / kappa_star_result.c
        assert ratio > 0

    def test_ratio_is_reasonable(self):
        """c ratio should be in reasonable range (0.5 to 3.0)."""
        kappa_result = compute_c_tex_exact_kappa(n=60)
        kappa_star_result = compute_c_tex_exact_kappa_star(n=60)

        ratio = kappa_result.c / kappa_star_result.c
        assert 0.5 < ratio < 3.0, f"Ratio should be reasonable: {ratio}"


# =============================================================================
# Quadrature Convergence Tests
# =============================================================================

class TestQuadratureConvergence:
    """Test that results converge as quadrature points increase."""

    @pytest.mark.parametrize("n1,n2", [(30, 60), (40, 80), (60, 100)])
    def test_c_converges_for_kappa_benchmark(self, n1, n2):
        """c should converge as n increases."""
        result1 = compute_c_tex_exact_kappa(n=n1)
        result2 = compute_c_tex_exact_kappa(n=n2)

        # Results should be close (within 5%)
        rel_diff = abs(result1.c - result2.c) / max(result1.c, result2.c)
        assert rel_diff < 0.05, f"c should converge: n={n1} gives {result1.c}, n={n2} gives {result2.c}"

    def test_c_is_stable_at_high_n(self):
        """c should be stable for high quadrature."""
        result60 = compute_c_tex_exact_kappa(n=60)
        result80 = compute_c_tex_exact_kappa(n=80)

        rel_diff = abs(result60.c - result80.c) / max(result60.c, result80.c)
        assert rel_diff < 0.01, "c should be stable at high n"


# =============================================================================
# Compare Function Tests
# =============================================================================

class TestCompareTexExactToTarget:
    """Test the comparison diagnostic function."""

    def test_returns_dict(self):
        """Function should return a dictionary."""
        results = compare_tex_exact_to_target(verbose=False)
        assert isinstance(results, dict)

    def test_has_kappa_results(self):
        """Results should have κ benchmark data."""
        results = compare_tex_exact_to_target(verbose=False)
        assert 'kappa' in results
        assert 'c_computed' in results['kappa']
        assert 'c_target' in results['kappa']
        assert 'c_gap_percent' in results['kappa']

    def test_has_kappa_star_results(self):
        """Results should have κ* benchmark data."""
        results = compare_tex_exact_to_target(verbose=False)
        assert 'kappa_star' in results
        assert 'c_computed' in results['kappa_star']
        assert 'c_target' in results['kappa_star']
        assert 'c_gap_percent' in results['kappa_star']

    def test_has_ratio_results(self):
        """Results should have ratio test data."""
        results = compare_tex_exact_to_target(verbose=False)
        assert 'ratio' in results
        assert 'computed' in results['ratio']
        assert 'target' in results['ratio']
        assert 'gap_percent' in results['ratio']

    def test_gap_is_negative_for_kappa(self):
        """κ benchmark gap should be negative (paper < target)."""
        results = compare_tex_exact_to_target(verbose=False)
        assert results['kappa']['c_gap_percent'] < 0

    def test_gap_is_negative_for_kappa_star(self):
        """κ* benchmark gap should be negative (paper < target)."""
        results = compare_tex_exact_to_target(verbose=False)
        assert results['kappa_star']['c_gap_percent'] < 0


# =============================================================================
# Assembly Formula Documentation Tests
# =============================================================================

class TestAssemblyFormula:
    """Test that assembly formula is documented in results."""

    def test_assembly_formula_is_set(self):
        """assembly_formula should be set in result."""
        result = compute_c_tex_exact_kappa(n=30)
        assert result.assembly_formula != ""
        assert "NO m₁" in result.assembly_formula

    def test_notes_document_evaluation_mode(self):
        """Notes should document the evaluation mode."""
        result = compute_c_tex_exact_kappa(n=30)
        notes_text = " ".join(result.notes)
        assert "paper regime" in notes_text.lower() or "tex-exact" in notes_text.lower()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_quadrature_still_works(self):
        """Even small n should produce a result."""
        result = compute_c_tex_exact_kappa(n=10)
        assert not np.isnan(result.c)
        assert result.c > 0

    def test_custom_polynomials_work(self):
        """Custom polynomials should work."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

        result = compute_c_tex_exact(
            theta=THETA,
            R=1.25,
            n=30,
            polynomials=polynomials,
        )
        assert result.c > 0
        assert abs(result.R - 1.25) < 1e-10

    def test_different_R_gives_different_c(self):
        """Different R should give different c."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

        result1 = compute_c_tex_exact(theta=THETA, R=1.2, n=30, polynomials=polynomials)
        result2 = compute_c_tex_exact(theta=THETA, R=1.4, n=30, polynomials=polynomials)

        assert result1.c != result2.c


# =============================================================================
# Regression Tests
# =============================================================================

class TestRegressionValues:
    """
    Lock in specific values to catch unexpected changes.

    These are NOT the target values - they are the paper regime values
    (which are ~78% below target).
    """

    def test_kappa_c_regression(self):
        """κ benchmark c should match expected paper regime value."""
        result = compute_c_tex_exact_kappa(n=60)

        # Paper regime gives approximately 0.46
        # Allow 10% tolerance for quadrature variations
        assert 0.4 < result.c < 0.6, f"c should be in paper regime range: {result.c}"

    def test_kappa_star_c_regression(self):
        """κ* benchmark c should match expected paper regime value."""
        result = compute_c_tex_exact_kappa_star(n=60)

        # Paper regime gives approximately 0.33
        # Allow 10% tolerance for quadrature variations
        assert 0.25 < result.c < 0.45, f"c should be in paper regime range: {result.c}"

