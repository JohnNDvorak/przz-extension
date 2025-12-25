"""
tests/test_canonical_evaluator.py
Phase 9.0B: Tests for the canonical evaluator entrypoint.

These tests verify that the canonical evaluator:
1. Matches the existing compute_c_paper_ordered results
2. Returns properly structured results
3. Correctly integrates with m1_policy
4. Provides accurate channel breakdowns
"""

import pytest
import math
from src.canonical_evaluator import (
    compute_c_canonical,
    compute_c_for_benchmark,
    CanonicalResult,
    KAPPA_BENCHMARK,
    KAPPA_STAR_BENCHMARK,
)
from src.m1_policy import M1Policy, M1Mode
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


class TestCanonicalEvaluatorBasic:
    """Basic functionality tests for compute_c_canonical."""

    @pytest.fixture
    def polys_kappa(self):
        """Load κ benchmark polynomials."""
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        """Load κ* benchmark polynomials."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_returns_canonical_result(self, polys_kappa):
        """compute_c_canonical should return a CanonicalResult."""
        result = compute_c_canonical(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa
        )
        assert isinstance(result, CanonicalResult)

    def test_c_and_kappa_computed(self, polys_kappa):
        """Result should have valid c and κ values."""
        result = compute_c_canonical(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa
        )
        assert result.c > 0
        assert 0 < result.kappa < 1
        # Verify κ = 1 - log(c)/R
        expected_kappa = 1.0 - math.log(result.c) / result.R
        assert abs(result.kappa - expected_kappa) < 1e-10

    def test_m1_mode_recorded(self, polys_kappa):
        """Result should record which m1 mode was used."""
        result = compute_c_canonical(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa
        )
        assert result.m1_mode == "K3_EMPIRICAL"
        assert result.m1_used > 0

    def test_parameters_recorded(self, polys_kappa):
        """Result should record all input parameters."""
        result = compute_c_canonical(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa, K=3
        )
        assert abs(result.theta - 4/7) < 1e-10
        assert abs(result.R - 1.3036) < 1e-10
        assert result.n == 40
        assert result.K == 3


class TestCanonicalMatchesExisting:
    """
    Gate tests: canonical evaluator must match existing compute_c_paper_ordered.

    This ensures the canonical evaluator is a proper wrapper, not a divergent
    implementation.
    """

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_matches_paper_ordered_kappa(self, polys_kappa):
        """Canonical result should match compute_c_paper_ordered for κ."""
        from src.evaluate import compute_c_paper_ordered

        # Compute via canonical
        canonical = compute_c_canonical(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa
        )

        # Compute via existing evaluator
        existing = compute_c_paper_ordered(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa,
            s12_pair_mode="triangle"
        )

        # Values should match within floating point tolerance
        assert abs(canonical.c - existing.total) < 1e-10, \
            f"c mismatch: canonical={canonical.c}, existing={existing.total}"

    def test_matches_paper_ordered_kappa_star(self, polys_kappa_star):
        """Canonical result should match compute_c_paper_ordered for κ*."""
        from src.evaluate import compute_c_paper_ordered

        canonical = compute_c_canonical(
            theta=4/7, R=1.1167, n=40, polynomials=polys_kappa_star
        )

        existing = compute_c_paper_ordered(
            theta=4/7, R=1.1167, n=40, polynomials=polys_kappa_star,
            s12_pair_mode="triangle"
        )

        assert abs(canonical.c - existing.total) < 1e-10, \
            f"c mismatch: canonical={canonical.c}, existing={existing.total}"


class TestBenchmarkConvenience:
    """Tests for the compute_c_for_benchmark convenience function."""

    def test_kappa_benchmark(self):
        """Test κ benchmark convenience function."""
        result = compute_c_for_benchmark("kappa", n=40)
        assert result.R == 1.3036
        assert abs(result.theta - 4/7) < 1e-10
        assert result.c > 0

    def test_kappa_star_benchmark(self):
        """Test κ* benchmark convenience function."""
        result = compute_c_for_benchmark("kappa_star", n=40)
        assert result.R == 1.1167
        assert abs(result.theta - 4/7) < 1e-10
        assert result.c > 0

    def test_unknown_benchmark_raises(self):
        """Unknown benchmark should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            compute_c_for_benchmark("unknown")


class TestM1PolicyIntegration:
    """Tests for m1_policy integration."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_default_mode_is_k3_empirical(self, polys_kappa):
        """Default m1 mode should be K3_EMPIRICAL."""
        result = compute_c_canonical(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa
        )
        assert result.m1_mode == "K3_EMPIRICAL"

    def test_explicit_m1_policy(self, polys_kappa):
        """Can pass explicit m1_policy."""
        policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
        result = compute_c_canonical(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa,
            m1_policy=policy
        )
        assert result.m1_mode == "K3_EMPIRICAL"

    def test_m1_value_is_correct(self, polys_kappa):
        """m1_used should match the formula for K3_EMPIRICAL."""
        import math
        result = compute_c_canonical(
            theta=4/7, R=1.3036, n=40, polynomials=polys_kappa
        )
        expected_m1 = math.exp(1.3036) + 5
        assert abs(result.m1_used - expected_m1) < 1e-10


class TestGapCalculations:
    """Tests for gap calculation methods on CanonicalResult."""

    def test_gap_vs_target(self):
        """Test gap_vs_target calculation."""
        result = CanonicalResult(
            c=2.1, kappa=0.42, S12_plus=1.0, S12_minus=0.5, S34=0.6,
            m1_used=8.0, m1_mode="K3_EMPIRICAL",
            R=1.3, theta=4/7, n=60, K=3
        )
        # Gap = (2.1 - 2.0) / 2.0 * 100 = 5%
        gap = result.gap_vs_target(2.0)
        assert abs(gap - 5.0) < 1e-10

    def test_kappa_gap_vs_target(self):
        """Test kappa_gap_vs_target calculation."""
        result = CanonicalResult(
            c=2.1, kappa=0.42, S12_plus=1.0, S12_minus=0.5, S34=0.6,
            m1_used=8.0, m1_mode="K3_EMPIRICAL",
            R=1.3, theta=4/7, n=60, K=3
        )
        # Gap = (0.42 - 0.40) / 0.40 * 100 = 5%
        gap = result.kappa_gap_vs_target(0.40)
        assert abs(gap - 5.0) < 1e-10


class TestBenchmarkTargetConstants:
    """Tests for the benchmark target constants."""

    def test_kappa_benchmark_constants(self):
        """KAPPA_BENCHMARK should have correct values."""
        assert abs(KAPPA_BENCHMARK["R"] - 1.3036) < 1e-10
        assert abs(KAPPA_BENCHMARK["theta"] - 4/7) < 1e-10
        assert abs(KAPPA_BENCHMARK["c_target"] - 2.13745440613217263636) < 1e-10
        assert abs(KAPPA_BENCHMARK["kappa_target"] - 0.417293962) < 1e-9

    def test_kappa_star_benchmark_constants(self):
        """KAPPA_STAR_BENCHMARK should have correct values."""
        assert abs(KAPPA_STAR_BENCHMARK["R"] - 1.1167) < 1e-10
        assert abs(KAPPA_STAR_BENCHMARK["theta"] - 4/7) < 1e-10
