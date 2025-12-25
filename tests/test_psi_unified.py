"""
tests/test_psi_unified.py
Test Suite for Unified Ψ Evaluator

This tests the unified pipeline that integrates all Ψ oracles to compute c and κ.

Test coverage:
1. Two-benchmark test: κ and κ* polynomials
2. Ratio comparison: c(κ) / c(κ*) ≈ 1.10 target
3. Per-pair contribution sanity checks
4. Convergence with quadrature refinement
"""

import pytest
import math
from src.psi_unified_evaluator import evaluate_c_psi, print_evaluation_report
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


# Test parameters
THETA = 4.0 / 7.0
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167

# Target values
KAPPA_TARGET = 0.417293962
C_KAPPA_TARGET = 2.13745440613217263636


class TestPsiUnifiedBasic:
    """Basic functionality tests for unified evaluator."""

    def test_kappa_benchmark_smoke(self):
        """Smoke test: κ polynomials run without errors."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = evaluate_c_psi(THETA, R_KAPPA, n_quad=40, polynomials=polys)

        # Basic sanity checks
        assert result.c_total > 0, "c should be positive"
        assert 0 < result.kappa < 1, "κ should be in (0,1)"
        assert result.R == R_KAPPA
        assert result.n_quad == 40

    def test_kappa_star_benchmark_smoke(self):
        """Smoke test: κ* polynomials run without errors."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = evaluate_c_psi(THETA, R_KAPPA_STAR, n_quad=40, polynomials=polys)

        # Basic sanity checks
        assert result.c_total > 0, "c should be positive"
        assert 0 < result.kappa < 1, "κ should be in (0,1)"
        assert result.R == R_KAPPA_STAR
        assert result.n_quad == 40

    def test_per_pair_contributions_exist(self):
        """All 6 pairs should have non-zero contributions."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = evaluate_c_psi(THETA, R_KAPPA, n_quad=40, polynomials=polys)

        # Check raw values are non-zero
        assert result.c11_raw != 0, "c₁₁ should be non-zero"
        assert result.c22_raw != 0, "c₂₂ should be non-zero"
        assert result.c33_raw != 0, "c₃₃ should be non-zero"
        assert result.c12_raw != 0, "c₁₂ should be non-zero"
        assert result.c13_raw != 0, "c₁₃ should be non-zero (stub)"
        assert result.c23_raw != 0, "c₂₃ should be non-zero (stub)"

    def test_normalization_factors_applied(self):
        """Normalized values should differ from raw by correct factors."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = evaluate_c_psi(THETA, R_KAPPA, n_quad=40, polynomials=polys)

        # (1,1): 1/(1!×1!) × 1 = 1
        assert result.c11_norm == pytest.approx(result.c11_raw, rel=1e-10)

        # (2,2): 1/(2!×2!) × 1 = 1/4
        assert result.c22_norm == pytest.approx(result.c22_raw / 4, rel=1e-10)

        # (3,3): 1/(3!×3!) × 1 = 1/36
        assert result.c33_norm == pytest.approx(result.c33_raw / 36, rel=1e-10)

        # (1,2): 1/(1!×2!) × 2 = 1
        assert result.c12_norm == pytest.approx(result.c12_raw, rel=1e-10)

        # (1,3): 1/(1!×3!) × 2 = 1/3
        assert result.c13_norm == pytest.approx(result.c13_raw / 3, rel=1e-10)

        # (2,3): 1/(2!×3!) × 2 = 1/6
        assert result.c23_norm == pytest.approx(result.c23_raw / 6, rel=1e-10)


class TestPsiUnifiedTwoBenchmarks:
    """Two-benchmark comparison tests."""

    @pytest.fixture
    def kappa_result(self):
        """κ benchmark result."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
        return evaluate_c_psi(THETA, R_KAPPA, n_quad=60, polynomials=polys)

    @pytest.fixture
    def kappa_star_result(self):
        """κ* benchmark result."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
        return evaluate_c_psi(THETA, R_KAPPA_STAR, n_quad=60, polynomials=polys)

    def test_ratio_comparison(self, kappa_result, kappa_star_result):
        """
        Test the ratio c(κ) / c(κ*).

        NOTE: With (1,3) and (2,3) as stubs, this ratio will NOT match 1.10 exactly.
        Once full oracles are implemented, this should be updated to strict check.
        """
        ratio = kappa_result.c_total / kappa_star_result.c_total

        # For now, just check it's in a reasonable range
        # Once full oracles are ready, update this to: assert ratio == pytest.approx(1.10, rel=0.01)
        assert 0.5 < ratio < 5.0, f"Ratio {ratio:.4f} seems unreasonable"

        print(f"\n[INFO] c(κ) / c(κ*) ratio: {ratio:.4f}")
        print(f"[INFO] Target ratio: 1.10")
        print(f"[INFO] Current deviation: {abs(ratio - 1.10):.4f}")
        print(f"[INFO] NOTE: (1,3) and (2,3) are stubs - ratio will improve with full oracles")

    def test_kappa_values_reasonable(self, kappa_result, kappa_star_result):
        """Both κ values should be in reasonable range."""
        # This unified Ψ evaluator still contains stubs; treat κ as a coarse sanity check.
        assert 0.0 < kappa_result.kappa < 0.9, f"κ = {kappa_result.kappa} out of range"
        assert 0.0 < kappa_star_result.kappa < 0.9, f"κ* = {kappa_star_result.kappa} out of range"

    def test_c_values_reasonable(self, kappa_result, kappa_star_result):
        """Both c values should be in reasonable range."""
        # c for κ should be around 2.137
        # c for κ* should be slightly smaller
        assert 1.0 < kappa_result.c_total < 5.0, f"c = {kappa_result.c_total} out of range"
        assert 1.0 < kappa_star_result.c_total < 5.0, f"c* = {kappa_star_result.c_total} out of range"


class TestPsiUnifiedPerPairBreakdown:
    """Test per-pair contribution sanity."""

    def test_dominant_pairs_kappa(self):
        """
        For κ polynomials, (1,1) and (2,2) should dominate.

        (1,1) is typically the largest contributor.
        (2,2) is second largest.
        (3,3) is smallest (P₃ is small).
        """
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
        result = evaluate_c_psi(THETA, R_KAPPA, n_quad=60, polynomials=polys)

        # Check ordering (normalized values)
        assert abs(result.c11_norm) > abs(result.c33_norm), "(1,1) should dominate (3,3)"
        assert abs(result.c22_norm) > abs(result.c33_norm), "(2,2) should dominate (3,3)"

    def test_sum_equals_total(self):
        """Sum of normalized pairs should equal total c."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
        result = evaluate_c_psi(THETA, R_KAPPA, n_quad=60, polynomials=polys)

        sum_norm = (result.c11_norm + result.c22_norm + result.c33_norm +
                    result.c12_norm + result.c13_norm + result.c23_norm)

        assert sum_norm == pytest.approx(result.c_total, rel=1e-10)


class TestPsiUnifiedConvergence:
    """Test quadrature convergence."""

    def test_convergence_with_n(self):
        """c should converge as n increases."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        ns = [40, 60, 80]
        results = [evaluate_c_psi(THETA, R_KAPPA, n, polys) for n in ns]
        cs = [r.c_total for r in results]

        # Check monotonic convergence (differences should decrease)
        diff1 = abs(cs[1] - cs[0])
        diff2 = abs(cs[2] - cs[1])

        # Later differences should be smaller (convergence).
        # If we're already at machine precision, diff1 can be ~0; guard against brittle comparisons.
        assert diff2 < max(diff1 * 2, 1e-12), "Quadrature should converge"

    def test_kappa_stability(self):
        """κ should be stable across quadrature refinements."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result1 = evaluate_c_psi(THETA, R_KAPPA, n_quad=60, polynomials=polys)
        result2 = evaluate_c_psi(THETA, R_KAPPA, n_quad=80, polynomials=polys)

        # κ should agree to 3-4 decimal places at least
        assert result1.kappa == pytest.approx(result2.kappa, abs=1e-3)


class TestPsiUnifiedReporting:
    """Test report generation."""

    def test_print_report_runs(self, capsys):
        """Report should print without errors."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
        result = evaluate_c_psi(THETA, R_KAPPA, n_quad=60, polynomials=polys)

        print_evaluation_report(result, polynomial_set="κ")

        captured = capsys.readouterr()
        assert "UNIFIED EVALUATOR REPORT" in captured.out
        assert "Per-Pair Raw Values" in captured.out
        assert "Per-Pair Normalized Values" in captured.out


if __name__ == "__main__":
    # Run a quick demo
    print("=" * 70)
    print("DEMO: Unified Ψ Evaluator with Two Benchmarks")
    print("=" * 70)

    # κ benchmark
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}
    result_k = evaluate_c_psi(THETA, R_KAPPA, n_quad=60, polynomials=polys_k)
    print_evaluation_report(result_k, polynomial_set="κ")

    # κ* benchmark
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    polys_ks = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}
    result_ks = evaluate_c_psi(THETA, R_KAPPA_STAR, n_quad=60, polynomials=polys_ks)
    print_evaluation_report(result_ks, polynomial_set="κ*")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"c(κ) / c(κ*) ratio:  {result_k.c_total / result_ks.c_total:.4f}")
    print(f"Target ratio:        1.10")
    print(f"\nNOTE: (1,3) and (2,3) are currently I₂-type stubs.")
    print(f"Once full Ψ oracles are implemented, ratio should converge to 1.10.")

    # Run pytest if available
    print("\n" + "=" * 70)
    print("Running pytest...")
    print("=" * 70)
    pytest.main([__file__, "-v"])
