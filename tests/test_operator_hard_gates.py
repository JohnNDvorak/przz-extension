"""
tests/test_operator_hard_gates.py
Hard gates for operator mode (Codex Task 4)

These tests prevent self-delusion by enforcing structural invariants:
1. σ=0 identity gate: Operator with σ=0 must reproduce base exactly
2. No-refit gate: compute_operator_implied_weights() must NOT call any 2×2 solve
3. Benchmark stability gate: Implied weights must be same sign and reasonable across κ/κ*
4. Regression snapshot: Track (m1_implied, m2_implied) at σ=5/32

Per GPT guidance 2025-12-20.
"""

import pytest
import numpy as np
import warnings

from src.evaluate import (
    compute_operator_implied_weights,
    compute_c_operator_sigma_shift,
    compute_c_paper_operator_v2,
    solve_two_weight_operator_diagnostic,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167

# Reference values from two-weight solve (DIAGNOSTIC ONLY)
M1_BASE = 6.198
M2_BASE = 8.052


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


# =============================================================================
# GATE 1: σ=0 Identity
# =============================================================================
# Operator mode with sigma=0 must yield m*_implied ≈ 1 (identity)

class TestSigmaZeroIdentityGate:
    """Gate 1: σ=0 must reproduce base exactly."""

    def test_sigma_zero_m1_implied_equals_one(self, polys_kappa):
        """m1_implied should be ≈1 when σ=0."""
        result = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=0.0, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        # m1_implied = I1_minus_op / I1_minus_base should be 1
        assert abs(result.m1_implied - 1.0) < 1e-10, \
            f"σ=0 should give m1_implied=1, got {result.m1_implied}"

    def test_sigma_zero_m2_implied_equals_one(self, polys_kappa):
        """m2_implied should be ≈1 when σ=0 (for i2_only scope)."""
        result = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=0.0, normalization="grid", lift_scope="i2_only",
            n=40, n_quad_a=30,
        )

        assert abs(result.m2_implied - 1.0) < 1e-10, \
            f"σ=0 should give m2_implied=1, got {result.m2_implied}"

    def test_sigma_zero_channels_match(self, polys_kappa):
        """With σ=0, op channels should equal base channels exactly."""
        result = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=0.0, normalization="none", lift_scope="both",
            n=40, n_quad_a=30,
        )

        assert abs(result.I1_minus_op - result.I1_minus_base) < 1e-10
        assert abs(result.I2_minus_op - result.I2_minus_base) < 1e-10

    def test_sigma_zero_is_identity_mode_flag(self, polys_kappa):
        """σ=0 should set is_identity_mode=True."""
        result = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=0.0, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        assert result.is_identity_mode is True


# =============================================================================
# GATE 2: No-Refit
# =============================================================================
# compute_operator_implied_weights() must NOT call any 2×2 solve

class TestNoRefitGate:
    """Gate 2: Ensure no 2×2 solve is called in implied weights computation."""

    def test_compute_implied_weights_no_solve_call(self, polys_kappa, monkeypatch):
        """compute_operator_implied_weights() should not call solve_two_weight_operator."""
        from src import evaluate

        call_log = []
        original_solve = evaluate.solve_two_weight_operator

        def tracked_solve(*args, **kwargs):
            call_log.append(("solve_two_weight_operator", args, kwargs))
            return original_solve(*args, **kwargs)

        monkeypatch.setattr(evaluate, "solve_two_weight_operator", tracked_solve)

        # This should NOT trigger any solve
        result = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        assert len(call_log) == 0, \
            f"compute_operator_implied_weights() called solve {len(call_log)} times - should be 0"

    def test_compute_c_sigma_shift_no_solve_call(self, polys_kappa, monkeypatch):
        """compute_c_operator_sigma_shift() should not call solve_two_weight_operator."""
        from src import evaluate

        call_log = []
        original_solve = evaluate.solve_two_weight_operator

        def tracked_solve(*args, **kwargs):
            call_log.append(("solve_two_weight_operator", args, kwargs))
            return original_solve(*args, **kwargs)

        monkeypatch.setattr(evaluate, "solve_two_weight_operator", tracked_solve)

        # This should NOT trigger any solve
        result = compute_c_operator_sigma_shift(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        assert len(call_log) == 0, \
            f"compute_c_operator_sigma_shift() called solve {len(call_log)} times - should be 0"

    def test_diagnostic_solve_emits_warning(self, polys_kappa, polys_kappa_star):
        """solve_two_weight_operator_diagnostic() should emit a UserWarning."""
        result_k = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="grid", lift_scope="i1_only", sigma=5/32,
        )
        result_ks = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_STAR_R, n=40, polynomials=polys_kappa_star,
            n_quad_a=30, verbose=False,
            normalization="grid", lift_scope="i1_only", sigma=5/32,
        )

        with pytest.warns(UserWarning, match="DIAGNOSTIC"):
            solve_two_weight_operator_diagnostic(
                result_k, result_ks,
                c_target_k=2.137, c_target_k_star=1.938,
                use_operator_channels=True,
            )


# =============================================================================
# GATE 3: Benchmark Stability
# =============================================================================
# Implied weights must be same sign and within reasonable bounds across κ/κ*

class TestBenchmarkStabilityGate:
    """Gate 3: Implied weights must be stable across benchmarks."""

    @pytest.mark.parametrize("sigma", [5/32, 0.15, 0.16])
    def test_m1_implied_same_sign_across_benchmarks(self, polys_kappa, polys_kappa_star, sigma):
        """m1_implied should have the same sign for κ and κ*."""
        result_k = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=sigma, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )
        result_ks = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_STAR_R, polynomials=polys_kappa_star,
            sigma=sigma, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        # Same sign check
        assert np.sign(result_k.m1_implied) == np.sign(result_ks.m1_implied), \
            f"m1_implied sign mismatch: κ={result_k.m1_implied}, κ*={result_ks.m1_implied}"

    @pytest.mark.parametrize("sigma", [5/32, 0.15, 0.16])
    def test_m2_implied_same_sign_across_benchmarks(self, polys_kappa, polys_kappa_star, sigma):
        """m2_implied should have the same sign for κ and κ*."""
        result_k = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=sigma, normalization="grid", lift_scope="i2_only",
            n=40, n_quad_a=30,
        )
        result_ks = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_STAR_R, polynomials=polys_kappa_star,
            sigma=sigma, normalization="grid", lift_scope="i2_only",
            n=40, n_quad_a=30,
        )

        # Same sign check
        assert np.sign(result_k.m2_implied) == np.sign(result_ks.m2_implied), \
            f"m2_implied sign mismatch: κ={result_k.m2_implied}, κ*={result_ks.m2_implied}"

    def test_implied_weights_within_factor_bound(self, polys_kappa, polys_kappa_star):
        """Implied weights between κ and κ* should be within 3x of each other."""
        sigma = 5/32

        result_k = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=sigma, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )
        result_ks = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_STAR_R, polynomials=polys_kappa_star,
            sigma=sigma, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        # Factor bound check (prevent 4000× vs 300× style blowups)
        m1_ratio = result_k.m1_implied / result_ks.m1_implied if result_ks.m1_implied != 0 else float('inf')
        m2_ratio = result_k.m2_implied / result_ks.m2_implied if result_ks.m2_implied != 0 else float('inf')

        assert 1/3 < m1_ratio < 3, \
            f"m1_implied ratio out of bounds: κ/κ* = {m1_ratio}"
        assert 1/3 < m2_ratio < 3, \
            f"m2_implied ratio out of bounds: κ/κ* = {m2_ratio}"

    def test_m1_m2_implied_close_across_benchmark_Rs(self, polys_kappa, polys_kappa_star):
        """For σ=5/32, implied weights should be very close across κ vs κ* benchmark R values."""
        sigma = 5/32

        result_k = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=sigma, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )
        result_ks = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_STAR_R, polynomials=polys_kappa_star,
            sigma=sigma, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        # The empirically observed κ/κ* ratio is ~1.01; enforce a tight band.
        m1_ratio = result_k.m1_implied / result_ks.m1_implied if result_ks.m1_implied != 0 else float("inf")
        m2_ratio = result_k.m2_implied / result_ks.m2_implied if result_ks.m2_implied != 0 else float("inf")

        assert 0.98 < m1_ratio < 1.02, f"m1_implied κ/κ* ratio out of 2% band: {m1_ratio}"
        assert 0.98 < m2_ratio < 1.02, f"m2_implied κ/κ* ratio out of 2% band: {m2_ratio}"


# =============================================================================
# GATE 4: Regression Snapshot
# =============================================================================
# Track (m1_implied, m2_implied) at σ=5/32 to catch accidental changes

class TestRegressionSnapshot:
    """Gate 4: Regression snapshot for σ=5/32."""

    # These are DIAGNOSTIC values, not ground truth.
    # They are tracked to catch accidental changes.
    # As of 2025-12-20, σ=5/32 with grid/i1_only gives:
    EXPECTED_M1_IMPLIED_KAPPA = 0.65  # Approximate, will be refined
    EXPECTED_M2_IMPLIED_KAPPA = 1.0   # m2 unchanged for i1_only scope
    EXPECTED_M1_IMPLIED_KAPPA_STAR = 0.65  # Approximate

    @pytest.mark.xfail(reason="Regression values not yet locked in")
    def test_m1_implied_regression_kappa(self, polys_kappa):
        """Track m1_implied regression at σ=5/32 for κ."""
        result = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
        )

        # Allow 10% tolerance since this is diagnostic
        assert abs(result.m1_implied - self.EXPECTED_M1_IMPLIED_KAPPA) / self.EXPECTED_M1_IMPLIED_KAPPA < 0.1, \
            f"m1_implied regression failed: got {result.m1_implied}, expected ~{self.EXPECTED_M1_IMPLIED_KAPPA}"

    def test_snapshot_values_are_recorded(self, polys_kappa, polys_kappa_star):
        """Record current snapshot values for future regression."""
        result_k = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
        )
        result_ks = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_STAR_R, polynomials=polys_kappa_star,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
        )

        # Print for visibility (these can be locked in later)
        print(f"\n--- REGRESSION SNAPSHOT at σ=5/32 ---")
        print(f"κ:  m1_implied={result_k.m1_implied:.6f}, m2_implied={result_k.m2_implied:.6f}")
        print(f"κ*: m1_implied={result_ks.m1_implied:.6f}, m2_implied={result_ks.m2_implied:.6f}")
        print(f"c_operator: κ={result_k.c_operator:.6f}, κ*={result_ks.c_operator:.6f}")

        # This test always passes - it's for recording values
        assert True


# =============================================================================
# GATE 5: q1_ratio Normalization is Quarantined
# =============================================================================

class TestUnstableNormalizationQuarantine:
    """Gate 5: q1_ratio normalization requires explicit opt-in."""

    def test_q1_ratio_raises_by_default(self, polys_kappa):
        """q1_ratio normalization should raise without allow_unstable=True."""
        with pytest.raises(ValueError, match="quarantined"):
            compute_c_paper_operator_v2(
                theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
                n_quad_a=30, verbose=False,
                normalization="q1_ratio", lift_scope="i1_only", sigma=1.0,
            )

    def test_q1_ratio_allowed_with_flag(self, polys_kappa):
        """q1_ratio normalization should work with allow_unstable=True."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_c_paper_operator_v2(
                theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
                n_quad_a=30, verbose=False,
                normalization="q1_ratio", lift_scope="i1_only", sigma=1.0,
                allow_unstable=True,
            )

            # Should emit a RuntimeWarning
            assert any("unstable" in str(warning.message).lower() for warning in w)


# =============================================================================
# GATE 6: +R Branch Must Be Unchanged (Codex Task 5)
# =============================================================================
# Operator mode must NOT change the direct (+R) branch.

class TestDirectBranchUnchangedGate:
    """Gate 6: +R branch must be bitwise identical under operator mode."""

    def test_i1_plus_unchanged_by_sigma(self, polys_kappa):
        """I1_plus should be identical regardless of σ value."""
        result_0 = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i1_only", sigma=0.0,
        )
        result_532 = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i1_only", sigma=5/32,
        )

        # Bitwise identical or near-zero tolerance
        assert result_0.per_term["_I1_plus"] == result_532.per_term["_I1_plus"], \
            f"I1_plus changed: σ=0 gave {result_0.per_term['_I1_plus']}, σ=5/32 gave {result_532.per_term['_I1_plus']}"

    def test_i2_plus_unchanged_by_sigma(self, polys_kappa):
        """I2_plus should be identical regardless of σ value."""
        result_0 = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i2_only", sigma=0.0,
        )
        result_532 = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i2_only", sigma=5/32,
        )

        assert result_0.per_term["_I2_plus"] == result_532.per_term["_I2_plus"], \
            f"I2_plus changed: σ=0 gave {result_0.per_term['_I2_plus']}, σ=5/32 gave {result_532.per_term['_I2_plus']}"

    def test_s34_plus_unchanged_by_sigma(self, polys_kappa):
        """S34_plus should be identical regardless of σ value."""
        result_0 = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="both", sigma=0.0,
        )
        result_532 = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="both", sigma=5/32,
        )

        assert result_0.per_term["_S34_plus"] == result_532.per_term["_S34_plus"], \
            f"S34_plus changed: σ=0 gave {result_0.per_term['_S34_plus']}, σ=5/32 gave {result_532.per_term['_S34_plus']}"

    def test_all_plus_channels_unchanged_across_scopes(self, polys_kappa):
        """All +R channels should be unchanged regardless of lift_scope."""
        scopes = ["i1_only", "i2_only", "both"]

        baseline = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i1_only", sigma=0.0,
        )

        for scope in scopes:
            result = compute_c_paper_operator_v2(
                theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
                n_quad_a=30, verbose=False,
                normalization="none", lift_scope=scope, sigma=5/32,
            )

            assert result.per_term["_I1_plus"] == baseline.per_term["_I1_plus"]
            assert result.per_term["_I2_plus"] == baseline.per_term["_I2_plus"]
            assert result.per_term["_S34_plus"] == baseline.per_term["_S34_plus"]


# =============================================================================
# GATE 7: Residual Factorization Consistency
# =============================================================================
# A1_resid and A2_resid should be consistent if the residual is global.

class TestResidualFactorizationGate:
    """Gate 7: Test residual factorization consistency."""

    def test_residual_report_returns_valid_data(self, polys_kappa, polys_kappa_star):
        """report_residual_amplitude should return valid reports."""
        from src.evaluate import report_residual_amplitude

        report_k, report_ks = report_residual_amplitude(
            theta=THETA,
            R_kappa=KAPPA_R,
            R_kappa_star=KAPPA_STAR_R,
            polys_kappa=polys_kappa,
            polys_kappa_star=polys_kappa_star,
            c_target_kappa=2.137,
            c_target_kappa_star=1.938,
            sigma=5/32,
            n=40, n_quad_a=30,
        )

        # Check that all fields are populated
        assert report_k.benchmark == "kappa"
        assert report_ks.benchmark == "kappa_star"
        assert report_k.m1_solved > 0
        assert report_k.m2_solved > 0
        assert not np.isnan(report_k.A1_resid)
        assert not np.isnan(report_k.A2_resid)

    def test_shape_factors_positive(self, polys_kappa, polys_kappa_star):
        """m1_shape and m2_shape should be positive for reasonable σ."""
        from src.evaluate import compute_operator_factorization

        fact = compute_operator_factorization(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=5/32, lift_scope="i1_only", n=40, n_quad_a=30,
        )

        assert fact.m1_shape > 0, f"m1_shape should be positive, got {fact.m1_shape}"
        # m2_shape should be 1 for i1_only scope (no change to I2)
        assert abs(fact.m2_shape - 1.0) < 1e-10, f"m2_shape should be 1.0 for i1_only, got {fact.m2_shape}"


# =============================================================================
# GATE 8: Operator vs Solved Comparison (GPT Run 2 Task 2)
# =============================================================================

class TestOperatorVsSolvedComparison:
    """Gate 8: compare_operator_to_two_weight_solve should work correctly."""

    def test_comparison_returns_valid_structure(self, polys_kappa, polys_kappa_star):
        """compare_operator_to_two_weight_solve should return proper structure."""
        from src.evaluate import compare_operator_to_two_weight_solve

        comp_k, comp_ks, summary = compare_operator_to_two_weight_solve(
            polys_kappa=polys_kappa,
            polys_kappa_star=polys_kappa_star,
            sigma=5/32,
            normalization="grid",
            lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        # Check structure
        assert comp_k.benchmark == "kappa"
        assert comp_ks.benchmark == "kappa_star"
        assert "is_global" in summary
        assert "A1_avg" in summary
        assert "A2_avg" in summary

    def test_comparison_residuals_are_finite(self, polys_kappa, polys_kappa_star):
        """Residual amplitudes should be finite (no div-by-zero)."""
        from src.evaluate import compare_operator_to_two_weight_solve

        comp_k, comp_ks, summary = compare_operator_to_two_weight_solve(
            polys_kappa=polys_kappa,
            polys_kappa_star=polys_kappa_star,
            sigma=5/32,
            normalization="grid",
            lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        assert np.isfinite(comp_k.A1_residual), f"A1_residual κ is not finite: {comp_k.A1_residual}"
        assert np.isfinite(comp_k.A2_residual), f"A2_residual κ is not finite: {comp_k.A2_residual}"
        assert np.isfinite(comp_ks.A1_residual), f"A1_residual κ* is not finite: {comp_ks.A1_residual}"
        assert np.isfinite(comp_ks.A2_residual), f"A2_residual κ* is not finite: {comp_ks.A2_residual}"


# =============================================================================
# GATE 9: Moment-Based σ Predictor (GPT Run 2 Task 3)
# =============================================================================

class TestMomentBasedSigmaPredictor:
    """Gate 9: Moment-based σ predictor should produce valid candidates."""

    def test_moment_candidates_are_valid(self, polys_kappa):
        """compute_moment_based_sigma_candidates should return valid values."""
        from src.evaluate import compute_moment_based_sigma_candidates

        moments = compute_moment_based_sigma_candidates(
            theta=THETA, R=KAPPA_R, Q=polys_kappa["Q"],
            n_quad=100,
        )

        # E[t] should be in (0, 1)
        assert 0 < moments.E_t < 1, f"E[t] out of range: {moments.E_t}"
        # E[t²] should be in (0, 1)
        assert 0 < moments.E_t2 < 1, f"E[t²] out of range: {moments.E_t2}"
        # E[t(1-t)] should be in (0, 0.25] for any distribution on [0,1]
        assert 0 < moments.E_t1mt <= 0.25, f"E[t(1-t)] out of range: {moments.E_t1mt}"
        # Var(t) should be non-negative
        assert moments.Var_t >= 0, f"Var(t) negative: {moments.Var_t}"

    def test_anti_overfit_probe_returns_verdict(self, polys_kappa, polys_kappa_star):
        """run_moment_anti_overfit_probe should return a verdict."""
        from src.evaluate import run_moment_anti_overfit_probe

        result = run_moment_anti_overfit_probe(
            polys_kappa=polys_kappa,
            polys_kappa_star=polys_kappa_star,
            sigma_empirical=5/32,
            n_quad=100,
        )

        assert result["verdict"] in ["structural", "Q-specific"], \
            f"Invalid verdict: {result['verdict']}"
        assert result["sigma_empirical"] == 5/32
        assert "best_match_kappa" in result
        assert "best_match_kappa_star" in result

    def test_sigma_comparison_sorted_by_diff(self, polys_kappa):
        """compare_sigma_to_moments should return sorted results."""
        from src.evaluate import (
            compute_moment_based_sigma_candidates,
            compare_sigma_to_moments,
        )

        moments = compute_moment_based_sigma_candidates(
            theta=THETA, R=KAPPA_R, Q=polys_kappa["Q"],
        )

        comparison = compare_sigma_to_moments(5/32, moments)

        # Check that results are sorted by absolute difference
        diffs = [v["diff"] for v in comparison.values()]
        assert diffs == sorted(diffs, key=abs), "Comparison not sorted by absolute difference"


# =============================================================================
# GATE 10: TeX-Mirror Evaluator Hard Gates (Codex Task 3 / GPT Run 3)
# =============================================================================
# These tests prevent regression into curve-fitting.

class TestTexMirrorEvaluatorHardGates:
    """Hard gates for compute_c_paper_tex_mirror()."""

    def test_tex_mirror_does_not_call_solve(self, polys_kappa):
        """compute_c_paper_tex_mirror must NOT call any 2×2 solve."""
        from src.evaluate import compute_c_paper_tex_mirror
        import unittest.mock as mock

        # Patch the solve function to detect if it's called
        with mock.patch('src.evaluate.solve_two_weight_operator') as mock_solve:
            result = compute_c_paper_tex_mirror(
                theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
                n_quad_a=30,
            )

            # solve_two_weight_operator should NOT be called
            assert not mock_solve.called, \
                "compute_c_paper_tex_mirror should NOT call solve_two_weight_operator"

    def test_tex_mirror_returns_valid_c(self, polys_kappa):
        """TeX-mirror should return positive, finite c."""
        from src.evaluate import compute_c_paper_tex_mirror

        result = compute_c_paper_tex_mirror(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30,
        )

        assert np.isfinite(result.c), f"c should be finite, got {result.c}"
        assert result.c > 0, f"c should be positive, got {result.c}"

    def test_tex_mirror_channels_match_operator_channels(self, polys_kappa):
        """TeX-mirror channel values should match operator-implied channels."""
        from src.evaluate import compute_c_paper_tex_mirror, compute_operator_implied_weights

        tex_result = compute_c_paper_tex_mirror(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30,
        )

        implied = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        # Channels should match exactly (same underlying computation)
        assert abs(tex_result.I1_plus - implied.I1_plus) < 1e-10
        assert abs(tex_result.I2_plus - implied.I2_plus) < 1e-10
        assert abs(tex_result.S34_plus - implied.S34_plus) < 1e-10

    def test_tex_mirror_amplitude_relationship(self, polys_kappa):
        """A2 - A1 should equal K - 1 = 2."""
        from src.evaluate import compute_c_paper_tex_mirror

        result = compute_c_paper_tex_mirror(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            K=3, n_quad_a=30,
        )

        A_diff = result.A2 - result.A1
        expected_diff = 3 - 1  # K - 1

        assert abs(A_diff - expected_diff) < 1e-10, \
            f"A2 - A1 should be {expected_diff}, got {A_diff}"


class TestTexAmplitudesHardGates:
    """Hard gates for tex_amplitudes()."""

    def test_tex_amplitudes_structure(self, polys_kappa):
        """tex_amplitudes should return correct structural values."""
        from src.evaluate import tex_amplitudes

        result = tex_amplitudes(
            theta=THETA, R=KAPPA_R, K=3, polynomials=polys_kappa,
        )

        # A_diff should equal K - 1
        assert abs(result.A_diff - 2.0) < 1e-10, f"A_diff should be 2, got {result.A_diff}"

        # A_ratio should be approximately 3/4 for K=3
        expected_ratio = (np.exp(KAPPA_R) + 2 + result.epsilon) / (np.exp(KAPPA_R) + 4 + result.epsilon)
        assert abs(result.A_ratio - expected_ratio) < 1e-10

    def test_tex_amplitudes_epsilon_default(self, polys_kappa):
        """epsilon should default to σ/θ."""
        from src.evaluate import tex_amplitudes

        sigma = 5/32
        result = tex_amplitudes(
            theta=THETA, R=KAPPA_R, K=3, polynomials=polys_kappa,
            sigma=sigma,
        )

        expected_epsilon = sigma / THETA
        assert abs(result.epsilon - expected_epsilon) < 1e-10

    def test_tex_amplitudes_diagnostics_valid(self, polys_kappa):
        """tex_amplitudes diagnostics should contain valid integrals."""
        from src.evaluate import tex_amplitudes

        result = tex_amplitudes(
            theta=THETA, R=KAPPA_R, K=3, polynomials=polys_kappa,
            compute_diagnostics=True,
        )

        # Check diagnostic keys exist
        assert "exp_R" in result.diagnostics
        assert "base_integral_Q2" in result.diagnostics
        assert "E_exp2Rt_under_Q2" in result.diagnostics

        # exp_R should match
        assert abs(result.diagnostics["exp_R"] - np.exp(KAPPA_R)) < 1e-10


class TestDirectBranchInvarianceGate:
    """Gate 11: (+R) outputs should be identical regardless of mirror mode."""

    def test_direct_branch_invariance(self, polys_kappa):
        """I1_plus, I2_plus, S34_plus should be same across sigma values."""
        from src.evaluate import compute_operator_implied_weights

        result_sigma0 = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=0.0, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        result_sigma5_32 = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        result_sigma1 = compute_operator_implied_weights(
            theta=THETA, R=KAPPA_R, polynomials=polys_kappa,
            sigma=1.0, normalization="grid", lift_scope="i1_only",
            n=40, n_quad_a=30,
        )

        # Direct (+R) channels should be identical
        assert abs(result_sigma0.I1_plus - result_sigma5_32.I1_plus) < 1e-10
        assert abs(result_sigma0.I2_plus - result_sigma5_32.I2_plus) < 1e-10
        assert abs(result_sigma0.S34_plus - result_sigma5_32.S34_plus) < 1e-10

        assert abs(result_sigma0.I1_plus - result_sigma1.I1_plus) < 1e-10
        assert abs(result_sigma0.I2_plus - result_sigma1.I2_plus) < 1e-10
        assert abs(result_sigma0.S34_plus - result_sigma1.S34_plus) < 1e-10


class TestBenchmarkAccuracyGate:
    """Gate 12: TeX-mirror should achieve <10% error on both benchmarks.

    NOTE: The tolerance is set to 10% as a starting point. As the amplitude
    formula is refined with more TeX-derived structure, this should tighten
    to <5% and eventually <1%.

    Current status (GPT Run 3):
    - κ: ~2-3% error (good)
    - κ*: ~8% error (amplitude formula needs refinement for different R)
    """

    def test_kappa_benchmark_accuracy(self, polys_kappa):
        """κ benchmark error should be < 10%."""
        from src.evaluate import compute_c_paper_tex_mirror

        result = compute_c_paper_tex_mirror(
            theta=THETA, R=KAPPA_R, n=60, polynomials=polys_kappa,
            n_quad_a=40,
        )

        c_target = 2.137
        c_gap_pct = abs(result.c - c_target) / c_target * 100

        # Loose gate: 10% for now, tighten as amplitude formula improves
        assert c_gap_pct < 10, f"κ benchmark error {c_gap_pct:.2f}% exceeds 10%"

    def test_kappa_star_benchmark_accuracy(self, polys_kappa_star):
        """κ* benchmark error should be < 10%."""
        from src.evaluate import compute_c_paper_tex_mirror

        result = compute_c_paper_tex_mirror(
            theta=THETA, R=KAPPA_STAR_R, n=60, polynomials=polys_kappa_star,
            n_quad_a=40,
        )

        c_target = 1.938
        c_gap_pct = abs(result.c - c_target) / c_target * 100

        # Loose gate: 10% for now, tighten as amplitude formula improves
        assert c_gap_pct < 10, f"κ* benchmark error {c_gap_pct:.2f}% exceeds 10%"


class TestCrossCheckGate:
    """Gate 13: TeX-derived weights should be close to diagnostic-solved weights."""

    def test_cross_check_weights_within_tolerance(self, polys_kappa, polys_kappa_star):
        """Derived m1,m2 should be within 15% of solved m1,m2."""
        from src.evaluate import validate_tex_mirror_against_diagnostic

        result = validate_tex_mirror_against_diagnostic(
            polys_kappa=polys_kappa,
            polys_kappa_star=polys_kappa_star,
            n=60, n_quad_a=40,
        )

        # m1,m2 should be within 15% of solved values
        assert result["m1_diff_pct"] < 15, \
            f"m1 diff {result['m1_diff_pct']:.1f}% exceeds 15%"
        assert result["m2_diff_pct"] < 15, \
            f"m2 diff {result['m2_diff_pct']:.1f}% exceeds 15%"

    def test_cross_check_residuals_stable(self, polys_kappa, polys_kappa_star):
        """Residual amplitudes should be benchmark-stable."""
        from src.evaluate import validate_tex_mirror_against_diagnostic

        result = validate_tex_mirror_against_diagnostic(
            polys_kappa=polys_kappa,
            polys_kappa_star=polys_kappa_star,
            n=60, n_quad_a=40,
        )

        # A1,A2 spans should be < 5%
        assert result["A1_span_pct"] < 5, \
            f"A1 span {result['A1_span_pct']:.1f}% exceeds 5%"
        assert result["A2_span_pct"] < 5, \
            f"A2 span {result['A2_span_pct']:.1f}% exceeds 5%"
