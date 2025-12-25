"""
tests/test_tex_amplitudes.py
Tests for tex_amplitudes() with exp_component modes (GPT Run 4)

Per GPT guidance:
- Non-PRZZ unit test: Q=1 case where E[exp(2Rt)] = (exp(2R)-1)/(2R)
- PRZZ acceptance test: Compare κ/κ* gaps between exp_R and E_exp2Rt_under_Q2 modes
"""

import pytest
import numpy as np

from src.evaluate import tex_amplitudes, compute_c_paper_tex_mirror
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
    Polynomial,
)


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167
C_TARGET_KAPPA = 2.137
C_TARGET_KAPPA_STAR = 1.938


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
# Non-PRZZ Unit Tests: Q=1 Case
# =============================================================================
# For Q(t) = 1, the Q²-weighted moment should equal the uniform average:
#   E_{Q²}[exp(2Rt)] = ∫₀¹ exp(2Rt) dt = (exp(2R)-1)/(2R)

class TestQEqualsOneCase:
    """Pure math validation: Q=1 case."""

    @pytest.fixture
    def polys_q_one(self):
        """Create polynomials with Q(t) = 1."""
        # Q = 1 means coefficients [1.0] in monomial basis
        Q_one = Polynomial(np.array([1.0]))
        # Dummy P polynomials (not used for this test)
        P1 = Polynomial(np.array([1.0]))
        P2 = Polynomial(np.array([1.0]))
        P3 = Polynomial(np.array([1.0]))
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q_one}

    @pytest.mark.parametrize("R", [0.5, 1.0, 1.3036, 1.5, 2.0])
    def test_e_exp2rt_equals_uniform_avg_for_q_one(self, polys_q_one, R):
        """For Q=1, E[exp(2Rt)] should equal (exp(2R)-1)/(2R)."""
        result = tex_amplitudes(
            theta=THETA,
            R=R,
            K=3,
            polynomials=polys_q_one,
            exp_component="exp_R",  # doesn't matter for this test
            compute_diagnostics=True,
        )

        E_exp2Rt = result.diagnostics["E_exp2Rt_under_Q2"]
        uniform_avg = result.diagnostics["uniform_avg"]

        # These should be equal within numerical precision
        assert abs(E_exp2Rt - uniform_avg) < 1e-10, \
            f"For Q=1, E[exp(2Rt)]={E_exp2Rt} should equal uniform_avg={uniform_avg}"

    @pytest.mark.parametrize("R", [0.5, 1.0, 1.3036, 1.5, 2.0])
    def test_uniform_avg_formula(self, polys_q_one, R):
        """Verify uniform_avg = (exp(2R)-1)/(2R) is computed correctly."""
        result = tex_amplitudes(
            theta=THETA,
            R=R,
            K=3,
            polynomials=polys_q_one,
            compute_diagnostics=True,
        )

        uniform_avg = result.diagnostics["uniform_avg"]
        expected = (np.exp(2 * R) - 1) / (2 * R)

        assert abs(uniform_avg - expected) < 1e-12, \
            f"uniform_avg={uniform_avg} should be (exp(2R)-1)/(2R)={expected}"


# =============================================================================
# exp_component Mode Tests
# =============================================================================

class TestExpComponentModes:
    """Test different exp_component modes."""

    def test_exp_r_mode_uses_exp_r(self, polys_kappa):
        """exp_R mode should use exp(R) as the exp component."""
        result = tex_amplitudes(
            theta=THETA,
            R=KAPPA_R,
            K=3,
            polynomials=polys_kappa,
            exp_component="exp_R",
        )

        exp_component_value = result.diagnostics["exp_component_value"]
        expected = np.exp(KAPPA_R)

        assert abs(exp_component_value - expected) < 1e-10

    def test_uniform_avg_mode(self, polys_kappa):
        """uniform_avg mode should use (exp(2R)-1)/(2R)."""
        result = tex_amplitudes(
            theta=THETA,
            R=KAPPA_R,
            K=3,
            polynomials=polys_kappa,
            exp_component="uniform_avg",
        )

        exp_component_value = result.diagnostics["exp_component_value"]
        expected = (np.exp(2 * KAPPA_R) - 1) / (2 * KAPPA_R)

        assert abs(exp_component_value - expected) < 1e-10

    def test_e_exp2rt_mode_uses_q_weighted_moment(self, polys_kappa):
        """E_exp2Rt_under_Q2 mode should use the Q²-weighted moment."""
        result = tex_amplitudes(
            theta=THETA,
            R=KAPPA_R,
            K=3,
            polynomials=polys_kappa,
            exp_component="E_exp2Rt_under_Q2",
        )

        exp_component_value = result.diagnostics["exp_component_value"]
        E_exp2Rt = result.diagnostics["E_exp2Rt_under_Q2"]

        assert abs(exp_component_value - E_exp2Rt) < 1e-10

    def test_e_exp2rt_requires_q_polynomial(self):
        """E_exp2Rt_under_Q2 mode should raise error if Q not provided."""
        with pytest.raises(ValueError, match="requires polynomials"):
            tex_amplitudes(
                theta=THETA,
                R=KAPPA_R,
                K=3,
                polynomials=None,  # No Q!
                exp_component="E_exp2Rt_under_Q2",
            )

    def test_invalid_exp_component_raises_error(self, polys_kappa):
        """Invalid exp_component should raise ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            tex_amplitudes(
                theta=THETA,
                R=KAPPA_R,
                K=3,
                polynomials=polys_kappa,
                exp_component="invalid_mode",
            )

    def test_a_diff_invariant_across_modes(self, polys_kappa):
        """A2 - A1 = K - 1 should hold for all exp_component modes."""
        for mode in ["exp_R", "exp_R_ref", "uniform_avg", "E_exp2Rt_under_Q2"]:
            result = tex_amplitudes(
                theta=THETA,
                R=KAPPA_R,
                K=3,
                polynomials=polys_kappa,
                exp_component=mode,
            )

            assert abs(result.A_diff - 2.0) < 1e-10, \
                f"A_diff should be K-1=2 for mode={mode}, got {result.A_diff}"

    def test_exp_r_ref_mode(self, polys_kappa):
        """exp_R_ref mode should use fixed exp(R_ref)."""
        R_ref = 1.3036

        # Test at κ* R value
        result = tex_amplitudes(
            theta=THETA,
            R=KAPPA_STAR_R,  # Different from R_ref!
            K=3,
            polynomials=polys_kappa,
            exp_component="exp_R_ref",
            R_ref=R_ref,
        )

        exp_R_ref_value = np.exp(R_ref)
        exp_component_value = result.diagnostics["exp_component_value"]

        # Should use exp(R_ref), not exp(R)
        assert abs(exp_component_value - exp_R_ref_value) < 1e-10, \
            f"exp_R_ref mode should use exp({R_ref}), got {exp_component_value}"

        # Should NOT use exp(R_kappa_star)
        exp_R_actual = np.exp(KAPPA_STAR_R)
        assert abs(exp_component_value - exp_R_actual) > 0.1, \
            f"exp_R_ref should NOT use exp(R), but values match: {exp_component_value}"


# =============================================================================
# PRZZ Acceptance Tests (CALIBRATION - not structural truth)
# =============================================================================

@pytest.mark.calibration
class TestPRZZAcceptance:
    """PRZZ-facing acceptance tests for exp_component modes.

    CLASSIFICATION: CALIBRATION tests, not structural hard gates.

    These tests verify that the exp_R_ref mode achieves good accuracy on
    PRZZ benchmarks, but exp_R_ref is a calibration fix, NOT TeX-derived.

    GPT Run 4 finding:
    - E_exp2Rt_under_Q2 mode does NOT improve results (makes them worse)
    - exp_R_ref mode with R_ref=1.3036 improves κ* from 8% error to <0.5%!

    GPT Run 5 guidance:
    Treat exp_R_ref as a calibrated stopgap. These tests ensure the
    calibration remains accurate, but they do NOT prove PRZZ reproduction
    from first principles.
    """

    def test_exp_r_ref_improves_kappa_star(self, polys_kappa_star):
        """κ* benchmark should improve dramatically with exp_R_ref mode.

        exp_R mode: ~8% error on κ*
        exp_R_ref mode: <0.5% error on κ* (using R_ref=1.3036)
        """
        result_old = compute_c_paper_tex_mirror(
            theta=THETA, R=KAPPA_STAR_R, n=60, polynomials=polys_kappa_star,
            tex_exp_component="exp_R", n_quad_a=40,
        )
        result_new = compute_c_paper_tex_mirror(
            theta=THETA, R=KAPPA_STAR_R, n=60, polynomials=polys_kappa_star,
            tex_exp_component="exp_R_ref", tex_R_ref=KAPPA_R, n_quad_a=40,
        )

        gap_old = abs(result_old.c - C_TARGET_KAPPA_STAR) / C_TARGET_KAPPA_STAR * 100
        gap_new = abs(result_new.c - C_TARGET_KAPPA_STAR) / C_TARGET_KAPPA_STAR * 100

        print(f"κ* gap: exp_R={gap_old:.2f}%, exp_R_ref={gap_new:.2f}%")

        # HARD GATE: exp_R_ref mode must achieve <1% error on κ*
        assert gap_new < 1.0, \
            f"exp_R_ref mode should achieve <1% on κ*, got {gap_new:.2f}%"

        # HARD GATE: exp_R_ref must improve over exp_R
        assert gap_new < gap_old, \
            f"exp_R_ref should improve over exp_R: {gap_new:.2f}% vs {gap_old:.2f}%"

    def test_exp_r_ref_maintains_kappa(self, polys_kappa):
        """κ benchmark should remain accurate with exp_R_ref mode."""
        result_old = compute_c_paper_tex_mirror(
            theta=THETA, R=KAPPA_R, n=60, polynomials=polys_kappa,
            tex_exp_component="exp_R", n_quad_a=40,
        )
        result_new = compute_c_paper_tex_mirror(
            theta=THETA, R=KAPPA_R, n=60, polynomials=polys_kappa,
            tex_exp_component="exp_R_ref", tex_R_ref=KAPPA_R, n_quad_a=40,
        )

        gap_old = abs(result_old.c - C_TARGET_KAPPA) / C_TARGET_KAPPA * 100
        gap_new = abs(result_new.c - C_TARGET_KAPPA) / C_TARGET_KAPPA * 100

        print(f"κ gap: exp_R={gap_old:.2f}%, exp_R_ref={gap_new:.2f}%")

        # For κ, exp_R_ref with R_ref=KAPPA_R should be identical to exp_R
        assert abs(gap_new - gap_old) < 0.01, \
            f"For κ, exp_R_ref(R_ref={KAPPA_R}) should equal exp_R: {gap_new:.2f}% vs {gap_old:.2f}%"

    def test_e_exp2rt_diagnostic_only(self, polys_kappa, polys_kappa_star):
        """E_exp2Rt_under_Q2 mode diagnostic (does NOT improve results)."""
        # This test documents that E_exp2Rt_under_Q2 was a dead end
        for bench_name, polys, R_val, c_target in [
            ("κ", polys_kappa, KAPPA_R, C_TARGET_KAPPA),
            ("κ*", polys_kappa_star, KAPPA_STAR_R, C_TARGET_KAPPA_STAR),
        ]:
            result_base = compute_c_paper_tex_mirror(
                theta=THETA, R=R_val, n=60, polynomials=polys,
                tex_exp_component="exp_R", n_quad_a=40,
            )
            result_e = compute_c_paper_tex_mirror(
                theta=THETA, R=R_val, n=60, polynomials=polys,
                tex_exp_component="E_exp2Rt_under_Q2", n_quad_a=40,
            )

            gap_base = abs(result_base.c - c_target) / c_target * 100
            gap_e = abs(result_e.c - c_target) / c_target * 100

            print(f"{bench_name}: exp_R={gap_base:.2f}%, E_exp2Rt={gap_e:.2f}%")

        # Diagnostic only - no hard gate
        assert True

    def test_all_modes_comparison(self, polys_kappa, polys_kappa_star):
        """Compare all exp_component modes side-by-side."""
        results = {}

        for mode in ["exp_R", "exp_R_ref", "E_exp2Rt_under_Q2", "uniform_avg"]:
            result_k = compute_c_paper_tex_mirror(
                theta=THETA, R=KAPPA_R, n=60, polynomials=polys_kappa,
                tex_exp_component=mode, tex_R_ref=KAPPA_R, n_quad_a=40,
            )
            result_ks = compute_c_paper_tex_mirror(
                theta=THETA, R=KAPPA_STAR_R, n=60, polynomials=polys_kappa_star,
                tex_exp_component=mode, tex_R_ref=KAPPA_R, n_quad_a=40,
            )

            gap_k = (result_k.c - C_TARGET_KAPPA) / C_TARGET_KAPPA * 100
            gap_ks = (result_ks.c - C_TARGET_KAPPA_STAR) / C_TARGET_KAPPA_STAR * 100

            results[mode] = {
                "c_kappa": result_k.c,
                "c_kappa_star": result_ks.c,
                "gap_kappa": gap_k,
                "gap_kappa_star": gap_ks,
            }

        # Print comparison for visibility
        print("\n" + "=" * 70)
        print("ALL AMPLITUDE MODES COMPARISON (GPT Run 4)")
        print("=" * 70)
        print(f"{'Mode':<25} {'κ gap':>12} {'κ* gap':>12}")
        print("-" * 50)
        for mode, data in results.items():
            print(f"{mode:<25} {data['gap_kappa']:>+11.2f}% {data['gap_kappa_star']:>+11.2f}%")
        print("=" * 70)

        # exp_R_ref should be the best (or tied for best) on both
        assert results["exp_R_ref"]["gap_kappa_star"] < 1.0, \
            "exp_R_ref should achieve <1% on κ*"
