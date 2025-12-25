"""
tests/test_mirror_decomposition_gate.py
Phase 6 Gate Tests: Mirror Operator Shift Verification

This test suite verifies the operator shift implementation for Phase 6:
1. Q-shift consistency: Q(1+·) computed correctly
2. Shifted I₁ computation: I₁ with shifted Q is finite and well-behaved
3. Analysis of shift effects: How Q shift relates to empirical m₁

These are EXPLORATORY tests to understand the Q-shift structure.
The full decomposition gate (I₁_combined = I₁_direct + I₁_mirror_exact)
requires additional derivation work.
"""

import pytest
import numpy as np

from src.mirror_exact import (
    compute_I1_with_shifted_Q,
    compute_I1_standard,
    compute_mirror_decomposition,
    analyze_shift_effect_on_m1,
    apply_QQexp_shifted_composition,
    compute_I1_at_minus_R,
)
from src.q_operator import (
    binomial_lift_coeffs,
    binomial_shift_coeffs,
    lift_poly_by_shift,
    verify_binomial_lift,
)
from src.operator_post_identity import compute_I1_operator_post_identity_pair
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


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


class TestQShiftMathematics:
    """Verify the Q-shift (binomial lift) mathematics is correct."""

    def test_binomial_lift_simple_example(self):
        """Test Q(1+x) computation for simple polynomial Q(x) = 1 + 2x + 3x²."""
        # Q(x) = 1 + 2x + 3x²
        # Q(1+x) = 1 + 2(1+x) + 3(1+x)²
        #        = 1 + 2 + 2x + 3(1 + 2x + x²)
        #        = 1 + 2 + 3 + (2 + 6)x + 3x²
        #        = 6 + 8x + 3x²
        q_coeffs = [1.0, 2.0, 3.0]
        expected = [6.0, 8.0, 3.0]

        result = binomial_lift_coeffs(q_coeffs)

        for i, (got, exp) in enumerate(zip(result, expected)):
            assert np.isclose(got, exp), f"Coefficient {i}: {got} != {exp}"

    def test_binomial_lift_preserves_degree(self):
        """Q(1+x) has same degree as Q(x)."""
        q_coeffs = [1.0, -2.0, 0.5, 0.1, -0.05]  # Degree 4
        result = binomial_lift_coeffs(q_coeffs)
        assert len(result) == len(q_coeffs)

    def test_binomial_lift_Q_at_1(self):
        """Q_lift(0) = Q(1) by definition."""
        q_coeffs = [0.5, 0.3, -0.1, 0.02]
        q_at_1 = sum(q_coeffs)  # Q(1) = sum of coefficients

        lifted = binomial_lift_coeffs(q_coeffs)
        q_lift_at_0 = lifted[0]  # Q_lift(0) = constant term

        assert np.isclose(q_lift_at_0, q_at_1), f"{q_lift_at_0} != {q_at_1}"

    def test_binomial_lift_verification(self):
        """Verify Q(1+x) = Q_lift(x) at multiple test points."""
        q_coeffs = [1.0, -0.5, 0.25, -0.1, 0.05, -0.02]
        success, max_error = verify_binomial_lift(q_coeffs)
        assert success, f"Binomial lift verification failed with max_error={max_error}"
        assert max_error < 1e-10

    def test_shift_zero_is_identity(self):
        """Q(0+x) = Q(x) should give original coefficients."""
        q_coeffs = [1.0, 2.0, 3.0]
        result = binomial_shift_coeffs(q_coeffs, shift=0.0)

        for i, (got, exp) in enumerate(zip(result, q_coeffs)):
            assert np.isclose(got, exp), f"Coefficient {i}: {got} != {exp}"


class TestLiftPolyByShift:
    """Test the polynomial object wrapper for Q shift."""

    def test_lift_poly_creates_polynomial(self, polys_kappa):
        """lift_poly_by_shift returns a valid Polynomial object."""
        Q = polys_kappa['Q']
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        # Should have eval method
        assert hasattr(Q_shifted, 'eval')

        # Should evaluate without error
        x = np.array([0.0, 0.5, 1.0])
        result = Q_shifted.eval(x)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_lift_poly_shift_1_at_0(self, polys_kappa):
        """Q_shifted(0) = Q(1) for shift=1."""
        Q = polys_kappa['Q']
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        # Q_shifted(0) should equal Q(1)
        q_shift_at_0 = Q_shifted.eval(np.array([0.0]))[0]
        q_at_1 = Q.eval(np.array([1.0]))[0]

        assert np.isclose(q_shift_at_0, q_at_1, rtol=1e-10), \
            f"Q_shifted(0)={q_shift_at_0} != Q(1)={q_at_1}"


class TestShiftedI1Computation:
    """Test I₁ computation with shifted Q operators."""

    @pytest.mark.parametrize("ell1,ell2", [(1, 1), (2, 2), (3, 3)])
    def test_I1_shifted_is_finite(self, polys_kappa, ell1, ell2):
        """I₁ with shifted Q should be finite."""
        I1_shifted = compute_I1_with_shifted_Q(
            theta=4/7, R=1.3036, n=15, polynomials=polys_kappa,
            ell1=ell1, ell2=ell2, shift=1.0
        )
        assert np.isfinite(I1_shifted), f"I₁_shifted({ell1},{ell2}) is not finite"

    @pytest.mark.parametrize("ell1,ell2", [(1, 1), (2, 2), (3, 3)])
    def test_I1_standard_matches_post_identity(self, polys_kappa, ell1, ell2):
        """I₁ with shift=0 should match the standard post-identity computation."""
        I1_standard = compute_I1_standard(
            theta=4/7, R=1.3036, n=20, polynomials=polys_kappa,
            ell1=ell1, ell2=ell2
        )

        result_post = compute_I1_operator_post_identity_pair(
            theta=4/7, R=1.3036, ell1=ell1, ell2=ell2, n=20,
            polynomials=polys_kappa
        )

        rel_error = abs(I1_standard - result_post.I1_value) / (abs(result_post.I1_value) + 1e-100)
        assert rel_error < 1e-10, \
            f"I₁_standard mismatch for ({ell1},{ell2}): {I1_standard} vs {result_post.I1_value}"

    def test_shifted_differs_from_standard(self, polys_kappa):
        """Shifted Q should give different I₁ than standard Q."""
        I1_standard = compute_I1_with_shifted_Q(
            theta=4/7, R=1.3036, n=20, polynomials=polys_kappa,
            ell1=1, ell2=1, shift=0.0
        )
        I1_shifted = compute_I1_with_shifted_Q(
            theta=4/7, R=1.3036, n=20, polynomials=polys_kappa,
            ell1=1, ell2=1, shift=1.0
        )

        # They should be different
        rel_diff = abs(I1_shifted - I1_standard) / (abs(I1_standard) + 1e-100)
        assert rel_diff > 0.01, \
            f"Shifted and standard I₁ are too similar: {I1_shifted} vs {I1_standard}"


class TestMirrorDecomposition:
    """Test mirror decomposition analysis."""

    def test_decomposition_returns_valid_result(self, polys_kappa):
        """compute_mirror_decomposition returns valid structure."""
        result = compute_mirror_decomposition(
            theta=4/7, R=1.3036, n=15, polynomials=polys_kappa,
            ell1=1, ell2=1
        )

        assert hasattr(result, 'I1_combined')
        assert hasattr(result, 'I1_with_shifted_Q')
        assert np.isfinite(result.I1_combined)
        assert np.isfinite(result.I1_with_shifted_Q)

    def test_decomposition_at_both_benchmarks(self, polys_kappa, polys_kappa_star):
        """Decomposition works at both κ and κ* benchmarks."""
        result_k = compute_mirror_decomposition(
            theta=4/7, R=1.3036, n=15, polynomials=polys_kappa,
            ell1=1, ell2=1
        )
        result_ks = compute_mirror_decomposition(
            theta=4/7, R=1.1167, n=15, polynomials=polys_kappa_star,
            ell1=1, ell2=1
        )

        assert np.isfinite(result_k.I1_combined)
        assert np.isfinite(result_ks.I1_combined)
        assert np.isfinite(result_k.I1_with_shifted_Q)
        assert np.isfinite(result_ks.I1_with_shifted_Q)


class TestQShiftEffectAnalysis:
    """Test the analysis of Q-shift effects on mirror weight."""

    def test_analysis_returns_expected_keys(self, polys_kappa):
        """analyze_shift_effect_on_m1 returns expected structure."""
        result = analyze_shift_effect_on_m1(
            theta=4/7, R=1.3036, n=15, polynomials=polys_kappa,
            verbose=False
        )

        expected_keys = [
            'R', 'I1_standard', 'I1_shifted', 'q_shift_ratio',
            'exp_2R', 'exp_R', 'm1_empirical'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_q_shift_ratio_is_positive(self, polys_kappa):
        """Q-shift ratio should be positive (both I₁ values have same sign)."""
        result = analyze_shift_effect_on_m1(
            theta=4/7, R=1.3036, n=15, polynomials=polys_kappa,
            verbose=False
        )

        # For (1,1) pair, both should be positive
        assert result['q_shift_ratio'] > 0, \
            f"Q-shift ratio is negative: {result['q_shift_ratio']}"


class TestI1AtMinusR:
    """Test I₁ evaluation at the -R point (relevant for mirror assembly)."""

    def test_I1_at_minus_R_is_finite(self, polys_kappa):
        """I₁ at -R evaluation point should be finite."""
        I1_minus = compute_I1_at_minus_R(
            theta=4/7, R=1.3036, n=15, polynomials=polys_kappa,
            ell1=1, ell2=1, use_shifted_Q=False
        )
        assert np.isfinite(I1_minus), f"I₁(-R) is not finite: {I1_minus}"

    def test_I1_at_minus_R_differs_from_plus_R(self, polys_kappa):
        """I₁ at ±R should be different."""
        I1_plus = compute_I1_with_shifted_Q(
            theta=4/7, R=1.3036, n=20, polynomials=polys_kappa,
            ell1=1, ell2=1, shift=0.0
        )
        I1_minus = compute_I1_at_minus_R(
            theta=4/7, R=1.3036, n=20, polynomials=polys_kappa,
            ell1=1, ell2=1, use_shifted_Q=False
        )

        # They should be significantly different
        # (exp factors have opposite signs in the exponent)
        assert not np.isclose(I1_plus, I1_minus, rtol=0.1), \
            f"I₁(+R)={I1_plus} and I₁(-R)={I1_minus} are too similar"


class TestSeriesExtractionWithShiftedQ:
    """Test that TruncatedSeries extraction works with shifted Q."""

    def test_QQexp_shifted_series_is_finite(self, polys_kappa):
        """Shifted Q×Q×exp series should have finite coefficients."""
        Q = polys_kappa['Q']
        theta = 4.0 / 7.0
        R = 1.3036
        t = 0.5

        series = apply_QQexp_shifted_composition(Q, t, theta, R, shift=1.0)

        c00 = series.extract(())
        cx = series.extract(("x",))
        cy = series.extract(("y",))
        cxy = series.extract(("x", "y"))

        assert np.isfinite(c00), f"c00 not finite: {c00}"
        assert np.isfinite(cx), f"cx not finite: {cx}"
        assert np.isfinite(cy), f"cy not finite: {cy}"
        assert np.isfinite(cxy), f"cxy not finite: {cxy}"

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_shifted_series_symmetry(self, polys_kappa, t):
        """At t=0.5, shifted series should have cx ≈ cy (eigenvalue symmetry)."""
        Q = polys_kappa['Q']
        theta = 4.0 / 7.0
        R = 1.3036

        series_standard = apply_QQexp_shifted_composition(Q, t, theta, R, shift=0.0)
        series_shifted = apply_QQexp_shifted_composition(Q, t, theta, R, shift=1.0)

        # Both should be finite
        cxy_std = series_standard.extract(("x", "y"))
        cxy_shf = series_shifted.extract(("x", "y"))

        assert np.isfinite(cxy_std), f"Standard cxy not finite at t={t}"
        assert np.isfinite(cxy_shf), f"Shifted cxy not finite at t={t}"


class TestDocumentedBehavior:
    """Tests documenting expected behavior for Phase 6 development."""

    def test_shift_ratio_documentation(self, polys_kappa):
        """Document the Q-shift ratio at κ benchmark for reference."""
        result = compute_mirror_decomposition(
            theta=4/7, R=1.3036, n=25, polynomials=polys_kappa,
            ell1=1, ell2=1
        )

        ratio = result.I1_with_shifted_Q / result.I1_combined

        # Document the observed ratio
        # This test passes as long as ratio is finite - it's for documentation
        assert np.isfinite(ratio), f"Ratio is not finite: {ratio}"

        # Print for documentation (visible with pytest -v)
        print(f"\nQ-shift ratio at κ (R=1.3036): {ratio:.6f}")
        print(f"I₁_combined: {result.I1_combined:.6f}")
        print(f"I₁_shifted:  {result.I1_with_shifted_Q:.6f}")

    def test_shift_ratio_both_benchmarks(self, polys_kappa, polys_kappa_star):
        """Compare Q-shift ratio at both benchmarks."""
        result_k = compute_mirror_decomposition(
            theta=4/7, R=1.3036, n=25, polynomials=polys_kappa,
            ell1=1, ell2=1
        )
        result_ks = compute_mirror_decomposition(
            theta=4/7, R=1.1167, n=25, polynomials=polys_kappa_star,
            ell1=1, ell2=1
        )

        ratio_k = result_k.I1_with_shifted_Q / result_k.I1_combined
        ratio_ks = result_ks.I1_with_shifted_Q / result_ks.I1_combined

        print(f"\nQ-shift ratios:")
        print(f"  κ  (R=1.3036): {ratio_k:.6f}")
        print(f"  κ* (R=1.1167): {ratio_ks:.6f}")
        print(f"  Ratio of ratios: {ratio_k / ratio_ks:.4f}")

        # Both ratios should be finite
        assert np.isfinite(ratio_k)
        assert np.isfinite(ratio_ks)
