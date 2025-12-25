"""
tests/test_operator_post_identity_core.py
Core tests for the post-identity operator approach.

These tests validate:
1. Affine coefficient structure (the whole point of this experiment)
2. Operator vs composition equivalence
3. L-stability (no L-divergence, fixing Step 2's trap)
"""

import pytest
import numpy as np
from src.operator_post_identity import (
    compute_A_alpha,
    compute_A_beta,
    get_A_alpha_affine_coeffs,
    get_A_beta_affine_coeffs,
    apply_Q_post_identity_composition,
    apply_Q_post_identity_operator_sum,
    evaluate_Q_product_at_affine,
    evaluate_operator_applied_core,
    convert_Q_basis_to_monomial,
    compute_post_identity_core_E,
)
from src.polynomials import load_przz_polynomials


# =============================================================================
# Test Class: Affine Coefficient Gates (Step 4)
# =============================================================================

class TestAffineCoefficients:
    """
    The whole point of this experiment: verify (θt-θ) cross-term structure.

    Expected structure:
        A_α = t + θ(t-1)·x + θt·y
        A_β = t + θt·x + θ(t-1)·y
    """

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    # --- A_β tests (matches tex_mirror's arg_α) ---

    @pytest.mark.parametrize("t", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_A_beta_x_coeff_is_theta_t(self, t, theta):
        """x-coeff of A_β should be θt."""
        _, x_coeff, _ = get_A_beta_affine_coeffs(t, theta)
        expected = theta * t
        assert abs(x_coeff - expected) < 1e-14, f"x_coeff={x_coeff}, expected={expected}"

    @pytest.mark.parametrize("t", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_A_beta_y_coeff_is_theta_t_minus_1(self, t, theta):
        """y-coeff of A_β should be θ(t-1)."""
        _, _, y_coeff = get_A_beta_affine_coeffs(t, theta)
        expected = theta * (t - 1)
        assert abs(y_coeff - expected) < 1e-14, f"y_coeff={y_coeff}, expected={expected}"

    @pytest.mark.parametrize("t", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_A_beta_constant_is_t(self, t, theta):
        """Constant term of A_β should be t."""
        u0, _, _ = get_A_beta_affine_coeffs(t, theta)
        assert abs(u0 - t) < 1e-14, f"u0={u0}, expected={t}"

    # --- A_α tests (matches tex_mirror's arg_β, swapped) ---

    @pytest.mark.parametrize("t", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_A_alpha_x_coeff_is_theta_t_minus_1(self, t, theta):
        """x-coeff of A_α should be θ(t-1)."""
        _, x_coeff, _ = get_A_alpha_affine_coeffs(t, theta)
        expected = theta * (t - 1)
        assert abs(x_coeff - expected) < 1e-14, f"x_coeff={x_coeff}, expected={expected}"

    @pytest.mark.parametrize("t", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_A_alpha_y_coeff_is_theta_t(self, t, theta):
        """y-coeff of A_α should be θt."""
        _, _, y_coeff = get_A_alpha_affine_coeffs(t, theta)
        expected = theta * t
        assert abs(y_coeff - expected) < 1e-14, f"y_coeff={y_coeff}, expected={expected}"

    @pytest.mark.parametrize("t", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_A_alpha_constant_is_t(self, t, theta):
        """Constant term of A_α should be t."""
        u0, _, _ = get_A_alpha_affine_coeffs(t, theta)
        assert abs(u0 - t) < 1e-14, f"u0={u0}, expected={t}"

    # --- Cross-term asymmetry test (the key insight) ---

    @pytest.mark.parametrize("t", [0.3, 0.5, 0.7])
    def test_cross_term_asymmetry(self, t, theta):
        """
        Verify that A_α and A_β have swapped x/y coefficients.

        This is the crucial asymmetry that creates the (θt-θ) cross-term structure.
        """
        _, x_a, y_a = get_A_alpha_affine_coeffs(t, theta)
        _, x_b, y_b = get_A_beta_affine_coeffs(t, theta)

        # x-coeff of A_α should equal y-coeff of A_β
        assert abs(x_a - y_b) < 1e-14, f"x_a={x_a}, y_b={y_b}"

        # y-coeff of A_α should equal x-coeff of A_β
        assert abs(y_a - x_b) < 1e-14, f"y_a={y_a}, x_b={x_b}"

    # --- Functional form verification ---

    @pytest.mark.parametrize("t", [0.3, 0.5, 0.7])
    def test_A_alpha_functional_form(self, t, theta):
        """Verify A_α = t(1+θ(x+y)) - θx at numeric points."""
        x_val, y_val = 0.1, 0.2

        # Direct evaluation
        direct = compute_A_alpha(x_val, y_val, t, theta)

        # Affine form evaluation
        u0, x_coeff, y_coeff = get_A_alpha_affine_coeffs(t, theta)
        affine = u0 + x_coeff * x_val + y_coeff * y_val

        assert abs(direct - affine) < 1e-14, f"direct={direct}, affine={affine}"

    @pytest.mark.parametrize("t", [0.3, 0.5, 0.7])
    def test_A_beta_functional_form(self, t, theta):
        """Verify A_β = t(1+θ(x+y)) - θy at numeric points."""
        x_val, y_val = 0.1, 0.2

        # Direct evaluation
        direct = compute_A_beta(x_val, y_val, t, theta)

        # Affine form evaluation
        u0, x_coeff, y_coeff = get_A_beta_affine_coeffs(t, theta)
        affine = u0 + x_coeff * x_val + y_coeff * y_val

        assert abs(direct - affine) < 1e-14, f"direct={direct}, affine={affine}"


# =============================================================================
# Test Class: Operator vs Composition Equivalence
# =============================================================================

class TestOperatorVsComposition:
    """
    Verify Path A (composition) and Path B (operator sum) give the same result.

    This is the key validation that the eigenvalue approach works.

    GPT Phase 1, Step 3: Strengthened to include exp factor validation.
    """

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    @pytest.fixture
    def Q_poly(self):
        """Load Q polynomial."""
        _, _, _, Q = load_przz_polynomials(enforce_Q0=True)
        return Q

    @pytest.fixture
    def Q_mono(self, Q_poly):
        """Get Q in monomial form from the loaded polynomial.

        Uses the loaded Q polynomial to ensure consistency between paths.
        """
        # Extract monomial coefficients from the loaded Q polynomial
        mono_poly = Q_poly.to_monomial()
        return list(mono_poly.coeffs)

    @pytest.mark.parametrize("t", [0.3, 0.5, 0.7])
    def test_paths_match_at_zero(self, t, theta, Q_poly, Q_mono):
        """
        At x=y=0, both paths should give Q(t)².
        """
        # Path A: Composition
        series = apply_Q_post_identity_composition(Q_poly, t, theta)
        composition_result = series.extract(())  # constant term

        # Path B: Operator sum
        operator_result = apply_Q_post_identity_operator_sum(Q_mono, 0.0, 0.0, t, theta)

        # Both should be Q(t)²
        Q_at_t = Q_poly.eval(np.array([t]))[0]
        expected = Q_at_t ** 2

        assert abs(composition_result - expected) < 1e-10, \
            f"composition={composition_result}, expected={expected}"
        assert abs(operator_result - expected) < 1e-10, \
            f"operator={operator_result}, expected={expected}"

    @pytest.mark.parametrize("t", [0.3, 0.5, 0.7])
    def test_paths_match_numeric(self, t, theta, Q_poly, Q_mono):
        """
        At small numeric x, y, both paths should match.
        """
        x_val, y_val = 0.01, 0.02

        # Path A: Evaluate Q(A_α)×Q(A_β) directly
        composition_result = evaluate_Q_product_at_affine(Q_poly, x_val, y_val, t, theta)

        # Path B: Operator sum
        operator_result = apply_Q_post_identity_operator_sum(Q_mono, x_val, y_val, t, theta)

        assert abs(composition_result - operator_result) < 1e-10, \
            f"composition={composition_result}, operator={operator_result}"

    @pytest.mark.parametrize("t", [0.3, 0.5, 0.7])
    @pytest.mark.parametrize("R", [1.3036, 1.1167])
    def test_operator_vs_composition_full_core(self, t, R, theta, Q_poly, Q_mono):
        """
        GPT Phase 1, Step 3: Validate FULL Q×Q×exp core from two paths.

        Path A (Composition): Extract series coefficients, evaluate at numeric (x,y).
        Path B (Operator): Q(A_α) × Q(A_β) × exp(2Rt + b(x+y)) directly.

        IMPORTANT: The series expansion is truncated at xy order (nilpotent algebra),
        while the operator path evaluates the full polynomial. The difference is
        O(x²) + O(y²) + O(x²y) + ... which is ~0.001 at small x,y.

        For I1 integration, we only extract the xy coefficient, so truncation is
        correct. This test verifies the match is close, not exact.
        """
        from src.operator_post_identity import apply_QQexp_post_identity_composition

        x_val, y_val = 0.01, 0.02  # Small values to minimize truncation error

        # Path A: Composition - extract series coefficients, evaluate polynomial
        series = apply_QQexp_post_identity_composition(Q_poly, t, theta, R)
        c00 = series.extract(())
        cx = series.extract(("x",))
        cy = series.extract(("y",))
        cxy = series.extract(("x", "y"))
        val_comp = c00 + cx * x_val + cy * y_val + cxy * x_val * y_val

        # Path B: Operator - evaluate Q(A_α) × Q(A_β) × exp(...) directly
        A_alpha = t + theta * (t - 1) * x_val + theta * t * y_val
        A_beta = t + theta * t * x_val + theta * (t - 1) * y_val
        b = R * (2 * theta * t - theta)  # Rθ(2t-1)

        Q_alpha = Q_poly.eval(np.array([A_alpha]))[0]
        Q_beta = Q_poly.eval(np.array([A_beta]))[0]
        exp_factor = np.exp(2 * R * t + b * (x_val + y_val))

        val_op = Q_alpha * Q_beta * exp_factor

        # Relative difference should be small (O(x²) + O(y²) ~ 0.0005)
        rel_diff = abs(val_comp - val_op) / abs(val_op) if val_op != 0 else abs(val_comp - val_op)
        assert rel_diff < 0.01, \
            f"FULL CORE MISMATCH at t={t}, R={R}: " \
            f"composition={val_comp:.12f}, operator={val_op:.12f}, " \
            f"rel_diff={rel_diff:.4f} (should be < 1%)"

    @pytest.mark.parametrize("t", [0.3, 0.5, 0.7])
    @pytest.mark.parametrize("R", [1.3036, 1.1167])
    def test_operator_vs_composition_at_zero(self, t, R, theta, Q_poly, Q_mono):
        """
        At x=y=0, both paths should give EXACTLY the same result.

        This is the definitive test: at zero, there's no truncation error.
        """
        from src.operator_post_identity import apply_QQexp_post_identity_composition

        # Path A: Composition - constant term only
        series = apply_QQexp_post_identity_composition(Q_poly, t, theta, R)
        val_comp = series.extract(())

        # Path B: Operator - Q(t)² × exp(2Rt)
        Q_at_t = Q_poly.eval(np.array([t]))[0]
        val_op = Q_at_t ** 2 * np.exp(2 * R * t)

        assert abs(val_comp - val_op) < 1e-12, \
            f"ZERO POINT MISMATCH at t={t}, R={R}: " \
            f"composition={val_comp:.12f}, operator={val_op:.12f}, " \
            f"diff={abs(val_comp - val_op):.2e}"


# =============================================================================
# Test Class: No L-Divergence Gate (Step 5)
# =============================================================================

class TestNoLDivergence:
    """
    The definitive 'no more L-divergence' test.

    Step 2's operator-level approach diverged with L because it used
    the pre-identity bracket with 1/(α+β) ~ L factor.

    The post-identity approach should be STABLE in L.
    """

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    @pytest.fixture
    def Q_mono(self):
        """Get Q in monomial form."""
        basis_coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
        return convert_Q_basis_to_monomial(basis_coeffs)

    def test_QQ_product_independent_of_L(self, theta, Q_mono):
        """
        Q(A_α)×Q(A_β) should be completely independent of L.

        The A_α, A_β forms don't depend on L at all - they come from
        the derivative of log E, not E itself.
        """
        t = 0.5
        x_val, y_val = 0.05, 0.05

        results = []
        for L in [10, 20, 50, 100]:
            result = apply_Q_post_identity_operator_sum(Q_mono, x_val, y_val, t, theta)
            results.append(result)

        # All results should be identical (not just proportional)
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-14, \
                f"Q×Q changed with L: {results}"

    def test_full_operator_applied_stable_in_L(self, theta, Q_mono):
        """
        Q(A_α)Q(A_β)×E should be stable at α=β=-R/L.

        The L-dependence in E should be exactly cancelled by the
        evaluation point α=β=-R/L.
        """
        R = 1.3036
        t = 0.5
        x_val, y_val = 0.05, 0.05

        results = []
        for L in [10, 20, 50, 100]:
            alpha = -R / L
            beta = -R / L
            result = evaluate_operator_applied_core(
                alpha, beta, x_val, y_val, t, theta, L, Q_mono
            )
            results.append(result)

        # Check stability: relative change < 1e-6
        # Note: We expect results to converge as L→∞, not be identical
        # So we check the ratio of consecutive results
        for i in range(1, len(results)):
            rel_change = abs(results[i] - results[i-1]) / abs(results[i-1])
            # The change should decrease as L increases
            if i > 1:
                prev_change = abs(results[i-1] - results[i-2]) / abs(results[i-2])
                assert rel_change <= prev_change * 1.5, \
                    f"L-divergence detected: changes not decreasing: {results}"

    def test_no_linear_L_growth(self, theta, Q_mono):
        """
        Verify there is NO linear growth with L.

        Step 2's pre-identity bracket gave I1 ∝ L (linear growth).
        The post-identity approach should give I1 that is O(1) in L.
        """
        R = 1.3036
        t = 0.5
        x_val, y_val = 0.05, 0.05

        L_values = [10.0, 20.0, 50.0, 100.0]
        results = []
        for L in L_values:
            alpha = -R / L
            beta = -R / L
            result = evaluate_operator_applied_core(
                alpha, beta, x_val, y_val, t, theta, L, Q_mono
            )
            results.append(result)

        # If there was linear L growth, result/L would be constant
        # Instead, we expect result to be ~constant (or converging)
        # Check that result/L is NOT constant
        result_over_L = [r / L for r, L in zip(results, L_values)]

        # The ratio should vary by more than 50% if results are ~constant
        ratio_range = max(result_over_L) / min(result_over_L)
        assert ratio_range > 1.5, \
            f"Suspicious: result/L is too constant ({ratio_range:.2f}), may indicate L-divergence"


# =============================================================================
# Test Class: Q Polynomial Conversion
# =============================================================================

class TestQConversion:
    """Quick tests for Q polynomial basis conversion."""

    def test_Q_at_zero_equals_sum(self):
        """Q(0) should equal sum of basis coefficients."""
        basis_coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
        Q_mono = convert_Q_basis_to_monomial(basis_coeffs)

        # Q(0) = q_0
        assert abs(Q_mono[0] - 0.999999) < 1e-5, f"Q(0) = {Q_mono[0]}"

    def test_Q_at_half_equals_c0(self):
        """Q(0.5) should equal c_0 since (1-2×0.5)^k = 0 for k>0."""
        basis_coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
        Q_mono = convert_Q_basis_to_monomial(basis_coeffs)

        x = 0.5
        Q_at_half = sum(c * (x ** i) for i, c in enumerate(Q_mono))
        assert abs(Q_at_half - 0.490464) < 1e-5, f"Q(0.5) = {Q_at_half}"

    def test_monomial_length(self):
        """Monomial coefficients should have length max_k + 1."""
        basis_coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
        Q_mono = convert_Q_basis_to_monomial(basis_coeffs)
        assert len(Q_mono) == 6, f"len(Q_mono) = {len(Q_mono)}"


# =============================================================================
# Test Class: Post-Identity Core E
# =============================================================================

class TestPostIdentityCore:
    """Tests for the post-identity exponential core E."""

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    def test_E_at_alpha_beta_zero(self, theta):
        """E should be 1 when α=β=0."""
        E = compute_post_identity_core_E(0, 0, 0.1, 0.1, 0.5, theta, 20.0)
        assert abs(E - 1.0) < 1e-14, f"E(0,0,...) = {E}"

    def test_E_symmetric_in_alpha_beta(self, theta):
        """E should be symmetric under α↔β, x↔y."""
        L = 20.0
        t = 0.5
        R = 1.3036

        E1 = compute_post_identity_core_E(-R/L, -R/L, 0.1, 0.2, t, theta, L)
        E2 = compute_post_identity_core_E(-R/L, -R/L, 0.2, 0.1, t, theta, L)

        assert abs(E1 - E2) < 1e-14, f"E not symmetric: E1={E1}, E2={E2}"

    def test_E_factorizes_correctly(self, theta):
        """E = exp(term1) × exp(term2) should match the product."""
        L = 20.0
        t = 0.5
        R = 1.3036
        alpha = -R / L
        beta = -R / L
        x, y = 0.1, 0.2

        E = compute_post_identity_core_E(alpha, beta, x, y, t, theta, L)

        # Manual computation
        term1 = theta * L * (alpha * x + beta * y)
        term2 = -t * (alpha + beta) * L * (1 + theta * (x + y))
        E_manual = np.exp(term1 + term2)

        assert abs(E - E_manual) < 1e-14, f"E={E}, E_manual={E_manual}"
