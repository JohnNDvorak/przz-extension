"""
tests/test_composition.py
Composition bridge tests: validate polynomial composition with nilpotent perturbations.

These tests bridge:
- polynomials.py (P, Q polynomials)
- series.py (TruncatedSeries with nilpotent variables)
- quadrature.py (grid-shaped coefficient arrays)

Key identity being tested:
    P(u + δ) where δ = Σ aᵢxᵢ (nilpotent vars)

Expands to:
    P(u) + P'(u)δ + P''(u)δ²/2! + ...

Under nilpotent rules (xᵢ² = 0):
    δ² = 2 Σᵢ<ⱼ aᵢaⱼxᵢxⱼ  (only cross terms survive)
    δ³ = 6 a₁a₂a₃ x₁x₂x₃  (for 3 vars)

So coefficient of x₁x₂...xₖ in P(u+δ) is:
    (∏ᵢ aᵢ) · P⁽ᵏ⁾(u)
"""

import numpy as np
import pytest
from src.series import TruncatedSeries
from src.polynomials import Polynomial, P1Polynomial, PellPolynomial, QPolynomial
from src.composition import compose_polynomial_on_affine


# =============================================================================
# Test Group A: Polynomial composition with 2 variables (scalar u)
# =============================================================================

class TestPolynomialComposition2Vars:
    """Test P(u + ax + by) expansion with scalar base point u."""

    def test_constant_polynomial(self):
        """P(x) = 5 gives P(u + δ) = 5 (constant, no derivatives)."""
        poly = Polynomial([5.0])  # constant
        var_names = ("x", "y")
        u0 = np.array(0.37)

        result = compose_polynomial_on_affine(poly, u0, {"x": 1.0, "y": 1.0}, var_names)

        # Only constant term
        np.testing.assert_allclose(result.extract(()), 5.0)
        np.testing.assert_allclose(result.extract(("x",)), 0.0)
        np.testing.assert_allclose(result.extract(("y",)), 0.0)
        np.testing.assert_allclose(result.extract(("x", "y")), 0.0)

    def test_linear_polynomial(self):
        """P(x) = 1 + 2x gives P(u + δ) = P(u) + 2δ."""
        poly = Polynomial([1.0, 2.0])  # 1 + 2x
        var_names = ("x", "y")
        u0 = np.array(0.5)
        a, b = 3.0, 4.0

        result = compose_polynomial_on_affine(poly, u0, {"x": a, "y": b}, var_names)

        # P(u) = 1 + 2*0.5 = 2.0
        np.testing.assert_allclose(result.extract(()), 2.0)
        # P'(u) = 2, so x coeff = a*P'(u) = 3*2 = 6
        np.testing.assert_allclose(result.extract(("x",)), a * 2.0)
        # y coeff = b*P'(u) = 4*2 = 8
        np.testing.assert_allclose(result.extract(("y",)), b * 2.0)
        # No second derivative for linear poly
        np.testing.assert_allclose(result.extract(("x", "y")), 0.0)

    def test_quadratic_polynomial(self):
        """
        P(x) = 1 + 2x + 3x² gives:
        - P(u)
        - x coeff: a*P'(u)
        - y coeff: b*P'(u)
        - xy coeff: ab*P''(u)
        """
        poly = Polynomial([1.0, 2.0, 3.0])  # 1 + 2x + 3x²
        var_names = ("x", "y")
        u0 = np.array(0.37)
        a, b = 2.0, 5.0

        result = compose_polynomial_on_affine(poly, u0, {"x": a, "y": b}, var_names)

        # P(u) = 1 + 2*0.37 + 3*0.37² = 1 + 0.74 + 0.4107 = 2.1507
        P_u = poly.eval(u0)
        np.testing.assert_allclose(result.extract(()), P_u)

        # P'(x) = 2 + 6x, P'(u) = 2 + 6*0.37 = 4.22
        P_prime_u = poly.eval_deriv(u0, 1)
        np.testing.assert_allclose(result.extract(("x",)), a * P_prime_u)
        np.testing.assert_allclose(result.extract(("y",)), b * P_prime_u)

        # P''(x) = 6, P''(u) = 6
        # xy coeff = ab * P''(u) (NOT ab * P''(u)/2!)
        # Because: δ² = 2ab*xy, and Taylor gives P''(u)/2! * δ² = P''(u)/2 * 2ab*xy = ab*P''(u)*xy
        P_double_prime_u = poly.eval_deriv(u0, 2)
        np.testing.assert_allclose(result.extract(("x", "y")), a * b * P_double_prime_u)

    def test_cubic_polynomial(self):
        """P(x) = 1 + 2x + 3x² + 4x³ with 2 variables."""
        poly = Polynomial([1.0, 2.0, 3.0, 4.0])  # 1 + 2x + 3x² + 4x³
        var_names = ("x", "y")
        u0 = np.array(0.25)
        a, b = 1.5, 2.5

        result = compose_polynomial_on_affine(poly, u0, {"x": a, "y": b}, var_names)

        P_u = poly.eval(u0)
        P_prime = poly.eval_deriv(u0, 1)
        P_double_prime = poly.eval_deriv(u0, 2)

        np.testing.assert_allclose(result.extract(()), P_u)
        np.testing.assert_allclose(result.extract(("x",)), a * P_prime)
        np.testing.assert_allclose(result.extract(("y",)), b * P_prime)
        np.testing.assert_allclose(result.extract(("x", "y")), a * b * P_double_prime)

        # Note: P'''(x) = 24, but with only 2 vars we can't get xyz term
        # The xy term is the highest we can extract


# =============================================================================
# Test Group B: Polynomial composition with 3 variables
# =============================================================================

class TestPolynomialComposition3Vars:
    """Test P(u + ax + by + cz) expansion - validates xyz coefficient."""

    def test_cubic_polynomial_3vars(self):
        """
        P(x) = 1 + 2x + 3x² + 4x³
        δ = ax + by + cz

        xyz coefficient = abc * P'''(u)
        """
        poly = Polynomial([1.0, 2.0, 3.0, 4.0])
        var_names = ("x", "y", "z")
        u0 = np.array(0.4)
        a, b, c = 2.0, 3.0, 5.0

        result = compose_polynomial_on_affine(
            poly, u0,
            {"x": a, "y": b, "z": c},
            var_names
        )

        P_u = poly.eval(u0)
        P_prime = poly.eval_deriv(u0, 1)
        P_double_prime = poly.eval_deriv(u0, 2)
        P_triple_prime = poly.eval_deriv(u0, 3)  # = 24 for this polynomial

        # Constant term
        np.testing.assert_allclose(result.extract(()), P_u)

        # Single variable terms
        np.testing.assert_allclose(result.extract(("x",)), a * P_prime)
        np.testing.assert_allclose(result.extract(("y",)), b * P_prime)
        np.testing.assert_allclose(result.extract(("z",)), c * P_prime)

        # Two-variable terms (xy, xz, yz)
        np.testing.assert_allclose(result.extract(("x", "y")), a * b * P_double_prime)
        np.testing.assert_allclose(result.extract(("x", "z")), a * c * P_double_prime)
        np.testing.assert_allclose(result.extract(("y", "z")), b * c * P_double_prime)

        # Three-variable term: xyz coefficient = abc * P'''(u)
        np.testing.assert_allclose(result.extract(("x", "y", "z")), a * b * c * P_triple_prime)

    def test_quadratic_polynomial_3vars_no_xyz(self):
        """Quadratic polynomial has no xyz term (P''' = 0)."""
        poly = Polynomial([1.0, 2.0, 3.0])  # degree 2
        var_names = ("x", "y", "z")
        u0 = np.array(0.5)

        result = compose_polynomial_on_affine(
            poly, u0,
            {"x": 2.0, "y": 3.0, "z": 4.0},
            var_names
        )

        # xyz term should be 0 (no P''')
        np.testing.assert_allclose(result.extract(("x", "y", "z")), 0.0)

    def test_identity_for_general_monomial(self):
        """
        Verify the general identity:
        Coefficient of x₁x₂...xₖ in P(u+δ) = (∏ᵢ aᵢ) · P⁽ᵏ⁾(u)

        Using P(x) = x⁴ and 4 variables.
        """
        poly = Polynomial([0.0, 0.0, 0.0, 0.0, 1.0])  # x⁴
        var_names = ("x1", "x2", "x3", "x4")
        u0 = np.array(0.3)
        a1, a2, a3, a4 = 1.0, 2.0, 3.0, 4.0

        result = compose_polynomial_on_affine(
            poly, u0,
            {"x1": a1, "x2": a2, "x3": a3, "x4": a4},
            var_names
        )

        # P(x) = x⁴
        # P'(x) = 4x³
        # P''(x) = 12x²
        # P'''(x) = 24x
        # P''''(x) = 24

        # x1x2x3x4 coefficient = a1*a2*a3*a4 * P''''(u) = 1*2*3*4 * 24 = 576
        expected = a1 * a2 * a3 * a4 * poly.eval_deriv(u0, 4)
        np.testing.assert_allclose(
            result.extract(("x1", "x2", "x3", "x4")),
            expected
        )


# =============================================================================
# Test Group C: Grid-shaped coefficients (array broadcasting)
# =============================================================================

class TestPolynomialCompositionArrays:
    """Test composition with grid-shaped u (numpy arrays)."""

    def test_2d_grid_base_point(self):
        """P(u + ax + by) where u is a 2D grid."""
        poly = Polynomial([1.0, 2.0, 3.0])  # 1 + 2x + 3x²
        var_names = ("x", "y")

        # u is a 7x5 grid
        u0 = np.linspace(0.1, 0.9, 35).reshape(7, 5)
        a, b = 2.0, 3.0  # scalar coefficients

        result = compose_polynomial_on_affine(poly, u0, {"x": a, "y": b}, var_names)

        # All extracted coefficients should have same shape as u0
        const = result.extract(())
        x_coeff = result.extract(("x",))
        y_coeff = result.extract(("y",))
        xy_coeff = result.extract(("x", "y"))

        assert const.shape == u0.shape
        assert x_coeff.shape == u0.shape
        assert y_coeff.shape == u0.shape
        assert xy_coeff.shape == u0.shape

        # Verify values
        np.testing.assert_allclose(const, poly.eval(u0))
        np.testing.assert_allclose(x_coeff, a * poly.eval_deriv(u0, 1))
        np.testing.assert_allclose(y_coeff, b * poly.eval_deriv(u0, 1))
        np.testing.assert_allclose(xy_coeff, a * b * poly.eval_deriv(u0, 2))

    def test_array_coefficients_in_delta(self):
        """δ = a(u,t)*x + b(u,t)*y where a, b are grid-shaped."""
        poly = Polynomial([1.0, 2.0, 3.0])
        var_names = ("x", "y")

        # u0 and coefficients are all 4x3 grids
        shape = (4, 3)
        u0 = np.linspace(0.2, 0.8, 12).reshape(shape)
        a = np.linspace(1.0, 2.0, 12).reshape(shape)
        b = np.linspace(0.5, 1.5, 12).reshape(shape)

        result = compose_polynomial_on_affine(poly, u0, {"x": a, "y": b}, var_names)

        # Verify shapes
        assert result.extract(()).shape == shape
        assert result.extract(("x",)).shape == shape
        assert result.extract(("y",)).shape == shape
        assert result.extract(("x", "y")).shape == shape

        # Verify values (element-wise)
        P_prime = poly.eval_deriv(u0, 1)
        P_double_prime = poly.eval_deriv(u0, 2)

        np.testing.assert_allclose(result.extract(()), poly.eval(u0))
        np.testing.assert_allclose(result.extract(("x",)), a * P_prime)
        np.testing.assert_allclose(result.extract(("y",)), b * P_prime)
        np.testing.assert_allclose(result.extract(("x", "y")), a * b * P_double_prime)

    def test_3vars_with_grid(self):
        """3 variables with grid-shaped base point."""
        poly = Polynomial([1.0, 1.0, 1.0, 1.0])  # 1 + x + x² + x³
        var_names = ("x", "y", "z")

        shape = (5, 5)
        u0 = np.linspace(0.1, 0.5, 25).reshape(shape)
        a, b, c = 2.0, 3.0, 4.0

        result = compose_polynomial_on_affine(
            poly, u0,
            {"x": a, "y": b, "z": c},
            var_names
        )

        # xyz coefficient
        xyz = result.extract(("x", "y", "z"))
        assert xyz.shape == shape
        np.testing.assert_allclose(xyz, a * b * c * poly.eval_deriv(u0, 3))


# =============================================================================
# Test Group D: Edge cases and integration with quadrature
# =============================================================================

class TestCompositionEdgeCases:
    """Edge cases and integration scenarios."""

    def test_zero_coefficients_in_delta(self):
        """Some coefficients in δ can be zero."""
        poly = Polynomial([1.0, 2.0, 3.0])
        var_names = ("x", "y")
        u0 = np.array(0.5)

        # δ = 2x + 0y = 2x
        result = compose_polynomial_on_affine(poly, u0, {"x": 2.0, "y": 0.0}, var_names)

        # y and xy terms should be zero
        np.testing.assert_allclose(result.extract(("y",)), 0.0)
        np.testing.assert_allclose(result.extract(("x", "y")), 0.0)

        # x term should be 2 * P'(u)
        np.testing.assert_allclose(result.extract(("x",)), 2.0 * poly.eval_deriv(u0, 1))

    def test_partial_variable_set(self):
        """Only some variables have nonzero coefficients."""
        poly = Polynomial([0.0, 0.0, 0.0, 1.0])  # x³
        var_names = ("x", "y", "z")
        u0 = np.array(0.5)

        # Only x has nonzero coefficient
        result = compose_polynomial_on_affine(
            poly, u0,
            {"x": 3.0, "y": 0.0, "z": 0.0},
            var_names
        )

        # Only x term should be nonzero
        P_prime = poly.eval_deriv(u0, 1)
        np.testing.assert_allclose(result.extract(("x",)), 3.0 * P_prime)
        np.testing.assert_allclose(result.extract(("y",)), 0.0)
        np.testing.assert_allclose(result.extract(("z",)), 0.0)

    def test_high_degree_polynomial_limited_by_vars(self):
        """High-degree polynomial but limited variables caps expansion."""
        # Degree 10 polynomial
        poly = Polynomial([1.0] * 11)
        var_names = ("x", "y")  # Only 2 vars
        u0 = np.array(0.3)

        result = compose_polynomial_on_affine(poly, u0, {"x": 1.0, "y": 1.0}, var_names)

        # Can only extract up to xy (2 vars)
        # Higher derivatives exist but can't be accessed with 2 vars
        P_double_prime = poly.eval_deriv(u0, 2)
        np.testing.assert_allclose(result.extract(("x", "y")), P_double_prime)

    def test_composition_then_exp(self):
        """Compose polynomial, then take exp - full PRZZ-style pipeline."""
        # This mimics: exp(R * P(u + δ))
        poly = Polynomial([0.5, 0.3])  # 0.5 + 0.3x (simple linear)
        var_names = ("x", "y")
        u0 = np.array(0.4)
        R = 1.3036  # PRZZ R value
        a, b = 0.5, 0.7

        # Step 1: P(u + δ)
        P_series = compose_polynomial_on_affine(poly, u0, {"x": a, "y": b}, var_names)

        # Step 2: R * P(u + δ)
        scaled = P_series * R

        # Step 3: exp(R * P(u + δ))
        result = scaled.exp()

        # Verify structure: exp(R*P(u)) * (1 + R*P'(u)*δ + ...)
        P_u = poly.eval(u0)
        P_prime = poly.eval_deriv(u0, 1)

        exp_RP = np.exp(R * P_u)

        # Constant term: exp(R*P(u))
        np.testing.assert_allclose(result.extract(()), exp_RP, rtol=1e-10)

        # x term: exp(R*P(u)) * R*a*P'(u)
        np.testing.assert_allclose(
            result.extract(("x",)),
            exp_RP * R * a * P_prime,
            rtol=1e-10
        )

        # y term: exp(R*P(u)) * R*b*P'(u)
        np.testing.assert_allclose(
            result.extract(("y",)),
            exp_RP * R * b * P_prime,
            rtol=1e-10
        )

        # xy term: exp(R*P(u)) * (R*a*P'(u)) * (R*b*P'(u)) / 1
        # Because for linear P, P'' = 0, so the xy term comes only from (δ)² in exp
        # exp(c + N) where N = R*P'(u)*(ax+by), N² = 2*R²*P'²*ab*xy
        # So xy coeff = exp(c) * R²*P'²*ab
        expected_xy = exp_RP * (R * P_prime) ** 2 * a * b
        np.testing.assert_allclose(result.extract(("x", "y")), expected_xy, rtol=1e-10)


# =============================================================================
# Test Group E: PRZZ polynomial wrappers (P1, Pell, Q)
# =============================================================================

class TestPRZZPolynomialComposition:
    """Test composition with PRZZ polynomial wrapper classes.

    These tests verify that compose_polynomial_on_affine works with:
    - P1Polynomial (no direct .degree, has .to_monomial())
    - PellPolynomial (no direct .degree, has .to_monomial())
    - QPolynomial (no direct .degree, has .to_monomial())

    This ensures the _get_poly_degree fallback logic works correctly.
    """

    def test_p1_polynomial_composition(self):
        """P1Polynomial composition matches its monomial equivalent."""
        # P1(x) = x + x(1-x)*P_tilde(1-x) with tilde_coeffs = [0.5, 0.3]
        p1 = P1Polynomial([0.5, 0.3])
        p1_mono = p1.to_monomial()

        var_names = ("x", "y")
        u0 = np.array(0.4)
        a, b = 2.0, 3.0

        # Compose with P1Polynomial directly
        result = compose_polynomial_on_affine(p1, u0, {"x": a, "y": b}, var_names)

        # Compare against monomial composition
        expected = compose_polynomial_on_affine(p1_mono, u0, {"x": a, "y": b}, var_names)

        np.testing.assert_allclose(result.extract(()), expected.extract(()))
        np.testing.assert_allclose(result.extract(("x",)), expected.extract(("x",)))
        np.testing.assert_allclose(result.extract(("y",)), expected.extract(("y",)))
        np.testing.assert_allclose(result.extract(("x", "y")), expected.extract(("x", "y")))

    def test_pell_polynomial_composition(self):
        """PellPolynomial composition matches its monomial equivalent."""
        # P_ell(x) = x*P_tilde(x) with tilde_coeffs = [1.0, 0.5, 0.2]
        pell = PellPolynomial([1.0, 0.5, 0.2])
        pell_mono = pell.to_monomial()

        var_names = ("x", "y", "z")
        u0 = np.array(0.3)
        a, b, c = 1.5, 2.0, 2.5

        # Compose with PellPolynomial directly
        result = compose_polynomial_on_affine(
            pell, u0, {"x": a, "y": b, "z": c}, var_names
        )

        # Compare against monomial composition
        expected = compose_polynomial_on_affine(
            pell_mono, u0, {"x": a, "y": b, "z": c}, var_names
        )

        np.testing.assert_allclose(result.extract(()), expected.extract(()))
        np.testing.assert_allclose(result.extract(("x",)), expected.extract(("x",)))
        np.testing.assert_allclose(result.extract(("x", "y")), expected.extract(("x", "y")))
        np.testing.assert_allclose(result.extract(("x", "y", "z")), expected.extract(("x", "y", "z")))

    def test_q_polynomial_composition_enforce_mode(self):
        """QPolynomial composition in enforce_Q0=True mode."""
        # Q(x) in (1-2x)^k basis with enforce_Q0=True
        q = QPolynomial({1: 0.3, 2: 0.2, 3: 0.1}, enforce_Q0=True)
        q_mono = q.to_monomial()

        var_names = ("x", "y")
        u0 = np.array(0.25)
        a, b = 1.0, 2.0

        # Compose with QPolynomial directly
        result = compose_polynomial_on_affine(q, u0, {"x": a, "y": b}, var_names)

        # Compare against monomial composition
        expected = compose_polynomial_on_affine(q_mono, u0, {"x": a, "y": b}, var_names)

        np.testing.assert_allclose(result.extract(()), expected.extract(()))
        np.testing.assert_allclose(result.extract(("x",)), expected.extract(("x",)))
        np.testing.assert_allclose(result.extract(("y",)), expected.extract(("y",)))
        np.testing.assert_allclose(result.extract(("x", "y")), expected.extract(("x", "y")))

    def test_q_polynomial_composition_paper_literal(self):
        """QPolynomial composition in enforce_Q0=False mode."""
        # Q(x) in (1-2x)^k basis with enforce_Q0=False
        q = QPolynomial({0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}, enforce_Q0=False)
        q_mono = q.to_monomial()

        var_names = ("x", "y")
        u0 = np.array(0.35)
        a, b = 0.8, 1.2

        # Compose with QPolynomial directly
        result = compose_polynomial_on_affine(q, u0, {"x": a, "y": b}, var_names)

        # Compare against monomial composition
        expected = compose_polynomial_on_affine(q_mono, u0, {"x": a, "y": b}, var_names)

        np.testing.assert_allclose(result.extract(()), expected.extract(()))
        np.testing.assert_allclose(result.extract(("x",)), expected.extract(("x",)))
        np.testing.assert_allclose(result.extract(("x", "y")), expected.extract(("x", "y")))

    def test_p1_with_grid_arrays(self):
        """P1Polynomial composition with grid-shaped base point."""
        p1 = P1Polynomial([0.2, 0.4, 0.1])

        var_names = ("x", "y")
        shape = (5, 4)
        u0 = np.linspace(0.1, 0.8, 20).reshape(shape)
        a, b = 1.5, 2.5

        result = compose_polynomial_on_affine(p1, u0, {"x": a, "y": b}, var_names)

        # Check shapes
        assert result.extract(()).shape == shape
        assert result.extract(("x",)).shape == shape
        assert result.extract(("y",)).shape == shape
        assert result.extract(("x", "y")).shape == shape

        # Verify values against monomial composition
        p1_mono = p1.to_monomial()
        expected = compose_polynomial_on_affine(p1_mono, u0, {"x": a, "y": b}, var_names)

        np.testing.assert_allclose(result.extract(()), expected.extract(()))
        np.testing.assert_allclose(result.extract(("x", "y")), expected.extract(("x", "y")))

    def test_przz_polynomial_degree_fallback(self):
        """Verify composition works even without .degree (pure protocol test).

        This test creates a minimal poly-like object that only has eval_deriv,
        to verify the fallback to n_vars works correctly.
        """
        class MinimalPoly:
            """Polynomial-like with only eval_deriv (no degree, no to_monomial)."""
            def eval_deriv(self, x, k):
                # Represents P(x) = 1 + x + x^2
                # P'(x) = 1 + 2x, P''(x) = 2, P'''(x) = 0
                x = np.asarray(x)
                if k == 0:
                    return 1.0 + x + x**2
                elif k == 1:
                    return 1.0 + 2*x
                elif k == 2:
                    return np.full_like(x, 2.0)
                else:
                    return np.zeros_like(x)

        poly = MinimalPoly()
        var_names = ("x", "y")
        u0 = np.array(0.5)
        a, b = 2.0, 3.0

        # Should work with fallback to n_vars=2
        result = compose_polynomial_on_affine(poly, u0, {"x": a, "y": b}, var_names)

        # P(u0) = 1 + 0.5 + 0.25 = 1.75
        np.testing.assert_allclose(result.extract(()), 1.75)
        # x coeff = a * P'(u0) = 2 * (1 + 2*0.5) = 2 * 2 = 4
        np.testing.assert_allclose(result.extract(("x",)), 4.0)
        # y coeff = b * P'(u0) = 3 * 2 = 6
        np.testing.assert_allclose(result.extract(("y",)), 6.0)
        # xy coeff = a * b * P''(u0) = 2 * 3 * 2 = 12
        np.testing.assert_allclose(result.extract(("x", "y")), 12.0)
