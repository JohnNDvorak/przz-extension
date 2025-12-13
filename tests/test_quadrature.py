"""
Unit tests for quadrature module.

Tests cover:
- Weight/node sanity (sum, bounds, positivity, symmetry)
- 1D monomial exactness (Gauss-Legendre property)
- 1D smooth function integration
- 2D separable monomial integration
- 2D convergence via reference-based test
- Grid shape and indexing verification
- Caching safety (read-only arrays)
"""

import math
import numpy as np
import pytest

from src.quadrature import gauss_legendre_01, tensor_grid_2d, tensor_grid_3d


# =============================================================================
# Weight Sanity Tests
# =============================================================================

class TestWeightSanity:
    """Tests for basic quadrature weight/node properties."""

    @pytest.mark.parametrize("n", [10, 20, 40, 60, 100])
    def test_weights_sum_to_one(self, n):
        """Weights should sum to 1 (integral of f=1 over [0,1])."""
        nodes, weights = gauss_legendre_01(n)
        assert abs(np.sum(weights) - 1.0) < 1e-14

    @pytest.mark.parametrize("n", [10, 20, 40, 60, 100])
    def test_nodes_in_unit_interval(self, n):
        """All nodes should be in [0, 1]."""
        nodes, weights = gauss_legendre_01(n)
        assert np.all(nodes >= 0.0)
        assert np.all(nodes <= 1.0)

    @pytest.mark.parametrize("n", [10, 20, 40, 60, 100])
    def test_weights_positive(self, n):
        """All weights should be strictly positive."""
        nodes, weights = gauss_legendre_01(n)
        assert np.all(weights > 0)

    @pytest.mark.parametrize("n", [10, 20, 40, 60, 100])
    def test_nodes_weights_symmetric(self, n):
        """Nodes and weights should be symmetric about 0.5."""
        nodes, weights = gauss_legendre_01(n)

        # Nodes: x_i + x_{n-1-i} should equal 1.0
        for i in range(n // 2):
            assert abs(nodes[i] + nodes[n - 1 - i] - 1.0) < 1e-14, (
                f"Node symmetry violated at i={i}: "
                f"{nodes[i]} + {nodes[n-1-i]} != 1.0"
            )

        # Weights: w_i == w_{n-1-i}
        for i in range(n // 2):
            assert abs(weights[i] - weights[n - 1 - i]) < 1e-14, (
                f"Weight symmetry violated at i={i}: "
                f"{weights[i]} != {weights[n-1-i]}"
            )

    def test_invalid_n_raises(self):
        """n < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            gauss_legendre_01(0)
        with pytest.raises(ValueError):
            gauss_legendre_01(-1)


# =============================================================================
# 1D Monomial Exactness Tests
# =============================================================================

class Test1DMonomialExactness:
    """Test Gauss-Legendre exactness: n-point rule is exact for polynomials up to degree 2n-1."""

    @pytest.mark.parametrize("n", [20, 40, 60])
    def test_monomial_exactness(self, n):
        """Integral of x^k from 0 to 1 equals 1/(k+1) for k = 0..2n-1."""
        nodes, weights = gauss_legendre_01(n)

        for k in range(2 * n):
            integral = np.sum(weights * nodes**k)
            expected = 1.0 / (k + 1)
            assert abs(integral - expected) < 1e-12, (
                f"n={n}, k={k}: got {integral}, expected {expected}"
            )

    def test_monomial_beyond_exactness_limit(self):
        """Integral of x^(2n) should NOT be exact (sanity check)."""
        n = 10
        k = 2 * n  # Beyond exactness limit
        nodes, weights = gauss_legendre_01(n)

        integral = np.sum(weights * nodes**k)
        expected = 1.0 / (k + 1)

        # Should have some error (though typically still small)
        # This is a sanity check that we understand the exactness limit
        error = abs(integral - expected)
        # Error should be nonzero but still reasonably small for k=20
        assert error > 1e-15, "Expected some error beyond exactness limit"


# =============================================================================
# 1D Smooth Function Tests
# =============================================================================

class Test1DSmoothFunction:
    """Test integration of smooth non-polynomial functions."""

    @pytest.mark.parametrize("n", [20, 40, 60])
    def test_exp_minus_x_squared(self, n):
        """Integral of exp(-x^2) from 0 to 1 equals (sqrt(pi)/2) * erf(1)."""
        nodes, weights = gauss_legendre_01(n)

        integral = np.sum(weights * np.exp(-nodes**2))
        expected = math.sqrt(math.pi) / 2 * math.erf(1)

        assert abs(integral - expected) < 1e-10, (
            f"n={n}: got {integral}, expected {expected}"
        )

    @pytest.mark.parametrize("n", [20, 40, 60])
    def test_sin_pi_x(self, n):
        """Integral of sin(pi*x) from 0 to 1 equals 2/pi."""
        nodes, weights = gauss_legendre_01(n)

        integral = np.sum(weights * np.sin(math.pi * nodes))
        expected = 2.0 / math.pi

        assert abs(integral - expected) < 1e-10, (
            f"n={n}: got {integral}, expected {expected}"
        )


# =============================================================================
# 2D Separable Tests
# =============================================================================

class Test2DSeparable:
    """Test 2D tensor product on separable integrands."""

    def test_separable_monomials(self):
        """Integral of u^a * t^b over [0,1]^2 equals 1/((a+1)(b+1))."""
        n = 40
        U, T, W = tensor_grid_2d(n)

        for a in range(5):
            for b in range(5):
                integral = np.sum(W * U**a * T**b)
                expected = 1.0 / ((a + 1) * (b + 1))
                assert abs(integral - expected) < 1e-12, (
                    f"a={a}, b={b}: got {integral}, expected {expected}"
                )

    def test_separable_exp_product(self):
        """Integral of exp(-u) * exp(-t) over [0,1]^2 equals (1-1/e)^2."""
        n = 40
        U, T, W = tensor_grid_2d(n)

        integral = np.sum(W * np.exp(-U) * np.exp(-T))
        expected = (1 - 1 / math.e) ** 2

        assert abs(integral - expected) < 1e-12


# =============================================================================
# 2D Convergence Tests (Reference-Based)
# =============================================================================

class Test2DConvergence:
    """Test 2D convergence against analytically-reduced 1D reference."""

    def test_2d_matches_1d_reference(self):
        """
        Test f(u,t) = exp(-ut) * sin(pi*u) with reference from 1D reduction.

        Analytic reduction:
        - Integrate over t: integral_0^1 exp(-ut) dt = (1 - exp(-u)) / u
        - 2D integral = integral_0^1 sin(pi*u) * (1 - exp(-u)) / u du

        We compute the reference using high-accuracy 1D quadrature (n=400).
        """
        # Compute reference value via high-accuracy 1D quadrature
        n_ref = 400
        nodes_ref, weights_ref = gauss_legendre_01(n_ref)

        # Integrand: sin(pi*u) * (1 - exp(-u)) / u
        # Handle u=0 via limit: lim_{u->0} (1-exp(-u))/u = 1
        def reduced_integrand(u):
            # Use np.where for vectorized handling of u near 0
            result = np.where(
                np.abs(u) < 1e-10,
                np.sin(math.pi * u),  # sin(pi*u) * 1 at u=0
                np.sin(math.pi * u) * (1 - np.exp(-u)) / u
            )
            return result

        reference = np.sum(weights_ref * reduced_integrand(nodes_ref))

        # Now compute via 2D quadrature and compare
        for n in [60, 80, 100]:
            U, T, W = tensor_grid_2d(n)
            integral_2d = np.sum(W * np.exp(-U * T) * np.sin(math.pi * U))

            error = abs(integral_2d - reference)
            assert error < 1e-10, (
                f"n={n}: 2D integral {integral_2d} differs from reference "
                f"{reference} by {error}"
            )

    def test_convergence_rate(self):
        """Verify error decreases with increasing n OR is at machine precision."""
        # Compute reference
        n_ref = 400
        nodes_ref, weights_ref = gauss_legendre_01(n_ref)

        def reduced_integrand(u):
            return np.where(
                np.abs(u) < 1e-10,
                np.sin(math.pi * u),
                np.sin(math.pi * u) * (1 - np.exp(-u)) / u
            )

        reference = np.sum(weights_ref * reduced_integrand(nodes_ref))

        # Compute errors at different n values
        errors = {}
        for n in [40, 60, 80, 100]:
            U, T, W = tensor_grid_2d(n)
            integral_2d = np.sum(W * np.exp(-U * T) * np.sin(math.pi * U))
            errors[n] = abs(integral_2d - reference)

        # Either: n=100 error < n=40 error (convergence)
        # OR: both errors are at machine precision (already converged)
        machine_eps = 1e-13
        if errors[40] > machine_eps:
            assert errors[100] < errors[40], (
                f"Error did not decrease: n=40 error={errors[40]}, "
                f"n=100 error={errors[100]}"
            )
        else:
            # Both should be at machine precision
            assert errors[100] < machine_eps, (
                f"Expected n=100 error near machine precision, got {errors[100]}"
            )


# =============================================================================
# 2D Shape Tests
# =============================================================================

class Test2DShapes:
    """Test 2D grid shapes and indexing."""

    @pytest.mark.parametrize("n", [10, 40, 100])
    def test_grid_shapes(self, n):
        """U, T, W should all have shape (n, n)."""
        U, T, W = tensor_grid_2d(n)
        assert U.shape == (n, n)
        assert T.shape == (n, n)
        assert W.shape == (n, n)

    def test_indexing_ij(self):
        """U varies along axis 0, T varies along axis 1."""
        n = 20
        U, T, W = tensor_grid_2d(n)
        nodes, _ = gauss_legendre_01(n)

        # U[i, :] should all equal nodes[i]
        for i in range(n):
            assert np.allclose(U[i, :], nodes[i])

        # T[:, j] should all equal nodes[j]
        for j in range(n):
            assert np.allclose(T[:, j], nodes[j])

    def test_weight_grid_is_outer_product(self):
        """W should equal outer product of 1D weights."""
        n = 30
        U, T, W = tensor_grid_2d(n)
        _, weights = gauss_legendre_01(n)

        expected_W = np.outer(weights, weights)
        assert np.allclose(W, expected_W)


# =============================================================================
# 3D Tests
# =============================================================================

class Test3DGrid:
    """Basic tests for 3D tensor grid."""

    def test_3d_shapes(self):
        """X, Y, Z, W should all have shape (n, n, n)."""
        n = 10
        X, Y, Z, W = tensor_grid_3d(n)
        assert X.shape == (n, n, n)
        assert Y.shape == (n, n, n)
        assert Z.shape == (n, n, n)
        assert W.shape == (n, n, n)

    def test_3d_separable_monomial(self):
        """Integral of x^a * y^b * z^c over [0,1]^3."""
        n = 20
        X, Y, Z, W = tensor_grid_3d(n)

        a, b, c = 2, 3, 4
        integral = np.sum(W * X**a * Y**b * Z**c)
        expected = 1.0 / ((a + 1) * (b + 1) * (c + 1))
        assert abs(integral - expected) < 1e-12

    def test_3d_weights_sum(self):
        """3D weights should sum to 1."""
        n = 15
        X, Y, Z, W = tensor_grid_3d(n)
        assert abs(np.sum(W) - 1.0) < 1e-14


# =============================================================================
# Caching Safety Tests
# =============================================================================

class TestCachingSafety:
    """Test that cached arrays are protected from mutation."""

    def test_cached_arrays_readonly(self):
        """Cached nodes and weights should be read-only."""
        # Clear cache to ensure fresh call
        gauss_legendre_01.cache_clear()

        nodes, weights = gauss_legendre_01(20)

        assert not nodes.flags.writeable, "nodes should be read-only"
        assert not weights.flags.writeable, "weights should be read-only"

        # Attempting to write should raise
        with pytest.raises((ValueError, TypeError)):
            nodes[0] = 999.0

        with pytest.raises((ValueError, TypeError)):
            weights[0] = 999.0

    def test_cache_returns_same_object(self):
        """Multiple calls with same n should return same cached objects."""
        gauss_legendre_01.cache_clear()

        nodes1, weights1 = gauss_legendre_01(30)
        nodes2, weights2 = gauss_legendre_01(30)

        # Should be the exact same objects (same id)
        assert nodes1 is nodes2, "Expected same cached nodes object"
        assert weights1 is weights2, "Expected same cached weights object"

    def test_different_n_gives_different_objects(self):
        """Different n values should give different objects."""
        gauss_legendre_01.cache_clear()

        nodes1, _ = gauss_legendre_01(20)
        nodes2, _ = gauss_legendre_01(30)

        assert nodes1 is not nodes2
