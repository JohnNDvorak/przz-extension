"""
tests/test_s12_backend_equivalence.py
Phase 27: S12 Backend Tests

Tests for the unified_general S12 backend implementation.
Includes regression tests against Phase 26B validated values.

Phase 26B proved that unified_general correctly implements the PRZZ bracket
with x^ℓ₁y^ℓ₂ coefficient extraction and ℓ₁!ℓ₂! normalization.

Created: 2025-12-26 (Phase 27)
"""

import pytest
import math

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.s12_backend import (
    compute_I1_backend,
    compute_I2_backend,
    compute_S12_pair,
    compute_S12_all_pairs,
    TRIANGLE_PAIRS,
    get_s12_factorial_normalization,
    get_s12_symmetry_factors,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def kappa_polynomials():
    """Load kappa benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def kappa_star_polynomials():
    """Load kappa* benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def kappa_params():
    """Parameters for kappa benchmark."""
    return {"R": 1.3036, "theta": 4 / 7}


@pytest.fixture
def kappa_star_params():
    """Parameters for kappa* benchmark."""
    return {"R": 1.1167, "theta": 4 / 7}


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestBasicFunctionality:
    """Test that basic backend functions work."""

    def test_compute_I1_unified_general_11(self, kappa_polynomials, kappa_params):
        """Test I₁ unified_general backend for (1,1)."""
        result = compute_I1_backend(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=1,
            ell2=1,
            polynomials=kappa_polynomials,
            backend="unified_general",
            n_quad=40,
        )
        assert result.backend == "unified_general"
        assert result.ell1 == 1
        assert result.ell2 == 1
        assert isinstance(result.value, float)
        assert math.isfinite(result.value)

    def test_compute_I1_dsl_11(self, kappa_polynomials, kappa_params):
        """Test I₁ DSL backend for (1,1)."""
        result = compute_I1_backend(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=1,
            ell2=1,
            polynomials=kappa_polynomials,
            backend="dsl",
            n_quad=40,
        )
        assert result.backend == "dsl"
        assert isinstance(result.value, float)
        assert math.isfinite(result.value)

    def test_compute_I2_unified_general_11(self, kappa_polynomials, kappa_params):
        """Test I₂ unified_general backend for (1,1)."""
        result = compute_I2_backend(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=1,
            ell2=1,
            polynomials=kappa_polynomials,
            backend="unified_general",
            n_quad=40,
        )
        assert result.backend == "unified_general"
        assert isinstance(result.value, float)
        assert math.isfinite(result.value)

    def test_compute_S12_pair_11(self, kappa_polynomials, kappa_params):
        """Test S12 pair computation for (1,1)."""
        result = compute_S12_pair(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=1,
            ell2=1,
            polynomials=kappa_polynomials,
            backend="unified_general",
            n_quad=40,
        )
        assert result.S12_value == result.I1_value + result.I2_value
        assert math.isfinite(result.S12_value)

    def test_compute_S12_all_pairs(self, kappa_polynomials, kappa_params):
        """Test S12 computation for all pairs."""
        result = compute_S12_all_pairs(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            polynomials=kappa_polynomials,
            backend="unified_general",
            n_quad=40,
        )
        assert len(result.pair_results) == 6
        assert all(key in result.pair_results for key in TRIANGLE_PAIRS)
        assert math.isfinite(result.total_normalized)


# =============================================================================
# PHASE 26B REGRESSION TESTS - I₁ VALUES
# =============================================================================

# Phase 26B validated I₁ values for kappa benchmark (R=1.3036)
PHASE_26B_I1_KAPPA = {
    "11": 4.13472939e-01,
    "22": 3.88388803e+00,
    "33": 2.86101708e+00,
    "12": -5.65031912e-01,
    "13": -5.81588275e-01,
    "23": 3.57146027e+00,
}


class TestPhase26BRegressionI1Kappa:
    """Regression tests: I₁ values must match Phase 26B validated values."""

    @pytest.mark.parametrize("pair_key,expected", list(PHASE_26B_I1_KAPPA.items()))
    def test_I1_regression(self, kappa_polynomials, kappa_params, pair_key, expected):
        """Test I₁ matches Phase 26B value for each pair."""
        ell1, ell2 = int(pair_key[0]), int(pair_key[1])

        result = compute_I1_backend(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=ell1,
            ell2=ell2,
            polynomials=kappa_polynomials,
            backend="unified_general",
            n_quad=60,
        )

        # Use 1e-6 relative tolerance (accounts for quadrature precision)
        if abs(expected) > 1e-10:
            rel_err = abs(result.value - expected) / abs(expected)
            assert rel_err < 1e-6, (
                f"I1({ell1},{ell2}) regression fail: "
                f"got {result.value:.8e}, expected {expected:.8e}, "
                f"rel_err={rel_err:.2e}"
            )
        else:
            assert abs(result.value - expected) < 1e-10


class TestPhase26BRegressionI1KappaStar:
    """Regression tests for kappa* benchmark (R=1.1167)."""

    def test_I1_11_kappa_star(self, kappa_star_polynomials, kappa_star_params):
        """Test I₁(1,1) produces finite value for kappa* benchmark."""
        result = compute_I1_backend(
            R=kappa_star_params["R"],
            theta=kappa_star_params["theta"],
            ell1=1,
            ell2=1,
            polynomials=kappa_star_polynomials,
            backend="unified_general",
            n_quad=60,
        )
        assert math.isfinite(result.value)
        # Kappa* I₁(1,1) should be positive and in reasonable range
        assert 0.1 < result.value < 1.0

    @pytest.mark.parametrize("pair_key", TRIANGLE_PAIRS)
    def test_I1_finite(self, kappa_star_polynomials, kappa_star_params, pair_key):
        """All I₁ values should be finite for kappa* benchmark."""
        ell1, ell2 = int(pair_key[0]), int(pair_key[1])

        result = compute_I1_backend(
            R=kappa_star_params["R"],
            theta=kappa_star_params["theta"],
            ell1=ell1,
            ell2=ell2,
            polynomials=kappa_star_polynomials,
            backend="unified_general",
            n_quad=60,
        )
        assert math.isfinite(result.value), f"I1({ell1},{ell2}) not finite"


# =============================================================================
# NORMALIZATION TESTS
# =============================================================================

class TestNormalization:
    """Test normalization factors are correct."""

    def test_factorial_normalization_values(self):
        """Test factorial normalization values."""
        norms = get_s12_factorial_normalization()

        # Verify expected values
        assert norms["11"] == 1.0  # 1/(1!*1!) = 1
        assert norms["22"] == 0.25  # 1/(2!*2!) = 1/4
        assert abs(norms["33"] - 1/36) < 1e-15  # 1/(3!*3!) = 1/36
        assert norms["12"] == 0.5  # 1/(1!*2!) = 1/2
        assert abs(norms["13"] - 1/6) < 1e-15  # 1/(1!*3!) = 1/6
        assert abs(norms["23"] - 1/12) < 1e-15  # 1/(2!*3!) = 1/12

    def test_symmetry_factors(self):
        """Test symmetry factors (triangle×2 convention)."""
        sym = get_s12_symmetry_factors()

        # Diagonal pairs have factor 1
        assert sym["11"] == 1.0
        assert sym["22"] == 1.0
        assert sym["33"] == 1.0

        # Off-diagonal pairs have factor 2
        assert sym["12"] == 2.0
        assert sym["13"] == 2.0
        assert sym["23"] == 2.0


# =============================================================================
# INVALID BACKEND TESTS
# =============================================================================

class TestInvalidBackend:
    """Test error handling for invalid backends."""

    def test_invalid_backend_I1(self, kappa_polynomials, kappa_params):
        """Test that invalid backend raises error for I₁."""
        with pytest.raises(ValueError, match="Unknown backend"):
            compute_I1_backend(
                R=kappa_params["R"],
                theta=kappa_params["theta"],
                ell1=1,
                ell2=1,
                polynomials=kappa_polynomials,
                backend="invalid_backend",
            )

    def test_invalid_backend_I2(self, kappa_polynomials, kappa_params):
        """Test that invalid backend raises error for I₂."""
        with pytest.raises(ValueError, match="Unknown backend"):
            compute_I2_backend(
                R=kappa_params["R"],
                theta=kappa_params["theta"],
                ell1=1,
                ell2=1,
                polynomials=kappa_polynomials,
                backend="invalid_backend",
            )


# =============================================================================
# SIGN CONVENTION TESTS
# =============================================================================

class TestSignConventions:
    """Verify sign conventions match Phase 26B."""

    def test_I1_12_is_negative(self, kappa_polynomials, kappa_params):
        """Test I₁(1,2) is negative ((-1)^{1+2} = -1)."""
        result = compute_I1_backend(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=1,
            ell2=2,
            polynomials=kappa_polynomials,
            backend="unified_general",
            n_quad=60,
        )
        assert result.value < 0, f"I1(1,2) should be negative, got {result.value}"

    def test_I1_13_is_negative(self, kappa_polynomials, kappa_params):
        """Test I₁(1,3) is negative ((-1)^{1+3} = +1 but pair has specific sign)."""
        result = compute_I1_backend(
            R=kappa_params["R"],
            theta=kappa_params["theta"],
            ell1=1,
            ell2=3,
            polynomials=kappa_polynomials,
            backend="unified_general",
            n_quad=60,
        )
        # Phase 26B shows (1,3) is negative: -5.81588275e-01
        assert result.value < 0, f"I1(1,3) should be negative, got {result.value}"

    def test_diagonal_pairs_positive(self, kappa_polynomials, kappa_params):
        """Test diagonal pairs (1,1), (2,2), (3,3) are positive."""
        for ell in [1, 2, 3]:
            result = compute_I1_backend(
                R=kappa_params["R"],
                theta=kappa_params["theta"],
                ell1=ell,
                ell2=ell,
                polynomials=kappa_polynomials,
                backend="unified_general",
                n_quad=60,
            )
            assert result.value > 0, f"I1({ell},{ell}) should be positive, got {result.value}"
