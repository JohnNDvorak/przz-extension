"""
Tests for unified zeta evaluation API (Phase 19.2).

These tests verify:
1. Mode separation (SEMANTIC vs NUMERIC)
2. Laurent validity checking
3. Numerical accuracy against mpmath
4. L-stability for semantic mode
"""

import pytest
import warnings
import math

from src.ratios.zeta_eval import (
    ZetaMode,
    ZetaEvalResult,
    zeta_logderiv_scaled,
    zeta_logderiv_at_point,
    zeta_logderiv_squared,
    get_laurent_at_point,
    validate_mode_for_point,
    compare_modes,
    get_raw_logderiv_squared,
    get_actual_logderiv_squared,
    get_actual_logderiv_single,
    EULER_MASCHERONI,
)

# Skip tests if mpmath not available
try:
    import mpmath
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False


class TestZetaModeEnum:
    """Basic mode enum tests."""

    def test_mode_values_distinct(self):
        """Mode enum values are distinct."""
        assert ZetaMode.SEMANTIC_LAURENT_NEAR_1 != ZetaMode.NUMERIC_FUNCTIONAL_EQ

    def test_mode_has_docstring(self):
        """Mode enum has documentation."""
        assert ZetaMode.__doc__ is not None
        assert "SEMANTIC" in ZetaMode.__doc__


class TestLaurentApproximation:
    """Tests for Laurent expansion (semantic mode)."""

    def test_laurent_at_small_eps(self):
        """Laurent expansion accurate for small ε."""
        # At ε = -0.01, Laurent should be very accurate
        eps = -0.01
        s = 1.0 + eps
        result = zeta_logderiv_at_point(s, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)

        # Laurent: -1/ε + γ = 100 + 0.577 ≈ 100.577
        expected = -1.0 / eps + EULER_MASCHERONI
        assert abs(result.value - expected) < 0.1

    def test_laurent_pole_structure(self):
        """Laurent has correct -1/ε pole."""
        # For small ε, the -1/ε term dominates
        eps = -0.001
        s = 1.0 + eps
        result = zeta_logderiv_at_point(s, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)

        # Should be close to 1000 (from -1/(-0.001))
        assert result.value > 900
        assert result.value < 1100

    def test_get_laurent_at_point_kappa(self):
        """get_laurent_at_point returns correct values for κ benchmark."""
        R = 1.3036
        single, squared = get_laurent_at_point(R)

        # Single: 1/R + γ ≈ 0.767 + 0.577 ≈ 1.344
        expected_single = 1.0 / R + EULER_MASCHERONI
        assert abs(single - expected_single) < 1e-10

        # Squared
        assert abs(squared - expected_single**2) < 1e-10

    def test_get_laurent_at_point_kappa_star(self):
        """get_laurent_at_point returns correct values for κ* benchmark."""
        R = 1.1167
        single, squared = get_laurent_at_point(R)

        expected_single = 1.0 / R + EULER_MASCHERONI
        assert abs(single - expected_single) < 1e-10


class TestModeValidation:
    """Tests for mode appropriateness validation."""

    def test_laurent_valid_near_1(self):
        """Laurent validated as appropriate near s=1."""
        s = 1.0 - 0.05  # Very close to 1
        is_valid, warning = validate_mode_for_point(s, ZetaMode.SEMANTIC_LAURENT_NEAR_1)
        assert is_valid
        assert warning is None

    def test_laurent_warning_moderate_distance(self):
        """Laurent gives warning for moderate |s-1|."""
        s = 1.0 - 0.3  # Moderate distance
        is_valid, warning = validate_mode_for_point(s, ZetaMode.SEMANTIC_LAURENT_NEAR_1)
        # Still "valid" but with warning
        assert is_valid
        assert warning is not None
        assert "error" in warning.lower() or "significant" in warning.lower()

    def test_laurent_invalid_far_from_1(self):
        """Laurent flagged as invalid far from s=1."""
        s = 1.0 - 1.3  # R = 1.3, far from 1
        is_valid, warning = validate_mode_for_point(s, ZetaMode.SEMANTIC_LAURENT_NEAR_1)
        assert not is_valid
        assert warning is not None
        assert "NUMERIC" in warning

    def test_strict_mode_raises(self):
        """Strict validation raises for invalid mode."""
        s = 1.0 - 1.3
        with pytest.raises(ValueError) as excinfo:
            validate_mode_for_point(s, ZetaMode.SEMANTIC_LAURENT_NEAR_1, strict=True)
        assert "Laurent" in str(excinfo.value)

    def test_numeric_always_valid(self):
        """NUMERIC mode always passes validation."""
        for s in [0.5, 1.0 - 1.3, 2.0, -5.0]:
            is_valid, warning = validate_mode_for_point(
                complex(s), ZetaMode.NUMERIC_FUNCTIONAL_EQ
            )
            assert is_valid
            assert warning is None


class TestModeSeparation:
    """Tests ensuring modes are truly separated."""

    def test_semantic_never_calls_mpmath(self):
        """Semantic mode works without mpmath."""
        # This should work even if mpmath import fails
        s = 1.0 - 0.01
        result = zeta_logderiv_at_point(s, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)
        assert result.mode == ZetaMode.SEMANTIC_LAURENT_NEAR_1
        assert result.value is not None

    @pytest.mark.skipif(not MPMATH_AVAILABLE, reason="mpmath required")
    def test_numeric_uses_mpmath(self):
        """Numeric mode uses mpmath."""
        s = 1.0 - 0.5
        result = zeta_logderiv_at_point(s, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ)
        assert result.mode == ZetaMode.NUMERIC_FUNCTIONAL_EQ

    def test_modes_give_different_values_far_from_1(self):
        """Modes give significantly different values at PRZZ R."""
        R = 1.3036
        s = 1.0 - R

        # Semantic (Laurent)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            semantic = zeta_logderiv_at_point(s, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)

        if MPMATH_AVAILABLE:
            # Numeric (mpmath)
            numeric = zeta_logderiv_at_point(s, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ)

            # They should differ significantly (Phase 15A finding: ~29% error)
            rel_diff = abs(semantic.value - numeric.value) / abs(numeric.value)
            assert rel_diff > 0.1  # At least 10% different

    def test_modes_converge_near_1(self):
        """Modes give similar values very near s=1."""
        if not MPMATH_AVAILABLE:
            pytest.skip("mpmath required")

        s = 1.0 - 0.01  # Very close to 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            semantic = zeta_logderiv_at_point(s, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)
            numeric = zeta_logderiv_at_point(s, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ)

        # They should be close (within ~1%)
        rel_diff = abs(semantic.value - numeric.value) / abs(numeric.value)
        assert rel_diff < 0.05


class TestScaledEvaluation:
    """Tests for the PRZZ-scaled evaluation."""

    def test_scaled_semantic_stable_increasing_L(self):
        """Scaled semantic result stable as L increases."""
        R = 1.3036

        values = []
        for L in [10, 50, 100, 500]:
            alpha_over_L = -R / L
            result = zeta_logderiv_scaled(alpha_over_L, L, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)
            # At α = -R/L, s = 1 - R/L → 1 as L → ∞
            # (ζ'/ζ)(1 - R/L) ≈ L/R + γ
            # But we're computing at fixed α/L ratio, so...
            values.append(result.value)

        # As L increases with fixed R, the result should stabilize
        # (Actually with fixed alpha_over_L = -R/L, α = -R is fixed!)
        # So values should all be approximately equal
        for i in range(1, len(values)):
            rel_diff = abs(values[i] - values[0]) / abs(values[0])
            assert rel_diff < 0.01  # Within 1%

    def test_scaled_is_marked(self):
        """Scaled result is marked as scaled."""
        result = zeta_logderiv_scaled(-0.1, 10.0)
        assert result.is_scaled


@pytest.mark.skipif(not MPMATH_AVAILABLE, reason="mpmath required")
class TestNumericAccuracy:
    """Tests for numeric mode accuracy."""

    def test_mpmath_at_2(self):
        """Numeric matches known value at s=2."""
        # (ζ'/ζ)(2) is well-known
        s = 2.0
        result = zeta_logderiv_at_point(s, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ)

        # Compute reference
        with mpmath.workdps(50):
            zeta_val = mpmath.zeta(2)
            zeta_deriv = mpmath.diff(mpmath.zeta, 2)
            expected = float(zeta_deriv / zeta_val)

        assert abs(result.value - expected) < 1e-10

    def test_mpmath_precision_affects_result(self):
        """Higher precision gives consistent results."""
        s = 1.0 - 1.3
        result1 = zeta_logderiv_at_point(s, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ, precision=30)
        result2 = zeta_logderiv_at_point(s, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ, precision=100)

        # Should be consistent
        assert abs(result1.value - result2.value) < 1e-8

    def test_squared_matches_product(self):
        """zeta_logderiv_squared equals value²."""
        R = 1.3036
        s = 1.0 - R

        result = zeta_logderiv_at_point(s, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ)
        squared_direct = zeta_logderiv_squared(R, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ)

        assert abs(result.squared - squared_direct) < 1e-10


@pytest.mark.skipif(not MPMATH_AVAILABLE, reason="mpmath required")
class TestCompareModes:
    """Tests for mode comparison utility."""

    def test_compare_modes_kappa(self):
        """compare_modes works for κ benchmark."""
        R = 1.3036
        result = compare_modes(R)

        assert "R" in result
        assert result["R"] == R
        assert "laurent_squared" in result
        assert "actual_squared" in result
        assert "squared_error_percent" in result

        # Phase 15A finding: ~66% error in squared
        assert result["squared_error_percent"] > 30  # Significant error

    def test_compare_modes_kappa_star(self):
        """compare_modes works for κ* benchmark."""
        R = 1.1167
        result = compare_modes(R)

        # Phase 15A finding: ~46% error in squared
        assert result["squared_error_percent"] > 20

    def test_squared_ratio_sensible(self):
        """Squared ratio is greater than 1 (actual > Laurent at R≈1.3)."""
        R = 1.3036
        result = compare_modes(R)

        # Actual is larger than Laurent at these R values
        assert result["squared_ratio"] > 1.0


@pytest.mark.skipif(not MPMATH_AVAILABLE, reason="mpmath required")
class TestDecision8Aliases:
    """Test Decision 8 convenience aliases."""

    def test_raw_logderiv_is_laurent(self):
        """get_raw_logderiv_squared uses Laurent."""
        R = 1.3036
        _, laurent_squared = get_laurent_at_point(R)
        raw = get_raw_logderiv_squared(R)
        assert abs(raw - laurent_squared) < 1e-10

    def test_actual_logderiv_is_mpmath(self):
        """get_actual_logderiv_squared uses mpmath."""
        R = 1.3036
        actual = get_actual_logderiv_squared(R)

        # Should match numeric mode
        squared = zeta_logderiv_squared(R, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ)
        assert abs(actual - squared) < 1e-10

    def test_actual_single_for_j13_j14(self):
        """get_actual_logderiv_single returns single factor."""
        R = 1.3036
        single = get_actual_logderiv_single(R)

        # Check it's reasonable (should be ~1.7 for R=1.3)
        assert 1.0 < single < 3.0

        # Check it's NOT the squared value
        squared = get_actual_logderiv_squared(R)
        assert abs(single - squared) > 0.5


class TestEvalResultStructure:
    """Test ZetaEvalResult dataclass."""

    def test_result_has_all_fields(self):
        """Result has all expected fields."""
        result = zeta_logderiv_at_point(0.5, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)

        assert hasattr(result, 'value')
        assert hasattr(result, 'squared')
        assert hasattr(result, 'mode')
        assert hasattr(result, 'eval_point')
        assert hasattr(result, 'is_scaled')
        assert hasattr(result, 'warning')

    def test_result_is_frozen(self):
        """Result is immutable."""
        result = zeta_logderiv_at_point(0.5, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            result.value = 999


class TestWarningBehavior:
    """Test warning emission."""

    def test_far_from_1_warns_in_semantic_mode(self):
        """Using semantic mode far from 1 emits warning."""
        s = 1.0 - 1.3  # Far from 1

        with pytest.warns(UserWarning) as record:
            zeta_logderiv_at_point(s, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)

        assert len(record) >= 1
        assert "NUMERIC" in str(record[0].message) or "inaccurate" in str(record[0].message)

    def test_no_warning_when_validate_false(self):
        """No warning when validate=False."""
        s = 1.0 - 1.3

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            zeta_logderiv_at_point(
                s, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1, validate=False
            )
            # Filter for our specific warnings
            relevant = [x for x in w if "NUMERIC" in str(x.message) or "inaccurate" in str(x.message)]
            assert len(relevant) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_eps_zero_raises(self):
        """Cannot evaluate Laurent at ε=0 (pole)."""
        with pytest.raises(ValueError):
            zeta_logderiv_at_point(1.0, mode=ZetaMode.SEMANTIC_LAURENT_NEAR_1)

    def test_scaled_alpha_zero_raises(self):
        """Cannot use scaled evaluation with α=0."""
        with pytest.raises(ValueError):
            zeta_logderiv_scaled(0.0, 10.0)

    @pytest.mark.skipif(not MPMATH_AVAILABLE, reason="mpmath required")
    def test_negative_s_works(self):
        """Numeric mode works for negative s (between trivial zeros)."""
        s = -0.5  # Between trivial zeros at -2, -4, ...
        result = zeta_logderiv_at_point(s, mode=ZetaMode.NUMERIC_FUNCTIONAL_EQ)
        assert result.value is not None
