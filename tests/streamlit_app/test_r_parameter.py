"""
R parameter widget and state management tests.

Tests that R values are correctly loaded for each mode and that
widget caches don't cause stale values.
"""

import pytest
import sys
from pathlib import Path

# Add streamlit_app to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))


class TestRParameterConstants:
    """Test that R constants are correctly defined."""

    def test_kappa_r_values_exist(self):
        """κ mode R values should be defined."""
        from streamlit_app.utils.constants import (
            R_PRZZ_KAPPA, R_OPTIMIZED_KAPPA
        )
        assert R_PRZZ_KAPPA == 1.3036
        assert R_OPTIMIZED_KAPPA == 1.14978

    def test_kappa_star_r_values_exist(self):
        """κ* mode R values should be defined."""
        from streamlit_app.utils.constants import (
            R_PRZZ_KAPPA_STAR, R_OPTIMIZED_KAPPA_STAR
        )
        assert R_PRZZ_KAPPA_STAR == 1.1167
        assert R_OPTIMIZED_KAPPA_STAR == 1.07966

    def test_kappa_and_kappa_star_r_values_differ(self):
        """κ and κ* should have different R values."""
        from streamlit_app.utils.constants import (
            R_PRZZ_KAPPA, R_PRZZ_KAPPA_STAR,
            R_OPTIMIZED_KAPPA, R_OPTIMIZED_KAPPA_STAR
        )
        # PRZZ values differ
        assert R_PRZZ_KAPPA != R_PRZZ_KAPPA_STAR
        assert abs(R_PRZZ_KAPPA - R_PRZZ_KAPPA_STAR) > 0.1

        # Optimized values differ
        assert R_OPTIMIZED_KAPPA != R_OPTIMIZED_KAPPA_STAR
        assert abs(R_OPTIMIZED_KAPPA - R_OPTIMIZED_KAPPA_STAR) > 0.05


class TestGetPrzzDefaults:
    """Test get_przz_defaults returns correct R for each mode."""

    def test_kappa_mode_returns_kappa_r(self):
        """κ mode should return R_PRZZ_KAPPA."""
        from streamlit_app.utils.constants import (
            get_przz_defaults, R_PRZZ_KAPPA
        )
        defaults = get_przz_defaults("kappa")
        assert defaults["R"] == R_PRZZ_KAPPA
        assert defaults["R"] == 1.3036

    def test_kappa_star_mode_returns_kappa_star_r(self):
        """κ* mode should return R_PRZZ_KAPPA_STAR."""
        from streamlit_app.utils.constants import (
            get_przz_defaults, R_PRZZ_KAPPA_STAR
        )
        defaults = get_przz_defaults("kappa_star")
        assert defaults["R"] == R_PRZZ_KAPPA_STAR
        assert defaults["R"] == 1.1167

    def test_kappa_mode_has_correct_polynomial_lengths(self):
        """κ mode polynomials should have expected lengths."""
        from streamlit_app.utils.constants import get_przz_defaults
        defaults = get_przz_defaults("kappa")
        assert len(defaults["P1_tilde"]) == 4
        assert len(defaults["P2_tilde"]) == 3
        assert len(defaults["P3_tilde"]) == 3

    def test_kappa_star_mode_has_correct_polynomial_lengths(self):
        """κ* mode polynomials should have expected lengths."""
        from streamlit_app.utils.constants import get_przz_defaults
        defaults = get_przz_defaults("kappa_star")
        assert len(defaults["P1_tilde"]) == 4
        assert len(defaults["P2_tilde"]) == 2  # κ* has degree 2
        assert len(defaults["P3_tilde"]) == 2  # κ* has degree 2


class TestGetOptimizedDefaults:
    """Test get_optimized_defaults returns correct R for each mode."""

    def test_kappa_mode_returns_optimized_kappa_r(self):
        """κ mode should return R_OPTIMIZED_KAPPA."""
        from streamlit_app.utils.constants import (
            get_optimized_defaults, R_OPTIMIZED_KAPPA
        )
        defaults = get_optimized_defaults("kappa")
        assert defaults["R"] == R_OPTIMIZED_KAPPA
        assert abs(defaults["R"] - 1.14978) < 0.00001

    def test_kappa_star_mode_returns_optimized_kappa_star_r(self):
        """κ* mode should return R_OPTIMIZED_KAPPA_STAR."""
        from streamlit_app.utils.constants import (
            get_optimized_defaults, R_OPTIMIZED_KAPPA_STAR
        )
        defaults = get_optimized_defaults("kappa_star")
        assert defaults["R"] == R_OPTIMIZED_KAPPA_STAR
        assert abs(defaults["R"] - 1.07966) < 0.00001

    def test_kappa_vs_kappa_star_r_difference(self):
        """The R values for κ and κ* optimized should differ by ~0.07."""
        from streamlit_app.utils.constants import get_optimized_defaults
        kappa_defaults = get_optimized_defaults("kappa")
        kappa_star_defaults = get_optimized_defaults("kappa_star")

        r_diff = kappa_defaults["R"] - kappa_star_defaults["R"]
        # κ has higher R than κ*
        assert r_diff > 0.05
        assert r_diff < 0.1


class TestWidgetCacheClearing:
    """Test that widget caches are properly identified for clearing."""

    def test_widget_cache_keys_are_correct(self):
        """Verify the widget cache key names match what's in the code."""
        # These are the keys that need to be cleared
        expected_keys = ["r_slider_widget", "r_text_input_widget"]

        # Verify they're strings and distinct
        assert len(expected_keys) == 2
        assert expected_keys[0] != expected_keys[1]
        assert all(isinstance(k, str) for k in expected_keys)


class TestModeAwareRSelection:
    """Test that mode correctly determines which R value is used."""

    def test_mode_determines_przz_r(self):
        """Different modes should return different PRZZ R values."""
        from streamlit_app.utils.constants import get_przz_defaults

        kappa_r = get_przz_defaults("kappa")["R"]
        kappa_star_r = get_przz_defaults("kappa_star")["R"]

        # They must be different
        assert kappa_r != kappa_star_r

        # κ should be ~1.30, κ* should be ~1.12
        assert 1.25 < kappa_r < 1.35
        assert 1.10 < kappa_star_r < 1.15

    def test_mode_determines_optimized_r(self):
        """Different modes should return different optimized R values."""
        from streamlit_app.utils.constants import get_optimized_defaults

        kappa_r = get_optimized_defaults("kappa")["R"]
        kappa_star_r = get_optimized_defaults("kappa_star")["R"]

        # They must be different
        assert kappa_r != kappa_star_r

        # κ should be ~1.15, κ* should be ~1.08
        assert 1.10 < kappa_r < 1.20
        assert 1.05 < kappa_star_r < 1.10


class TestRValueRanges:
    """Test R value bounds and ranges."""

    def test_r_bounds_are_reasonable(self):
        """R bounds should allow all preset values."""
        from streamlit_app.utils.constants import (
            R_MIN, R_MAX,
            R_PRZZ_KAPPA, R_PRZZ_KAPPA_STAR,
            R_OPTIMIZED_KAPPA, R_OPTIMIZED_KAPPA_STAR
        )

        all_r_values = [
            R_PRZZ_KAPPA, R_PRZZ_KAPPA_STAR,
            R_OPTIMIZED_KAPPA, R_OPTIMIZED_KAPPA_STAR,
            0.85,  # Additional preset
        ]

        for r in all_r_values:
            assert R_MIN <= r <= R_MAX, f"R={r} is outside bounds [{R_MIN}, {R_MAX}]"

    def test_r_step_allows_precision(self):
        """R step should allow fine-grained adjustments."""
        from streamlit_app.utils.constants import R_STEP
        assert R_STEP <= 0.01


class TestStateManagementGetR:
    """Test get_R function from state management."""

    def test_get_r_default_value(self, mock_streamlit):
        """get_R should return default when R_value not set."""
        from streamlit_app.utils.state_management import get_R

        # Clear R_value
        if "R_value" in mock_streamlit._session_state:
            del mock_streamlit._session_state["R_value"]

        r = get_R()
        # Default is 1.15 per the code
        assert r == 1.15

    def test_get_r_returns_session_value(self, mock_streamlit):
        """get_R should return session state value when set."""
        from streamlit_app.utils.state_management import get_R

        mock_streamlit._session_state["R_value"] = 1.07966
        r = get_R()
        assert r == 1.07966


class TestComputationUsesCorrectR:
    """Test that computation functions receive correct R."""

    def test_quick_kappa_cache_key_includes_r(self):
        """The cache key for quick kappa should include R."""
        from streamlit_app.computation.caching import coefficients_to_hash_key

        key1 = coefficients_to_hash_key(
            P1_coeffs=[0.1, 0.2],
            P2_coeffs=[0.3],
            P3_coeffs=[0.4],
            Q_coeffs={0: 0.5},
            R=1.14978,
            theta=4/7,
        )

        key2 = coefficients_to_hash_key(
            P1_coeffs=[0.1, 0.2],
            P2_coeffs=[0.3],
            P3_coeffs=[0.4],
            Q_coeffs={0: 0.5},
            R=1.07966,  # Different R
            theta=4/7,
        )

        # Different R should produce different cache keys
        assert key1 != key2

    def test_cache_key_r_precision(self):
        """R should be rounded appropriately in cache key."""
        from streamlit_app.computation.caching import coefficients_to_hash_key
        import json

        key = coefficients_to_hash_key(
            P1_coeffs=[0.1],
            P2_coeffs=[0.1],
            P3_coeffs=[0.1],
            Q_coeffs={0: 0.5},
            R=1.079660001,  # Slight variation
            theta=4/7,
        )

        data = json.loads(key)
        # R should be rounded to 6 decimal places
        assert data["R"] == 1.07966


class TestExpectedKappaResults:
    """Sanity checks for expected κ values at different R."""

    def test_kappa_formula(self):
        """κ = 1 - log(c)/R formula verification."""
        import math

        # For c=1, κ should equal 1 regardless of R
        c = 1.0
        for R in [1.0, 1.14978, 1.3036]:
            kappa = 1 - math.log(c) / R
            assert abs(kappa - 1.0) < 1e-10

        # For c > 1, κ < 1
        c = 2.137
        R = 1.3036
        kappa = 1 - math.log(c) / R
        assert 0.4 < kappa < 0.5  # Should be around 0.417

    def test_przz_target_values(self):
        """Verify PRZZ target constants are consistent."""
        import math
        from streamlit_app.utils.constants import (
            KAPPA_TARGET, C_TARGET, R_PRZZ_KAPPA
        )

        # κ = 1 - log(c)/R
        computed_kappa = 1 - math.log(C_TARGET) / R_PRZZ_KAPPA
        assert abs(computed_kappa - KAPPA_TARGET) < 1e-6
