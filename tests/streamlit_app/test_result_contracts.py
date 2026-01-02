"""
Result contract validation tests.

Tests that visualization functions handle all result dict variants correctly:
- full_result (22 keys) - normal case
- quick_result (4 keys) - may be passed incorrectly
- None - pre-computation state
- error_result - computation failed partially

These tests catch KeyError patterns before deployment.
"""

import pytest
import sys
from pathlib import Path

# Add streamlit_app to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))


class TestResultContractValidation:
    """Test that result dicts have expected keys."""

    def test_quick_result_has_4_keys(self, quick_result, quick_result_keys):
        """Quick result must have exactly 4 keys."""
        assert set(quick_result.keys()) == quick_result_keys

    def test_full_result_has_22_keys(self, full_result, full_result_keys):
        """Full result must have all 22 keys."""
        assert set(full_result.keys()) == full_result_keys

    def test_per_pair_structure(self, full_result, per_pair_keys):
        """Per-pair sub-dicts must have expected keys."""
        per_pair = full_result["per_pair"]
        assert per_pair is not None

        for pair_key, pair_data in per_pair.items():
            if isinstance(pair_data, dict) and "error" not in pair_data:
                assert set(pair_data.keys()) == per_pair_keys, (
                    f"Pair {pair_key} missing keys: {per_pair_keys - set(pair_data.keys())}"
                )


class TestDecompositionWaterfallContracts:
    """Test decomposition_waterfall.py handles all result types."""

    def test_with_full_result(self, full_result, mock_streamlit):
        """Should work with full result."""
        from streamlit_app.visualizations.decomposition_waterfall import (
            create_decomposition_waterfall,
            render_decomposition,
        )

        # Should not raise
        fig = create_decomposition_waterfall(full_result)
        assert fig is not None

    def test_with_none_result(self, none_result, mock_streamlit):
        """Should show info message with None."""
        from streamlit_app.visualizations.decomposition_waterfall import (
            render_decomposition,
        )

        # Should not raise
        render_decomposition(none_result)

        # Should have shown info message
        assert len(mock_streamlit._infos) > 0
        assert "Compute Full Result" in mock_streamlit._infos[0]

    def test_with_quick_result_returns_none(self, quick_result, mock_streamlit):
        """
        FIXED: Should return None when required keys are missing.
        """
        from streamlit_app.visualizations.decomposition_waterfall import (
            create_decomposition_waterfall,
        )

        # Should return None, not raise KeyError
        result = create_decomposition_waterfall(quick_result)
        assert result is None


class TestFullCalculationContracts:
    """Test full_calculation.py handles all result types."""

    def test_display_full_result_with_valid_result(self, full_result, mock_streamlit):
        """Should display full result without error."""
        from streamlit_app.computation.full_calculation import display_full_result

        # Should not raise
        display_full_result(full_result)

    def test_display_full_result_with_none_kappa_rigorous(
        self, full_result_with_none_values, mock_streamlit
    ):
        """Should handle None kappa_rigorous gracefully."""
        from streamlit_app.computation.full_calculation import display_full_result

        # Should not raise
        display_full_result(full_result_with_none_values)

    def test_display_quick_result(self, quick_result, mock_streamlit):
        """Should display quick result metrics."""
        from streamlit_app.computation.full_calculation import display_quick_result

        kappa = quick_result["kappa"]
        c = quick_result["c"]
        R = 1.3036

        # Should not raise
        display_quick_result(kappa, c, R)


class TestIntegralBreakdownContracts:
    """Test create_integral_breakdown handles all result types."""

    def test_with_full_result(self, full_result, mock_streamlit):
        """Should create bar chart with full result."""
        from streamlit_app.visualizations.decomposition_waterfall import (
            create_integral_breakdown,
        )

        fig = create_integral_breakdown(full_result)
        assert fig is not None

    def test_with_quick_result_returns_none(self, quick_result, mock_streamlit):
        """
        FIXED: Should return None when required keys are missing.
        """
        from streamlit_app.visualizations.decomposition_waterfall import (
            create_integral_breakdown,
        )

        # Should return None, not raise KeyError
        result = create_integral_breakdown(quick_result)
        assert result is None


class TestRenderDecompositionCorrectionFactors:
    """Test correction factor display handles missing keys."""

    def test_correction_factors_with_full_result(self, full_result, mock_streamlit):
        """Should display correction factors."""
        from streamlit_app.visualizations.decomposition_waterfall import (
            render_decomposition,
        )

        render_decomposition(full_result)

        # Check metrics were created
        assert len(mock_streamlit._metrics) >= 4

    def test_correction_factors_with_missing_keys_handles_gracefully(self, mock_streamlit):
        """
        FIXED: Should handle missing g_I1, g_I2, etc gracefully.
        """
        from streamlit_app.visualizations.decomposition_waterfall import (
            render_decomposition,
        )

        partial_result = {
            "S12_plus": 1.0,
            "S12_minus": 0.1,
            "S34": 0.5,
            "m": 8.0,
            "c": 2.0,
            "kappa": 0.4,
            "I1_plus": 0.5,
            "I1_minus": 0.05,
            "I2_plus": 0.4,
            "I2_minus": 0.04,
            "I3_plus": 0.2,
            "I4_plus": 0.1,
            # Missing: g_I1, g_I2, g_total, base
        }

        # Should not raise KeyError
        render_decomposition(partial_result)


class TestNoneResultHandling:
    """Test all render functions handle None result gracefully."""

    def test_decomposition_with_none(self, none_result, mock_streamlit):
        """render_decomposition should show info for None."""
        from streamlit_app.visualizations.decomposition_waterfall import (
            render_decomposition,
        )

        render_decomposition(none_result)
        assert len(mock_streamlit._infos) > 0


class TestErrorResultHandling:
    """Test handling of results with error sub-dicts."""

    def test_per_pair_with_error_key(self, error_result, mock_streamlit):
        """per_pair may contain {"error": "message"} instead of data."""
        per_pair = error_result["per_pair"]

        # Check structure
        assert "error" in per_pair

    def test_error_bounds_with_error_key(self, error_result, mock_streamlit):
        """error_bounds may contain {"error": "message"}."""
        eb = error_result["error_bounds"]

        assert "error" in eb


class TestKeyAccessPatterns:
    """
    Direct tests for specific KeyError-prone patterns.

    These tests document the exact lines that have bugs.
    """

    def test_decomposition_waterfall_line_23_28(self, quick_result):
        """
        decomposition_waterfall.py lines 23-28:
            S12_plus = result["S12_plus"]  # KeyError
            S12_minus = result["S12_minus"]
            S34 = result["S34"]
            m = result["m"]
        """
        # These keys don't exist in quick_result
        with pytest.raises(KeyError, match="S12_plus"):
            _ = quick_result["S12_plus"]

    def test_decomposition_waterfall_line_99_107(self, quick_result):
        """
        decomposition_waterfall.py lines 99-107:
            values = [
                result["I1_plus"],  # KeyError
                result["I1_minus"],
                ...
            ]
        """
        with pytest.raises(KeyError, match="I1_plus"):
            _ = quick_result["I1_plus"]

    def test_safe_get_pattern(self, quick_result):
        """Demonstrate safe .get() pattern."""
        # Safe access returns None or default
        S12_plus = quick_result.get("S12_plus", 0.0)
        assert S12_plus == 0.0

        I1_plus = quick_result.get("I1_plus")
        assert I1_plus is None


class TestQuickToFullTransition:
    """Test handling when quick_result is passed where full_result expected."""

    def test_quick_result_missing_decomposition_keys(
        self, quick_result, full_result_keys
    ):
        """Document missing keys when quick_result used."""
        quick_keys = set(quick_result.keys())
        full_keys = full_result_keys

        missing = full_keys - quick_keys
        expected_missing = {
            "R", "theta", "K",
            "S12_plus", "S12_minus", "S34", "m",
            "I1_plus", "I1_minus", "I2_plus", "I2_minus",
            "I3_plus", "I4_plus",
            "g_I1", "g_I2", "g_total", "base",
            "error_bounds", "kappa_rigorous", "per_pair",
        }

        assert missing == expected_missing
