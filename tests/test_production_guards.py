"""
tests/test_production_guards.py
Production Guards - Ensure kappa_engine never regresses to anchored code.

These tests explicitly verify that:
1. kappa_engine uses ONLY first-principles formulas
2. kappa_engine doesn't import calibrated constants
3. The public API is stable
4. No quiet calibration creep

Created: 2025-12-27 (Phase 46++)
"""

import pytest
import inspect
import ast
from pathlib import Path

from src.kappa_engine import (
    KappaEngine,
    KappaResult,
    IntegralComponents,
    CorrectionFactors,
    compute_g_I1,
    compute_g_I2,
    compute_base,
    compute_mirror_multiplier,
    compute_c_from_integrals,
    compute_kappa_from_c,
    compute_przz_kappa,
    validate_przz_benchmarks,
)


class TestNoAnchoredImports:
    """Verify kappa_engine doesn't import calibrated constants."""

    def test_no_correction_policy_import(self):
        """kappa_engine must NOT import from correction_policy."""
        source_path = Path(__file__).parent.parent / "src" / "kappa_engine.py"
        source = source_path.read_text()

        # Parse the source to find imports
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "correction_policy" not in alias.name, \
                        "kappa_engine imports correction_policy"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "correction_policy" not in node.module, \
                        "kappa_engine imports from correction_policy"

    def test_no_g_first_principles_import(self):
        """kappa_engine must NOT import from g_first_principles."""
        source_path = Path(__file__).parent.parent / "src" / "kappa_engine.py"
        source = source_path.read_text()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "g_first_principles" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "g_first_principles" not in node.module

    def test_no_calibrated_constants_in_source(self):
        """kappa_engine must NOT contain calibrated constant values."""
        source_path = Path(__file__).parent.parent / "src" / "kappa_engine.py"
        source = source_path.read_text()

        # The calibrated constants
        calibrated_values = [
            "1.00091428",  # G_I1_CALIBRATED
            "1.01945154",  # G_I2_CALIBRATED
        ]

        for val in calibrated_values:
            # Allow in comments but not as actual values
            lines_with_val = [
                line for line in source.split('\n')
                if val in line and not line.strip().startswith('#')
            ]
            assert len(lines_with_val) == 0, \
                f"Found calibrated value {val} in non-comment line"


class TestPublicAPIStability:
    """Verify public API is stable."""

    def test_compute_g_I1_signature(self):
        """compute_g_I1 must accept theta and K."""
        sig = inspect.signature(compute_g_I1)
        params = list(sig.parameters.keys())
        assert "theta" in params
        assert "K" in params

    def test_compute_g_I2_signature(self):
        """compute_g_I2 must accept theta and K."""
        sig = inspect.signature(compute_g_I2)
        params = list(sig.parameters.keys())
        assert "theta" in params
        assert "K" in params

    def test_compute_base_signature(self):
        """compute_base must accept R and K."""
        sig = inspect.signature(compute_base)
        params = list(sig.parameters.keys())
        assert "R" in params
        assert "K" in params

    def test_kappa_engine_has_from_przz_kappa(self):
        """KappaEngine must have from_przz_kappa factory."""
        assert hasattr(KappaEngine, 'from_przz_kappa')
        assert callable(KappaEngine.from_przz_kappa)

    def test_kappa_engine_has_from_przz_kappa_star(self):
        """KappaEngine must have from_przz_kappa_star factory."""
        assert hasattr(KappaEngine, 'from_przz_kappa_star')
        assert callable(KappaEngine.from_przz_kappa_star)

    def test_kappa_engine_has_compute_kappa(self):
        """KappaEngine must have compute_kappa method."""
        assert hasattr(KappaEngine, 'compute_kappa')

    def test_kappa_result_has_required_fields(self):
        """KappaResult must have kappa, c, integrals, corrections."""
        result_fields = {f.name for f in KappaResult.__dataclass_fields__.values()}
        assert "kappa" in result_fields
        assert "c" in result_fields
        assert "integrals" in result_fields
        assert "corrections" in result_fields


class TestFirstPrinciplesFormulas:
    """Verify formulas are first-principles (not calibrated)."""

    def test_g_I1_matches_closed_form(self):
        """g_I1 must match: 1 + θ(1-θ)(2(K-1)+θ) / (8K(2K+1)²)"""
        theta = 4 / 7
        K = 3

        g = compute_g_I1(theta, K)

        # Closed-form calculation
        numerator = theta * (1 - theta) * (2 * (K - 1) + theta)
        denominator = 8 * K * (2 * K + 1) ** 2
        expected = 1 + numerator / denominator

        assert g == pytest.approx(expected, rel=1e-14), \
            f"g_I1 doesn't match closed-form: {g} vs {expected}"

    def test_g_I2_matches_closed_form(self):
        """g_I2 must match: 1 + θ(2-θ) / (2K(2K+1))"""
        theta = 4 / 7
        K = 3

        g = compute_g_I2(theta, K)

        # Closed-form calculation
        expected = 1 + theta * (2 - theta) / (2 * K * (2 * K + 1))

        assert g == pytest.approx(expected, rel=1e-14), \
            f"g_I2 doesn't match closed-form: {g} vs {expected}"

    def test_base_matches_closed_form(self):
        """base must match: exp(R) + (2K-1)"""
        import math
        R = 1.3036
        K = 3

        base = compute_base(R, K)
        expected = math.exp(R) + (2 * K - 1)

        assert base == pytest.approx(expected, rel=1e-14), \
            f"base doesn't match closed-form: {base} vs {expected}"

    def test_formulas_are_not_calibrated_constants(self):
        """Computed values must NOT equal the calibrated constants."""
        theta = 4 / 7
        K = 3

        # The calibrated constants from correction_policy
        G_I1_CALIBRATED = 1.00091428
        G_I2_CALIBRATED = 1.01945154

        g_I1 = compute_g_I1(theta, K)
        g_I2 = compute_g_I2(theta, K)

        # They should be close (within 0.01%) but not identical
        assert abs(g_I1 - G_I1_CALIBRATED) > 1e-6, \
            "g_I1 equals calibrated constant - possible regression!"
        assert abs(g_I2 - G_I2_CALIBRATED) > 1e-6, \
            "g_I2 equals calibrated constant - possible regression!"


class TestConvenienceFunctions:
    """Test convenience functions work correctly."""

    def test_compute_przz_kappa_returns_result(self):
        """compute_przz_kappa must return KappaResult."""
        result = compute_przz_kappa(n_quad=40)
        assert isinstance(result, KappaResult)
        assert 0 < result.kappa < 1

    def test_validate_przz_benchmarks_returns_dict(self):
        """validate_przz_benchmarks must return validation dict."""
        validation = validate_przz_benchmarks(tolerance_pct=1.0, n_quad=40)

        assert isinstance(validation, dict)
        assert "kappa" in validation
        assert "kappa_star" in validation
        assert "passed" in validation["kappa"]
        assert "passed" in validation["kappa_star"]


class TestDocumentation:
    """Verify documentation is present and correct."""

    def test_module_has_docstring(self):
        """kappa_engine must have module docstring."""
        import src.kappa_engine as module
        assert module.__doc__ is not None
        assert "production" in module.__doc__.lower()
        assert "first-principles" in module.__doc__.lower()

    def test_kappa_engine_has_docstring(self):
        """KappaEngine class must have docstring."""
        assert KappaEngine.__doc__ is not None

    def test_compute_kappa_has_docstring(self):
        """compute_kappa must have docstring."""
        assert KappaEngine.compute_kappa.__doc__ is not None
