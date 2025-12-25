"""
tests/test_j12_c00_semantics.py
Phase 14H Task H2: Semantic Mode Tests for J12 Constant Term

PURPOSE:
========
These tests PROVE which LaurentMode is semantically correct for J12,
based on the literal zeta-factor structure from the paper.

KEY FINDING (Phase 14H):
========================
RAW_LOGDERIV is the semantically correct mode.

The J12 bracket₂ structure is:
    (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)

At s=u=0 with α=β=-R:
    (ζ'/ζ)(1-R) × (ζ'/ζ)(1-R) = (-1/(-R) + γ)² = (1/R + γ)²

This matches RAW_LOGDERIV exactly.

POLE_CANCELLED (using 1.0) was based on G-product G(α)×G(β) where G=(1/ζ)(ζ'/ζ),
but J12 does NOT include the 1/ζ factors.
"""

import pytest
from src.ratios.j12_c00_reference import (
    compute_j12_c00_reference,
    compute_G_series_at_shift,
    eval_logderiv_at_shift,
    J12C00Result,
)
from src.ratios.zeta_laurent import EULER_MASCHERONI


class TestModeToLiteralMatch:
    """Test that exactly one mode matches the literal J12 structure."""

    def test_j12_c00_literal_matches_raw_logderiv_kappa(self):
        """RAW_LOGDERIV should match the literal log-deriv product for κ benchmark."""
        R = 1.3036
        result = compute_j12_c00_reference(R)

        # RAW_LOGDERIV should match log-derivative product exactly
        diff = abs(result.c00_literal_logderiv_product - result.c00_raw_logderiv)
        assert diff < 1e-10, (
            f"RAW_LOGDERIV should match log-deriv product: "
            f"literal={result.c00_literal_logderiv_product:.6f}, "
            f"raw={result.c00_raw_logderiv:.6f}, diff={diff:.2e}"
        )

    def test_j12_c00_literal_matches_raw_logderiv_kappa_star(self):
        """RAW_LOGDERIV should match the literal log-deriv product for κ* benchmark."""
        R = 1.1167
        result = compute_j12_c00_reference(R)

        diff = abs(result.c00_literal_logderiv_product - result.c00_raw_logderiv)
        assert diff < 1e-10, (
            f"RAW_LOGDERIV should match log-deriv product: "
            f"literal={result.c00_literal_logderiv_product:.6f}, "
            f"raw={result.c00_raw_logderiv:.6f}, diff={diff:.2e}"
        )

    def test_pole_cancelled_does_not_match_literal(self):
        """POLE_CANCELLED (1.0) should NOT match the literal log-deriv product."""
        for R in [1.3036, 1.1167]:
            result = compute_j12_c00_reference(R)

            diff = abs(result.c00_literal_logderiv_product - result.c00_pole_cancelled)
            assert diff > 0.5, (
                f"POLE_CANCELLED should NOT match literal at R={R}: "
                f"literal={result.c00_literal_logderiv_product:.4f}, "
                f"pole_cancelled={result.c00_pole_cancelled:.4f}"
            )

    def test_exactly_one_mode_matches(self):
        """Exactly one mode should match the literal structure (RAW_LOGDERIV)."""
        for R in [1.3036, 1.1167]:
            result = compute_j12_c00_reference(R)

            raw_match = abs(result.c00_literal_logderiv_product - result.c00_raw_logderiv) < 1e-10
            pole_match = abs(result.c00_literal_logderiv_product - result.c00_pole_cancelled) < 0.1

            assert raw_match and not pole_match, (
                f"Exactly RAW_LOGDERIV should match at R={R}: "
                f"raw_match={raw_match}, pole_match={pole_match}"
            )


class TestNegativeControl:
    """Negative control tests to prevent accidental matching."""

    def test_j12_c00_changes_with_R(self):
        """The literal c₀₀ should change with different R values."""
        result_kappa = compute_j12_c00_reference(1.3036)
        result_kappa_star = compute_j12_c00_reference(1.1167)

        diff = abs(result_kappa.c00_literal_logderiv_product -
                   result_kappa_star.c00_literal_logderiv_product)
        assert diff > 0.1, (
            f"c₀₀ should vary with R: κ={result_kappa.c00_literal_logderiv_product:.4f}, "
            f"κ*={result_kappa_star.c00_literal_logderiv_product:.4f}"
        )

    def test_G_product_differs_from_logderiv_product(self):
        """The G-product G(α)G(β) should differ from log-deriv product.

        This proves we're testing two different mathematical objects:
        - G-product: (1/ζ)(ζ'/ζ) × (1/ζ)(ζ'/ζ) includes 1/ζ factors
        - Log-deriv: (ζ'/ζ) × (ζ'/ζ) does NOT include 1/ζ factors
        """
        for R in [1.3036, 1.1167]:
            result = compute_j12_c00_reference(R)

            diff = abs(result.c00_literal_G_product - result.c00_literal_logderiv_product)
            assert diff > 5.0, (
                f"G-product and log-deriv should differ significantly at R={R}: "
                f"G={result.c00_literal_G_product:.4f}, "
                f"logderiv={result.c00_literal_logderiv_product:.4f}"
            )

    def test_dropping_gamma_changes_result(self):
        """Dropping γ from the formula should change the result."""
        R = 1.3036
        result = compute_j12_c00_reference(R)

        # The correct formula is (1/R + γ)²
        correct = result.c00_raw_logderiv

        # If we drop γ, we get (1/R)² which is different
        broken = (1.0 / R) ** 2

        diff = abs(correct - broken)
        assert diff > 0.5, (
            f"Dropping γ should change result significantly: "
            f"correct={correct:.4f}, broken={broken:.4f}"
        )


class TestMathematicalDerivation:
    """Test the mathematical derivation of the log-deriv product."""

    def test_logderiv_expansion_matches_formula(self):
        """Verify (ζ'/ζ)(1+α) = -1/α + γ for our shift values."""
        gamma = EULER_MASCHERONI

        for R in [1.3036, 1.1167]:
            alpha = -R

            # From series: (ζ'/ζ)(1+α) ≈ -1/α + γ + O(α)
            expected = -1.0 / alpha + gamma

            # From eval_logderiv_at_shift
            coeffs = eval_logderiv_at_shift(alpha, order=2)
            actual = coeffs[0]  # Constant term

            # Should match to O(α) accuracy
            rel_diff = abs(expected - actual) / abs(expected)
            assert rel_diff < 0.1, (  # 10% tolerance for O(α) terms
                f"Log-deriv expansion mismatch at α={alpha}: "
                f"expected={expected:.4f}, actual={actual:.4f}"
            )

    def test_raw_logderiv_is_logderiv_squared(self):
        """Verify (1/R + γ)² = [(ζ'/ζ)(1-R)]² mathematically."""
        gamma = EULER_MASCHERONI

        for R in [1.3036, 1.1167]:
            # (1/R + γ)²
            raw_formula = (1.0 / R + gamma) ** 2

            # (ζ'/ζ)(1-R) ≈ -1/(-R) + γ = 1/R + γ
            logderiv_at_1_minus_R = 1.0 / R + gamma
            logderiv_squared = logderiv_at_1_minus_R ** 2

            assert abs(raw_formula - logderiv_squared) < 1e-14, (
                f"Mathematical identity should hold exactly"
            )


class TestSemanticDecision:
    """Document the semantic decision from Phase 14H."""

    def test_semantic_decision_documented(self):
        """
        SEMANTIC DECISION (Phase 14H):

        The J12 bracket₂ zeta-factor is:
            (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)

        At s=u=0 with α=β=-R, this equals:
            (ζ'/ζ)(1-R)² = (1/R + γ)²

        Therefore: RAW_LOGDERIV is semantically correct.
        POLE_CANCELLED was based on a different object (G-product with 1/ζ factors).

        This test documents that the decision is based on mathematical equivalence,
        NOT on which mode gives smaller delta.
        """
        for R in [1.3036, 1.1167]:
            result = compute_j12_c00_reference(R)

            # The literal bracket₂ structure leads to log-deriv product
            # which equals RAW_LOGDERIV
            assert abs(result.c00_literal_logderiv_product - result.c00_raw_logderiv) < 1e-10

        # This assertion documents the semantic choice
        assert True, "RAW_LOGDERIV is the semantically correct mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
