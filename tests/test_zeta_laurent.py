"""
tests/test_zeta_laurent.py
Phase 14C Task C2: Tests for Laurent series machinery.

Tests verify:
1. Pole structure: ζ'/ζ ~ -1/s
2. Constant term: γ_E ≈ 0.5772
3. Cross-check with prime sum at small positive s
4. Product series for J12-J14 reductions
"""

import pytest
import numpy as np
from src.ratios.zeta_laurent import (
    EULER_MASCHERONI,
    STIELTJES_GAMMA1,
    LaurentSeries,
    zeta_series,
    inv_zeta_series,
    zeta_logderiv_series,
    inv_zeta_times_logderiv_series,
    logderiv_product_series,
    j12_main_term_coefficient,
    j13_main_term_sign,
    j14_main_term_sign,
)
from src.ratios.zeta_logderiv import (
    zeta_log_deriv,
    zeta_log_deriv_1_plus_eps,
)


class TestConstants:
    """Test that mathematical constants are correct."""

    def test_euler_mascheroni_value(self):
        """γ ≈ 0.5772156649..."""
        assert abs(EULER_MASCHERONI - 0.5772156649) < 1e-9

    def test_stieltjes_gamma1_is_negative(self):
        """γ₁ ≈ -0.0728... is negative."""
        assert STIELTJES_GAMMA1 < 0
        assert abs(STIELTJES_GAMMA1 + 0.0728158454836767) < 1e-10


class TestZetaSeries:
    """Test the ζ(1+s) Laurent series."""

    def test_zeta_has_pole_coeff_1(self):
        """ζ(1+s) has pole coefficient 1."""
        series = zeta_series(order=3)
        assert abs(series.pole_coeff - 1.0) < 1e-14

    def test_zeta_constant_term_is_gamma(self):
        """Constant term of ζ(1+s) is γ."""
        series = zeta_series(order=3)
        assert abs(series.coeffs[0] - EULER_MASCHERONI) < 1e-14


class TestInvZetaSeries:
    """Test the 1/ζ(1+s) Taylor series."""

    def test_inv_zeta_vanishes_at_zero(self):
        """1/ζ(1+s) vanishes at s=0."""
        coeffs = inv_zeta_series(order=4)
        assert abs(coeffs[0]) < 1e-14, "c₀ should be 0"

    def test_inv_zeta_linear_coeff_is_1(self):
        """[s^1] of 1/ζ(1+s) is 1."""
        coeffs = inv_zeta_series(order=4)
        assert abs(coeffs[1] - 1.0) < 1e-14, "c₁ should be 1"

    def test_inv_zeta_quadratic_coeff_is_minus_gamma(self):
        """[s^2] of 1/ζ(1+s) is -γ."""
        coeffs = inv_zeta_series(order=4)
        assert abs(coeffs[2] + EULER_MASCHERONI) < 1e-14, "c₂ should be -γ"


class TestZetaLogderivSeries:
    """Test the ζ'/ζ(1+s) Laurent series."""

    def test_logderiv_pole_is_minus_1(self):
        """
        ζ'/ζ(1+s) has pole coefficient -1.

        This is CRITICAL for sign conventions in J12-J14.
        """
        series = zeta_logderiv_series(order=3)
        assert abs(series.pole_coeff + 1.0) < 1e-14, "Pole should be -1"

    def test_logderiv_constant_term_is_gamma(self):
        """Constant term of ζ'/ζ(1+s) is γ."""
        series = zeta_logderiv_series(order=3)
        assert abs(series.coeffs[0] - EULER_MASCHERONI) < 1e-14

    def test_logderiv_matches_evaluator_at_small_s(self):
        """
        Cross-check: Laurent series should match numerical evaluator.

        At s=0.1: ζ'/ζ(1.1) from series vs from evaluator.
        """
        series = zeta_logderiv_series(order=4)
        s = 0.1

        # From Laurent: -1/s + γ + γ₁s + ...
        laurent_value = series.eval_at(s)

        # From numerical evaluator
        numerical_value = zeta_log_deriv_1_plus_eps(s, order=4)

        # Should match within 1%
        rel_error = abs(laurent_value - numerical_value) / abs(numerical_value)
        assert rel_error < 0.01, (
            f"Laurent={laurent_value}, Numerical={numerical_value}, "
            f"rel_error={rel_error:.4f}"
        )


class TestInvZetaTimesLogderiv:
    """Test the (1/ζ)(ζ'/ζ)(1+s) series."""

    def test_product_constant_term_is_minus_1(self):
        """
        [s^0] of (1/ζ)(ζ'/ζ)(1+s) is -1.

        This is key for understanding the main-term structure.
        """
        coeffs = inv_zeta_times_logderiv_series(order=3)
        assert abs(coeffs[0] + 1.0) < 1e-14, "c₀ should be -1"

    def test_product_linear_term_is_2gamma(self):
        """[s^1] of (1/ζ)(ζ'/ζ)(1+s) is 2γ."""
        coeffs = inv_zeta_times_logderiv_series(order=3)
        assert abs(coeffs[1] - 2 * EULER_MASCHERONI) < 1e-10, (
            f"c₁ should be 2γ ≈ {2*EULER_MASCHERONI}, got {coeffs[1]}"
        )


class TestLogderivProductSeries:
    """Test the product series for J12."""

    def test_product_at_przz_point(self):
        """
        At α=β=-R, the product should be well-defined.

        (ζ'/ζ)(1-R+s) × (ζ'/ζ)(1-R+u) evaluated at s=u=0.
        """
        R = 1.3036
        alpha = -R
        beta = -R

        s_coeffs, u_coeffs = logderiv_product_series(alpha, beta, order=3)

        # At s=u=0, we want [s^0][u^0]
        # This should be finite (no pole at s=u=0 since α,β are away from 0)
        c00 = s_coeffs[0] * u_coeffs[0]
        assert np.isfinite(c00), f"[s^0 u^0] should be finite, got {c00}"

        # The product at s=u=0 should be approximately:
        # (ζ'/ζ(1-R))² = (-1/(-R) + γ)² = (1/R + γ)²
        expected = (1.0 / R + EULER_MASCHERONI) ** 2
        assert abs(c00 - expected) / abs(expected) < 0.05, (
            f"c00={c00}, expected≈{expected}"
        )


class TestJ13J14MainTermSigns:
    """Test that J13/J14 have correct main-term signs."""

    def test_j13_has_negative_sign(self):
        """
        J13 main-term has leading MINUS sign.

        PRZZ TeX Lines 1551-1564: I₃ prefactor is -1/θ.
        """
        sign = j13_main_term_sign()
        assert sign == -1, "J13 main term should have negative sign"

    def test_j14_has_negative_sign(self):
        """
        J14 main-term has leading MINUS sign.

        Symmetric with J13.
        """
        sign = j14_main_term_sign()
        assert sign == -1, "J14 main term should have negative sign"


class TestLaurentSeriesClass:
    """Test the LaurentSeries helper class."""

    def test_repr_shows_pole(self):
        """String representation should show pole term."""
        series = LaurentSeries(pole_coeff=1.0, coeffs=(0.5,))
        rep = repr(series)
        assert "/s" in rep, "Should show pole term"

    def test_taylor_coeff_negative_index(self):
        """taylor_coeff(-1) returns pole coefficient."""
        series = LaurentSeries(pole_coeff=2.5, coeffs=(1.0, 0.5))
        assert abs(series.taylor_coeff(-1) - 2.5) < 1e-14

    def test_taylor_coeff_beyond_order(self):
        """taylor_coeff beyond order returns 0."""
        series = LaurentSeries(pole_coeff=1.0, coeffs=(0.5,))
        assert abs(series.taylor_coeff(5)) < 1e-14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
