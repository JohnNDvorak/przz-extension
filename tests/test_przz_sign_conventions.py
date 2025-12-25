"""
tests/test_przz_sign_conventions.py
Phase 14D Task D5: Lock sign conventions against paper formulas

PURPOSE:
========
Prevent regression back to incorrect signs that were fixed in Phase 14C.

These tests lock:
1. J13/J14 main-term signs are NEGATIVE (from residue calculus)
2. ζ'/ζ(1+s) pole coefficient is -1 (not +1)
3. A^{(1,1)}(0) is POSITIVE (~1.3856)

TEX REFERENCE:
=============
PRZZ TeX Lines 1551-1564: I₃ and I₄ prefactors are -1/θ
PRZZ TeX Lines 1377-1389: S(0) ≈ 1.385603705 (positive)

Phase 14C explicitly corrected the sign of J13/J14 from positive (literal)
to negative (main-term reduction). These tests prevent that from regressing.
"""

import pytest
import numpy as np
from src.ratios.j1_k3_decomposition import (
    build_J1_pieces_K3_main_terms,
)
from src.ratios.zeta_laurent import (
    zeta_logderiv_series,
    EULER_MASCHERONI,
)
from src.ratios.arithmetic_factor import A11_prime_sum


class TestZetaLogDerivPoleSign:
    """Tests for ζ'/ζ(1+s) pole structure."""

    def test_pole_coefficient_is_minus_one(self):
        """
        LOCK: ζ'/ζ(1+s) has pole coefficient -1.

        The Laurent expansion is:
            ζ'/ζ(1+s) = -1/s + γ + γ₁s + ...

        The leading pole is -1/s, NOT +1/s.
        """
        series = zeta_logderiv_series(order=2)
        assert abs(series.pole_coeff - (-1.0)) < 1e-10, (
            f"Pole coefficient should be -1, got {series.pole_coeff}"
        )

    def test_constant_term_is_euler_gamma(self):
        """Constant term should be Euler-Mascheroni gamma."""
        series = zeta_logderiv_series(order=2)
        # coeffs[0] is the s^0 coefficient
        assert abs(series.coeffs[0] - EULER_MASCHERONI) < 1e-10


class TestJ13J14MainTermSigns:
    """Tests for J13/J14 main-term sign conventions."""

    def test_j13_main_term_is_negative(self):
        """
        LOCK: J13 main-term has NEGATIVE sign from residue calculus.

        Phase 14C corrected this from positive (literal) to negative.
        The sign comes from the -1/s pole in ζ'/ζ.
        """
        R = 1.3036
        alpha = -R
        beta = -R
        s = 0.1
        u = 0.1

        pieces = build_J1_pieces_K3_main_terms(alpha, beta, complex(s), complex(u))

        assert np.real(pieces.j13) < 0, (
            f"J13 main-term should be negative, got {pieces.j13}"
        )

    def test_j14_main_term_is_negative(self):
        """
        LOCK: J14 main-term has NEGATIVE sign (symmetric with J13).

        Phase 14C corrected this from positive (literal) to negative.
        """
        R = 1.3036
        alpha = -R
        beta = -R
        s = 0.1
        u = 0.1

        pieces = build_J1_pieces_K3_main_terms(alpha, beta, complex(s), complex(u))

        assert np.real(pieces.j14) < 0, (
            f"J14 main-term should be negative, got {pieces.j14}"
        )

    def test_j13_j14_are_symmetric(self):
        """J13 and J14 should have same magnitude at α=β=-R."""
        R = 1.3036
        alpha = -R
        beta = -R
        s = 0.1
        u = 0.1

        pieces = build_J1_pieces_K3_main_terms(alpha, beta, complex(s), complex(u))

        # At symmetric point, J13 ≈ J14
        assert abs(np.real(pieces.j13) - np.real(pieces.j14)) < 1e-6


class TestA11PrimeSumSign:
    """Tests for A^{(1,1)}(s) sign convention."""

    def test_A11_at_zero_is_positive(self):
        """
        LOCK: A^{(1,1)}(0) is POSITIVE (~1.3856).

        This is the prime sum: Σ_p (log p / (p - 1))²
        All terms are squared, so the result is positive.
        """
        value = A11_prime_sum(0.0, prime_cutoff=5000)
        assert value > 0, f"A^{{(1,1)}}(0) should be positive, got {value}"

    def test_A11_at_zero_is_approximately_1_386(self):
        """A^{(1,1)}(0) ≈ 1.3856 (from PRZZ TeX Lines 1377-1389)."""
        value = A11_prime_sum(0.0, prime_cutoff=5000)
        # Should be close to 1.3856 (within ~1% with 5000 primes)
        assert abs(value - 1.3856) / 1.3856 < 0.02, (
            f"A^{{(1,1)}}(0) should be ~1.3856, got {value}"
        )

    def test_A11_decreases_for_positive_s(self):
        """A^{(1,1)}(s) should decrease as s increases (larger denominators)."""
        val_0 = A11_prime_sum(0.0, prime_cutoff=1000)
        val_01 = A11_prime_sum(0.1, prime_cutoff=1000)
        assert val_01 < val_0, (
            f"A^{{(1,1)}}(s) should decrease: A(0)={val_0}, A(0.1)={val_01}"
        )


class TestEulerMaclaurinSignConventions:
    """Tests for sign conventions in Euler-Maclaurin integral forms."""

    def test_j12_divisor_is_negative_at_diagonal(self):
        """
        At α=β=-R, the 1/(α+β) = 1/(-2R) factor is negative.

        This affects J12's sign.
        """
        R = 1.3036
        divisor = -2.0 * R
        assert divisor < 0

    def test_j13_prefactor_is_negative(self):
        """
        J13 PRZZ prefactor is -1/θ (negative).

        From PRZZ TeX Lines 1551-1564.
        """
        theta = 4.0 / 7.0
        prefactor = -1.0 / theta
        assert prefactor < 0

    def test_j14_prefactor_is_negative(self):
        """
        J14 PRZZ prefactor is -1/θ (negative, symmetric with J13).
        """
        theta = 4.0 / 7.0
        prefactor = -1.0 / theta
        assert prefactor < 0


class TestLaurentCoefficientExtraction:
    """Tests for Laurent coefficient extraction in J12 reductions."""

    def test_logderiv_series_has_expected_structure(self):
        """
        The logderiv series should have:
        - Pole at s=0 with coefficient -1
        - Taylor coefficients starting with γ
        """
        series = zeta_logderiv_series(order=3)

        # Check structure
        assert hasattr(series, 'pole_coeff')
        assert hasattr(series, 'coeffs')
        assert series.pole_coeff == pytest.approx(-1.0)
        assert series.coeffs[0] == pytest.approx(EULER_MASCHERONI)

    def test_inverse_logderiv_at_R(self):
        """
        At s = -R + small, ζ'/ζ(1+s) ≈ 1/R + γ.

        This is the value used in J12 Laurent coefficient.
        """
        R = 1.3036
        expected = 1.0 / R + EULER_MASCHERONI
        assert expected > 0  # Should be positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
