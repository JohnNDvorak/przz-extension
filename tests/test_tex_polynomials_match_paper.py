"""
tests/test_tex_polynomials_match_paper.py
Phase 7D: TeX Polynomial Regression Anchor

Verify PRZZ TeX polynomial coefficients are correctly transcribed.

PRZZ TeX sources:
- κ polynomials: Lines 2567-2586
- κ* polynomials: Lines 2587-2598

Constraints:
- P₁(0) = 0, P₁(1) = 1
- P₂(0) = P₃(0) = 0
- Q(0) = 1

This is a STRICT REGRESSION GATE. Any transcription errors here cascade.
"""

import numpy as np
import pytest
from pathlib import Path
import json

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
    P1Polynomial,
    PellPolynomial,
    QPolynomial,
)


# =============================================================================
# TeX Reference Values (from PRZZ TeX)
# =============================================================================

# κ benchmark (TeX lines 2567-2586)
TEX_KAPPA_P1_TILDE = [0.261076, -1.071007, -0.236840, 0.260233]
TEX_KAPPA_P2_TILDE = [1.048274, 1.319912, -0.940058]
TEX_KAPPA_P3_TILDE = [0.522811, -0.686510, -0.049923]
TEX_KAPPA_Q_BASIS = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
TEX_KAPPA_Q0_SUM = 0.999999  # As printed in paper (not exactly 1.0)

# κ* benchmark (TeX lines 2587-2598)
TEX_KAPPA_STAR_P1_TILDE = [0.052703, -0.657999, -0.003193, -0.101832]
TEX_KAPPA_STAR_P2_TILDE = [1.049837, -0.097446]
TEX_KAPPA_STAR_P3_TILDE = [0.035113, -0.156465]
TEX_KAPPA_STAR_Q_BASIS = {0: 0.483777, 1: 0.516223}
TEX_KAPPA_STAR_Q0_SUM = 1.0  # Exactly 1.0 for linear Q


# =============================================================================
# Phase 7D Test Suite: κ Benchmark Polynomials
# =============================================================================

class TestKappaPolynomialCoefficients:
    """Verify κ benchmark polynomial coefficients match TeX."""

    @pytest.fixture
    def polys(self):
        """Load κ benchmark polynomials."""
        return load_przz_polynomials(enforce_Q0=False)

    def test_p1_tilde_coefficients_match_tex(self, polys):
        """P₁ tilde coefficients match TeX lines 2567-2586."""
        P1, _, _, _ = polys
        np.testing.assert_allclose(
            P1.tilde_coeffs,
            TEX_KAPPA_P1_TILDE,
            rtol=1e-5,
            err_msg="P₁ tilde coefficients do not match TeX"
        )

    def test_p2_tilde_coefficients_match_tex(self, polys):
        """P₂ tilde coefficients match TeX lines 2567-2586."""
        _, P2, _, _ = polys
        np.testing.assert_allclose(
            P2.tilde_coeffs,
            TEX_KAPPA_P2_TILDE,
            rtol=1e-5,
            err_msg="P₂ tilde coefficients do not match TeX"
        )

    def test_p3_tilde_coefficients_match_tex(self, polys):
        """P₃ tilde coefficients match TeX lines 2567-2586."""
        _, _, P3, _ = polys
        np.testing.assert_allclose(
            P3.tilde_coeffs,
            TEX_KAPPA_P3_TILDE,
            rtol=1e-5,
            err_msg="P₃ tilde coefficients do not match TeX"
        )

    def test_q_basis_coefficients_match_tex(self, polys):
        """Q basis coefficients match TeX lines 2567-2586."""
        _, _, _, Q = polys
        for k, expected in TEX_KAPPA_Q_BASIS.items():
            actual = Q.basis_coeffs.get(k, 0.0)
            assert abs(actual - expected) < 1e-5, \
                f"Q basis coeff for k={k}: expected {expected}, got {actual}"

    def test_q_zero_sum_matches_tex(self, polys):
        """Q(0) = sum of coefficients matches TeX printed value."""
        _, _, _, Q = polys
        q0 = Q.Q_at_zero()
        assert abs(q0 - TEX_KAPPA_Q0_SUM) < 1e-5, \
            f"Q(0) = {q0}, expected {TEX_KAPPA_Q0_SUM}"


class TestKappaPolynomialConstraints:
    """Verify κ benchmark polynomial constraints are satisfied."""

    @pytest.fixture
    def polys(self):
        """Load κ benchmark polynomials."""
        return load_przz_polynomials(enforce_Q0=False)

    def test_p1_at_zero_is_zero(self, polys):
        """P₁(0) = 0 by construction."""
        P1, _, _, _ = polys
        val = P1.eval(np.array([0.0]))[0]
        assert abs(val) < 1e-14, f"P₁(0) = {val}, expected 0"

    def test_p1_at_one_is_one(self, polys):
        """P₁(1) = 1 by construction."""
        P1, _, _, _ = polys
        val = P1.eval(np.array([1.0]))[0]
        assert abs(val - 1.0) < 1e-14, f"P₁(1) = {val}, expected 1"

    def test_p2_at_zero_is_zero(self, polys):
        """P₂(0) = 0 by construction."""
        _, P2, _, _ = polys
        val = P2.eval(np.array([0.0]))[0]
        assert abs(val) < 1e-14, f"P₂(0) = {val}, expected 0"

    def test_p3_at_zero_is_zero(self, polys):
        """P₃(0) = 0 by construction."""
        _, _, P3, _ = polys
        val = P3.eval(np.array([0.0]))[0]
        assert abs(val) < 1e-14, f"P₃(0) = {val}, expected 0"

    def test_q_at_zero_approximately_one(self, polys):
        """Q(0) ≈ 1 (within paper precision)."""
        _, _, _, Q = polys
        val = Q.eval(np.array([0.0]))[0]
        # Paper prints 0.999999, so tolerance is 1e-5
        assert abs(val - 1.0) < 1e-5, f"Q(0) = {val}, expected ~1.0"

    def test_q_at_zero_matches_sum(self, polys):
        """Q(0) via eval equals sum of basis coefficients."""
        _, _, _, Q = polys
        val_eval = Q.eval(np.array([0.0]))[0]
        val_sum = Q.Q_at_zero()
        assert abs(val_eval - val_sum) < 1e-14, \
            f"Q(0) via eval = {val_eval}, via sum = {val_sum}"


# =============================================================================
# Phase 7D Test Suite: κ* Benchmark Polynomials
# =============================================================================

class TestKappaStarPolynomialCoefficients:
    """Verify κ* benchmark polynomial coefficients match TeX."""

    @pytest.fixture
    def polys(self):
        """Load κ* benchmark polynomials."""
        return load_przz_polynomials_kappa_star(enforce_Q0=False)

    def test_p1_tilde_coefficients_match_tex(self, polys):
        """P₁ tilde coefficients match TeX lines 2587-2598."""
        P1, _, _, _ = polys
        np.testing.assert_allclose(
            P1.tilde_coeffs,
            TEX_KAPPA_STAR_P1_TILDE,
            rtol=1e-5,
            err_msg="κ* P₁ tilde coefficients do not match TeX"
        )

    def test_p2_tilde_coefficients_match_tex(self, polys):
        """P₂ tilde coefficients match TeX lines 2587-2598."""
        _, P2, _, _ = polys
        np.testing.assert_allclose(
            P2.tilde_coeffs,
            TEX_KAPPA_STAR_P2_TILDE,
            rtol=1e-5,
            err_msg="κ* P₂ tilde coefficients do not match TeX"
        )

    def test_p3_tilde_coefficients_match_tex(self, polys):
        """P₃ tilde coefficients match TeX lines 2587-2598."""
        _, _, P3, _ = polys
        np.testing.assert_allclose(
            P3.tilde_coeffs,
            TEX_KAPPA_STAR_P3_TILDE,
            rtol=1e-5,
            err_msg="κ* P₃ tilde coefficients do not match TeX"
        )

    def test_q_basis_coefficients_match_tex(self, polys):
        """Q basis coefficients match TeX lines 2587-2598."""
        _, _, _, Q = polys
        for k, expected in TEX_KAPPA_STAR_Q_BASIS.items():
            actual = Q.basis_coeffs.get(k, 0.0)
            assert abs(actual - expected) < 1e-5, \
                f"κ* Q basis coeff for k={k}: expected {expected}, got {actual}"

    def test_q_zero_sum_matches_tex(self, polys):
        """Q(0) = sum of coefficients matches TeX printed value."""
        _, _, _, Q = polys
        q0 = Q.Q_at_zero()
        assert abs(q0 - TEX_KAPPA_STAR_Q0_SUM) < 1e-5, \
            f"κ* Q(0) = {q0}, expected {TEX_KAPPA_STAR_Q0_SUM}"


class TestKappaStarPolynomialConstraints:
    """Verify κ* benchmark polynomial constraints are satisfied."""

    @pytest.fixture
    def polys(self):
        """Load κ* benchmark polynomials."""
        return load_przz_polynomials_kappa_star(enforce_Q0=False)

    def test_p1_at_zero_is_zero(self, polys):
        """P₁(0) = 0 by construction."""
        P1, _, _, _ = polys
        val = P1.eval(np.array([0.0]))[0]
        assert abs(val) < 1e-14, f"κ* P₁(0) = {val}, expected 0"

    def test_p1_at_one_is_one(self, polys):
        """P₁(1) = 1 by construction."""
        P1, _, _, _ = polys
        val = P1.eval(np.array([1.0]))[0]
        assert abs(val - 1.0) < 1e-14, f"κ* P₁(1) = {val}, expected 1"

    def test_p2_at_zero_is_zero(self, polys):
        """P₂(0) = 0 by construction."""
        _, P2, _, _ = polys
        val = P2.eval(np.array([0.0]))[0]
        assert abs(val) < 1e-14, f"κ* P₂(0) = {val}, expected 0"

    def test_p3_at_zero_is_zero(self, polys):
        """P₃(0) = 0 by construction."""
        _, _, P3, _ = polys
        val = P3.eval(np.array([0.0]))[0]
        assert abs(val) < 1e-14, f"κ* P₃(0) = {val}, expected 0"

    def test_q_at_zero_exactly_one(self, polys):
        """Q(0) = 1 exactly for linear Q."""
        _, _, _, Q = polys
        val = Q.eval(np.array([0.0]))[0]
        assert abs(val - 1.0) < 1e-14, f"κ* Q(0) = {val}, expected 1.0"


# =============================================================================
# Polynomial Degree Verification
# =============================================================================

class TestKappaPolynomialDegrees:
    """Verify κ polynomial degrees match TeX specification."""

    @pytest.fixture
    def polys(self):
        return load_przz_polynomials(enforce_Q0=False)

    def test_p1_has_4_tilde_coefficients(self, polys):
        """P₁ has 4 tilde coefficients → degree 5 polynomial."""
        P1, _, _, _ = polys
        assert len(P1.tilde_coeffs) == 4

    def test_p2_has_3_tilde_coefficients(self, polys):
        """P₂ has 3 tilde coefficients → degree 3 polynomial."""
        _, P2, _, _ = polys
        assert len(P2.tilde_coeffs) == 3

    def test_p3_has_3_tilde_coefficients(self, polys):
        """P₃ has 3 tilde coefficients → degree 3 polynomial."""
        _, _, P3, _ = polys
        assert len(P3.tilde_coeffs) == 3

    def test_q_has_degree_5(self, polys):
        """Q has degree 5 (powers 0, 1, 3, 5)."""
        _, _, _, Q = polys
        max_power = max(Q.basis_coeffs.keys())
        assert max_power == 5


class TestKappaStarPolynomialDegrees:
    """Verify κ* polynomial degrees match TeX specification."""

    @pytest.fixture
    def polys(self):
        return load_przz_polynomials_kappa_star(enforce_Q0=False)

    def test_p1_has_4_tilde_coefficients(self, polys):
        """P₁ has 4 tilde coefficients → degree 5 polynomial."""
        P1, _, _, _ = polys
        assert len(P1.tilde_coeffs) == 4

    def test_p2_has_2_tilde_coefficients(self, polys):
        """κ* P₂ has 2 tilde coefficients → degree 2 polynomial."""
        _, P2, _, _ = polys
        assert len(P2.tilde_coeffs) == 2

    def test_p3_has_2_tilde_coefficients(self, polys):
        """κ* P₃ has 2 tilde coefficients → degree 2 polynomial."""
        _, _, P3, _ = polys
        assert len(P3.tilde_coeffs) == 2

    def test_q_is_linear(self, polys):
        """κ* Q is linear (powers 0, 1 only)."""
        _, _, _, Q = polys
        max_power = max(Q.basis_coeffs.keys())
        assert max_power == 1, f"κ* Q should be linear, got max power {max_power}"


# =============================================================================
# JSON File Integrity Tests
# =============================================================================

class TestJsonFileIntegrity:
    """Verify JSON parameter files are correctly formatted."""

    def test_kappa_json_exists(self):
        """κ parameters JSON file exists."""
        path = Path(__file__).parent.parent / "data" / "przz_parameters.json"
        assert path.exists(), f"Missing: {path}"

    def test_kappa_star_json_exists(self):
        """κ* parameters JSON file exists."""
        path = Path(__file__).parent.parent / "data" / "przz_parameters_kappa_star.json"
        assert path.exists(), f"Missing: {path}"

    def test_kappa_json_has_required_fields(self):
        """κ JSON has all required fields."""
        path = Path(__file__).parent.parent / "data" / "przz_parameters.json"
        with open(path) as f:
            data = json.load(f)

        assert "configuration" in data
        assert "polynomials" in data
        assert "targets" in data

        assert "R" in data["configuration"]
        assert "theta" in data["configuration"]

        assert "P1" in data["polynomials"]
        assert "P2" in data["polynomials"]
        assert "P3" in data["polynomials"]
        assert "Q" in data["polynomials"]

    def test_kappa_star_json_has_required_fields(self):
        """κ* JSON has all required fields."""
        path = Path(__file__).parent.parent / "data" / "przz_parameters_kappa_star.json"
        with open(path) as f:
            data = json.load(f)

        assert "configuration" in data
        assert "polynomials" in data
        assert "targets" in data


# =============================================================================
# Cross-Benchmark Consistency
# =============================================================================

class TestCrossBenchmarkConsistency:
    """Verify both benchmarks use consistent theta."""

    def test_theta_matches_four_sevenths(self):
        """Both benchmarks use θ = 4/7."""
        kappa = load_przz_polynomials(enforce_Q0=False)
        kappa_star = load_przz_polynomials_kappa_star(enforce_Q0=False)

        # Get theta from JSON
        path_k = Path(__file__).parent.parent / "data" / "przz_parameters.json"
        path_ks = Path(__file__).parent.parent / "data" / "przz_parameters_kappa_star.json"

        with open(path_k) as f:
            data_k = json.load(f)
        with open(path_ks) as f:
            data_ks = json.load(f)

        theta_k = data_k["configuration"]["theta"]
        theta_ks = data_ks["configuration"]["theta"]

        expected = 4.0 / 7.0
        assert abs(theta_k - expected) < 1e-14, f"κ theta = {theta_k}"
        assert abs(theta_ks - expected) < 1e-14, f"κ* theta = {theta_ks}"

    def test_r_values_are_different(self):
        """κ and κ* use different R values."""
        path_k = Path(__file__).parent.parent / "data" / "przz_parameters.json"
        path_ks = Path(__file__).parent.parent / "data" / "przz_parameters_kappa_star.json"

        with open(path_k) as f:
            data_k = json.load(f)
        with open(path_ks) as f:
            data_ks = json.load(f)

        R_k = data_k["configuration"]["R"]
        R_ks = data_ks["configuration"]["R"]

        assert abs(R_k - 1.3036) < 1e-4, f"κ R = {R_k}"
        assert abs(R_ks - 1.1167) < 1e-4, f"κ* R = {R_ks}"
        assert R_k != R_ks, "R values should differ between benchmarks"
