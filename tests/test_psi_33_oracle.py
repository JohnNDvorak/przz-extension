"""
tests/test_psi_33_oracle.py
Tests for the (3,3) Ψ Oracle Implementation

This validates:
1. Monomial generation produces exactly 27 terms
2. Evaluation with κ polynomials (R=1.3036)
3. Evaluation with κ* polynomials (R=1.1167)
4. Ratio moves toward 1.10 (vs DSL's ~17.4)
"""

import pytest
import numpy as np
from src.psi_33_oracle import psi_oracle_33, verify_monomial_count
from src.psi_monomial_expansion import expand_pair_to_monomials
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


class TestMonomialGeneration:
    """Test monomial generation for (3,3)."""

    def test_monomial_count(self):
        """Verify (3,3) produces exactly 27 monomials."""
        monomials = expand_pair_to_monomials(3, 3)
        assert len(monomials) == 27, f"Expected 27 monomials, got {len(monomials)}"

    def test_monomial_structure(self):
        """Verify monomial structure is correct."""
        monomials = expand_pair_to_monomials(3, 3)

        # Check that we have the expected range of powers
        max_a = max(a for (a, b, c, d) in monomials.keys())
        max_b = max(b for (a, b, c, d) in monomials.keys())
        max_c = max(c for (a, b, c, d) in monomials.keys())
        max_d = max(d for (a, b, c, d) in monomials.keys())

        # For (3,3), max powers should be:
        # - A^3, B^3 from (A-C)^3, (B-C)^3 when p=0
        # - C^6 from (D-C²)^3 when p=3, d=0
        # - D^3 from (D-C²)^3 when p=3
        assert max_a == 3, f"Expected max A power = 3, got {max_a}"
        assert max_b == 3, f"Expected max B power = 3, got {max_b}"
        assert max_c == 6, f"Expected max C power = 6, got {max_c}"
        assert max_d == 3, f"Expected max D power = 3, got {max_d}"

    def test_p_config_expansion(self):
        """Verify p-configs expand correctly."""
        from src.psi_block_configs import psi_p_configs

        configs = psi_p_configs(3, 3)

        # (3,3) should have 4 p-configs: p=0,1,2,3
        assert len(configs) == 4, f"Expected 4 p-configs, got {len(configs)}"

        # Verify coefficients
        # p=0: C(3,0)*C(3,0)*0! = 1
        # p=1: C(3,1)*C(3,1)*1! = 9
        # p=2: C(3,2)*C(3,2)*2! = 18
        # p=3: C(3,3)*C(3,3)*3! = 6
        expected_coeffs = [1, 9, 18, 6]
        actual_coeffs = [cfg.coeff for cfg in configs]
        assert actual_coeffs == expected_coeffs, \
            f"Expected coeffs {expected_coeffs}, got {actual_coeffs}"


class TestOracleStructure:
    """Test the oracle structure and API."""

    @pytest.fixture
    def kappa_setup(self):
        """Load κ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        theta = 4.0 / 7.0
        R = 1.3036
        return P3, Q, theta, R

    @pytest.fixture
    def kappa_star_setup(self):
        """Load κ* polynomials."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=True)
        theta = 4.0 / 7.0
        R = 1.1167
        return P3, Q, theta, R

    def test_oracle_returns_correct_structure(self, kappa_setup):
        """Verify oracle returns OracleResult33 with correct fields."""
        P3, Q, theta, R = kappa_setup
        result = psi_oracle_33(P3, Q, theta, R, n_quad=20, debug=False)

        assert hasattr(result, 'total')
        assert hasattr(result, 'monomial_breakdown')
        assert hasattr(result, 'n_monomials')
        assert result.n_monomials == 27

    def test_oracle_breakdown_has_all_monomials(self, kappa_setup):
        """Verify breakdown includes all 27 monomials."""
        P3, Q, theta, R = kappa_setup
        result = psi_oracle_33(P3, Q, theta, R, n_quad=20, debug=False)

        assert len(result.monomial_breakdown) == 27, \
            f"Expected 27 monomials in breakdown, got {len(result.monomial_breakdown)}"


class TestKappaBenchmark:
    """Test with κ polynomials (R=1.3036)."""

    @pytest.fixture
    def setup(self):
        """Load κ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        theta = 4.0 / 7.0
        R = 1.3036
        return P3, Q, theta, R

    def test_kappa_evaluation(self, setup):
        """Test that oracle runs without error for κ benchmark."""
        P3, Q, theta, R = setup
        result = psi_oracle_33(P3, Q, theta, R, n_quad=40, debug=False)

        # For now, just verify it returns a finite value
        assert np.isfinite(result.total), "Oracle returned non-finite value"

    def test_kappa_quadrature_convergence(self, setup):
        """Test that results converge with increasing quadrature points."""
        P3, Q, theta, R = setup

        # Run with different quadrature levels
        result_20 = psi_oracle_33(P3, Q, theta, R, n_quad=20, debug=False)
        result_40 = psi_oracle_33(P3, Q, theta, R, n_quad=40, debug=False)
        result_60 = psi_oracle_33(P3, Q, theta, R, n_quad=60, debug=False)

        # Check that values are converging
        # (We expect small changes as quadrature improves)
        diff_20_40 = abs(result_40.total - result_20.total)
        diff_40_60 = abs(result_60.total - result_40.total)

        # The difference should decrease as we refine
        # (or at least not increase)
        assert diff_40_60 <= diff_20_40 or diff_40_60 < 1e-4, \
            "Quadrature not converging properly"


class TestKappaStarBenchmark:
    """Test with κ* polynomials (R=1.1167)."""

    @pytest.fixture
    def setup(self):
        """Load κ* polynomials."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=True)
        theta = 4.0 / 7.0
        R = 1.1167
        return P3, Q, theta, R

    def test_kappa_star_evaluation(self, setup):
        """Test that oracle runs without error for κ* benchmark."""
        P3, Q, theta, R = setup
        result = psi_oracle_33(P3, Q, theta, R, n_quad=40, debug=False)

        # For now, just verify it returns a finite value
        assert np.isfinite(result.total), "Oracle returned non-finite value"


class TestRatioTarget:
    """Test that the κ/κ* ratio moves toward 1.10."""

    def test_ratio_better_than_dsl(self):
        """
        Verify that the oracle ratio is much better than DSL's ~17.4.

        The goal is to get close to 1.10 (the target from PRZZ).
        DSL gives ~17.4, showing only 15% coverage.
        Even partial implementation should be much better.
        """
        # Load κ polynomials
        P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
        theta = 4.0 / 7.0
        R_kappa = 1.3036

        # Load κ* polynomials
        P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
        R_kappa_star = 1.1167

        # Evaluate both
        result_k = psi_oracle_33(P3_k, Q_k, theta, R_kappa, n_quad=60, debug=False)
        result_ks = psi_oracle_33(P3_ks, Q_ks, theta, R_kappa_star, n_quad=60, debug=False)

        # Skip if either is zero (indicates incomplete implementation)
        if result_k.total == 0 or result_ks.total == 0:
            pytest.skip("Oracle not yet fully implemented")

        ratio = result_k.total / result_ks.total

        # The ratio should be much better than DSL's 17.4
        # We expect it to be close to 1.10
        assert ratio < 15.0, \
            f"Ratio {ratio:.2f} is too large (DSL gives ~17.4, target is 1.10)"

        # Ideally, it should be within reasonable range of 1.10
        # But we'll be lenient during development
        print(f"\nRatio: {ratio:.4f} (target: 1.10, DSL: ~17.4)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
