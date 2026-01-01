"""
tests/test_k4_computation.py
Tests for K=4 computation (Phase 48)

These tests validate that:
1. K=4 term builders work correctly
2. K=4 engine computes without errors
3. K=3 pairs give same values in K=4 engine as in K=3 engine
4. Quadrature is stable across different n values
5. Results are in plausible ranges

Created: 2025-12-28 (Phase 48)
"""

import pytest
import numpy as np
from typing import List, Tuple


class TestK4TermBuilders:
    """Tests for K=4 term builders in terms_k3_d1.py."""

    def test_make_all_terms_k4_returns_10_pairs(self):
        """make_all_terms_k4 should return exactly 10 pairs."""
        from src.terms_k3_d1 import make_all_terms_k4

        theta = 4/7
        R = 1.3036
        terms = make_all_terms_k4(theta, R)

        assert len(terms) == 10
        expected_keys = {"11", "22", "33", "44", "12", "13", "14", "23", "24", "34"}
        assert set(terms.keys()) == expected_keys

    def test_make_all_terms_k4_v2_returns_10_pairs(self):
        """make_all_terms_k4_v2 should return exactly 10 pairs."""
        from src.terms_k3_d1 import make_all_terms_k4_v2

        theta = 4/7
        R = 1.3036
        terms = make_all_terms_k4_v2(theta, R)

        assert len(terms) == 10

    def test_each_pair_has_4_terms(self):
        """Each pair should have exactly 4 terms (I1, I2, I3, I4)."""
        from src.terms_k3_d1 import make_all_terms_k4_v2

        theta = 4/7
        R = 1.3036
        terms = make_all_terms_k4_v2(theta, R)

        for pk, pair_terms in terms.items():
            assert len(pair_terms) == 4, f"Pair {pk} has {len(pair_terms)} terms, expected 4"

    def test_k4_pairs_use_p4_polynomial(self):
        """K=4 pairs should reference P4 polynomial."""
        from src.terms_k3_d1 import make_all_terms_k4_v2

        theta = 4/7
        R = 1.3036
        terms = make_all_terms_k4_v2(theta, R)

        # Check pair (4,4)
        terms_44 = terms["44"]
        i1_term = terms_44[0]
        poly_names = [pf.poly_name for pf in i1_term.poly_factors]
        assert "P4" in poly_names, "Pair (4,4) I1 should use P4"

        # Check pair (1,4)
        terms_14 = terms["14"]
        i1_term = terms_14[0]
        poly_names = [pf.poly_name for pf in i1_term.poly_factors]
        assert "P4" in poly_names, "Pair (1,4) I1 should use P4"

    def test_one_minus_u_power_for_44(self):
        """Pair (4,4) I1 should have (1-u)^6 power."""
        from src.terms_k3_d1 import make_all_terms_k4_v2

        theta = 4/7
        R = 1.3036
        terms = make_all_terms_k4_v2(theta, R)

        i1_44 = terms["44"][0]
        # (1-u) power for I1 at (4,4): (4-1) + (4-1) = 6
        assert len(i1_44.poly_prefactors) > 0, "Should have poly prefactor for (1-u)^6"


class TestK4Engine:
    """Tests for KappaEngineK4."""

    def test_engine_from_baseline_creates_engine(self):
        """KappaEngineK4.from_baseline should create valid engine."""
        from src.kappa_engine_k4 import KappaEngineK4

        engine = KappaEngineK4.from_baseline(n_quad=40)

        assert engine.K == 4
        assert len(engine.P4_coeffs) > 0
        assert engine.theta == pytest.approx(4/7, rel=1e-10)

    def test_engine_compute_kappa_runs(self):
        """Engine should compute κ without errors."""
        from src.kappa_engine_k4 import KappaEngineK4

        engine = KappaEngineK4.from_baseline(n_quad=40)
        result = engine.compute_kappa()

        assert result is not None
        assert result.kappa > 0
        assert result.c > 0

    def test_engine_k4_has_higher_base(self):
        """K=4 should have higher base than K=3."""
        from src.kappa_engine import compute_base

        base_k3 = compute_base(1.3036, 3)
        base_k4 = compute_base(1.3036, 4)

        assert base_k4 > base_k3
        # K=4: exp(R) + 7 vs K=3: exp(R) + 5
        assert base_k4 - base_k3 == pytest.approx(2.0, rel=1e-10)


class TestK4vsK3IntegralConsistency:
    """Tests that K=3 pairs give same values in K=4 engine."""

    def test_k3_pairs_same_integrals_when_p4_zero(self):
        """K=3 pairs should give same S12, S34 when P4=0."""
        from src.kappa_engine import KappaEngine
        from src.kappa_engine_k4 import KappaEngineK4
        from src.polynomials import load_przz_polynomials

        # K=3 engine
        engine_k3 = KappaEngine.from_przz_kappa(n_quad=40)
        result_k3 = engine_k3.compute_kappa()

        # K=4 engine with P4=0
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        engine_k4 = KappaEngineK4(
            P1_coeffs=P1.tilde_coeffs.tolist(),
            P2_coeffs=P2.tilde_coeffs.tolist(),
            P3_coeffs=P3.tilde_coeffs.tolist(),
            P4_coeffs=[0.0, 0.0, 0.0],
            Q_coeffs=Q.to_monomial().coeffs.tolist(),
            theta=4/7,
            R=1.3036,
            n_quad=40,
        )
        result_k4 = engine_k4.compute_kappa()

        # Integrals should match
        assert result_k4.integrals.S12_plus == pytest.approx(result_k3.integrals.S12_plus, rel=1e-8)
        assert result_k4.integrals.S12_minus == pytest.approx(result_k3.integrals.S12_minus, rel=1e-8)
        assert result_k4.integrals.S34_plus == pytest.approx(result_k3.integrals.S34_plus, rel=1e-8)

    def test_k3_pairs_different_kappa_due_to_higher_base(self):
        """κ should be lower in K=4 engine due to higher base (even with P4=0)."""
        from src.kappa_engine import KappaEngine
        from src.kappa_engine_k4 import KappaEngineK4
        from src.polynomials import load_przz_polynomials

        # K=3 engine
        engine_k3 = KappaEngine.from_przz_kappa(n_quad=40)
        result_k3 = engine_k3.compute_kappa()

        # K=4 engine with P4=0
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        engine_k4 = KappaEngineK4(
            P1_coeffs=P1.tilde_coeffs.tolist(),
            P2_coeffs=P2.tilde_coeffs.tolist(),
            P3_coeffs=P3.tilde_coeffs.tolist(),
            P4_coeffs=[0.0, 0.0, 0.0],
            Q_coeffs=Q.to_monomial().coeffs.tolist(),
            theta=4/7,
            R=1.3036,
            n_quad=40,
        )
        result_k4 = engine_k4.compute_kappa()

        # K=4 κ should be lower because higher base increases c
        assert result_k4.kappa < result_k3.kappa
        # The difference should be significant (not just noise)
        assert result_k3.kappa - result_k4.kappa > 0.1


class TestK4QuadratureStability:
    """Tests for quadrature stability."""

    def test_quadrature_converges(self):
        """K=4 results should converge as n increases."""
        from src.kappa_engine_k4 import KappaEngineK4

        results = []
        for n in [30, 40, 50]:
            engine = KappaEngineK4.from_baseline(n_quad=n)
            result = engine.compute_kappa()
            results.append(result.c)

        # Check convergence (difference should decrease)
        diff_1 = abs(results[1] - results[0])
        diff_2 = abs(results[2] - results[1])

        # The second difference should be smaller or similar
        assert diff_2 <= diff_1 * 2, "Quadrature should be converging"


class TestK4PlausibleRanges:
    """Tests for plausible result ranges."""

    def test_c_in_plausible_range(self):
        """c should be in plausible range (1.5 < c < 4.0)."""
        from src.kappa_engine_k4 import KappaEngineK4

        engine = KappaEngineK4.from_baseline(n_quad=40)
        result = engine.compute_kappa()

        assert 1.5 < result.c < 4.0, f"c={result.c} outside plausible range"

    def test_kappa_in_plausible_range(self):
        """κ should be in plausible range (0.1 < κ < 0.6)."""
        from src.kappa_engine_k4 import KappaEngineK4

        engine = KappaEngineK4.from_baseline(n_quad=40)
        result = engine.compute_kappa()

        assert 0.1 < result.kappa < 0.6, f"κ={result.kappa} outside plausible range"

    def test_correction_factors_near_one(self):
        """g_I1 and g_I2 should be near 1."""
        from src.kappa_engine_k4 import KappaEngineK4

        engine = KappaEngineK4.from_baseline(n_quad=40)
        result = engine.compute_kappa()

        assert 0.99 < result.corrections.g_I1 < 1.01
        assert 0.99 < result.corrections.g_I2 < 1.02


class TestK4PairNormalization:
    """Tests for pair normalization factors."""

    def test_factorial_norm_for_k4_pairs(self):
        """Check factorial normalization for K=4 pairs."""
        from src.evaluator.pairs import factorial_norm

        # (4,4): 1/(4! × 4!) = 1/576
        assert factorial_norm(4, 4) == pytest.approx(1/576, rel=1e-10)

        # (1,4): 1/(1! × 4!) = 1/24
        assert factorial_norm(1, 4) == pytest.approx(1/24, rel=1e-10)

        # (2,4): 1/(2! × 4!) = 1/48
        assert factorial_norm(2, 4) == pytest.approx(1/48, rel=1e-10)

        # (3,4): 1/(3! × 4!) = 1/144
        assert factorial_norm(3, 4) == pytest.approx(1/144, rel=1e-10)

    def test_symmetry_factor_for_k4_pairs(self):
        """Check symmetry factors for K=4 pairs."""
        from src.evaluator.pairs import symmetry_factor

        # Diagonal pairs: factor = 1
        assert symmetry_factor(4, 4) == 1.0

        # Off-diagonal pairs: factor = 2
        assert symmetry_factor(1, 4) == 2.0
        assert symmetry_factor(2, 4) == 2.0
        assert symmetry_factor(3, 4) == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
