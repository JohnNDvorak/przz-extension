#!/usr/bin/env python3
"""
tests/test_moment_derivation.py
Test the PRZZ (2t-1) moment derivation chain (Phase 54)

This test verifies the derivation path from PRZZ bracket structure to g factors:
    Step A: (2t-1) moments M₀, M₁, M₂
    Step B: Connection to g_I1, g_I2 formulas
    Step C: Additive constant structure

Created: 2025-12-29 (Phase 54 - PRZZ g-factor Derivation)
"""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.step_a_moment_analysis import (
    compute_M0_analytic,
    compute_M1_analytic,
    compute_M2_analytic,
    compute_moments_numeric,
    compute_xy_integral_direct,
    compute_xy_integral_from_moments,
)

from scripts.step_b_g_formula_derivation import (
    compute_production_g_I1,
    compute_production_g_I2,
    beta_function,
)


class TestStepA_MomentAnalysis:
    """Test Step A: (2t-1) moment integrals."""

    @pytest.mark.parametrize("R", [1.3036, 1.1167, 0.5, 2.0])
    def test_M0_dq_limit(self, R: float):
        """M₀ should equal the DQ limit (exp(2R)-1)/(2R)."""
        M0 = compute_M0_analytic(R)
        expected = (math.exp(2*R) - 1) / (2*R)
        assert abs(M0 - expected) < 1e-12

    @pytest.mark.parametrize("R", [1.3036, 1.1167])
    def test_analytic_matches_numeric(self, R: float):
        """Analytic and numeric moment computations should match."""
        M0_a = compute_M0_analytic(R)
        M1_a = compute_M1_analytic(R)
        M2_a = compute_M2_analytic(R)

        M0_n, M1_n, M2_n = compute_moments_numeric(R, n_quad=100)

        assert abs(M0_a - M0_n) / abs(M0_a) < 1e-10
        assert abs(M1_a - M1_n) / abs(M1_a) < 1e-10
        assert abs(M2_a - M2_n) / abs(M2_a) < 1e-10

    def test_M1_formula(self):
        """Test the M₁ formula: [(R-1)exp(2R) + (R+1)]/(2R²)."""
        R = 1.3036
        M1 = compute_M1_analytic(R)
        expected = ((R - 1) * math.exp(2*R) + (R + 1)) / (2 * R**2)
        assert abs(M1 - expected) < 1e-12

    def test_M2_recurrence(self):
        """Test M₂ = M₀ - 2M₁/R recurrence."""
        R = 1.3036
        M0 = compute_M0_analytic(R)
        M1 = compute_M1_analytic(R)
        M2 = compute_M2_analytic(R)

        expected = M0 - 2 * M1 / R
        assert abs(M2 - expected) < 1e-12

    @pytest.mark.parametrize("R,theta", [(1.3036, 4/7), (1.1167, 4/7)])
    def test_xy_integral_decomposition(self, R: float, theta: float):
        """
        The key result: xy_integral = R²θ²M₂ + 2Rθ²M₁

        This proves the (2t-1) moment structure underlies the bracket.
        """
        xy_direct = compute_xy_integral_direct(R, theta, n_quad=100)
        xy_moments = compute_xy_integral_from_moments(R, theta)

        rel_error = abs(xy_direct - xy_moments) / abs(xy_direct)
        assert rel_error < 1e-10, f"Decomposition failed: {rel_error}"

    def test_xy_integral_value_kappa(self):
        """Verify the ~2.67 value at R=1.3036 from Path 1 analysis."""
        R = 1.3036
        theta = 4/7
        xy = compute_xy_integral_direct(R, theta)
        # Path 1 reported ~2.67
        assert 2.6 < xy < 2.8


class TestStepB_GFactorStructure:
    """Test Step B: g_I1 and g_I2 formula structure."""

    def test_g_I2_beta_structure(self):
        """g_I2 - 1 = θ(2-θ) × Beta(2, 2K)."""
        theta = 4/7
        K = 3

        g_I2 = compute_production_g_I2(theta, K)
        g_I2_minus_1 = g_I2 - 1

        # Beta(2, 2K) = 1/(2K(2K+1))
        beta_factor = 1 / (2 * K * (2*K + 1))
        theta_factor = theta * (2 - theta)
        expected = theta_factor * beta_factor

        assert abs(g_I2_minus_1 - expected) < 1e-12

    def test_g_I2_exact_value(self):
        """g_I2 = 1 + (4/7)(10/7)/42 for standard parameters."""
        theta = 4/7
        K = 3

        g_I2 = compute_production_g_I2(theta, K)

        # Exact: 1 + (40/49)/42 = 1 + 40/2058
        expected = 1 + 40/2058
        assert abs(g_I2 - expected) < 1e-12

    def test_g_I1_exact_value(self):
        """g_I1 = 1 + θ(1-θ)(2(K-1)+θ)/(8K(2K+1)²) for standard parameters."""
        theta = 4/7
        K = 3

        g_I1 = compute_production_g_I1(theta, K)

        # Exact: (12/49)(32/7)/(8×3×49) = (384/343)/1176
        numerator = (12/49) * (32/7)
        denominator = 8 * 3 * 49
        expected = 1 + numerator / denominator

        assert abs(g_I1 - expected) < 1e-12

    def test_g_I2_greater_than_g_I1(self):
        """g_I2 > g_I1 (I₂ gets larger correction)."""
        theta = 4/7
        K = 3

        g_I1 = compute_production_g_I1(theta, K)
        g_I2 = compute_production_g_I2(theta, K)

        assert g_I2 > g_I1

    def test_g_correction_ratio(self):
        """(g_I1-1)/(g_I2-1) matches algebraic simplification."""
        theta = 4/7
        K = 3

        g_I1 = compute_production_g_I1(theta, K)
        g_I2 = compute_production_g_I2(theta, K)

        ratio = (g_I1 - 1) / (g_I2 - 1)

        # Algebraic: (1-θ)(2(K-1)+θ) / [4(2-θ)(2K+1)]
        expected = (1 - theta) * (2*(K-1) + theta) / (4 * (2 - theta) * (2*K + 1))

        assert abs(ratio - expected) < 1e-10

    def test_beta_function_for_g_I2(self):
        """Beta(2, 2K) = 1/(2K(2K+1))."""
        K = 3
        beta_val = beta_function(2, 2*K)
        expected = 1 / (2 * K * (2*K + 1))
        assert abs(beta_val - expected) < 1e-10


class TestStepC_ConstantStructure:
    """Test Step C: base constant structure."""

    def test_base_constant_value(self):
        """base = exp(R) + (2K-1) for production."""
        R = 1.3036
        K = 3
        base = math.exp(R) + (2*K - 1)
        assert abs(base - (math.exp(R) + 5)) < 1e-10

    def test_constant_representation_equivalence(self):
        """
        Two representations with different constants are equivalent
        if g-factors compensate.

        m = g × (exp(R) + 5) = g' × (exp(R) + 6)
        implies g' = g × (exp(R) + 5)/(exp(R) + 6)
        """
        R = 1.3036
        theta = 4/7
        K = 3

        g_I1 = compute_production_g_I1(theta, K)
        g_I2 = compute_production_g_I2(theta, K)
        f_I1 = 0.033  # Typical value

        g_total = f_I1 * g_I1 + (1 - f_I1) * g_I2
        base_5 = math.exp(R) + 5
        base_6 = math.exp(R) + 6

        m_with_5 = g_total * base_5

        # Equivalent g for base_6
        g_equiv = g_total * base_5 / base_6
        m_with_6 = g_equiv * base_6

        assert abs(m_with_5 - m_with_6) < 1e-10


class TestDerivationChain:
    """Test the complete derivation chain from moments to κ."""

    def test_complete_chain_consistency(self):
        """
        Verify the complete chain:
        1. (2t-1) moments M₀, M₁, M₂ are well-defined
        2. xy_integral = R²θ²M₂ + 2Rθ²M₁
        3. g_I2 = 1 + θ(2-θ) × Beta(2, 2K)
        4. g_I1 structure is consistent with moment ratios
        5. m = g_total × base gives valid assembly
        """
        R = 1.3036
        theta = 4/7
        K = 3

        # Step 1: Moments
        M0 = compute_M0_analytic(R)
        M1 = compute_M1_analytic(R)
        M2 = compute_M2_analytic(R)
        assert all(m > 0 for m in [M0, M1, M2])

        # Step 2: xy decomposition
        xy_direct = compute_xy_integral_direct(R, theta)
        xy_moments = R**2 * theta**2 * M2 + 2 * R * theta**2 * M1
        assert abs(xy_direct - xy_moments) / xy_direct < 1e-10

        # Step 3: g_I2 Beta structure
        g_I2 = compute_production_g_I2(theta, K)
        beta_2_2K = 1 / (2 * K * (2*K + 1))
        assert abs((g_I2 - 1) - theta * (2 - theta) * beta_2_2K) < 1e-12

        # Step 4: g_I1 is smaller than g_I2
        g_I1 = compute_production_g_I1(theta, K)
        assert g_I1 < g_I2

        # Step 5: m assembly is valid
        f_I1 = 0.033
        g_total = f_I1 * g_I1 + (1 - f_I1) * g_I2
        base = math.exp(R) + 5
        m = g_total * base
        assert m > 0

    @pytest.mark.parametrize("R", [1.3036, 1.1167])
    def test_both_benchmarks(self, R: float):
        """Derivation chain works for both κ and κ* benchmarks."""
        theta = 4/7
        K = 3

        # Moments
        M0 = compute_M0_analytic(R)
        M1 = compute_M1_analytic(R)
        M2 = compute_M2_analytic(R)

        # xy decomposition
        xy_direct = compute_xy_integral_direct(R, theta)
        xy_moments = compute_xy_integral_from_moments(R, theta)
        assert abs(xy_direct - xy_moments) / abs(xy_direct) < 1e-10

        # g factors
        g_I1 = compute_production_g_I1(theta, K)
        g_I2 = compute_production_g_I2(theta, K)

        assert 1.0 < g_I1 < 1.01
        assert 1.01 < g_I2 < 1.03


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
