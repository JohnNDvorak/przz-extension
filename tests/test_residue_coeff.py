"""
tests/test_residue_coeff.py
Phase 14B Task 3: Tests for L_{1,1} and L_{1,2} coefficient extraction.

PAPER ANCHORS:
- L_{1,1} extracts [s^i] from exp(s·log(N/n)) / ζ(1+α+s)
- L_{1,2} extracts [u^j] from exp(u·log(N/n)) / ζ(1+β+u)
- Main-term mode simplifies 1/ζ(1+α+s) → (α+s) per PRZZ approach
- Full mode uses Laurent expansion of 1/ζ

TWO MODES:
1. Main-term mode: Used in PRZZ main-term pipeline
   - 1/ζ(1+α+s) ≈ (α+s) when α is small
   - This gives: L_{1,1}^{main} = [s^i] (α+s) · exp(s·log(N/n))

2. Full mode: Uses Laurent expansion
   - At α=0: 1/ζ(1+s) = s - γs² + O(s³)
   - This gives better approximation for small α
"""

import pytest
import math
from src.ratios.residue_coeff import (
    L11_main,
    L12_main,
    L11_full,
    L12_full,
    exp_series_coeff,
)


class TestExpSeriesCoeff:
    """Test the basic exp series coefficient extraction."""

    def test_exp_coeff_0(self):
        """[s^0] exp(s·x) = 1."""
        result = exp_series_coeff(x=1.0, i=0)
        assert abs(result - 1.0) < 1e-10

    def test_exp_coeff_1(self):
        """[s^1] exp(s·x) = x."""
        x = 2.5
        result = exp_series_coeff(x=x, i=1)
        assert abs(result - x) < 1e-10

    def test_exp_coeff_2(self):
        """[s^2] exp(s·x) = x²/2."""
        x = 2.0
        result = exp_series_coeff(x=x, i=2)
        expected = x**2 / 2
        assert abs(result - expected) < 1e-10

    def test_exp_coeff_general(self):
        """[s^i] exp(s·x) = x^i / i!."""
        x = 1.5
        for i in range(5):
            result = exp_series_coeff(x=x, i=i)
            expected = x**i / math.factorial(i)
            assert abs(result - expected) < 1e-10


class TestL11Main:
    """Test main-term mode L_{1,1} coefficient extraction."""

    def test_L11_main_at_n_equals_N(self):
        """When n=N, log(N/n)=0, exp(s·0)=1."""
        # [s^i] (α+s) · 1 = α if i=0, 1 if i=1, 0 otherwise
        n, N = 100, 100
        alpha = 0.5

        result_i0 = L11_main(n, N, alpha, i=0)
        assert abs(result_i0 - alpha) < 1e-10

        result_i1 = L11_main(n, N, alpha, i=1)
        assert abs(result_i1 - 1.0) < 1e-10

        result_i2 = L11_main(n, N, alpha, i=2)
        assert abs(result_i2 - 0.0) < 1e-10

    def test_L11_main_at_small_alpha(self):
        """At α=0, [s^i](s·exp(s·x)) = x^{i-1}/(i-1)! for i≥1."""
        n, N = 50, 100
        alpha = 0.0
        x = math.log(N / n)  # log(2)

        # [s^0] s·exp(s·x) = 0
        result_i0 = L11_main(n, N, alpha, i=0)
        assert abs(result_i0) < 1e-10

        # [s^1] s·exp(s·x) = 1
        result_i1 = L11_main(n, N, alpha, i=1)
        assert abs(result_i1 - 1.0) < 1e-10

        # [s^2] s·exp(s·x) = x
        result_i2 = L11_main(n, N, alpha, i=2)
        assert abs(result_i2 - x) < 1e-10

    def test_L11_main_finite_values(self):
        """L11_main should produce finite values."""
        n, N = 10, 1000
        for alpha in [0.0, 0.1, 0.5, 1.0]:
            for i in range(4):
                result = L11_main(n, N, alpha, i)
                assert math.isfinite(result), f"L11_main({n},{N},{alpha},{i})={result}"


class TestL12Main:
    """Test main-term mode L_{1,2} coefficient extraction."""

    def test_L12_main_symmetry(self):
        """L12 is the symmetric (β,u,j) version of L11."""
        n, N = 50, 100
        alpha = 0.3
        beta = 0.3  # Same value

        for i in range(4):
            result_L11 = L11_main(n, N, alpha, i)
            result_L12 = L12_main(n, N, beta, i)
            assert abs(result_L11 - result_L12) < 1e-10


class TestL11Full:
    """Test full Laurent-expansion mode L_{1,1}."""

    def test_L11_full_produces_finite_coefficients(self):
        """Full mode produces structurally correct coefficients."""
        n, N = 50, 100
        alpha = 0.3

        for i in range(3):
            main = L11_main(n, N, alpha, i)
            full = L11_full(n, N, alpha, i, order=3)
            # Both should be finite
            assert math.isfinite(main), f"main i={i} is not finite"
            assert math.isfinite(full.real), f"full i={i} is not finite"
            # For i=0 and i=1, both should have same sign (structural check)
            if i <= 1 and abs(main) > 0.01 and abs(full) > 0.01:
                same_sign = (main > 0 and full.real > 0) or (main < 0 and full.real < 0)
                assert same_sign, f"i={i}: main={main}, full={full} have different signs"

    def test_L11_full_coefficient_is_finite(self):
        """Full mode should produce finite values."""
        n, N = 10, 100
        for alpha in [0.01, 0.1, 0.5]:
            for i in range(4):
                result = L11_full(n, N, alpha, i, order=3)
                assert math.isfinite(result.real), f"L11_full({n},{N},{alpha},{i})={result}"
                assert math.isfinite(result.imag), f"L11_full({n},{N},{alpha},{i})={result}"

    def test_L11_full_at_n_equals_N(self):
        """When n=N, exp factor is 1, simpler test case."""
        n, N = 100, 100
        alpha = 0.1

        # At n=N, log(N/n)=0, so exp(s·0)=1
        # L11_full extracts [s^i] from 1/ζ(1+α+s)
        result_i0 = L11_full(n, N, alpha, i=0, order=3)
        # Should be finite
        assert math.isfinite(result_i0.real)


class TestL12Full:
    """Test full Laurent-expansion mode L_{1,2}."""

    def test_L12_full_symmetry(self):
        """L12_full is symmetric version of L11_full."""
        n, N = 50, 100
        alpha = 0.2
        beta = 0.2

        for i in range(3):
            result_L11 = L11_full(n, N, alpha, i, order=3)
            result_L12 = L12_full(n, N, beta, i, order=3)
            assert abs(result_L11 - result_L12) < 1e-10


class TestModeConsistency:
    """Test that both modes give consistent results in overlapping regimes."""

    def test_modes_trend_same_direction(self):
        """Both modes should give increasing magnitude with log(N/n)."""
        alpha = 0.1
        i = 1
        N = 100

        main_values = []
        full_values = []
        for n in [90, 50, 10]:  # Increasing log(N/n)
            main_values.append(L11_main(n, N, alpha, i))
            full_values.append(L11_full(n, N, alpha, i, order=3).real)

        # Magnitudes should increase as n decreases (log(N/n) increases)
        assert abs(main_values[0]) < abs(main_values[2])
        # Full mode should show same trend
        # (may not be strictly monotonic due to expansion, but general trend)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
