"""
tests/test_zeta_logderiv.py
Phase 14B Task 2: Tests for ζ'/ζ evaluator.

PAPER ANCHORS:
- ζ'/ζ(s) = -Σ_p log(p)/(p^s - 1) for Re(s) > 1
- ζ'/ζ(1+ε) = -1/ε + γ_E + O(ε) (Laurent expansion near pole)
- Used in J₁₂ (double product), J₁₃, J₁₄ bracket terms
"""

import pytest
import math
from src.ratios.zeta_logderiv import (
    zeta_log_deriv_1_plus_eps,
    zeta_log_deriv_prime_sum,
    EULER_MASCHERONI,
)


class TestZetaLogDerivLaurent:
    """Test Laurent expansion of ζ'/ζ(1+ε) near ε=0."""

    def test_pole_term_dominates_near_zero(self):
        """ζ'/ζ(1+ε) ~ -1/ε as ε→0."""
        for eps in [0.1, 0.01, 0.001]:
            result = zeta_log_deriv_1_plus_eps(eps)
            pole_term = -1.0 / eps

            # The pole should dominate
            ratio = result / pole_term
            assert 0.8 < ratio < 1.2, (
                f"At ε={eps}: ζ'/ζ = {result}, -1/ε = {pole_term}, ratio = {ratio}"
            )

    def test_constant_term_is_euler_mascheroni(self):
        """ζ'/ζ(1+ε) = -1/ε + γ_E + O(ε)."""
        # Extract constant by computing: ζ'/ζ(1+ε) + 1/ε
        for eps in [0.01, 0.001]:
            result = zeta_log_deriv_1_plus_eps(eps, order=2)
            residual = result + 1.0 / eps

            # Should be close to Euler-Mascheroni
            assert abs(residual - EULER_MASCHERONI) < 0.1, (
                f"At ε={eps}: residual = {residual}, expected γ = {EULER_MASCHERONI}"
            )

    def test_is_negative_for_positive_eps(self):
        """ζ'/ζ(1+ε) < 0 for small positive ε (due to -1/ε pole)."""
        for eps in [0.01, 0.1, 0.5]:
            result = zeta_log_deriv_1_plus_eps(eps)
            assert result < 0, f"ζ'/ζ(1+{eps}) = {result} should be negative"

    def test_higher_order_terms(self):
        """Test that order=3 gives better approximation than order=2."""
        eps = 0.1
        order2 = zeta_log_deriv_1_plus_eps(eps, order=2)
        order3 = zeta_log_deriv_1_plus_eps(eps, order=3)

        # They should be close but not identical
        assert abs(order2 - order3) < 0.5
        assert order2 != order3  # But different


class TestZetaLogDerivPrimeSum:
    """Test prime sum evaluation of ζ'/ζ(s) for Re(s) > 1."""

    def test_is_negative_for_large_s(self):
        """ζ'/ζ(s) = -Σ log(p)/(p^s-1) < 0 for Re(s) > 1."""
        for s in [1.5, 2.0, 3.0]:
            result = zeta_log_deriv_prime_sum(s, prime_cutoff=1000)
            # Result is complex but for real s > 1, should have negative real part
            assert result.real < 0, f"ζ'/ζ({s}) = {result} should be negative"

    def test_decreases_in_magnitude_with_s(self):
        """As s → ∞, ζ'/ζ(s) → 0."""
        result_2 = abs(zeta_log_deriv_prime_sum(2.0, prime_cutoff=1000))
        result_3 = abs(zeta_log_deriv_prime_sum(3.0, prime_cutoff=1000))
        result_5 = abs(zeta_log_deriv_prime_sum(5.0, prime_cutoff=1000))

        assert result_2 > result_3 > result_5, (
            f"|ζ'/ζ| should decrease: {result_2} > {result_3} > {result_5}"
        )

    def test_convergence_with_prime_cutoff(self):
        """Prime sum should converge as cutoff increases."""
        s = 2.0
        result_100 = zeta_log_deriv_prime_sum(s, prime_cutoff=100)
        result_1000 = zeta_log_deriv_prime_sum(s, prime_cutoff=1000)
        result_5000 = zeta_log_deriv_prime_sum(s, prime_cutoff=5000)

        # Values should be getting closer
        diff1 = abs(result_1000 - result_100)
        diff2 = abs(result_5000 - result_1000)
        assert diff2 < diff1, "Should converge as cutoff increases"


class TestZetaLogDerivSymmetry:
    """Test symmetry and consistency properties."""

    def test_complex_argument_finite(self):
        """ζ'/ζ should be finite for complex arguments with Re > 1."""
        s = 2.0 + 0.5j
        result = zeta_log_deriv_prime_sum(s, prime_cutoff=100)
        assert math.isfinite(abs(result))


class TestEulerMascheroniConstant:
    """Test that we have the right Euler-Mascheroni constant."""

    def test_euler_mascheroni_value(self):
        """γ ≈ 0.5772156649..."""
        assert abs(EULER_MASCHERONI - 0.5772156649015329) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
