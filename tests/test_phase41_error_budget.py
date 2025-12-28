#!/usr/bin/env python3
"""
tests/test_phase41_error_budget.py
Phase 41: Error Budget Attribution Tests

Validates the attribution formulas used in Phase 41 residual analysis.

Created: 2025-12-27 (Phase 41)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import math


class TestErrorBudgetFormulas:
    """Test the error budget attribution formulas."""

    def test_m_needed_formula_closes_gap(self):
        """m_needed = (c_target - S12+ - S34) / S12- should close the gap exactly."""
        # Example values from Phase 40
        c_target = 2.137454
        S12_plus = 0.7975
        S12_minus = 0.2201
        S34 = -0.6002

        m_needed = (c_target - S12_plus - S34) / S12_minus

        # Verify formula works: using m_needed should hit c_target exactly
        c_check = S12_plus + m_needed * S12_minus + S34
        assert abs(c_check - c_target) < 1e-10, f"c_check={c_check}, c_target={c_target}"

    def test_S34_needed_formula_closes_gap(self):
        """S34_needed = c_target - S12+ - m_derived * S12- should close the gap exactly."""
        c_target = 2.137454
        S12_plus = 0.7975
        S12_minus = 0.2201
        m_derived = 8.8007

        S34_needed = c_target - S12_plus - m_derived * S12_minus

        # Verify formula works: using S34_needed should hit c_target exactly
        c_check = S12_plus + m_derived * S12_minus + S34_needed
        assert abs(c_check - c_target) < 1e-10, f"c_check={c_check}, c_target={c_target}"

    def test_both_attributions_equivalent_at_target(self):
        """When c_computed == c_target, both delta_m and delta_S34 should be zero."""
        c_target = 2.137454
        S12_plus = 0.7975
        S12_minus = 0.2201
        m_derived = 8.8007

        # Compute S34 such that c_computed == c_target
        S34_exact = c_target - S12_plus - m_derived * S12_minus

        # Now compute attributions
        m_needed = (c_target - S12_plus - S34_exact) / S12_minus
        S34_needed = c_target - S12_plus - m_derived * S12_minus

        delta_m = m_needed / m_derived - 1
        delta_S34 = S34_needed - S34_exact

        # Both should be essentially zero
        assert abs(delta_m) < 1e-10, f"delta_m={delta_m}"
        assert abs(delta_S34) < 1e-10, f"delta_S34={delta_S34}"

    def test_m_needed_increases_when_c_too_low(self):
        """When c_computed < c_target, m_needed > m_derived."""
        c_target = 2.137454
        S12_plus = 0.7975
        S12_minus = 0.2201
        m_derived = 8.8007
        S34 = -0.6002

        c_computed = S12_plus + m_derived * S12_minus + S34
        # This gives c_computed < c_target (verified in Phase 40)

        m_needed = (c_target - S12_plus - S34) / S12_minus

        # If c_computed < c_target, then m_needed > m_derived (need bigger multiplier)
        if c_computed < c_target:
            assert m_needed > m_derived, f"m_needed={m_needed} should be > m_derived={m_derived}"

    def test_S34_needed_less_negative_when_c_too_low(self):
        """When c_computed < c_target, S34_needed should be less negative than S34."""
        c_target = 2.137454
        S12_plus = 0.7975
        S12_minus = 0.2201
        m_derived = 8.8007
        S34 = -0.6002  # Negative

        c_computed = S12_plus + m_derived * S12_minus + S34

        S34_needed = c_target - S12_plus - m_derived * S12_minus

        # If c_computed < c_target and S34 is negative, S34_needed should be "less negative"
        if c_computed < c_target:
            assert S34_needed > S34, f"S34_needed={S34_needed} should be > S34={S34}"


class TestConvergenceDetection:
    """Test convergence detection logic."""

    def test_converged_sequence(self):
        """Sequence with decreasing deltas should be marked converged."""
        values = [1.0, 1.00001, 1.000010001, 1.0000100001]
        threshold = 1e-5

        deltas = [float('nan')]
        for i in range(1, len(values)):
            delta = abs((values[i] - values[i-1]) / values[i-1])
            deltas.append(delta)

        # Final delta should be below threshold
        converged = deltas[-1] < threshold
        assert converged, f"Final delta {deltas[-1]} should be < {threshold}"

    def test_not_converged_sequence(self):
        """Sequence with large deltas should NOT be marked converged."""
        values = [1.0, 1.01, 1.02, 1.03]
        threshold = 1e-5

        deltas = [float('nan')]
        for i in range(1, len(values)):
            delta = abs((values[i] - values[i-1]) / values[i-1])
            deltas.append(delta)

        converged = deltas[-1] < threshold
        assert not converged, f"Final delta {deltas[-1]} should be >= {threshold}"

    def test_convergence_with_negative_values(self):
        """Convergence should work correctly with negative values."""
        values = [-0.6002, -0.60019, -0.600189, -0.6001889]
        threshold = 1e-5

        deltas = [float('nan')]
        for i in range(1, len(values)):
            if abs(values[i-1]) > 1e-15:
                delta = abs((values[i] - values[i-1]) / values[i-1])
            else:
                delta = abs(values[i] - values[i-1])
            deltas.append(delta)

        # The sequence is converging
        assert deltas[-1] < deltas[-2], "Deltas should decrease"


class TestDecisionGateLogic:
    """Test decision gate interpretation logic."""

    def test_opposite_m_shifts_detection(self):
        """Detect when m shifts have opposite signs."""
        delta_m_kappa = +0.0015  # Needs increase
        delta_m_kappa_star = -0.0002  # Needs decrease

        m_shifts_same_sign = (delta_m_kappa * delta_m_kappa_star) > 0
        assert not m_shifts_same_sign, "Opposite m shifts should give False"

    def test_same_m_shifts_detection(self):
        """Detect when m shifts have same sign."""
        delta_m_kappa = +0.0015
        delta_m_kappa_star = +0.0008

        m_shifts_same_sign = (delta_m_kappa * delta_m_kappa_star) > 0
        assert m_shifts_same_sign, "Same sign m shifts should give True"

    def test_route_a_condition(self):
        """Route A: m same sign, S34 opposite sign."""
        # This means m has systematic error
        delta_m_k1, delta_m_k2 = +0.001, +0.002  # Same sign
        delta_S34_k1, delta_S34_k2 = +0.001, -0.001  # Opposite sign

        m_same = (delta_m_k1 * delta_m_k2) > 0
        S34_same = (delta_S34_k1 * delta_S34_k2) > 0

        route_a = m_same and not S34_same
        assert route_a, "Should be Route A"

    def test_route_b_condition(self):
        """Route B: S34 same sign, m opposite sign."""
        # This means S34 has systematic error
        delta_m_k1, delta_m_k2 = +0.001, -0.002  # Opposite sign
        delta_S34_k1, delta_S34_k2 = +0.001, +0.002  # Same sign

        m_same = (delta_m_k1 * delta_m_k2) > 0
        S34_same = (delta_S34_k1 * delta_S34_k2) > 0

        route_b = S34_same and not m_same
        assert route_b, "Should be Route B"


class TestRealValues:
    """Test with actual Phase 40 values."""

    def test_kappa_attribution_signs(self):
        """Verify kappa needs m increase (positive delta_m)."""
        # From Phase 40:
        # kappa: c_gap = -0.14% (computed < target)
        # Therefore m_needed > m_derived (positive delta_m)

        c_target = 2.137454406132173
        S12_plus = 0.7975  # Approximate
        S12_minus = 0.2201
        S34 = -0.6002
        m_derived = 8.8007

        c_computed = S12_plus + m_derived * S12_minus + S34
        m_needed = (c_target - S12_plus - S34) / S12_minus
        delta_m = m_needed / m_derived - 1

        # Phase 40 says kappa needs m increase
        assert delta_m > 0, f"delta_m={delta_m} should be positive for kappa"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
