"""
tests/test_mirror_transform_harness.py
Phase 12.0: Validation tests for mirror transform harness.

Tests that the harness produces values consistent with the canonical evaluator.
"""

import pytest
from src.mirror_transform_harness import MirrorTransformHarness, run_harness_comparison
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import compute_c_paper_with_mirror


class TestS34Validation:
    """Test that harness S34 matches canonical evaluator."""

    @pytest.fixture
    def kappa_polynomials(self):
        """Load κ benchmark polynomials."""
        P1, P2, P3, Q = load_przz_polynomials()
        return {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    @pytest.fixture
    def kappa_star_polynomials(self):
        """Load κ* benchmark polynomials."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    def test_s34_matches_canonical_kappa(self, kappa_polynomials):
        """S34 from harness should match canonical evaluator for κ benchmark."""
        theta = 4/7
        R = 1.3036
        n = 40

        # Run harness
        harness = MirrorTransformHarness(theta, R, n, kappa_polynomials)
        harness_result = harness.run()

        # Run canonical evaluator
        canonical = compute_c_paper_with_mirror(
            theta=theta, R=R, n=n, polynomials=kappa_polynomials,
            pair_mode='triangle'
        )
        canonical_s34 = canonical.per_term['_S34_triangle_total']

        # Should match exactly (same computation path)
        assert abs(harness_result.S34_total - canonical_s34) < 1e-10, (
            f"S34 mismatch: harness={harness_result.S34_total}, "
            f"canonical={canonical_s34}"
        )

    def test_s34_matches_canonical_kappa_star(self, kappa_star_polynomials):
        """S34 from harness should match canonical evaluator for κ* benchmark."""
        theta = 4/7
        R = 1.1167
        n = 40

        # Run harness
        harness = MirrorTransformHarness(theta, R, n, kappa_star_polynomials)
        harness_result = harness.run()

        # Run canonical evaluator
        canonical = compute_c_paper_with_mirror(
            theta=theta, R=R, n=n, polynomials=kappa_star_polynomials,
            pair_mode='triangle'
        )
        canonical_s34 = canonical.per_term['_S34_triangle_total']

        # Should match exactly
        assert abs(harness_result.S34_total - canonical_s34) < 1e-10, (
            f"S34 mismatch: harness={harness_result.S34_total}, "
            f"canonical={canonical_s34}"
        )

    def test_s34_per_pair_nonzero(self, kappa_polynomials):
        """Each S34 per-pair value should be non-zero (not placeholder)."""
        theta = 4/7
        R = 1.3036
        n = 40

        harness = MirrorTransformHarness(theta, R, n, kappa_polynomials)
        result = harness.run()

        # Check that at least the diagonal pairs have non-trivial values
        for pair_key in ["11", "22"]:
            assert result.S34_pair[pair_key] != 0.0, (
                f"S34 for pair {pair_key} is still zero (placeholder?)"
            )

    def test_s34_total_negative(self, kappa_polynomials):
        """S34 total should be negative (per PRZZ structure)."""
        theta = 4/7
        R = 1.3036
        n = 40

        harness = MirrorTransformHarness(theta, R, n, kappa_polynomials)
        result = harness.run()

        # PRZZ I3/I4 terms contribute negative values to c
        assert result.S34_total < 0, (
            f"S34_total should be negative, got {result.S34_total}"
        )


class TestHarnessStructure:
    """Test harness internal structure and consistency."""

    @pytest.fixture
    def kappa_polynomials(self):
        P1, P2, P3, Q = load_przz_polynomials()
        return {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    def test_harness_runs_without_error(self, kappa_polynomials):
        """Basic smoke test."""
        theta = 4/7
        R = 1.3036
        n = 40

        harness = MirrorTransformHarness(theta, R, n, kappa_polynomials)
        result = harness.run()

        assert result is not None
        assert result.S12_direct_total > 0
        assert result.S12_basis_total > 0

    def test_c_assembly_consistency(self, kappa_polynomials):
        """Check c assembly formulas are internally consistent."""
        theta = 4/7
        R = 1.3036
        n = 40

        harness = MirrorTransformHarness(theta, R, n, kappa_polynomials)
        result = harness.run()

        import numpy as np

        # c_direct_only = S12_direct + S34
        expected_direct = result.S12_direct_total + result.S34_total
        assert abs(result.c_direct_only - expected_direct) < 1e-10

        # c_with_empirical = S12_direct + m1 * S12_basis + S34
        m1 = np.exp(R) + 5
        expected_empirical = (
            result.S12_direct_total +
            m1 * result.S12_basis_total +
            result.S34_total
        )
        assert abs(result.c_with_empirical - expected_empirical) < 1e-10

        # c_with_operator = S12_direct + S12_operator_mirror + S34
        expected_operator = (
            result.S12_direct_total +
            result.S12_operator_mirror_total +
            result.S34_total
        )
        assert abs(result.c_with_operator - expected_operator) < 1e-10

    def test_m1_implied_calculation(self, kappa_polynomials):
        """Check m1_implied = S12_operator_mirror / S12_basis."""
        theta = 4/7
        R = 1.3036
        n = 40

        harness = MirrorTransformHarness(theta, R, n, kappa_polynomials)
        result = harness.run()

        expected_m1 = result.S12_operator_mirror_total / result.S12_basis_total
        assert abs(result.m1_implied - expected_m1) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
