"""
tests/test_logfactor_split_reconstruction.py
Phase 35A.2: Reconstruction gate test for log factor split.

This test validates that the split coefficients correctly reconstruct
the actual I1 term value.

For each triangle pair:
1. Get I1_term from terms_k3_d1
2. Compute I1 value using canonical evaluator
3. Compute F_x, F_y, F_xy using evaluate_term_with_split
4. Reconstruct: I1_reconstructed = (1/θ)×F_xy + F_x + F_y
5. Assert I1_value ≈ I1_reconstructed

This validates:
- Correct coefficient extraction
- Correct integration weights/prefactors
- No hidden normalization mismatches

Created: 2025-12-26 (Phase 35A.2)
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term
from src.unified_s12.logfactor_split import (
    split_logfactor_for_pair,
    evaluate_term_with_split,
    LogFactorSplit,
)


class TestReconstructionGate:
    """Test that split coefficients reconstruct I1 exactly."""

    @pytest.fixture
    def kappa_polys(self):
        """Load κ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def kappa_star_polys(self):
        """Load κ* polynomials."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def params(self):
        """Standard test parameters."""
        return {
            "theta": 4/7,
            "R": 1.3036,
            "K": 3,
            "n_quad": 60,
        }

    def test_reconstruction_pair_11(self, kappa_polys, params):
        """Test that split reconstructs I1 for pair (1,1)."""
        self._test_reconstruction_for_pair("11", kappa_polys, params)

    def test_reconstruction_pair_22(self, kappa_polys, params):
        """Test that split reconstructs I1 for pair (2,2)."""
        self._test_reconstruction_for_pair("22", kappa_polys, params)

    def test_reconstruction_pair_33(self, kappa_polys, params):
        """Test that split reconstructs I1 for pair (3,3)."""
        self._test_reconstruction_for_pair("33", kappa_polys, params)

    def test_reconstruction_pair_12(self, kappa_polys, params):
        """Test that split reconstructs I1 for pair (1,2)."""
        self._test_reconstruction_for_pair("12", kappa_polys, params)

    def test_reconstruction_pair_13(self, kappa_polys, params):
        """Test that split reconstructs I1 for pair (1,3)."""
        self._test_reconstruction_for_pair("13", kappa_polys, params)

    def test_reconstruction_pair_23(self, kappa_polys, params):
        """Test that split reconstructs I1 for pair (2,3)."""
        self._test_reconstruction_for_pair("23", kappa_polys, params)

    def test_reconstruction_kappa_star(self, kappa_star_polys, params):
        """Test reconstruction with κ* polynomials at R=1.1167."""
        params_ks = params.copy()
        params_ks["R"] = 1.1167
        self._test_reconstruction_for_pair("11", kappa_star_polys, params_ks)

    def _test_reconstruction_for_pair(self, pair_key, polynomials, params):
        """
        Core reconstruction test for a specific pair.

        The identity being tested:
            I1_canonical = (1/θ)×F_xy + F_y + F_x

        where:
            - F_xy, F_x, F_y are from the PURE series (before log factor)
            - The log factor (1/θ + x + y) is applied via product rule
        """
        theta = params["theta"]
        R = params["R"]
        n_quad = params["n_quad"]

        # Get the I1 term for this pair
        terms = make_all_terms_k3(theta, R, kernel_regime="paper")
        I1_term = terms[pair_key][0]

        # Method 1: Canonical evaluation (includes log factor multiplication)
        canonical_result = evaluate_term(I1_term, polynomials, n_quad, R=R, theta=theta)
        I1_canonical = canonical_result.value

        # Method 2: Split evaluation - extract pure F coefficients
        coeffs = evaluate_term_with_split(I1_term, polynomials, n_quad, theta, R)

        F_xy = coeffs.get("F_xy", 0.0)
        F_x = coeffs.get("F_x", 0.0)
        F_y = coeffs.get("F_y", 0.0)

        # Reconstruct using product rule:
        # d²/dxdy [(1/θ + x + y) × F] = (1/θ)×F_xy + F_y + F_x
        I1_reconstructed = (1/theta) * F_xy + F_y + F_x

        # Check reconstruction matches canonical
        rel_error = abs(I1_reconstructed - I1_canonical) / abs(I1_canonical) if I1_canonical != 0 else 0

        assert rel_error < 1e-10, (
            f"Reconstruction failed for pair {pair_key}:\n"
            f"  I1_canonical = {I1_canonical:.10f}\n"
            f"  I1_reconstructed = {I1_reconstructed:.10f}\n"
            f"  (1/θ)×F_xy = {F_xy/theta:.10f}\n"
            f"  F_y = {F_y:.10f}\n"
            f"  F_x = {F_x:.10f}\n"
            f"  Relative error = {rel_error:.2e}"
        )


class TestCorrectionFactorInterpretation:
    """Test that the correction factor interpretation is correct."""

    @pytest.fixture
    def kappa_polys(self):
        """Load κ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_correction_factor_formula(self, kappa_polys):
        """
        Test that correction_factor = total_coeff / main_coeff
        and that this equals 1 + θ × (F_x + F_y) / F_xy
        """
        theta = 4/7
        R = 1.3036
        K = 3
        n_quad = 60

        split = split_logfactor_for_pair("11", theta, R, K, kappa_polys, n_quad)

        # Check that correction_factor = total / main
        expected_corr = split.total_coeff / split.main_coeff
        assert abs(split.correction_factor - expected_corr) < 1e-10

        # Check the alternative formula: 1 + θ × (F_x + F_y) / F_xy
        # Note: cross_from_x = F_y, cross_from_y = F_x
        F_xy = theta * split.main_coeff  # main_coeff = (1/θ)×F_xy
        F_x = split.cross_from_y_term
        F_y = split.cross_from_x_term
        cross_sum = F_x + F_y

        alt_corr = 1 + theta * cross_sum / F_xy
        assert abs(split.correction_factor - alt_corr) < 1e-10, (
            f"Correction formula mismatch:\n"
            f"  correction_factor = {split.correction_factor:.10f}\n"
            f"  1 + θ×(F_x+F_y)/F_xy = {alt_corr:.10f}"
        )


class TestAllPairsFullFormula:
    """Test the full formula from start to finish for all pairs."""

    @pytest.fixture
    def kappa_polys(self):
        """Load κ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_all_pairs_reconstruction(self, kappa_polys):
        """
        Run the full formula for all 6 pairs and verify reconstruction.

        This is the "no drift / no hidden mismatch" test GPT requested.
        """
        theta = 4/7
        R = 1.3036
        K = 3
        n_quad = 60

        pair_keys = ["11", "22", "33", "12", "13", "23"]
        terms = make_all_terms_k3(theta, R, kernel_regime="paper")

        results = []

        for pair_key in pair_keys:
            I1_term = terms[pair_key][0]

            # Canonical evaluation
            canonical_result = evaluate_term(I1_term, kappa_polys, n_quad, R=R, theta=theta)
            I1_canonical = canonical_result.value

            # Split evaluation
            coeffs = evaluate_term_with_split(I1_term, kappa_polys, n_quad, theta, R)
            F_xy = coeffs.get("F_xy", 0.0)
            F_x = coeffs.get("F_x", 0.0)
            F_y = coeffs.get("F_y", 0.0)

            # Reconstruct
            I1_reconstructed = (1/theta) * F_xy + F_y + F_x

            # Compute relative error
            rel_error = abs(I1_reconstructed - I1_canonical) / abs(I1_canonical) if I1_canonical != 0 else 0

            results.append({
                "pair_key": pair_key,
                "I1_canonical": I1_canonical,
                "I1_reconstructed": I1_reconstructed,
                "rel_error": rel_error,
            })

            # Each pair should reconstruct exactly
            assert rel_error < 1e-10, f"Pair {pair_key} failed reconstruction with error {rel_error:.2e}"

        # Print summary for debugging
        print("\n" + "="*70)
        print("RECONSTRUCTION GATE TEST SUMMARY")
        print("="*70)
        for r in results:
            status = "✓" if r["rel_error"] < 1e-10 else "✗"
            print(f"  {r['pair_key']}: canonical={r['I1_canonical']:+.6f}, "
                  f"reconstructed={r['I1_reconstructed']:+.6f}, "
                  f"error={r['rel_error']:.2e} {status}")
        print()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
