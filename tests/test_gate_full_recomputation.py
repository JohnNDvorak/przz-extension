#!/usr/bin/env python3
"""
Gate 1: Full Recomputation, No Approximations

GPT's first validation gate for the κ = 0.5213 claim:
- Re-score candidate with full KappaEngine evaluator (no fast mode)
- Match c to at least 1e-8 across runs
- Verify against stored NOLH point 17 values

Created: 2025-12-28 (GPT Critical Review)
"""

import json
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kappa_engine import KappaEngine


def load_optimal_polynomials():
    """Load the validated optimal polynomials."""
    path = Path(__file__).parent.parent / "data" / "optimal_polynomials.json"
    with open(path) as f:
        return json.load(f)


class TestFullRecomputationReproducibility:
    """Test that full KappaEngine gives identical results across runs."""

    def test_five_independent_runs_match(self):
        """Run full KappaEngine 5 times, verify c matches to 1e-8."""
        data = load_optimal_polynomials()

        R = 1.3036
        theta = 4/7
        n_quad = 60  # Use higher than NOLH's 40 for validation

        c_values = []
        kappa_values = []

        for i in range(5):
            # Fresh engine instance each time
            engine = KappaEngine(
                P1_coeffs=data['P1_tilde'],
                P2_coeffs=data['P2_tilde'],
                P3_coeffs=data['P3_tilde'],
                Q_coeffs=data['Q_mono'],
                theta=theta,
                K=3,
                R=R,
                n_quad=n_quad,
            )
            result = engine.compute_kappa()
            c_values.append(result.c)
            kappa_values.append(result.kappa)

        c_values = np.array(c_values)
        kappa_values = np.array(kappa_values)

        # All c values should match to 1e-8 relative tolerance
        c_mean = np.mean(c_values)
        c_max_deviation = np.max(np.abs(c_values - c_mean))
        rel_deviation = c_max_deviation / c_mean

        print(f"\n  Full Recomputation (5 runs, n_quad={n_quad}):")
        print(f"  c values: {c_values}")
        print(f"  c_mean: {c_mean:.10f}")
        print(f"  c_max_deviation: {c_max_deviation:.2e}")
        print(f"  rel_deviation: {rel_deviation:.2e}")

        assert rel_deviation < 1e-8, f"c values differ by {rel_deviation:.2e}"

        # Also check kappa
        kappa_mean = np.mean(kappa_values)
        kappa_max_deviation = np.max(np.abs(kappa_values - kappa_mean))

        print(f"  κ_mean: {kappa_mean:.10f}")
        print(f"  κ_max_deviation: {kappa_max_deviation:.2e}")

        assert kappa_max_deviation < 1e-8, f"κ values differ by {kappa_max_deviation:.2e}"

    def test_full_vs_nolh_candidate_match(self):
        """Compare full KappaEngine result against stored NOLH point 17."""
        data = load_optimal_polynomials()

        R = 1.3036
        theta = 4/7

        # Use n_quad=40 to exactly match NOLH evaluation
        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=3,
            R=R,
            n_quad=40,
        )
        result = engine.compute_kappa()

        # Stored values from NOLH optimization
        stored_c = data['kappa_benchmark']['c']
        stored_kappa = data['kappa_benchmark']['kappa']

        c_rel_diff = abs(result.c - stored_c) / stored_c
        kappa_abs_diff = abs(result.kappa - stored_kappa)

        print(f"\n  Full vs NOLH Candidate (n_quad=40):")
        print(f"  Computed c: {result.c:.10f}")
        print(f"  Stored c:   {stored_c:.10f}")
        print(f"  c rel_diff: {c_rel_diff:.2e}")
        print(f"  Computed κ: {result.kappa:.10f}")
        print(f"  Stored κ:   {stored_kappa:.10f}")
        print(f"  κ abs_diff: {kappa_abs_diff:.2e}")

        # Should match within float precision
        assert c_rel_diff < 1e-6, f"c mismatch: {c_rel_diff:.2e}"
        assert kappa_abs_diff < 1e-4, f"κ mismatch: {kappa_abs_diff:.2e}"


class TestQuadratureConvergence:
    """Test convergence across quadrature refinement."""

    def test_quadrature_convergence_table(self):
        """Generate n_quad=40/60/80/100 convergence table."""
        data = load_optimal_polynomials()

        R = 1.3036
        theta = 4/7

        n_values = [40, 60, 80, 100]
        results = []

        print(f"\n  Quadrature Convergence Table:")
        print(f"  {'n_quad':>8} | {'c':>14} | {'κ':>12} | {'Δc':>12}")
        print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}")

        prev_c = None
        for n in n_values:
            engine = KappaEngine(
                P1_coeffs=data['P1_tilde'],
                P2_coeffs=data['P2_tilde'],
                P3_coeffs=data['P3_tilde'],
                Q_coeffs=data['Q_mono'],
                theta=theta,
                K=3,
                R=R,
                n_quad=n,
            )
            result = engine.compute_kappa()

            delta_c = 0 if prev_c is None else abs(result.c - prev_c)
            results.append({'n': n, 'c': result.c, 'kappa': result.kappa, 'delta_c': delta_c})

            print(f"  {n:>8} | {result.c:>14.10f} | {result.kappa:>12.10f} | {delta_c:>12.2e}")
            prev_c = result.c

        # Check convergence: delta between n=80 and n=100 should be small
        delta_80_100 = results[3]['delta_c']
        assert delta_80_100 < 1e-6, f"Not converged: Δc(80→100) = {delta_80_100:.2e}"

        # All kappa values should be > 0.5 (the claim)
        for r in results:
            assert r['kappa'] > 0.5, f"κ < 0.5 at n_quad={r['n']}: {r['kappa']}"


class TestDecompositionMatch:
    """Test that decomposition components match stored values."""

    def test_decomposition_components_match(self):
        """Verify S12(+R), S12(-R), S34(+R), m match stored values."""
        data = load_optimal_polynomials()

        R = 1.3036
        theta = 4/7

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=3,
            R=R,
            n_quad=60,
        )
        result = engine.compute_kappa()

        stored = data['decomposition']

        print(f"\n  Decomposition Component Match:")
        print(f"  {'Component':>12} | {'Computed':>12} | {'Stored':>12} | {'Rel Diff':>12}")
        print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

        components = [
            ('S12(+R)', result.integrals.S12_plus, stored['S12_plus']),
            ('S12(-R)', result.integrals.S12_minus, stored['S12_minus']),
            ('S34(+R)', result.integrals.S34_plus, stored['S34_plus']),
            ('m', result.corrections.m, stored['m']),
        ]

        for name, computed, stored_val in components:
            rel_diff = abs(computed - stored_val) / (abs(stored_val) + 1e-15)
            print(f"  {name:>12} | {computed:>12.6f} | {stored_val:>12.6f} | {rel_diff:>12.2e}")
            assert rel_diff < 0.01, f"{name} mismatch: {rel_diff:.2e}"


class TestGate1Summary:
    """Comprehensive Gate 1 summary."""

    def test_full_gate1_summary(self):
        """Run full Gate 1 summary with pass/fail status."""
        data = load_optimal_polynomials()

        print("\n" + "=" * 70)
        print("GATE 1: FULL RECOMPUTATION (GPT Critical Review)")
        print("=" * 70)

        R = 1.3036
        theta = 4/7

        all_passed = True

        # Test 1: Reproducibility
        c_values = []
        for i in range(5):
            engine = KappaEngine(
                P1_coeffs=data['P1_tilde'],
                P2_coeffs=data['P2_tilde'],
                P3_coeffs=data['P3_tilde'],
                Q_coeffs=data['Q_mono'],
                theta=theta,
                K=3,
                R=R,
                n_quad=60,
            )
            result = engine.compute_kappa()
            c_values.append(result.c)

        c_values = np.array(c_values)
        c_mean = np.mean(c_values)
        c_max_dev = np.max(np.abs(c_values - c_mean))
        rel_dev = c_max_dev / c_mean

        test1_pass = rel_dev < 1e-8
        status1 = "PASS" if test1_pass else "FAIL"
        print(f"\n  Reproducibility (5 runs): {status1}")
        print(f"    c_mean = {c_mean:.10f}")
        print(f"    rel_deviation = {rel_dev:.2e}")
        all_passed &= test1_pass

        # Test 2: Match stored values
        stored_c = data['kappa_benchmark']['c']
        c_match_diff = abs(c_mean - stored_c) / stored_c
        test2_pass = c_match_diff < 1e-4
        status2 = "PASS" if test2_pass else "FAIL"
        print(f"\n  Match stored NOLH: {status2}")
        print(f"    Computed c = {c_mean:.10f}")
        print(f"    Stored c   = {stored_c:.10f}")
        print(f"    rel_diff   = {c_match_diff:.2e}")
        all_passed &= test2_pass

        # Test 3: κ > 0.5
        kappa = result.kappa
        test3_pass = kappa > 0.5
        status3 = "PASS" if test3_pass else "FAIL"
        print(f"\n  κ > 0.5 claim: {status3}")
        print(f"    Computed κ = {kappa:.10f}")
        all_passed &= test3_pass

        print("\n" + "=" * 70)
        overall = "PASS" if all_passed else "FAIL"
        print(f"GATE 1 OVERALL: {overall}")
        print("=" * 70)

        assert all_passed, "Gate 1 failed"


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GATE 1: FULL RECOMPUTATION - Quick Check")
    print("=" * 70)

    data = load_optimal_polynomials()

    R = 1.3036
    theta = 4/7

    # Run 3 times
    for i in range(3):
        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=3,
            R=R,
            n_quad=60,
        )
        result = engine.compute_kappa()
        print(f"  Run {i+1}: c = {result.c:.10f}, κ = {result.kappa:.10f}")
