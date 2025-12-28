#!/usr/bin/env python3
"""
Phase 45.2: Out-of-Sample Stability Test

Tests that the anchored constants (g_I1, g_I2) behave sanely on polynomial sets
beyond the κ and κ* benchmarks used to calibrate them.

TEST SUITE:
1. Q=1 microcases (should give correction near baseline)
2. "κ-like" Q (real κ polynomial)
3. "κ*-like" Q (real κ* polynomial - lower degree)
4. Randomized Q under boundary constraints

STABILITY GATE:
- Correction should remain small and stable (0.1–0.5%)
- No blow-up on edge cases (< 5%)

Created: 2025-12-27 (Phase 45.2)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from src.polynomials import Polynomial, load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.g_functional import compute_I1_I2_totals
from src.evaluator.correction_policy import (
    CorrectionMode,
    get_g_correction,
    compute_g_baseline,
    compute_g_anchored,
    G_I1_CALIBRATED,
    G_I2_CALIBRATED,
)


@dataclass
class StabilityTestResult:
    """Result of a stability test."""
    name: str
    R: float
    f_I1: float
    g_total: float
    g_baseline: float
    correction_pct: float  # (g_total / g_baseline - 1) * 100
    passed: bool
    threshold: float


def create_q_one_polynomial() -> Polynomial:
    """Create Q(x) = 1 (constant polynomial)."""
    return Polynomial(coeffs=np.array([1.0]))


def create_random_q_polynomial(seed: int = 42, degree: int = 3) -> Polynomial:
    """
    Create a random Q polynomial satisfying Q(0)=1, Q(1)=-1.

    Uses random coefficients for x, x^2, ... x^(degree) terms,
    then adjusts constant and sum to satisfy boundary conditions.
    """
    rng = np.random.RandomState(seed)

    # Generate random interior coefficients
    interior = rng.uniform(-0.5, 0.5, size=degree)

    # Q(x) = c0 + c1*x + c2*x^2 + ... + cn*x^n
    # Q(0) = c0 = 1
    # Q(1) = c0 + c1 + c2 + ... + cn = -1
    # => c1 + c2 + ... + cn = -2

    # Set c0 = 1
    c0 = 1.0

    # Adjust last coefficient to satisfy Q(1) = -1
    # sum(interior) + adjust = -2
    adjust = -2.0 - np.sum(interior[:-1])
    interior[-1] = adjust

    coeffs = np.concatenate([[c0], interior])
    return Polynomial(coeffs=coeffs)


def compute_f_I1(R: float, theta: float, polynomials: dict, n_quad: int = 60) -> float:
    """Compute the I1 fraction f_I1 = I1(-R) / (I1(-R) + I2(-R))."""
    I1_minus, I2_minus = compute_I1_I2_totals(-R, theta, polynomials, n_quad)
    S12_minus = I1_minus + I2_minus
    if abs(S12_minus) < 1e-15:
        return 0.5  # Avoid division by zero
    return I1_minus / S12_minus


def run_stability_test(
    name: str,
    polynomials: dict,
    R: float,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
    threshold: float = 5.0,  # Max allowed deviation %
) -> StabilityTestResult:
    """
    Run a stability test for a given polynomial set.

    Returns:
        StabilityTestResult with pass/fail status
    """
    # Compute f_I1 for this polynomial set
    f_I1 = compute_f_I1(R, theta, polynomials, n_quad)

    # Compute g using anchored mode
    g_total = compute_g_anchored(f_I1)

    # Get baseline for comparison
    g_baseline = compute_g_baseline(theta, K)

    # Compute correction percentage
    correction_pct = (g_total / g_baseline - 1) * 100

    # Check if within threshold
    passed = abs(correction_pct) <= threshold

    return StabilityTestResult(
        name=name,
        R=R,
        f_I1=f_I1,
        g_total=g_total,
        g_baseline=g_baseline,
        correction_pct=correction_pct,
        passed=passed,
        threshold=threshold,
    )


def create_q_modified_polynomials(base_polynomials: dict, Q: Polynomial) -> dict:
    """Create a polynomial dict with a modified Q polynomial."""
    return {
        "P1": base_polynomials["P1"],
        "P2": base_polynomials["P2"],
        "P3": base_polynomials["P3"],
        "Q": Q,
    }


def main():
    print()
    print("=" * 80)
    print("  PHASE 45.2: OUT-OF-SAMPLE STABILITY TEST")
    print("=" * 80)
    print()
    print("Testing that anchored constants (g_I1, g_I2) behave sanely")
    print("on polynomial sets beyond the κ/κ* benchmarks.")
    print()
    print(f"Calibrated constants:")
    print(f"  g_I1 = {G_I1_CALIBRATED:.8f}")
    print(f"  g_I2 = {G_I2_CALIBRATED:.8f}")
    print()

    # Load base polynomial sets
    P1, P2, P3, Q_kappa = load_przz_polynomials()
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q_kappa}

    P1s, P2s, P3s, Q_kappa_star = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Q_kappa_star}

    # Create synthetic polynomial sets
    Q_one = create_q_one_polynomial()

    theta = 4 / 7
    K = 3
    g_baseline = compute_g_baseline(theta, K)

    results: List[StabilityTestResult] = []

    # -------------------------------------------------------------------------
    # Test Suite 1: Q=1 microcases
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("Test Suite 1: Q=1 Microcases")
    print("  Expected: correction should be near 0% (baseline is exact for Q=1)")
    print("-" * 80)

    for name, base_polys, R in [
        ("κ polys, Q=1, R=1.3036", polys_kappa, 1.3036),
        ("κ* polys, Q=1, R=1.1167", polys_kappa_star, 1.1167),
        ("κ polys, Q=1, R=1.0", polys_kappa, 1.0),
        ("κ polys, Q=1, R=1.5", polys_kappa, 1.5),
    ]:
        modified = create_q_modified_polynomials(base_polys, Q_one)
        result = run_stability_test(name, modified, R, theta, K, threshold=0.5)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         f_I1={result.f_I1:.4f}, g={result.g_total:.6f}, correction={result.correction_pct:+.4f}%")

    print()

    # -------------------------------------------------------------------------
    # Test Suite 2: Real Q polynomials (κ and κ*)
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("Test Suite 2: Real Q Polynomials (κ and κ*)")
    print("  Expected: correction should be within ±0.5% (what we calibrated for)")
    print("-" * 80)

    for name, polys, R in [
        ("κ benchmark", polys_kappa, 1.3036),
        ("κ* benchmark", polys_kappa_star, 1.1167),
    ]:
        result = run_stability_test(name, polys, R, theta, K, threshold=0.5)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         f_I1={result.f_I1:.4f}, g={result.g_total:.6f}, correction={result.correction_pct:+.4f}%")

    print()

    # -------------------------------------------------------------------------
    # Test Suite 3: Cross-matched R values
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("Test Suite 3: Cross-Matched R Values")
    print("  Testing κ polys at κ* R value and vice versa")
    print("-" * 80)

    for name, polys, R in [
        ("κ polys at R=1.1167", polys_kappa, 1.1167),
        ("κ* polys at R=1.3036", polys_kappa_star, 1.3036),
    ]:
        result = run_stability_test(name, polys, R, theta, K, threshold=1.0)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         f_I1={result.f_I1:.4f}, g={result.g_total:.6f}, correction={result.correction_pct:+.4f}%")

    print()

    # -------------------------------------------------------------------------
    # Test Suite 4: Random Q polynomials
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("Test Suite 4: Random Q Polynomials")
    print("  Testing stability with random Q satisfying Q(0)=1, Q(1)=-1")
    print("-" * 80)

    for seed in [42, 123, 456, 789]:
        Q_random = create_random_q_polynomial(seed=seed, degree=3)
        modified = create_q_modified_polynomials(polys_kappa, Q_random)
        name = f"Random Q (seed={seed})"
        result = run_stability_test(name, modified, 1.3036, theta, K, threshold=5.0)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         f_I1={result.f_I1:.4f}, g={result.g_total:.6f}, correction={result.correction_pct:+.4f}%")

    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print()

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    print(f"Tests passed: {passed_count}/{total_count}")
    print()

    # Statistics
    corrections = [r.correction_pct for r in results]
    print(f"Correction statistics:")
    print(f"  Min: {min(corrections):+.4f}%")
    print(f"  Max: {max(corrections):+.4f}%")
    print(f"  Mean: {np.mean(corrections):+.4f}%")
    print(f"  Std: {np.std(corrections):.4f}%")
    print()

    if passed_count == total_count:
        print("STATUS: ALL TESTS PASSED")
        print("  The anchored constants are stable on out-of-sample polynomials.")
        return True
    else:
        print("STATUS: SOME TESTS FAILED")
        print("  Review the failed tests above.")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.correction_pct:+.4f}% (threshold: ±{r.threshold}%)")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
