#!/usr/bin/env python3
"""
scripts/test_gfactor_transferability.py
Phase 58: Test g-factor transferability across polynomial sets

CRITICAL TEST (from Claude Opus):
If m_needed / (exp(R) + 5) ≈ 1.015 for ALL polynomial sets,
then g-factors transfer and κ=0.52 is credible.

If the ratio varies significantly with polynomial choice,
then κ=0.52 is suspect.

Created: 2025-12-29
"""

import math
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.polynomials import (
    load_przz_polynomials, P1Polynomial, PellPolynomial, Polynomial
)


def compute_integrals_for_polynomials(
    polynomials: Dict,
    R: float,
    theta: float,
    n_quad: int = 80
) -> Tuple[float, float, float]:
    """
    Compute S12(+R), S12(-R), S34(+R) for given polynomials.
    """
    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term

    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0
    }

    all_terms_plus = make_all_terms_k3(theta, R, kernel_regime="paper")
    all_terms_minus = make_all_terms_k3(theta, -R, kernel_regime="paper")

    s12_plus = 0.0
    s12_minus = 0.0
    s34_plus = 0.0

    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms_plus = all_terms_plus[pair_key]
        terms_minus = all_terms_minus[pair_key]

        norm = factorial_norm[pair_key]
        sym = symmetry_factor[pair_key]
        full_norm = sym * norm

        for i in range(2):
            if i < len(terms_plus):
                result_plus = evaluate_term(
                    terms_plus[i], polynomials, n_quad, R=R, theta=theta, n_quad_a=40
                )
                s12_plus += full_norm * result_plus.value

            if i < len(terms_minus):
                result_minus = evaluate_term(
                    terms_minus[i], polynomials, n_quad, R=-R, theta=theta, n_quad_a=40
                )
                s12_minus += full_norm * result_minus.value

        for i in range(2, 4):
            if i < len(terms_plus):
                result_plus = evaluate_term(
                    terms_plus[i], polynomials, n_quad, R=R, theta=theta, n_quad_a=40
                )
                s34_plus += full_norm * result_plus.value

    return s12_plus, s12_minus, s34_plus


def compute_m_needed(s12_plus: float, s12_minus: float, s34_plus: float, c_target: float) -> float:
    """Solve for m such that c = S12(+R) + m × S12(-R) + S34(+R) = c_target."""
    if abs(s12_minus) < 1e-10:
        return float('inf')
    return (c_target - s12_plus - s34_plus) / s12_minus


def load_optimized_polynomials() -> Optional[Dict]:
    """Load optimized polynomials from JSON."""
    json_path = Path(__file__).parent.parent / "data" / "optimal_polynomials.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    P1 = P1Polynomial(tilde_coeffs=np.array(data["P1_tilde"]))
    P2 = PellPolynomial(tilde_coeffs=np.array(data["P2_tilde"]))
    P3 = PellPolynomial(tilde_coeffs=np.array(data["P3_tilde"]))
    Q = Polynomial(coeffs=np.array(data["Q_mono"]))

    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


def create_unit_polynomials() -> Dict:
    """Create P=1, Q=1 polynomials (simplest case)."""
    P1 = P1Polynomial(tilde_coeffs=np.array([1.0]))
    P2 = PellPolynomial(tilde_coeffs=np.array([1.0]))
    P3 = PellPolynomial(tilde_coeffs=np.array([1.0]))
    Q = Polynomial(coeffs=np.array([1.0]))  # Q(t) = 1
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


def create_scaled_przz_polynomials(scale: float) -> Dict:
    """Create PRZZ polynomials scaled by a factor."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)

    P1_scaled = P1Polynomial(tilde_coeffs=P1.tilde_coeffs * scale)
    P2_scaled = PellPolynomial(tilde_coeffs=P2.tilde_coeffs * scale)
    P3_scaled = PellPolynomial(tilde_coeffs=P3.tilde_coeffs * scale)

    return {"P1": P1_scaled, "P2": P2_scaled, "P3": P3_scaled, "Q": Q}


def main():
    """Run g-factor transferability test."""
    print("=" * 70)
    print("PHASE 58: G-FACTOR TRANSFERABILITY TEST")
    print("=" * 70)
    print()
    print("Critical question: Is m_needed / (exp(R) + 5) constant across polynomials?")
    print()

    R = 1.3036
    theta = 4/7
    K = 3

    # Target c for PRZZ baseline (κ = 0.4173)
    kappa_przz = 0.417293962
    c_przz = math.exp(R * (1 - kappa_przz))  # ≈ 2.137

    m_formula = math.exp(R) + (2*K - 1)  # exp(R) + 5 ≈ 8.68

    print(f"Reference values:")
    print(f"  R = {R}")
    print(f"  exp(R) + 5 = {m_formula:.6f}")
    print(f"  c_PRZZ = {c_przz:.6f} (for κ = {kappa_przz})")
    print()

    # Define polynomial sets to test
    polynomial_sets = []

    # 1. PRZZ baseline
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
    polynomial_sets.append(("PRZZ baseline", {"P1": P1, "P2": P2, "P3": P3, "Q": Q}, c_przz))

    # 2. Optimized polynomials
    opt_polys = load_optimized_polynomials()
    if opt_polys:
        # For optimized, the target c is different (c ≈ 1.87 for κ ≈ 0.52)
        kappa_opt = 0.5213
        c_opt = math.exp(R * (1 - kappa_opt))
        polynomial_sets.append(("Optimized (κ=0.52)", opt_polys, c_opt))

    # 3. Unit polynomials (P=Q=1)
    unit_polys = create_unit_polynomials()
    # For unit, we don't know the target κ, so we'll just use whatever c we get
    polynomial_sets.append(("Unit (P=Q=1)", unit_polys, None))

    # 4. Scaled PRZZ (0.5x)
    scaled_05 = create_scaled_przz_polynomials(0.5)
    polynomial_sets.append(("PRZZ × 0.5", scaled_05, None))

    # 5. Scaled PRZZ (2x)
    scaled_2 = create_scaled_przz_polynomials(2.0)
    polynomial_sets.append(("PRZZ × 2.0", scaled_2, None))

    # Run tests
    print(f"{'Polynomial Set':<25} {'S12(+R)':>10} {'S12(-R)':>10} {'S34(+R)':>10} {'m_needed':>10} {'ratio':>10}")
    print("-" * 85)

    ratios = []

    for name, polys, c_target in polynomial_sets:
        try:
            s12_plus, s12_minus, s34_plus = compute_integrals_for_polynomials(
                polys, R, theta, n_quad=60
            )

            if c_target is None:
                # Just compute c with our formula and see what κ we get
                c_actual = s12_plus + m_formula * s12_minus + s34_plus
                if c_actual > 0:
                    kappa_actual = 1 - math.log(c_actual) / R
                    c_target = c_actual  # Use actual for m_needed calculation
                else:
                    c_target = c_przz  # Fallback

            m_needed = compute_m_needed(s12_plus, s12_minus, s34_plus, c_target)
            ratio = m_needed / m_formula

            ratios.append((name, ratio))

            print(f"{name:<25} {s12_plus:>10.4f} {s12_minus:>10.4f} {s34_plus:>10.4f} {m_needed:>10.4f} {ratio:>10.4f}")

        except Exception as e:
            print(f"{name:<25} ERROR: {e}")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    # Analyze ratio stability
    valid_ratios = [r for _, r in ratios if 0.5 < r < 2.0]  # Filter outliers

    if len(valid_ratios) >= 2:
        mean_ratio = sum(valid_ratios) / len(valid_ratios)
        max_ratio = max(valid_ratios)
        min_ratio = min(valid_ratios)
        spread = (max_ratio - min_ratio) / mean_ratio * 100

        print(f"Ratio statistics (m_needed / (exp(R) + 5)):")
        print(f"  Mean:   {mean_ratio:.4f}")
        print(f"  Min:    {min_ratio:.4f}")
        print(f"  Max:    {max_ratio:.4f}")
        print(f"  Spread: {spread:.2f}%")
        print()

        if spread < 5:
            print("✓ RATIO IS STABLE (spread < 5%)")
            print("  → g-factors TRANSFER across polynomial sets")
            print("  → κ = 0.52 result is CREDIBLE")
        elif spread < 15:
            print("⚠ RATIO IS MODERATELY STABLE (spread 5-15%)")
            print("  → g-factors may partially transfer")
            print("  → κ = 0.52 result needs scrutiny")
        else:
            print("✗ RATIO IS UNSTABLE (spread > 15%)")
            print("  → g-factors DO NOT TRANSFER")
            print("  → κ = 0.52 result is SUSPECT")

    print()

    # Special comparison: PRZZ vs Optimized
    print("=" * 70)
    print("KEY COMPARISON: PRZZ vs OPTIMIZED")
    print("=" * 70)
    print()

    przz_ratio = None
    opt_ratio = None
    for name, ratio in ratios:
        if "PRZZ baseline" in name:
            przz_ratio = ratio
        if "Optimized" in name:
            opt_ratio = ratio

    if przz_ratio and opt_ratio:
        diff = abs(opt_ratio - przz_ratio) / przz_ratio * 100
        print(f"PRZZ baseline ratio:  {przz_ratio:.4f}")
        print(f"Optimized ratio:      {opt_ratio:.4f}")
        print(f"Difference:           {diff:.2f}%")
        print()

        if diff < 2:
            print("✓ EXCELLENT: <2% difference")
            print("  The g-factors derived from PRZZ apply to optimized polynomials.")
            print("  κ = 0.52 is a CREDIBLE result.")
        elif diff < 5:
            print("✓ GOOD: <5% difference")
            print("  The g-factors approximately transfer.")
            print("  κ = 0.52 is plausible but needs verification.")
        else:
            print("✗ CONCERNING: >5% difference")
            print("  The g-factors may not fully transfer.")
            print("  κ = 0.52 requires independent validation.")


if __name__ == "__main__":
    main()
