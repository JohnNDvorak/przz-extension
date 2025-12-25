"""
run_analytic_gate_test.py
Analytic Gate: Verify DSL with P=Q=1 (constant polynomials)

GPT Recommendation (2025-12-19)
-------------------------------
With P₁=P₂=P₃=1 and Q=1:
- All derivatives vanish → I₁, I₃, I₄ = 0
- Only I₂ survives (no derivatives taken)

For I₂ with P=Q=1:
    I₂ = ∫∫ Q(t)² exp(2Rt) P₁(u)P₂(u) du dt
       = ∫₀¹ exp(2Rt) dt × ∫₀¹ 1 du
       = (exp(2R) - 1) / (2R)

This provides a non-benchmark oracle to verify integration correctness.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from src.evaluate import evaluate_term, evaluate_c_full
from src.polynomials import Polynomial
from src.terms_k3_d1 import make_all_terms_11_v2, make_all_terms_22_v2, make_all_terms_33_v2

# Constants
THETA = 4.0 / 7.0
N_QUAD = 60
N_QUAD_A = 40


def make_constant_polynomials() -> Dict:
    """Create P₁=P₂=P₃=Q=1 (all constant)."""
    P_one = Polynomial([1.0])  # P(x) = 1
    Q_one = Polynomial([1.0])  # Q(t) = 1
    return {
        "P1": P_one,
        "P2": P_one,
        "P3": P_one,
        "Q": Q_one,
    }


def analytic_I2(R: float, theta: float = THETA) -> float:
    """Analytically compute I₂ for P=Q=1.

    I₂ = (1/θ) × ∫₀¹ dt ∫₀¹ du  Q(t)² exp(2Rt) P(u)²
       = (1/θ) × ∫₀¹ exp(2Rt) dt × ∫₀¹ 1 du
       = (1/θ) × (exp(2R) - 1) / (2R)

    Note: The 1/θ prefactor comes from the PRZZ formula structure.
    """
    if abs(R) < 1e-10:
        return 1.0 / theta  # Limit as R → 0
    return (math.exp(2 * R) - 1) / (2 * R * theta)


def run_pair_test(pair_name: str, terms_func, polynomials: Dict, R: float):
    """Test a single pair with P=Q=1."""
    terms = terms_func(THETA, R)

    print(f"\n{pair_name} pair (P=Q=1):")
    print("-" * 50)

    total = 0.0
    for i, term in enumerate(terms):
        try:
            result = evaluate_term(
                term, polynomials, N_QUAD, R=R, theta=THETA, n_quad_a=N_QUAD_A
            )
            val = result.value
            print(f"  I{i+1}: {val:+.10f}")
            total += val
        except Exception as e:
            print(f"  I{i+1}: ERROR - {e}")

    print(f"  Total: {total:+.10f}")
    return total


def main():
    print("=" * 78)
    print("ANALYTIC GATE: P=Q=1 (Constant Polynomials)")
    print("=" * 78)
    print()
    print("With P₁=P₂=P₃=Q=1:")
    print("  - All derivatives vanish → I₁, I₃, I₄ should be ~0")
    print("  - Only I₂ survives (no derivatives)")
    print("  - Analytic I₂ = (exp(2R) - 1) / (2R)")
    print()

    polynomials = make_constant_polynomials()

    # Test at κ benchmark R
    R = 1.3036
    analytic = analytic_I2(R)

    print(f"R = {R}")
    print(f"Analytic I₂ = (exp(2×{R}) - 1) / (2×{R}) = {analytic:.10f}")
    print()

    # Test each diagonal pair
    print("-" * 78)
    print("Diagonal pairs (should give ~analytic_I₂ for I₂, ~0 for others)")
    print("-" * 78)

    pairs = [
        ("11", make_all_terms_11_v2),
        ("22", make_all_terms_22_v2),
        ("33", make_all_terms_33_v2),
    ]

    for pair_name, terms_func in pairs:
        run_pair_test(pair_name, terms_func, polynomials, R)

    # Compute full c
    print()
    print("=" * 78)
    print("Full c computation with P=Q=1")
    print("=" * 78)

    try:
        result = evaluate_c_full(
            theta=THETA,
            R=R,
            n=N_QUAD,
            polynomials=polynomials,
            n_quad_a=N_QUAD_A,
            kernel_regime="raw",  # No Case C transform for P=1
        )
        print(f"\nc_computed (raw regime): {result.total:+.10f}")

    except Exception as e:
        print(f"ERROR: {e}")

    # Also test the paper regime for comparison
    print()
    try:
        result_paper = evaluate_c_full(
            theta=THETA,
            R=R,
            n=N_QUAD,
            polynomials=polynomials,
            n_quad_a=N_QUAD_A,
            kernel_regime="paper",
        )
        print(f"c_computed (paper regime): {result_paper.total:+.10f}")
    except Exception as e:
        print(f"Paper regime ERROR: {e}")

    print()
    print("=" * 78)
    print("ANALYSIS")
    print("=" * 78)
    print()
    print("For P=Q=1:")
    print(f"  Analytic I₂ = (1/θ) × (exp(2R)-1)/(2R) = {analytic:.6f}")
    print()
    print("NOTE: I₁, I₃, I₄ are NOT necessarily zero with P=1 because:")
    print("  - Exp factors exp(Rx/θ), exp(Ry/θ) still depend on x, y")
    print("  - Algebraic prefactors (θ(x+y)+1)/θ depend on x, y")
    print("  - Only the P(x+u)P(y+u) part becomes constant")
    print()
    print("KEY CHECK: I₂ should match analytic value within numerical precision")
    print(f"  Expected I₂: {analytic:.6f}")
    print(f"  Ratio check: I₂_computed / I₂_analytic should be ~1.0")


if __name__ == "__main__":
    main()
