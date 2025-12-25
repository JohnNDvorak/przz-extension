"""
src/ratio_reversal_diagnostic.py
Diagnostic Script for Ratio Reversal Investigation

This script investigates WHY PRZZ's formula produces a ratio reversal:
- Naive ∫P² gives const ratio 1.71 (κ > κ*)
- PRZZ needs const ratio 0.94 (κ < κ*)

We test four candidate mechanisms:
1. Derivative term subtraction (I₁, I₃, I₄ subtract from I₂)
2. (1-u)^{ℓ₁+ℓ₂} weights
3. Case C kernels K_ω(u;R)
4. Ψ sign patterns

The goal: identify which mechanism(s) produce const_κ/const_κ* ≈ 0.94
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, Dict
from math import exp, factorial
from dataclasses import dataclass

# Import polynomial loaders
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.composition import PolyLike


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


@dataclass
class RatioAnalysis:
    """Results of ratio analysis for one mechanism."""
    mechanism: str
    kappa_value: float
    kappa_star_value: float
    ratio: float
    target_ratio: float
    error_pct: float

    def print_summary(self):
        print(f"\n{self.mechanism}:")
        print(f"  κ value:  {self.kappa_value:8.4f}")
        print(f"  κ* value: {self.kappa_star_value:8.4f}")
        print(f"  Ratio:    {self.ratio:8.4f} (target: {self.target_ratio:.4f})")
        print(f"  Error:    {self.error_pct:+7.2f}%")


# =============================================================================
# MECHANISM 1: Simple P² integrals (naive expectation)
# =============================================================================

def compute_naive_p_squared(polys: Dict[str, PolyLike], n_quad: int = 80) -> float:
    """
    Compute naive (1/θ) × Σ_{pairs} ∫P_i(u)·P_j(u)du

    This is what we'd expect if there were no derivative terms,
    no (1-u) weights, no Case C kernels.
    """
    u_nodes, u_weights = gauss_legendre_01(n_quad)

    P1 = polys["P1"]
    P2 = polys["P2"]
    P3 = polys["P3"]

    # Evaluate polynomials
    P1_u = P1.eval(u_nodes)
    P2_u = P2.eval(u_nodes)
    P3_u = P3.eval(u_nodes)

    # Cross-integrals
    I11 = np.sum(u_weights * P1_u * P1_u)
    I22 = np.sum(u_weights * P2_u * P2_u)
    I33 = np.sum(u_weights * P3_u * P3_u)
    I12 = np.sum(u_weights * P1_u * P2_u)
    I13 = np.sum(u_weights * P1_u * P3_u)
    I23 = np.sum(u_weights * P2_u * P3_u)

    # With symmetry factors and factorial normalization
    theta = 4.0 / 7.0
    total = (1.0 / theta) * (
        1 * I11 / (factorial(1) * factorial(1)) +
        1 * I22 / (factorial(2) * factorial(2)) +
        1 * I33 / (factorial(3) * factorial(3)) +
        2 * I12 / (factorial(1) * factorial(2)) +
        2 * I13 / (factorial(1) * factorial(3)) +
        2 * I23 / (factorial(2) * factorial(3))
    )

    return total


# =============================================================================
# MECHANISM 2: Derivative term contributions
# =============================================================================

def compute_derivative_contribution(
    pair: Tuple[int, int],
    polys: Dict[str, PolyLike],
    theta: float,
    n_quad: int = 80
) -> Dict[str, float]:
    """
    For a given pair (ℓ₁, ℓ₂), compute the RELATIVE contribution of
    derivative terms vs base I₂ term.

    Returns:
        {
            'I2': base integral value,
            'I1': derivative contribution (approx),
            'I3': derivative contribution (approx),
            'I4': derivative contribution (approx),
            'total_deriv': sum of I1+I3+I4,
            'ratio_deriv_to_I2': (I1+I3+I4) / I2
        }

    This is a ROUGH estimate - true derivatives need full oracle.
    We approximate using P' magnitudes.
    """
    u_nodes, u_weights = gauss_legendre_01(n_quad)

    ell1, ell2 = pair
    P_names = ["P1", "P2", "P3"]
    P1 = polys[P_names[ell1 - 1]]
    P2 = polys[P_names[ell2 - 1]]

    # I₂: base integral (no derivatives)
    P1_u = P1.eval(u_nodes)
    P2_u = P2.eval(u_nodes)
    I2 = np.sum(u_weights * P1_u * P2_u)

    # Derivative terms: use P' as proxy for derivative contribution
    P1_prime_u = P1.eval_deriv(u_nodes, 1)
    P2_prime_u = P2.eval_deriv(u_nodes, 1)

    # I₁ ~ ∫P'₁ P'₂ (mixed derivative)
    I1_approx = np.sum(u_weights * P1_prime_u * P2_prime_u)

    # I₃ ~ ∫P'₁ P₂ (single derivative)
    I3_approx = np.sum(u_weights * P1_prime_u * P2_u)

    # I₄ ~ ∫P₁ P'₂ (single derivative)
    I4_approx = np.sum(u_weights * P1_u * P2_prime_u)

    total_deriv = I1_approx + abs(I3_approx) + abs(I4_approx)

    return {
        'I2': I2,
        'I1_approx': I1_approx,
        'I3_approx': I3_approx,
        'I4_approx': I4_approx,
        'total_deriv': total_deriv,
        'ratio_deriv_to_I2': total_deriv / I2 if abs(I2) > 1e-12 else 0.0
    }


# =============================================================================
# MECHANISM 3: (1-u)^{ℓ₁+ℓ₂} weights
# =============================================================================

def compute_with_1_minus_u_weights(
    polys: Dict[str, PolyLike],
    n_quad: int = 80
) -> float:
    """
    Compute (1/θ) × Σ_{pairs} ∫(1-u)^{ℓ₁+ℓ₂} P_i(u)·P_j(u)du

    This includes the (1-u) suppression factor that appears in I₁, I₃, I₄.
    Higher pairs get suppressed more near u=1.
    """
    u_nodes, u_weights = gauss_legendre_01(n_quad)

    P1 = polys["P1"]
    P2 = polys["P2"]
    P3 = polys["P3"]

    # Evaluate polynomials
    P1_u = P1.eval(u_nodes)
    P2_u = P2.eval(u_nodes)
    P3_u = P3.eval(u_nodes)

    # (1-u) factor
    one_minus_u = 1.0 - u_nodes

    # Weighted cross-integrals with (1-u)^{ℓ₁+ℓ₂}
    # (1,1): (1-u)^2
    I11 = np.sum(u_weights * (one_minus_u**2) * P1_u * P1_u)

    # (2,2): (1-u)^4
    I22 = np.sum(u_weights * (one_minus_u**4) * P2_u * P2_u)

    # (3,3): (1-u)^6
    I33 = np.sum(u_weights * (one_minus_u**6) * P3_u * P3_u)

    # (1,2): (1-u)^3
    I12 = np.sum(u_weights * (one_minus_u**3) * P1_u * P2_u)

    # (1,3): (1-u)^4
    I13 = np.sum(u_weights * (one_minus_u**4) * P1_u * P3_u)

    # (2,3): (1-u)^5
    I23 = np.sum(u_weights * (one_minus_u**5) * P2_u * P3_u)

    # With symmetry factors and factorial normalization
    theta = 4.0 / 7.0
    total = (1.0 / theta) * (
        1 * I11 / (factorial(1) * factorial(1)) +
        1 * I22 / (factorial(2) * factorial(2)) +
        1 * I33 / (factorial(3) * factorial(3)) +
        2 * I12 / (factorial(1) * factorial(2)) +
        2 * I13 / (factorial(1) * factorial(3)) +
        2 * I23 / (factorial(2) * factorial(3))
    )

    return total


# =============================================================================
# MECHANISM 4: Per-pair derivative analysis
# =============================================================================

def analyze_derivative_ratios_by_pair(n_quad: int = 80):
    """
    Analyze how derivative terms differ between κ and κ* for each pair.

    This reveals which pairs are most affected by derivative subtraction.
    """
    theta = 4.0 / 7.0

    # Load polynomials
    polys_k = load_przz_polynomials(enforce_Q0=True)
    polys_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    P1_k, P2_k, P3_k, Q_k = polys_k
    P1_ks, P2_ks, P3_ks, Q_ks = polys_ks

    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    print("\n" + "=" * 70)
    print("DERIVATIVE TERM ANALYSIS BY PAIR")
    print("=" * 70)
    print("\nRatio of (I1+I3+I4)/I2 for each pair:")
    print(f"{'Pair':<8} {'κ ratio':>12} {'κ* ratio':>12} {'Diff':>12}")
    print("-" * 70)

    for pair in pairs:
        result_k = compute_derivative_contribution(pair, polys_kappa, theta, n_quad)
        result_ks = compute_derivative_contribution(pair, polys_kappa_star, theta, n_quad)

        ratio_k = result_k['ratio_deriv_to_I2']
        ratio_ks = result_ks['ratio_deriv_to_I2']
        diff = ratio_k - ratio_ks

        print(f"  {pair}    {ratio_k:12.4f} {ratio_ks:12.4f} {diff:+12.4f}")


# =============================================================================
# MAIN DIAGNOSTIC
# =============================================================================

def run_ratio_reversal_diagnostic():
    """
    Run the complete ratio reversal diagnostic.

    This tests each mechanism and identifies which produces the
    correct ratio.
    """
    print("=" * 70)
    print("RATIO REVERSAL DIAGNOSTIC")
    print("=" * 70)

    # Target values
    print("\nTarget const ratios:")
    print("  Naive ∫P²:     1.71 (κ > κ*) — WRONG direction")
    print("  PRZZ needs:    0.94 (κ < κ*) — CORRECT direction")
    print("  t-integral:    1.17 (nearly benchmark-independent)")
    print("  Combined:      1.17 × 0.94 = 1.10 ✓")

    # Load polynomials
    print("\nLoading polynomials...")
    polys_k = load_przz_polynomials(enforce_Q0=True)
    polys_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    P1_k, P2_k, P3_k, Q_k = polys_k
    P1_ks, P2_ks, P3_ks, Q_ks = polys_ks

    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    n_quad = 80
    target_ratio = 0.94

    results = []

    # ==========================================================================
    # TEST 1: Naive P² integrals
    # ==========================================================================
    print("\n" + "-" * 70)
    print("TEST 1: Naive ∫P_i·P_j (no derivatives, no weights)")
    print("-" * 70)

    naive_k = compute_naive_p_squared(polys_kappa, n_quad)
    naive_ks = compute_naive_p_squared(polys_kappa_star, n_quad)
    naive_ratio = naive_k / naive_ks

    result1 = RatioAnalysis(
        mechanism="Naive ∫P²",
        kappa_value=naive_k,
        kappa_star_value=naive_ks,
        ratio=naive_ratio,
        target_ratio=target_ratio,
        error_pct=100 * (naive_ratio - target_ratio) / target_ratio
    )
    result1.print_summary()
    results.append(result1)

    # ==========================================================================
    # TEST 2: With (1-u)^{ℓ₁+ℓ₂} weights
    # ==========================================================================
    print("\n" + "-" * 70)
    print("TEST 2: ∫(1-u)^{ℓ₁+ℓ₂} P_i·P_j (with suppression weights)")
    print("-" * 70)

    weighted_k = compute_with_1_minus_u_weights(polys_kappa, n_quad)
    weighted_ks = compute_with_1_minus_u_weights(polys_kappa_star, n_quad)
    weighted_ratio = weighted_k / weighted_ks

    result2 = RatioAnalysis(
        mechanism="With (1-u)^{ℓ₁+ℓ₂}",
        kappa_value=weighted_k,
        kappa_star_value=weighted_ks,
        ratio=weighted_ratio,
        target_ratio=target_ratio,
        error_pct=100 * (weighted_ratio - target_ratio) / target_ratio
    )
    result2.print_summary()
    results.append(result2)

    # ==========================================================================
    # TEST 3: Derivative term analysis
    # ==========================================================================
    print("\n" + "-" * 70)
    print("TEST 3: Derivative term contributions")
    print("-" * 70)

    analyze_derivative_ratios_by_pair(n_quad)

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: RATIO COMPARISON")
    print("=" * 70)

    print(f"\n{'Mechanism':<30} {'Ratio':>10} {'Target':>10} {'Error':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r.mechanism:<30} {r.ratio:10.4f} {r.target_ratio:10.4f} {r.error_pct:+9.1f}%")

    # Identify best match
    best = min(results, key=lambda r: abs(r.error_pct))

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if abs(best.error_pct) < 10:
        print(f"\n✓ {best.mechanism} MATCHES target ratio within 10%!")
        print(f"  Ratio: {best.ratio:.4f} (target: {best.target_ratio:.4f})")
        print(f"  Error: {best.error_pct:+.2f}%")
    else:
        print(f"\n✗ None of the tested mechanisms match the target ratio.")
        print(f"  Best: {best.mechanism} with {best.error_pct:+.2f}% error")
        print("\nPossible explanations:")
        print("  1. Need full derivative terms (not just approximations)")
        print("  2. Case C kernels have significant effect")
        print("  3. Ψ sign patterns create additional cancellations")
        print("  4. Combination of multiple mechanisms")


if __name__ == "__main__":
    run_ratio_reversal_diagnostic()
