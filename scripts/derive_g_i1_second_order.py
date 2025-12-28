#!/usr/bin/env python3
"""
scripts/derive_g_i1_second_order.py
Explore second-order corrections for g_I1

The θ(2-θ) formula gives excellent g_I2 (within 0.0015% of calibrated).
But g_I1 = 1.0 has a 0.09% gap from calibrated g_I1 = 1.00091.

This script explores whether g_I1 also has a second-order correction.

OBSERVATION:
The gap ratio is:
  epsilon_I1 / epsilon_I2 = 0.00091 / 0.00584 ≈ 0.156

where:
  epsilon_I2 = g_I2_calibrated - g_baseline = 1.01945 - 1.01361 = 0.00584

This ratio might have structure involving θ, K, or f_I1.

Created: 2025-12-27
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def explore_g_i1_formulas():
    print("=" * 70)
    print("EXPLORING SECOND-ORDER CORRECTIONS FOR g_I1")
    print("=" * 70)

    theta = 4/7
    K = 3
    f_I1_kappa = 0.233
    f_I1_star = 0.326

    # Known values
    g_I1_calibrated = 1.00091428
    g_I2_calibrated = 1.01945154
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    epsilon_I1 = g_I1_calibrated - 1.0
    epsilon_I2 = g_I2_calibrated - g_baseline
    beta = theta / (2 * K * (2 * K + 1))

    print(f"\nKnown values:")
    print(f"  theta = {theta:.10f}")
    print(f"  K = {K}")
    print(f"  g_baseline = {g_baseline:.10f}")
    print(f"  beta = θ/(2K(2K+1)) = {beta:.10f}")
    print(f"")
    print(f"  g_I1_calibrated = {g_I1_calibrated:.10f}")
    print(f"  g_I2_calibrated = {g_I2_calibrated:.10f}")
    print(f"")
    print(f"  epsilon_I1 = g_I1 - 1.0 = {epsilon_I1:.10f}")
    print(f"  epsilon_I2 = g_I2 - g_baseline = {epsilon_I2:.10f}")

    # Key ratios
    print(f"\nKey ratios:")
    print(f"  epsilon_I1 / epsilon_I2 = {epsilon_I1 / epsilon_I2:.6f}")
    print(f"  epsilon_I1 / beta = {epsilon_I1 / beta:.6f}")
    print(f"  epsilon_I2 / beta = {epsilon_I2 / beta:.6f} (this is 1-θ = {1-theta:.6f})")

    # The (2-θ) factor for g_I2 came from:
    # epsilon_I2 / beta = 1 - θ = 3/7
    # So: epsilon_I2 = beta × (1-θ) = θ(1-θ)/(2K(2K+1)) = θ(2-θ)/(2K(2K+1)) - θ/(2K(2K+1))

    print(f"\n" + "=" * 70)
    print("HYPOTHESIS EXPLORATION")
    print("=" * 70)

    # The key insight: epsilon_I2 = beta × (1-θ)
    # Maybe epsilon_I1 has a similar structure?

    # Hypothesis A: epsilon_I1 = beta × α for some constant α
    alpha_needed = epsilon_I1 / beta
    print(f"\nHypothesis A: epsilon_I1 = beta × α")
    print(f"  α needed = {alpha_needed:.6f}")
    print(f"  Note: 1-θ = {1-theta:.6f}")
    print(f"  Ratio α/(1-θ) = {alpha_needed / (1-theta):.6f}")

    # Hypothesis B: epsilon_I1 = f_I1 × epsilon_I2 × some_factor
    factor_kappa = epsilon_I1 / (f_I1_kappa * epsilon_I2)
    factor_star = epsilon_I1 / (f_I1_star * epsilon_I2)
    print(f"\nHypothesis B: epsilon_I1 = f_I1 × epsilon_I2 × factor")
    print(f"  For f_I1 = {f_I1_kappa} (kappa): factor = {factor_kappa:.6f}")
    print(f"  For f_I1 = {f_I1_star} (kappa*): factor = {factor_star:.6f}")

    # Hypothesis C: epsilon_I1 = θ × (1-θ)² / denominator
    # Let's find what denominator would work
    numerator_c = theta * (1 - theta)**2
    denom_needed_c = numerator_c / epsilon_I1
    print(f"\nHypothesis C: epsilon_I1 = θ(1-θ)² / denom")
    print(f"  θ(1-θ)² = {numerator_c:.10f}")
    print(f"  denom needed = {denom_needed_c:.4f}")
    print(f"  For comparison: 2K(2K+1) = {2*K*(2*K+1)}")
    print(f"  For comparison: 2K(2K+1)² = {2*K*(2*K+1)**2}")

    # Hypothesis D: epsilon_I1 = epsilon_I2 × θ × some_factor
    factor_d = epsilon_I1 / (epsilon_I2 * theta)
    print(f"\nHypothesis D: epsilon_I1 = epsilon_I2 × θ × factor")
    print(f"  factor = {factor_d:.6f}")
    print(f"  1/(2K) = {1/(2*K):.6f}")
    print(f"  1/(2K+1) = {1/(2*K+1):.6f}")
    print(f"  1/(K(2K+1)) = {1/(K*(2*K+1)):.6f}")

    # Hypothesis E: Direct formula exploration
    print(f"\n" + "=" * 70)
    print("FORMULA SEARCH")
    print("=" * 70)

    formulas = {
        "θ²(1-θ)/(2K(2K+1))": theta**2 * (1-theta) / (2*K*(2*K+1)),
        "θ(1-θ)²/(2K(2K+1))": theta * (1-theta)**2 / (2*K*(2*K+1)),
        "θ(1-θ)/(2K(2K+1)²)": theta * (1-theta) / (2*K*(2*K+1)**2),
        "θ²(1-θ)/(2K(2K+1)²)": theta**2 * (1-theta) / (2*K*(2*K+1)**2),
        "θ(1-θ)/(K(2K+1)²)": theta * (1-theta) / (K*(2*K+1)**2),
        "θ²(1-θ)²/(2K(2K+1))": theta**2 * (1-theta)**2 / (2*K*(2*K+1)),
        "(1-θ)×β": (1-theta) * beta,
        "θ×(1-θ)×β": theta * (1-theta) * beta,
        "(1-θ)²×β": (1-theta)**2 * beta,
        "θ×β²": theta * beta**2,
        "(1-θ)×β²×(2K+1)": (1-theta) * beta**2 * (2*K+1),
    }

    print(f"\nTarget: epsilon_I1 = {epsilon_I1:.10f}")
    print(f"\nFormula values:")

    best_formula = None
    best_ratio = float('inf')

    for name, value in formulas.items():
        ratio = value / epsilon_I1 if epsilon_I1 != 0 else float('inf')
        error_pct = abs(ratio - 1) * 100
        print(f"  {name:30s} = {value:.10f}  (ratio: {ratio:.4f}, error: {error_pct:.1f}%)")

        if abs(ratio - 1) < abs(best_ratio - 1):
            best_ratio = ratio
            best_formula = name

    print(f"\nBest formula: {best_formula} with ratio {best_ratio:.4f}")

    # Try formulas involving f_I1
    print(f"\n" + "-" * 70)
    print("FORMULAS INVOLVING f_I1")
    print("-" * 70)

    for f_I1, label in [(f_I1_kappa, "kappa"), (f_I1_star, "kappa*")]:
        print(f"\nFor f_I1 = {f_I1} ({label}):")

        f_formulas = {
            "f_I1 × epsilon_I2": f_I1 * epsilon_I2,
            "f_I1 × θ(1-θ)/(2K(2K+1))": f_I1 * theta * (1-theta) / (2*K*(2*K+1)),
            "f_I1² × epsilon_I2": f_I1**2 * epsilon_I2,
            "f_I1 × (1-θ) × β": f_I1 * (1-theta) * beta,
            "f_I1 × θ × β": f_I1 * theta * beta,
        }

        for name, value in f_formulas.items():
            ratio = value / epsilon_I1 if epsilon_I1 != 0 else float('inf')
            error_pct = abs(ratio - 1) * 100
            print(f"    {name:35s} = {value:.10f}  (ratio: {ratio:.4f}, error: {error_pct:.1f}%)")


def main():
    explore_g_i1_formulas()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The epsilon_I1 = 0.00091 does not match any simple formula involving
θ, K, and the Beta moment structure.

However, the θ(2-θ) formula for g_I2 already achieves ~0.02% accuracy
on both benchmarks. The remaining gap from g_I1 = 1.0 vs 1.00091 is:
- Only 0.09% of g_I1 itself
- Contributes ~0.02% to the final c error (weighted by f_I1)

For practical purposes, the current formula is essentially exact:
  g_I1 = 1.0
  g_I2 = 1 + θ(2-θ)/(2K(2K+1))

The 0.02% residual is within typical numerical tolerances and
may come from:
1. Higher-order Q polynomial effects
2. Quadrature discretization
3. Interaction between I1 and I2 that our model doesn't capture
""")


if __name__ == "__main__":
    main()
