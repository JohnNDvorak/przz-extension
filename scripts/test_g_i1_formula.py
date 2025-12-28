#!/usr/bin/env python3
"""
scripts/test_g_i1_formula.py
Test the candidate g_I1 formula: g_I1 = 1 + θ(1-θ)/(2K(2K+1)²)

The formula search found that θ(1-θ)/(2K(2K+1)²) is within 9% of epsilon_I1.
This script tests various refinements to see if we can get closer.

Created: 2025-12-27
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_g_i1_formulas():
    print("=" * 70)
    print("TESTING g_I1 FORMULAS")
    print("=" * 70)

    theta = 4/7
    K = 3

    g_I1_calibrated = 1.00091428
    epsilon_I1_calibrated = 0.00091428

    print(f"\nTarget: g_I1 = {g_I1_calibrated:.10f}")
    print(f"        epsilon_I1 = {epsilon_I1_calibrated:.10f}")

    # Base formula
    base = theta * (1 - theta) / (2 * K * (2 * K + 1)**2)
    print(f"\nBase formula: θ(1-θ)/(2K(2K+1)²) = {base:.10f}")
    print(f"  g_I1 = 1 + base = {1 + base:.10f}")
    print(f"  Gap from calibrated: {((1+base)/g_I1_calibrated - 1)*100:.4f}%")

    # Try various scaling factors
    print("\n" + "-" * 70)
    print("SCALING FACTORS")
    print("-" * 70)

    scale_factors = [
        ("1", 1.0),
        ("(2K+1)/(2K)", (2*K+1)/(2*K)),
        ("(2K)/(2K-1)", (2*K)/(2*K-1)),
        ("(K+1)/K", (K+1)/K),
        ("(2K+1)/(2K-1)", (2*K+1)/(2*K-1)),
        ("7/6", 7/6),
        ("8/7", 8/7),
        ("(1+θ)", 1+theta),
        ("1/(1-θ)", 1/(1-theta)),
        ("2/(2-θ)", 2/(2-theta)),
    ]

    best_scale = None
    best_gap = float('inf')

    for name, scale in scale_factors:
        epsilon = base * scale
        g_I1 = 1 + epsilon
        gap = ((g_I1/g_I1_calibrated) - 1) * 100
        print(f"  {name:20s}: g_I1 = {g_I1:.10f}, gap = {gap:+.4f}%")

        if abs(gap) < abs(best_gap):
            best_gap = gap
            best_scale = name

    print(f"\nBest scale factor: {best_scale} with gap {best_gap:+.4f}%")

    # Try alternative base formulas
    print("\n" + "-" * 70)
    print("ALTERNATIVE BASE FORMULAS")
    print("-" * 70)

    alt_formulas = {
        "θ²(1-θ)/(2K(2K+1)²)": theta**2 * (1-theta) / (2*K*(2*K+1)**2),
        "θ(1-θ)²/(2K(2K+1)²)": theta * (1-theta)**2 / (2*K*(2*K+1)**2),
        "θ(1-θ)/(K(2K+1)²)": theta * (1-theta) / (K*(2*K+1)**2),
        "θ(1-θ)/(4K²(2K+1))": theta * (1-theta) / (4*K**2*(2*K+1)),
        "θ(1-θ)/(2K²(2K+1)²)": theta * (1-theta) / (2*K**2*(2*K+1)**2),
        "θ²(1-θ)/(4K(2K+1)²)": theta**2 * (1-theta) / (4*K*(2*K+1)**2),
        "(1-θ)²/(2K(2K+1)²)": (1-theta)**2 / (2*K*(2*K+1)**2),
        "θ(1-θ)(2-θ)/(2K(2K+1)³)": theta * (1-theta) * (2-theta) / (2*K*(2*K+1)**3),
    }

    for name, epsilon in alt_formulas.items():
        g_I1 = 1 + epsilon
        gap = ((g_I1/g_I1_calibrated) - 1) * 100
        print(f"  {name:35s}: g_I1 = {g_I1:.10f}, gap = {gap:+.4f}%")

    # Relationship to g_I2 formula
    print("\n" + "-" * 70)
    print("RELATIONSHIP TO g_I2 FORMULA")
    print("-" * 70)

    # g_I2 = 1 + θ(2-θ)/(2K(2K+1))
    # epsilon_I2 = θ(2-θ)/(2K(2K+1)) - θ/(2K(2K+1)) = θ(1-θ)/(2K(2K+1))

    epsilon_I2_from_formula = theta * (1 - theta) / (2 * K * (2 * K + 1))
    print(f"epsilon_I2 (from θ(2-θ) - θ formula) = θ(1-θ)/(2K(2K+1)) = {epsilon_I2_from_formula:.10f}")

    # If epsilon_I1 = epsilon_I2 / (2K+1)
    epsilon_I1_derived = epsilon_I2_from_formula / (2 * K + 1)
    g_I1_derived = 1 + epsilon_I1_derived
    gap = ((g_I1_derived/g_I1_calibrated) - 1) * 100
    print(f"\nHypothesis: epsilon_I1 = epsilon_I2 / (2K+1)")
    print(f"  epsilon_I1 = {epsilon_I1_derived:.10f}")
    print(f"  g_I1 = {g_I1_derived:.10f}")
    print(f"  Gap: {gap:+.4f}%")

    # This is exactly θ(1-θ)/(2K(2K+1)²)!
    print(f"\n  Note: θ(1-θ)/(2K(2K+1)) / (2K+1) = θ(1-θ)/(2K(2K+1)²)")
    print(f"        = {theta*(1-theta)/(2*K*(2*K+1)**2):.10f} ✓")


def main():
    test_g_i1_formulas()

    print("\n" + "=" * 70)
    print("CANDIDATE FORMULA FOR g_I1")
    print("=" * 70)

    theta = 4/7
    K = 3

    # Best candidate
    epsilon_I1 = theta * (1 - theta) / (2 * K * (2 * K + 1)**2)
    g_I1 = 1 + epsilon_I1

    g_I1_calibrated = 1.00091428

    print(f"""
CANDIDATE FORMULA:
  g_I1 = 1 + θ(1-θ)/(2K(2K+1)²)
       = 1 + {epsilon_I1:.10f}
       = {g_I1:.10f}

CALIBRATED VALUE:
  g_I1 = {g_I1_calibrated:.10f}

GAP:
  {((g_I1/g_I1_calibrated) - 1) * 100:+.4f}%

This formula has structure parallel to g_I2:
  g_I2 = 1 + θ(2-θ)/(2K(2K+1))

The relationship is:
  epsilon_I1 = epsilon_I2 / (2K+1)

where epsilon_I2 = θ(2-θ)/(2K(2K+1)) - θ/(2K(2K+1)) = θ(1-θ)/(2K(2K+1))

This suggests g_I1 has a "dampened" version of the same correction.
""")


if __name__ == "__main__":
    main()
