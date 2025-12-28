#!/usr/bin/env python3
"""
Phase 45: I1/I2 Split Analysis for First-Principles g Derivation

Solves for g_I1 and g_I2 that simultaneously satisfy both benchmarks.

Given:
  c = I1(+R) + g_I1 × base × I1(-R) + I2(+R) + g_I2 × base × I2(-R) + S34

For κ and κ*, we have 2 equations with 2 unknowns (g_I1, g_I2).

If a solution exists where g_I1 ≈ g_baseline and g_I2 ≈ 1.0 (or vice versa),
this would confirm the I1/I2 split hypothesis and allow deriving the empirical
α and f_ref from first principles.

Created: 2025-12-27 (Phase 45)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.g_functional import compute_I1_I2_totals
from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term


@dataclass
class BenchmarkData:
    """Data for a single benchmark."""
    name: str
    R: float
    c_target: float
    polynomials: Dict

    # Computed values
    I1_plus: float = 0.0
    I1_minus: float = 0.0
    I2_plus: float = 0.0
    I2_minus: float = 0.0
    S34: float = 0.0
    base: float = 0.0  # exp(R) + (2K-1)

    # I1 fraction
    f_I1: float = 0.0


def compute_S34(theta: float, R: float, polynomials: Dict, n_quad: int = 60) -> float:
    """Compute S34 = I3 + I4 (no mirror)."""
    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")

    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    S34 = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms[pair_key]
        norm = factorial_norm[pair_key]
        sym = symmetry_factor[pair_key]
        full_norm = sym * norm

        for term in terms[2:4]:
            result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
            S34 += full_norm * result.value

    return S34


def compute_benchmark_data(
    name: str,
    R: float,
    c_target: float,
    polynomials: Dict,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> BenchmarkData:
    """Compute all data for a benchmark."""
    data = BenchmarkData(name=name, R=R, c_target=c_target, polynomials=polynomials)

    # Compute I1 and I2 at +R and -R
    data.I1_plus, data.I2_plus = compute_I1_I2_totals(R, theta, polynomials, n_quad)
    data.I1_minus, data.I2_minus = compute_I1_I2_totals(-R, theta, polynomials, n_quad)

    # Compute S34
    data.S34 = compute_S34(theta, R, polynomials, n_quad)

    # Compute base multiplier
    data.base = math.exp(R) + (2 * K - 1)

    # Compute I1 fraction at -R
    S12_minus = data.I1_minus + data.I2_minus
    data.f_I1 = data.I1_minus / S12_minus if abs(S12_minus) > 1e-15 else 0.0

    return data


def solve_for_g_I1_g_I2(
    data1: BenchmarkData,
    data2: BenchmarkData,
) -> Tuple[float, float, bool]:
    """
    Solve the 2x2 linear system for g_I1 and g_I2.

    Equations:
    g_I1 * (base1 * I1_minus1) + g_I2 * (base1 * I2_minus1) = c_target1 - I1_plus1 - I2_plus1 - S34_1
    g_I1 * (base2 * I1_minus2) + g_I2 * (base2 * I2_minus2) = c_target2 - I1_plus2 - I2_plus2 - S34_2

    Returns:
        (g_I1, g_I2, success)
    """
    # Build coefficient matrix A and RHS vector b
    a1 = data1.base * data1.I1_minus
    b1 = data1.base * data1.I2_minus
    c1 = data1.c_target - data1.I1_plus - data1.I2_plus - data1.S34

    a2 = data2.base * data2.I1_minus
    b2 = data2.base * data2.I2_minus
    c2 = data2.c_target - data2.I1_plus - data2.I2_plus - data2.S34

    # Solve: [a1 b1] [g_I1]   [c1]
    #        [a2 b2] [g_I2] = [c2]
    A = np.array([[a1, b1], [a2, b2]])
    b = np.array([c1, c2])

    det = np.linalg.det(A)
    if abs(det) < 1e-15:
        return 0.0, 0.0, False

    solution = np.linalg.solve(A, b)
    g_I1, g_I2 = solution

    return float(g_I1), float(g_I2), True


def verify_solution(
    data: BenchmarkData,
    g_I1: float,
    g_I2: float,
) -> Tuple[float, float]:
    """Verify the solution by computing c and comparing to target."""
    c_computed = (data.I1_plus + g_I1 * data.base * data.I1_minus +
                  data.I2_plus + g_I2 * data.base * data.I2_minus + data.S34)
    gap_pct = (c_computed / data.c_target - 1) * 100
    return c_computed, gap_pct


def compute_empirical_params(
    g_I1: float,
    g_I2: float,
    f_I1_1: float,
    f_I1_2: float,
    g_baseline: float,
    theta: float,
    K: int,
) -> Tuple[float, float]:
    """
    Derive α and f_ref from the I1/I2 split solution.

    The formula is:
      g(f_I1) = g_baseline + delta_g
      delta_g = -α × (θ/(2K(2K+1))) × (f_I1 - f_ref)

    We have g_total for each benchmark:
      g_total = f_I1 × g_I1 + (1 - f_I1) × g_I2  [weighted average]

    This gives us two equations to solve for α and f_ref.
    """
    beta_factor = theta / (2 * K * (2 * K + 1))

    # Compute g_total for each benchmark from the weighted formula
    g_total_1 = f_I1_1 * g_I1 + (1 - f_I1_1) * g_I2
    g_total_2 = f_I1_2 * g_I1 + (1 - f_I1_2) * g_I2

    # delta_g = g_total - g_baseline
    delta_g_1 = g_total_1 - g_baseline
    delta_g_2 = g_total_2 - g_baseline

    # From the formula: delta_g = -α × beta_factor × (f_I1 - f_ref)
    # We have two points (f_I1_1, delta_g_1) and (f_I1_2, delta_g_2)
    # Slope = delta_g / (f_I1 - f_ref)

    # Linear fit: delta_g = slope * f_I1 + intercept
    # slope = -α × beta_factor
    # intercept = α × beta_factor × f_ref

    slope = (delta_g_2 - delta_g_1) / (f_I1_2 - f_I1_1)
    intercept = delta_g_1 - slope * f_I1_1

    # α = -slope / beta_factor
    # f_ref = -intercept / slope = intercept / (α × beta_factor)
    alpha_derived = -slope / beta_factor
    f_ref_derived = intercept / (alpha_derived * beta_factor) if abs(alpha_derived * beta_factor) > 1e-15 else 0.0

    return alpha_derived, f_ref_derived


def main():
    """Main entry point."""
    theta = 4 / 7
    K = 3
    n_quad = 60

    c_target_kappa = 2.13745440613217263636
    c_target_kappa_star = 1.9379524112

    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    print()
    print("=" * 90)
    print("PHASE 45: I1/I2 SPLIT ANALYSIS FOR FIRST-PRINCIPLES g DERIVATION")
    print("=" * 90)
    print()
    print(f"Baseline g = 1 + θ/(2K(2K+1)) = {g_baseline:.6f}")
    print(f"θ/(2K(2K+1)) = {theta / (2 * K * (2 * K + 1)):.6f}")
    print()

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polynomials_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Compute benchmark data
    print("Computing benchmark data...")
    data_kappa = compute_benchmark_data("kappa", 1.3036, c_target_kappa, polynomials_kappa, theta, K, n_quad)
    data_kappa_star = compute_benchmark_data("kappa*", 1.1167, c_target_kappa_star, polynomials_kappa_star, theta, K, n_quad)

    print()
    print("=" * 90)
    print("BENCHMARK DATA")
    print("=" * 90)

    for data in [data_kappa, data_kappa_star]:
        print()
        print(f"{data.name} (R={data.R}):")
        print(f"  I1_plus = {data.I1_plus:.6f}, I1_minus = {data.I1_minus:.6f}")
        print(f"  I2_plus = {data.I2_plus:.6f}, I2_minus = {data.I2_minus:.6f}")
        print(f"  S34 = {data.S34:.6f}")
        print(f"  base = exp(R) + 5 = {data.base:.6f}")
        print(f"  f_I1 = {data.f_I1:.4f} (I1 fraction at -R)")
        print(f"  c_target = {data.c_target:.6f}")

    # Solve for g_I1 and g_I2
    print()
    print("=" * 90)
    print("SOLVING FOR g_I1 AND g_I2")
    print("=" * 90)
    print()
    print("Solving the 2x2 system:")
    print("  g_I1 × (base1 × I1_minus1) + g_I2 × (base1 × I2_minus1) = c1")
    print("  g_I1 × (base2 × I1_minus2) + g_I2 × (base2 × I2_minus2) = c2")
    print()

    g_I1, g_I2, success = solve_for_g_I1_g_I2(data_kappa, data_kappa_star)

    if not success:
        print("ERROR: Failed to solve the system (singular matrix)")
        return

    print(f"Solution:")
    print(f"  g_I1 = {g_I1:.8f}")
    print(f"  g_I2 = {g_I2:.8f}")
    print()
    print(f"Comparison to baseline:")
    print(f"  g_baseline = {g_baseline:.8f}")
    print(f"  g_I1 / g_baseline = {g_I1 / g_baseline:.6f}")
    print(f"  g_I2 / g_baseline = {g_I2 / g_baseline:.6f}")
    print(f"  g_I1 - g_baseline = {g_I1 - g_baseline:+.6f}")
    print(f"  g_I2 - g_baseline = {g_I2 - g_baseline:+.6f}")

    # Verify solution
    print()
    print("=" * 90)
    print("VERIFICATION")
    print("=" * 90)

    for data in [data_kappa, data_kappa_star]:
        c_computed, gap_pct = verify_solution(data, g_I1, g_I2)
        print()
        print(f"{data.name}:")
        print(f"  c_computed = {c_computed:.10f}")
        print(f"  c_target   = {data.c_target:.10f}")
        print(f"  gap = {gap_pct:+.6f}%")

    # Compute what g_total would be with the weighted formula
    print()
    print("=" * 90)
    print("WEIGHTED g FORMULA ANALYSIS")
    print("=" * 90)
    print()
    print("If g_total = f_I1 × g_I1 + (1 - f_I1) × g_I2:")
    print()

    for data in [data_kappa, data_kappa_star]:
        g_total_weighted = data.f_I1 * g_I1 + (1 - data.f_I1) * g_I2

        # What g is actually needed for this benchmark?
        S12_minus = data.I1_minus + data.I2_minus
        c_residual = data.c_target - data.I1_plus - data.I2_plus - data.S34
        g_needed = c_residual / (data.base * S12_minus)

        print(f"{data.name}:")
        print(f"  f_I1 = {data.f_I1:.4f}")
        print(f"  g_total (weighted) = {g_total_weighted:.6f}")
        print(f"  g_needed (uniform) = {g_needed:.6f}")
        print(f"  Gap: {(g_total_weighted / g_needed - 1) * 100:+.4f}%")
        print()

    # Derive α and f_ref
    print("=" * 90)
    print("DERIVING α AND f_ref FROM I1/I2 SPLIT")
    print("=" * 90)
    print()

    # Method 1: From the uniform g_needed values
    g_needed_1 = (data_kappa.c_target - data_kappa.I1_plus - data_kappa.I2_plus - data_kappa.S34) / \
                 (data_kappa.base * (data_kappa.I1_minus + data_kappa.I2_minus))
    g_needed_2 = (data_kappa_star.c_target - data_kappa_star.I1_plus - data_kappa_star.I2_plus - data_kappa_star.S34) / \
                 (data_kappa_star.base * (data_kappa_star.I1_minus + data_kappa_star.I2_minus))

    delta_g_1 = g_needed_1 - g_baseline
    delta_g_2 = g_needed_2 - g_baseline

    f_I1_1 = data_kappa.f_I1
    f_I1_2 = data_kappa_star.f_I1

    beta_factor = theta / (2 * K * (2 * K + 1))

    # Linear fit: delta_g = slope * f_I1 + intercept
    slope = (delta_g_2 - delta_g_1) / (f_I1_2 - f_I1_1)
    intercept = delta_g_1 - slope * f_I1_1

    # α = -slope / beta_factor
    alpha_derived = -slope / beta_factor
    # f_ref: find where delta_g = 0
    # 0 = slope * f_ref + intercept → f_ref = -intercept / slope
    f_ref_derived = -intercept / slope if abs(slope) > 1e-15 else 0.0

    print("Method 1: From uniform g_needed values")
    print("-" * 60)
    print(f"  κ:  f_I1 = {f_I1_1:.4f}, g_needed = {g_needed_1:.6f}, delta_g = {delta_g_1:+.6f}")
    print(f"  κ*: f_I1 = {f_I1_2:.4f}, g_needed = {g_needed_2:.6f}, delta_g = {delta_g_2:+.6f}")
    print()
    print(f"  Linear fit: delta_g = slope × f_I1 + intercept")
    print(f"  slope = {slope:.6f}")
    print(f"  intercept = {intercept:.6f}")
    print()
    print(f"  Derived parameters:")
    print(f"    α_derived = -slope / β_factor = {alpha_derived:.4f}")
    print(f"    f_ref_derived = -intercept / slope = {f_ref_derived:.4f}")
    print()
    print(f"  Empirical parameters (Phase 44):")
    print(f"    α_empirical = 1.3625")
    print(f"    f_ref_empirical = 0.3154")
    print()
    print(f"  Comparison:")
    print(f"    α: derived {alpha_derived:.4f} vs empirical 1.3625 → gap {(alpha_derived / 1.3625 - 1) * 100:+.2f}%")
    print(f"    f_ref: derived {f_ref_derived:.4f} vs empirical 0.3154 → gap {(f_ref_derived / 0.3154 - 1) * 100:+.2f}%")

    # Method 2: From the g_I1, g_I2 solution
    print()
    print("Method 2: From g_I1, g_I2 solution via weighted formula")
    print("-" * 60)

    # Compute g_total for each benchmark using weighted formula
    g_total_1 = f_I1_1 * g_I1 + (1 - f_I1_1) * g_I2
    g_total_2 = f_I1_2 * g_I1 + (1 - f_I1_2) * g_I2

    delta_g_w_1 = g_total_1 - g_baseline
    delta_g_w_2 = g_total_2 - g_baseline

    slope_w = (delta_g_w_2 - delta_g_w_1) / (f_I1_2 - f_I1_1)
    intercept_w = delta_g_w_1 - slope_w * f_I1_1

    alpha_w = -slope_w / beta_factor
    f_ref_w = -intercept_w / slope_w if abs(slope_w) > 1e-15 else 0.0

    print(f"  g_total (weighted formula):")
    print(f"    κ:  f_I1 = {f_I1_1:.4f}, g_total = {g_total_1:.6f}, delta_g = {delta_g_w_1:+.6f}")
    print(f"    κ*: f_I1 = {f_I1_2:.4f}, g_total = {g_total_2:.6f}, delta_g = {delta_g_w_2:+.6f}")
    print()
    print(f"  Derived from weighted formula:")
    print(f"    α = {alpha_w:.4f}")
    print(f"    f_ref = {f_ref_w:.4f}")

    # Physical interpretation
    print()
    print("=" * 90)
    print("PHYSICAL INTERPRETATION")
    print("=" * 90)
    print()

    print("The I1/I2 split solution shows:")
    print()
    print(f"  g_I1 = {g_I1:.6f} (correction for I1 terms with log factor)")
    print(f"  g_I2 = {g_I2:.6f} (correction for I2 terms without log factor)")
    print()

    if g_I1 > g_I2:
        print("  → I1 needs LARGER correction than I2")
        print("  → This is consistent with I1 having the log factor cross-terms")
    else:
        print("  → I2 needs LARGER correction than I1")
        print("  → This is UNEXPECTED based on log factor structure")

    print()
    print("The weighted formula g_total = f_I1 × g_I1 + (1 - f_I1) × g_I2 explains why:")
    print("  - Low f_I1 (κ = 23%) → g_total closer to g_I2")
    print("  - High f_I1 (κ* = 33%) → g_total closer to g_I1")
    print()

    # Check if this matches the empirical correction pattern
    print("Checking if pattern matches empirical correction direction:")
    if g_I1 < g_I2:
        print("  - g_I1 < g_I2")
        print("  - Low f_I1 → higher g_total (more I2 weight)")
        print("  - High f_I1 → lower g_total (more I1 weight)")
        print("  - This matches κ needing g UP and κ* needing g DOWN ✓")
    else:
        print("  - g_I1 > g_I2")
        print("  - Low f_I1 → lower g_total (more I2 weight)")
        print("  - High f_I1 → higher g_total (more I1 weight)")
        print("  - This is OPPOSITE to the empirical pattern!")


if __name__ == "__main__":
    main()
