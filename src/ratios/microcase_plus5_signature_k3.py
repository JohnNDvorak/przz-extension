"""
src/ratios/microcase_plus5_signature_k3.py
Phase 14D Task D4: Microcase script for +5 signature diagnosis

PURPOSE:
========
Minimal, deterministic script that:
1. Uses actual PRZZ polynomials (from D1 loader)
2. Computes the decomposition m₁ ≈ A × exp(R) + B
3. Prints per-piece J₁ contributions separately
4. Shows where the +5 should come from

This is a diagnostic tool for debugging the +5 gate tests.

EXPECTED BEHAVIOR:
=================
With correct implementation:
- B ≈ 5 (= 2K-1 for K=3)
- A > 0 and A ≈ 1
- J₁₅ should be the primary contributor to the +5 constant

CURRENT STATUS:
==============
Phase 14D wired real polynomials, but the Euler-Maclaurin formulas
give B ≈ -0.27, not B ≈ 5. This script helps diagnose what's missing.

TeX REFERENCE:
=============
PRZZ Lines 1502-1511: m₁ structure with mirror combination
The "+5" comes from 2K-1 for K=3 pieces (combinatorial factor).
"""

from __future__ import annotations
import numpy as np
from scipy import integrate

from src.ratios.przz_polynomials import (
    load_przz_k3_polynomials,
    PrzzK3Polynomials,
    KAPPA_R,
    KAPPA_STAR_R,
)
from src.ratios.j1_euler_maclaurin import (
    decompose_m1_using_integrals,
    compute_J1_as_integrals,
    compute_m1_with_mirror_assembly,
)
from src.ratios.arithmetic_factor import A11_prime_sum
from src.ratios.zeta_laurent import EULER_MASCHERONI


def compute_polynomial_integrals(polys: PrzzK3Polynomials) -> dict:
    """
    Compute various polynomial integrals that appear in J₁ formulas.

    Returns:
        Dictionary with named integral values
    """
    def P1(u):
        return float(polys.P1.eval(np.array([u]))[0])

    def P2(u):
        return float(polys.P2.eval(np.array([u]))[0])

    def P3(u):
        return float(polys.P3.eval(np.array([u]))[0])

    # Various integrals that appear in PRZZ
    integrals = {}

    # ∫ P₁(u)² du
    integrals["int_P1_squared"], _ = integrate.quad(lambda u: P1(u)**2, 0, 1)

    # ∫ P₁(u)P₂(u) du
    integrals["int_P1_P2"], _ = integrate.quad(lambda u: P1(u)*P2(u), 0, 1)

    # ∫ P₁(u)P₃(u) du
    integrals["int_P1_P3"], _ = integrate.quad(lambda u: P1(u)*P3(u), 0, 1)

    # ∫ (1-u) P₁(u)² du
    integrals["int_1mu_P1_sq"], _ = integrate.quad(
        lambda u: (1-u)*P1(u)**2, 0, 1
    )

    # ∫ (1-u) P₁(u)P₂(u) du
    integrals["int_1mu_P1_P2"], _ = integrate.quad(
        lambda u: (1-u)*P1(u)*P2(u), 0, 1
    )

    return integrals


def run_microcase(benchmark: str = "kappa", verbose: bool = True) -> dict:
    """
    Run the microcase analysis for a given benchmark.

    Args:
        benchmark: "kappa" or "kappa_star"
        verbose: If True, print detailed output

    Returns:
        Dictionary with decomposition results
    """
    # Load real PRZZ polynomials
    polys = load_przz_k3_polynomials(benchmark)
    R = polys.R
    theta = polys.theta

    if verbose:
        print("=" * 70)
        print(f"MICROCASE +5 SIGNATURE ANALYSIS")
        print(f"Benchmark: {benchmark}, R={R}")
        print("=" * 70)
        print()

    # Compute decomposition using Euler-Maclaurin integrals
    decomp = decompose_m1_using_integrals(theta=theta, R=R, polys=polys)

    A = decomp["exp_coefficient"]
    B = decomp["constant_offset"]

    if verbose:
        print("DECOMPOSITION: m₁ ≈ A × exp(R) + B")
        print("-" * 50)
        print(f"  A (exp coefficient): {A:.6f}")
        print(f"  B (constant offset): {B:.6f}")
        print(f"  Target B: 5 (= 2K-1 for K=3)")
        print(f"  Gap from target: {B - 5:.6f}")
        print()

        print("PER-PIECE CONTRIBUTIONS:")
        print("-" * 50)
        total_exp = 0.0
        total_const = 0.0
        for name, contrib in decomp["per_piece_contribution"].items():
            exp_c = contrib["exp_coefficient"]
            const = contrib["constant"]
            total_exp += exp_c
            total_const += const
            print(f"  {name}:")
            print(f"    exp coefficient: {exp_c:+.6f}")
            print(f"    constant:        {const:+.6f}")
        print("-" * 50)
        print(f"  TOTAL:")
        print(f"    exp coefficient: {total_exp:+.6f} (should match A)")
        print(f"    constant:        {total_const:+.6f} (should match B)")
        print()

    # Compute polynomial integrals for diagnostic
    poly_ints = compute_polynomial_integrals(polys)

    if verbose:
        print("POLYNOMIAL INTEGRALS:")
        print("-" * 50)
        for name, val in poly_ints.items():
            print(f"  {name}: {val:.6f}")
        print()

    # A^{(1,1)}(0) value
    A11_val = A11_prime_sum(0.0, prime_cutoff=5000)

    if verbose:
        print("KEY PARAMETERS:")
        print("-" * 50)
        print(f"  R: {R}")
        print(f"  θ: {theta:.10f}")
        print(f"  exp(R): {np.exp(R):.6f}")
        print(f"  A^{{(1,1)}}(0): {A11_val:.6f}")
        print(f"  1/R + γ: {1.0/R + EULER_MASCHERONI:.6f}")
        print(f"  (1/R + γ)²: {(1.0/R + EULER_MASCHERONI)**2:.6f}")
        print()

        # Expected J15 contribution
        expected_j15 = A11_val * poly_ints["int_P1_P2"]
        print("EXPECTED J₁₅ CONTRIBUTION:")
        print("-" * 50)
        print(f"  J₁₅ = A^{{(1,1)}}(0) × ∫P₁P₂ du")
        print(f"       = {A11_val:.4f} × {poly_ints['int_P1_P2']:.4f}")
        print(f"       = {expected_j15:.6f}")
        print()

        print("GAP ANALYSIS:")
        print("-" * 50)
        print(f"  Current B: {B:.6f}")
        print(f"  Target B: 5.000000")
        print(f"  Gap: {5.0 - B:.6f}")
        print()
        print("  If B came from J₁₅ alone, we'd need:")
        print(f"    ∫P₁P₂ du = 5 / A^{{(1,1)}}(0) = {5.0/A11_val:.6f}")
        print(f"  But actual ∫P₁P₂ du = {poly_ints['int_P1_P2']:.6f}")
        print()

        print("=" * 70)

    return {
        "A": A,
        "B": B,
        "target_B": 5.0,
        "gap": B - 5.0,
        "per_piece": decomp["per_piece_contribution"],
        "polynomial_integrals": poly_ints,
        "A11_value": A11_val,
        "R": R,
        "benchmark": benchmark,
    }


def run_microcase_with_mirror(benchmark: str = "kappa", verbose: bool = True) -> dict:
    """
    Run Phase 14E microcase analysis with mirror assembly.

    This is the corrected version that applies mirror assembly before
    extracting A and B coefficients.

    Args:
        benchmark: "kappa" or "kappa_star"
        verbose: If True, print detailed output

    Returns:
        Dictionary with decomposition results
    """
    # Load real PRZZ polynomials
    polys = load_przz_k3_polynomials(benchmark)
    R = polys.R
    theta = polys.theta

    if verbose:
        print("=" * 70)
        print(f"PHASE 14E: MICROCASE WITH MIRROR ASSEMBLY")
        print(f"Benchmark: {benchmark}, R={R}")
        print("=" * 70)
        print()

    # Compute decomposition using mirror assembly (Phase 14E)
    decomp = compute_m1_with_mirror_assembly(theta=theta, R=R, polys=polys, K=3)

    A = decomp["exp_coefficient"]
    B = decomp["constant_offset"]
    target = decomp["target_constant"]

    if verbose:
        print("PRZZ MIRROR ASSEMBLY FORMULA:")
        print("-" * 50)
        print("  c = I₁₂(+R) + m × I₁₂(-R) + I₃₄(+R)")
        print(f"  where m = exp(R) + {target} = {decomp['mirror_multiplier']:.4f}")
        print()

        print("COMPONENT TOTALS:")
        print("-" * 50)
        print(f"  I₁₂(+R): {decomp['i12_plus_total']:.6f}")
        print(f"  I₁₂(-R): {decomp['i12_minus_total']:.6f}")
        print(f"  I₃₄(+R): {decomp['i34_plus_total']:.6f}")
        print()

        print("I₁₂ PIECES (at +R):")
        for name, val in decomp['i12_plus_pieces'].items():
            print(f"  {name}: {val:+.6f}")
        print()

        print("I₁₂ PIECES (at -R):")
        for name, val in decomp['i12_minus_pieces'].items():
            print(f"  {name}: {val:+.6f}")
        print()

        print("I₃₄ PIECES (at +R):")
        for name, val in decomp['i34_plus_pieces'].items():
            print(f"  {name}: {val:+.6f}")
        print()

        print("ASSEMBLED RESULT:")
        print("-" * 50)
        print(f"  Total: {decomp['assembled_total']:.6f}")
        print()

        print("DECOMPOSITION: m₁ ≈ A × exp(R) + B")
        print("-" * 50)
        print(f"  A (exp coefficient): {A:.6f}")
        print(f"  B (constant offset): {B:.6f}")
        print(f"  Target B: {target} (= 2K-1 for K=3)")
        gap = B - target
        rel_gap = gap / target * 100
        print(f"  Gap from target: {gap:+.6f} ({rel_gap:+.1f}%)")
        print()

        # Phase 14F normalized metrics
        print("PHASE 14F NORMALIZED METRICS:")
        print("-" * 50)
        print(f"  D = I₁₂(+R) + I₃₄(+R): {decomp['D']:.6f}")
        print(f"  delta = D/A:           {decomp['delta']:.6f}")
        print(f"  B/A = 5 + delta:       {decomp['B_over_A']:.6f}")
        print(f"  Target B/A:            5.000000")
        ba_gap = decomp['B_over_A'] - 5.0
        ba_rel_gap = ba_gap / 5.0 * 100
        print(f"  Gap from target:       {ba_gap:+.6f} ({ba_rel_gap:+.1f}%)")
        print()

        print("=" * 70)

    return {
        "A": A,
        "B": B,
        "target_B": target,
        "gap": B - target,
        "gap_percent": (B - target) / target * 100,
        "i12_plus_total": decomp['i12_plus_total'],
        "i12_minus_total": decomp['i12_minus_total'],
        "i34_plus_total": decomp['i34_plus_total'],
        "mirror_multiplier": decomp['mirror_multiplier'],
        "assembled_total": decomp['assembled_total'],
        "R": R,
        "benchmark": benchmark,
        "method": "mirror_assembly",
        # Phase 14F: Normalized metrics
        "D": decomp['D'],
        "delta": decomp['delta'],
        "B_over_A": decomp['B_over_A'],
    }


def compare_approaches(benchmark: str = "kappa"):
    """Compare Phase 14D (no mirror) vs Phase 14E (with mirror)."""
    polys = load_przz_k3_polynomials(benchmark)
    R = polys.R
    theta = polys.theta

    # Phase 14D: individual pieces, no mirror
    old = decompose_m1_using_integrals(theta=theta, R=R, polys=polys)

    # Phase 14E: mirror assembly
    new = compute_m1_with_mirror_assembly(theta=theta, R=R, polys=polys, K=3)

    print(f"\nCOMPARISON: {benchmark.upper()} (R={R})")
    print("-" * 50)
    print(f"  Phase 14D B (no mirror):   {old['constant_offset']:+.4f}")
    print(f"  Phase 14E B (with mirror): {new['constant_offset']:+.4f}")
    print(f"  Target B:                  {new['target_constant']:+.4f}")
    print(f"  Improvement:               {new['constant_offset'] - old['constant_offset']:+.4f}")

    return old, new


def compare_benchmarks():
    """Run microcase for both kappa and kappa* benchmarks."""
    print("\n" + "=" * 70)
    print("COMPARING KAPPA AND KAPPA* BENCHMARKS")
    print("=" * 70 + "\n")

    result_k = run_microcase("kappa", verbose=True)
    print("\n")
    result_ks = run_microcase("kappa_star", verbose=True)

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<25} {'kappa':<15} {'kappa*':<15}")
    print("-" * 55)
    print(f"  {'A (exp coef)':<25} {result_k['A']:.6f}       {result_ks['A']:.6f}")
    print(f"  {'B (constant)':<25} {result_k['B']:.6f}       {result_ks['B']:.6f}")
    print(f"  {'Gap from 5':<25} {result_k['gap']:.6f}       {result_ks['gap']:.6f}")
    print(f"  {'A^(1,1)(0)':<25} {result_k['A11_value']:.6f}       {result_ks['A11_value']:.6f}")
    print("=" * 70)


def compare_benchmarks_with_mirror():
    """Compare both benchmarks using Phase 14E/14F mirror assembly."""
    print("\n" + "=" * 70)
    print("PHASE 14F: COMPARING BENCHMARKS WITH NORMALIZED METRICS")
    print("=" * 70 + "\n")

    result_k = run_microcase_with_mirror("kappa", verbose=False)
    result_ks = run_microcase_with_mirror("kappa_star", verbose=False)

    print(f"  {'Metric':<25} {'kappa':<15} {'kappa*':<15}")
    print("-" * 55)
    print(f"  {'A (exp coef)':<25} {result_k['A']:.6f}       {result_ks['A']:.6f}")
    print(f"  {'B (constant)':<25} {result_k['B']:.6f}       {result_ks['B']:.6f}")
    print(f"  {'Target B':<25} {result_k['target_B']:.6f}       {result_ks['target_B']:.6f}")
    print(f"  {'Gap from target (B)':<25} {result_k['gap']:+.6f}      {result_ks['gap']:+.6f}")
    print(f"  {'Gap % (B)':<25} {result_k['gap_percent']:+.1f}%          {result_ks['gap_percent']:+.1f}%")
    print("-" * 55)
    print("  PHASE 14F NORMALIZED METRICS:")
    print(f"  {'D = I₁₂(+R) + I₃₄(+R)':<25} {result_k['D']:.6f}       {result_ks['D']:.6f}")
    print(f"  {'delta = D/A':<25} {result_k['delta']:.6f}       {result_ks['delta']:.6f}")
    print(f"  {'B/A (normalized)':<25} {result_k['B_over_A']:.6f}       {result_ks['B_over_A']:.6f}")
    print(f"  {'Target B/A':<25} {'5.000000':<15} {'5.000000':<15}")
    ba_gap_k = (result_k['B_over_A'] - 5.0) / 5.0 * 100
    ba_gap_ks = (result_ks['B_over_A'] - 5.0) / 5.0 * 100
    print(f"  {'Gap % (B/A)':<25} {ba_gap_k:+.1f}%           {ba_gap_ks:+.1f}%")
    print("=" * 70)

    return result_k, result_ks


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# PHASE 14E: +5 GATE MICROCASE ANALYSIS")
    print("#" * 70)

    # Compare Phase 14D vs Phase 14E
    print("\n" + "=" * 70)
    print("PHASE 14D vs PHASE 14E COMPARISON")
    print("=" * 70)
    compare_approaches("kappa")
    compare_approaches("kappa_star")

    # Run Phase 14E mirror assembly analysis
    print("\n")
    run_microcase_with_mirror("kappa", verbose=True)

    print("\n")
    run_microcase_with_mirror("kappa_star", verbose=True)

    # Summary comparison
    compare_benchmarks_with_mirror()
