#!/usr/bin/env python3
"""
scripts/run_phase30_dual_benchmark_decomposition.py
Phase 30.1: Dual Benchmark Decomposition

This script prints exact components for BOTH benchmarks from a single code path,
allowing direct comparison to identify the source of the κ* 9.29% gap vs κ's 1.35% gap.

Output structure for each benchmark:
- Polynomial provenance (loader, degrees, fingerprint)
- S12 components (paper regime): S12(+R), S12(-R), ratio
- S34 component: S34(+R)
- Assembly: m, c computed, c target, gap

Created: 2025-12-26 (Phase 30)
"""

import sys
import math
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, Any

sys.path.insert(0, ".")

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.evaluate import compute_c_paper_with_mirror


# =============================================================================
# Benchmark configurations
# =============================================================================

BENCHMARKS = {
    "kappa": {
        "loader": load_przz_polynomials,
        "loader_name": "load_przz_polynomials()",
        "R": 1.3036,
        "c_target": 2.13745440613217,
        "kappa_target": 0.417293962,
    },
    "kappa_star": {
        "loader": load_przz_polynomials_kappa_star,
        "loader_name": "load_przz_polynomials_kappa_star()",
        "R": 1.1167,
        "c_target": 1.9379524124677437,
        "kappa_target": 0.407511457,
    },
}


@dataclass
class BenchmarkDecomposition:
    """Full decomposition of a benchmark computation."""
    benchmark: str
    loader_name: str
    R: float
    theta: float

    # Polynomial info
    Q_degree: int
    P2_degree: int
    P3_degree: int
    fingerprint: str

    # S12 components
    S12_plus: float
    S12_minus: float
    S12_ratio: float

    # S34 component
    S34_plus: float

    # Assembly
    m_empirical: float
    c_computed: float
    c_target: float
    c_gap_pct: float

    # Kappa
    kappa_computed: float
    kappa_target: float
    kappa_gap_pct: float


def get_poly_degree(poly) -> int:
    """Get the degree of a polynomial."""
    # Handle different polynomial types
    if hasattr(poly, 'degree'):
        return poly.degree
    if hasattr(poly, 'tilde_coeffs'):
        return len(poly.tilde_coeffs)
    if hasattr(poly, 'coeffs'):
        return len(poly.coeffs) - 1
    # For QPolynomial with basis representation
    if hasattr(poly, 'coeffs_in_basis_terms'):
        # Degree is 2*(number of terms - 1) for Legendre-like basis
        return 2 * (len(poly.coeffs_in_basis_terms) - 1) + 1
    return -1


def get_poly_coeffs_for_fingerprint(poly) -> str:
    """Get polynomial coefficients as a string for fingerprinting."""
    if hasattr(poly, 'tilde_coeffs'):
        coeffs = poly.tilde_coeffs
    elif hasattr(poly, 'coeffs'):
        coeffs = poly.coeffs
    elif hasattr(poly, 'coeffs_in_basis_terms'):
        coeffs = poly.coeffs_in_basis_terms
    else:
        coeffs = []
    # Round to 8 decimal places for stability
    return str([round(float(c), 8) for c in coeffs])


def compute_poly_fingerprint(polys: Dict) -> str:
    """Compute stable fingerprint from polynomial coefficients."""
    data = []
    for key in sorted(polys.keys()):
        coeffs = get_poly_coeffs_for_fingerprint(polys[key])
        data.append(f"{key}:{coeffs}")
    return hashlib.sha256("|".join(data).encode()).hexdigest()[:16]


def compute_decomposition(benchmark: str, n_quad: int = 60) -> BenchmarkDecomposition:
    """Compute full decomposition for a benchmark."""
    config = BENCHMARKS[benchmark]

    # Load polynomials
    P1, P2, P3, Q = config["loader"]()
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    R = config["R"]
    theta = 4 / 7

    # Get polynomial info
    Q_degree = get_poly_degree(Q)
    P2_degree = get_poly_degree(P2)
    P3_degree = get_poly_degree(P3)
    fingerprint = compute_poly_fingerprint(polys)

    # Compute c using paper regime with mirror
    result = compute_c_paper_with_mirror(
        theta=theta,
        R=R,
        n=n_quad,
        polynomials=polys,
        pair_mode="triangle",  # Use triangle mode for consistency
        use_factorial_normalization=True,
        mode="main",
        K=3,
        mirror_mode="empirical_scalar",
    )

    # Extract components from per_term
    per_term = result.per_term or {}
    S12_plus = per_term.get("_S12_plus_total", float('nan'))
    S12_minus = per_term.get("_S12_minus_total", float('nan'))
    S34_plus = per_term.get("_I3_I4_plus_total", float('nan'))
    m_empirical = per_term.get("_mirror_multiplier", float('nan'))

    # Compute ratio
    S12_ratio = S12_plus / S12_minus if abs(S12_minus) > 1e-15 else float('inf')

    # Get c
    c_computed = result.total
    c_target = config["c_target"]
    c_gap_pct = 100 * (c_computed - c_target) / c_target

    # Compute kappa
    kappa_computed = 1 - math.log(c_computed) / R if c_computed > 0 else float('-inf')
    kappa_target = config["kappa_target"]
    kappa_gap_pct = 100 * (kappa_computed - kappa_target) / kappa_target

    return BenchmarkDecomposition(
        benchmark=benchmark,
        loader_name=config["loader_name"],
        R=R,
        theta=theta,
        Q_degree=Q_degree,
        P2_degree=P2_degree,
        P3_degree=P3_degree,
        fingerprint=fingerprint,
        S12_plus=S12_plus,
        S12_minus=S12_minus,
        S12_ratio=S12_ratio,
        S34_plus=S34_plus,
        m_empirical=m_empirical,
        c_computed=c_computed,
        c_target=c_target,
        c_gap_pct=c_gap_pct,
        kappa_computed=kappa_computed,
        kappa_target=kappa_target,
        kappa_gap_pct=kappa_gap_pct,
    )


def print_decomposition(d: BenchmarkDecomposition) -> None:
    """Print formatted decomposition."""
    print(f"\n{'=' * 60}")
    print(f"=== BENCHMARK: {d.benchmark.upper()} ===")
    print(f"{'=' * 60}")

    print(f"\nPolynomial Provenance:")
    print(f"  loader: {d.loader_name}")
    print(f"  R: {d.R}, theta: {d.theta:.6f}")
    print(f"  Q degree: {d.Q_degree}, P2 degree: {d.P2_degree}, P3 degree: {d.P3_degree}")
    print(f"  Fingerprint: {d.fingerprint}")

    print(f"\nS12 Components (paper regime):")
    print(f"  S12(+R): {d.S12_plus:.6f}")
    print(f"  S12(-R): {d.S12_minus:.6f}")
    print(f"  Ratio S12(+R)/S12(-R): {d.S12_ratio:.4f}")

    print(f"\nS34 Component:")
    print(f"  S34(+R): {d.S34_plus:.6f}")

    print(f"\nAssembly:")
    print(f"  m_empirical = exp(R) + 5 = {d.m_empirical:.4f}")
    print(f"  c = S12(+R) + m x S12(-R) + S34(+R)")
    print(f"    = {d.S12_plus:.6f} + {d.m_empirical:.4f} x {d.S12_minus:.6f} + {d.S34_plus:.6f}")
    print(f"  c_computed = {d.c_computed:.6f}")
    print(f"  c_target   = {d.c_target:.6f}")
    print(f"  c_gap      = {d.c_gap_pct:+.2f}%")

    print(f"\nKappa:")
    print(f"  kappa_computed = {d.kappa_computed:.6f}")
    print(f"  kappa_target   = {d.kappa_target:.6f}")
    print(f"  kappa_gap      = {d.kappa_gap_pct:+.2f}%")


def print_comparison(kappa: BenchmarkDecomposition, kappa_star: BenchmarkDecomposition) -> None:
    """Print side-by-side comparison."""
    print(f"\n{'=' * 60}")
    print("=== COMPARISON: kappa vs kappa* ===")
    print(f"{'=' * 60}")

    print(f"\n{'Component':<25} {'kappa':>15} {'kappa*':>15} {'Ratio (k/k*)':>15}")
    print("-" * 70)

    # Polynomial degrees
    print(f"{'P2 degree':<25} {kappa.P2_degree:>15} {kappa_star.P2_degree:>15} {'(DIFFERENT!)' if kappa.P2_degree != kappa_star.P2_degree else ''}")
    print(f"{'P3 degree':<25} {kappa.P3_degree:>15} {kappa_star.P3_degree:>15} {'(DIFFERENT!)' if kappa.P3_degree != kappa_star.P3_degree else ''}")
    print(f"{'Q degree':<25} {kappa.Q_degree:>15} {kappa_star.Q_degree:>15} {'(DIFFERENT!)' if kappa.Q_degree != kappa_star.Q_degree else ''}")

    print()
    print(f"{'S12(+R)':<25} {kappa.S12_plus:>15.6f} {kappa_star.S12_plus:>15.6f} {kappa.S12_plus / kappa_star.S12_plus:>15.4f}")
    print(f"{'S12(-R)':<25} {kappa.S12_minus:>15.6f} {kappa_star.S12_minus:>15.6f} {kappa.S12_minus / kappa_star.S12_minus:>15.4f}")
    print(f"{'S12 ratio (+R/-R)':<25} {kappa.S12_ratio:>15.4f} {kappa_star.S12_ratio:>15.4f} {kappa.S12_ratio / kappa_star.S12_ratio:>15.4f}")

    print()
    print(f"{'S34(+R)':<25} {kappa.S34_plus:>15.6f} {kappa_star.S34_plus:>15.6f} {kappa.S34_plus / kappa_star.S34_plus if abs(kappa_star.S34_plus) > 1e-10 else float('nan'):>15.4f}")

    print()
    print(f"{'m_empirical':<25} {kappa.m_empirical:>15.4f} {kappa_star.m_empirical:>15.4f} {kappa.m_empirical / kappa_star.m_empirical:>15.4f}")
    print(f"{'c_computed':<25} {kappa.c_computed:>15.6f} {kappa_star.c_computed:>15.6f} {kappa.c_computed / kappa_star.c_computed:>15.4f}")
    print(f"{'c_target':<25} {kappa.c_target:>15.6f} {kappa_star.c_target:>15.6f} {kappa.c_target / kappa_star.c_target:>15.4f}")

    print()
    print(f"{'c_gap %':<25} {kappa.c_gap_pct:>15.2f}% {kappa_star.c_gap_pct:>15.2f}%")
    print(f"{'kappa_gap %':<25} {kappa.kappa_gap_pct:>15.2f}% {kappa_star.kappa_gap_pct:>15.2f}%")

    # Fingerprints
    print()
    print(f"\nFingerprints:")
    print(f"  kappa:      {kappa.fingerprint}")
    print(f"  kappa*:     {kappa_star.fingerprint}")
    print(f"  Match:      {'YES (PROBLEM!)' if kappa.fingerprint == kappa_star.fingerprint else 'NO (correct - different polynomials)'}")

    # Diagnosis
    print(f"\n{'=' * 60}")
    print("=== DIAGNOSIS ===")
    print(f"{'=' * 60}")

    if abs(kappa.c_gap_pct) < 3 and abs(kappa_star.c_gap_pct) > 5:
        print("\nSYMPTOM: kappa OK (~1-3%), kappa* BAD (>5%)")
        print("\nPOSSIBLE CAUSES:")

        # Check polynomial differences
        if kappa.P2_degree != kappa_star.P2_degree:
            print(f"  [1] P2 degree differs: {kappa.P2_degree} vs {kappa_star.P2_degree}")
            print("      This is EXPECTED - kappa* has simpler polynomials")

        if kappa.fingerprint == kappa_star.fingerprint:
            print("  [!!!] SAME FINGERPRINT - wrong polynomials being used!")
            print("      LIKELY ROOT CAUSE: kappa polynomials used for kappa* benchmark")

        # Check which component has largest ratio discrepancy
        target_c_ratio = kappa.c_target / kappa_star.c_target
        computed_c_ratio = kappa.c_computed / kappa_star.c_computed
        ratio_diff = abs(computed_c_ratio - target_c_ratio) / target_c_ratio

        print(f"\n  c ratio (computed vs target):")
        print(f"    target:   {target_c_ratio:.4f}")
        print(f"    computed: {computed_c_ratio:.4f}")
        print(f"    diff:     {ratio_diff:.2%}")

        if ratio_diff > 0.05:
            # Which component is most off?
            S12_plus_ratio = kappa.S12_plus / kappa_star.S12_plus
            S12_minus_ratio = kappa.S12_minus / kappa_star.S12_minus
            S34_ratio = kappa.S34_plus / kappa_star.S34_plus if abs(kappa_star.S34_plus) > 1e-10 else float('nan')

            print(f"\n  Component ratios (kappa/kappa*):")
            print(f"    S12(+R): {S12_plus_ratio:.4f}")
            print(f"    S12(-R): {S12_minus_ratio:.4f}")
            print(f"    S34(+R): {S34_ratio:.4f}")

    else:
        print("\nBoth benchmarks have similar accuracy - no parity issue detected.")


def main():
    """Run dual benchmark decomposition."""
    print("Phase 30.1: Dual Benchmark Decomposition")
    print("=" * 60)
    print("Computing decomposition for both benchmarks...")

    # Compute decompositions
    kappa = compute_decomposition("kappa")
    kappa_star = compute_decomposition("kappa_star")

    # Print individual decompositions
    print_decomposition(kappa)
    print_decomposition(kappa_star)

    # Print comparison
    print_comparison(kappa, kappa_star)

    return {"kappa": kappa, "kappa_star": kappa_star}


if __name__ == "__main__":
    main()
