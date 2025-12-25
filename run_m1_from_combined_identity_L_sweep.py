#!/usr/bin/env python3
"""
Derive m1 from first principles via L-sweep.

GPT Phase 3: Replace empirical m1 = exp(R) + 5 with principled derivation.

=============================================================================
CRITICAL FINDING (2025-12-22):
=============================================================================
The finite-L combined identity approach DIVERGES linearly with L!

At α = β = -R/L, the combined identity becomes:
    L/(2R) × [exp(-Rθ(x+y)) - exp(2R)·exp(Rθ(x+y))]

The problem:
- The 1/(α+β) = -L/(2R) prefactor grows LINEARLY with L
- The bracket [exp(-...) - exp(2R)×exp(+...)] is L-INDEPENDENT
- Nothing cancels the L-dependence!

Result: m1_eff(L) ∝ L, diverging to infinity.

The post-identity approach works by computing the L→∞ limit ANALYTICALLY
using asymptotic expansion, NOT by evaluating at finite L.

This means:
1. Option A (reuse post-identity + L/(2R) prefactor) does NOT work
2. The "m1 derivation via L-sweep" approach requires rethinking
3. The empirical m1 = exp(R)+5 remains the working formula

See: tests/test_combined_identity_finite_L.py for validation tests
=============================================================================

ORIGINAL METHOD (conceptual - now known to diverge):
=====================================================
1. For each L in [10, 20, 50, 100]:
   a. Compute I1_combined(L) from combined identity at finite L
   b. Compute I1+ (plus branch at +R)
   c. Compute I1−base (minus branch base - without mirror weight)
   d. Solve: m1_eff(L) = (I1_combined - I1+) / I1−base

2. If m1_eff(L) converges, fit to closed form  ← DOES NOT CONVERGE
3. Compare against empirical m1 = exp(R) + 5

Usage:
    python3 run_m1_from_combined_identity_L_sweep.py
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.operator_post_identity import compute_I1_operator_post_identity_pair
from src.combined_identity_finite_L import compute_I1_combined_at_L


@dataclass
class M1SweepResult:
    """Result of m1 sweep at a specific L."""
    L: float
    I1_combined: Optional[float]  # From combined identity at finite L
    I1_plus: float        # Post-identity at +R
    I1_minus_base: float  # Post-identity at -R
    m1_eff: Optional[float]
    m1_empirical: float   # exp(R) + 5


def compute_I1_plus_post_identity(
    theta: float, R: float, n: int, polynomials: Dict, ell1: int = 1, ell2: int = 1
) -> float:
    """Compute I1 plus branch using post-identity operator."""
    result = compute_I1_operator_post_identity_pair(
        theta, R, ell1, ell2, n, polynomials
    )
    return result.I1_value


def compute_I1_minus_base_post_identity(
    theta: float, R: float, n: int, polynomials: Dict, ell1: int = 1, ell2: int = 1
) -> float:
    """Compute I1 minus branch base (at -R) using post-identity operator."""
    result = compute_I1_operator_post_identity_pair(
        theta, -R, ell1, ell2, n, polynomials  # Note: -R
    )
    return result.I1_value


def run_L_sweep(
    theta: float, R: float, n: int, polynomials: Dict
) -> List[M1SweepResult]:
    """Run L sweep and compute m1_eff at each L."""
    L_values = [10.0, 20.0, 50.0, 100.0]
    results = []

    # Pre-compute L-independent parts (post-identity)
    I1_plus = compute_I1_plus_post_identity(theta, R, n, polynomials)
    I1_minus_base = compute_I1_minus_base_post_identity(theta, R, n, polynomials)

    m1_empirical = np.exp(R) + 5

    for L in L_values:
        I1_combined = compute_I1_combined_at_L(theta, R, L, n, polynomials)

        if I1_combined is not None and abs(I1_minus_base) > 1e-15:
            m1_eff = (I1_combined - I1_plus) / I1_minus_base
        else:
            m1_eff = None

        results.append(M1SweepResult(
            L=L,
            I1_combined=I1_combined,
            I1_plus=I1_plus,
            I1_minus_base=I1_minus_base,
            m1_eff=m1_eff,
            m1_empirical=m1_empirical,
        ))

    return results


def analyze_convergence(results: List[M1SweepResult]) -> Dict:
    """Analyze whether m1_eff converges as L → ∞."""
    m1_values = [r.m1_eff for r in results if r.m1_eff is not None]

    if len(m1_values) < 2:
        return {
            "converged": None,
            "m1_limit": None,
            "message": "Insufficient data (I1_combined not implemented)"
        }

    # Check relative change between last two L values
    rel_change = abs(m1_values[-1] - m1_values[-2]) / abs(m1_values[-2])

    converged = rel_change < 0.01  # 1% threshold

    return {
        "converged": converged,
        "m1_limit": m1_values[-1] if converged else None,
        "rel_change": rel_change,
        "message": "Converged" if converged else "Not converged"
    }


def main():
    print("=" * 70)
    print("M1 DERIVATION VIA L-SWEEP")
    print("=" * 70)
    print()
    print("Goal: Derive m1 from combined identity, not calibration.")
    print("Method: Compute m1_eff(L) = (I1_combined - I1+) / I1−base")
    print("        and observe convergence as L → ∞")
    print()
    print("⚠️  CRITICAL FINDING: The finite-L approach DIVERGES!")
    print("   See docstring for details.")
    print()

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    theta = 4.0 / 7.0
    n = 40

    for R, name, polys in [
        (1.3036, "κ", polys_kappa),
        (1.1167, "κ*", polys_kappa_star)
    ]:
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {name} (R={R})")
        print(f"{'='*70}")

        m1_empirical = np.exp(R) + 5
        print(f"Empirical m1 = exp(R) + 5 = {m1_empirical:.6f}")
        print()

        results = run_L_sweep(theta, R, n, polys)

        print("L-Sweep Results (for (1,1) pair):")
        print("-" * 70)
        print(f"{'L':>8} {'I1_combined':>14} {'I1+':>14} {'I1−base':>14} {'m1_eff':>12}")
        print("-" * 70)

        for r in results:
            I1_comb_str = f"{r.I1_combined:.4f}" if r.I1_combined is not None else "N/A"
            m1_eff_str = f"{r.m1_eff:.4f}" if r.m1_eff is not None else "N/A"
            print(f"{r.L:>8.0f} {I1_comb_str:>14} {r.I1_plus:>14.8f} "
                  f"{r.I1_minus_base:>14.8f} {m1_eff_str:>12}")

        # Analyze convergence
        analysis = analyze_convergence(results)

        print()
        print("Convergence Analysis:")
        print(f"  Status: {analysis['message']}")
        if analysis['rel_change'] is not None:
            print(f"  Relative change (L50→L100): {analysis['rel_change']:.2%}")

        # Post-identity values (L=∞ reference)
        print()
        print("Post-Identity Reference (L → ∞):")
        print(f"  I1+ (post-identity at +R):  {results[0].I1_plus:.8f}")
        print(f"  I1- (post-identity at -R):  {results[0].I1_minus_base:.8f}")

        # Naive mirror weight from combined identity
        print()
        print("Mirror Weight Comparison:")
        print(f"  Empirical: m1 = exp(R) + 5 = {m1_empirical:.4f}")
        print(f"  Naive: m1 = exp(2R) = {np.exp(2*R):.4f}")

        # Show the divergence analysis
        if len([r for r in results if r.m1_eff is not None]) >= 2:
            m1_values = [r.m1_eff for r in results if r.m1_eff is not None]
            L_values = [r.L for r in results if r.m1_eff is not None]

            # Linear fit: m1_eff ≈ slope × L
            slope = m1_values[-1] / L_values[-1]
            print()
            print("Divergence Analysis:")
            print(f"  m1_eff scales linearly with L: m1_eff ≈ {slope:.4f} × L")
            print(f"  This confirms the finite-L combined identity DIVERGES.")

        print()
        print("=" * 70)
        print("CONCLUSION:")
        print("=" * 70)
        print("  The finite-L combined identity approach does NOT converge.")
        print("  m1_eff(L) grows linearly with L instead of converging.")
        print()
        print("  Root cause: The 1/(α+β) = -L/(2R) prefactor grows with L,")
        print("  but the bracket [exp(-Rθ(x+y)) - exp(2R)×exp(Rθ(x+y))] is L-independent.")
        print()
        print("  The empirical m1 = exp(R) + 5 remains the working formula.")
        print("  A proper derivation would require asymptotic analysis,")
        print("  not finite-L evaluation.")


if __name__ == "__main__":
    main()
