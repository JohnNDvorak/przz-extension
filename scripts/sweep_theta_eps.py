#!/usr/bin/env python3
"""
scripts/sweep_theta_eps.py
Phase 48.3: θ-Robustness Gate

GPT's critical requirement: Show κ > 0.5 for θ = 4/7 - ε.

PRZZ works with θ = 4/7 − ε and then takes limits.
If κ > 0.5 requires being exactly at θ = 4/7 and collapses at θ = 0.5710,
referees will force you to address that.

This script sweeps ε ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2} and shows:
1. κ stays > 0.5 for at least one nontrivial ε
2. Reports the observed slope dκ/dθ numerically

Created: 2025-12-28 (Phase 48 - Adversarial Verification)
"""

import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kappa_engine import KappaEngine


def load_optimal_polynomials():
    """Load the validated optimal polynomials."""
    path = Path(__file__).parent.parent / "data" / "optimal_polynomials.json"
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("PHASE 48.3: θ-ROBUSTNESS GATE")
    print("Sweeping θ = 4/7 - ε to verify κ > 0.5 is not knife-edge")
    print("=" * 70)

    data = load_optimal_polynomials()

    R = 1.3036
    theta_exact = 4/7
    K = 3
    n_quad = 60

    # Epsilon values to sweep
    epsilon_values = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    results = []

    print(f"\n  Parameters:")
    print(f"    R = {R}")
    print(f"    θ_exact = 4/7 = {theta_exact:.10f}")
    print(f"    K = {K}")
    print(f"    n_quad = {n_quad}")

    print(f"\n  {'ε':>10} | {'θ':>12} | {'c':>14} | {'κ':>12} | Status")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*14}-+-{'-'*12}-+-------")

    for eps in epsilon_values:
        theta = theta_exact - eps

        engine = KappaEngine(
            P1_coeffs=data['P1_tilde'],
            P2_coeffs=data['P2_tilde'],
            P3_coeffs=data['P3_tilde'],
            Q_coeffs=data['Q_mono'],
            theta=theta,
            K=K,
            R=R,
            n_quad=n_quad,
        )
        result = engine.compute_kappa()

        kappa = result.kappa
        c = result.c
        status = "κ>0.5 ✓" if kappa > 0.5 else "κ≤0.5 ✗"

        results.append({
            'eps': eps,
            'theta': theta,
            'c': c,
            'kappa': kappa,
            'passes': kappa > 0.5
        })

        eps_str = f"{eps:.0e}" if eps > 0 else "0"
        print(f"  {eps_str:>10} | {theta:.10f} | {c:>14.10f} | {kappa:>12.10f} | {status}")

    # Compute slope dκ/dθ
    print("\n" + "-" * 70)
    print("SLOPE ANALYSIS")
    print("-" * 70)

    # Use finite difference between ε=0 and ε=1e-3
    r0 = next(r for r in results if r['eps'] == 0.0)
    r1 = next(r for r in results if r['eps'] == 1e-3)

    delta_kappa = r1['kappa'] - r0['kappa']
    delta_theta = r1['theta'] - r0['theta']
    slope = delta_kappa / delta_theta

    print(f"\n  Slope dκ/dθ (from ε=0 to ε=1e-3):")
    print(f"    Δκ = {delta_kappa:+.10f}")
    print(f"    Δθ = {delta_theta:+.10f}")
    print(f"    dκ/dθ ≈ {slope:.6f}")

    # Check for stability
    all_pass = all(r['passes'] for r in results)
    max_eps_passing = max(r['eps'] for r in results if r['passes'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  All ε values pass (κ > 0.5): {'YES ✓' if all_pass else 'NO ✗'}")
    print(f"  Largest ε where κ > 0.5: {max_eps_passing:.0e}" if max_eps_passing > 0 else "  Largest ε where κ > 0.5: only at ε=0")

    if all_pass:
        print(f"\n  CONCLUSION: κ > 0.5 is ROBUST for θ ∈ [{theta_exact - 1e-2:.4f}, {theta_exact:.4f}]")
        print(f"              This is NOT a knife-edge result.")
    else:
        failing = [r for r in results if not r['passes']]
        print(f"\n  WARNING: κ < 0.5 at ε = {[r['eps'] for r in failing]}")

    print("\n" + "=" * 70)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
