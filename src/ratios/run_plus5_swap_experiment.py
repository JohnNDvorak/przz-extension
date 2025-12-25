"""
src/ratios/run_plus5_swap_experiment.py
Phase 14F.2: 2x2 (R, polynomial) swap experiment

PURPOSE:
========
Determine whether delta is polynomial-driven or R-driven.

Experiment matrix:
| R           | polynomials |
|-------------|-------------|
| κ R=1.3036  | κ polys     |
| κ R=1.3036  | κ* polys    |
| κ* R=1.1167 | κ polys     |
| κ* R=1.1167 | κ* polys    |

Interpretation:
- If delta tracks polys → Euler-Maclaurin formulas are degree-sensitive
- If delta tracks R → Laurent-factor handling (1/R+γ)² is the driver
"""

from __future__ import annotations
import numpy as np

from src.ratios.przz_polynomials import load_przz_k3_polynomials
from src.ratios.j1_euler_maclaurin import compute_m1_with_mirror_assembly


def run_swap_experiment(verbose: bool = True) -> dict:
    """
    Run the 2x2 (R, polynomial) swap experiment.

    Returns:
        Dictionary with all experiment results
    """
    # Load both polynomial sets
    polys_k = load_przz_k3_polynomials("kappa")
    polys_ks = load_przz_k3_polynomials("kappa_star")

    R_k = polys_k.R  # 1.3036
    R_ks = polys_ks.R  # 1.1167
    theta = 4.0 / 7.0

    # 2x2 experiment matrix
    configs = [
        {"R": R_k, "R_name": "κ R", "polys": polys_k, "poly_name": "κ polys"},
        {"R": R_k, "R_name": "κ R", "polys": polys_ks, "poly_name": "κ* polys"},
        {"R": R_ks, "R_name": "κ* R", "polys": polys_k, "poly_name": "κ polys"},
        {"R": R_ks, "R_name": "κ* R", "polys": polys_ks, "poly_name": "κ* polys"},
    ]

    results = []
    for cfg in configs:
        decomp = compute_m1_with_mirror_assembly(
            theta=theta, R=cfg["R"], polys=cfg["polys"], K=3
        )
        results.append({
            "R_name": cfg["R_name"],
            "R": cfg["R"],
            "poly_name": cfg["poly_name"],
            "A": decomp["exp_coefficient"],
            "B": decomp["constant_offset"],
            "D": decomp["D"],
            "delta": decomp["delta"],
            "B_over_A": decomp["B_over_A"],
        })

    if verbose:
        print("=" * 80)
        print("PHASE 14F: 2x2 SWAP EXPERIMENT")
        print("=" * 80)
        print()
        print("Purpose: Determine whether delta is polynomial-driven or R-driven")
        print()

        # Print results table
        print("RESULTS:")
        print("-" * 80)
        header = f"{'R':<12} {'Polys':<12} {'A':<10} {'B':<10} {'D':<10} {'delta':<10} {'B/A':<10}"
        print(header)
        print("-" * 80)

        for r in results:
            line = (
                f"{r['R_name']:<12} "
                f"{r['poly_name']:<12} "
                f"{r['A']:<10.4f} "
                f"{r['B']:<10.4f} "
                f"{r['D']:<10.4f} "
                f"{r['delta']:<10.4f} "
                f"{r['B_over_A']:<10.4f}"
            )
            print(line)

        print("-" * 80)
        print()

        # Analysis: Does delta track R or polynomials?
        print("ANALYSIS:")
        print("-" * 80)

        # Compare fixed R, varying polys
        delta_k_polys_at_kR = results[0]["delta"]  # κ R + κ polys
        delta_ks_polys_at_kR = results[1]["delta"]  # κ R + κ* polys
        delta_k_polys_at_ksR = results[2]["delta"]  # κ* R + κ polys
        delta_ks_polys_at_ksR = results[3]["delta"]  # κ* R + κ* polys

        print()
        print("Effect of changing POLYNOMIALS (fixed R):")
        print(f"  At κ R:   κ polys → delta={delta_k_polys_at_kR:.4f}, κ* polys → delta={delta_ks_polys_at_kR:.4f}")
        print(f"  Change:   {delta_ks_polys_at_kR - delta_k_polys_at_kR:+.4f}")
        print(f"  At κ* R:  κ polys → delta={delta_k_polys_at_ksR:.4f}, κ* polys → delta={delta_ks_polys_at_ksR:.4f}")
        print(f"  Change:   {delta_ks_polys_at_ksR - delta_k_polys_at_ksR:+.4f}")

        print()
        print("Effect of changing R (fixed polynomials):")
        print(f"  κ polys:  κ R → delta={delta_k_polys_at_kR:.4f}, κ* R → delta={delta_k_polys_at_ksR:.4f}")
        print(f"  Change:   {delta_k_polys_at_ksR - delta_k_polys_at_kR:+.4f}")
        print(f"  κ* polys: κ R → delta={delta_ks_polys_at_kR:.4f}, κ* R → delta={delta_ks_polys_at_ksR:.4f}")
        print(f"  Change:   {delta_ks_polys_at_ksR - delta_ks_polys_at_kR:+.4f}")

        # Determine which driver is dominant
        poly_effect = abs(delta_ks_polys_at_kR - delta_k_polys_at_kR) + abs(delta_ks_polys_at_ksR - delta_k_polys_at_ksR)
        R_effect = abs(delta_k_polys_at_ksR - delta_k_polys_at_kR) + abs(delta_ks_polys_at_ksR - delta_ks_polys_at_kR)

        print()
        print(f"Total polynomial effect: {poly_effect:.4f}")
        print(f"Total R effect: {R_effect:.4f}")
        print()

        if poly_effect > R_effect * 1.5:
            print("CONCLUSION: delta is primarily POLYNOMIAL-DRIVEN")
            print("  → Euler-Maclaurin formulas are degree-sensitive")
        elif R_effect > poly_effect * 1.5:
            print("CONCLUSION: delta is primarily R-DRIVEN")
            print("  → Laurent-factor handling (1/R+γ)² is the main driver")
        else:
            print("CONCLUSION: delta is driven by BOTH R and polynomials")
            print("  → Mixed effects from Euler-Maclaurin and Laurent factors")

        print()
        print("=" * 80)

    return {
        "results": results,
        "R_k": R_k,
        "R_ks": R_ks,
    }


if __name__ == "__main__":
    run_swap_experiment(verbose=True)
