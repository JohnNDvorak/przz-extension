#!/usr/bin/env python3
"""
scripts/run_constrained_sweep.py
Run NOLH optimization with coefficient magnitude constraints

This script tests whether the κ = 0.521 result holds when polynomial
coefficients are constrained to match Conrey/PRZZ scale:

| Paper | Max |Coefficient| | κ Result |
|-------|-------------------|----------|
| Conrey 2011 | 0.077 | 0.4105 |
| PRZZ 2019 | 0.687 | 0.4173 |
| Our unconstrained | 2.409 | 0.5211 |

If constrained optimization (||P||_∞ ≤ 1.0) yields κ ~ 0.42, it confirms
the 0.52 result is an artifact of optimization in a regime where error
bounds haven't been validated.

Created: 2025-12-29 (Phase 65)
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.nolh_optimization.design import generate_nolh_design
from scripts.nolh_optimization.runner import run_nolh_batch


def run_constrained_sweep(
    max_coeff: float,
    n_samples: int = 49,
    n_quad: int = 40,
    seed: int = 42,
    output_dir: str = "results/constrained_sweeps",
    verbose: bool = True,
):
    """
    Run NOLH optimization with coefficient magnitude constraint.

    Args:
        max_coeff: Maximum coefficient magnitude (e.g., 1.0, 2.0, 4.0)
        n_samples: Number of NOLH design points
        n_quad: Quadrature points for integration
        seed: Random seed
        output_dir: Directory for results
        verbose: Print progress
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"CONSTRAINED OPTIMIZATION: ||P||_∞ ≤ {max_coeff}")
        print(f"{'='*70}")

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate constrained design
    # Use PRZZ baseline as center (not optimized values which violate bounds)
    design = generate_nolh_design(
        n_samples=n_samples,
        seed=seed,
        p1_pct=0.5,
        p2_pct=0.5,  # Wider search since constrained
        p3_pct=0.5,  # Wider search since constrained
        q_pct=0.25,
        use_optimized_p2p3=False,  # Start from PRZZ, not large-coeff optimum
        fix_Q=True,  # Keep Q at PRZZ to reduce dimensionality
        max_coeff_magnitude=max_coeff,
    )

    if verbose:
        print(f"\nDesign parameters:")
        print(f"  Samples: {design.n_samples}")
        print(f"  Coefficient cap: {max_coeff}")
        print(f"  Bounds applied: {len([b for b in design.bounds if abs(b[0]) <= max_coeff + 0.01 and abs(b[1]) <= max_coeff + 0.01])}/{len(design.bounds)} params")

        # Show actual bounds
        print(f"\nParameter bounds (constrained to ±{max_coeff}):")
        for i, (name, (lo, hi)) in enumerate(zip(design.param_names, design.bounds)):
            constrained = " [CONSTRAINED]" if abs(lo) == max_coeff or abs(hi) == max_coeff else ""
            print(f"  {name}: [{lo:.4f}, {hi:.4f}]{constrained}")

    # Run optimization
    checkpoint_path = out_path / f"checkpoint_cap{max_coeff}.json"
    results = run_nolh_batch(
        design=design,
        n_quad=n_quad,
        checkpoint_path=str(checkpoint_path),
        checkpoint_interval=10,
        verbose=verbose,
        fix_Q=True,
    )

    # Save final results
    results_path = out_path / f"results_cap{max_coeff}.json"
    results.save(str(results_path))

    if verbose:
        print(f"\nResults saved to: {results_path}")

    return results


def run_all_sweeps():
    """Run sweeps with multiple coefficient caps and compare results."""
    print("\n" + "="*70)
    print("CONSTRAINED OPTIMIZATION COMPARISON")
    print("Testing whether κ = 0.521 is an artifact of large coefficients")
    print("="*70)

    # Define coefficient caps to test
    # 1.0 = Conrey/PRZZ scale
    # 2.0 = 2× PRZZ (moderate)
    # 4.0 = 4× PRZZ (relaxed but still constrained)
    caps = [1.0, 2.0, 4.0]

    # Also run unconstrained for comparison
    caps.append(None)  # None = unconstrained

    all_results = {}

    for cap in caps:
        cap_label = f"{cap}" if cap else "unconstrained"

        if cap is None:
            # Unconstrained: use larger bounds starting from optimized values
            design = generate_nolh_design(
                n_samples=49,
                seed=42,
                p1_pct=0.5,
                p2_pct=0.5,
                p3_pct=0.5,
                use_optimized_p2p3=True,  # Start from large-coeff optimum
                fix_Q=True,
                max_coeff_magnitude=None,
            )

            print(f"\n{'='*70}")
            print("UNCONSTRAINED OPTIMIZATION (baseline)")
            print(f"{'='*70}")

            results = run_nolh_batch(
                design=design,
                n_quad=40,
                checkpoint_path="results/constrained_sweeps/checkpoint_unconstrained.json",
                checkpoint_interval=10,
                verbose=True,
                fix_Q=True,
            )
            results.save("results/constrained_sweeps/results_unconstrained.json")
        else:
            results = run_constrained_sweep(max_coeff=cap, verbose=True)

        if results.best:
            all_results[cap_label] = {
                'c': results.best.c,
                'kappa': results.best.kappa,
                'n_valid': results.n_valid,
                'P1': results.best.P1_tilde,
                'P2': results.best.P2_tilde,
                'P3': results.best.P3_tilde,
                'max_coeff_actual': max(
                    max(abs(x) for x in results.best.P1_tilde),
                    max(abs(x) for x in results.best.P2_tilde),
                    max(abs(x) for x in results.best.P3_tilde),
                ),
            }
        else:
            all_results[cap_label] = {'error': 'No valid results'}

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Cap':<15} {'κ':<10} {'c':<12} {'Max |coeff|':<12} {'Valid':<8}")
    print("-"*60)

    for cap_label, result in all_results.items():
        if 'error' in result:
            print(f"{cap_label:<15} {'ERROR':<10}")
        else:
            cap_display = cap_label if cap_label != "None" else "∞"
            print(f"{cap_display:<15} {result['kappa']:.4f}     {result['c']:.6f}   {result['max_coeff_actual']:.4f}       {result['n_valid']}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    przz_kappa = 0.4173
    target_kappa = 0.5211

    if "1.0" in all_results and 'kappa' in all_results["1.0"]:
        constrained_kappa = all_results["1.0"]['kappa']
        improvement_over_przz = (constrained_kappa - przz_kappa) / przz_kappa * 100

        print(f"\nWith ||P||_∞ ≤ 1.0 (Conrey/PRZZ scale):")
        print(f"  κ = {constrained_kappa:.4f}")
        print(f"  Improvement over PRZZ: {improvement_over_przz:+.2f}%")

        if constrained_kappa < 0.43:
            print(f"\n  → CONFIRMS: κ = 0.521 is likely an artifact of large coefficients")
            print(f"  → Within PRZZ coefficient bounds, improvement is modest (~{improvement_over_przz:.1f}%)")
        else:
            print(f"\n  → SUPPORTS: Improvement may be real even with constrained coefficients")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
        'analysis': {
            'przz_kappa': przz_kappa,
            'target_kappa': target_kappa,
        }
    }

    summary_path = Path("results/constrained_sweeps/summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nFull summary saved to: {summary_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run constrained NOLH optimization")
    parser.add_argument("--cap", type=float, default=None,
                       help="Coefficient magnitude cap (default: run all)")
    parser.add_argument("--samples", type=int, default=49,
                       help="Number of NOLH samples (default: 49)")
    parser.add_argument("--quad", type=int, default=40,
                       help="Quadrature points (default: 40)")
    parser.add_argument("--all", action="store_true",
                       help="Run all sweeps (1.0, 2.0, 4.0, unconstrained)")

    args = parser.parse_args()

    if args.all or args.cap is None:
        run_all_sweeps()
    else:
        run_constrained_sweep(
            max_coeff=args.cap,
            n_samples=args.samples,
            n_quad=args.quad,
        )
