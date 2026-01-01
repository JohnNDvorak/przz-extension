#!/usr/bin/env python3
"""
scripts/run_overnight_optimization.py
Overnight parallel optimization with NOLH + Differential Evolution refinement

Runs 6 configurations in parallel:
- κ (R=1.3036): Cap=1.0, Cap=2.0, Unconstrained
- κ* (R=1.1167): Cap=1.0, Cap=2.0, Unconstrained

Each configuration:
1. Deep NOLH exploration (500 samples)
2. Differential Evolution refinement of best point

Created: 2025-12-29 (Phase 65b)
Updated: 2025-12-30 (Phase 66 - added κ* and DE)
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, List
import traceback

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.nolh_optimization.design import (
    generate_nolh_design,
    P1_PRZZ, P2_PRZZ, P3_PRZZ,
    R_KAPPA, R_KAPPA_STAR,
)
from scripts.nolh_optimization.runner import run_nolh_batch, NOLHResult
from scripts.nolh_optimization.diff_evolution import differential_evolution_refine


# Configuration for overnight runs - 6 configs total
CONFIGS = [
    # κ configs (R=1.3036, 10 parameters)
    {
        "name": "kappa_cap_1.0",
        "cap": 1.0,
        "samples": 500,
        "seed": 42,
        "polynomial_set": "kappa",
        "R": R_KAPPA,
        "use_optimized": False,
    },
    {
        "name": "kappa_cap_2.0",
        "cap": 2.0,
        "samples": 500,
        "seed": 43,
        "polynomial_set": "kappa",
        "R": R_KAPPA,
        "use_optimized": False,
    },
    {
        "name": "kappa_unconstrained",
        "cap": None,
        "samples": 500,
        "seed": 44,
        "polynomial_set": "kappa",
        "R": R_KAPPA,
        "use_optimized": True,  # Start from overnight best
    },
    # κ* configs (R=1.1167, 8 parameters)
    {
        "name": "kappa_star_cap_1.0",
        "cap": 1.0,
        "samples": 500,
        "seed": 45,
        "polynomial_set": "kappa_star",
        "R": R_KAPPA_STAR,
        "use_optimized": False,
    },
    {
        "name": "kappa_star_cap_2.0",
        "cap": 2.0,
        "samples": 500,
        "seed": 46,
        "polynomial_set": "kappa_star",
        "R": R_KAPPA_STAR,
        "use_optimized": False,
    },
    {
        "name": "kappa_star_unconstrained",
        "cap": None,
        "samples": 500,
        "seed": 47,
        "polynomial_set": "kappa_star",
        "R": R_KAPPA_STAR,
        "use_optimized": False,
    },
]


def run_single_config(
    config: Dict,
    output_dir: Path,
    n_quad: int = 60,
    de_maxiter: int = 200,
    de_popsize: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Run NOLH + Differential Evolution refinement for a single configuration.

    Returns dict with results and metadata.
    """
    name = config["name"]
    cap = config["cap"]
    samples = config["samples"]
    seed = config["seed"]
    use_optimized = config.get("use_optimized", False)
    polynomial_set = config.get("polynomial_set", "kappa")
    R = config.get("R", R_KAPPA)

    start_time = time.time()
    result = {
        "config": config,
        "start_time": datetime.now().isoformat(),
        "status": "running",
    }

    try:
        if verbose:
            print(f"\n{'='*70}")
            print(f"CONFIGURATION: {name}")
            print(f"{'='*70}")
            print(f"Polynomial set: {polynomial_set}")
            print(f"R: {R}")
            print(f"Cap: {cap if cap else 'None (unconstrained)'}")
            print(f"Samples: {samples}")
            print(f"Quadrature: n={n_quad}")
            print(f"DE: maxiter={de_maxiter}, popsize={de_popsize}")

        # Phase 1: Deep NOLH exploration
        if verbose:
            print(f"\n--- Phase 1: NOLH Exploration ({samples} samples) ---")

        design = generate_nolh_design(
            n_samples=samples,
            seed=seed,
            p1_pct=0.5,
            p2_pct=0.5,
            p3_pct=0.5,
            use_optimized_p2p3=use_optimized,
            fix_Q=True,
            max_coeff_magnitude=cap,
            polynomial_set=polynomial_set,
        )

        checkpoint_path = output_dir / f"checkpoint_{name}.json"
        nolh_results = run_nolh_batch(
            design=design,
            n_quad=n_quad,
            R=R,
            checkpoint_path=str(checkpoint_path),
            checkpoint_interval=20,
            verbose=verbose,
            fix_Q=True,
            polynomial_set=polynomial_set,
        )

        # Save NOLH results
        nolh_path = output_dir / f"nolh_{name}.json"
        nolh_results.save(str(nolh_path))

        if not nolh_results.best:
            result["status"] = "failed"
            result["error"] = "No valid NOLH results"
            return result

        result["nolh"] = {
            "n_valid": nolh_results.n_valid,
            "best_c": nolh_results.best.c,
            "best_kappa": nolh_results.best.kappa,
        }

        if verbose:
            print(f"\nNOLH Best: c={nolh_results.best.c:.6f}, κ={nolh_results.best.kappa:.4f}")

        # Phase 2: Differential Evolution refinement
        if verbose:
            print(f"\n--- Phase 2: Differential Evolution Refinement ---")

        refined_result, refine_stats = differential_evolution_refine(
            start_result=nolh_results.best,
            bounds=design.bounds,
            param_names=design.param_names,
            R=R,
            n_quad=n_quad,
            fix_Q=True,
            polynomial_set=polynomial_set,
            maxiter=de_maxiter,
            popsize=de_popsize,
            seed=seed + 1000,
            verbose=verbose,
        )

        result["refinement"] = {
            "initial_c": refine_stats.initial_c,
            "final_c": refine_stats.final_c,
            "initial_kappa": refine_stats.initial_kappa,
            "final_kappa": refine_stats.final_kappa,
            "improvement_pct": refine_stats.improvement_pct,
            "n_iterations": refine_stats.n_iterations,
            "n_evaluations": refine_stats.n_evaluations,
            "convergence": refine_stats.convergence,
            "message": refine_stats.message,
        }

        result["final"] = {
            "c": refined_result.c,
            "kappa": refined_result.kappa,
            "P1_tilde": refined_result.P1_tilde,
            "P2_tilde": refined_result.P2_tilde,
            "P3_tilde": refined_result.P3_tilde,
            "Q_mono": refined_result.Q_mono,
            "params": refined_result.params,
        }

        result["status"] = "success"
        result["elapsed_sec"] = time.time() - start_time
        result["end_time"] = datetime.now().isoformat()

        # Save final result
        result_path = output_dir / f"final_{name}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        if verbose:
            print(f"\n{'='*70}")
            print(f"FINAL RESULT: {name}")
            print(f"{'='*70}")
            print(f"c = {refined_result.c:.10f}")
            print(f"κ = {refined_result.kappa:.10f}")
            print(f"Total improvement: {refine_stats.improvement_pct:.4f}%")
            print(f"Elapsed: {result['elapsed_sec']:.1f}s")

        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["elapsed_sec"] = time.time() - start_time
        result["end_time"] = datetime.now().isoformat()

        # Save error result
        error_path = output_dir / f"error_{name}.json"
        with open(error_path, 'w') as f:
            json.dump(result, f, indent=2)

        if verbose:
            print(f"\nERROR in {name}: {e}")
            traceback.print_exc()

        return result


def run_parallel(
    configs: List[Dict],
    output_dir: Path,
    n_quad: int = 60,
    de_maxiter: int = 200,
    de_popsize: int = 10,
    max_workers: int = 6,
) -> Dict:
    """Run all configurations in parallel."""
    print(f"\n{'#'*70}")
    print("OVERNIGHT OPTIMIZATION - PARALLEL MODE")
    print(f"{'#'*70}")
    print(f"Configurations: {len(configs)}")
    print(f"Workers: {max_workers}")
    print(f"DE: maxiter={de_maxiter}, popsize={de_popsize}")
    print(f"Output: {output_dir}")

    all_results = {}
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_config, config, output_dir, n_quad, de_maxiter, de_popsize, True): config["name"]
            for config in configs
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                all_results[name] = result
                print(f"\nCompleted: {name} - {result['status']}")
            except Exception as e:
                all_results[name] = {"status": "error", "error": str(e)}
                print(f"\nFailed: {name} - {e}")

    total_elapsed = time.time() - start_time

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "parallel",
        "total_elapsed_sec": total_elapsed,
        "results": all_results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print_summary(all_results, total_elapsed)
    return all_results


def run_sequential(
    configs: List[Dict],
    output_dir: Path,
    n_quad: int = 60,
    de_maxiter: int = 200,
    de_popsize: int = 10,
) -> Dict:
    """Run all configurations sequentially."""
    print(f"\n{'#'*70}")
    print("OVERNIGHT OPTIMIZATION - SEQUENTIAL MODE")
    print(f"{'#'*70}")
    print(f"Configurations: {len(configs)}")
    print(f"DE: maxiter={de_maxiter}, popsize={de_popsize}")
    print(f"Output: {output_dir}")

    all_results = {}
    start_time = time.time()

    for config in configs:
        result = run_single_config(config, output_dir, n_quad, de_maxiter, de_popsize, verbose=True)
        all_results[config["name"]] = result

    total_elapsed = time.time() - start_time

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "sequential",
        "total_elapsed_sec": total_elapsed,
        "results": all_results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print_summary(all_results, total_elapsed)
    return all_results


def print_summary(results: Dict, elapsed: float):
    """Print final summary table."""
    print(f"\n{'#'*70}")
    print("OVERNIGHT OPTIMIZATION - SUMMARY")
    print(f"{'#'*70}")
    print(f"\nTotal elapsed: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")

    print(f"\n{'Config':<25} {'Status':<10} {'R':<8} {'κ (NOLH)':<12} {'κ (Final)':<12} {'Improvement':<12}")
    print("-" * 85)

    for name, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            R = result["config"].get("R", 1.3036)
            nolh_kappa = result["nolh"]["best_kappa"]
            final_kappa = result["final"]["kappa"]
            improvement = result["refinement"]["improvement_pct"]
            print(f"{name:<25} {status:<10} {R:<8.4f} {nolh_kappa:<12.4f} {final_kappa:<12.4f} {improvement:+.4f}%")
        else:
            error = result.get("error", "unknown")[:30]
            print(f"{name:<25} {status:<10} {error}")

    # Compare to baselines
    print(f"\n--- Comparison to PRZZ Baselines ---")
    przz_kappa = 0.4173      # κ baseline (R=1.3036)
    przz_kappa_star = 0.4075 # κ* baseline (R=1.1167)

    for name, result in results.items():
        if result.get("status") == "success":
            final_kappa = result["final"]["kappa"]
            polynomial_set = result["config"].get("polynomial_set", "kappa")
            baseline = przz_kappa_star if polynomial_set == "kappa_star" else przz_kappa
            baseline_name = "κ*" if polynomial_set == "kappa_star" else "κ"
            vs_przz = (final_kappa - baseline) / baseline * 100
            print(f"{name}: κ={final_kappa:.4f}, vs PRZZ {baseline_name}: {vs_przz:+.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run overnight optimization with NOLH + Differential Evolution"
    )
    parser.add_argument(
        "--mode", choices=["parallel", "sequential"], default="parallel",
        help="Execution mode (default: parallel)"
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="NOLH samples per config (default: 500)"
    )
    parser.add_argument(
        "--quad", type=int, default=60,
        help="Quadrature points (default: 60)"
    )
    parser.add_argument(
        "--de-maxiter", type=int, default=200,
        help="DE maximum iterations (default: 200)"
    )
    parser.add_argument(
        "--de-popsize", type=int, default=10,
        help="DE population size multiplier (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/kappa_kappa_star_YYYYMMDD_HHMM)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Run only specific config (e.g., kappa_cap_1.0, kappa_star_unconstrained)"
    )
    parser.add_argument(
        "--kappa-only", action="store_true",
        help="Run only κ configs (R=1.3036)"
    )
    parser.add_argument(
        "--kappa-star-only", action="store_true",
        help="Run only κ* configs (R=1.1167)"
    )

    args = parser.parse_args()

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = Path(f"results/kappa_kappa_star_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Update configs with command-line samples
    configs = [c.copy() for c in CONFIGS]
    for config in configs:
        config["samples"] = args.samples

    # Filter to specific config if requested
    if args.config:
        configs = [c for c in configs if c["name"] == args.config]
        if not configs:
            print(f"Error: Unknown config '{args.config}'")
            print(f"Available: {[c['name'] for c in CONFIGS]}")
            sys.exit(1)
    elif args.kappa_only:
        configs = [c for c in configs if c.get("polynomial_set") == "kappa"]
    elif args.kappa_star_only:
        configs = [c for c in configs if c.get("polynomial_set") == "kappa_star"]

    print(f"\nConfigurations to run: {[c['name'] for c in configs]}")

    # Run
    if args.mode == "parallel":
        run_parallel(configs, output_dir, args.quad, args.de_maxiter, args.de_popsize, len(configs))
    else:
        run_sequential(configs, output_dir, args.quad, args.de_maxiter, args.de_popsize)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
