#!/usr/bin/env python3
"""
scripts/run_nolh_exploration.py
Main Entry Point for NOLH Multi-Parameter Optimization

Runs a 49-point NOLH exploration of the 13-parameter K=3 polynomial space.
Identifies optimal regions and parameter importance.

Usage:
    python scripts/run_nolh_exploration.py [--samples N] [--quad N] [--resume]

Created: 2025-12-28 (Phase 49)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.nolh_optimization import (
    generate_nolh_design,
    run_nolh_batch,
    compute_main_effects,
    fit_response_surface,
)
from scripts.nolh_optimization.analysis import print_analysis_summary
from scripts.nolh_optimization.runner import NOLHBatchResults


# =============================================================================
# CONSTANTS
# =============================================================================

RESULTS_DIR = Path(__file__).parent.parent / "candidate_files"
DESIGN_PATH = RESULTS_DIR / "nolh_design.json"
RESULTS_PATH = RESULTS_DIR / "nolh_results.json"
CHECKPOINT_PATH = RESULTS_DIR / "nolh_checkpoint.json"
BEST_PATH = RESULTS_DIR / "nolh_best.json"
ANALYSIS_PATH = RESULTS_DIR / "nolh_analysis.json"


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def run_exploration(
    n_samples: int = 49,
    n_quad: int = 40,
    resume: bool = True,
    seed: int = 42,
) -> NOLHBatchResults:
    """
    Run full NOLH exploration.

    Args:
        n_samples: Number of NOLH design points
        n_quad: Quadrature points
        resume: Resume from checkpoint if exists
        seed: Random seed

    Returns:
        NOLHBatchResults with all evaluations
    """
    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load or generate design
    if resume and DESIGN_PATH.exists():
        print(f"Loading existing design from {DESIGN_PATH}")
        design = generate_nolh_design(n_samples=n_samples, seed=seed)
        # Verify it matches
        if design.n_samples != n_samples:
            print(f"Warning: Design has {design.n_samples} samples, requested {n_samples}")
    else:
        print(f"Generating new NOLH design with {n_samples} samples...")
        design = generate_nolh_design(n_samples=n_samples, seed=seed)
        design.save(str(DESIGN_PATH))
        print(f"Design saved to {DESIGN_PATH}")

    # Run batch evaluation
    checkpoint = str(CHECKPOINT_PATH) if resume else None
    results = run_nolh_batch(
        design=design,
        n_quad=n_quad,
        checkpoint_path=checkpoint,
        checkpoint_interval=5,
        verbose=True,
    )

    # Save final results
    results.save(str(RESULTS_PATH))
    print(f"\nResults saved to {RESULTS_PATH}")

    return results


def analyze_results(results: NOLHBatchResults) -> dict:
    """
    Perform statistical analysis on results.

    Returns:
        Analysis summary dict
    """
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    analysis = {}

    # Main effects
    effects = compute_main_effects(results)
    analysis["main_effects"] = effects.effects
    analysis["std_errors"] = effects.std_errors
    analysis["top_params"] = [name for name, _ in effects.top_n(5)]

    print("\nTop 5 Parameters (by effect on c):")
    for name, effect in effects.top_n(5):
        se = effects.std_errors[name]
        direction = "decreases c" if effect < 0 else "increases c"
        significance = "|effect|/SE" if se > 0 else "N/A"
        if se > 0 and se != float('inf'):
            t_stat = abs(effect) / se
            print(f"  {name}: {effect:+.4f} (SE={se:.4f}, t={t_stat:.2f}) - {direction}")
        else:
            print(f"  {name}: {effect:+.4f} - {direction}")

    # Response surface (if enough points)
    if results.n_valid >= 20:
        try:
            surface = fit_response_surface(results)
            analysis["r_squared"] = surface.r_squared
            analysis["linear_coeffs"] = surface.linear_coeffs
            print(f"\nResponse Surface R²: {surface.r_squared:.4f}")
        except Exception as e:
            print(f"\nResponse surface fitting failed: {e}")
            analysis["r_squared"] = None

    # Best point summary
    best = results.best
    if best:
        analysis["best_point"] = {
            "point_id": best.point_id,
            "c": best.c,
            "kappa": best.kappa,
            "params": best.params,
        }

    return analysis


def save_best_candidate(results: NOLHBatchResults):
    """Save best candidate to JSON."""
    best = results.best
    if not best:
        print("No valid results to save as best candidate")
        return

    candidate = {
        "source": f"nolh_point_{best.point_id}",
        "timestamp": datetime.now().isoformat(),
        "c": best.c,
        "kappa": best.kappa,
        "params": best.params,
        "P1_tilde": best.P1_tilde,
        "P2_tilde": best.P2_tilde,
        "P3_tilde": best.P3_tilde,
        "Q_mono": best.Q_mono,
    }

    with open(BEST_PATH, 'w') as f:
        json.dump(candidate, f, indent=2)

    print(f"Best candidate saved to {BEST_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NOLH Multi-Parameter Optimization for K=3 Polynomials"
    )
    parser.add_argument(
        "--samples", type=int, default=49,
        help="Number of NOLH design points (default: 49)"
    )
    parser.add_argument(
        "--quad", type=int, default=40,
        help="Quadrature points (default: 40)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh (don't resume from checkpoint)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NOLH Multi-Parameter Optimization")
    print("Phase 49: K=3 Polynomial Space Exploration")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  Design points: {args.samples}")
    print(f"  Quadrature: n={args.quad}")
    print(f"  Seed: {args.seed}")
    print(f"  Resume: {not args.no_resume}")

    # Run exploration
    results = run_exploration(
        n_samples=args.samples,
        n_quad=args.quad,
        resume=not args.no_resume,
        seed=args.seed,
    )

    # Print summary
    print_analysis_summary(results)

    # Statistical analysis
    analysis = analyze_results(results)

    # Save analysis
    with open(ANALYSIS_PATH, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved to {ANALYSIS_PATH}")

    # Save best candidate
    save_best_candidate(results)

    # Final summary
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    if results.best:
        baseline_c = 2.0165371858
        baseline_kappa = 0.4620
        improvement = (results.best.kappa - baseline_kappa) * 100
        print(f"\nBest result:")
        print(f"  Point: {results.best.point_id}")
        print(f"  c:     {results.best.c:.6f} (baseline: {baseline_c:.6f})")
        print(f"  κ:     {results.best.kappa:.4f} (baseline: {baseline_kappa:.4f})")
        print(f"  Improvement: {improvement:+.2f}%")


if __name__ == "__main__":
    main()
