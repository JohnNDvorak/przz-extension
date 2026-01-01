"""
scripts/nolh_optimization/perturbation.py
Perturbation-based local refinement for polynomial optimization

After NOLH exploration finds a good region, this module refines the best
point using small random perturbations to escape local minima.

Created: 2025-12-29 (Phase 65b)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import time
from datetime import datetime

from .runner import NOLHResult, evaluate_design_point
from .design import get_parameter_bounds


@dataclass
class PerturbationResult:
    """Result of perturbation refinement."""
    initial_c: float
    initial_kappa: float
    final_c: float
    final_kappa: float
    improvement_pct: float
    n_rounds: int
    n_evaluations: int
    elapsed_sec: float
    best_params: Dict[str, float]
    history: List[Dict]  # Per-round history


def generate_perturbations(
    center: np.ndarray,
    n_perturbations: int,
    pct: float,
    bounds: List[Tuple[float, float]],
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate random perturbations around a center point.

    Args:
        center: Current best parameter values
        n_perturbations: Number of perturbations to generate
        pct: Percentage perturbation (e.g., 0.05 for ±5%)
        bounds: Parameter bounds (lo, hi) for each dimension
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_perturbations, n_params) with perturbed points
    """
    if seed is not None:
        np.random.seed(seed)

    n_params = len(center)
    perturbations = np.zeros((n_perturbations, n_params))

    for i in range(n_perturbations):
        for j in range(n_params):
            # Random perturbation in [-pct, +pct] of current value
            delta = center[j] * pct * (2 * np.random.random() - 1)
            new_val = center[j] + delta

            # Clip to bounds
            lo, hi = bounds[j]
            new_val = max(lo, min(hi, new_val))
            perturbations[i, j] = new_val

    return perturbations


def perturbation_refine(
    start_result: NOLHResult,
    bounds: List[Tuple[float, float]],
    param_names: List[str],
    max_rounds: int = 10,
    perturbation_pct: float = 0.05,
    n_perturbations: int = 50,
    n_quad: int = 60,
    R: float = 1.3036,
    theta: float = 4/7,
    fix_Q: bool = True,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Tuple[NOLHResult, PerturbationResult]:
    """
    Refine a starting point using local perturbations.

    Args:
        start_result: NOLHResult from NOLH exploration
        bounds: Parameter bounds
        param_names: Parameter names
        max_rounds: Maximum refinement rounds
        perturbation_pct: Perturbation percentage (default 5%)
        n_perturbations: Perturbations per round
        n_quad: Quadrature points
        R: PRZZ R parameter
        theta: PRZZ theta parameter
        fix_Q: If True, Q is fixed at PRZZ values
        verbose: Print progress
        seed: Random seed

    Returns:
        (best_result, refinement_stats)
    """
    start_time = time.time()

    # Extract starting point
    current_best = start_result
    current_params = np.array([start_result.params[name] for name in param_names])

    initial_c = current_best.c
    initial_kappa = current_best.kappa

    if verbose:
        print(f"\nPerturbation Refinement")
        print("=" * 60)
        print(f"Starting: c={initial_c:.6f}, κ={initial_kappa:.4f}")
        print(f"Perturbation: ±{perturbation_pct*100:.0f}%, {n_perturbations} per round")

    history = []
    n_evaluations = 0
    no_improvement_count = 0

    for round_num in range(max_rounds):
        round_seed = seed + round_num if seed else None

        # Generate perturbations
        perturbations = generate_perturbations(
            current_params, n_perturbations, perturbation_pct, bounds, seed=round_seed
        )

        # Evaluate all perturbations
        round_results = []
        for i, perturbed in enumerate(perturbations):
            result = evaluate_design_point(
                sample=perturbed,
                point_id=i,
                param_names=param_names,
                n_quad=n_quad,
                R=R,
                theta=theta,
                fix_Q=fix_Q,
            )
            if result.valid:
                round_results.append((perturbed, result))
            n_evaluations += 1

        if not round_results:
            if verbose:
                print(f"  Round {round_num}: No valid results")
            no_improvement_count += 1
            if no_improvement_count >= 2:
                break
            continue

        # Find best in this round
        best_perturbed, best_result = min(round_results, key=lambda x: x[1].c)

        round_info = {
            'round': round_num,
            'n_valid': len(round_results),
            'best_c': best_result.c,
            'current_best_c': current_best.c,
        }

        if best_result.c < current_best.c:
            improvement = (current_best.c - best_result.c) / current_best.c * 100
            round_info['improvement_pct'] = improvement

            if verbose:
                print(f"  Round {round_num}: c={best_result.c:.6f} (↓{improvement:.4f}%), "
                      f"κ={best_result.kappa:.4f}")

            current_best = best_result
            current_params = best_perturbed
            no_improvement_count = 0
        else:
            round_info['improvement_pct'] = 0
            if verbose:
                print(f"  Round {round_num}: No improvement (best c={best_result.c:.6f})")
            no_improvement_count += 1

        history.append(round_info)

        if no_improvement_count >= 2:
            if verbose:
                print(f"  Converged (no improvement in 2 rounds)")
            break

    elapsed = time.time() - start_time
    improvement_pct = (initial_c - current_best.c) / initial_c * 100

    if verbose:
        print("-" * 60)
        print(f"Final: c={current_best.c:.6f}, κ={current_best.kappa:.4f}")
        print(f"Total improvement: {improvement_pct:.4f}%")
        print(f"Evaluations: {n_evaluations}, Rounds: {len(history)}")
        print(f"Time: {elapsed:.1f}s")

    stats = PerturbationResult(
        initial_c=initial_c,
        initial_kappa=initial_kappa,
        final_c=current_best.c,
        final_kappa=current_best.kappa,
        improvement_pct=improvement_pct,
        n_rounds=len(history),
        n_evaluations=n_evaluations,
        elapsed_sec=elapsed,
        best_params=current_best.params,
        history=history,
    )

    return current_best, stats


def adaptive_perturbation_refine(
    start_result: NOLHResult,
    bounds: List[Tuple[float, float]],
    param_names: List[str],
    max_rounds: int = 20,
    initial_pct: float = 0.10,
    min_pct: float = 0.01,
    decay_rate: float = 0.7,
    n_perturbations: int = 50,
    n_quad: int = 60,
    R: float = 1.3036,
    theta: float = 4/7,
    fix_Q: bool = True,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> Tuple[NOLHResult, PerturbationResult]:
    """
    Adaptive perturbation refinement with decreasing step size.

    Starts with larger perturbations and shrinks when no improvement found,
    allowing both exploration and fine-tuning.

    Args:
        initial_pct: Starting perturbation (default 10%)
        min_pct: Minimum perturbation (default 1%)
        decay_rate: Factor to reduce perturbation on no improvement (default 0.7)
        (other args same as perturbation_refine)
    """
    start_time = time.time()

    current_best = start_result
    current_params = np.array([start_result.params[name] for name in param_names])

    initial_c = current_best.c
    initial_kappa = current_best.kappa
    current_pct = initial_pct

    if verbose:
        print(f"\nAdaptive Perturbation Refinement")
        print("=" * 60)
        print(f"Starting: c={initial_c:.6f}, κ={initial_kappa:.4f}")
        print(f"Initial perturbation: ±{initial_pct*100:.0f}%, min: ±{min_pct*100:.0f}%")

    history = []
    n_evaluations = 0

    for round_num in range(max_rounds):
        round_seed = seed + round_num if seed else None

        # Generate perturbations at current scale
        perturbations = generate_perturbations(
            current_params, n_perturbations, current_pct, bounds, seed=round_seed
        )

        # Evaluate
        round_results = []
        for i, perturbed in enumerate(perturbations):
            result = evaluate_design_point(
                sample=perturbed,
                point_id=i,
                param_names=param_names,
                n_quad=n_quad,
                R=R,
                theta=theta,
                fix_Q=fix_Q,
            )
            if result.valid:
                round_results.append((perturbed, result))
            n_evaluations += 1

        if not round_results:
            current_pct *= decay_rate
            if current_pct < min_pct:
                if verbose:
                    print(f"  Reached minimum perturbation, stopping")
                break
            continue

        best_perturbed, best_result = min(round_results, key=lambda x: x[1].c)

        round_info = {
            'round': round_num,
            'pct': current_pct,
            'n_valid': len(round_results),
            'best_c': best_result.c,
        }

        if best_result.c < current_best.c:
            improvement = (current_best.c - best_result.c) / current_best.c * 100
            round_info['improvement_pct'] = improvement

            if verbose:
                print(f"  Round {round_num} (±{current_pct*100:.1f}%): "
                      f"c={best_result.c:.6f} (↓{improvement:.4f}%), κ={best_result.kappa:.4f}")

            current_best = best_result
            current_params = best_perturbed
        else:
            round_info['improvement_pct'] = 0
            current_pct *= decay_rate

            if verbose:
                print(f"  Round {round_num}: No improvement, reducing to ±{current_pct*100:.1f}%")

            if current_pct < min_pct:
                if verbose:
                    print(f"  Reached minimum perturbation, stopping")
                break

        history.append(round_info)

    elapsed = time.time() - start_time
    improvement_pct = (initial_c - current_best.c) / initial_c * 100

    if verbose:
        print("-" * 60)
        print(f"Final: c={current_best.c:.6f}, κ={current_best.kappa:.4f}")
        print(f"Total improvement: {improvement_pct:.4f}%")
        print(f"Evaluations: {n_evaluations}")
        print(f"Time: {elapsed:.1f}s")

    stats = PerturbationResult(
        initial_c=initial_c,
        initial_kappa=initial_kappa,
        final_c=current_best.c,
        final_kappa=current_best.kappa,
        improvement_pct=improvement_pct,
        n_rounds=len(history),
        n_evaluations=n_evaluations,
        elapsed_sec=elapsed,
        best_params=current_best.params,
        history=history,
    )

    return current_best, stats


if __name__ == "__main__":
    # Quick test
    from .design import generate_nolh_design
    from .runner import run_nolh_batch

    print("Perturbation Refinement Test")
    print("=" * 60)

    # Generate small NOLH design
    design = generate_nolh_design(
        n_samples=10,
        seed=42,
        fix_Q=True,
        max_coeff_magnitude=1.0,
    )

    # Run NOLH
    print("\nPhase 1: NOLH Exploration (10 samples)")
    results = run_nolh_batch(design, n_quad=40, verbose=True, fix_Q=True)

    if results.best:
        # Refine best
        print("\nPhase 2: Perturbation Refinement")
        refined, stats = perturbation_refine(
            results.best,
            bounds=design.bounds,
            param_names=design.param_names,
            max_rounds=5,
            perturbation_pct=0.05,
            n_perturbations=20,
            n_quad=40,
            fix_Q=True,
            verbose=True,
        )

        print(f"\nImprovement: {stats.improvement_pct:.4f}%")
