"""
scripts/nolh_optimization/diff_evolution.py
Differential Evolution refinement for polynomial optimization

Uses scipy.optimize.differential_evolution for global optimization
starting from best NOLH point.

Created: 2025-12-30 (Phase 66)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import time
from datetime import datetime

from scipy.optimize import differential_evolution

from .runner import NOLHResult, evaluate_design_point


@dataclass
class DEResult:
    """Result of Differential Evolution optimization."""
    initial_c: float
    initial_kappa: float
    final_c: float
    final_kappa: float
    improvement_pct: float
    n_evaluations: int
    n_iterations: int
    elapsed_sec: float
    best_params: Dict[str, float]
    convergence: bool
    message: str


def differential_evolution_refine(
    start_result: NOLHResult,
    bounds: List[Tuple[float, float]],
    param_names: List[str],
    R: float = 1.3036,
    theta: float = 4/7,
    n_quad: int = 60,
    fix_Q: bool = True,
    polynomial_set: str = "kappa",
    maxiter: int = 200,
    popsize: int = 10,
    mutation: Tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    tol: float = 1e-7,
    atol: float = 0,
    seed: Optional[int] = None,
    verbose: bool = True,
    callback_interval: int = 10,
) -> Tuple[NOLHResult, DEResult]:
    """
    Refine best point using Differential Evolution.

    Uses scipy's differential_evolution with 'best1bin' strategy,
    starting from the best NOLH result.

    Args:
        start_result: NOLHResult from NOLH exploration phase
        bounds: Parameter bounds [(lo, hi), ...]
        param_names: Parameter names
        R: PRZZ R parameter (1.3036 for κ, 1.1167 for κ*)
        theta: PRZZ theta parameter (default 4/7)
        n_quad: Quadrature points
        fix_Q: If True, Q is fixed at baseline values
        polynomial_set: "kappa" or "kappa_star"
        maxiter: Maximum iterations (generations)
        popsize: Population size multiplier (actual pop = popsize * n_params)
        mutation: Mutation constant (F) or range for dithering
        recombination: Recombination constant (CR)
        tol: Relative tolerance for convergence
        atol: Absolute tolerance for convergence
        seed: Random seed for reproducibility
        verbose: Print progress
        callback_interval: Print progress every N iterations

    Returns:
        (best_result, de_stats) tuple
    """
    start_time = time.time()

    initial_c = start_result.c
    initial_kappa = start_result.kappa
    n_evals = [0]  # Use list to allow mutation in nested function
    n_iters = [0]
    best_so_far = [initial_c]

    if verbose:
        print(f"\nDifferential Evolution Refinement")
        print("=" * 60)
        print(f"Starting: c={initial_c:.6f}, κ={initial_kappa:.4f}")
        print(f"Parameters: {len(param_names)}, R={R}")
        print(f"Settings: maxiter={maxiter}, popsize={popsize}, mutation={mutation}")

    # Objective function
    def objective(params):
        n_evals[0] += 1
        result = evaluate_design_point(
            sample=params,
            point_id=n_evals[0],
            param_names=param_names,
            n_quad=n_quad,
            R=R,
            theta=theta,
            fix_Q=fix_Q,
            polynomial_set=polynomial_set,
        )
        if result.valid:
            if result.c < best_so_far[0]:
                best_so_far[0] = result.c
            return result.c
        else:
            return 1e10  # Invalid configuration

    # Callback for progress reporting
    def callback(xk, convergence):
        n_iters[0] += 1
        if verbose and n_iters[0] % callback_interval == 0:
            current_c = objective(xk)
            n_evals[0] -= 1  # Don't count callback evaluation
            improvement = (initial_c - current_c) / initial_c * 100
            print(f"  Iter {n_iters[0]}: c={current_c:.6f} ({improvement:+.4f}%), "
                  f"evals={n_evals[0]}, conv={convergence:.2e}")

    # Initialize population with best NOLH result
    x0 = np.array([start_result.params[name] for name in param_names])

    # Run Differential Evolution
    result = differential_evolution(
        objective,
        bounds=bounds,
        x0=x0,
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        tol=tol,
        atol=atol,
        seed=seed,
        callback=callback if verbose else None,
        disp=False,  # We handle our own progress output
        polish=False,  # Don't do final L-BFGS-B polish
        init='latinhypercube',
        updating='immediate',
    )

    elapsed = time.time() - start_time

    # Evaluate final point to get full result
    final_result = evaluate_design_point(
        sample=result.x,
        point_id=0,
        param_names=param_names,
        n_quad=n_quad,
        R=R,
        theta=theta,
        fix_Q=fix_Q,
        polynomial_set=polynomial_set,
    )

    improvement_pct = (initial_c - final_result.c) / initial_c * 100

    if verbose:
        print("-" * 60)
        print(f"Final: c={final_result.c:.6f}, κ={final_result.kappa:.4f}")
        print(f"Total improvement: {improvement_pct:.4f}%")
        print(f"Evaluations: {n_evals[0]}, Iterations: {n_iters[0]}")
        print(f"Convergence: {result.success} ({result.message})")
        print(f"Time: {elapsed:.1f}s")

    stats = DEResult(
        initial_c=initial_c,
        initial_kappa=initial_kappa,
        final_c=final_result.c,
        final_kappa=final_result.kappa,
        improvement_pct=improvement_pct,
        n_evaluations=n_evals[0],
        n_iterations=n_iters[0],
        elapsed_sec=elapsed,
        best_params=final_result.params,
        convergence=result.success,
        message=result.message,
    )

    return final_result, stats


if __name__ == "__main__":
    # Quick test
    from .design import generate_nolh_design
    from .runner import run_nolh_batch

    print("Differential Evolution Refinement Test")
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
        # Refine with DE
        print("\nPhase 2: Differential Evolution Refinement")
        refined, stats = differential_evolution_refine(
            results.best,
            bounds=design.bounds,
            param_names=design.param_names,
            maxiter=20,
            popsize=5,
            n_quad=40,
            fix_Q=True,
            verbose=True,
        )

        print(f"\nImprovement: {stats.improvement_pct:.4f}%")
        print(f"Convergence: {stats.convergence}")
