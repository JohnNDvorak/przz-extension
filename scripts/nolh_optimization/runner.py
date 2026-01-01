"""
scripts/nolh_optimization/runner.py
NOLH Batch Execution for K=3 Polynomial Optimization

Evaluates NOLH design points using KappaEngine and handles:
- Sequential or parallel execution
- Checkpointing and resume
- Error handling and timeouts

Created: 2025-12-28 (Phase 49)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np
import json
import time
import math
from pathlib import Path
from datetime import datetime
import sys

from .design import NOLHDesign, sample_to_polynomials, polynomials_to_engine_format


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NOLHResult:
    """Result of evaluating a single NOLH design point."""
    point_id: int
    params: Dict[str, float]  # Parameter values
    c: float                   # Main-term constant
    kappa: float              # Proportion bound
    valid: bool               # Whether computation succeeded
    error: Optional[str] = None
    elapsed_sec: float = 0.0

    # Polynomial coefficients used
    P1_tilde: List[float] = field(default_factory=list)
    P2_tilde: List[float] = field(default_factory=list)
    P3_tilde: List[float] = field(default_factory=list)
    Q_mono: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "point_id": self.point_id,
            "params": self.params,
            "c": self.c,
            "kappa": self.kappa,
            "valid": self.valid,
            "error": self.error,
            "elapsed_sec": self.elapsed_sec,
            "P1_tilde": self.P1_tilde,
            "P2_tilde": self.P2_tilde,
            "P3_tilde": self.P3_tilde,
            "Q_mono": self.Q_mono,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NOLHResult":
        """Create from dict."""
        return cls(
            point_id=d["point_id"],
            params=d["params"],
            c=d["c"],
            kappa=d["kappa"],
            valid=d["valid"],
            error=d.get("error"),
            elapsed_sec=d.get("elapsed_sec", 0.0),
            P1_tilde=d.get("P1_tilde", []),
            P2_tilde=d.get("P2_tilde", []),
            P3_tilde=d.get("P3_tilde", []),
            Q_mono=d.get("Q_mono", []),
        )


@dataclass
class NOLHBatchResults:
    """Container for all NOLH batch results."""
    design: NOLHDesign
    results: List[NOLHResult]
    start_time: str
    end_time: Optional[str] = None
    n_quad: int = 40

    @property
    def n_valid(self) -> int:
        """Number of valid results."""
        return sum(1 for r in self.results if r.valid)

    @property
    def best(self) -> Optional[NOLHResult]:
        """Best result (lowest c among valid)."""
        valid = [r for r in self.results if r.valid]
        if not valid:
            return None
        return min(valid, key=lambda r: r.c)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "design": self.design.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "n_quad": self.n_quad,
            "summary": {
                "n_total": len(self.results),
                "n_valid": self.n_valid,
                "best_id": self.best.point_id if self.best else None,
                "best_c": self.best.c if self.best else None,
                "best_kappa": self.best.kappa if self.best else None,
            }
        }

    def save(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "NOLHBatchResults":
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            d = json.load(f)
        return cls(
            design=NOLHDesign.from_dict(d["design"]),
            results=[NOLHResult.from_dict(r) for r in d["results"]],
            start_time=d["start_time"],
            end_time=d.get("end_time"),
            n_quad=d.get("n_quad", 40),
        )


# =============================================================================
# SINGLE POINT EVALUATION
# =============================================================================

def evaluate_design_point(
    sample: np.ndarray,
    point_id: int,
    param_names: List[str],
    n_quad: int = 40,
    R: float = 1.3036,
    theta: float = 4/7,
    fix_Q: bool = False,
    polynomial_set: str = "kappa",
) -> NOLHResult:
    """
    Evaluate a single NOLH design point.

    Args:
        sample: Array of parameter values (10 if fix_Q, 13 otherwise for κ; 8/11 for κ*)
        point_id: Identifier for this point
        param_names: Names of parameters
        n_quad: Quadrature points for integration
        R: PRZZ R parameter (1.3036 for κ, 1.1167 for κ*)
        theta: PRZZ theta parameter
        fix_Q: If True, sample only has P1/P2/P3 params; use baseline Q values
        polynomial_set: "kappa" or "kappa_star" - determines polynomial structure

    Returns:
        NOLHResult with c, kappa, and metadata
    """
    start = time.time()

    # Convert sample to polynomial format
    polys = sample_to_polynomials(sample, fix_Q=fix_Q, polynomial_set=polynomial_set)
    engine_polys = polynomials_to_engine_format(polys, polynomial_set=polynomial_set)

    # Build params dict for logging
    params = {name: float(sample[i]) for i, name in enumerate(param_names)}

    try:
        from src.kappa_engine import KappaEngine

        engine = KappaEngine(
            P1_coeffs=engine_polys["P1_tilde"],
            P2_coeffs=engine_polys["P2_tilde"],
            P3_coeffs=engine_polys["P3_tilde"],
            Q_coeffs=engine_polys["Q_mono"],
            theta=theta,
            K=3,
            R=R,
            n_quad=n_quad,
        )

        result = engine.compute_kappa()

        # Validate result
        c = result.c
        kappa = result.kappa

        if not np.isfinite(c) or not np.isfinite(kappa):
            raise ValueError(f"Non-finite result: c={c}, kappa={kappa}")
        if c <= 0:
            raise ValueError(f"Non-positive c: {c}")
        if not (0 < kappa < 1):
            raise ValueError(f"kappa out of range: {kappa}")

        elapsed = time.time() - start

        return NOLHResult(
            point_id=point_id,
            params=params,
            c=c,
            kappa=kappa,
            valid=True,
            error=None,
            elapsed_sec=elapsed,
            P1_tilde=engine_polys["P1_tilde"],
            P2_tilde=engine_polys["P2_tilde"],
            P3_tilde=engine_polys["P3_tilde"],
            Q_mono=engine_polys["Q_mono"],
        )

    except Exception as e:
        elapsed = time.time() - start
        return NOLHResult(
            point_id=point_id,
            params=params,
            c=float('nan'),
            kappa=float('nan'),
            valid=False,
            error=str(e),
            elapsed_sec=elapsed,
            P1_tilde=engine_polys["P1_tilde"],
            P2_tilde=engine_polys["P2_tilde"],
            P3_tilde=engine_polys["P3_tilde"],
            Q_mono=engine_polys["Q_mono"],
        )


# =============================================================================
# BATCH EXECUTION
# =============================================================================

def run_nolh_batch(
    design: NOLHDesign,
    n_quad: int = 40,
    R: float = 1.3036,
    theta: float = 4/7,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 5,
    verbose: bool = True,
    fix_Q: bool = False,
    polynomial_set: str = "kappa",
) -> NOLHBatchResults:
    """
    Run all NOLH design points sequentially.

    Args:
        design: NOLH design to evaluate
        n_quad: Quadrature points
        R: PRZZ R parameter (1.3036 for κ, 1.1167 for κ*)
        theta: PRZZ theta parameter
        checkpoint_path: Path to save intermediate results
        checkpoint_interval: Save checkpoint every N points
        verbose: Print progress
        fix_Q: If True, use baseline Q values (design only has P1/P2/P3 params)
        polynomial_set: "kappa" or "kappa_star" - determines polynomial structure

    Returns:
        NOLHBatchResults with all evaluations
    """
    start_time = datetime.now().isoformat()
    results = []

    # Resume from checkpoint if exists
    start_idx = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            existing = NOLHBatchResults.load(checkpoint_path)
            results = existing.results
            start_idx = len(results)
            if verbose:
                print(f"Resuming from checkpoint: {start_idx}/{design.n_samples} points done")
        except Exception as e:
            if verbose:
                print(f"Could not load checkpoint: {e}")

    # Get baseline for comparison
    baseline_c = 2.0165371858  # Current optimized κ result

    if verbose:
        print(f"\nNOLH Batch Evaluation")
        print("=" * 60)
        print(f"Design points: {design.n_samples}")
        print(f"Parameters: {design.n_params} ({'Q fixed' if fix_Q else 'Q variable'})")
        print(f"Quadrature: n={n_quad}")
        print(f"Baseline c: {baseline_c:.6f}")
        print()

    for i in range(start_idx, design.n_samples):
        sample = design.samples[i]

        if verbose:
            print(f"[{i+1}/{design.n_samples}] Point {i}...", end=" ", flush=True)

        result = evaluate_design_point(
            sample=sample,
            point_id=i,
            param_names=design.param_names,
            n_quad=n_quad,
            R=R,
            theta=theta,
            fix_Q=fix_Q,
            polynomial_set=polynomial_set,
        )

        results.append(result)

        if verbose:
            if result.valid:
                delta = (result.c / baseline_c - 1) * 100
                marker = " <-- BEST" if result.c < baseline_c else ""
                print(f"c={result.c:.6f} ({delta:+.3f}%) κ={result.kappa:.4f}{marker}")
            else:
                print(f"FAILED: {result.error}")

        # Checkpoint
        if checkpoint_path and (i + 1) % checkpoint_interval == 0:
            batch = NOLHBatchResults(
                design=design,
                results=results,
                start_time=start_time,
                n_quad=n_quad,
            )
            batch.save(checkpoint_path)
            if verbose:
                print(f"  [Checkpoint saved: {len(results)} points]")

    # Final save
    end_time = datetime.now().isoformat()
    batch = NOLHBatchResults(
        design=design,
        results=results,
        start_time=start_time,
        end_time=end_time,
        n_quad=n_quad,
    )

    if verbose:
        print()
        print("=" * 60)
        print(f"Completed: {batch.n_valid}/{design.n_samples} valid")
        if batch.best:
            print(f"Best: point {batch.best.point_id}, c={batch.best.c:.6f}, κ={batch.best.kappa:.4f}")

    return batch


def run_nolh_batch_parallel(
    design: NOLHDesign,
    n_quad: int = 40,
    n_workers: int = 4,
    R: float = 1.3036,
    theta: float = 4/7,
    verbose: bool = True,
    fix_Q: bool = False,
    polynomial_set: str = "kappa",
) -> NOLHBatchResults:
    """
    Run NOLH design points in parallel using ProcessPoolExecutor.

    Args:
        design: NOLH design to evaluate
        n_quad: Quadrature points
        n_workers: Number of parallel workers
        R: PRZZ R parameter (1.3036 for κ, 1.1167 for κ*)
        theta: PRZZ theta parameter
        verbose: Print progress
        fix_Q: If True, use baseline Q values
        polynomial_set: "kappa" or "kappa_star" - determines polynomial structure

    Returns:
        NOLHBatchResults with all evaluations
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    start_time = datetime.now().isoformat()

    if verbose:
        print(f"\nNOLH Parallel Batch Evaluation")
        print("=" * 60)
        print(f"Design points: {design.n_samples}")
        print(f"Workers: {n_workers}")
        print(f"Quadrature: n={n_quad}")
        print()

    # Prepare arguments for each point
    args_list = [
        (design.samples[i], i, design.param_names, n_quad, R, theta, fix_Q, polynomial_set)
        for i in range(design.n_samples)
    ]

    results = [None] * design.n_samples

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(evaluate_design_point, *args): args[1]  # args[1] is point_id
            for args in args_list
        }

        completed = 0
        for future in as_completed(futures):
            point_id = futures[future]
            try:
                result = future.result()
                results[point_id] = result
                completed += 1
                if verbose:
                    if result.valid:
                        print(f"[{completed}/{design.n_samples}] Point {point_id}: c={result.c:.6f}")
                    else:
                        print(f"[{completed}/{design.n_samples}] Point {point_id}: FAILED")
            except Exception as e:
                results[point_id] = NOLHResult(
                    point_id=point_id,
                    params={},
                    c=float('nan'),
                    kappa=float('nan'),
                    valid=False,
                    error=str(e),
                )
                completed += 1

    end_time = datetime.now().isoformat()

    batch = NOLHBatchResults(
        design=design,
        results=results,
        start_time=start_time,
        end_time=end_time,
        n_quad=n_quad,
    )

    if verbose:
        print()
        print("=" * 60)
        print(f"Completed: {batch.n_valid}/{design.n_samples} valid")
        if batch.best:
            print(f"Best: point {batch.best.point_id}, c={batch.best.c:.6f}, κ={batch.best.kappa:.4f}")

    return batch


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from .design import generate_nolh_design

    print("NOLH Runner Test")
    print("=" * 60)

    # Generate small test design
    design = generate_nolh_design(n_samples=5, seed=42)

    print(f"\nRunning {design.n_samples} test points...")

    # Run batch
    results = run_nolh_batch(
        design=design,
        n_quad=40,
        verbose=True,
    )

    print(f"\nResults summary:")
    print(f"  Valid: {results.n_valid}/{design.n_samples}")
    if results.best:
        print(f"  Best c: {results.best.c:.6f}")
        print(f"  Best κ: {results.best.kappa:.4f}")

    # Save test
    test_path = "/tmp/nolh_results_test.json"
    results.save(test_path)
    print(f"\nSaved to: {test_path}")
