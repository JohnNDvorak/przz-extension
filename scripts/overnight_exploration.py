#!/usr/bin/env python3
"""
scripts/overnight_exploration.py
Phase 47c: Overnight Polynomial Exploration

Runs comprehensive exploration of P2/P3 polynomial space overnight.

Strategies:
1. Line search along winning direction (alpha = 0 to 25)
2. Random jitter with multiple seeds and sigma values
3. Combination of direction + jitter

Results saved to overnight_results.json

Usage:
    python scripts/overnight_exploration.py

Created: 2025-12-28 (Phase 47c overnight)
"""

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kappa_engine import KappaEngine


@dataclass
class ExplorationResult:
    id: str
    c: float
    kappa: float
    delta_c_pct: float
    strategy: str
    params: dict
    converged: bool


# Targets
C_TARGET = 2.13745440613217263636
KAPPA_TARGET = 0.417293962


def load_baseline():
    with open('candidate_files/przz_baseline.json') as f:
        return json.load(f)


def load_winner():
    with open('candidate_files/P23_jitter_small_2.json') as f:
        return json.load(f)


def score(P1, P2, P3, Q, R=1.3036, n_quad=60):
    engine = KappaEngine(
        P1_coeffs=P1,
        P2_coeffs=P2,
        P3_coeffs=P3,
        Q_coeffs=Q,
        theta=4/7,
        K=3,
        R=R,
        n_quad=n_quad,
    )
    result = engine.compute_kappa()
    return result.c, result.kappa


def check_convergence(P1, P2, P3, Q, R=1.3036):
    """Check if result converges across quadrature levels."""
    c40, _ = score(P1, P2, P3, Q, R, n_quad=40)
    c60, _ = score(P1, P2, P3, Q, R, n_quad=60)
    drift = abs(c60 - c40) / c40 if c40 > 0 else 1.0
    return drift < 1e-6


def explore_line_search(baseline, winner, results: List):
    """Strategy 1: Line search along winning direction."""
    print("\n[STRATEGY 1] Line search along winning direction")
    print("-" * 60)

    d2 = np.array(winner['P2_tilde']) - np.array(baseline['P2_tilde'])
    d3 = np.array(winner['P3_tilde']) - np.array(baseline['P3_tilde'])

    best_c = C_TARGET
    best_alpha = 0

    for alpha in np.arange(0, 25.5, 0.5):
        P2 = list(np.array(baseline['P2_tilde']) + alpha * d2)
        P3 = list(np.array(baseline['P3_tilde']) + alpha * d3)

        c, k = score(baseline['P1_tilde'], P2, P3, baseline['Q_mono'], n_quad=40)
        delta = (c / C_TARGET - 1) * 100

        if c < best_c:
            best_c = c
            best_alpha = alpha

        results.append(ExplorationResult(
            id=f"line_alpha_{alpha:.1f}",
            c=c,
            kappa=k,
            delta_c_pct=delta,
            strategy="line_search",
            params={"alpha": alpha},
            converged=False,  # Will verify best ones
        ))

        if alpha % 5 == 0:
            print(f"  alpha={alpha:.1f}: c={c:.6f}, Δc={delta:+.4f}%")

    print(f"  Best alpha: {best_alpha:.1f}, c={best_c:.6f}")
    return best_alpha, d2, d3


def explore_random_jitter(baseline, results: List, n_seeds=100):
    """Strategy 2: Random jitter on P2/P3."""
    print("\n[STRATEGY 2] Random jitter on P2/P3")
    print("-" * 60)

    sigmas = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]

    for sigma in sigmas:
        print(f"  sigma={sigma*100:.1f}%: ", end="", flush=True)
        improvements = 0

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)

            P2 = list(np.array(baseline['P2_tilde']) * (1 + sigma * rng.normal(size=3)))
            P3 = list(np.array(baseline['P3_tilde']) * (1 + sigma * rng.normal(size=3)))

            c, k = score(baseline['P1_tilde'], P2, P3, baseline['Q_mono'], n_quad=40)
            delta = (c / C_TARGET - 1) * 100

            if delta < 0:
                improvements += 1

            results.append(ExplorationResult(
                id=f"jitter_s{sigma}_seed{seed}",
                c=c,
                kappa=k,
                delta_c_pct=delta,
                strategy="random_jitter",
                params={"sigma": sigma, "seed": seed},
                converged=False,
            ))

        print(f"{improvements}/{n_seeds} improved")


def explore_direction_plus_jitter(baseline, d2, d3, best_alpha, results: List, n_seeds=50):
    """Strategy 3: Best direction + additional jitter."""
    print("\n[STRATEGY 3] Best direction + additional jitter")
    print("-" * 60)

    # Start from best alpha
    P2_base = np.array(baseline['P2_tilde']) + best_alpha * d2
    P3_base = np.array(baseline['P3_tilde']) + best_alpha * d3

    for sigma in [0.005, 0.01, 0.02]:
        print(f"  alpha={best_alpha:.1f} + sigma={sigma*100:.1f}%: ", end="", flush=True)
        improvements = 0

        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)  # Different seed range

            P2 = list(P2_base * (1 + sigma * rng.normal(size=3)))
            P3 = list(P3_base * (1 + sigma * rng.normal(size=3)))

            c, k = score(baseline['P1_tilde'], P2, P3, baseline['Q_mono'], n_quad=40)
            delta = (c / C_TARGET - 1) * 100

            if delta < -0.4:  # Better than best_alpha alone
                improvements += 1

            results.append(ExplorationResult(
                id=f"dir_jitter_a{best_alpha}_s{sigma}_seed{seed}",
                c=c,
                kappa=k,
                delta_c_pct=delta,
                strategy="direction_plus_jitter",
                params={"alpha": best_alpha, "sigma": sigma, "seed": seed},
                converged=False,
            ))

        print(f"{improvements}/{n_seeds} beat direction alone")


def verify_top_candidates(baseline, results: List, top_n=20):
    """Verify top candidates at high quadrature."""
    print("\n[VERIFICATION] Checking top candidates at n=60,80")
    print("-" * 60)

    # Sort by c
    sorted_results = sorted(results, key=lambda r: r.c)

    verified = []
    for r in sorted_results[:top_n]:
        # Reconstruct polynomials based on strategy
        if r.strategy == "line_search":
            with open('candidate_files/P23_jitter_small_2.json') as f:
                winner = json.load(f)
            d2 = np.array(winner['P2_tilde']) - np.array(baseline['P2_tilde'])
            d3 = np.array(winner['P3_tilde']) - np.array(baseline['P3_tilde'])
            alpha = r.params["alpha"]
            P2 = list(np.array(baseline['P2_tilde']) + alpha * d2)
            P3 = list(np.array(baseline['P3_tilde']) + alpha * d3)

        elif r.strategy == "random_jitter":
            sigma = r.params["sigma"]
            seed = r.params["seed"]
            rng = np.random.default_rng(seed)
            P2 = list(np.array(baseline['P2_tilde']) * (1 + sigma * rng.normal(size=3)))
            P3 = list(np.array(baseline['P3_tilde']) * (1 + sigma * rng.normal(size=3)))

        elif r.strategy == "direction_plus_jitter":
            with open('candidate_files/P23_jitter_small_2.json') as f:
                winner = json.load(f)
            d2 = np.array(winner['P2_tilde']) - np.array(baseline['P2_tilde'])
            d3 = np.array(winner['P3_tilde']) - np.array(baseline['P3_tilde'])
            alpha = r.params["alpha"]
            sigma = r.params["sigma"]
            seed = r.params["seed"]
            rng = np.random.default_rng(1000 + seed)
            P2_base = np.array(baseline['P2_tilde']) + alpha * d2
            P3_base = np.array(baseline['P3_tilde']) + alpha * d3
            P2 = list(P2_base * (1 + sigma * rng.normal(size=3)))
            P3 = list(P3_base * (1 + sigma * rng.normal(size=3)))
        else:
            continue

        # Verify at higher quadrature
        c60, k60 = score(baseline['P1_tilde'], P2, P3, baseline['Q_mono'], n_quad=60)
        c80, k80 = score(baseline['P1_tilde'], P2, P3, baseline['Q_mono'], n_quad=80)

        drift = abs(c80 - c60) / c60 if c60 > 0 else 1.0
        converged = drift < 1e-5

        delta = (c80 / C_TARGET - 1) * 100

        verified.append({
            "id": r.id,
            "c_n80": c80,
            "kappa_n80": k80,
            "delta_c_pct": delta,
            "converged": converged,
            "drift": drift,
            "strategy": r.strategy,
            "params": r.params,
            "P2_tilde": P2,
            "P3_tilde": P3,
        })

        status = "CONVERGED" if converged else "DRIFT"
        print(f"  {r.id}: c={c80:.6f}, Δc={delta:+.4f}%, {status}")

    return verified


def main():
    start_time = time.time()
    print("=" * 70)
    print("OVERNIGHT POLYNOMIAL EXPLORATION")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    baseline = load_baseline()
    winner = load_winner()

    results: List[ExplorationResult] = []

    # Strategy 1: Line search
    best_alpha, d2, d3 = explore_line_search(baseline, winner, results)

    # Strategy 2: Random jitter
    explore_random_jitter(baseline, results, n_seeds=200)

    # Strategy 3: Direction + jitter
    explore_direction_plus_jitter(baseline, d2, d3, best_alpha, results, n_seeds=100)

    # Verify top candidates
    verified = verify_top_candidates(baseline, results, top_n=30)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)
    print(f"Total candidates explored: {len(results)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")

    # Best results
    best = min(verified, key=lambda v: v["c_n80"])
    print(f"\nBEST RESULT:")
    print(f"  ID: {best['id']}")
    print(f"  c = {best['c_n80']:.10f}")
    print(f"  kappa = {best['kappa_n80']:.10f}")
    print(f"  Delta c vs PRZZ: {best['delta_c_pct']:+.6f}%")
    print(f"  Converged: {best['converged']}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": elapsed / 60,
        "total_explored": len(results),
        "baseline_c": C_TARGET,
        "baseline_kappa": KAPPA_TARGET,
        "best": best,
        "top_10": verified[:10],
        "all_verified": verified,
    }

    with open("overnight_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to overnight_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
