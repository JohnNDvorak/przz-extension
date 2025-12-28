#!/usr/bin/env python3
"""
Phase 41.1: Error Budget Attribution Analysis

Computes m_needed and S34_needed for each benchmark to identify
whether the residual gap is attributable to m or S34.

GPT's Attribution Formulas:
1. m_needed = (c_target - S12+ - S34) / S12-
2. S34_needed = c_target - S12+ - m_derived * S12-

Decision gate:
- If kappa and kappa* require opposite shifts in m but consistent shifts
  in S34 (or vice versa), identify the real source of the residual.

Created: 2025-12-27 (Phase 41)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass
from typing import Dict, List

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.decomposition import compute_decomposition, compute_mirror_multiplier
from src.mirror_transform_paper_exact import compute_S12_paper_sum


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""
    name: str
    R: float
    theta: float
    c_target: float
    kappa_target: float
    loader: callable


@dataclass
class ErrorBudget:
    """Error budget attribution for a single benchmark."""

    # Input parameters
    benchmark: str
    R: float
    theta: float
    K: int

    # Component values
    S12_plus: float
    S12_minus: float
    S34: float

    # Derived values
    m_derived: float         # m from production formula
    c_computed: float        # c = S12+ + m_derived * S12- + S34
    c_target: float
    c_gap_pct: float         # (c_computed/c_target - 1) * 100

    # Attribution: what m would be needed to hit target
    m_needed: float          # (c_target - S12+ - S34) / S12-
    delta_m: float           # (m_needed/m_derived - 1) as ratio

    # Attribution: what S34 would be needed to hit target
    S34_needed: float        # c_target - S12+ - m_derived * S12-
    delta_S34: float         # S34_needed - S34 (absolute)
    delta_S34_pct: float     # (S34_needed/S34 - 1) * 100 if S34 != 0


def get_benchmarks() -> List[BenchmarkConfig]:
    """Return benchmark configurations."""
    theta = 4 / 7
    return [
        BenchmarkConfig(
            name="kappa",
            R=1.3036,
            theta=theta,
            c_target=2.137454406132173,
            kappa_target=0.417293962,
            loader=load_przz_polynomials,
        ),
        BenchmarkConfig(
            name="kappa_star",
            R=1.1167,
            theta=theta,
            c_target=1.9379524124677437,
            kappa_target=0.407511457,
            loader=load_przz_polynomials_kappa_star,
        ),
    ]


def compute_error_budget(config: BenchmarkConfig, K: int = 3, n_quad: int = 60) -> ErrorBudget:
    """
    Compute error budget attribution for a benchmark.

    Args:
        config: Benchmark configuration
        K: Number of mollifier pieces
        n_quad: Quadrature points

    Returns:
        ErrorBudget with all attribution values
    """
    # Load polynomials
    P1, P2, P3, Q = config.loader()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Compute decomposition using canonical function
    decomp = compute_decomposition(
        theta=config.theta,
        R=config.R,
        K=K,
        polynomials=polynomials,
        kernel_regime="paper",
        n_quad=n_quad,
        mirror_formula="derived",
    )

    # Extract components
    S12_plus = decomp.S12_plus
    S12_minus = decomp.S12_minus
    S34 = decomp.S34
    m_derived = decomp.mirror_mult
    c_computed = decomp.total

    # Compute gaps
    c_gap_pct = (c_computed / config.c_target - 1) * 100

    # Attribution 1: What m would hit the target?
    # c_target = S12+ + m_needed * S12- + S34
    # m_needed = (c_target - S12+ - S34) / S12-
    m_needed = (config.c_target - S12_plus - S34) / S12_minus
    delta_m = (m_needed / m_derived - 1)  # as ratio

    # Attribution 2: What S34 would hit the target?
    # c_target = S12+ + m_derived * S12- + S34_needed
    # S34_needed = c_target - S12+ - m_derived * S12-
    S34_needed = config.c_target - S12_plus - m_derived * S12_minus
    delta_S34 = S34_needed - S34
    delta_S34_pct = ((S34_needed / S34 - 1) * 100) if abs(S34) > 1e-15 else float('inf')

    return ErrorBudget(
        benchmark=config.name,
        R=config.R,
        theta=config.theta,
        K=K,
        S12_plus=S12_plus,
        S12_minus=S12_minus,
        S34=S34,
        m_derived=m_derived,
        c_computed=c_computed,
        c_target=config.c_target,
        c_gap_pct=c_gap_pct,
        m_needed=m_needed,
        delta_m=delta_m,
        S34_needed=S34_needed,
        delta_S34=delta_S34,
        delta_S34_pct=delta_S34_pct,
    )


def print_error_budget_table(budgets: List[ErrorBudget]) -> None:
    """Print error budget attribution table."""
    print("=" * 80)
    print("PHASE 41.1: ERROR BUDGET ATTRIBUTION")
    print("=" * 80)
    print()

    # Component values table
    print("COMPONENT VALUES")
    print("-" * 80)
    print(f"{'Benchmark':<12} | {'S12+':<12} | {'S12-':<12} | {'S34':<12} | {'m_derived':<10}")
    print("-" * 80)
    for b in budgets:
        print(f"{b.benchmark:<12} | {b.S12_plus:<12.6f} | {b.S12_minus:<12.6f} | {b.S34:<12.6f} | {b.m_derived:<10.6f}")
    print()

    # Computed vs target
    print("COMPUTED VS TARGET")
    print("-" * 80)
    print(f"{'Benchmark':<12} | {'c_computed':<14} | {'c_target':<14} | {'c_gap %':<12}")
    print("-" * 80)
    for b in budgets:
        print(f"{b.benchmark:<12} | {b.c_computed:<14.8f} | {b.c_target:<14.8f} | {b.c_gap_pct:+.4f}%")
    print()

    # m Attribution
    print("M ATTRIBUTION: What m would hit target?")
    print("-" * 80)
    print(f"{'Benchmark':<12} | {'m_derived':<12} | {'m_needed':<12} | {'delta_m %':<12}")
    print("-" * 80)
    for b in budgets:
        delta_m_pct = b.delta_m * 100
        print(f"{b.benchmark:<12} | {b.m_derived:<12.6f} | {b.m_needed:<12.6f} | {delta_m_pct:+.4f}%")
    print()

    # S34 Attribution
    print("S34 ATTRIBUTION: What S34 would hit target?")
    print("-" * 80)
    print(f"{'Benchmark':<12} | {'S34':<12} | {'S34_needed':<12} | {'delta_S34':<12} | {'delta %':<10}")
    print("-" * 80)
    for b in budgets:
        print(f"{b.benchmark:<12} | {b.S34:<12.6f} | {b.S34_needed:<12.6f} | {b.delta_S34:+.8f} | {b.delta_S34_pct:+.4f}%")
    print()


def print_decision_gate(budgets: List[ErrorBudget]) -> Dict:
    """Print decision gate analysis. Returns decision info."""
    print("=" * 80)
    print("DECISION GATE ANALYSIS")
    print("=" * 80)
    print()

    if len(budgets) < 2:
        print("Need at least 2 benchmarks for decision gate.")
        return {"decision": "insufficient_data"}

    b1, b2 = budgets[0], budgets[1]

    # Check m shifts
    m_shift_same_sign = (b1.delta_m * b2.delta_m) > 0

    # Check S34 shifts
    S34_shift_same_sign = (b1.delta_S34 * b2.delta_S34) > 0

    print(f"  {b1.benchmark}: delta_m = {b1.delta_m*100:+.4f}%, delta_S34 = {b1.delta_S34:+.8f}")
    print(f"  {b2.benchmark}: delta_m = {b2.delta_m*100:+.4f}%, delta_S34 = {b2.delta_S34:+.8f}")
    print()

    print(f"  M shifts same direction:   {'YES' if m_shift_same_sign else 'NO'}")
    print(f"  S34 shifts same direction: {'YES' if S34_shift_same_sign else 'NO'}")
    print()

    decision = {}

    if m_shift_same_sign and not S34_shift_same_sign:
        print("  INTERPRETATION: m has systematic error, S34 is benchmark-dependent")
        print("  ROUTE A RECOMMENDED: Derive polynomial-aware g(P,Q,R,K,theta)")
        decision = {"route": "A", "reason": "m_systematic_S34_varies"}
    elif S34_shift_same_sign and not m_shift_same_sign:
        print("  INTERPRETATION: S34 has systematic error, m is correct per-benchmark")
        print("  ROUTE B RECOMMENDED: Audit S34 derivation for missing factor")
        decision = {"route": "B", "reason": "S34_systematic_m_varies"}
    elif m_shift_same_sign and S34_shift_same_sign:
        print("  INTERPRETATION: Both have systematic errors in same direction")
        print("  CHECK: May be overall normalization issue")
        decision = {"route": "both", "reason": "both_systematic"}
    else:
        print("  INTERPRETATION: Opposite shifts in BOTH m and S34")
        print("  This confirms Phase 40 finding: no single correction works")
        print("  RECOMMENDATION: Accept ±0.15% as production accuracy OR")
        print("                  implement polynomial-aware derived functional")
        decision = {"route": "functional", "reason": "both_vary"}

    return decision


def main():
    """Main entry point."""
    benchmarks = get_benchmarks()
    K = 3
    n_quad = 60

    print()
    print(f"Computing error budgets for K={K}, n_quad={n_quad}...")
    print()

    budgets = [compute_error_budget(config, K, n_quad) for config in benchmarks]

    print_error_budget_table(budgets)
    decision = print_decision_gate(budgets)

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("  1. Run Phase 41.2 (component convergence) to verify numerical stability")
    print("  2. Based on decision gate, proceed to Route A or Route B")
    print("  3. If Route A: Derive g(P,Q,R,K,theta) correction factor")
    print("  4. If Route B: Audit S34 for missing normalization/factor")
    print()

    return budgets, decision


if __name__ == "__main__":
    main()
