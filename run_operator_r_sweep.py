"""
run_operator_r_sweep.py
Claude Run 2: R-sweep for the GO variant (grid/i1_only)

Tests the winning variant across R ∈ [1.0, 1.5] to check weight stability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

from src.evaluate import (
    compute_c_paper_operator_unified,
    solve_two_weight_operator,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0

# Base weights from 2×2 solve (for reference)
M1_BASE = 6.198
M2_BASE = 8.052


@dataclass
class RSweepResult:
    """Result for one R value."""
    R_kappa: float
    R_kappa_star: float
    m1: float
    m2: float
    cond: float
    det: float
    m1_delta: float
    m2_delta: float
    is_reasonable: bool


def run_r_sweep(
    R_values: List[float],
    polys_kappa: dict,
    polys_kappa_star: dict,
    c_target_kappa: float,
    c_target_kappa_star: float,
    n: int,
    n_quad_a: int,
    normalization: str = "grid",
    lift_scope: str = "i1_only",
) -> List[RSweepResult]:
    """Run R-sweep for the specified variant."""

    results = []

    # For κ*, scale R proportionally to maintain same relative shift
    # Original: κ at R=1.3036, κ* at R=1.1167
    # Ratio: 1.1167/1.3036 = 0.8566
    R_RATIO = 1.1167 / 1.3036

    for R_k in R_values:
        R_ks = R_k * R_RATIO

        # Run for κ benchmark
        result_k = compute_c_paper_operator_unified(
            theta=THETA, R=R_k, n=n, polynomials=polys_kappa,
            n_quad_a=n_quad_a, verbose=False,
            normalization=normalization, lift_scope=lift_scope,
        )

        # Run for κ* benchmark
        result_ks = compute_c_paper_operator_unified(
            theta=THETA, R=R_ks, n=n, polynomials=polys_kappa_star,
            n_quad_a=n_quad_a, verbose=False,
            normalization=normalization, lift_scope=lift_scope,
        )

        # Solve 2×2 system
        op_solve = solve_two_weight_operator(
            result_k, result_ks,
            c_target_k=c_target_kappa, c_target_k_star=c_target_kappa_star,
            use_operator_channels=True,
        )

        m1 = op_solve["m1"]
        m2 = op_solve["m2"]
        cond = op_solve["cond"]
        det = op_solve["det"]

        m1_delta = m1 - M1_BASE
        m2_delta = m2 - M2_BASE

        # Reasonable if positive and within ×2 of base
        is_reasonable = (
            m1 > 0 and m2 > 0 and
            0.5 * M1_BASE < m1 < 2.0 * M1_BASE and
            0.5 * M2_BASE < m2 < 2.0 * M2_BASE
        )

        results.append(RSweepResult(
            R_kappa=R_k,
            R_kappa_star=R_ks,
            m1=m1, m2=m2, cond=cond, det=det,
            m1_delta=m1_delta, m2_delta=m2_delta,
            is_reasonable=is_reasonable,
        ))

    return results


def main():
    parser = argparse.ArgumentParser(description="R-sweep for GO variant (Claude Run 2)")
    parser.add_argument("--n", type=int, default=60, help="u/t quadrature points")
    parser.add_argument("--n-quad-a", type=int, default=40, help="Case C a-integral quadrature points")
    parser.add_argument("--normalization", type=str, default="grid", help="Normalization mode")
    parser.add_argument("--scope", type=str, default="i1_only", help="Lift scope")
    args = parser.parse_args()

    print("=" * 100)
    print(f"R-SWEEP FOR GO VARIANT (Claude Run 2)")
    print(f"Variant: {args.normalization}/{args.scope}")
    print("=" * 100)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}

    # Reference c targets (at original R values)
    KAPPA_C_TARGET = 2.137
    KAPPA_STAR_C_TARGET = 1.938

    # R values to sweep
    R_values = [1.0, 1.15, 1.3036, 1.4, 1.5]

    print(f"Base weights: m1={M1_BASE:.3f}, m2={M2_BASE:.3f}")
    print(f"Quadrature: n={args.n}, n_quad_a={args.n_quad_a}")
    print()

    # Run sweep
    print("Running R-sweep...")
    results = run_r_sweep(
        R_values=R_values,
        polys_kappa=polys_kappa,
        polys_kappa_star=polys_kappa_star,
        c_target_kappa=KAPPA_C_TARGET,
        c_target_kappa_star=KAPPA_STAR_C_TARGET,
        n=args.n, n_quad_a=args.n_quad_a,
        normalization=args.normalization,
        lift_scope=args.scope,
    )

    # Results table
    print()
    print("=" * 100)
    print("R-SWEEP RESULTS")
    print("=" * 100)
    print()
    print(f"{'R_κ':>8} {'R_κ*':>8} {'m1':>10} {'m2':>10} {'cond':>8} {'m1_Δ':>10} {'m2_Δ':>10} {'OK?':>5}")
    print("-" * 100)

    for r in results:
        ok_flag = "GO" if r.is_reasonable and r.cond < 25 else "NO"
        print(f"{r.R_kappa:>8.4f} {r.R_kappa_star:>8.4f} {r.m1:>10.2f} {r.m2:>10.2f} {r.cond:>8.1f} {r.m1_delta:>+10.2f} {r.m2_delta:>+10.2f} {ok_flag:>5}")

    print()

    # Stability analysis
    print("=" * 100)
    print("STABILITY ANALYSIS")
    print("=" * 100)
    print()

    go_results = [r for r in results if r.is_reasonable and r.cond < 25]

    if len(go_results) == len(results):
        print("ALL R values give GO verdict")
        print("→ Operator variant is R-stable")
    elif len(go_results) >= 3:
        print(f"{len(go_results)}/{len(results)} R values give GO verdict")
        print("→ Operator variant is reasonably stable across R")
    elif len(go_results) > 0:
        print(f"Only {len(go_results)}/{len(results)} R values give GO verdict")
        print("→ Operator variant has limited R-stability")
    else:
        print("NO R values give GO verdict")
        print("→ Operator variant fails R-stability test")

    print()

    # Weight variation
    m1_values = [r.m1 for r in results if r.m1 > 0]
    m2_values = [r.m2 for r in results if r.m2 > 0]

    if m1_values and m2_values:
        m1_min, m1_max = min(m1_values), max(m1_values)
        m2_min, m2_max = min(m2_values), max(m2_values)

        print("Weight ranges across positive-weight R values:")
        print(f"  m1: [{m1_min:.2f}, {m1_max:.2f}] (span: {m1_max - m1_min:.2f})")
        print(f"  m2: [{m2_min:.2f}, {m2_max:.2f}] (span: {m2_max - m2_min:.2f})")
        print()

        # Is the span small compared to the base values?
        m1_span_ratio = (m1_max - m1_min) / M1_BASE
        m2_span_ratio = (m2_max - m2_min) / M2_BASE

        if m1_span_ratio < 0.5 and m2_span_ratio < 0.5:
            print("Weight variation is small (<50% of base)")
            print("→ Q-lift operator captures stable structure")
        else:
            print(f"Weight variation is large (m1: {m1_span_ratio*100:.0f}%, m2: {m2_span_ratio*100:.0f}% of base)")
            print("→ Q-lift operator is R-sensitive")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
