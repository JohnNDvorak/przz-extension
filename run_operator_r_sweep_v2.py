"""
run_operator_r_sweep_v2.py
R-sweep for top GO variants from variant matrix (Claude Run 2)

Tests the best variants across R ∈ [1.0, 1.5] to check weight stability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

from src.evaluate import (
    compute_c_paper_operator_v2,
    solve_two_weight_operator,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0


@dataclass
class RSweepResult:
    """Result for one R value."""
    R_kappa: float
    R_kappa_star: float
    m1: float
    m2: float
    cond: float
    m1_base: float
    m2_base: float
    cond_base: float
    m1_delta: float
    m2_delta: float
    is_go: bool


def run_r_sweep(
    R_values: List[float],
    polys_kappa: dict,
    polys_kappa_star: dict,
    c_target_kappa: float,
    c_target_kappa_star: float,
    n: int,
    n_quad_a: int,
    normalization: str,
    scope: str,
    sigma: float,
) -> List[RSweepResult]:
    """Run R-sweep for the specified variant."""

    results = []
    R_RATIO = 1.1167 / 1.3036  # κ* to κ ratio

    for R_k in R_values:
        R_ks = R_k * R_RATIO

        result_k = compute_c_paper_operator_v2(
            theta=THETA, R=R_k, n=n, polynomials=polys_kappa,
            n_quad_a=n_quad_a, verbose=False,
            normalization=normalization, lift_scope=scope, sigma=sigma,
        )

        result_ks = compute_c_paper_operator_v2(
            theta=THETA, R=R_ks, n=n, polynomials=polys_kappa_star,
            n_quad_a=n_quad_a, verbose=False,
            normalization=normalization, lift_scope=scope, sigma=sigma,
        )

        base_solve = solve_two_weight_operator(
            result_k, result_ks,
            c_target_k=c_target_kappa, c_target_k_star=c_target_kappa_star,
            use_operator_channels=False,
        )
        op_solve = solve_two_weight_operator(
            result_k, result_ks,
            c_target_k=c_target_kappa, c_target_k_star=c_target_kappa_star,
            use_operator_channels=True,
        )

        m1 = op_solve["m1"]
        m2 = op_solve["m2"]
        cond = op_solve["cond"]

        m1_base = base_solve["m1"]
        m2_base = base_solve["m2"]
        cond_base = base_solve["cond"]

        m1_delta = m1 - m1_base
        m2_delta = m2 - m2_base

        is_go = (
            m1 > 0 and m2 > 0 and cond < 25 and
            m1_base / 3 < m1 < m1_base * 3 and
            m2_base / 3 < m2 < m2_base * 3
        )

        results.append(RSweepResult(
            R_kappa=R_k, R_kappa_star=R_ks,
            m1=m1, m2=m2, cond=cond,
            m1_base=m1_base, m2_base=m2_base, cond_base=cond_base,
            m1_delta=m1_delta, m2_delta=m2_delta,
            is_go=is_go,
        ))

    return results


def main():
    parser = argparse.ArgumentParser(description="R-sweep for top GO variants (Claude Run 2)")
    parser.add_argument("--n", type=int, default=60, help="u/t quadrature points")
    parser.add_argument("--n-quad-a", type=int, default=40, help="Case C a-integral quadrature points")
    args = parser.parse_args()

    print("=" * 100)
    print("R-SWEEP FOR TOP GO VARIANTS (Claude Run 2)")
    print("=" * 100)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}

    KAPPA_C_TARGET = 2.137
    KAPPA_STAR_C_TARGET = 1.938

    R_values = [1.0, 1.15, 1.3036, 1.4, 1.5]

    # Top variants to test (from variant matrix results)
    variants = [
        ("l2", "i2_only", 0.5),
        ("grid", "i2_only", 0.5),
        ("grid", "i1_only", 0.5),
        ("l2", "i1_only", 0.5),
    ]

    print(f"Quadrature: n={args.n}, n_quad_a={args.n_quad_a}")
    print()

    for norm, scope, sigma in variants:
        print("=" * 100)
        print(f"VARIANT: {norm}/{scope}/σ={sigma}")
        print("=" * 100)
        print()

        results = run_r_sweep(
            R_values=R_values,
            polys_kappa=polys_kappa,
            polys_kappa_star=polys_kappa_star,
            c_target_kappa=KAPPA_C_TARGET,
            c_target_kappa_star=KAPPA_STAR_C_TARGET,
            n=args.n, n_quad_a=args.n_quad_a,
            normalization=norm, scope=scope, sigma=sigma,
        )

        print(
            f"{'R_κ':>8} {'R_κ*':>8}  "
            f"{'m1_base':>10} {'m2_base':>10} {'cond_b':>7}  "
            f"{'m1_op':>10} {'m2_op':>10} {'cond_o':>7}  "
            f"{'Δm1':>10} {'Δm2':>10} {'OK':>5}"
        )
        print("-" * 110)

        go_count = 0
        for r in results:
            ok_flag = "GO" if r.is_go else "NO"
            if r.is_go:
                go_count += 1
            print(
                f"{r.R_kappa:>8.4f} {r.R_kappa_star:>8.4f}  "
                f"{r.m1_base:>10.2f} {r.m2_base:>10.2f} {r.cond_base:>7.1f}  "
                f"{r.m1:>10.2f} {r.m2:>10.2f} {r.cond:>7.1f}  "
                f"{r.m1_delta:>+10.2f} {r.m2_delta:>+10.2f} {ok_flag:>5}"
            )

        print()
        
        # Weight stability analysis
        m1_values = [r.m1 for r in results if r.m1 > 0]
        m2_values = [r.m2 for r in results if r.m2 > 0]
        
        if m1_values and m2_values:
            m1_span = max(m1_values) - min(m1_values)
            m2_span = max(m2_values) - min(m2_values)
            base_m1_values = [r.m1_base for r in results if r.m1_base > 0]
            base_m2_values = [r.m2_base for r in results if r.m2_base > 0]
            m1_ref = base_m1_values[len(base_m1_values) // 2] if base_m1_values else 1.0
            m2_ref = base_m2_values[len(base_m2_values) // 2] if base_m2_values else 1.0

            m1_span_pct = m1_span / m1_ref * 100
            m2_span_pct = m2_span / m2_ref * 100
            
            print(f"Weight stability: m1 span = {m1_span:.2f} ({m1_span_pct:.0f}%), "
                  f"m2 span = {m2_span:.2f} ({m2_span_pct:.0f}%)")
            print(f"R-sweep verdict: {go_count}/{len(results)} GO")
            
            if m2_span_pct < 30:
                print("-> m2 is R-stable (<30% variation)")
            if m1_span_pct < 50:
                print("-> m1 is reasonably R-stable (<50% variation)")
        
        print()

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print("Key finding: σ=0.5 (half-shift) works better than σ=1.0 (full shift)")
    print("Best candidates for further investigation:")
    print("  1. l2/i2_only/σ=0.5 - Best conditioning")
    print("  2. grid/i1_only/σ=0.5 - Most stable weights")
    print()


if __name__ == "__main__":
    main()
