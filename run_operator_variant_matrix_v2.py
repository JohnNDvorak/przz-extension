"""
run_operator_variant_matrix_v2.py
Full operator variant matrix with sigma sweep (Codex Task 5)

Tests all combinations of:
- Normalizations: none, l2, grid
- Scopes: i1_only, i2_only, both (controls)
- Sigmas: 0.0, 0.5, 1.0

Reports operator-solved (m1, m2) via 2×2 system for each variant and compares
against the dynamically-computed base-solve (no hard-coded fitted weights).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple
import sys

from src.evaluate import (
    compute_c_paper_operator_v2,
    solve_two_weight_operator,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_C_TARGET = 2.137
KAPPA_STAR_R = 1.1167
KAPPA_STAR_C_TARGET = 1.938


@dataclass
class VariantResult:
    """Result for one operator variant."""
    normalization: str
    scope: str
    sigma: float
    m1_op: float
    m2_op: float
    cond_op: float
    det_op: float
    m1_base: float
    m2_base: float
    cond_base: float
    # Derived metrics
    m1_delta: float  # m1_op - m1_base
    m2_delta: float
    verdict: str  # "GO", "WARN", "FAIL", "FAIL-HARD", "IDENTITY"


def evaluate_verdict(
    m1: float,
    m2: float,
    cond: float,
    *,
    base_m1: float,
    base_m2: float,
    is_identity: bool = False,
) -> str:
    """Determine verdict based on standardized criteria."""
    if is_identity:
        return "IDENTITY"
    
    # FAIL-HARD: condition number > 60
    if cond > 60:
        return "FAIL-HARD"
    
    # FAIL: negative weights
    if m1 <= 0 or m2 <= 0:
        return "FAIL"
    
    # WARN: condition number > 25
    if cond > 25:
        return "WARN"
    
    # Heuristic sanity band (within 3x of base solve)
    if not (base_m1 / 3 < m1 < base_m1 * 3 and base_m2 / 3 < m2 < base_m2 * 3):
        return "WARN"

    return "GO"


def run_variant(
    normalization: str,
    scope: str,
    sigma: float,
    polys_kappa: dict,
    polys_kappa_star: dict,
    n: int,
    n_quad_a: int,
    base_m1: float,
    base_m2: float,
) -> VariantResult:
    """Run a single operator variant and return results."""
    
    is_identity = abs(sigma) < 1e-15
    
    # Run for κ benchmark
    result_k = compute_c_paper_operator_v2(
        theta=THETA, R=KAPPA_R, n=n, polynomials=polys_kappa,
        n_quad_a=n_quad_a, verbose=False,
        normalization=normalization, lift_scope=scope, sigma=sigma,
    )
    
    # Run for κ* benchmark
    result_ks = compute_c_paper_operator_v2(
        theta=THETA, R=KAPPA_STAR_R, n=n, polynomials=polys_kappa_star,
        n_quad_a=n_quad_a, verbose=False,
        normalization=normalization, lift_scope=scope, sigma=sigma,
    )
    
    # Solve 2×2 system with operator channels
    op_solve = solve_two_weight_operator(
        result_k, result_ks,
        c_target_k=KAPPA_C_TARGET, c_target_k_star=KAPPA_STAR_C_TARGET,
        use_operator_channels=True,
    )
    
    m1_op = op_solve["m1"]
    m2_op = op_solve["m2"]
    cond_op = op_solve["cond"]
    det_op = op_solve["det"]

    # Compute deltas
    m1_delta = m1_op - base_m1
    m2_delta = m2_op - base_m2

    verdict = evaluate_verdict(
        m1_op,
        m2_op,
        cond_op,
        base_m1=base_m1,
        base_m2=base_m2,
        is_identity=is_identity,
    )

    return VariantResult(
        normalization=normalization,
        scope=scope,
        sigma=sigma,
        m1_op=m1_op,
        m2_op=m2_op,
        cond_op=cond_op,
        det_op=det_op,
        m1_base=base_m1,
        m2_base=base_m2,
        cond_base=float("nan"),
        m1_delta=m1_delta,
        m2_delta=m2_delta,
        verdict=verdict,
    )


def print_summary_row(r: VariantResult):
    """Print a single summary row."""
    print(f"{r.normalization:8s} {r.scope:12s} {r.sigma:4.1f}  "
          f"{r.m1_op:10.2f} {r.m2_op:10.2f} {r.cond_op:8.1f}  "
          f"{r.m1_delta:+10.2f} {r.m2_delta:+10.2f}  {r.verdict}")


def main():
    parser = argparse.ArgumentParser(
        description="Full operator variant matrix (Codex Task 5)"
    )
    parser.add_argument("--n", type=int, default=60, 
                        help="u/t quadrature points")
    parser.add_argument("--n-quad-a", type=int, default=40,
                        help="Case C a-integral quadrature points")
    parser.add_argument("--theta", type=str, default="4/7",
                        help="theta parameter (default: 4/7)")
    args = parser.parse_args()
    
    # Parse theta
    if "/" in args.theta:
        num, denom = args.theta.split("/")
        theta = float(num) / float(denom)
    else:
        theta = float(args.theta)
    
    global THETA
    THETA = theta
    
    print("=" * 110)
    print("OPERATOR VARIANT MATRIX V2 (Comprehensive)")
    print("=" * 110)
    print()
    print(f"Quadrature: n={args.n}, n_quad_a={args.n_quad_a}, theta={theta:.6f}")
    print()
    
    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}
    
    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}
    
    # Compute dynamic base-solve (sigma=0 identity; base channels only).
    base_k = compute_c_paper_operator_v2(
        theta=THETA, R=KAPPA_R, n=args.n, polynomials=polys_kappa,
        n_quad_a=args.n_quad_a, verbose=False,
        normalization="none", lift_scope="both", sigma=0.0,
    )
    base_ks = compute_c_paper_operator_v2(
        theta=THETA, R=KAPPA_STAR_R, n=args.n, polynomials=polys_kappa_star,
        n_quad_a=args.n_quad_a, verbose=False,
        normalization="none", lift_scope="both", sigma=0.0,
    )
    base_solve = solve_two_weight_operator(
        base_k, base_ks,
        c_target_k=KAPPA_C_TARGET, c_target_k_star=KAPPA_STAR_C_TARGET,
        use_operator_channels=False,
    )
    base_m1 = base_solve["m1"]
    base_m2 = base_solve["m2"]
    base_cond = base_solve["cond"]

    print(f"Base solve: m1={base_m1:.6f}, m2={base_m2:.6f}, cond={base_cond:.2f}")
    print()

    # Define variants to test
    normalizations = ["none", "l2", "grid"]
    scopes = ["i1_left_only", "i1_right_only", "i1_only", "i2_only", "both"]
    sigmas = [0.0, 0.5, 1.0]
    
    # Run all variants
    results: List[VariantResult] = []
    
    print("Running variants...")
    total = len(normalizations) * len(scopes) * len(sigmas)
    count = 0
    
    for norm in normalizations:
        for scope in scopes:
            for sigma in sigmas:
                count += 1
                print(f"  [{count}/{total}] {norm}/{scope}/σ={sigma:.1f}...", 
                      end="", flush=True)
                try:
                    result = run_variant(
                        normalization=norm,
                        scope=scope,
                        sigma=sigma,
                        polys_kappa=polys_kappa,
                        polys_kappa_star=polys_kappa_star,
                        n=args.n,
                        n_quad_a=args.n_quad_a,
                        base_m1=base_m1,
                        base_m2=base_m2,
                    )
                    results.append(result)
                    print(f" m1={result.m1_op:+.1f}, m2={result.m2_op:+.1f}, "
                          f"cond={result.cond_op:.0f} [{result.verdict}]")
                except Exception as e:
                    print(f" ERROR: {e}")
    
    print()
    
    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    print("=" * 110)
    print("RESULTS TABLE (Standardized Metrics - Codex Task 4)")
    print("=" * 110)
    print()
    print(f"{'Norm':<8} {'Scope':<12} {'σ':>4}  "
          f"{'m1_op':>10} {'m2_op':>10} {'cond_op':>8}  "
          f"{'Δm1':>10} {'Δm2':>10}  Verdict")
    print("-" * 110)
    
    for r in results:
        print_summary_row(r)
    
    print()
    
    # =========================================================================
    # CONTROL CHECKS (Claude Run 3)
    # =========================================================================
    print("=" * 110)
    print("CONTROL CHECKS")
    print("=" * 110)
    print()
    
    # Check 1: sigma=0 should reproduce base
    sigma0_results = [r for r in results if abs(r.sigma) < 1e-15]
    if sigma0_results:
        print("1. σ=0 Identity Check:")
        for r in sigma0_results:
            delta_m1 = abs(r.m1_op - r.m1_base)
            delta_m2 = abs(r.m2_op - r.m2_base)
            ok = delta_m1 < 1e-6 and delta_m2 < 1e-6
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"   {r.normalization}/{r.scope}: Δm1={delta_m1:.2e}, Δm2={delta_m2:.2e} {status}")
        print()
    
    # Check 2: "both" scope should misbehave
    both_results = [r for r in results if r.scope == "both" and r.sigma > 0.5]
    if both_results:
        print("2. 'both' Scope Negative Control:")
        all_bad = all(r.verdict in ["FAIL", "FAIL-HARD", "WARN"] for r in both_results)
        status = "✓ EXPECTED" if all_bad else "⚠ UNEXPECTED"
        print(f"   All 'both' scope variants have issues: {status}")
        for r in both_results:
            print(f"   {r.normalization}/both/σ={r.sigma:.1f}: "
                  f"m1={r.m1_op:.1f}, m2={r.m2_op:.1f}, cond={r.cond_op:.0f} [{r.verdict}]")
        print()
    
    # =========================================================================
    # GO VARIANTS
    # =========================================================================
    print("=" * 110)
    print("GO VARIANTS (Best Candidates)")
    print("=" * 110)
    print()
    
    go_variants = [r for r in results if r.verdict == "GO"]
    
    if go_variants:
        # Sort by conditioning
        go_variants.sort(key=lambda r: r.cond_op)
        
        print(f"Found {len(go_variants)} GO variants:")
        print()
        print(f"{'Rank':>4} {'Norm':<8} {'Scope':<12} {'σ':>4}  "
              f"{'m1_op':>10} {'m2_op':>10} {'cond_op':>8}")
        print("-" * 70)
        
        for i, r in enumerate(go_variants[:5], 1):
            print(f"{i:>4} {r.normalization:<8} {r.scope:<12} {r.sigma:>4.1f}  "
                  f"{r.m1_op:>10.2f} {r.m2_op:>10.2f} {r.cond_op:>8.1f}")
        
        print()
        print("→ Top variants recommended for R-sweep (Claude Run 2)")
    else:
        print("No GO variants found.")
        print()
        warn_variants = [r for r in results if r.verdict == "WARN"]
        if warn_variants:
            print(f"Found {len(warn_variants)} WARN variants (partial success):")
            for r in warn_variants[:3]:
                print(f"   {r.normalization}/{r.scope}/σ={r.sigma:.1f}: "
                      f"m1={r.m1_op:.1f}, m2={r.m2_op:.1f}, cond={r.cond_op:.0f}")
    
    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
