"""
run_operator_variant_matrix.py
Claude Run 1: Compare operator variants (scopes × normalizations)

Tests all combinations of:
- Scopes: both, i1_only, i2_only
- Normalizations: none, l2, grid

Reports operator-solved (m1, m2) via 2×2 system for each variant.
GO/NO-GO criteria clearly evaluated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

from src.evaluate import (
    compute_c_paper_operator_unified,
    solve_two_weight_operator,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_C_TARGET = 2.137
KAPPA_STAR_R = 1.1167
KAPPA_STAR_C_TARGET = 1.938

# Target weights from base 2×2 solve
M1_BASE_TARGET = 6.198
M2_BASE_TARGET = 8.052


@dataclass
class VariantResult:
    """Result for one operator variant."""
    normalization: str
    scope: str
    m1_op: float
    m2_op: float
    cond_op: float
    det_op: float
    m1_delta: float  # m1_op - m1_base
    m2_delta: float
    is_positive: bool
    is_well_conditioned: bool
    is_reasonable: bool  # within ×2 of base


def run_variant(
    normalization: str,
    scope: str,
    polys_kappa: dict,
    polys_kappa_star: dict,
    n: int,
    n_quad_a: int,
) -> VariantResult:
    """Run a single operator variant and return results."""

    # Run for κ benchmark
    result_k = compute_c_paper_operator_unified(
        theta=THETA, R=KAPPA_R, n=n, polynomials=polys_kappa,
        n_quad_a=n_quad_a, verbose=False,
        normalization=normalization, lift_scope=scope,
    )

    # Run for κ* benchmark
    result_ks = compute_c_paper_operator_unified(
        theta=THETA, R=KAPPA_STAR_R, n=n, polynomials=polys_kappa_star,
        n_quad_a=n_quad_a, verbose=False,
        normalization=normalization, lift_scope=scope,
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

    # Compute deltas from base
    m1_delta = m1_op - M1_BASE_TARGET
    m2_delta = m2_op - M2_BASE_TARGET

    # Check criteria
    is_positive = m1_op > 0 and m2_op > 0
    is_well_conditioned = cond_op < 25
    is_reasonable = (
        is_positive and
        0.5 * M1_BASE_TARGET < m1_op < 2.0 * M1_BASE_TARGET and
        0.5 * M2_BASE_TARGET < m2_op < 2.0 * M2_BASE_TARGET
    )

    return VariantResult(
        normalization=normalization,
        scope=scope,
        m1_op=m1_op,
        m2_op=m2_op,
        cond_op=cond_op,
        det_op=det_op,
        m1_delta=m1_delta,
        m2_delta=m2_delta,
        is_positive=is_positive,
        is_well_conditioned=is_well_conditioned,
        is_reasonable=is_reasonable,
    )


def main():
    parser = argparse.ArgumentParser(description="Operator variant matrix test (Claude Run 1)")
    parser.add_argument("--n", type=int, default=60, help="u/t quadrature points")
    parser.add_argument("--n-quad-a", type=int, default=40, help="Case C a-integral quadrature points")
    args = parser.parse_args()

    print("=" * 100)
    print("OPERATOR VARIANT MATRIX (Claude Run 1)")
    print("=" * 100)
    print()
    print(f"Base weights: m1={M1_BASE_TARGET:.3f}, m2={M2_BASE_TARGET:.3f}")
    print(f"Quadrature: n={args.n}, n_quad_a={args.n_quad_a}")
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}

    # Define variants to test
    normalizations = ["none", "l2", "grid"]
    scopes = ["both", "i1_only", "i2_only"]

    # Run all variants
    results: List[VariantResult] = []

    print("Running variants...")
    for norm in normalizations:
        for scope in scopes:
            result = run_variant(
                normalization=norm,
                scope=scope,
                polys_kappa=polys_kappa,
                polys_kappa_star=polys_kappa_star,
                n=args.n,
                n_quad_a=args.n_quad_a,
            )
            results.append(result)
            print(f"  {norm:8s} / {scope:10s}: m1={result.m1_op:8.2f}, m2={result.m2_op:8.2f}, cond={result.cond_op:6.1f}")

    print()

    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    print("=" * 100)
    print("RESULTS TABLE")
    print("=" * 100)
    print()
    print(f"{'Norm':<10} {'Scope':<12} {'m1_op':>10} {'m2_op':>10} {'cond':>8} {'m1_Δ':>10} {'m2_Δ':>10} {'pos?':>6} {'cond<25':>8} {'OK?':>5}")
    print("-" * 100)

    for r in results:
        pos_flag = "✓" if r.is_positive else "✗"
        cond_flag = "✓" if r.is_well_conditioned else "✗"
        ok_flag = "GO" if r.is_reasonable and r.is_well_conditioned else "NO"

        print(f"{r.normalization:<10} {r.scope:<12} {r.m1_op:>10.2f} {r.m2_op:>10.2f} {r.cond_op:>8.1f} {r.m1_delta:>+10.2f} {r.m2_delta:>+10.2f} {pos_flag:>6} {cond_flag:>8} {ok_flag:>5}")

    print()

    # =========================================================================
    # GO/NO-GO VERDICT
    # =========================================================================
    print("=" * 100)
    print("GO/NO-GO VERDICT")
    print("=" * 100)
    print()

    go_variants = [r for r in results if r.is_reasonable and r.is_well_conditioned]
    partial_variants = [r for r in results if r.is_positive and r.is_well_conditioned and not r.is_reasonable]

    if go_variants:
        print("🟢 GO: Found variant(s) with positive weights, cond<25, and weights within ×2 of base:")
        for r in go_variants:
            print(f"   - {r.normalization}/{r.scope}: m1={r.m1_op:.2f}, m2={r.m2_op:.2f}, cond={r.cond_op:.1f}")
        print()
        print("→ Proceed to Claude Run 2 (R-sweep) with best GO variant")
    elif partial_variants:
        print("🟡 PARTIAL: Found variant(s) with positive weights and cond<25, but weights outside ×2 range:")
        for r in partial_variants:
            print(f"   - {r.normalization}/{r.scope}: m1={r.m1_op:.2f}, m2={r.m2_op:.2f}, cond={r.cond_op:.1f}")
        print()
        print("→ Q-lift captures some structure but not the full mechanism")
    else:
        print("🔴 NO-GO: All variants have either:")
        print("   - Negative weights, or")
        print("   - cond > 25, or")
        print("   - Weights wildly different from base")
        print()
        print("→ Stop iterating on Q-lift as the operator mechanism")
        print("→ Pivot to next TeX-derived missing piece (not Q-shift)")

    print()

    # =========================================================================
    # BEST VARIANT DETAILS
    # =========================================================================
    # Find the variant with best conditioning among positive-weight variants
    positive_results = [r for r in results if r.is_positive]
    if positive_results:
        best = min(positive_results, key=lambda r: r.cond_op)
        print("=" * 100)
        print(f"BEST VARIANT (lowest cond among positive): {best.normalization}/{best.scope}")
        print("=" * 100)
        print(f"  m1_op = {best.m1_op:.4f} (base: {M1_BASE_TARGET:.4f}, Δ = {best.m1_delta:+.4f})")
        print(f"  m2_op = {best.m2_op:.4f} (base: {M2_BASE_TARGET:.4f}, Δ = {best.m2_delta:+.4f})")
        print(f"  cond  = {best.cond_op:.2f}")
        print(f"  det   = {best.det_op:.4e}")


if __name__ == "__main__":
    main()
