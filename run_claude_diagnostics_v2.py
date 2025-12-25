"""
run_claude_diagnostics_v2.py
GPT Run 2 Claude Diagnostics

Claude Task 1: Operator implied vs solved report at fixed settings
Claude Task 2: R-sweep implied weights (not solved)
Claude Task 3: Moment-based anti-overfit probe
"""

from __future__ import annotations

import numpy as np

from src.evaluate import (
    compare_operator_to_two_weight_solve,
    compute_operator_implied_weights,
    run_moment_anti_overfit_probe,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167
SIGMA = 5/32


def main():
    print("=" * 90)
    print("GPT RUN 2: CLAUDE DIAGNOSTICS")
    print("=" * 90)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}

    print(f"σ = {SIGMA:.6f} = 5/32")
    print(f"θ = {THETA:.6f} = 4/7")
    print(f"normalization = grid")
    print(f"lift_scope = i1_only")
    print()

    # =========================================================================
    # Claude Task 1: Operator implied vs solved report
    # =========================================================================
    print("=" * 90)
    print("CLAUDE TASK 1: OPERATOR IMPLIED vs SOLVED (σ=5/32, grid, i1_only)")
    print("=" * 90)
    print()

    comp_k, comp_ks, summary = compare_operator_to_two_weight_solve(
        polys_kappa=polys_kappa,
        polys_kappa_star=polys_kappa_star,
        sigma=SIGMA,
        normalization="grid",
        lift_scope="i1_only",
        n=60, n_quad_a=40,
    )

    print(f"{'Benchmark':<12} {'m1_implied':>12} {'m2_implied':>12} {'m1_solved':>12} {'m2_solved':>12} {'A1_resid':>10} {'A2_resid':>10}")
    print("-" * 85)
    print(f"{'κ':<12} {comp_k.m1_implied:>12.4f} {comp_k.m2_implied:>12.4f} {comp_k.m1_solved:>12.4f} {comp_k.m2_solved:>12.4f} {comp_k.A1_residual:>10.4f} {comp_k.A2_residual:>10.4f}")
    print(f"{'κ*':<12} {comp_ks.m1_implied:>12.4f} {comp_ks.m2_implied:>12.4f} {comp_ks.m1_solved:>12.4f} {comp_ks.m2_solved:>12.4f} {comp_ks.A1_residual:>10.4f} {comp_ks.A2_residual:>10.4f}")
    print()

    print("SUMMARY:")
    print(f"  A1_avg = {summary['A1_avg']:.4f}")
    print(f"  A2_avg = {summary['A2_avg']:.4f}")
    print(f"  A1_span = {summary['A1_span']:.2%}")
    print(f"  A2_span = {summary['A2_span']:.2%}")
    print(f"  is_global = {summary['is_global']}")
    print(f"  A_ratio κ = {summary['A_ratio_kappa']:.4f}")
    print(f"  A_ratio κ* = {summary['A_ratio_kappa_star']:.4f}")
    print()

    if summary['is_global']:
        print("✓ Residual is GLOBAL (A1, A2 stable across κ/κ*)")
    else:
        print("✗ Residual is BENCHMARK-DEPENDENT")
    print()

    # =========================================================================
    # Claude Task 2: R-sweep implied weights
    # =========================================================================
    print("=" * 90)
    print("CLAUDE TASK 2: R-SWEEP IMPLIED WEIGHTS (not solved)")
    print("=" * 90)
    print()

    R_values = [1.0, 1.15, 1.3036, 1.4, 1.5]

    print(f"{'R':>8} {'m1_impl(κ)':>12} {'m2_impl(κ)':>12} {'m1_impl(κ*)':>12} {'m2_impl(κ*)':>12} {'ratio_m1':>10}")
    print("-" * 70)

    for R_val in R_values:
        # Compute implied weights for κ polynomials at this R
        implied_k = compute_operator_implied_weights(
            theta=THETA, R=R_val, polynomials=polys_kappa,
            sigma=SIGMA, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
        )

        # Compute implied weights for κ* polynomials at this R
        implied_ks = compute_operator_implied_weights(
            theta=THETA, R=R_val, polynomials=polys_kappa_star,
            sigma=SIGMA, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
        )

        ratio = implied_k.m1_implied / implied_ks.m1_implied if implied_ks.m1_implied != 0 else float('inf')

        print(f"{R_val:>8.4f} {implied_k.m1_implied:>12.4f} {implied_k.m2_implied:>12.4f} "
              f"{implied_ks.m1_implied:>12.4f} {implied_ks.m2_implied:>12.4f} {ratio:>10.4f}")

    print()

    # Check smoothness across R
    print("SUCCESS CRITERION: implied weights should vary smoothly with R")
    print("  and should NOT diverge between κ and κ*")
    print()

    # =========================================================================
    # Claude Task 3: Moment-based anti-overfit probe
    # =========================================================================
    print("=" * 90)
    print("CLAUDE TASK 3: MOMENT-BASED ANTI-OVERFIT PROBE")
    print("=" * 90)
    print()

    probe = run_moment_anti_overfit_probe(
        polys_kappa=polys_kappa,
        polys_kappa_star=polys_kappa_star,
        sigma_empirical=SIGMA,
        n_quad=200,
    )

    print(f"σ_empirical = {probe['sigma_empirical']:.6f} = 5/32")
    print()

    print("KAPPA (R=1.3036) MOMENTS:")
    m_k = probe["kappa_moments"]
    print(f"  E[t] = {m_k.E_t:.6f}")
    print(f"  E[t²] = {m_k.E_t2:.6f}")
    print(f"  E[t(1-t)] = {m_k.E_t1mt:.6f}")
    print(f"  Var(t) = {m_k.Var_t:.6f}")
    print()

    print("KAPPA SIGMA CANDIDATE COMPARISON:")
    for name, data in probe["kappa_comparison"].items():
        diff_sign = "+" if data["diff"] >= 0 else ""
        print(f"  {name:<16} = {data['value']:>10.6f}  (σ - candidate = {diff_sign}{data['diff']:.6f}, rel = {data['rel_diff_pct']:.1f}%)")
    print()

    print(f"Best match for κ: {probe['best_match_kappa'][0]} (diff = {probe['best_match_kappa'][1]:.6f})")
    print(f"Best match for κ*: {probe['best_match_kappa_star'][0]} (diff = {probe['best_match_kappa_star'][1]:.6f})")
    print()

    print("=" * 90)
    print(f"VERDICT: σ = 5/32 is {probe['verdict'].upper()}")
    print("=" * 90)
    print()

    if probe["verdict"] == "structural":
        print("σ ≈ 5/32 matches a moment-derived candidate consistently across κ and κ*.")
        print("This suggests a DERIVABLE structural relationship.")
    else:
        print("σ ≈ 5/32 does NOT match a single moment across both benchmarks.")
        print("This suggests σ is encoding Q-specific structure (still valid, just not universal).")
    print()

    print("=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
