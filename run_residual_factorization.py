"""
run_residual_factorization.py
Run residual factorization report at σ=5/32 (optimal shift)
"""

from __future__ import annotations

import argparse
import numpy as np

from src.evaluate import (
    report_residual_amplitude,
    fit_amplitude_candidates,
    rank_amplitude_fits,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167
KAPPA_C_TARGET = 2.137
KAPPA_STAR_C_TARGET = 1.938


def main():
    parser = argparse.ArgumentParser(description="Residual Factorization Report")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--n-quad-a", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=5/32)
    args = parser.parse_args()

    sigma = args.sigma

    print("=" * 80)
    print("RESIDUAL FACTORIZATION REPORT")
    print("=" * 80)
    print()
    print(f"σ = {sigma:.6f} ({sigma} = 5/32)" if sigma == 5/32 else f"σ = {sigma:.6f}")
    print(f"θ = {THETA:.6f} = 4/7")
    print(f"σ/θ = {sigma/THETA:.6f} = {sigma*7/4:.6f} × (4/7)")
    print(f"Quadrature: n={args.n}, n_quad_a={args.n_quad_a}")
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}

    # Run residual factorization report
    print("=" * 80)
    print("RESIDUAL FACTORIZATION (Claude Task A)")
    print("=" * 80)
    print()

    report_k, report_ks = report_residual_amplitude(
        polys_kappa=polys_kappa,
        polys_kappa_star=polys_kappa_star,
        theta=THETA,
        R_kappa=KAPPA_R,
        R_kappa_star=KAPPA_STAR_R,
        c_target_kappa=KAPPA_C_TARGET,
        c_target_kappa_star=KAPPA_STAR_C_TARGET,
        sigma=sigma,
        n=args.n,
        n_quad_a=args.n_quad_a,
    )

    print(f"{'Benchmark':<12} {'m1_solved':>12} {'m2_solved':>12} {'m1_shape':>10} {'m2_shape':>10} {'A1_resid':>10} {'A2_resid':>10} {'A_ratio':>10}")
    print("-" * 90)
    print(f"{'κ':<12} {report_k.m1_solved:>12.4f} {report_k.m2_solved:>12.4f} {report_k.m1_shape:>10.4f} {report_k.m2_shape:>10.4f} {report_k.A1_resid:>10.4f} {report_k.A2_resid:>10.4f} {report_k.A_ratio:>10.4f}")
    print(f"{'κ*':<12} {report_ks.m1_solved:>12.4f} {report_ks.m2_solved:>12.4f} {report_ks.m1_shape:>10.4f} {report_ks.m2_shape:>10.4f} {report_ks.A1_resid:>10.4f} {report_ks.A2_resid:>10.4f} {report_ks.A_ratio:>10.4f}")
    print()

    # Check if A_resid is global (A1_resid ≈ A2_resid)
    a1_avg = (report_k.A1_resid + report_ks.A1_resid) / 2
    a2_avg = (report_k.A2_resid + report_ks.A2_resid) / 2
    ratio_span_1 = abs(report_k.A1_resid - report_ks.A1_resid) / a1_avg if a1_avg != 0 else float('inf')
    ratio_span_2 = abs(report_k.A2_resid - report_ks.A2_resid) / a2_avg if a2_avg != 0 else float('inf')

    print("=" * 80)
    print("RESIDUAL STRUCTURE ANALYSIS")
    print("=" * 80)
    print()
    print(f"A1_resid average: {a1_avg:.4f}")
    print(f"A2_resid average: {a2_avg:.4f}")
    print(f"A1 span (relative): {ratio_span_1:.2%}")
    print(f"A2 span (relative): {ratio_span_2:.2%}")
    print()

    is_global = ratio_span_1 < 0.20 and ratio_span_2 < 0.20
    if is_global:
        print("✓ Residual appears GLOBAL (A1, A2 stable across benchmarks)")
        print(f"  Global A1 ≈ {a1_avg:.4f}")
        print(f"  Global A2 ≈ {a2_avg:.4f}")
    else:
        print("✗ Residual is CHANNEL-STRUCTURED (varies by benchmark)")
        print(f"  A1: κ={report_k.A1_resid:.4f}, κ*={report_ks.A1_resid:.4f}")
        print(f"  A2: κ={report_k.A2_resid:.4f}, κ*={report_ks.A2_resid:.4f}")
    print()

    # R-sweep for amplitude fingerprint (Claude Task B preview)
    # Use both benchmarks at each R-pair to maintain solvability
    print("=" * 80)
    print("R-SWEEP RESIDUAL FINGERPRINT (Claude Task B)")
    print("=" * 80)
    print()

    from src.evaluate import (
        compute_operator_factorization,
        solve_two_weight_operator,
        compute_c_paper_operator_v2,
    )

    # Sweep R for κ benchmark, keeping κ* fixed
    R_values = [1.0, 1.15, 1.3036, 1.4, 1.5]
    A1_resid_values = []
    A2_resid_values = []

    print(f"{'R_κ':>8} {'m1_shape':>10} {'m2_shape':>10} {'m1_solved':>10} {'m2_solved':>10} {'A1_resid':>10} {'A2_resid':>10}")
    print("-" * 75)

    for R_val in R_values:
        # Compute with varying R for κ
        result_k_R = compute_c_paper_operator_v2(
            theta=THETA, R=R_val, n=args.n, polynomials=polys_kappa,
            n_quad_a=args.n_quad_a, verbose=False,
            normalization="grid", lift_scope="i1_only", sigma=sigma,
        )

        # Keep κ* at fixed R for stability
        result_ks_fixed = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_STAR_R, n=args.n, polynomials=polys_kappa_star,
            n_quad_a=args.n_quad_a, verbose=False,
            normalization="grid", lift_scope="i1_only", sigma=sigma,
        )

        # Get shape factors at this R
        fact = compute_operator_factorization(
            polynomials=polys_kappa, theta=THETA, R=R_val, sigma=sigma,
            n=args.n, n_quad_a=args.n_quad_a,
        )

        # c_target scales with R: c = exp(R*(1-κ))
        c_target_k = np.exp(R_val * (1 - 0.417293962))
        c_target_ks = KAPPA_STAR_C_TARGET  # Fixed

        # Solve 2x2 system
        op_solve = solve_two_weight_operator(
            result_k_R, result_ks_fixed,
            c_target_k=c_target_k, c_target_k_star=c_target_ks,
            use_operator_channels=True,
        )

        m1_solved = op_solve["m1"]
        m2_solved = op_solve["m2"]

        A1_resid = m1_solved / fact.m1_shape if fact.m1_shape != 0 else float('inf')
        A2_resid = m2_solved / fact.m2_shape if fact.m2_shape != 0 else float('inf')

        A1_resid_values.append(A1_resid)
        A2_resid_values.append(A2_resid)

        print(f"{R_val:>8.4f} {fact.m1_shape:>10.4f} {fact.m2_shape:>10.4f} {m1_solved:>10.4f} {m2_solved:>10.4f} {A1_resid:>10.4f} {A2_resid:>10.4f}")

    print()

    # Fit amplitude candidates
    print("=" * 80)
    print("AMPLITUDE CANDIDATE FITTING")
    print("=" * 80)
    print()

    rankings = rank_amplitude_fits(R_values, A1_resid_values, A2_resid_values, THETA)

    print(f"{'Rank':>4} {'Candidate':>30} {'A1 RMSE':>12} {'A2 RMSE':>12} {'Total RMSE':>12}")
    print("-" * 75)

    for i, (name, fit_data) in enumerate(rankings[:6], 1):
        a1_rmse = fit_data.get('rmse_A1', float('inf'))
        a2_rmse = fit_data.get('rmse_A2', float('inf'))
        total_rmse = fit_data.get('rmse_total', float('inf'))
        print(f"{i:>4} {name:>30} {a1_rmse:>12.4f} {a2_rmse:>12.4f} {total_rmse:>12.4f}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
