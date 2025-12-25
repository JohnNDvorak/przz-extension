"""
run_sigma_fine_sweep.py
Fine σ sweep to find optimal shift magnitude

Sweeps σ ∈ [0.3, 0.7] with step 0.05 for the best variants.
"""

from __future__ import annotations

import argparse
from typing import List
import numpy as np

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

M1_BASE = 6.198
M2_BASE = 8.052


def evaluate_sigma(
    sigma: float,
    normalization: str,
    scope: str,
    polys_kappa: dict,
    polys_kappa_star: dict,
    n: int,
    n_quad_a: int,
) -> dict:
    """Evaluate a single sigma value."""
    
    result_k = compute_c_paper_operator_v2(
        theta=THETA, R=KAPPA_R, n=n, polynomials=polys_kappa,
        n_quad_a=n_quad_a, verbose=False,
        normalization=normalization, lift_scope=scope, sigma=sigma,
    )
    
    result_ks = compute_c_paper_operator_v2(
        theta=THETA, R=KAPPA_STAR_R, n=n, polynomials=polys_kappa_star,
        n_quad_a=n_quad_a, verbose=False,
        normalization=normalization, lift_scope=scope, sigma=sigma,
    )
    
    op_solve = solve_two_weight_operator(
        result_k, result_ks,
        c_target_k=KAPPA_C_TARGET, c_target_k_star=KAPPA_STAR_C_TARGET,
        use_operator_channels=True,
    )
    
    m1 = op_solve["m1"]
    m2 = op_solve["m2"]
    cond = op_solve["cond"]
    
    # Distance from base weights
    dist = np.sqrt((m1 - M1_BASE)**2 + (m2 - M2_BASE)**2)
    
    is_go = (
        m1 > 0 and m2 > 0 and cond < 25 and
        M1_BASE / 3 < m1 < M1_BASE * 3 and
        M2_BASE / 3 < m2 < M2_BASE * 3
    )
    
    return {
        "sigma": sigma,
        "m1": m1,
        "m2": m2,
        "cond": cond,
        "dist": dist,
        "is_go": is_go,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine σ sweep")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--n-quad-a", type=int, default=40)
    args = parser.parse_args()
    
    print("=" * 90)
    print("FINE σ SWEEP TO FIND OPTIMAL SHIFT MAGNITUDE")
    print("=" * 90)
    print()
    
    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}
    
    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}
    
    print(f"Base reference: m1={M1_BASE:.3f}, m2={M2_BASE:.3f}")
    print(f"Quadrature: n={args.n}, n_quad_a={args.n_quad_a}")
    print()
    
    # Sigma values to test
    sigmas = np.arange(0.25, 0.76, 0.05)
    
    # Test best variants
    variants = [
        ("l2", "i2_only"),
        ("grid", "i1_only"),
    ]
    
    for norm, scope in variants:
        print("=" * 90)
        print(f"VARIANT: {norm}/{scope}")
        print("=" * 90)
        print()
        print(f"{'σ':>6} {'m1':>10} {'m2':>10} {'cond':>8} {'dist':>8} {'OK':>5}")
        print("-" * 50)
        
        results = []
        for sigma in sigmas:
            r = evaluate_sigma(
                sigma=sigma,
                normalization=norm,
                scope=scope,
                polys_kappa=polys_kappa,
                polys_kappa_star=polys_kappa_star,
                n=args.n,
                n_quad_a=args.n_quad_a,
            )
            results.append(r)
            
            ok_flag = "GO" if r["is_go"] else "NO"
            print(f"{r['sigma']:>6.2f} {r['m1']:>10.2f} {r['m2']:>10.2f} "
                  f"{r['cond']:>8.1f} {r['dist']:>8.2f} {ok_flag:>5}")
        
        print()
        
        # Find optimal sigma (minimum distance to base among GO variants)
        go_results = [r for r in results if r["is_go"]]
        if go_results:
            best = min(go_results, key=lambda r: r["dist"])
            print(f"OPTIMAL σ = {best['sigma']:.2f}")
            print(f"  m1 = {best['m1']:.3f} (base: {M1_BASE:.3f}, Δ = {best['m1']-M1_BASE:+.3f})")
            print(f"  m2 = {best['m2']:.3f} (base: {M2_BASE:.3f}, Δ = {best['m2']-M2_BASE:+.3f})")
            print(f"  cond = {best['cond']:.2f}")
            print(f"  distance = {best['dist']:.3f}")
        else:
            print("No GO variants found in this range")
        
        print()
    
    # Special theoretical values to test
    print("=" * 90)
    print("THEORETICAL σ VALUES")
    print("=" * 90)
    print()
    
    # θ = 4/7, so 1/θ = 7/4 = 1.75, θ/2 = 2/7 ≈ 0.286
    theoretical_sigmas = [
        (1.0/THETA, "1/θ"),
        (THETA, "θ"),
        (THETA/2, "θ/2"),
        (1.0/2, "1/2"),
        (2.0/THETA, "2/θ"),
    ]
    
    for sigma, name in theoretical_sigmas:
        if 0 < sigma < 2:  # Reasonable range
            r = evaluate_sigma(
                sigma=sigma,
                normalization="grid",
                scope="i1_only",
                polys_kappa=polys_kappa,
                polys_kappa_star=polys_kappa_star,
                n=args.n,
                n_quad_a=args.n_quad_a,
            )
            ok = "GO" if r["is_go"] else "NO"
            print(f"σ = {name:8s} = {sigma:.4f}: m1={r['m1']:8.2f}, m2={r['m2']:8.2f}, "
                  f"cond={r['cond']:6.1f}, dist={r['dist']:6.2f} [{ok}]")
    
    print()


if __name__ == "__main__":
    main()
