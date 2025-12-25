"""
run_two_benchmark_gate.py
Two-Benchmark Gate Test (Paper-Truth DSL)

Tests the DSL evaluator (`src/evaluate.py` + term tables) against both:
- κ benchmark (R=1.3036, c_target=2.137)
- κ* benchmark (R=1.1167, c_target=1.938)

The gate passes if:
1. Both c values are within 5% of targets
2. Ratio of c values is within 20% of target ratio (1.10)
"""

import json
import numpy as np
from math import exp, log

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import compute_c_paper, compute_kappa, evaluate_c_full


def run_two_benchmark_gate(n_quad: int = 60, n_quad_a: int = 40, verbose: bool = True):
    """Run the two-benchmark gate test."""
    print("=" * 70)
    print("TWO-BENCHMARK GATE TEST - DSL PAPER EVALUATOR")
    print("=" * 70)

    # === Benchmark 1: κ (R=1.3036) ===
    print("\n--- Benchmark 1: κ (R=1.3036) ---")
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    R_kappa = 1.3036
    theta = 4.0 / 7.0
    c_target_kappa = 2.13745440613217

    c_kappa = compute_c_paper(
        theta=theta,
        R=R_kappa,
        n=n_quad,
        polynomials={"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k},
        return_breakdown=verbose,
        n_quad_a=n_quad_a,
    ).total
    kappa_computed = compute_kappa(c_kappa, R_kappa)

    error_kappa = (c_kappa - c_target_kappa) / c_target_kappa * 100
    print(f"  c computed: {c_kappa:.6f}")
    print(f"  c target:   {c_target_kappa:.6f}")
    print(f"  Error:      {error_kappa:+.2f}%")
    print(f"  κ computed: {kappa_computed:.6f}")
    print(f"  κ target:   0.417294")

    # === Benchmark 2: κ* (R=1.1167) ===
    print("\n--- Benchmark 2: κ* (R=1.1167) ---")
    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    R_kappa_star = 1.1167
    c_target_kappa_star = 1.9379524124677437

    c_kappa_star = compute_c_paper(
        theta=theta,
        R=R_kappa_star,
        n=n_quad,
        polynomials={"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s},
        return_breakdown=verbose,
        n_quad_a=n_quad_a,
    ).total
    kappa_star_computed = compute_kappa(c_kappa_star, R_kappa_star)

    error_kappa_star = (c_kappa_star - c_target_kappa_star) / c_target_kappa_star * 100
    print(f"  c computed: {c_kappa_star:.6f}")
    print(f"  c target:   {c_target_kappa_star:.6f}")
    print(f"  Error:      {error_kappa_star:+.2f}%")
    print(f"  κ* computed: {kappa_star_computed:.6f}")
    print(f"  κ* target:   0.407511")

    # === Ratio Analysis ===
    print("\n--- Ratio Analysis ---")
    target_ratio = c_target_kappa / c_target_kappa_star
    computed_ratio = c_kappa / c_kappa_star
    ratio_error = (computed_ratio - target_ratio) / target_ratio * 100

    print(f"  Target ratio:   {target_ratio:.4f}")
    print(f"  Computed ratio: {computed_ratio:.4f}")
    print(f"  Ratio error:    {ratio_error:+.2f}%")

    # === Gate Results ===
    print("\n" + "=" * 70)
    print("GATE RESULTS")
    print("=" * 70)

    gate1_pass = abs(error_kappa) < 5.0
    gate2_pass = abs(error_kappa_star) < 5.0
    gate3_pass = abs(ratio_error) < 20.0

    print(f"  Gate 1 (κ within 5%):     {'PASS' if gate1_pass else 'FAIL'} ({error_kappa:+.2f}%)")
    print(f"  Gate 2 (κ* within 5%):    {'PASS' if gate2_pass else 'FAIL'} ({error_kappa_star:+.2f}%)")
    print(f"  Gate 3 (ratio within 20%): {'PASS' if gate3_pass else 'FAIL'} ({ratio_error:+.2f}%)")

    overall_pass = gate1_pass and gate2_pass and gate3_pass
    print(f"\n  OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    if verbose:
        # Show per-pair breakdown for both benchmarks (raw pair contributions)
        print("\n" + "=" * 70)
        print("PER-PAIR BREAKDOWN (DSL PAPER)")
        print("=" * 70)

        def _pair_raw(polys, R_val):
            res = evaluate_c_full(
                theta=theta,
                R=R_val,
                n=n_quad,
                polynomials=polys,
                return_breakdown=True,
                kernel_regime="paper",
                n_quad_a=n_quad_a,
            )
            return {
                "11": res.per_term.get("_c11_raw", float("nan")),
                "22": res.per_term.get("_c22_raw", float("nan")),
                "33": res.per_term.get("_c33_raw", float("nan")),
                "12": res.per_term.get("_c12_raw", float("nan")),
                "13": res.per_term.get("_c13_raw", float("nan")),
                "23": res.per_term.get("_c23_raw", float("nan")),
            }

        raw_k = _pair_raw({"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}, R_kappa)
        raw_s = _pair_raw({"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}, R_kappa_star)

        print("\n  pair     κ raw        κ* raw       ratio")
        print("-" * 55)
        for pair in ["11", "22", "33", "12", "13", "23"]:
            a = raw_k[pair]
            b = raw_s[pair]
            r = a / b if b != 0 else float("inf")
            print(f"  {pair}: {a:+.6f}   {b:+.6f}   {r:.4f}")

        print("-" * 55)
        print(f"  Total:  {c_kappa:+.6f}   {c_kappa_star:+.6f}   {computed_ratio:.4f}")

    return overall_pass, {
        'c_kappa': c_kappa,
        'c_kappa_star': c_kappa_star,
        'error_kappa': error_kappa,
        'error_kappa_star': error_kappa_star,
        'ratio_error': ratio_error
    }


if __name__ == "__main__":
    run_two_benchmark_gate()
