"""
run_operator_mode_diagnostic.py
Operator Mode Diagnostic: Q → Q_lift in mirror branch (GPT Guidance 2025-12-20)

This script runs the operator mode evaluation for both κ and κ* benchmarks
and reports the implied weights (m1_implied, m2_implied).

This script intentionally does NOT treat any fitted (m₁, m₂) as truth.
Instead it reports:
- implied (m₁, m₂) from operator structure (no targets)
- solved (m₁, m₂) from the two-benchmark system (uses targets, for comparison only)
- residual ratios: (m_solved / m_implied) per benchmark
"""

from __future__ import annotations

import argparse
from src.evaluate import compute_c_paper_operator_q_shift
from src.evaluate import solve_two_weight_coefficients
from src.evaluate import solve_two_weight_operator
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0

# Benchmarks
KAPPA_R = 1.3036
KAPPA_C_TARGET = 2.137
KAPPA_STAR_R = 1.1167
KAPPA_STAR_C_TARGET = 1.938


def _safe_ratio(num: float, den: float) -> float:
    if den == 0.0:
        return float("inf") if num != 0.0 else float("nan")
    return num / den


def main() -> None:
    parser = argparse.ArgumentParser(description="Operator-mode implied weight diagnostic (κ and κ*).")
    parser.add_argument("--n", type=int, default=40, help="u/t quadrature points")
    parser.add_argument("--n-quad-a", type=int, default=30, help="Case C a-integral quadrature points")
    parser.add_argument("--theta", type=float, default=THETA, help="theta parameter")
    args = parser.parse_args()

    print("=" * 80)
    print("OPERATOR MODE DIAGNOSTIC: Q → Q_lift in mirror branch")
    print("=" * 80)
    print()

    # Load polynomials
    print("Loading polynomials...")
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}

    n_quad = args.n
    n_quad_a = args.n_quad_a

    print(f"Using n={n_quad}, n_quad_a={n_quad_a}")
    print()

    # =========================================================================
    # RUN κ BENCHMARK
    # =========================================================================
    print("=" * 80)
    print("κ BENCHMARK (R=1.3036)")
    print("=" * 80)
    print()

    result_k = compute_c_paper_operator_q_shift(
        theta=args.theta,
        R=KAPPA_R,
        n=n_quad,
        polynomials=polys_kappa,
        n_quad_a=n_quad_a,
        verbose=True,
    )

    c_k = result_k.total
    c_gap_k = (c_k - KAPPA_C_TARGET) / KAPPA_C_TARGET * 100
    m1_k = result_k.per_term["_m1_implied"]
    m2_k = result_k.per_term["_m2_implied"]

    print()
    print(f"  c_computed:  {c_k:.6f}")
    print(f"  c_target:    {KAPPA_C_TARGET:.6f}")
    print(f"  c_gap:       {c_gap_k:+.2f}%")
    print()

    # =========================================================================
    # RUN κ* BENCHMARK
    # =========================================================================
    print("=" * 80)
    print("κ* BENCHMARK (R=1.1167)")
    print("=" * 80)
    print()

    result_s = compute_c_paper_operator_q_shift(
        theta=args.theta,
        R=KAPPA_STAR_R,
        n=n_quad,
        polynomials=polys_kappa_star,
        n_quad_a=n_quad_a,
        verbose=True,
    )

    c_s = result_s.total
    c_gap_s = (c_s - KAPPA_STAR_C_TARGET) / KAPPA_STAR_C_TARGET * 100
    m1_s = result_s.per_term["_m1_implied"]
    m2_s = result_s.per_term["_m2_implied"]

    print()
    print(f"  c_computed:  {c_s:.6f}")
    print(f"  c_target:    {KAPPA_STAR_C_TARGET:.6f}")
    print(f"  c_gap:       {c_gap_s:+.2f}%")
    print()

    # =========================================================================
    # TWO-WEIGHT SOLVE (for comparison only; uses benchmark targets)
    # =========================================================================
    channels_k = {
        "_I1_plus": float(result_k.per_term["_I1_plus"]),
        "_I2_plus": float(result_k.per_term["_I2_plus"]),
        "_I1_minus": float(result_k.per_term["_I1_minus_base"]),
        "_I2_minus": float(result_k.per_term["_I2_minus_base"]),
        "_S34_plus": float(result_k.per_term["_S34_plus"]),
    }
    channels_s = {
        "_I1_plus": float(result_s.per_term["_I1_plus"]),
        "_I2_plus": float(result_s.per_term["_I2_plus"]),
        "_I1_minus": float(result_s.per_term["_I1_minus_base"]),
        "_I2_minus": float(result_s.per_term["_I2_minus_base"]),
        "_S34_plus": float(result_s.per_term["_S34_plus"]),
    }
    m1_solved, m2_solved, det = solve_two_weight_coefficients(
        channels_k,
        channels_s,
        c_target_kappa=KAPPA_C_TARGET,
        c_target_kappa_star=KAPPA_STAR_C_TARGET,
    )

    # =========================================================================
    # TWO-WEIGHT SOLVE USING OPERATOR CHANNELS (Codex Run 2)
    # =========================================================================
    # This solves the same 2×2 system but with operator-mode minus channels
    # instead of base minus channels.
    op_solve = solve_two_weight_operator(
        result_k,
        result_s,
        c_target_k=KAPPA_C_TARGET,
        c_target_k_star=KAPPA_STAR_C_TARGET,
        use_operator_channels=True,
    )
    m1_op_solved = op_solve["m1"]
    m2_op_solved = op_solve["m2"]
    det_op = op_solve["det"]
    cond_op = op_solve["cond"]

    # Also solve with base channels for direct comparison
    base_solve = solve_two_weight_operator(
        result_k,
        result_s,
        c_target_k=KAPPA_C_TARGET,
        c_target_k_star=KAPPA_STAR_C_TARGET,
        use_operator_channels=False,
    )
    m1_base_solved = base_solve["m1"]
    m2_base_solved = base_solve["m2"]

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 80)
    print("SUMMARY: OPERATOR-SOLVED vs BASE-SOLVED WEIGHTS (GPT Codex Run 2)")
    print("=" * 80)
    print()

    # Primary comparison: Base-solved vs Operator-solved
    print("Two-Weight Solutions (2×2 system using both benchmarks):")
    print()
    print(f"{'Mode':<20} {'m1':>12} {'m2':>12} {'ratio':>10} {'det':>12} {'cond':>10}")
    print("-" * 80)
    print(f"{'BASE (I_minus_base)':<20} {m1_base_solved:>12.4f} {m2_base_solved:>12.4f} {_safe_ratio(m1_base_solved, m2_base_solved):>10.4f} {base_solve['det']:>12.4e} {base_solve['cond']:>10.2f}")
    print(f"{'OPERATOR (I_minus_op)':<20} {m1_op_solved:>12.4f} {m2_op_solved:>12.4f} {_safe_ratio(m1_op_solved, m2_op_solved):>10.4f} {det_op:>12.4e} {cond_op:>10.2f}")
    print()

    # Compute effect sizes: how much operator mode changes the required weights.
    # If |m_op_solved| << |m_base_solved|, operator mode is explaining a large factor.
    ratio_m1 = _safe_ratio(m1_base_solved, m1_op_solved)
    ratio_m2 = _safe_ratio(m2_base_solved, m2_op_solved)
    gain_m1 = abs(ratio_m1)
    gain_m2 = abs(ratio_m2)

    print("Weight reduction factors |m_base_solved| / |m_op_solved|:")
    print(f"  m1: {gain_m1:.4f}   (base/op = {ratio_m1:.4f})")
    print(f"  m2: {gain_m2:.4f}   (base/op = {ratio_m2:.4f})")
    print()

    # Secondary: implied weights per benchmark (for comparison)
    print("-" * 80)
    print("Implied Weights per Benchmark (naive ratio I_op/I_base):")
    print(f"{'Benchmark':<15} {'m1_implied':>12} {'m2_implied':>12} {'ratio':>10}")
    print("-" * 55)
    print(f"{'κ':<15} {m1_k:>12.4f} {m2_k:>12.4f} {_safe_ratio(m1_k, m2_k):>10.4f}")
    print(f"{'κ*':<15} {m1_s:>12.4f} {m2_s:>12.4f} {_safe_ratio(m1_s, m2_s):>10.4f}")
    print()

    # =========================================================================
    # VERDICT
    # =========================================================================
    print("=" * 80)
    # Verdict uses the *absolute* reduction factor. Values > 1 mean operator mode
    # reduces the required weights; values < 1 mean it makes them larger.
    reduction = min(gain_m1, gain_m2)
    if reduction > 3.0:
        print("RESULT: Operator mode reduces required weights by >3× (promising)")
    elif reduction > 1.2:
        print("RESULT: Operator mode reduces required weights modestly (>1.2×)")
    else:
        print("RESULT: Operator mode does NOT reduce required weights (≤1.2×)")
        print("        → Q-shift alone does not explain the missing amplification")
        print("        → Missing structure beyond Q → Q(1+x)")
    print("=" * 80)

    # =========================================================================
    # CHANNEL BREAKDOWN
    # =========================================================================
    print()
    print("=" * 80)
    print("CHANNEL VALUES")
    print("=" * 80)
    print()
    print("κ benchmark:")
    print(f"  I1_plus:       {result_k.per_term['_I1_plus']:+.8f}")
    print(f"  I1_minus_base: {result_k.per_term['_I1_minus_base']:+.8f}")
    print(f"  I1_minus_op:   {result_k.per_term['_I1_minus_op']:+.8f}")
    print(f"  I2_plus:       {result_k.per_term['_I2_plus']:+.8f}")
    print(f"  I2_minus_base: {result_k.per_term['_I2_minus_base']:+.8f}")
    print(f"  I2_minus_op:   {result_k.per_term['_I2_minus_op']:+.8f}")
    print(f"  S34_plus:      {result_k.per_term['_S34_plus']:+.8f}")
    print()
    print("κ* benchmark:")
    print(f"  I1_plus:       {result_s.per_term['_I1_plus']:+.8f}")
    print(f"  I1_minus_base: {result_s.per_term['_I1_minus_base']:+.8f}")
    print(f"  I1_minus_op:   {result_s.per_term['_I1_minus_op']:+.8f}")
    print(f"  I2_plus:       {result_s.per_term['_I2_plus']:+.8f}")
    print(f"  I2_minus_base: {result_s.per_term['_I2_minus_base']:+.8f}")
    print(f"  I2_minus_op:   {result_s.per_term['_I2_minus_op']:+.8f}")
    print(f"  S34_plus:      {result_s.per_term['_S34_plus']:+.8f}")


if __name__ == "__main__":
    main()
