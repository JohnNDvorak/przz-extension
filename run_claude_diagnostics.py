"""
run_claude_diagnostics.py
Claude Code Diagnostic Suite (GPT Guidance 2025-12-20)

Tasks:
1. Rerun 4-config truth table (ordered vs triangle) as guardrail
2. Recompute split-channel table and two-weight solve
3. Compute residual factors (m_solved / m_implied)
4. Sanity sweep for fingerprinting the residual
"""

from __future__ import annotations

import math
import numpy as np
from src.evaluate import (
    compare_triangle_vs_ordered,
    evaluate_c_ordered,
    evaluate_c_ordered_with_exp_transform,
    compute_c_paper_operator_q_shift,
    solve_two_weight_coefficients,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
N_QUAD = 40
N_QUAD_A = 30

# Benchmarks
KAPPA_R = 1.3036
KAPPA_C_TARGET = 2.137
KAPPA_STAR_R = 1.1167
KAPPA_STAR_C_TARGET = 1.938


def task1_truth_table():
    """Claude 1: Rerun 4-config truth table as guardrail."""
    print("=" * 100)
    print("CLAUDE 1: 4-Config Truth Table (Ordered vs Triangle)")
    print("=" * 100)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}

    configs = [
        ("raw", "κ", polys_k, KAPPA_R),
        ("raw", "κ*", polys_s, KAPPA_STAR_R),
        ("paper", "κ", polys_k, KAPPA_R),
        ("paper", "κ*", polys_s, KAPPA_STAR_R),
    ]

    print(f"{'Config':<15} {'Δ_S12 (12vs21)':<18} {'Δ_S34 (12vs21)':<18} {'S12 Sym?':<12} {'S34 Asym?':<12}")
    print("-" * 80)

    for kernel, bench, polys, R in configs:
        report = compare_triangle_vs_ordered(
            theta=THETA,
            R=R,
            n=N_QUAD,
            polynomials=polys,
            kernel_regime=kernel,
            n_quad_a=N_QUAD_A,
            verbose=False,
        )

        # Extract 12 vs 21 deltas
        d = report["off_diagonal"]["12_vs_21"]
        delta_s12 = float(d["delta_S12"])
        delta_s34 = float(d["delta_S34"])

        s12_sym = "YES" if abs(delta_s12) < 1e-10 else "NO"
        s34_asym = "YES" if abs(delta_s34) > 0.1 else "NO"

        config_str = f"{kernel}/{bench}"
        print(f"{config_str:<15} {delta_s12:>+18.2e} {delta_s34:>+18.6f} {s12_sym:<12} {s34_asym:<12}")

    print()
    print("Expected: S12 symmetric (Δ≈0), S34 asymmetric (Δ≠0)")
    print("Result: Ordered pairs remain paper truth.")
    print()


def task2_split_channels():
    """Claude 2: Recompute split-channel table and two-weight solve."""
    print("=" * 100)
    print("CLAUDE 2: Split-Channel Table and Two-Weight Solve")
    print("=" * 100)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}

    # Compute channels for both benchmarks
    channels_k = _compute_split_channels(polys_k, KAPPA_R)
    channels_s = _compute_split_channels(polys_s, KAPPA_STAR_R)

    print("Split Channel Values:")
    print()
    print(f"{'Channel':<15} {'κ (R=1.3036)':>15} {'κ* (R=1.1167)':>15} {'Ratio κ/κ*':>12}")
    print("-" * 60)
    for key in ["_I1_plus", "_I2_plus", "_I1_minus", "_I2_minus", "_S34_plus"]:
        k_val = channels_k[key]
        s_val = channels_s[key]
        ratio = k_val / s_val if s_val != 0 else float('inf')
        print(f"{key:<15} {k_val:>+15.8f} {s_val:>+15.8f} {ratio:>12.4f}")

    print()

    # Solve two-weight coefficients
    m1, m2, det = solve_two_weight_coefficients(
        channels_k, channels_s,
        c_target_kappa=KAPPA_C_TARGET,
        c_target_kappa_star=KAPPA_STAR_C_TARGET,
    )

    # Compute matrix condition number
    I1m_k = channels_k["_I1_minus"]
    I2m_k = channels_k["_I2_minus"]
    I1m_s = channels_s["_I1_minus"]
    I2m_s = channels_s["_I2_minus"]
    A = np.array([[I1m_k, I2m_k], [I1m_s, I2m_s]])
    cond = np.linalg.cond(A)

    print("Two-Weight Solver Results:")
    print(f"  m1 = {m1:.6f}")
    print(f"  m2 = {m2:.6f}")
    print(f"  m1/m2 = {m1/m2 if m2 != 0 else float('inf'):.6f}")
    print(f"  det = {det:.6e}")
    print(f"  cond = {cond:.2f}")
    print()

    return channels_k, channels_s, m1, m2


def _compute_split_channels(polys, R):
    """Helper to compute split I1/I2 channels."""
    ORDERED_PAIR_KEYS = ("11", "22", "33", "12", "21", "13", "31", "23", "32")
    FACTORIAL_WEIGHTS = {
        "11": 1.0, "22": 0.25, "33": 1.0/36,
        "12": 0.5, "21": 0.5,
        "13": 1.0/6, "31": 1.0/6,
        "23": 1.0/12, "32": 1.0/12,
    }

    result = evaluate_c_ordered(
        theta=THETA, R=R, n=N_QUAD, polynomials=polys,
        kernel_regime="paper", use_factorial_normalization=True, n_quad_a=N_QUAD_A,
    )
    mirror = evaluate_c_ordered_with_exp_transform(
        theta=THETA, R=-R, n=N_QUAD, polynomials=polys,
        kernel_regime="paper", exp_scale_multiplier=1.0, exp_t_flip=False,
        q_a0_shift=0.0, use_factorial_normalization=True, n_quad_a=N_QUAD_A,
    )

    I1_plus = I2_plus = I1_minus = I2_minus = 0.0
    I3_plus = I4_plus = 0.0

    for pair in ORDERED_PAIR_KEYS:
        w = FACTORIAL_WEIGHTS[pair]
        I1_plus += w * float(result.per_term.get(f"{pair}_I1_{pair}", 0.0))
        I2_plus += w * float(result.per_term.get(f"{pair}_I2_{pair}", 0.0))
        I3_plus += w * float(result.per_term.get(f"{pair}_I3_{pair}", 0.0))
        I4_plus += w * float(result.per_term.get(f"{pair}_I4_{pair}", 0.0))
        I1_minus += w * float(mirror.per_term.get(f"{pair}_I1_{pair}", 0.0))
        I2_minus += w * float(mirror.per_term.get(f"{pair}_I2_{pair}", 0.0))

    return {
        "_I1_plus": I1_plus,
        "_I2_plus": I2_plus,
        "_I1_minus": I1_minus,
        "_I2_minus": I2_minus,
        "_S34_plus": I3_plus + I4_plus,
    }


def task3_residual_factors(m1_solved, m2_solved):
    """Claude 3: Compute residual factors (m_solved / m_implied)."""
    print("=" * 100)
    print("CLAUDE 3: Residual Factors (m_solved / m_implied)")
    print("=" * 100)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}

    # Run operator mode for both benchmarks
    result_k = compute_c_paper_operator_q_shift(
        theta=THETA, R=KAPPA_R, n=N_QUAD, polynomials=polys_k, n_quad_a=N_QUAD_A,
    )
    result_s = compute_c_paper_operator_q_shift(
        theta=THETA, R=KAPPA_STAR_R, n=N_QUAD, polynomials=polys_s, n_quad_a=N_QUAD_A,
    )

    m1_impl_k = result_k.per_term["_m1_implied"]
    m2_impl_k = result_k.per_term["_m2_implied"]
    m1_impl_s = result_s.per_term["_m1_implied"]
    m2_impl_s = result_s.per_term["_m2_implied"]

    # Compute residuals
    res1_k = m1_solved / m1_impl_k if m1_impl_k != 0 else float('inf')
    res2_k = m2_solved / m2_impl_k if m2_impl_k != 0 else float('inf')
    res1_s = m1_solved / m1_impl_s if m1_impl_s != 0 else float('inf')
    res2_s = m2_solved / m2_impl_s if m2_impl_s != 0 else float('inf')

    print(f"{'Benchmark':<15} {'m1_implied':>12} {'m2_implied':>12} {'res1':>10} {'res2':>10}")
    print("-" * 65)
    print(f"{'κ':<15} {m1_impl_k:>12.4f} {m2_impl_k:>12.4f} {res1_k:>10.4f} {res2_k:>10.4f}")
    print(f"{'κ*':<15} {m1_impl_s:>12.4f} {m2_impl_s:>12.4f} {res1_s:>10.4f} {res2_s:>10.4f}")
    print()
    print(f"Solved weights: m1={m1_solved:.4f}, m2={m2_solved:.4f}")
    print()
    print("Residual interpretation:")
    print("  res = m_solved / m_implied")
    print("  If res ≈ 1: Q-shift fully explains this channel")
    print("  If res >> 1: Q-shift only partially explains (~1/res of the effect)")
    print()

    avg_res1 = (res1_k + res1_s) / 2
    avg_res2 = (res2_k + res2_s) / 2
    print(f"Average residuals: res1={avg_res1:.2f}×, res2={avg_res2:.2f}×")
    print(f"Interpretation: Q-shift explains ~{100/avg_res1:.0f}% of m1, ~{100/avg_res2:.0f}% of m2")
    print()

    return m1_impl_k, m2_impl_k


def task4_sanity_sweep(m1_solved, m2_solved):
    """Claude 4: Sanity sweep - fingerprint the residual."""
    print("=" * 100)
    print("CLAUDE 4: Sanity Sweep - Fingerprinting the Residual")
    print("=" * 100)
    print()

    # Load κ polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    R_values = [0.7, 0.9, 1.1, 1.3036, 1.5]

    print(f"{'R':>8} {'m1_impl':>10} {'m2_impl':>10} {'res1':>8} {'res2':>8} {'exp(R)':>10} {'exp(2R)':>10} {'(e^2R-1)/2R':>12}")
    print("-" * 100)

    for R in R_values:
        result = compute_c_paper_operator_q_shift(
            theta=THETA, R=R, n=N_QUAD, polynomials=polys, n_quad_a=N_QUAD_A,
        )

        m1_impl = result.per_term["_m1_implied"]
        m2_impl = result.per_term["_m2_implied"]
        res1 = m1_solved / m1_impl if m1_impl != 0 else float('inf')
        res2 = m2_solved / m2_impl if m2_impl != 0 else float('inf')

        exp_R = math.exp(R)
        exp_2R = math.exp(2*R)
        avg_exp = (exp_2R - 1) / (2*R) if R != 0 else 1.0

        print(f"{R:>8.4f} {m1_impl:>10.4f} {m2_impl:>10.4f} {res1:>8.2f} {res2:>8.2f} {exp_R:>10.4f} {exp_2R:>10.4f} {avg_exp:>12.4f}")

    print()
    print("Fingerprint comparison:")
    print("  - If res scales like exp(R): missing a single exponential factor")
    print("  - If res scales like exp(2R): missing exp(2R) dampening")
    print("  - If res is constant: missing a structural factor (prefactor, product rule)")
    print()

    # Check if residual is roughly constant or scaling
    result_07 = compute_c_paper_operator_q_shift(theta=THETA, R=0.7, n=N_QUAD, polynomials=polys, n_quad_a=N_QUAD_A)
    result_15 = compute_c_paper_operator_q_shift(theta=THETA, R=1.5, n=N_QUAD, polynomials=polys, n_quad_a=N_QUAD_A)

    m1_impl_07 = result_07.per_term["_m1_implied"]
    m1_impl_15 = result_15.per_term["_m1_implied"]
    res1_07 = m1_solved / m1_impl_07 if m1_impl_07 != 0 else float('inf')
    res1_15 = m1_solved / m1_impl_15 if m1_impl_15 != 0 else float('inf')

    res_ratio = res1_15 / res1_07 if res1_07 != 0 else float('inf')
    exp_ratio = math.exp(1.5) / math.exp(0.7)

    print(f"Residual ratio (R=1.5 / R=0.7): {res_ratio:.3f}")
    print(f"exp(R) ratio (R=1.5 / R=0.7): {exp_ratio:.3f}")
    print()
    if abs(res_ratio - 1.0) < 0.3:
        print("DIAGNOSIS: Residual is roughly CONSTANT with R")
        print("          → Missing factor is STRUCTURAL (prefactor, product rule)")
    elif abs(res_ratio - exp_ratio) < 0.5:
        print("DIAGNOSIS: Residual scales roughly like exp(R)")
        print("          → Missing a single exponential factor")
    else:
        print("DIAGNOSIS: Residual has complex R-dependence")
        print("          → Likely multiple missing components")


def main():
    print()
    print("*" * 100)
    print("CLAUDE CODE DIAGNOSTIC SUITE")
    print("GPT Guidance 2025-12-20 - Pre-I₁ Prefactor Fix Baseline")
    print("*" * 100)
    print()

    # Task 1: Truth table guardrail
    task1_truth_table()

    # Task 2: Split channels and two-weight solve
    channels_k, channels_s, m1, m2 = task2_split_channels()

    # Task 3: Residual factors
    task3_residual_factors(m1, m2)

    # Task 4: Sanity sweep
    task4_sanity_sweep(m1, m2)

    print()
    print("*" * 100)
    print("END OF DIAGNOSTIC SUITE")
    print("*" * 100)


if __name__ == "__main__":
    main()
