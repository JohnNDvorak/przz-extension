"""
src/test_case_c_derivatives.py
Test Case C derivatives for I3/I4 terms.

This script computes c using Case C structure for all terms and compares
to raw computation at both R benchmarks.

Key hypothesis: Case C derivatives may change sign patterns in ways that
INCREASE c (because some derivative terms are negative).
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, Tuple

from src.polynomials import load_przz_polynomials
from src.quadrature import tensor_grid_2d
from src.case_c_kernel import (
    compute_case_c_kernel,
    compute_case_c_kernel_derivative,
)
from src.evaluate import evaluate_terms
from src.terms_k3_d1 import make_all_terms_k3


THETA = 4/7
R1 = 1.3036
R2 = 1.1167
KAPPA1 = 0.417293962
KAPPA2 = 0.407511457


def compute_c_raw(R: float, n_quad: int = 60) -> Tuple[float, Dict[str, float]]:
    """Compute c using raw (no Case C) computation."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    factorial_norm = {'11': 1.0, '22': 1.0/4, '33': 1.0/36, '12': 1.0/2, '13': 1.0/6, '23': 1.0/12}
    symmetry = {'11': 1.0, '22': 1.0, '33': 1.0, '12': 2.0, '13': 2.0, '23': 2.0}

    terms_all = make_all_terms_k3(THETA, R)
    total = 0.0
    per_pair = {}

    for pair_key, terms in terms_all.items():
        result = evaluate_terms(terms, polys, n_quad, return_breakdown=True)
        norm = factorial_norm[pair_key] * symmetry[pair_key]
        contrib = result.total * norm
        per_pair[pair_key] = contrib
        total += contrib

    return total, per_pair


def compute_c_case_c(R: float, n_quad: int = 60, n_quad_a: int = 30) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Compute c using Case C structure for all terms.

    This is a simplified computation that approximates the full Case C structure.
    It replaces P with K for ω > 0 polynomials and uses K' for derivatives.

    NOTE: This is not the full DSL implementation, but gives insight into
    how Case C affects the derivative terms.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    U, T, W = tensor_grid_2d(n_quad)
    U_flat = U.flatten()
    T_flat = T.flatten()
    W_flat = W.flatten()

    factorial_norm = {'11': 1.0, '22': 1.0/4, '33': 1.0/36, '12': 1.0/2, '13': 1.0/6, '23': 1.0/12}
    symmetry = {'11': 1.0, '22': 1.0, '33': 1.0, '12': 2.0, '13': 2.0, '23': 2.0}

    # Polynomial info: (P_left, P_right, omega_left, omega_right, (1-u) power for I3/I4)
    pairs_info = {
        '11': (P1, P1, 0, 0, 1, 1),  # ω_left, ω_right, power_I3, power_I4
        '12': (P1, P2, 0, 1, 1, 2),
        '13': (P1, P3, 0, 2, 1, 3),
        '22': (P2, P2, 1, 1, 2, 2),
        '23': (P2, P3, 1, 2, 2, 3),
        '33': (P3, P3, 2, 2, 3, 3),
    }

    Q_sq = Q.eval(T_flat) ** 2
    exp_2R = np.exp(2 * R * T_flat)

    total = 0.0
    per_pair = {}

    for pair_key, (P_left, P_right, omega_l, omega_r, power_I3, power_I4) in pairs_info.items():
        # Compute kernels and derivatives
        if omega_l == 0:
            K_left = P_left.eval(U_flat)
            K_left_deriv = P_left.eval_deriv(U_flat, 1)
        else:
            K_left = compute_case_c_kernel(P_left.eval, U_flat, omega_l, R, THETA, n_quad_a)
            K_left_deriv = compute_case_c_kernel_derivative(
                P_left.eval, lambda x: P_left.eval_deriv(x, 1),
                U_flat, omega_l, R, THETA, n_quad_a
            )

        if omega_r == 0:
            K_right = P_right.eval(U_flat)
            K_right_deriv = P_right.eval_deriv(U_flat, 1)
        else:
            K_right = compute_case_c_kernel(P_right.eval, U_flat, omega_r, R, THETA, n_quad_a)
            K_right_deriv = compute_case_c_kernel_derivative(
                P_right.eval, lambda x: P_right.eval_deriv(x, 1),
                U_flat, omega_r, R, THETA, n_quad_a
            )

        # I₂: (1/θ) × K_left × K_right × Q² × exp(2Rt)
        I2 = float(np.sum(W_flat * K_left * K_right * Q_sq * exp_2R)) / THETA

        # I₃: -(1/θ) × K'_left × K_right × (1-u)^{power_I3} × Q² × exp(2Rt)
        # This is simplified - real DSL extracts derivatives from series expansion
        poly_pref_I3 = (1 - U_flat) ** power_I3
        I3 = -float(np.sum(W_flat * K_left_deriv * K_right * poly_pref_I3 * Q_sq * exp_2R)) / THETA

        # I₄: -(1/θ) × K_left × K'_right × (1-u)^{power_I4} × Q² × exp(2Rt)
        poly_pref_I4 = (1 - U_flat) ** power_I4
        I4 = -float(np.sum(W_flat * K_left * K_right_deriv * poly_pref_I4 * Q_sq * exp_2R)) / THETA

        # I₁: (1/θ) × K'_left × K'_right × (1-u)^{power_I1} × Q² × exp(2Rt)
        # power_I1 = ℓ₁ + ℓ₂
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])
        poly_pref_I1 = (1 - U_flat) ** (ell1 + ell2)
        I1 = float(np.sum(W_flat * K_left_deriv * K_right_deriv * poly_pref_I1 * Q_sq * exp_2R)) / THETA

        # Apply normalization
        norm = factorial_norm[pair_key] * symmetry[pair_key]
        pair_total = (I1 + I2 + I3 + I4) * norm

        per_pair[pair_key] = {
            'I1': I1 * norm,
            'I2': I2 * norm,
            'I3': I3 * norm,
            'I4': I4 * norm,
            'total': pair_total,
        }
        total += pair_total

    return total, per_pair


def compare_raw_vs_case_c(verbose: bool = True):
    """Compare raw and Case C computations at both R benchmarks."""
    c1_target = math.exp(R1 * (1 - KAPPA1))
    c2_target = math.exp(R2 * (1 - KAPPA2))
    target_ratio = c1_target / c2_target

    # Raw computation
    c1_raw, pairs1_raw = compute_c_raw(R1)
    c2_raw, pairs2_raw = compute_c_raw(R2)

    # Case C computation
    c1_cc, pairs1_cc = compute_c_case_c(R1)
    c2_cc, pairs2_cc = compute_c_case_c(R2)

    if verbose:
        print('=' * 80)
        print('CASE C DERIVATIVES: RAW vs CASE C COMPARISON')
        print('=' * 80)

        print(f'\nTarget R-sensitivity: {(target_ratio-1)*100:.2f}%')
        print(f'Target c(R1): {c1_target:.6f}')
        print(f'Target c(R2): {c2_target:.6f}')

        print('\n--- RAW COMPUTATION ---')
        print(f'  c(R1) = {c1_raw:.10f}')
        print(f'  c(R2) = {c2_raw:.10f}')
        print(f'  Ratio = {c1_raw/c2_raw:.6f} ({(c1_raw/c2_raw-1)*100:.2f}%)')
        print(f'  Gap at R1: {(c1_raw - c1_target)/c1_target*100:+.2f}%')
        print(f'  Gap at R2: {(c2_raw - c2_target)/c2_target*100:+.2f}%')

        print('\n--- CASE C COMPUTATION ---')
        print(f'  c(R1) = {c1_cc:.10f}')
        print(f'  c(R2) = {c2_cc:.10f}')
        print(f'  Ratio = {c1_cc/c2_cc:.6f} ({(c1_cc/c2_cc-1)*100:.2f}%)')
        print(f'  Gap at R1: {(c1_cc - c1_target)/c1_target*100:+.2f}%')
        print(f'  Gap at R2: {(c2_cc - c2_target)/c2_target*100:+.2f}%')

        print('\n--- PER-PAIR COMPARISON (R=1.3036) ---')
        print(f'  Pair    Raw Total      Case C Total    Change')
        print('  ' + '-' * 55)
        for pair in ['11', '12', '13', '22', '23', '33']:
            raw_val = pairs1_raw[pair]
            cc_val = pairs1_cc[pair]['total']
            change_pct = (cc_val - raw_val) / abs(raw_val) * 100 if abs(raw_val) > 1e-15 else 0
            print(f'  ({pair[0]},{pair[1]})  {raw_val:>+14.8f}  {cc_val:>+14.8f}  {change_pct:>+8.2f}%')

        print('\n--- TERM-LEVEL BREAKDOWN FOR (2,2) PAIR ---')
        # Compare I1, I2, I3, I4 for (2,2) which is worst offender
        raw_22_r1 = evaluate_terms(
            make_all_terms_k3(THETA, R1)['22'],
            {'P1': load_przz_polynomials(enforce_Q0=True)[0],
             'P2': load_przz_polynomials(enforce_Q0=True)[1],
             'P3': load_przz_polynomials(enforce_Q0=True)[2],
             'Q': load_przz_polynomials(enforce_Q0=True)[3]},
            60, return_breakdown=True
        )
        norm_22 = 1.0/4 * 1.0

        print(f'  Term      Raw(R1)        CaseC(R1)')
        print('  ' + '-' * 45)
        for term in ['I1', 'I2', 'I3', 'I4']:
            raw_val = raw_22_r1.per_term.get(f'{term}_22', 0) * norm_22
            cc_val = pairs1_cc['22'][term]
            print(f'  {term}      {raw_val:>+12.8f}  {cc_val:>+12.8f}')

        print('\n--- ANALYSIS ---')
        raw_r_sens = (c1_raw/c2_raw - 1) * 100
        cc_r_sens = (c1_cc/c2_cc - 1) * 100
        target_r_sens = (target_ratio - 1) * 100

        print(f'  Raw R-sensitivity:    {raw_r_sens:.2f}%')
        print(f'  Case C R-sensitivity: {cc_r_sens:.2f}%')
        print(f'  Target R-sensitivity: {target_r_sens:.2f}%')

        if abs(cc_r_sens - target_r_sens) < abs(raw_r_sens - target_r_sens):
            print('\n  ✓ Case C IMPROVES R-sensitivity matching!')
        else:
            print('\n  ✗ Case C does NOT improve R-sensitivity')

        if c1_cc > c1_raw:
            print(f'  ✓ Case C INCREASES c at R1 ({c1_raw:.4f} → {c1_cc:.4f})')
        else:
            print(f'  ✗ Case C DECREASES c at R1 ({c1_raw:.4f} → {c1_cc:.4f})')

        print('=' * 80)

    return {
        'c1_raw': c1_raw, 'c2_raw': c2_raw,
        'c1_cc': c1_cc, 'c2_cc': c2_cc,
        'c1_target': c1_target, 'c2_target': c2_target,
        'pairs1_raw': pairs1_raw, 'pairs2_raw': pairs2_raw,
        'pairs1_cc': pairs1_cc, 'pairs2_cc': pairs2_cc,
    }


if __name__ == '__main__':
    compare_raw_vs_case_c(verbose=True)
