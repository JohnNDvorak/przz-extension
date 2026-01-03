#!/usr/bin/env python3
"""
Unit Test: Symbolic I_j vs Numeric Engine

Step 1 of GPT marching plan: Lock the target functional.

This script compares symbolic integrals (raw regime) against numeric engine
(paper regime) at multiple R values to precisely identify where discrepancies
appear.

Key question: Is the mismatch purely due to Case C attenuation?

Usage:
    python -m src.path_a.unit_test_symbolic
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sympy import N
from typing import Dict, List, Tuple
import math

from src.path_a.optimal_coeffs import R, R_star_approx
from src.path_a.symbolic_pairs import compute_pair
from src.polynomials import P1Polynomial, PellPolynomial, QPolynomial


def get_optimal_polynomials() -> Dict:
    """Load optimal polynomial objects for numeric engine."""
    from src.path_a.optimal_coeffs import P1_tilde, P2_tilde, P3_tilde, Q_coeffs

    # P1 tilde coefficients
    p1_tilde = [
        float(P1_tilde['a0']), float(P1_tilde['a1']),
        float(P1_tilde['a2']), float(P1_tilde['a3'])
    ]

    # P2 tilde coefficients
    p2_tilde = [
        float(P2_tilde['b0']), float(P2_tilde['b1']), float(P2_tilde['b2'])
    ]

    # P3 tilde coefficients
    p3_tilde = [
        float(P3_tilde['c0']), float(P3_tilde['c1']), float(P3_tilde['c2'])
    ]

    # Q coefficients in (1-2x)^k basis
    q_basis = {
        0: float(Q_coeffs['q0']),
        1: float(Q_coeffs['q1']),
        3: float(Q_coeffs['q3']),
        5: float(Q_coeffs['q5']),
    }

    P1 = P1Polynomial(tilde_coeffs=p1_tilde)
    P2 = PellPolynomial(tilde_coeffs=p2_tilde)
    P3 = PellPolynomial(tilde_coeffs=p3_tilde)

    Q = QPolynomial(basis_coeffs=q_basis, enforce_Q0=False)

    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


def compute_symbolic_I2(ell1: int, ell2: int, R_val: float) -> float:
    """Compute I2 symbolically at given R."""
    results = compute_pair(ell1, ell2, verbose=False)
    if 'I2' not in results or 'expr' not in results['I2']:
        return float('nan')
    expr = results['I2']['expr']
    return float(N(expr.subs(R, R_val), 30))


def compute_numeric_I2_raw(ell1: int, ell2: int, R_val: float, polys: Dict) -> float:
    """Compute I2 numerically using RAW regime."""
    from src.unified_i2_general import compute_I2_unified_general
    theta = 4/7
    result = compute_I2_unified_general(
        R_val, theta, ell1, ell2, polys,
        n_quad_u=60, n_quad_t=60, include_Q=True
    )
    return result.I2_value


def compute_numeric_I2_paper(ell1: int, ell2: int, R_val: float, polys: Dict) -> float:
    """Compute I2 numerically using PAPER regime (Case C attenuation)."""
    from src.unified_i2_paper import compute_I2_unified_paper
    theta = 4/7
    result = compute_I2_unified_paper(
        R_val, theta, ell1, ell2, polys,
        n_quad_u=60, n_quad_t=60, n_quad_a=40, include_Q=True
    )
    return result.I2_value


def run_I2_comparison(R_values: List[float]) -> None:
    """Compare I2 across symbolic, numeric raw, and numeric paper."""
    print("=" * 80)
    print("UNIT TEST: I2 Comparison (Symbolic vs Numeric Raw vs Numeric Paper)")
    print("=" * 80)

    polys = get_optimal_polynomials()
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    for R_val in R_values:
        print(f"\n{'='*60}")
        print(f"R = {R_val}")
        print(f"{'='*60}")
        print(f"{'Pair':>8} {'Symbolic':>14} {'Num Raw':>14} {'Num Paper':>14} {'Sym/Raw':>10} {'Raw/Paper':>10}")
        print("-" * 80)

        for ell1, ell2 in pairs:
            sym_val = compute_symbolic_I2(ell1, ell2, R_val)
            raw_val = compute_numeric_I2_raw(ell1, ell2, R_val, polys)
            paper_val = compute_numeric_I2_paper(ell1, ell2, R_val, polys)

            sym_raw_ratio = sym_val / raw_val if abs(raw_val) > 1e-15 else float('inf')
            raw_paper_ratio = raw_val / paper_val if abs(paper_val) > 1e-15 else float('inf')

            print(f"({ell1},{ell2}):  {sym_val:>14.8f} {raw_val:>14.8f} {paper_val:>14.8f} {sym_raw_ratio:>10.4f} {raw_paper_ratio:>10.4f}")


def compute_symbolic_sum_at_R(R_val: float) -> Tuple[float, float]:
    """Compute symbolic S12 and S34 at given R."""
    from sympy import factorial, Rational

    S12 = 0.0
    S34 = 0.0

    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    for ell1, ell2 in pairs:
        sym_factor = 2 if ell1 != ell2 else 1
        fact_norm = 1.0 / (math.factorial(ell1) * math.factorial(ell2))
        weight = sym_factor * fact_norm

        results = compute_pair(ell1, ell2, verbose=False)

        I1 = I2 = I3 = I4 = 0.0
        for name, var in [('I1', 'I1'), ('I2', 'I2'), ('I3', 'I3'), ('I4', 'I4')]:
            if name in results and 'expr' in results[name]:
                val = float(N(results[name]['expr'].subs(R, R_val), 30))
                if name == 'I1':
                    I1 = val
                elif name == 'I2':
                    I2 = val
                elif name == 'I3':
                    I3 = val
                elif name == 'I4':
                    I4 = val

        S12 += weight * (I1 + I2)
        S34 += weight * (I3 + I4)

    return S12, S34


def compare_assembly(R_values: List[float]) -> None:
    """Compare full assembly: symbolic vs numeric (paper regime with mirror)."""
    print("\n" + "=" * 80)
    print("ASSEMBLY COMPARISON: Symbolic vs Numeric Paper+Mirror")
    print("=" * 80)

    polys = get_optimal_polynomials()
    theta = 4/7
    K = 3

    for R_val in R_values:
        print(f"\n--- R = {R_val} ---")

        # Symbolic computation
        S12_sym, S34_sym = compute_symbolic_sum_at_R(R_val)
        S12_sym_minus, _ = compute_symbolic_sum_at_R(-R_val)

        # Mirror formula (symbolic)
        G = 1.015  # correction factor
        M_sym = G * (np.exp(R_val) + 5)
        c_sym = S12_sym + M_sym * S12_sym_minus + S34_sym

        print(f"  SYMBOLIC (raw regime):")
        print(f"    S12(+R) = {S12_sym:.10f}")
        print(f"    S12(-R) = {S12_sym_minus:.10f}")
        print(f"    S34(+R) = {S34_sym:.10f}")
        print(f"    M(R)    = {M_sym:.10f}")
        print(f"    c(R)    = {c_sym:.10f}")

        # Numeric paper regime - need to compute manually
        try:
            from src.unified_i1_paper import compute_I1_unified_paper
            from src.unified_i2_paper import compute_I2_unified_paper

            S12_num = 0.0
            S34_num = 0.0
            S12_num_minus = 0.0

            pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

            for ell1, ell2 in pairs:
                sym_factor = 2 if ell1 != ell2 else 1
                fact_norm = 1.0 / (math.factorial(ell1) * math.factorial(ell2))
                weight = sym_factor * fact_norm

                # I1 and I2 at +R
                I1_res = compute_I1_unified_paper(R_val, theta, ell1, ell2, polys)
                I2_res = compute_I2_unified_paper(R_val, theta, ell1, ell2, polys)
                I1_val = I1_res.I1_value
                I2_val = I2_res.I2_value

                # For I3 and I4, we use I1 with boundary terms - approximate for now
                # Note: actual I3, I4 need separate computation

                S12_num += weight * (I1_val + I2_val)

                # I1 and I2 at -R (for mirror)
                I1_res_neg = compute_I1_unified_paper(-R_val, theta, ell1, ell2, polys)
                I2_res_neg = compute_I2_unified_paper(-R_val, theta, ell1, ell2, polys)
                S12_num_minus += weight * (I1_res_neg.I1_value + I2_res_neg.I2_value)

            m_num = np.exp(R_val) + 5  # no G factor in numeric
            c_num = S12_num + m_num * S12_num_minus  # incomplete (missing S34)

            print(f"\n  NUMERIC (paper regime, I1+I2 only):")
            print(f"    S12(+R) = {S12_num:.10f}")
            print(f"    S12(-R) = {S12_num_minus:.10f}")
            print(f"    m(R)    = {m_num:.10f}")

            print(f"\n  RATIOS:")
            ratio_S12 = S12_sym / S12_num if abs(S12_num) > 1e-15 else float('inf')
            print(f"    S12 symbolic/numeric = {ratio_S12:.4f}")

        except Exception as e:
            print(f"  NUMERIC: Failed - {e}")


def main():
    print("=" * 80)
    print("PATH A UNIT TESTS: Locking the Target Functional")
    print("=" * 80)
    print(f"\nR* ≈ {R_star_approx}")

    # Test at multiple R values as specified in GPT marching plan
    R_values = [0.8, 1.0, R_star_approx, 1.4]

    # 1. I2 comparison (cleanest integral - no derivatives)
    print("\n" + "=" * 80)
    print("STEP 1: I2 COMPARISON")
    print("=" * 80)
    run_I2_comparison(R_values)

    # 2. Full assembly comparison
    print("\n" + "=" * 80)
    print("STEP 2: ASSEMBLY COMPARISON")
    print("=" * 80)
    compare_assembly([R_star_approx])

    print("\n" + "=" * 80)
    print("UNIT TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
