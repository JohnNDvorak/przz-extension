#!/usr/bin/env python3
"""
Phase 45.3: Split-Q Microcase Derivation

This script derives g_I1 and g_I2 from first principles using the Split-Q
microcase matrix. The key insight is that by running experiments where I1
and I2 have DIFFERENT Q modes, we can isolate each component's contribution.

MICROCASE MATRIX:
| Case | I1 Q-mode | I2 Q-mode | What it measures |
|------|-----------|-----------|------------------|
| A | Q=1 | Q=1 | Pure baseline (no Q effects) |
| B | Q=1 | Q=real | Q effect on I2 ONLY |
| C | Q=real | Q=1 | Q effect on I1 ONLY |
| D | Q=real | Q=real | Full case |

DERIVATION:
delta_c_I1 = c(C) - c(A)   # Q's effect through I1
delta_c_I2 = c(B) - c(A)   # Q's effect through I2

delta_g_I1 = delta_c_I1 / (base × I1_baseline(-R))
delta_g_I2 = delta_c_I2 / (base × I2_baseline(-R))

g_I1_derived = 1.0 + delta_g_I1     # I1 self-corrects to 1.0
g_I2_derived = g_baseline + delta_g_I2  # I2 needs baseline + Q effect

Created: 2025-12-27 (Phase 45.3)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.correction_policy import (
    compute_g_baseline,
    compute_base,
    G_I1_CALIBRATED,
    G_I2_CALIBRATED,
)
from src.unified_i1_paper import compute_I1_unified_paper
from src.unified_i2_paper import compute_I2_unified_paper
from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term


@dataclass
class SplitQResult:
    """Result of Split-Q computation for a single microcase."""
    name: str
    q_I1: str        # "real" or "none"
    q_I2: str        # "real" or "none"
    I1_total: float
    I2_total: float
    S12_total: float
    S34_total: float
    c_value: float


def compute_I1_I2_with_split_q(
    R: float,
    theta: float,
    polynomials: Dict,
    q_mode_I1: str,  # "real" or "none"
    q_mode_I2: str,  # "real" or "none"
    n_quad: int = 60,
    n_quad_a: int = 40,
) -> Tuple[float, float]:
    """
    Compute I1 and I2 totals with separate Q modes for each.

    Args:
        R: PRZZ R parameter
        theta: θ parameter
        polynomials: Dict with P1, P2, P3, Q
        q_mode_I1: "real" (include Q) or "none" (Q=1) for I1
        q_mode_I2: "real" (include Q) or "none" (Q=1) for I2
        n_quad: Quadrature points
        n_quad_a: Quadrature points for Case C a-integral

    Returns:
        (I1_total, I2_total)
    """
    # Factorial normalization
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    s_factor = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    include_Q_I1 = (q_mode_I1 == "real")
    include_Q_I2 = (q_mode_I2 == "real")

    I1_total = 0.0
    I2_total = 0.0

    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        ell1, ell2 = int(pair_key[0]), int(pair_key[1])
        norm = f_norm[pair_key] * s_factor[pair_key]

        # I1 computation
        I1_result = compute_I1_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=n_quad_a,
            include_Q=include_Q_I1,
        )
        I1_total += norm * I1_result.I1_value

        # I2 computation
        I2_result = compute_I2_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=n_quad_a,
            include_Q=include_Q_I2,
        )
        I2_total += norm * I2_result.I2_value

    return I1_total, I2_total


def compute_S34(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """Compute S34 = I3 + I4 (always uses real Q)."""
    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")

    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    S34 = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms[pair_key]
        norm = factorial_norm[pair_key]
        sym = symmetry_factor[pair_key]

        for term in terms[2:4]:  # I3 and I4
            result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
            S34 += sym * norm * result.value

    return S34


def compute_c_split_q(
    R: float,
    theta: float,
    polynomials: Dict,
    q_mode_I1: str,
    q_mode_I2: str,
    K: int = 3,
    n_quad: int = 60,
) -> SplitQResult:
    """
    Compute c with split Q modes for I1 and I2.

    Uses the mirror formula:
    c = I1(+R) + g*base*I1(-R) + I2(+R) + g*base*I2(-R) + S34

    For this experiment, we use g=g_baseline for both I1 and I2 to measure
    the raw Q effect, not the corrected c.
    """
    g_baseline = compute_g_baseline(theta, K)
    base = compute_base(R, K)
    m = g_baseline * base

    # Compute I1/I2 at +R and -R with split Q modes
    I1_plus, I2_plus = compute_I1_I2_with_split_q(
        R, theta, polynomials, q_mode_I1, q_mode_I2, n_quad
    )
    I1_minus, I2_minus = compute_I1_I2_with_split_q(
        -R, theta, polynomials, q_mode_I1, q_mode_I2, n_quad
    )

    # S34 always uses real Q (it's not part of the I1/I2 split experiment)
    S34 = compute_S34(theta, R, polynomials, n_quad)

    # Assemble c using baseline g
    S12_plus = I1_plus + I2_plus
    S12_minus = I1_minus + I2_minus
    c = S12_plus + m * S12_minus + S34

    name = f"I1_Q={q_mode_I1}, I2_Q={q_mode_I2}"

    return SplitQResult(
        name=name,
        q_I1=q_mode_I1,
        q_I2=q_mode_I2,
        I1_total=I1_plus + m * I1_minus,
        I2_total=I2_plus + m * I2_minus,
        S12_total=S12_plus + m * S12_minus,
        S34_total=S34,
        c_value=c,
    )


def run_split_q_derivation(
    polyset_name: str,
    polynomials: Dict,
    R: float,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
):
    """Run Split-Q derivation for a polynomial set."""
    print(f"\n{'='*70}")
    print(f"  {polyset_name} (R = {R})")
    print(f"{'='*70}")

    g_baseline = compute_g_baseline(theta, K)
    base = compute_base(R, K)

    # Run all 4 microcases
    print("\nMicrocase Matrix:")
    print("-" * 70)

    results = {}
    for q_I1, q_I2, case_name in [
        ("none", "none", "A"),   # Pure baseline
        ("none", "real", "B"),   # Q → I2 only
        ("real", "none", "C"),   # Q → I1 only
        ("real", "real", "D"),   # Full
    ]:
        result = compute_c_split_q(R, theta, polynomials, q_I1, q_I2, K, n_quad)
        results[case_name] = result
        print(f"  Case {case_name}: I1_Q={q_I1:4s}, I2_Q={q_I2:4s} -> c = {result.c_value:.10f}")

    print()

    # Compute I1/I2 baselines at -R with Q=1
    I1_baseline_minus, I2_baseline_minus = compute_I1_I2_with_split_q(
        -R, theta, polynomials, "none", "none", n_quad
    )

    print("Baseline I1/I2 at -R (Q=1):")
    print(f"  I1(-R) = {I1_baseline_minus:.10f}")
    print(f"  I2(-R) = {I2_baseline_minus:.10f}")
    print()

    # Compute deltas
    c_A = results["A"].c_value
    c_B = results["B"].c_value
    c_C = results["C"].c_value
    c_D = results["D"].c_value

    delta_c_I1 = c_C - c_A  # Q effect on I1
    delta_c_I2 = c_B - c_A  # Q effect on I2
    delta_c_total = c_D - c_A  # Total Q effect

    print("Delta c from Q effects:")
    print(f"  delta_c_I1 = c(C) - c(A) = {delta_c_I1:+.10f}")
    print(f"  delta_c_I2 = c(B) - c(A) = {delta_c_I2:+.10f}")
    print(f"  delta_c_total = c(D) - c(A) = {delta_c_total:+.10f}")
    print(f"  Additivity check: I1+I2 = {delta_c_I1 + delta_c_I2:+.10f}")
    print(f"  Additivity gap: {abs(delta_c_I1 + delta_c_I2 - delta_c_total):.10f}")
    print()

    # Convert to delta_g
    delta_g_I1 = delta_c_I1 / (base * I1_baseline_minus)
    delta_g_I2 = delta_c_I2 / (base * I2_baseline_minus)

    print("Delta g from Q effects:")
    print(f"  delta_g_I1 = {delta_g_I1:+.10f}")
    print(f"  delta_g_I2 = {delta_g_I2:+.10f}")
    print()

    # Derive g_I1 and g_I2
    # Hypothesis: I1 self-corrects to 1.0, I2 needs baseline + Q effect
    g_I1_derived = 1.0 + delta_g_I1
    g_I2_derived = g_baseline + delta_g_I2

    print("DERIVED VALUES:")
    print("-" * 70)
    print(f"  g_I1_derived = 1.0 + delta_g_I1 = {g_I1_derived:.8f}")
    print(f"  g_I2_derived = g_baseline + delta_g_I2 = {g_I2_derived:.8f}")
    print()

    print("COMPARISON TO CALIBRATED VALUES:")
    print("-" * 70)
    print(f"  g_I1: derived={g_I1_derived:.8f} vs calibrated={G_I1_CALIBRATED:.8f}")
    print(f"        difference = {g_I1_derived - G_I1_CALIBRATED:+.8f} ({(g_I1_derived/G_I1_CALIBRATED - 1)*100:+.4f}%)")
    print()
    print(f"  g_I2: derived={g_I2_derived:.8f} vs calibrated={G_I2_CALIBRATED:.8f}")
    print(f"        difference = {g_I2_derived - G_I2_CALIBRATED:+.8f} ({(g_I2_derived/G_I2_CALIBRATED - 1)*100:+.4f}%)")
    print()

    return {
        "g_I1_derived": g_I1_derived,
        "g_I2_derived": g_I2_derived,
        "delta_g_I1": delta_g_I1,
        "delta_g_I2": delta_g_I2,
        "additivity_gap": abs(delta_c_I1 + delta_c_I2 - delta_c_total),
    }


def main():
    print()
    print("=" * 70)
    print("  PHASE 45.3: SPLIT-Q FIRST-PRINCIPLES DERIVATION")
    print("=" * 70)
    print()
    print("Deriving g_I1 and g_I2 from Split-Q microcase experiments.")
    print("NO c_target values used - derivation from integrals only.")
    print()
    print(f"Calibrated targets (for comparison):")
    print(f"  g_I1 = {G_I1_CALIBRATED:.8f}")
    print(f"  g_I2 = {G_I2_CALIBRATED:.8f}")

    # Load polynomial sets
    P1, P2, P3, Q = load_przz_polynomials()
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    theta = 4 / 7
    K = 3

    # Run derivation for both benchmarks
    result_kappa = run_split_q_derivation(
        "κ BENCHMARK", polys_kappa, R=1.3036, theta=theta, K=K, n_quad=60
    )

    result_kappa_star = run_split_q_derivation(
        "κ* BENCHMARK", polys_kappa_star, R=1.1167, theta=theta, K=K, n_quad=60
    )

    # Summary
    print()
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print()

    print("Derived values from κ benchmark:")
    print(f"  g_I1 = {result_kappa['g_I1_derived']:.8f}")
    print(f"  g_I2 = {result_kappa['g_I2_derived']:.8f}")
    print()

    print("Derived values from κ* benchmark:")
    print(f"  g_I1 = {result_kappa_star['g_I1_derived']:.8f}")
    print(f"  g_I2 = {result_kappa_star['g_I2_derived']:.8f}")
    print()

    print("Consistency check (derived values should be similar across benchmarks):")
    g_I1_diff = abs(result_kappa['g_I1_derived'] - result_kappa_star['g_I1_derived'])
    g_I2_diff = abs(result_kappa['g_I2_derived'] - result_kappa_star['g_I2_derived'])
    print(f"  g_I1 difference: {g_I1_diff:.8f}")
    print(f"  g_I2 difference: {g_I2_diff:.8f}")
    print()

    # Average derived values
    g_I1_avg = (result_kappa['g_I1_derived'] + result_kappa_star['g_I1_derived']) / 2
    g_I2_avg = (result_kappa['g_I2_derived'] + result_kappa_star['g_I2_derived']) / 2

    print("Average derived values:")
    print(f"  g_I1_avg = {g_I1_avg:.8f} (calibrated: {G_I1_CALIBRATED:.8f})")
    print(f"  g_I2_avg = {g_I2_avg:.8f} (calibrated: {G_I2_CALIBRATED:.8f})")
    print()

    # Check success
    g_I1_match = abs(g_I1_avg - G_I1_CALIBRATED) < 0.01
    g_I2_match = abs(g_I2_avg - G_I2_CALIBRATED) < 0.01

    if g_I1_match and g_I2_match:
        print("STATUS: DERIVATION SUCCESSFUL")
        print("  The Split-Q method derives g_I1 and g_I2 from first principles!")
    else:
        print("STATUS: DERIVATION NEEDS REFINEMENT")
        print("  The derived values don't match calibrated values closely enough.")
        print("  Consider: frozen-Q decomposition, different baseline assumptions.")


if __name__ == "__main__":
    main()
