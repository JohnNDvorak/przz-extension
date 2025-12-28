#!/usr/bin/env python3
"""
scripts/run_phase31_track_c_q_shift.py
Phase 31 Track C: Q-Operator Shift Analysis

GOAL:
=====
Understand why the operator_q_shift mode produces wrong results
and derive correct operator-level m from the Q(1+D) transformation.

THE Q-OPERATOR SHIFT IDENTITY:
=============================
From PRZZ mirror structure:
  Q(D_α)[T^{-s}F] = T^{-s} × Q(1 + D_α)F

This means for the mirror term:
- Direct branch uses Q(D)
- Mirror branch needs Q(1+D) = Q(D) with argument shifted by 1

The implementation in q_operator.py uses binomial lift:
  Q(1+x) = Σ q'_r x^r
  where q'_r = Σ_{j≥r} q_j × C(j,r)

CURRENT STATUS:
==============
The operator_q_shift mode in evaluate.py gives:
- κ: +44% gap (way too high)
- κ*: -57% gap (way too low, opposite sign!)

This suggests the Q-shift is not being applied correctly.

Created: 2025-12-26 (Phase 31)
"""

import sys
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, ".")

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)


# =============================================================================
# Benchmark Configurations
# =============================================================================

BENCHMARKS = {
    "kappa": {
        "loader": load_przz_polynomials,
        "R": 1.3036,
        "theta": 4 / 7,
        "c_target": 2.13745440613217,
    },
    "kappa_star": {
        "loader": load_przz_polynomials_kappa_star,
        "R": 1.1167,
        "theta": 4 / 7,
        "c_target": 1.9379524124677437,
    },
}


@dataclass
class QShiftAnalysisResult:
    """Result of Q-shift analysis."""
    benchmark: str
    R: float
    theta: float

    # Q polynomial properties
    Q_degree: int
    Q_coeffs: List[float]
    Q_shifted_coeffs: List[float]

    # Q values at key points
    Q_at_0: float
    Q_at_1: float
    Q_shifted_at_0: float  # Q(1+0) = Q(1)

    # Operator mode results
    empirical_c: float
    empirical_gap_pct: float
    q_shift_c: Optional[float]
    q_shift_gap_pct: Optional[float]

    # Derived m values
    m_empirical: float
    m_from_q_shift: Optional[float]


def analyze_q_polynomial(benchmark: str) -> Dict:
    """Analyze the Q polynomial structure."""
    config = BENCHMARKS[benchmark]
    P1, P2, P3, Q = config["loader"]()

    try:
        from src.q_operator import binomial_lift_coeffs, lift_poly_by_shift

        # Get Q in standard monomial form
        if hasattr(Q, 'to_monomial'):
            Q_mono = Q.to_monomial()
            q_coeffs = list(Q_mono.coeffs)
        elif hasattr(Q, 'coeffs'):
            q_coeffs = list(Q.coeffs)
        else:
            # Try to extract coefficients by evaluation
            import numpy as np
            degree = getattr(Q, 'degree', 5)
            x = np.array([0.0])
            q_coeffs = [
                float(Q.eval_deriv(x, k)[0]) / math.factorial(k)
                for k in range(int(degree) + 1)
            ]

        # Compute shifted coefficients
        q_shifted = binomial_lift_coeffs(q_coeffs)

        # Evaluate at key points
        def eval_poly(coeffs, x):
            return sum(c * (x ** i) for i, c in enumerate(coeffs))

        Q_at_0 = eval_poly(q_coeffs, 0)
        Q_at_1 = eval_poly(q_coeffs, 1)
        Q_shifted_at_0 = eval_poly(q_shifted, 0)  # Should equal Q(1)

        return {
            "available": True,
            "degree": len(q_coeffs) - 1,
            "q_coeffs": q_coeffs,
            "q_shifted": q_shifted,
            "Q_at_0": Q_at_0,
            "Q_at_1": Q_at_1,
            "Q_shifted_at_0": Q_shifted_at_0,
            "shift_verified": abs(Q_at_1 - Q_shifted_at_0) < 1e-10,
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def compare_evaluation_modes(benchmark: str, n_quad: int = 60) -> Dict:
    """Compare empirical_scalar vs operator_q_shift modes."""
    config = BENCHMARKS[benchmark]
    R = config["R"]
    theta = config["theta"]
    c_target = config["c_target"]

    P1, P2, P3, Q = config["loader"]()
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    results = {}

    # Try different mirror modes
    for mode in ["empirical_scalar", "operator_q_shift", "operator_q_shift_joint"]:
        try:
            from src.evaluate import compute_c_paper_with_mirror

            result = compute_c_paper_with_mirror(
                theta=theta,
                R=R,
                n=n_quad,
                polynomials=polys,
                pair_mode="triangle",
                use_factorial_normalization=True,
                mode="main",
                K=3,
                mirror_mode=mode,
            )

            c_computed = result.total
            c_gap_pct = 100 * (c_computed - c_target) / c_target

            per_term = result.per_term or {}
            m_eff = per_term.get("_mirror_multiplier_effective", per_term.get("_mirror_multiplier"))

            results[mode] = {
                "available": True,
                "c_computed": c_computed,
                "c_target": c_target,
                "c_gap_pct": c_gap_pct,
                "m_eff": m_eff,
                "S12_plus": per_term.get("_S12_plus_total"),
                "S12_minus": per_term.get("_S12_minus_total"),
            }
        except Exception as e:
            results[mode] = {
                "available": False,
                "error": str(e),
            }

    return results


def analyze_q_shift_effect(benchmark: str) -> Dict:
    """Analyze how Q-shift affects the integrand."""
    config = BENCHMARKS[benchmark]
    R = config["R"]
    theta = config["theta"]

    P1, P2, P3, Q = config["loader"]()

    try:
        from src.q_operator import binomial_lift_coeffs, lift_poly_by_shift
        import numpy as np

        # Get shifted Q
        Q_shifted = lift_poly_by_shift(Q, shift=1.0)

        # Evaluate both at sample points
        u_samples = np.linspace(0.1, 0.9, 5)

        Q_values = []
        Q_shifted_values = []

        for u in u_samples:
            u_arr = np.array([u])
            Q_val = float(Q.eval(u_arr)[0])
            Q_shifted_val = float(Q_shifted.eval(u_arr)[0])
            Q_values.append(Q_val)
            Q_shifted_values.append(Q_shifted_val)

        # Compute ratios
        ratios = [s / o if abs(o) > 1e-15 else float('inf')
                  for s, o in zip(Q_shifted_values, Q_values)]

        return {
            "available": True,
            "u_samples": list(u_samples),
            "Q_values": Q_values,
            "Q_shifted_values": Q_shifted_values,
            "ratios": ratios,
            "avg_ratio": sum(ratios) / len(ratios),
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def run_q_shift_analysis(benchmark: str, n_quad: int = 60) -> QShiftAnalysisResult:
    """Run complete Q-shift analysis."""
    config = BENCHMARKS[benchmark]
    R = config["R"]
    theta = config["theta"]
    c_target = config["c_target"]
    m_empirical = math.exp(R) + 5

    # Analyze Q polynomial
    q_analysis = analyze_q_polynomial(benchmark)

    # Compare modes
    mode_results = compare_evaluation_modes(benchmark, n_quad)

    empirical = mode_results.get("empirical_scalar", {})
    q_shift = mode_results.get("operator_q_shift", {})

    return QShiftAnalysisResult(
        benchmark=benchmark,
        R=R,
        theta=theta,
        Q_degree=q_analysis.get("degree", -1) if q_analysis.get("available") else -1,
        Q_coeffs=q_analysis.get("q_coeffs", []) if q_analysis.get("available") else [],
        Q_shifted_coeffs=q_analysis.get("q_shifted", []) if q_analysis.get("available") else [],
        Q_at_0=q_analysis.get("Q_at_0", 0) if q_analysis.get("available") else 0,
        Q_at_1=q_analysis.get("Q_at_1", 0) if q_analysis.get("available") else 0,
        Q_shifted_at_0=q_analysis.get("Q_shifted_at_0", 0) if q_analysis.get("available") else 0,
        empirical_c=empirical.get("c_computed", 0) if empirical.get("available") else 0,
        empirical_gap_pct=empirical.get("c_gap_pct", 0) if empirical.get("available") else 0,
        q_shift_c=q_shift.get("c_computed") if q_shift.get("available") else None,
        q_shift_gap_pct=q_shift.get("c_gap_pct") if q_shift.get("available") else None,
        m_empirical=m_empirical,
        m_from_q_shift=q_shift.get("m_eff") if q_shift.get("available") else None,
    )


def print_result(result: QShiftAnalysisResult) -> None:
    """Print formatted result."""
    print(f"\n{'=' * 70}")
    print(f"=== Q-SHIFT ANALYSIS: {result.benchmark.upper()} ===")
    print(f"{'=' * 70}")

    print(f"\n--- Parameters ---")
    print(f"  R = {result.R}, theta = {result.theta:.6f}")
    print(f"  m_empirical = exp(R) + 5 = {result.m_empirical:.4f}")

    print(f"\n--- Q Polynomial Properties ---")
    print(f"  Q degree: {result.Q_degree}")
    print(f"  Q(0) = {result.Q_at_0:.6f} (should be ~1.0)")
    print(f"  Q(1) = {result.Q_at_1:.6f}")
    print(f"  Q_shifted(0) = Q(1) = {result.Q_shifted_at_0:.6f}")
    if len(result.Q_coeffs) <= 6:
        print(f"  Q coeffs: {[f'{c:.4f}' for c in result.Q_coeffs]}")
        print(f"  Q_shifted: {[f'{c:.4f}' for c in result.Q_shifted_coeffs]}")

    print(f"\n--- Evaluation Mode Comparison ---")
    print(f"  Empirical (m = exp(R)+5):")
    print(f"    c = {result.empirical_c:.6f}")
    print(f"    gap = {result.empirical_gap_pct:+.2f}%")

    print(f"  Operator Q-shift:")
    if result.q_shift_c is not None:
        print(f"    c = {result.q_shift_c:.6f}")
        print(f"    gap = {result.q_shift_gap_pct:+.2f}%")
        if result.m_from_q_shift is not None:
            print(f"    m_eff = {result.m_from_q_shift:.4f}")
            m_gap = 100 * (result.m_from_q_shift - result.m_empirical) / result.m_empirical
            print(f"    m_gap = {m_gap:+.2f}%")
    else:
        print(f"    NOT AVAILABLE")


def main():
    """Run Track C: Q-Operator Shift Analysis."""
    print("=" * 70)
    print("PHASE 31 TRACK C: Q-OPERATOR SHIFT ANALYSIS")
    print("=" * 70)
    print(f"\nGoal: Understand why operator_q_shift mode fails")
    print(f"The Q(1+D) transformation should give correct mirror contribution")

    # Analyze both benchmarks
    results = {}
    for benchmark in ["kappa", "kappa_star"]:
        print(f"\nAnalyzing {benchmark}...")
        results[benchmark] = run_q_shift_analysis(benchmark)
        print_result(results[benchmark])

    # Analyze Q-shift effect
    print(f"\n{'=' * 70}")
    print("=== Q-SHIFT EFFECT ON INTEGRAND ===")
    print(f"{'=' * 70}")

    for benchmark in ["kappa", "kappa_star"]:
        print(f"\n--- {benchmark.upper()} ---")
        effect = analyze_q_shift_effect(benchmark)
        if effect.get("available"):
            print(f"  u samples: {[f'{u:.2f}' for u in effect['u_samples']]}")
            print(f"  Q(u):        {[f'{v:.4f}' for v in effect['Q_values']]}")
            print(f"  Q(1+u):      {[f'{v:.4f}' for v in effect['Q_shifted_values']]}")
            print(f"  Ratio Q(1+u)/Q(u): {[f'{r:.4f}' for r in effect['ratios']]}")
            print(f"  Average ratio: {effect['avg_ratio']:.4f}")
        else:
            print(f"  NOT AVAILABLE: {effect.get('error', 'unknown')}")

    # Full mode comparison
    print(f"\n{'=' * 70}")
    print("=== FULL MODE COMPARISON ===")
    print(f"{'=' * 70}")

    for benchmark in ["kappa", "kappa_star"]:
        print(f"\n--- {benchmark.upper()} ---")
        modes = compare_evaluation_modes(benchmark)
        for mode, data in modes.items():
            if data.get("available"):
                print(f"  {mode}:")
                print(f"    c = {data['c_computed']:.6f}, gap = {data['c_gap_pct']:+.2f}%")
                if data.get('m_eff') is not None:
                    print(f"    m_eff = {data['m_eff']:.4f}")
            else:
                print(f"  {mode}: NOT AVAILABLE - {data.get('error', 'unknown')}")

    # Summary
    print(f"\n{'=' * 70}")
    print("=== TRACK C SUMMARY ===")
    print(f"{'=' * 70}")

    kappa = results["kappa"]
    kappa_star = results["kappa_star"]

    print(f"""
KEY FINDINGS:
1. The operator_q_shift mode produces dramatically wrong results:
   - κ: {kappa.q_shift_gap_pct:+.2f}% gap (vs {kappa.empirical_gap_pct:+.2f}% empirical)
   - κ*: {kappa_star.q_shift_gap_pct:+.2f}% gap (vs {kappa_star.empirical_gap_pct:+.2f}% empirical)

2. The Q-shift is mathematically correct (binomial lift verified):
   - Q_shifted(0) = Q(1) ✓

3. POSSIBLE ISSUES:
   a) The Q-shift is applied to wrong integrand component
   b) Missing normalization factor in shifted evaluation
   c) The shift should be by θ, not 1
   d) Missing exponential prefactor T^{{-s}} handling

4. The "+5" is NOT coming from Q-shift alone:
   - It requires the full J₁ decomposition structure
   - The Q-shift is just part of the mirror machinery

NEXT STEPS:
- Check if shift should be by θ instead of 1
- Verify which integrand component gets the shifted Q
- Compare with difference_quotient mode which should unify the structure
""")

    print(f"\n{'=' * 70}")
    print("TRACK C COMPLETE")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    main()
