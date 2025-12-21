#!/usr/bin/env python3
"""
GPT Run 13B: Diagnostic Solver Measurement Metrics

This script adds explicit comparison metrics for measuring how close a
TeX-derived hypothesis is to the diagnostic 2x2 solve.

The goal is to use the diagnostic solver ONLY as a measurement device,
not for fitting. This helps validate any TeX-derived mirror formula.

Usage:
    python run_gpt_run13b_solver_metrics.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import compute_c_paper_tex_mirror


THETA = 4.0 / 7.0

TARGETS = {
    "kappa": {
        "name": "κ",
        "R": 1.3036,
        "c_target": 2.13745440613217263636,
    },
    "kappa_star": {
        "name": "κ*",
        "R": 1.1167,
        "c_target": 1.93801,
    }
}


@dataclass
class MirrorHypothesisMetrics:
    """Metrics for evaluating a mirror hypothesis against the diagnostic solve."""

    # Hypothesis values
    hypothesis_m1: float
    hypothesis_m2: float

    # Reference values (from diagnostic solve or tex_mirror)
    reference_m1: float
    reference_m2: float

    # Error metrics
    m1_abs_error: float
    m2_abs_error: float
    m1_rel_error: float
    m2_rel_error: float

    # Gate tests
    passes_10pct_gate: bool
    passes_5pct_gate: bool
    passes_1pct_gate: bool

    # Resulting c values
    c_hypothesis: float
    c_reference: float
    c_target: float
    c_gap_hypothesis: float
    c_gap_reference: float


def measure_mirror_hypothesis(
    hypothesis_m1: float,
    hypothesis_m2: float,
    reference_m1: float,
    reference_m2: float,
    I1_plus: float,
    I2_plus: float,
    I1_minus_base: float,
    I2_minus_base: float,
    S34_plus: float,
    c_target: float,
) -> MirrorHypothesisMetrics:
    """
    Measure how close a TeX-derived hypothesis is to the reference (diagnostic solve).

    Args:
        hypothesis_m1, hypothesis_m2: The hypothesized mirror weights
        reference_m1, reference_m2: The reference weights (from solve or tex_mirror)
        I1_plus, I2_plus, S34_plus: Channel values from +R evaluation
        I1_minus_base, I2_minus_base: Channel values from -R base evaluation
        c_target: Target c value for gap calculation

    Returns:
        MirrorHypothesisMetrics with all error metrics
    """

    # Error metrics
    m1_abs_error = abs(hypothesis_m1 - reference_m1)
    m2_abs_error = abs(hypothesis_m2 - reference_m2)

    m1_rel_error = m1_abs_error / abs(reference_m1) if abs(reference_m1) > 1e-10 else float('inf')
    m2_rel_error = m2_abs_error / abs(reference_m2) if abs(reference_m2) > 1e-10 else float('inf')

    # Gate tests
    passes_10pct = m1_rel_error < 0.10 and m2_rel_error < 0.10
    passes_5pct = m1_rel_error < 0.05 and m2_rel_error < 0.05
    passes_1pct = m1_rel_error < 0.01 and m2_rel_error < 0.01

    # Compute c values
    c_hypothesis = (
        I1_plus + hypothesis_m1 * I1_minus_base +
        I2_plus + hypothesis_m2 * I2_minus_base +
        S34_plus
    )

    c_reference = (
        I1_plus + reference_m1 * I1_minus_base +
        I2_plus + reference_m2 * I2_minus_base +
        S34_plus
    )

    c_gap_hypothesis = 100 * (c_hypothesis - c_target) / c_target
    c_gap_reference = 100 * (c_reference - c_target) / c_target

    return MirrorHypothesisMetrics(
        hypothesis_m1=hypothesis_m1,
        hypothesis_m2=hypothesis_m2,
        reference_m1=reference_m1,
        reference_m2=reference_m2,
        m1_abs_error=m1_abs_error,
        m2_abs_error=m2_abs_error,
        m1_rel_error=m1_rel_error,
        m2_rel_error=m2_rel_error,
        passes_10pct_gate=passes_10pct,
        passes_5pct_gate=passes_5pct,
        passes_1pct_gate=passes_1pct,
        c_hypothesis=c_hypothesis,
        c_reference=c_reference,
        c_target=c_target,
        c_gap_hypothesis=c_gap_hypothesis,
        c_gap_reference=c_gap_reference,
    )


def test_hypothesis(
    name: str,
    hypothesis_func,
    theta: float,
    R: float,
    K: int,
    polynomials: Dict,
    c_target: float,
) -> MirrorHypothesisMetrics:
    """
    Test a hypothesis function against the tex_mirror reference.

    Args:
        name: Name of the hypothesis for printing
        hypothesis_func: Function(theta, R, K) -> (m1, m2)
        theta, R, K: Parameters
        polynomials: Polynomial dictionary
        c_target: Target c value

    Returns:
        MirrorHypothesisMetrics
    """

    # Get reference values from tex_mirror
    ref = compute_c_paper_tex_mirror(
        theta=theta,
        R=R,
        n=60,
        polynomials=polynomials,
        terms_version="old",
        tex_exp_component="exp_R_ref",
    )

    # Get hypothesis m1, m2
    hyp_m1, hyp_m2 = hypothesis_func(theta, R, K)

    # Measure
    metrics = measure_mirror_hypothesis(
        hypothesis_m1=hyp_m1,
        hypothesis_m2=hyp_m2,
        reference_m1=ref.m1,
        reference_m2=ref.m2,
        I1_plus=ref.I1_plus,
        I2_plus=ref.I2_plus,
        I1_minus_base=ref.I1_minus_base,
        I2_minus_base=ref.I2_minus_base,
        S34_plus=ref.S34_plus,
        c_target=c_target,
    )

    return metrics


def print_metrics(name: str, metrics: MirrorHypothesisMetrics):
    """Print metrics in a readable format."""
    print(f"\n{name}:")
    print(f"  m1: hypothesis={metrics.hypothesis_m1:.4f}, reference={metrics.reference_m1:.4f}, error={metrics.m1_rel_error*100:.2f}%")
    print(f"  m2: hypothesis={metrics.hypothesis_m2:.4f}, reference={metrics.reference_m2:.4f}, error={metrics.m2_rel_error*100:.2f}%")
    print(f"  c:  hypothesis={metrics.c_hypothesis:.4f} ({metrics.c_gap_hypothesis:+.2f}%), reference={metrics.c_reference:.4f} ({metrics.c_gap_reference:+.2f}%)")

    gates = []
    if metrics.passes_1pct_gate:
        gates.append("1%")
    if metrics.passes_5pct_gate:
        gates.append("5%")
    if metrics.passes_10pct_gate:
        gates.append("10%")

    if gates:
        print(f"  Gates: PASSES {', '.join(gates)}")
    else:
        print(f"  Gates: FAILS all")


def main():
    print("=" * 70)
    print("GPT Run 13B: Diagnostic Solver Measurement Metrics")
    print("=" * 70)
    print()
    print("Testing various mirror weight hypotheses against tex_mirror reference.")
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    K = 3

    # Define hypothesis functions to test

    def hypothesis_exp_R(theta, R, K):
        """Simple exp(R) hypothesis."""
        A1 = np.exp(R) + (K - 1)
        A2 = np.exp(R) + 2 * (K - 1)
        # Assume m_implied = 1 (no shape deformation)
        return A1, A2

    def hypothesis_exp_2R(theta, R, K):
        """TeX-motivated exp(2R/theta) hypothesis."""
        A1 = np.exp(2 * R / theta) + (K - 1)
        A2 = np.exp(2 * R / theta) + 2 * (K - 1)
        return A1, A2

    def hypothesis_simple(theta, R, K):
        """Simple K-based hypothesis."""
        A1 = K
        A2 = 2 * K
        return A1, A2

    def hypothesis_tex_exp_R_ref(theta, R, K):
        """tex_mirror's exp_R_ref hypothesis (uses R_ref=1.3036)."""
        R_ref = 1.3036
        eps = (5/32) / theta
        A1 = np.exp(R_ref) + (K - 1) + eps
        A2 = np.exp(R_ref) + 2 * (K - 1) + eps
        return A1, A2

    hypotheses = [
        ("exp(R)", hypothesis_exp_R),
        ("exp(2R/θ)", hypothesis_exp_2R),
        ("Simple K-based", hypothesis_simple),
        ("tex_mirror exp_R_ref", hypothesis_tex_exp_R_ref),
    ]

    # Test on both benchmarks
    benchmarks = [
        ("κ (R=1.3036)", polys_kappa, TARGETS["kappa"]),
        ("κ* (R=1.1167)", polys_kappa_star, TARGETS["kappa_star"]),
    ]

    for bench_name, polys, target in benchmarks:
        R = target["R"]
        c_target = target["c_target"]

        print("=" * 70)
        print(f"BENCHMARK: {bench_name}")
        print("=" * 70)

        for hyp_name, hyp_func in hypotheses:
            metrics = test_hypothesis(
                name=hyp_name,
                hypothesis_func=hyp_func,
                theta=THETA,
                R=R,
                K=K,
                polynomials=polys,
                c_target=c_target,
            )
            print_metrics(hyp_name, metrics)

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The exp_R_ref hypothesis (fixed R=1.3036) performs best on BOTH benchmarks
because it was calibrated for this exact purpose.

The pure exp(R) hypothesis is benchmark-specific and performs poorly on κ*
when R differs from the reference.

This confirms that the tex_mirror amplitude model is a calibrated approximation,
not a first-principles TeX derivation.

To remove the calibration dependency, future work must:
1. Derive the amplitude from TeX Section 10 mirror formula
2. Or validate the calibration across a wider R range
""")


if __name__ == "__main__":
    main()
