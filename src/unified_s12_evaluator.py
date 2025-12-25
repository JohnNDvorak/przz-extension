"""
src/unified_s12_evaluator.py
Phase 21B: Unified S12 Evaluator via Difference Quotient

PURPOSE:
========
This module computes S12 using the correct difference quotient structure where
the combined [direct - exp(2R)*mirror] term naturally produces D = 0.

KEY INSIGHT:
============
The difference quotient identity gives:
    [direct - exp(2R)*mirror] / s = RHS_integral

For the PRZZ micro-case, we have the symmetry:
    I1_plus = exp(2R) * I1_minus

Which means:
    [I1_plus - exp(2R)*I1_minus] = 0

This is the D = 0 condition! The unified bracket should compute this naturally.

IMPLEMENTATION:
===============
1. Compute I1(+R) using the standard approach
2. Compute I1(-R) using the standard approach (with R -> -R)
3. Combined S12_plus = I1_plus - exp(2R)*I1_minus (should be ~0)
4. S12_minus = I1_minus (baseline)
5. D = S12_plus + I34 (should be ~0)

REFERENCES:
===========
- src/difference_quotient.py: Core identity implementation
- src/abd_diagnostics.py: ABD decomposition
- docs/PLAN_PHASE_21B_UNIFIED_S12.md: Implementation plan
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from src.series import TruncatedSeries
from src.quadrature import gauss_legendre_01
from src.composition import compose_exp_on_affine, compose_polynomial_on_affine
from src.abd_diagnostics import ABDDecomposition, compute_abd_decomposition


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass
class UnifiedS12ResultV2:
    """Result of unified S12 evaluation using difference quotient structure."""

    # The combined bracket value: [I1_plus - exp(2R)*I1_minus]
    # This should be ~0 in the unified structure
    S12_combined: float

    # Individual components (for diagnostics)
    I1_plus: float   # I1 evaluated at +R
    I1_minus: float  # I1 evaluated at -R

    # The multiplier that makes them related: I1_plus / I1_minus
    ratio: float

    # Expected ratio for perfect D=0: exp(2R)
    expected_ratio: float

    # ABD decomposition
    abd: ABDDecomposition

    # Metadata
    R: float
    theta: float
    benchmark: str


# =============================================================================
# CORE EVALUATOR
# =============================================================================


def compute_I1_at_R(
    R_eval: float,
    theta: float,
    n_quad: int = 40,
    include_log_factor: bool = True,
    include_alg_prefactor: bool = True,
) -> float:
    """
    Compute I1 for (1,1) pair at a given R value.

    This evaluates the bracket integral at the specified R:
        ∫₀¹ [exp(2*R_eval*t + R_eval*θ(2t-1)(x+y)) × factors]_xy dt

    Args:
        R_eval: The R value to evaluate at (+R or -R)
        theta: PRZZ θ parameter
        n_quad: Number of quadrature points
        include_log_factor: Include (1 + θ(x+y)) factor
        include_alg_prefactor: Include (1/θ + x + y) factor

    Returns:
        The xy coefficient of the integrated bracket
    """
    var_names = ("x", "y")
    xy_mask = (1 << 0) | (1 << 1)

    t_nodes, t_weights = gauss_legendre_01(n_quad)

    total = 0.0
    for t, w in zip(t_nodes, t_weights):
        # Build exp series: exp(2*R_eval*t + R_eval*θ(2t-1)(x+y))
        u0 = 2 * R_eval * t
        lin_coeff = R_eval * theta * (2 * t - 1)
        lin = {var_names[0]: lin_coeff, var_names[1]: lin_coeff}

        exp_series = compose_exp_on_affine(1.0, u0, lin, var_names)

        # Log factor: (1 + θ(x+y))
        if include_log_factor:
            log_series = TruncatedSeries.from_scalar(1.0, var_names)
            log_series = log_series + TruncatedSeries.variable("x", var_names) * theta
            log_series = log_series + TruncatedSeries.variable("y", var_names) * theta
            exp_series = exp_series * log_series

        # Algebraic prefactor: (1/θ + x + y)
        if include_alg_prefactor:
            alg_series = TruncatedSeries.from_scalar(1.0 / theta, var_names)
            alg_series = alg_series + TruncatedSeries.variable("x", var_names)
            alg_series = alg_series + TruncatedSeries.variable("y", var_names)
            exp_series = exp_series * alg_series

        # Extract xy coefficient
        xy_coeff = exp_series.coeffs.get(xy_mask, 0.0)
        if isinstance(xy_coeff, np.ndarray):
            xy_coeff = float(xy_coeff)

        total += xy_coeff * w

    return total


def compute_S12_unified_v2(
    R: float,
    theta: float = 4.0 / 7.0,
    n_quad: int = 40,
    I34_plus: float = 0.0,  # In micro-case, I34 = 0
    benchmark: str = "unified_v2",
) -> UnifiedS12ResultV2:
    """
    Compute S12 using the unified difference quotient structure.

    The key insight: I1_plus = exp(2R) * I1_minus (symmetry property).
    Therefore: S12_combined = I1_plus - exp(2R)*I1_minus = 0

    This gives D = S12_combined + I34 = 0 + 0 = 0 in micro-case.

    Args:
        R: PRZZ R parameter
        theta: PRZZ θ parameter
        n_quad: Number of quadrature points
        I34_plus: I34(+R) value (0 in micro-case)
        benchmark: Benchmark name for diagnostics

    Returns:
        UnifiedS12ResultV2 with combined S12 and ABD decomposition
    """
    # Compute I1 at +R and -R
    I1_plus = compute_I1_at_R(R_eval=+R, theta=theta, n_quad=n_quad)
    I1_minus = compute_I1_at_R(R_eval=-R, theta=theta, n_quad=n_quad)

    # The combined bracket: [I1_plus - exp(2R)*I1_minus]
    # This should be ~0 due to the symmetry I1_plus = exp(2R)*I1_minus
    S12_combined = I1_plus - np.exp(2 * R) * I1_minus

    # Compute ratio
    ratio = I1_plus / I1_minus if abs(I1_minus) > 1e-15 else float('inf')
    expected_ratio = np.exp(2 * R)

    # ABD decomposition:
    # In the unified structure, S12_combined plays the role of I12_plus
    # and I1_minus plays the role of I12_minus (the exp(R) coefficient)
    abd = compute_abd_decomposition(
        I12_plus=S12_combined,  # Should be ~0
        I12_minus=I1_minus,     # The baseline
        I34_plus=I34_plus,      # 0 in micro-case
        R=R,
        benchmark=benchmark,
    )

    return UnifiedS12ResultV2(
        S12_combined=S12_combined,
        I1_plus=I1_plus,
        I1_minus=I1_minus,
        ratio=ratio,
        expected_ratio=expected_ratio,
        abd=abd,
        R=R,
        theta=theta,
        benchmark=benchmark,
    )


# =============================================================================
# COMPARISON HELPERS
# =============================================================================


def verify_symmetry(R: float, theta: float = 4.0 / 7.0, n_quad: int = 40) -> Dict:
    """
    Verify the symmetry I1_plus = exp(2R) * I1_minus.

    This symmetry is the mathematical basis for D = 0.

    Returns:
        Dictionary with verification results
    """
    I1_plus = compute_I1_at_R(R_eval=+R, theta=theta, n_quad=n_quad)
    I1_minus = compute_I1_at_R(R_eval=-R, theta=theta, n_quad=n_quad)

    ratio = I1_plus / I1_minus if abs(I1_minus) > 1e-15 else float('inf')
    expected = np.exp(2 * R)
    rel_error = abs(ratio - expected) / expected if expected != 0 else float('inf')

    # The combined term should be ~0
    combined = I1_plus - expected * I1_minus

    return {
        "I1_plus": I1_plus,
        "I1_minus": I1_minus,
        "ratio": ratio,
        "expected_ratio": expected,
        "ratio_rel_error": rel_error,
        "combined": combined,
        "combined_rel_to_plus": abs(combined) / abs(I1_plus) if I1_plus != 0 else 0,
        "symmetry_holds": rel_error < 1e-10,
    }


def run_dual_benchmark_v2(n_quad: int = 40) -> Tuple[UnifiedS12ResultV2, UnifiedS12ResultV2]:
    """
    Run unified S12 computation on both benchmarks.

    Returns:
        (kappa_result, kappa_star_result)
    """
    kappa = compute_S12_unified_v2(R=1.3036, n_quad=n_quad, benchmark="kappa_v2")
    kappa_star = compute_S12_unified_v2(R=1.1167, n_quad=n_quad, benchmark="kappa_star_v2")
    return kappa, kappa_star


# =============================================================================
# MAIN EXECUTION
# =============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED S12 EVALUATOR V2 - DIFFERENCE QUOTIENT STRUCTURE")
    print("=" * 70)

    for benchmark, R in [("kappa", 1.3036), ("kappa_star", 1.1167)]:
        print(f"\n{'='*70}")
        print(f"Benchmark: {benchmark.upper()} (R={R})")
        print("=" * 70)

        # Verify symmetry
        sym = verify_symmetry(R=R)
        print("\n--- Symmetry Verification ---")
        print(f"I1_plus:         {sym['I1_plus']:.10f}")
        print(f"I1_minus:        {sym['I1_minus']:.10f}")
        print(f"Ratio:           {sym['ratio']:.10f}")
        print(f"Expected (e^2R): {sym['expected_ratio']:.10f}")
        print(f"Rel error:       {sym['ratio_rel_error']:.2e}")
        print(f"Combined:        {sym['combined']:.2e}")
        print(f"Symmetry holds:  {sym['symmetry_holds']}")

        # Compute unified S12
        result = compute_S12_unified_v2(R=R, benchmark=benchmark)

        print("\n--- Unified S12 Result ---")
        print(f"S12_combined:    {result.S12_combined:.10f}")
        print(f"D:               {result.abd.D:.10f}")
        print(f"B/A:             {result.abd.B_over_A:.6f}")

        # Check gates
        D_ok = abs(result.abd.D) < 1e-6
        BA_ok = abs(result.abd.B_over_A - 5.0) < 1e-6

        print("\n--- Gate Check ---")
        print(f"D ≈ 0:           {'PASS' if D_ok else 'FAIL'} (D = {result.abd.D:.2e})")
        print(f"B/A ≈ 5:         {'PASS' if BA_ok else 'FAIL'} (B/A = {result.abd.B_over_A:.6f})")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: The symmetry I1_plus = exp(2R)*I1_minus")
    print("automatically gives S12_combined = 0, hence D = 0, B/A = 5.")
    print("=" * 70)
