"""
src/unified_s12_microcases.py
Phase 25.3: P=Q=1 Microcase Ladder

PURPOSE:
========
Compare unified and empirical evaluators with P=Q=1 (constant polynomials).
This isolates the bracket structure from polynomial interactions.

DIAGNOSTIC LOGIC:
=================
- If P=Q=1 DISAGREES: Gap is bracket structure, NOT polynomials
  → Look for missing prefactor or structural issue
- If P=Q=1 MATCHES but full disagrees: Gap is polynomial/Q mixing
  → Look for eigenvalue mapping or Q application issue

Created: 2025-12-25
Phase: 25.3
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import math


# =============================================================================
# CONSTANTS
# =============================================================================

THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass
class MicrocaseResult:
    """Result of microcase (P=Q=1) evaluation."""
    I1_11_value: float
    evaluator: str  # "unified" or "empirical"
    R: float
    theta: float
    n_quad: int


@dataclass
class MicrocaseComparison:
    """Comparison between unified and empirical P=Q=1 evaluations."""
    unified_I1_11: float
    empirical_I1_11: float
    rel_diff: float
    abs_diff: float
    R: float
    theta: float
    n_quad: int
    agrees: bool  # True if rel_diff < 1%


# =============================================================================
# EMPIRICAL MICROCASE (uses existing infrastructure)
# =============================================================================


def empirical_I1_with_P1_Q1(
    theta: float = THETA,
    R: float = KAPPA_R,
    n: int = 60,
) -> float:
    """
    Compute I1 for (1,1) pair using empirical evaluator with P=Q=1.

    Uses the existing evaluate_I1_with_P1_Q1 from src/evaluate.py.

    Args:
        theta: PRZZ theta parameter (default 4/7)
        R: PRZZ R parameter (default 1.3036)
        n: Quadrature points

    Returns:
        I1 value from empirical evaluator
    """
    from src.evaluate import evaluate_I1_with_P1_Q1
    return evaluate_I1_with_P1_Q1(theta, R, n)


# =============================================================================
# UNIFIED MICROCASE
# =============================================================================


def unified_I1_with_P1_Q1(
    theta: float = THETA,
    R: float = KAPPA_R,
    n: int = 60,
) -> float:
    """
    Compute I1 for (1,1) pair using unified evaluator with P=Q=1.

    With P=Q=1:
    - P(x+u) = 1, P(y+u) = 1
    - Q(A_alpha) = 1, Q(A_beta) = 1
    - Only exp and log factors contribute

    This tests the pure bracket structure without polynomial complications.

    Args:
        theta: PRZZ theta parameter (default 4/7)
        R: PRZZ R parameter (default 1.3036)
        n: Quadrature points

    Returns:
        I1 value from unified evaluator with constant polynomials
    """
    from src.unified_s12_evaluator_v3 import compute_I1_unified_v3
    from src.polynomials import Polynomial

    # Create constant polynomial P(x) = Q(x) = 1
    one_poly = Polynomial([1.0])
    polynomials = {"P1": one_poly, "P2": one_poly, "P3": one_poly, "Q": one_poly}

    # Compute I1 for (1,1) pair with unified evaluator
    result = compute_I1_unified_v3(
        R=R,
        theta=theta,
        ell1=1,
        ell2=1,
        polynomials=polynomials,
        n_quad_u=n,
        n_quad_t=n,
        include_Q=True,  # Q=1 so this is equivalent to include_Q=False
    )

    return result.I1_value


def unified_I2_analytic_P1_Q1(
    theta: float = THETA,
    R: float = KAPPA_R,
) -> float:
    """
    Compute I2 for (1,1) pair analytically with P=Q=1.

    I2 has no xy derivatives (it's the "scalar" integral), so with P=Q=1:
        I2 = (1/theta) * integral of exp(2Rt) over t * integral of 1 over u
           = (1/theta) * (exp(2R) - 1) / (2R) * 1
           = (1/theta) * F(R)

    where F(R) = (exp(2R) - 1) / (2R) is the scalar t-integral factor.

    Args:
        theta: PRZZ theta parameter
        R: PRZZ R parameter

    Returns:
        I2 analytic value with P=Q=1
    """
    if abs(R) < 1e-15:
        F_R = 1.0  # L'Hôpital limit
    else:
        F_R = (math.exp(2 * R) - 1) / (2 * R)

    return F_R / theta


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================


def compare_microcase_I1(
    theta: float = THETA,
    R: float = KAPPA_R,
    n: int = 60,
) -> MicrocaseComparison:
    """
    Compare I1 between unified and empirical for P=Q=1.

    Args:
        theta: PRZZ theta parameter
        R: PRZZ R parameter
        n: Quadrature points

    Returns:
        MicrocaseComparison with unified, empirical, and relative difference
    """
    unified = unified_I1_with_P1_Q1(theta, R, n)
    empirical = empirical_I1_with_P1_Q1(theta, R, n)

    abs_diff = abs(unified - empirical)
    rel_diff = abs_diff / abs(empirical) if empirical != 0 else float('inf')

    return MicrocaseComparison(
        unified_I1_11=unified,
        empirical_I1_11=empirical,
        rel_diff=rel_diff,
        abs_diff=abs_diff,
        R=R,
        theta=theta,
        n_quad=n,
        agrees=rel_diff < 0.01,  # 1% tolerance
    )


def run_microcase_ladder(
    n: int = 60,
) -> Tuple[MicrocaseComparison, MicrocaseComparison]:
    """
    Run P=Q=1 microcase on both benchmarks.

    Returns:
        (kappa_comparison, kappa_star_comparison)
    """
    kappa = compare_microcase_I1(theta=THETA, R=KAPPA_R, n=n)
    kappa_star = compare_microcase_I1(theta=THETA, R=KAPPA_STAR_R, n=n)
    return kappa, kappa_star


def print_microcase_report(kappa: MicrocaseComparison, kappa_star: MicrocaseComparison) -> None:
    """Print formatted microcase comparison report."""
    print("=" * 70)
    print("PHASE 25.3: P=Q=1 MICROCASE LADDER")
    print("=" * 70)
    print()
    print("DIAGNOSTIC LOGIC:")
    print("  - If DISAGREES: Gap is bracket structure, NOT polynomials")
    print("  - If MATCHES but full disagrees: Gap is polynomial/Q mixing")
    print()

    print("KAPPA (R=1.3036):")
    print(f"  Unified I1:    {kappa.unified_I1_11:>12.6e}")
    print(f"  Empirical I1:  {kappa.empirical_I1_11:>12.6e}")
    print(f"  Relative diff: {kappa.rel_diff*100:>12.4f}%")
    print(f"  Agreement:     {kappa.agrees}")
    print()

    print("KAPPA* (R=1.1167):")
    print(f"  Unified I1:    {kappa_star.unified_I1_11:>12.6e}")
    print(f"  Empirical I1:  {kappa_star.empirical_I1_11:>12.6e}")
    print(f"  Relative diff: {kappa_star.rel_diff*100:>12.4f}%")
    print(f"  Agreement:     {kappa_star.agrees}")
    print()

    print("DIAGNOSIS:")
    if kappa.agrees and kappa_star.agrees:
        print("  P=Q=1 MATCHES for both benchmarks")
        print("  → Full gap must come from polynomial/Q interactions")
        print("  → Investigate eigenvalue mapping and Q application")
    else:
        print("  P=Q=1 DISAGREES for at least one benchmark")
        print("  → Gap is in bracket structure, NOT polynomials")
        print("  → Look for missing prefactor or structural issue")

    print("=" * 70)
