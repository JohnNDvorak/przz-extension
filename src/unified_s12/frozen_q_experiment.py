"""
src/unified_s12/frozen_q_experiment.py
Phase 37: Frozen-Q Experiment

This module implements the "frozen-Q" experiment to isolate how Q causes deviation.

HYPOTHESIS TO TEST:
===================
The Q polynomial causes ~0.4% deviation from the Beta moment prediction.
This could be from:

1. Q-DERIVATIVE HITS: When we take d²/dxdy of the integrand, Q gets differentiated.
   - If this is the cause, frozen-Q should eliminate the deviation.

2. Q REWEIGHTING T-MEASURE: Q(t) modifies the effective integration weights.
   - If this is the cause, frozen-Q should show the SAME deviation.

EXPERIMENT:
===========
- Normal Q: Q(Arg_α(x,y,t)) × Q(Arg_β(x,y,t)) where arguments depend on x,y
- Frozen Q: Q(t) × Q(t) = Q(t)² - arguments evaluated at x=y=0

By comparing the correction factor with normal vs frozen Q, we determine
which mechanism is responsible for the deviation.

Created: 2025-12-26 (Phase 37)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math
import numpy as np

from src.quadrature import gauss_legendre_01
from src.series_bivariate import BivariateSeries, build_exp_bracket, build_log_factor
from src.unified_i1_paper import build_P_factor_paper, omega_for_ell, _extract_poly_coeffs


@dataclass
class FrozenQResult:
    """Result of frozen-Q experiment for one pair."""

    ell1: int
    ell2: int

    # I1 values
    I1_normal_Q: float     # With full Q(Arg_α)×Q(Arg_β)
    I1_frozen_Q: float     # With frozen Q(t)²
    I1_no_Q: float         # With Q=1

    # Derived
    ratio_frozen_vs_no_Q: float  # I1_frozen_Q / I1_no_Q
    ratio_normal_vs_no_Q: float  # I1_normal_Q / I1_no_Q

    @property
    def Q_derivative_effect(self) -> float:
        """Effect from Q derivatives (normal - frozen)."""
        return self.I1_normal_Q - self.I1_frozen_Q

    @property
    def Q_reweight_effect(self) -> float:
        """Effect from Q reweighting t-measure (frozen - no_Q)."""
        return self.I1_frozen_Q - self.I1_no_Q


def build_frozen_Q_factor(
    Q_coeffs: list,
    t: float,
    max_dx: int,
    max_dy: int,
) -> BivariateSeries:
    """
    Build frozen-Q factor: Q(t)² as a constant series.

    This evaluates Q at the frozen eigenvalue (x=y=0), giving Q(t).
    The product Q(Arg_α) × Q(Arg_β) at x=y=0 gives Q(t)².

    Args:
        Q_coeffs: Q polynomial coefficients [c0, c1, c2, ...]
        t: The t quadrature point
        max_dx, max_dy: Maximum degrees for series

    Returns:
        BivariateSeries with constant coefficient Q(t)²
    """
    # Evaluate Q(t)
    Q_at_t = sum(c * t**i for i, c in enumerate(Q_coeffs))

    # Q(t)² as constant series
    return BivariateSeries(
        max_dx=max_dx,
        max_dy=max_dy,
        coeffs={(0, 0): Q_at_t * Q_at_t}
    )


def compute_I1_with_Q_mode(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    q_mode: str = "normal",  # "normal", "frozen", or "none"
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I1 with configurable Q handling.

    Args:
        R, theta: PRZZ parameters
        ell1, ell2: Pair indices
        polynomials: Dict with P1, P2, P3, Q
        q_mode: "normal" (full Q), "frozen" (Q(t)²), or "none" (Q=1)
        n_quad_u, n_quad_t, n_quad_a: Quadrature points

    Returns:
        I1 value with specified Q handling
    """
    from src.series_bivariate import build_Q_factor

    max_dx = ell1
    max_dy = ell2

    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    if P_ell1 is None or P_ell2 is None:
        raise ValueError(f"Missing polynomial P{ell1} or P{ell2}")

    Q_coeffs = _extract_poly_coeffs(Q) if Q is not None else None
    one_minus_u_power = ell1 + ell2

    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power

        for t, t_w in zip(t_nodes, t_weights):
            # 1. Exp factor
            a0 = 2 * R * t
            a_xy = R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)

            # 2. Log factor
            log_factor = build_log_factor(theta, max_dx, max_dy)

            # 3. P factors (paper regime)
            P_x = build_P_factor_paper(
                P_ell1, u, "x", omega1, R, theta, max_dx, max_dy, n_quad_a
            )
            P_y = build_P_factor_paper(
                P_ell2, u, "y", omega2, R, theta, max_dx, max_dy, n_quad_a
            )

            # Build bracket
            bracket = exp_factor * log_factor * P_x * P_y

            # 4. Q factor based on mode
            if q_mode == "normal" and Q_coeffs is not None:
                # Full Q(Arg_α) × Q(Arg_β) with x,y dependence
                Q_alpha = build_Q_factor(
                    Q_coeffs,
                    a0=t,
                    ax=theta * (t - 1),
                    ay=theta * t,
                    max_dx=max_dx,
                    max_dy=max_dy,
                )
                Q_beta = build_Q_factor(
                    Q_coeffs,
                    a0=t,
                    ax=theta * t,
                    ay=theta * (t - 1),
                    max_dx=max_dx,
                    max_dy=max_dy,
                )
                bracket = bracket * Q_alpha * Q_beta

            elif q_mode == "frozen" and Q_coeffs is not None:
                # Frozen Q(t)² - no x,y dependence
                Q_frozen = build_frozen_Q_factor(Q_coeffs, t, max_dx, max_dy)
                bracket = bracket * Q_frozen

            # q_mode == "none" -> no Q multiplication

            # 5. Extract coefficient
            coeff = bracket.extract(ell1, ell2)
            total += coeff * one_minus_u_factor * u_w * t_w

    # Apply factorial normalization
    total *= math.factorial(ell1) * math.factorial(ell2)

    # Apply sign convention for off-diagonal
    if ell1 != ell2:
        sign = (-1) ** (ell1 + ell2)
        total *= sign

    return total


def run_frozen_q_experiment(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
    verbose: bool = True,
) -> Dict[str, FrozenQResult]:
    """
    Run the frozen-Q experiment for all pairs.

    Computes I1 with three Q modes:
    1. normal: Full Q(Arg_α) × Q(Arg_β)
    2. frozen: Q(t)² (x=y=0 in argument)
    3. none: Q=1

    Returns:
        Dict mapping pair_key -> FrozenQResult
    """
    pairs = ["11", "22", "33", "12", "13", "23"]
    results = {}

    if verbose:
        print("=" * 70)
        print("PHASE 37: FROZEN-Q EXPERIMENT")
        print("=" * 70)
        print()
        print(f"Parameters: θ={theta:.6f}, R={R}")
        print()
        print("Hypothesis: Q polynomial causes ~0.4% deviation from Beta moment.")
        print("- If frozen-Q eliminates deviation → Q-derivative hits are cause")
        print("- If frozen-Q shows same deviation → Q reweighting t-measure is cause")
        print()

    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        I1_normal = compute_I1_with_Q_mode(
            R, theta, ell1, ell2, polynomials, q_mode="normal", n_quad_u=n_quad
        )
        I1_frozen = compute_I1_with_Q_mode(
            R, theta, ell1, ell2, polynomials, q_mode="frozen", n_quad_u=n_quad
        )
        I1_no_Q = compute_I1_with_Q_mode(
            R, theta, ell1, ell2, polynomials, q_mode="none", n_quad_u=n_quad
        )

        ratio_frozen = I1_frozen / I1_no_Q if abs(I1_no_Q) > 1e-15 else float('inf')
        ratio_normal = I1_normal / I1_no_Q if abs(I1_no_Q) > 1e-15 else float('inf')

        results[pair_key] = FrozenQResult(
            ell1=ell1,
            ell2=ell2,
            I1_normal_Q=I1_normal,
            I1_frozen_Q=I1_frozen,
            I1_no_Q=I1_no_Q,
            ratio_frozen_vs_no_Q=ratio_frozen,
            ratio_normal_vs_no_Q=ratio_normal,
        )

    if verbose:
        print("PER-PAIR RESULTS")
        print("-" * 70)
        print(f"{'Pair':<6} | {'I1(Q=1)':<12} | {'I1(frozen)':<12} | {'I1(normal)':<12} | {'Δ deriv':<10} | {'Δ reweight':<10}")
        print("-" * 70)

        for pair_key in pairs:
            r = results[pair_key]
            deriv_pct = (r.I1_normal_Q - r.I1_frozen_Q) / abs(r.I1_no_Q) * 100 if abs(r.I1_no_Q) > 1e-15 else 0
            reweight_pct = (r.I1_frozen_Q - r.I1_no_Q) / abs(r.I1_no_Q) * 100 if abs(r.I1_no_Q) > 1e-15 else 0

            print(f"({r.ell1},{r.ell2})  | {r.I1_no_Q:+.6f}   | {r.I1_frozen_Q:+.6f}   | {r.I1_normal_Q:+.6f}   | {deriv_pct:+.4f}%   | {reweight_pct:+.4f}%")

        print()

    return results


def analyze_frozen_q_results(
    results: Dict[str, FrozenQResult],
    verbose: bool = True,
) -> Dict:
    """
    Analyze frozen-Q results to determine Q deviation mechanism.

    Returns analysis dict with:
    - total_deriv_effect: Total effect from Q derivatives
    - total_reweight_effect: Total effect from Q t-reweighting
    - dominant_mechanism: "derivative" or "reweight"
    """
    # Factorial normalization matching production
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    # Aggregate S12 values for each Q mode
    S12_no_Q = 0.0
    S12_frozen = 0.0
    S12_normal = 0.0

    for pair_key, r in results.items():
        norm = f_norm[pair_key] * symmetry[pair_key]
        S12_no_Q += r.I1_no_Q * norm
        S12_frozen += r.I1_frozen_Q * norm
        S12_normal += r.I1_normal_Q * norm

    # Compute effects as percentage of S12_no_Q
    deriv_effect_pct = (S12_normal - S12_frozen) / abs(S12_no_Q) * 100
    reweight_effect_pct = (S12_frozen - S12_no_Q) / abs(S12_no_Q) * 100
    total_Q_effect_pct = (S12_normal - S12_no_Q) / abs(S12_no_Q) * 100

    dominant = "derivative" if abs(deriv_effect_pct) > abs(reweight_effect_pct) else "reweight"

    if verbose:
        print("AGGREGATE ANALYSIS")
        print("-" * 70)
        print(f"  S12 (Q=1):     {S12_no_Q:+.8f}")
        print(f"  S12 (frozen):  {S12_frozen:+.8f}")
        print(f"  S12 (normal):  {S12_normal:+.8f}")
        print()
        print(f"  Q derivative effect:   {deriv_effect_pct:+.4f}% of S12")
        print(f"  Q reweight effect:     {reweight_effect_pct:+.4f}% of S12")
        print(f"  Total Q effect:        {total_Q_effect_pct:+.4f}% of S12")
        print()
        print(f"  DOMINANT MECHANISM: {dominant.upper()}")
        print()

        if abs(deriv_effect_pct) > abs(reweight_effect_pct):
            print("  Interpretation: The Q deviation is primarily from Q being differentiated")
            print("    → When we take d²/dxdy, Q(Arg(x,y,t)) gets chain-rule derivatives")
            print("    → These Q' and Q'' terms create the deviation from Beta moment")
        else:
            print("  Interpretation: The Q deviation is primarily from Q reweighting")
            print("    → Q(t)² changes the effective t-integration measure")
            print("    → This happens even before derivatives are extracted")

    return {
        "S12_no_Q": S12_no_Q,
        "S12_frozen": S12_frozen,
        "S12_normal": S12_normal,
        "deriv_effect_pct": deriv_effect_pct,
        "reweight_effect_pct": reweight_effect_pct,
        "total_Q_effect_pct": total_Q_effect_pct,
        "dominant_mechanism": dominant,
    }


if __name__ == "__main__":
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    theta = 4 / 7
    R = 1.3036

    results = run_frozen_q_experiment(R, theta, polynomials, n_quad=60)
    analysis = analyze_frozen_q_results(results)
