#!/usr/bin/env python3
"""
GPT Run 13A: Direct Mirror-Integrand Experiment

This script implements the PRZZ TeX mirror formula DIRECTLY (lines 1530-1532)
and compares it to the current tex_mirror shape×amplitude model.

The TeX formula shows the +R and -R terms are COMBINED into a single integral,
not evaluated separately. This is fundamentally different from our current
tex_mirror approach of `I(+R) + m × I(-R)`.

TeX Formula (I₁ for (1,1), lines 1530-1532):
-----------------------------------------
I₁ = T Φ̂(0) d²/dxdy [(θ(x+y)+1)/θ] ∫∫ (1-u)² P₁(x+u) P₂(y+u)
     × exp(R[θt(x+y) - θy + t]) × exp(R[θt(x+y) - θx + t])
     × Q(θt(x+y) - θy + t) × Q(θt(x+y) - θx + t) |_{x=y=0} du dt

At x=y=0, this simplifies considerably.

Usage:
    python run_gpt_run13a_direct_mirror.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
    Polynomial,
)
from src.quadrature import gauss_legendre_01
from src.evaluate import compute_c_paper_tex_mirror


THETA = 4.0 / 7.0

# PRZZ Targets
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


def eval_poly(P, u: float) -> float:
    """Evaluate polynomial at point u using its eval method."""
    # All polynomial classes have an eval(x) method
    return float(P.eval(np.array([u]))[0])


def eval_poly_derivative(P, u: float) -> float:
    """Evaluate polynomial first derivative at point u."""
    # Convert to monomial for derivative evaluation
    if hasattr(P, 'to_monomial'):
        mono = P.to_monomial()
    else:
        mono = P
    return float(mono.eval_deriv(np.array([u]), 1)[0])


def compute_I1_direct_11(
    theta: float,
    R: float,
    P1: Polynomial,
    P2: Polynomial,
    Q: Polynomial,
    n_quad: int = 60,
) -> float:
    """
    Compute I₁ for (1,1) pair using DIRECT TeX formula.

    From TeX lines 1530-1532:
    I₁ = T Φ̂(0) d²/dxdy [(θ(x+y)+1)/θ] ∫∫ (1-u)² P₁(x+u) P₂(y+u)
         × exp(R[θt(x+y) - θy + t]) × exp(R[θt(x+y) - θx + t])
         × Q(θt(x+y) - θy + t) × Q(θt(x+y) - θx + t) |_{x=y=0} du dt

    At x=y=0, the Q arguments simplify:
    - Q(θ×0×t - θ×0 + t) = Q(t)
    - exp(R[t]) × exp(R[t]) = exp(2Rt)

    BUT we need d²/dxdy which brings in derivatives of P and prefactor.

    Using Taylor expansion and derivative extraction:
    - P₁(x+u) = P₁(u) + x P₁'(u) + O(x²)
    - P₂(y+u) = P₂(u) + y P₂'(u) + O(y²)
    - d²/dxdy[P₁(x+u)P₂(y+u)] = P₁'(u) P₂'(u)

    The prefactor (θ(x+y)+1)/θ adds:
    - d/dx[(θ(x+y)+1)/θ] = 1
    - d²/dxdy[(θ(x+y)+1)/θ × f(x,y)] = 1 × d/dy[f] + (θ(x+y)+1)/θ × d²/dxdy[f]

    At x=y=0: = d/dy[f]|_{0,0} + 1/θ × d²/dxdy[f]|_{0,0}

    This is getting complex. Let me use finite differences for verification.
    """
    nodes, weights = gauss_legendre_01(n_quad)

    # We need to compute d²/dxdy of the integrand at x=y=0
    # Using finite differences for this verification experiment

    eps = 1e-6

    def integrand(x: float, y: float, u: float, t: float) -> float:
        """Full integrand including all factors."""
        # (1-u)^2 factor
        omu_sq = (1 - u) ** 2

        # P factors (shifted)
        P1_val = eval_poly(P1, x + u)
        P2_val = eval_poly(P2, y + u)

        # Exponential arguments
        # θt(x+y) - θy + t and θt(x+y) - θx + t
        exp_arg1 = R * (theta * t * (x + y) - theta * y + t)
        exp_arg2 = R * (theta * t * (x + y) - theta * x + t)
        exp_factor = np.exp(exp_arg1 + exp_arg2)

        # Q arguments
        q_arg1 = theta * t * (x + y) - theta * y + t
        q_arg2 = theta * t * (x + y) - theta * x + t
        Q1_val = eval_poly(Q, q_arg1)
        Q2_val = eval_poly(Q, q_arg2)

        # Algebraic prefactor
        prefactor = (theta * (x + y) + 1) / theta

        return prefactor * omu_sq * P1_val * P2_val * exp_factor * Q1_val * Q2_val

    def integral_at(x: float, y: float) -> float:
        """Compute the double integral at given x, y."""
        result = 0.0
        for i, (u, wu) in enumerate(zip(nodes, weights)):
            for j, (t, wt) in enumerate(zip(nodes, weights)):
                result += wu * wt * integrand(x, y, u, t)
        return result

    # Compute d²/dxdy using central finite differences
    # d²f/dxdy ≈ (f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)) / (4h²)
    f_pp = integral_at(eps, eps)
    f_pm = integral_at(eps, -eps)
    f_mp = integral_at(-eps, eps)
    f_mm = integral_at(-eps, -eps)

    d2_dxdy = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)

    return d2_dxdy


def compute_I2_direct_11(
    theta: float,
    R: float,
    P1: Polynomial,
    P2: Polynomial,
    Q: Polynomial,
    n_quad: int = 60,
) -> float:
    """
    Compute I₂ for (1,1) pair using DIRECT TeX formula.

    From TeX line 1548:
    I₂ = T Φ̂(0)/θ ∫∫ Q(t)² exp(2Rt) P₁(u) P₂(u) dt du

    This has no derivatives - straightforward double integral.
    (Dropping the T Φ̂(0) normalization for comparison.)
    """
    nodes, weights = gauss_legendre_01(n_quad)

    result = 0.0
    for i, (u, wu) in enumerate(zip(nodes, weights)):
        for j, (t, wt) in enumerate(zip(nodes, weights)):
            P1_val = eval_poly(P1, u)
            P2_val = eval_poly(P2, u)
            Q_val = eval_poly(Q, t)
            exp_factor = np.exp(2 * R * t)

            integrand = (1.0 / theta) * (Q_val ** 2) * exp_factor * P1_val * P2_val
            result += wu * wt * integrand

    return result


def compute_I3_direct_11(
    theta: float,
    R: float,
    P1: Polynomial,
    P2: Polynomial,
    Q: Polynomial,
    n_quad: int = 60,
) -> float:
    """
    Compute I₃ for (1,1) pair using DIRECT TeX formula.

    From TeX lines 1562-1563:
    I₃ = -T Φ̂(0) (1+θx)/θ d/dx ∫∫ (1-u) P₁(x+u) P₂(u)
         × exp(R[t+θxt]) × exp(R[-θx + t + θxt])
         × Q(t+θxt) × Q(-θx + t + θxt) dt du |_{x=0}
    """
    nodes, weights = gauss_legendre_01(n_quad)

    eps = 1e-6

    def integrand(x: float, u: float, t: float) -> float:
        """Full I₃ integrand."""
        omu = 1 - u

        P1_val = eval_poly(P1, x + u)
        P2_val = eval_poly(P2, u)

        # Exponential arguments
        exp_arg1 = R * (t + theta * x * t)
        exp_arg2 = R * (-theta * x + t + theta * x * t)
        exp_factor = np.exp(exp_arg1 + exp_arg2)

        # Q arguments
        q_arg1 = t + theta * x * t
        q_arg2 = -theta * x + t + theta * x * t
        Q1_val = eval_poly(Q, q_arg1)
        Q2_val = eval_poly(Q, q_arg2)

        # Algebraic prefactor
        prefactor = (1 + theta * x) / theta

        return -prefactor * omu * P1_val * P2_val * exp_factor * Q1_val * Q2_val

    def integral_at(x: float) -> float:
        result = 0.0
        for i, (u, wu) in enumerate(zip(nodes, weights)):
            for j, (t, wt) in enumerate(zip(nodes, weights)):
                result += wu * wt * integrand(x, u, t)
        return result

    # d/dx using central finite difference
    df_dx = (integral_at(eps) - integral_at(-eps)) / (2 * eps)

    return df_dx


def compute_I4_direct_11(
    theta: float,
    R: float,
    P1: Polynomial,
    P2: Polynomial,
    Q: Polynomial,
    n_quad: int = 60,
) -> float:
    """
    Compute I₄ for (1,1) pair using DIRECT TeX formula.

    From TeX lines 1568-1569:
    I₄ = -T Φ̂(0) (1+θy)/θ d/dy ∫∫ (1-u) P₁(u) P₂(y+u)
         × exp(R[t+θyt]) × exp(R[-θy + t + θyt])
         × Q(t+θyt) × Q(-θy + t + θyt) dt du |_{y=0}
    """
    nodes, weights = gauss_legendre_01(n_quad)

    eps = 1e-6

    def integrand(y: float, u: float, t: float) -> float:
        """Full I₄ integrand."""
        omu = 1 - u

        P1_val = eval_poly(P1, u)
        P2_val = eval_poly(P2, y + u)

        # Exponential arguments
        exp_arg1 = R * (t + theta * y * t)
        exp_arg2 = R * (-theta * y + t + theta * y * t)
        exp_factor = np.exp(exp_arg1 + exp_arg2)

        # Q arguments
        q_arg1 = t + theta * y * t
        q_arg2 = -theta * y + t + theta * y * t
        Q1_val = eval_poly(Q, q_arg1)
        Q2_val = eval_poly(Q, q_arg2)

        # Algebraic prefactor
        prefactor = (1 + theta * y) / theta

        return -prefactor * omu * P1_val * P2_val * exp_factor * Q1_val * Q2_val

    def integral_at(y: float) -> float:
        result = 0.0
        for i, (u, wu) in enumerate(zip(nodes, weights)):
            for j, (t, wt) in enumerate(zip(nodes, weights)):
                result += wu * wt * integrand(y, u, t)
        return result

    # d/dy using central finite difference
    df_dy = (integral_at(eps) - integral_at(-eps)) / (2 * eps)

    return df_dy


def main():
    print("=" * 70)
    print("GPT Run 13A: Direct Mirror-Integrand Experiment")
    print("=" * 70)
    print()
    print("This compares DIRECT TeX formula (lines 1530-1532) with tex_mirror model.")
    print("The TeX combines +R/-R into a single integral; tex_mirror separates them.")
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    benchmarks = [
        ("κ", polys_kappa, TARGETS["kappa"]),
        ("κ*", polys_kappa_star, TARGETS["kappa_star"]),
    ]

    # =========================================================================
    # SECTION 1: PAIR-LEVEL ANALYSIS (Direct I-terms for individual pairs)
    # =========================================================================
    print("=" * 70)
    print("SECTION 1: PAIR-LEVEL ANALYSIS")
    print("=" * 70)
    print()
    print("Computing DIRECT TeX I-terms for individual pairs.")
    print("These values represent ONE pair's contribution, NOT the full assembly.")
    print()
    print("NOTE: tex_mirror does not expose per-pair breakdowns in its public")
    print("interface. For a true pair-level comparison, we would need to:")
    print("  1. Extract per-pair values from tex_mirror's internal computation")
    print("  2. Compare direct (ℓ₁,ℓ₂) against tex_mirror's (ℓ₁,ℓ₂) contribution")
    print()

    for bench_name, polys, target in benchmarks:
        R = target["R"]

        print(f"\nBenchmark: {bench_name} (R={R})")
        print("-" * 50)

        # Compute direct I-terms for (1,1)
        P1 = polys["P1"]
        Q = polys["Q"]

        I1_direct = compute_I1_direct_11(THETA, R, P1, P1, Q, n_quad=60)
        I2_direct = compute_I2_direct_11(THETA, R, P1, P1, Q, n_quad=60)
        I3_direct = compute_I3_direct_11(THETA, R, P1, P1, Q, n_quad=60)
        I4_direct = compute_I4_direct_11(THETA, R, P1, P1, Q, n_quad=60)
        sum_direct = I1_direct + I2_direct + I3_direct + I4_direct

        print(f"(1,1) Pair - Direct TeX formulas:")
        print(f"  I₁ = {I1_direct:+.6f}")
        print(f"  I₂ = {I2_direct:+.6f}")
        print(f"  I₃ = {I3_direct:+.6f}")
        print(f"  I₄ = {I4_direct:+.6f}")
        print(f"  Sum(1,1) = {sum_direct:+.6f}")

    # =========================================================================
    # SECTION 2: FULL-ASSEMBLY ANALYSIS (tex_mirror totals)
    # =========================================================================
    print()
    print("=" * 70)
    print("SECTION 2: FULL-ASSEMBLY ANALYSIS")
    print("=" * 70)
    print()
    print("tex_mirror assembles ALL 6 pairs: (1,1), (2,2), (3,3), (1,2), (1,3), (2,3)")
    print("These values are TOTALS, not single-pair contributions.")
    print()

    for bench_name, polys, target in benchmarks:
        R = target["R"]
        c_target = target["c_target"]

        print(f"\nBenchmark: {bench_name} (R={R})")
        print("-" * 50)

        # Get tex_mirror result
        tex_result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R,
            n=60,
            polynomials=polys,
            terms_version="old",
            tex_exp_component="exp_R_ref",
        )

        print(f"tex_mirror model (ALL pairs combined):")
        print(f"  I1_plus (total)  = {tex_result.I1_plus:+.6f}")
        print(f"  I2_plus (total)  = {tex_result.I2_plus:+.6f}")
        print(f"  S34_plus (total) = {tex_result.S34_plus:+.6f}")
        print(f"  m1, m2           = {tex_result.m1:.4f}, {tex_result.m2:.4f}")
        print(f"  c (assembled)    = {tex_result.c:.6f}")
        print(f"  c_target         = {c_target:.6f}")
        print(f"  c gap            = {100 * (tex_result.c - c_target) / c_target:+.2f}%")

    # =========================================================================
    # SECTION 3: COMPARISON INTERPRETATION
    # =========================================================================
    print()
    print("=" * 70)
    print("SECTION 3: COMPARISON INTERPRETATION")
    print("=" * 70)
    print("""
CRITICAL: Section 1 and Section 2 values are NOT directly comparable!

  Section 1: Direct I-terms for (1,1) pair ONLY
  Section 2: tex_mirror totals across ALL 6 pairs

Why they differ:
  - Direct (1,1) is ONE pair; tex_mirror sums 6 pairs
  - tex_mirror uses shape×amplitude factorization
  - tex_mirror separates +R/-R; TeX combines them

What a proper comparison would need:
  1. Extract tex_mirror's per-pair breakdown (not currently exposed)
  2. Compute direct I-terms for ALL 6 pairs with correct (1-u) powers
  3. Sum direct values across pairs
  4. Compare: Direct total vs tex_mirror total

The numbers above should be interpreted as:
  - Section 1: Shows WHAT direct TeX computation produces for one pair
  - Section 2: Shows WHAT tex_mirror produces for full assembly
  - NOT as a direct A=B comparison!
""")

    # =========================================================================
    # SECTION 4: (2,2) Pair for reference (incomplete - wrong power)
    # =========================================================================
    print()
    print("=" * 70)
    print("SECTION 4: (2,2) Pair Reference (INCOMPLETE)")
    print("=" * 70)
    print()
    print("WARNING: The functions below use (1-u)^2 for all pairs.")
    print("For (2,2), the correct power is (1-u)^4 (OLD) or (1-u)^2 (V2).")
    print("These values are for ILLUSTRATION only, not accurate (2,2) values.")
    print()

    for bench_name, polys, target in benchmarks:
        R = target["R"]

        print(f"\nBenchmark: {bench_name} (R={R})")
        print("-" * 50)

        # For (2,2), use P2 twice
        P2_poly = polys["P2"]
        Q = polys["Q"]

        I1_direct = compute_I1_direct_11(THETA, R, P2_poly, P2_poly, Q, n_quad=60)
        I2_direct = compute_I2_direct_11(THETA, R, P2_poly, P2_poly, Q, n_quad=60)
        I3_direct = compute_I3_direct_11(THETA, R, P2_poly, P2_poly, Q, n_quad=60)
        I4_direct = compute_I4_direct_11(THETA, R, P2_poly, P2_poly, Q, n_quad=60)

        print(f"(2,2) Pair - Direct (WRONG POWER - using (1-u)² instead of (1-u)⁴):")
        print(f"  I₁ = {I1_direct:+.6f}")
        print(f"  I₂ = {I2_direct:+.6f}")
        print(f"  I₃ = {I3_direct:+.6f}")
        print(f"  I₄ = {I4_direct:+.6f}")
        print(f"  Sum = {I1_direct + I2_direct + I3_direct + I4_direct:+.6f}")

    # =========================================================================
    # SECTION 5: STRUCTURAL ANALYSIS
    # =========================================================================
    print()
    print("=" * 70)
    print("SECTION 5: STRUCTURAL ANALYSIS")
    print("=" * 70)
    print("""
KEY OBSERVATIONS:

1. The TeX formula (lines 1530-1532) combines +R and -R branches into a
   SINGLE integral with Q arguments that depend on (x,y) before setting x=y=0.

2. At x=y=0, the Q arguments simplify to just Q(t), but the DERIVATIVE
   d²/dxdy brings in P'(u) terms and modifies the structure.

3. The current tex_mirror model approximates this as:
   c = I₁(+R) + m₁×I₁(-R) + I₂(+R) + m₂×I₂(-R) + S₃₄(+R)

   where +R and -R are evaluated SEPARATELY with fitted multipliers.

4. The direct TeX formula does NOT separate +R and -R this way.
   The mirror structure is built into the combined formula.

IMPLICATION:

The tex_mirror model's shape×amplitude factorization is an APPROXIMATION
of the true TeX structure. The multipliers m₁, m₂ are calibrated to
reproduce the TeX's combined +R/-R behavior, but they are not derived
from first principles.

This explains why:
- The tex_mirror model achieves <1% accuracy (good approximation)
- But requires calibration (exp_R_ref mode)
- And breaks when V2 terms are used (different (1-u) powers change the
  integrand structure in ways the calibration doesn't account for)

NEXT STEPS:

1. To fully implement TeX-derived mirror, would need to:
   - Compute the COMBINED integral (not separate +R/-R)
   - Handle the derivative operator correctly
   - Avoid the shape×amplitude factorization

2. Alternative: Keep tex_mirror as a validated approximation, but
   understand its limitations are structural, not just calibration.
""")


if __name__ == "__main__":
    main()
