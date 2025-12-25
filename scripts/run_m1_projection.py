#!/usr/bin/env python3
"""
scripts/run_m1_projection.py
Phase 13.2: Derive m₁ as a projection coefficient.

GOAL: Understand where m₁ = exp(R) + 5 comes from.

APPROACH:
=========
The empirical formula that works is:
    c = S12(+R) + m₁ × S12(-R) + S34(+R)

where m₁ = exp(R) + 5 for K=3.

We can solve for m₁ algebraically:
    m₁_needed = (c_target - S12(+R) - S34(+R)) / S12(-R)

This tells us what m₁ value is REQUIRED to hit c_target.

FINDINGS FROM PHASE 13:
=======================
1. The operator-derived mirror gives S12_op_mirror ≈ S12(+R), NOT m₁ × S12(-R)
2. So the operator approach gives c_op = S12(+R) + S12(+R) + S34 = 2×S12(+R) + S34
3. This doesn't match c_target

The question is: what is the mathematical origin of the −R evaluation?
"""

import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import compute_c_paper_with_mirror
from src.mirror_transform_harness import MirrorTransformHarness


def compute_components(theta: float, R: float, n: int, polynomials: dict):
    """Compute S12(+R), S12(-R), and S34(+R) components."""
    harness = MirrorTransformHarness(
        theta, R, n, polynomials,
        use_t_dependent=True,
        use_t_flip_exp=True
    )
    result = harness.run(verbose=False)

    return {
        'S12_plus_R': result.S12_direct_total,
        'S12_minus_R': result.S12_basis_total,
        'S12_operator_mirror': result.S12_operator_mirror_total,
        'S34': result.S34_total,
        'c_with_operator': result.c_with_operator,
        'c_with_empirical': result.c_with_empirical,
        'm1_implied': result.m1_implied,
    }


def compute_m1_needed(S12_plus_R: float, S12_minus_R: float, S34: float, c_target: float) -> float:
    """
    Compute what m₁ is NEEDED to produce c_target.

    From: c = S12(+R) + m₁ × S12(-R) + S34(+R)
    Solve: m₁ = (c_target - S12(+R) - S34) / S12(-R)
    """
    if abs(S12_minus_R) < 1e-15:
        return float('inf')
    return (c_target - S12_plus_R - S34) / S12_minus_R


def main():
    print("=" * 70)
    print("PHASE 13.2: m₁ PROJECTION COEFFICIENT DERIVATION")
    print("=" * 70)

    theta = 4.0 / 7.0
    n = 40

    # κ benchmark (R=1.3036)
    print("\n" + "=" * 50)
    print("κ BENCHMARK (R=1.3036)")
    print("=" * 50)

    R_kappa = 1.3036
    c_target_kappa = 2.137
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_kappa = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    comp_kappa = compute_components(theta, R_kappa, n, polynomials_kappa)

    m1_needed_kappa = compute_m1_needed(
        comp_kappa['S12_plus_R'],
        comp_kappa['S12_minus_R'],
        comp_kappa['S34'],
        c_target_kappa
    )
    m1_empirical_kappa = np.exp(R_kappa) + 5

    print(f"\nComponents:")
    print(f"  S12(+R):              {comp_kappa['S12_plus_R']:.6f}")
    print(f"  S12(-R):              {comp_kappa['S12_minus_R']:.6f}")
    print(f"  S12_operator_mirror:  {comp_kappa['S12_operator_mirror']:.6f}")
    print(f"  S34(+R):              {comp_kappa['S34']:.6f}")

    print(f"\nc values:")
    print(f"  c_target (PRZZ):      {c_target_kappa:.6f}")
    print(f"  c_with_empirical:     {comp_kappa['c_with_empirical']:.6f}")
    print(f"  c_with_operator:      {comp_kappa['c_with_operator']:.6f}")

    print(f"\nm₁ analysis:")
    print(f"  m1_needed to hit c:   {m1_needed_kappa:.4f}")
    print(f"  m1_empirical (exp+5): {m1_empirical_kappa:.4f}")
    print(f"  m1_operator_implied:  {comp_kappa['m1_implied']:.4f}")
    print(f"  Ratio needed/empirical: {m1_needed_kappa / m1_empirical_kappa:.4f}")

    # κ* benchmark (R=1.1167)
    print("\n" + "=" * 50)
    print("κ* BENCHMARK (R=1.1167)")
    print("=" * 50)

    R_kappa_star = 1.1167
    c_target_kappa_star = 1.94
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    polynomials_kappa_star = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    comp_kappa_star = compute_components(theta, R_kappa_star, n, polynomials_kappa_star)

    m1_needed_kappa_star = compute_m1_needed(
        comp_kappa_star['S12_plus_R'],
        comp_kappa_star['S12_minus_R'],
        comp_kappa_star['S34'],
        c_target_kappa_star
    )
    m1_empirical_kappa_star = np.exp(R_kappa_star) + 5

    print(f"\nComponents:")
    print(f"  S12(+R):              {comp_kappa_star['S12_plus_R']:.6f}")
    print(f"  S12(-R):              {comp_kappa_star['S12_minus_R']:.6f}")
    print(f"  S12_operator_mirror:  {comp_kappa_star['S12_operator_mirror']:.6f}")
    print(f"  S34(+R):              {comp_kappa_star['S34']:.6f}")

    print(f"\nc values:")
    print(f"  c_target (PRZZ):      {c_target_kappa_star:.6f}")
    print(f"  c_with_empirical:     {comp_kappa_star['c_with_empirical']:.6f}")
    print(f"  c_with_operator:      {comp_kappa_star['c_with_operator']:.6f}")

    print(f"\nm₁ analysis:")
    print(f"  m1_needed to hit c:   {m1_needed_kappa_star:.4f}")
    print(f"  m1_empirical (exp+5): {m1_empirical_kappa_star:.4f}")
    print(f"  m1_operator_implied:  {comp_kappa_star['m1_implied']:.4f}")
    print(f"  Ratio needed/empirical: {m1_needed_kappa_star / m1_empirical_kappa_star:.4f}")

    # R-sweep analysis
    print("\n" + "=" * 50)
    print("R-SWEEP ANALYSIS: What m₁ is needed at each R?")
    print("=" * 50)

    R_values = [1.0, 1.1, 1.2, 1.3, 1.4]
    print("\nR      | m1_needed | exp(R)+5  | diff   | ratio")
    print("-" * 55)

    for R in R_values:
        # For R-sweep, use κ polynomials (just varying R)
        comp = compute_components(theta, R, n, polynomials_kappa)

        # Approximate c_target from PRZZ relationship
        # c = exp(R(1-κ)) where κ ≈ 0.417
        kappa_approx = 0.417
        c_target_approx = np.exp(R * (1 - kappa_approx))

        m1_needed = compute_m1_needed(
            comp['S12_plus_R'],
            comp['S12_minus_R'],
            comp['S34'],
            c_target_approx
        )
        m1_emp = np.exp(R) + 5
        diff = m1_needed - m1_emp
        ratio = m1_needed / m1_emp if m1_emp > 0 else float('inf')

        print(f"{R:.2f}   | {m1_needed:9.4f} | {m1_emp:9.4f} | {diff:6.2f} | {ratio:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
KEY FINDINGS:
1. The operator-derived mirror gives S12_op ≈ S12(+R), NOT m₁ × S12(-R)
2. The I₂ gate test confirms: I₂_op_mirror = I₂(+R) exactly
3. The empirical formula m₁ = exp(R) + 5 is an EFFECTIVE coefficient

INTERPRETATION:
The empirical formula encodes the DIFFERENCE between:
- What PRZZ TeX says the mirror should be (giving c_target)
- What evaluation at +R gives

The "+5" term cannot come from the operator transformation alone.
It appears to be a combinatorial correction from mollifier piece assembly.

OPEN QUESTION:
What is the mathematical origin of the "+2K-1" term (5 for K=3)?
Is it in PRZZ TeX somewhere? Or is it an emergent property of optimization?
""")


if __name__ == "__main__":
    main()
