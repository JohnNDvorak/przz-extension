"""
run_q_variation_test.py
Test: Does the "+5" constant come from Q-operator structure?

GPT Recommendation (2025-12-19)
-------------------------------
This is "huge and cheap" - will cut search tree in half.

Use varying Q polynomials while holding P fixed:
- Q₀(t) = 1  (constant)
- Q₁(t) = 1 + t  (linear, still Q(0)=1)
- Q₂(t) = 1 + t + t²  (quadratic)

Then recompute the required multiplier.

If the constant "5" tracks deg(Q) or # of nonzero coefficients:
  → The "+5" comes from Q operator structure (Q(1+D) shift)

If "5" persists even when Q=1:
  → The "+5" is NOT from Q - it's from log(N^{x+y}T) prefactor structure

This test isolates whether Q-operator shift is the source of the empirical constant.
"""

from __future__ import annotations

import math
from typing import Dict, Union

import numpy as np

from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials, Polynomial


# Constants
THETA = 4.0 / 7.0
N_QUAD = 60
N_QUAD_A = 40
R = 1.3036  # κ benchmark
C_TARGET = 2.137


def make_constant_Q() -> Polynomial:
    """Q(t) = 1 (constant)"""
    return Polynomial([1.0])


def make_linear_Q() -> Polynomial:
    """Q(t) = 1 + t (linear, Q(0)=1)"""
    return Polynomial([1.0, 1.0])


def make_quadratic_Q() -> Polynomial:
    """Q(t) = 1 + t + t² (quadratic, Q(0)=1)"""
    return Polynomial([1.0, 1.0, 1.0])


def make_cubic_Q() -> Polynomial:
    """Q(t) = 1 + t + t² + t³ (cubic, Q(0)=1)"""
    return Polynomial([1.0, 1.0, 1.0, 1.0])


def run_test(name: str, Q_func: Polynomial, polynomials_base: Dict):
    """Run mirror assembly with custom Q polynomial."""
    # Create polynomial dict with custom Q
    polynomials = {
        "P1": polynomials_base["P1"],
        "P2": polynomials_base["P2"],
        "P3": polynomials_base["P3"],
        "Q": Q_func,
    }

    # Run with standard mirror multiplier
    result = compute_c_paper_with_mirror(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        n_quad_a=N_QUAD_A,
        K=3,
    )

    c_computed = result.total
    direct_c = result.per_term.get("_direct_c", 0.0)
    mirror_I12 = result.per_term.get("_mirror_I12", 0.0)
    m_used = result.per_term.get("_mirror_multiplier", 0.0)

    # What m would be needed to hit a hypothetical target?
    # We can't use C_TARGET since Q changes the relationship
    # Instead, compute m_needed to make c = some reference
    # Actually, let's just report the structure

    print(f"\n{name}:")
    print(f"  direct_c:     {direct_c:+.8f}")
    print(f"  mirror_I12:   {mirror_I12:+.8f}")
    print(f"  m_used:       {m_used:.6f}  (exp(R)+5)")
    print(f"  c_computed:   {c_computed:+.8f}")

    # Ratio: how much does mirror contribute relative to direct?
    if direct_c != 0:
        ratio = (m_used * mirror_I12) / direct_c
        print(f"  mirror_contrib/direct: {ratio:+.4f}")

    return {
        "name": name,
        "direct_c": direct_c,
        "mirror_I12": mirror_I12,
        "c_computed": c_computed,
    }


def main():
    print("=" * 78)
    print("Q-VARIATION TEST: Does '+5' come from Q-operator structure?")
    print("=" * 78)
    print()
    print("Hypothesis: If '+5' is from Q(1+D) operator shift,")
    print("then changing Q should change the required multiplier structure.")
    print()
    print("Test: Use Q = 1, Q = 1+t, Q = 1+t+t², Q = 1+t+t²+t³")
    print("      with the SAME P polynomials from κ benchmark")
    print()

    # Load base polynomials (P1, P2, P3)
    P1, P2, P3, Q_original = load_przz_polynomials(enforce_Q0=True)
    polys_base = {"P1": P1, "P2": P2, "P3": P3}

    # Test with different Q polynomials
    results = []

    # Original Q (for reference)
    print("-" * 78)
    print("Reference: Original PRZZ Q polynomial")
    result_orig = compute_c_paper_with_mirror(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials={"P1": P1, "P2": P2, "P3": P3, "Q": Q_original},
        n_quad_a=N_QUAD_A,
        K=3,
    )
    print(f"  c_computed: {result_orig.total:+.8f}  (target: {C_TARGET})")
    print(f"  gap: {(result_orig.total - C_TARGET) / C_TARGET * 100:+.2f}%")

    # Test Q variations
    print("-" * 78)
    print("Q-variation tests:")

    results.append(run_test("Q(t) = 1 (constant)", make_constant_Q(), polys_base))
    results.append(run_test("Q(t) = 1 + t", make_linear_Q(), polys_base))
    results.append(run_test("Q(t) = 1 + t + t²", make_quadratic_Q(), polys_base))
    results.append(run_test("Q(t) = 1 + t + t² + t³", make_cubic_Q(), polys_base))

    # Analysis
    print()
    print("=" * 78)
    print("ANALYSIS")
    print("=" * 78)
    print()

    # Check: does the mirror_I12 / direct_c ratio change with Q?
    print("mirror_I12 / direct_c ratio across Q variations:")
    for r in results:
        if r["direct_c"] != 0:
            ratio = r["mirror_I12"] / r["direct_c"]
            print(f"  {r['name']:30s}: {ratio:+.6f}")

    print()

    # Key question: is direct_c itself changing significantly?
    if results[0]["direct_c"] != 0:
        q1_direct = results[0]["direct_c"]
        print(f"direct_c relative to Q=1 case:")
        for r in results:
            rel = r["direct_c"] / q1_direct if q1_direct != 0 else float('inf')
            print(f"  {r['name']:30s}: {rel:.6f}× Q=1 value")

    print()
    print("CONCLUSION:")
    print("-" * 78)

    # Check if mirror_I12 changes substantially with Q
    q1_mirror = results[0]["mirror_I12"]
    q3_mirror = results[3]["mirror_I12"]

    if abs(q1_mirror) > 1e-10 and abs(q3_mirror) > 1e-10:
        mirror_change = q3_mirror / q1_mirror
        if abs(mirror_change - 1.0) > 0.5:
            print("  → mirror_I12 DOES change significantly with Q complexity")
            print("  → The '+5' MAY be related to Q operator structure")
        else:
            print("  → mirror_I12 does NOT change much with Q complexity")
            print("  → The '+5' is likely NOT from Q operator structure")
            print("  → Look elsewhere: log(N^{x+y}T) prefactor, (1-u) weights, etc.")
    else:
        print("  → Insufficient mirror_I12 magnitude for comparison")


if __name__ == "__main__":
    main()
