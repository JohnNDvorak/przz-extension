#!/usr/bin/env python3
"""
scripts/test_gemini_operator_shift.py
Phase 58: Test Gemini's "Operator Shift" hypothesis

Gemini claims:
1. The mirror term should use Q(1-x), not Q(x)
2. With this shift, the weight exp(2R) gives correct physics
3. Our "shim" m = exp(R)+5 works for PRZZ due to symmetry but breaks under optimization

This script tests whether:
- exp(2R) with Q(1-x) shift gives κ ≈ 0.417 for PRZZ polynomials
- If so, what happens with optimized polynomials?

Created: 2025-12-29
"""

import math
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.polynomials import (
    load_przz_polynomials, P1Polynomial, PellPolynomial, Polynomial
)


def shift_polynomial_Q_to_1_minus_x(Q: Polynomial) -> Polynomial:
    """
    Transform Q(x) -> Q(1-x).

    If Q(x) = sum_i c_i * x^i, then
    Q(1-x) = sum_i c_i * (1-x)^i
           = sum_i c_i * sum_k binom(i,k) * (-1)^k * x^k
    """
    coeffs = Q.coeffs
    n = len(coeffs)
    new_coeffs = np.zeros(n)

    for i, c in enumerate(coeffs):
        for k in range(i + 1):
            bin_coeff = math.comb(i, k)
            term = c * bin_coeff * ((-1) ** k)
            new_coeffs[k] += term

    return Polynomial(new_coeffs)


def compute_integrals_with_Q_shift(
    polynomials: dict,
    polynomials_shifted: dict,
    R: float,
    theta: float,
    n_quad: int = 60
):
    """
    Compute integrals for Gemini's formula:
    - Main: S12 at -R with Q (standard)
    - Mirror: S12 at +R with Q(1-x) (shifted)
    - Cross: S34 at +R with Q (standard)
    """
    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term

    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0
    }

    # Generate terms
    all_terms_plus = make_all_terms_k3(theta, R, kernel_regime="paper")
    all_terms_minus = make_all_terms_k3(theta, -R, kernel_regime="paper")

    s_main = 0.0       # S12 at -R with standard Q
    s_mirror = 0.0     # S12 at +R with shifted Q(1-x)
    s_cross = 0.0      # S34 at +R with standard Q

    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms_plus = all_terms_plus[pair_key]
        terms_minus = all_terms_minus[pair_key]

        norm = factorial_norm[pair_key]
        sym = symmetry_factor[pair_key]
        full_norm = sym * norm

        # I1 and I2 components
        for i in range(2):
            # Main term: -R with standard Q
            if i < len(terms_minus):
                result_main = evaluate_term(
                    terms_minus[i], polynomials, n_quad, R=-R, theta=theta, n_quad_a=40
                )
                s_main += full_norm * result_main.value

            # Mirror term: +R with SHIFTED Q(1-x)
            if i < len(terms_plus):
                result_mirror = evaluate_term(
                    terms_plus[i], polynomials_shifted, n_quad, R=R, theta=theta, n_quad_a=40
                )
                s_mirror += full_norm * result_mirror.value

        # Cross terms: +R with standard Q
        for i in range(2, 4):
            if i < len(terms_plus):
                result_cross = evaluate_term(
                    terms_plus[i], polynomials, n_quad, R=R, theta=theta, n_quad_a=40
                )
                s_cross += full_norm * result_cross.value

    return s_main, s_mirror, s_cross


def main():
    print("=" * 70)
    print("PHASE 58: TEST GEMINI'S OPERATOR SHIFT HYPOTHESIS")
    print("=" * 70)
    print()
    print("Gemini's claim: Use Q(1-x) for mirror term, then exp(2R) works.")
    print()

    R = 1.3036
    theta = 4/7
    K = 3

    # Target values
    kappa_target = 0.417293962
    c_target = math.exp(R * (1 - kappa_target))

    print(f"PRZZ targets:")
    print(f"  κ_target = {kappa_target}")
    print(f"  c_target = {c_target:.6f}")
    print()

    # Load PRZZ polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
    Q_mono = Q.to_monomial()

    # Create shifted Q(1-x)
    Q_shifted = shift_polynomial_Q_to_1_minus_x(Q_mono)

    print("Q polynomial transformation:")
    print(f"  Q(x) coeffs:     {Q_mono.coeffs[:4]}...")
    print(f"  Q(1-x) coeffs:   {Q_shifted.coeffs[:4]}...")
    print()

    # Standard polynomials
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    polynomials_shifted = {"P1": P1, "P2": P2, "P3": P3, "Q": Q_shifted}

    # Compute integrals with Gemini's approach
    print("Computing integrals with Gemini's operator shift...")
    s_main, s_mirror_shifted, s_cross = compute_integrals_with_Q_shift(
        polynomials, polynomials_shifted, R, theta, n_quad=60
    )

    print()
    print("Gemini's Integrals:")
    print(f"  S_main(-R, Q)           = {s_main:.6f}")
    print(f"  S_mirror(+R, Q(1-x))    = {s_mirror_shifted:.6f}")
    print(f"  S_cross(+R, Q)          = {s_cross:.6f}")
    print()

    # Also compute our standard integrals for comparison
    from scripts.test_m_derivation import compute_raw_integrals
    s12_plus, s12_minus, s34_plus = compute_raw_integrals(R, theta, n_quad=60)

    print("Our Standard Integrals:")
    print(f"  S12(+R, Q)   = {s12_plus:.6f}")
    print(f"  S12(-R, Q)   = {s12_minus:.6f}")
    print(f"  S34(+R, Q)   = {s34_plus:.6f}")
    print()

    # Test assembly formulas
    print("=" * 70)
    print("ASSEMBLY FORMULA COMPARISON")
    print("=" * 70)
    print()

    exp_R = math.exp(R)
    exp_2R = math.exp(2 * R)
    m_shim = exp_R + (2*K - 1)

    formulas = [
        # (name, c_formula, description)
        ("Gemini (e^2R, shifted Q)",
         s_main + exp_2R * s_mirror_shifted + exp_R * s_cross,
         "c = S_main(-R) + e^(2R)×S_mirror(+R,shifted) + e^R×S_cross"),
        ("Gemini (e^2R, unshifted)",
         s12_minus + exp_2R * s12_plus + exp_R * s34_plus,
         "c = S12(-R) + e^(2R)×S12(+R) + e^R×S34 [PROVEN WRONG]"),
        ("Our formula (m≈8.8)",
         s12_plus + m_shim * s12_minus + s34_plus,
         "c = S12(+R) + m×S12(-R) + S34(+R)"),
    ]

    print(f"{'Formula':<30} {'c':>12} {'κ':>12} {'Gap from target':>18}")
    print("-" * 75)

    for name, c, desc in formulas:
        if c > 0:
            kappa = 1 - math.log(c) / R
            gap = kappa - kappa_target
            gap_pct = gap / kappa_target * 100
            print(f"{name:<30} {c:>12.6f} {kappa:>12.6f} {gap_pct:>+17.2f}%")
        else:
            print(f"{name:<30} {c:>12.6f} {'N/A':>12} {'INVALID':>18}")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    # Check which formula matches target
    c_gemini_shifted = s_main + exp_2R * s_mirror_shifted + exp_R * s_cross
    c_ours = s12_plus + m_shim * s12_minus + s34_plus

    if c_gemini_shifted > 0:
        kappa_gemini = 1 - math.log(c_gemini_shifted) / R
        gap_gemini = abs(kappa_gemini - kappa_target) / kappa_target * 100
    else:
        gap_gemini = float('inf')

    kappa_ours = 1 - math.log(c_ours) / R
    gap_ours = abs(kappa_ours - kappa_target) / kappa_target * 100

    print(f"Gemini's formula gap: {gap_gemini:.2f}%")
    print(f"Our formula gap:      {gap_ours:.2f}%")
    print()

    if gap_gemini < gap_ours and gap_gemini < 5:
        print("⚠ GEMINI'S FORMULA IS CLOSER TO TARGET")
        print("  The operator shift hypothesis may have merit!")
        print("  Further investigation required.")
    elif gap_ours < gap_gemini:
        print("✓ OUR FORMULA REMAINS CLOSER TO TARGET")
        print("  Gemini's operator shift does not improve accuracy.")
    else:
        print("⚠ BOTH FORMULAS HAVE SIGNIFICANT GAPS")
        print("  Neither matches PRZZ baseline well.")

    print()

    # Key diagnostic: How much does the Q shift change the mirror integral?
    print("KEY DIAGNOSTIC:")
    print(f"  S_mirror with Q(x):     {s12_plus:.6f}")
    print(f"  S_mirror with Q(1-x):   {s_mirror_shifted:.6f}")
    print(f"  Ratio:                  {s_mirror_shifted/s12_plus:.4f}")
    print()
    print("If Gemini is right, the shifted mirror should be much smaller,")
    print("compensating for the large exp(2R) weight.")


if __name__ == "__main__":
    main()
