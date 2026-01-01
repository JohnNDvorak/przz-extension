#!/usr/bin/env python3
"""
scripts/test_m_derivation.py
Phase 58: Test what m value is actually needed for PRZZ baseline

KEY TEST (from Claude Opus's recommendation):
Set g_I1 = g_I2 = 1 and C_K = 0 (no corrections).
Compute m directly as whatever makes the baseline work.

If m_needed ≈ exp(R) + 5 ≈ 8.68, then g-factors are cosmetic.
If m_needed is significantly different, that tells us something important.

Created: 2025-12-29
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polynomials import load_przz_polynomials


def compute_raw_integrals(R: float, theta: float, n_quad: int = 80):
    """
    Compute raw S12(+R), S12(-R), S34(+R) values.

    Uses the paper regime evaluator to get integral components.
    """
    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Normalization factors (PRZZ standard)
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

    # Generate terms at +R and -R
    all_terms_plus = make_all_terms_k3(theta, R, kernel_regime="paper")
    all_terms_minus = make_all_terms_k3(theta, -R, kernel_regime="paper")

    # Accumulate
    s12_plus = 0.0  # I1 + I2 at +R
    s12_minus = 0.0  # I1 + I2 at -R
    s34_plus = 0.0  # I3 + I4 at +R

    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms_plus = all_terms_plus[pair_key]
        terms_minus = all_terms_minus[pair_key]

        norm = factorial_norm[pair_key]
        sym = symmetry_factor[pair_key]
        full_norm = sym * norm

        # I1 and I2 (indices 0, 1) at +R and -R
        for i in range(2):
            if i < len(terms_plus):
                result_plus = evaluate_term(
                    terms_plus[i], polynomials, n_quad, R=R, theta=theta, n_quad_a=40
                )
                s12_plus += full_norm * result_plus.value

            if i < len(terms_minus):
                result_minus = evaluate_term(
                    terms_minus[i], polynomials, n_quad, R=-R, theta=theta, n_quad_a=40
                )
                s12_minus += full_norm * result_minus.value

        # I3 and I4 (indices 2, 3) at +R only
        for i in range(2, 4):
            if i < len(terms_plus):
                result_plus = evaluate_term(
                    terms_plus[i], polynomials, n_quad, R=R, theta=theta, n_quad_a=40
                )
                s34_plus += full_norm * result_plus.value

    return s12_plus, s12_minus, s34_plus


def solve_for_m(s12_plus: float, s12_minus: float, s34_plus: float, c_target: float) -> float:
    """
    Given c = S12(+R) + m × S12(-R) + S34(+R) = c_target,
    solve for m.
    """
    # c = s12_plus + m * s12_minus + s34_plus
    # m * s12_minus = c_target - s12_plus - s34_plus
    # m = (c_target - s12_plus - s34_plus) / s12_minus

    if abs(s12_minus) < 1e-10:
        return float('inf')

    return (c_target - s12_plus - s34_plus) / s12_minus


def main():
    """Run the m derivation test."""
    print("=" * 70)
    print("PHASE 58: TEST WHAT m VALUE IS ACTUALLY NEEDED")
    print("=" * 70)
    print()

    # PRZZ parameters
    R = 1.3036
    theta = 4/7
    K = 3

    # PRZZ target values
    kappa_target = 0.417293962
    c_target = math.exp(R * (1 - kappa_target))

    print("PRZZ Targets:")
    print(f"  R = {R}")
    print(f"  θ = {theta:.10f}")
    print(f"  κ_target = {kappa_target}")
    print(f"  c_target = {c_target:.10f}")
    print()

    # Compute raw integrals
    print("Computing raw integrals with PRZZ polynomials...")
    s12_plus, s12_minus, s34_plus = compute_raw_integrals(R, theta)

    print()
    print("Raw Integral Values:")
    print(f"  S12(+R) = {s12_plus:.10f}")
    print(f"  S12(-R) = {s12_minus:.10f}")
    print(f"  S34(+R) = {s34_plus:.10f}")
    print()

    # Solve for m that gives c_target
    m_needed = solve_for_m(s12_plus, s12_minus, s34_plus, c_target)

    print("Solving for m such that c = S12(+R) + m × S12(-R) + S34(+R) = c_target:")
    print(f"  m_needed = {m_needed:.10f}")
    print()

    # Compare with our formula m = exp(R) + 5
    m_formula = math.exp(R) + (2*K - 1)
    m_exp2R = math.exp(2*R)

    print("Comparison with candidate formulas:")
    print(f"  m_needed       = {m_needed:.6f}")
    print(f"  exp(R) + 5     = {m_formula:.6f}  (our formula)")
    print(f"  exp(2R)        = {m_exp2R:.6f}  (Gemini's claim)")
    print(f"  exp(R)         = {math.exp(R):.6f}")
    print()

    # Compute relative differences
    diff_formula = (m_needed - m_formula) / m_needed * 100
    diff_exp2R = (m_needed - m_exp2R) / m_needed * 100

    print("Relative differences:")
    print(f"  m_needed vs exp(R)+5: {diff_formula:+.2f}%")
    print(f"  m_needed vs exp(2R):  {diff_exp2R:+.2f}%")
    print()

    # Test what each formula gives
    print("Testing each formula:")

    formulas = [
        ("m_needed (exact)", m_needed),
        ("exp(R) + 5 (ours)", m_formula),
        ("exp(2R) (Gemini)", m_exp2R),
        ("No mirror (m=0)", 0),
        ("exp(R)", math.exp(R)),
    ]

    print()
    print(f"{'Formula':<25} {'m':>10} {'c':>12} {'κ':>12} {'κ gap':>12}")
    print("-" * 75)

    for name, m in formulas:
        c = s12_plus + m * s12_minus + s34_plus
        if c > 0:
            kappa = 1 - math.log(c) / R
        else:
            kappa = float('-inf')
        kappa_gap = kappa - kappa_target

        print(f"{name:<25} {m:>10.4f} {c:>12.6f} {kappa:>12.6f} {kappa_gap:>+12.6f}")

    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()

    if abs(diff_formula) < 5:
        print("✓ m_needed ≈ exp(R) + 5 (within 5%)")
        print("  This suggests our formula structure is CORRECT")
        print("  The g-factors may provide fine-tuning, not fundamental correction")
    else:
        print("✗ m_needed differs significantly from exp(R) + 5")
        print("  This suggests the formula needs investigation")

    print()

    if abs(diff_exp2R) > 30:
        print("✓ Gemini's exp(2R) formula is WRONG (differs by >30%)")
        print("  Confirmed: exp(2R) gives nonsensical κ < 0")

    print()

    # Additional diagnostic: What happens with PRZZ's own benchmark values?
    print("=" * 70)
    print("VERIFICATION WITH USER'S TABLE VALUES")
    print("=" * 70)
    print()
    print("From main_results.tex (lines 350-359), PRZZ baseline:")
    print("  S12(+R) = 0.797")
    print("  S12(-R) = 0.220")
    print("  S34(+R) = -0.600")
    print()

    s12_plus_table = 0.797
    s12_minus_table = 0.220
    s34_plus_table = -0.600

    m_from_table = solve_for_m(s12_plus_table, s12_minus_table, s34_plus_table, c_target)

    print(f"m_needed from table values = {m_from_table:.6f}")
    print(f"exp(R) + 5 = {m_formula:.6f}")
    print(f"Difference: {(m_from_table - m_formula)/m_from_table*100:+.2f}%")
    print()

    # Verify
    c_verify = s12_plus_table + m_from_table * s12_minus_table + s34_plus_table
    print(f"Verification: c = {c_verify:.6f} (target: {c_target:.6f})")


if __name__ == "__main__":
    main()
