#!/usr/bin/env python3
"""
Compare raw vs paper regime for I₁ and I₃/I₄ to find exact normalization factors.

Goal: Find factors F₁, F₃₄ such that:
    I₁_symbolic × F₁ = I₁_KappaEngine
    S₃₄_symbolic × F₃₄ = S₃₄_KappaEngine

Expected from previous analysis:
    I₁ ratio ≈ 4.58 → F₁ ≈ 0.218
    S₃₄ ratio ≈ 1.63 → F₃₄ ≈ 0.612 ≈ 30/49?
"""
import numpy as np
import math
from fractions import Fraction

from src.kappa_engine import KappaEngine
from src.path_a.optimal_coeffs import Q_coeffs, R_star_approx

# Polynomial coefficients in tilde form (as lists, not dicts with Rationals)
P1_list = [-2.0, 0.9375, 1.0, -0.6]
P2_list = [0.5241, 1.3199, -0.9401]
P3_list = [0.1367, -0.6865, -0.0499]


def expand_q_to_monomial(q0, q1, q3, q5):
    """
    Q(t) = q0 + q1(1-2t) + q3(1-2t)³ + q5(1-2t)⁵
    Convert to Q(t) = c0 + c1·t + c2·t² + c3·t³ + c4·t⁴ + c5·t⁵
    """
    c0 = q0 + q1 + q3 + q5
    c1 = -2*q1 - 6*q3 - 10*q5
    c2 = 12*q3 + 40*q5
    c3 = -8*q3 - 80*q5
    c4 = 80*q5
    c5 = -32*q5
    return [float(c0), float(c1), float(c2), float(c3), float(c4), float(c5)]


Q_mono = expand_q_to_monomial(
    float(Q_coeffs['q0']), float(Q_coeffs['q1']),
    float(Q_coeffs['q3']), float(Q_coeffs['q5'])
)

R_star = float(R_star_approx)
theta = 4/7


def create_polynomial_dict():
    """Create polynomial dictionary for unified engines."""
    from src.polynomials import make_P1_from_tilde, make_Pell_from_tilde, make_Q_from_basis

    # Create polynomial objects
    P1 = make_P1_from_tilde(P1_list)
    P2 = make_Pell_from_tilde(P2_list)  # PellPolynomial for P2
    P3 = make_Pell_from_tilde(P3_list)  # PellPolynomial for P3
    Q = make_Q_from_basis({0: float(Q_coeffs['q0']), 1: float(Q_coeffs['q1']),
                           3: float(Q_coeffs['q3']), 5: float(Q_coeffs['q5'])})

    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


def compute_I1_raw_vs_paper():
    """Compare raw vs paper regime for I₁ across all pairs."""
    from src.unified_i1_paper import compute_I1_unified_paper
    from src.unified_i1_general import compute_I1_unified_general

    print("=" * 70)
    print("COMPARING RAW vs PAPER REGIME FOR I₁")
    print("=" * 70)

    polynomials = create_polynomial_dict()

    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    print(f"\nAt R* = {R_star:.6f}")
    print(f"θ = {theta:.10f}")
    print()
    print(f"{'Pair':^8} {'Raw I₁':^15} {'Paper I₁':^15} {'Ratio':^10} {'Factor':^10}")
    print("-" * 70)

    results = {}

    for ell1, ell2 in pairs:
        # Raw regime
        raw_result = compute_I1_unified_general(
            R_star, theta, ell1, ell2, polynomials,
            n_quad_u=80, n_quad_t=80,
            include_Q=True, apply_factorial_norm=True
        )

        # Paper regime
        paper_result = compute_I1_unified_paper(
            R_star, theta, ell1, ell2, polynomials,
            n_quad_u=80, n_quad_t=80, n_quad_a=60,
            include_Q=True, apply_factorial_norm=True
        )

        raw_val = raw_result.I1_value
        paper_val = paper_result.I1_value

        ratio = raw_val / paper_val if abs(paper_val) > 1e-15 else float('inf')
        factor = 1.0 / ratio if abs(ratio) > 1e-15 else 0.0

        results[(ell1, ell2)] = {
            'raw': raw_val,
            'paper': paper_val,
            'ratio': ratio,
            'factor': factor,
            'omega1': paper_result.omega1,
            'omega2': paper_result.omega2,
        }

        print(f"({ell1},{ell2}){' ':^4} {raw_val:>+15.10f} {paper_val:>+15.10f} {ratio:>10.4f} {factor:>10.6f}")

    return results


def compute_I34_raw_vs_paper():
    """Compare raw vs paper regime for I₃/I₄ across all pairs."""
    from src.przz_exact_i34 import compute_I34_all_pairs

    print("\n" + "=" * 70)
    print("RAW REGIME FOR I₃/I₄")
    print("=" * 70)

    polynomials = create_polynomial_dict()

    # Raw I₃₄ - returns {"I3": {...}, "I4": {...}}
    raw_results = compute_I34_all_pairs(theta, R_star, polynomials, n_quad=80)

    print(f"\nAt R* = {R_star:.6f}")
    print(f"θ = {theta:.10f}")
    print()
    print(f"{'Pair':^8} {'I₃':^15} {'I₄':^15} {'I₃₄':^15}")
    print("-" * 60)

    # Combine I3 and I4 per pair
    combined_results = {}
    raw_S34 = 0.0
    for key in raw_results["I3"].keys():
        I3_val = raw_results["I3"][key].value
        I4_val = raw_results["I4"][key].value
        I34_val = I3_val + I4_val

        # Off-diagonal pairs count twice
        ell1, ell2 = int(key[0]), int(key[1])
        mult = 2 if ell1 != ell2 else 1
        raw_S34 += mult * I34_val

        combined_results[key] = {
            'I3': I3_val,
            'I4': I4_val,
            'I34': I34_val,
        }
        print(f"{key:^8} {I3_val:>+15.10f} {I4_val:>+15.10f} {I34_val:>+15.10f}")

    print(f"\n  Raw S₃₄ total = {raw_S34:+.10f}")

    return combined_results, raw_S34


def analyze_factors():
    """Main analysis to find exact normalization factors."""
    print("=" * 70)
    print("COMPUTING WEIGHTED SUMS AND GLOBAL FACTORS")
    print("=" * 70)

    # Get KappaEngine values using correct initialization
    engine = KappaEngine(P1_list, P2_list, P3_list, Q_mono, theta=theta, K=3, R=R_star)
    result = engine.compute_kappa()

    print(f"\nKappaEngine values at R* = {R_star:.6f}:")
    print(f"  I₁(+R) = {result.integrals.I1_plus:.10f}")
    print(f"  I₂(+R) = {result.integrals.I2_plus:.10f}")
    print(f"  S₁₂(+R) = {result.integrals.S12_plus:.10f}")
    print(f"  S₃₄(+R) = {result.integrals.S34_plus:.10f}")
    print(f"  c = {result.c:.10f}")

    # Verify c = 1
    if abs(result.c - 1.0) > 0.01:
        print(f"\n  ⚠️ WARNING: c = {result.c:.6f} ≠ 1.0")
        print("     Check that polynomials are correct!")
    else:
        print(f"\n  ✓ c ≈ 1.0 verified")

    # Compare with raw I₁
    i1_results = compute_I1_raw_vs_paper()

    # Weighted sum of raw I₁ (with 2x factor for off-diagonal)
    raw_I1_weighted = sum(
        r['raw'] * (2 if ell1 != ell2 else 1)
        for (ell1, ell2), r in i1_results.items()
    )
    paper_I1_weighted = sum(
        r['paper'] * (2 if ell1 != ell2 else 1)
        for (ell1, ell2), r in i1_results.items()
    )

    print(f"\n  Weighted sum of raw I₁ = {raw_I1_weighted:.10f}")
    print(f"  Weighted sum of paper I₁ = {paper_I1_weighted:.10f}")
    print(f"  KappaEngine I₁(+R) = {result.integrals.I1_plus:.10f}")

    # Factor for I₁
    I1_ratio = raw_I1_weighted / result.integrals.I1_plus
    I1_factor = 1.0 / I1_ratio
    print(f"\n  I₁ RATIO (raw/engine) = {I1_ratio:.6f}")
    print(f"  I₁ FACTOR (engine/raw) = {I1_factor:.6f}")

    # Now for S₃₄
    i34_results, raw_S34 = compute_I34_raw_vs_paper()

    S34_ratio = raw_S34 / result.integrals.S34_plus
    S34_factor = 1.0 / S34_ratio

    print(f"\n  S₃₄ RATIO (raw/engine) = {S34_ratio:.6f}")
    print(f"  S₃₄ FACTOR (engine/raw) = {S34_factor:.6f}")

    # Try to find simple rational approximations
    print("\n" + "=" * 70)
    print("SEARCHING FOR RATIONAL FACTORS")
    print("=" * 70)

    for name, factor in [("I₁", I1_factor), ("S₃₄", S34_factor)]:
        print(f"\n{name} factor = {factor:.10f}")
        print("  Checking rational combinations with small integers...")

        # Check common combinations
        theta_frac = Fraction(4, 7)
        K = 3

        candidates = [
            ("1/θ", 1/float(theta_frac)),
            ("θ", float(theta_frac)),
            ("1-θ", float(1 - theta_frac)),
            ("1+θ", float(1 + theta_frac)),
            ("θ/(1+θ)", float(theta_frac/(1 + theta_frac))),
            ("(1-θ)/(1+θ)", float((1-theta_frac)/(1+theta_frac))),
            ("1/(2K+1)", 1/(2*K+1)),
            ("1/(2K-1)", 1/(2*K-1)),
            ("3/(2K+1)", 3/(2*K+1)),
            ("(1-θ)·θ", float((1-theta_frac)*theta_frac)),
            ("30/49", 30/49),
            ("6/49", 6/49),
            ("7/49", 7/49),
            ("3/14", 3/14),
            ("θ²", float(theta_frac**2)),
            ("1/7", 1/7),
            ("2/7", 2/7),
            ("3/7", 3/7),
            ("1/6", 1/6),
            ("1/5", 1/5),
            ("1/4", 1/4),
            ("1/3", 1/3),
            # Additional candidates for Case C kernels
            ("1/(1+θ)", 1/float(1+theta_frac)),
            ("(1-θ)²", float((1-theta_frac)**2)),
            ("θ(1-θ)", float(theta_frac*(1-theta_frac))),
            ("1/(K*(2K+1))", 1/(K*(2*K+1))),
            ("θ/(K*(2K+1))", float(theta_frac)/(K*(2*K+1))),
        ]

        for desc, val in candidates:
            if abs(factor - val) < 0.02:
                print(f"  CLOSE MATCH: {desc} = {val:.6f} (diff = {factor - val:.6f})")
            if abs(factor * val - 1) < 0.02:
                print(f"  RECIPROCAL MATCH: 1/{desc} = {1/val:.6f} (diff = {factor - 1/val:.6f})")

    return result, i1_results, raw_S34


def compute_per_pair_I34_paper():
    """Compute I₃₄ in paper regime using unified_i34_paper."""
    print("\n" + "=" * 70)
    print("PAPER REGIME I₃₄ PER PAIR")
    print("=" * 70)

    try:
        from src.unified_i34_paper import compute_I34_unified_paper
    except ImportError:
        print("  unified_i34_paper not available")
        return None

    polynomials = create_polynomial_dict()
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    print(f"\nAt R* = {R_star:.6f}")
    print()
    print(f"{'Pair':^8} {'Paper I₃':^15} {'Paper I₄':^15} {'Paper I₃₄':^15}")
    print("-" * 60)

    total = 0.0
    for ell1, ell2 in pairs:
        result = compute_I34_unified_paper(
            R_star, theta, ell1, ell2, polynomials,
            n_quad_u=80, n_quad_t=80, n_quad_a=60,
            include_Q=True, apply_factorial_norm=True
        )

        mult = 2 if ell1 != ell2 else 1
        I34 = result.I3_value + result.I4_value
        total += mult * I34

        print(f"({ell1},{ell2}){' ':^4} {result.I3_value:>+15.10f} {result.I4_value:>+15.10f} {I34:>+15.10f}")

    print(f"\n  Total paper S₃₄ = {total:+.10f}")
    return total


if __name__ == "__main__":
    result, i1_results, raw_S34 = analyze_factors()

    print("\n" + "=" * 70)
    print("PER-PAIR ANALYSIS: RAW/PAPER RATIO")
    print("=" * 70)

    print(f"\n{'Pair':^8} {'ω₁':^4} {'ω₂':^4} {'Ratio':^10} {'Notes':^30}")
    print("-" * 70)

    for (ell1, ell2), r in i1_results.items():
        notes = ""
        if r['omega1'] == 0 and r['omega2'] == 0:
            notes = "Both Case B (raw)"
        elif r['omega1'] > 0 and r['omega2'] > 0:
            notes = "Both Case C (attenuated)"
        else:
            notes = "Mixed Case B/C"

        print(f"({ell1},{ell2}){' ':^4} {r['omega1']:^4} {r['omega2']:^4} {r['ratio']:>10.4f} {notes:^30}")

    # Paper regime I34
    try:
        paper_S34 = compute_per_pair_I34_paper()
        if paper_S34 is not None:
            print(f"\n  Comparison:")
            print(f"    Raw S₃₄ = {raw_S34:+.10f}")
            print(f"    Paper S₃₄ = {paper_S34:+.10f}")
            print(f"    KappaEngine S₃₄ = {result.integrals.S34_plus:+.10f}")
    except Exception as e:
        print(f"\n  Could not compute paper I34: {e}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
