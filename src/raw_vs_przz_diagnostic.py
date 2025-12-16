"""
src/raw_vs_przz_diagnostic.py
Diagnostic: What does our raw computation actually compute vs PRZZ?

KEY QUESTION:
Our c_raw = 1.95 is BELOW target 2.137.
Case C kernel ratios are < 1 (would make c smaller).
So why is the target LARGER than our raw?

HYPOTHESES:
1. (1,1) pair should match exactly (both pieces are Case B)
2. Case C pairs have different structure in PRZZ
3. There's something else INCREASING c that we're missing

This diagnostic tests hypothesis 1: does (1,1) match between us and PRZZ?
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, Any

from src.polynomials import load_przz_polynomials
from src.evaluate import evaluate_terms, evaluate_term, compute_kappa
from src.terms_k3_d1 import make_all_terms_11, make_I1_11, make_I2_11
from src.quadrature import tensor_grid_2d, gauss_legendre_01


THETA = 4/7
R = 1.3036
C_TARGET = 2.13745440613217263636


def compute_przz_style_11_manually(
    R: float,
    theta: float = THETA,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute (1,1) terms manually following PRZZ structure.

    PRZZ Section 6.2.1 gives:
    - I₁: (θ(x+y)+1)/θ × ∂²/∂x∂y [∫∫ (1−u)² P₁(x+u)P₁(y+u) Q(...)² exp(R...) dudt]
    - I₂: (1/θ) × ∫∫ P₁(u)² Q(t)² exp(2Rt) dudt

    For (1,1), both P factors are P₁ (Case B), so no Case C modification.

    We compute these manually and compare with DSL.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    U, T, W = tensor_grid_2d(n_quad)

    # ==========================================================================
    # I₂: Decoupled term (no derivatives needed)
    # I₂ = (1/θ) × ∫∫ P₁(u)² Q(t)² exp(2Rt) du dt
    # ==========================================================================

    P1_sq = P1.eval(U) ** 2
    Q_sq = Q.eval(T) ** 2
    exp_2R = np.exp(2 * R * T)

    I2_integrand = P1_sq * Q_sq * exp_2R / theta
    I2_manual = float(np.sum(W * I2_integrand))

    # ==========================================================================
    # I₁: Main coupled term (requires derivative extraction)
    # At x=y=0: P₁(x+u)P₁(y+u) → P₁(u)P₁(u)
    # Algebraic prefactor: (θ(x+y)+1)/θ → 1/θ at x=y=0
    # We need the xy-coefficient of the full integrand.
    #
    # Structure: (1/θ + S) × (1-u)² × P₁(X+u)P₁(Y+u) × Q(arg_α)Q(arg_β) × exp(R×...)
    # where S = x+y, X = x, Y = y
    #
    # The xy coefficient involves:
    # - Derivative of (1/θ + S) w.r.t. S → contributes to linear terms
    # - Derivatives of P₁ → P₁'(u)
    # - Derivatives of Q and exp through their arguments
    # ==========================================================================

    # For manual I₁, we'd need to trace through the full derivative extraction.
    # Instead, let's use DSL for I₁ and verify I₂ matches.

    # DSL computation
    term_I2 = make_I2_11(theta, R)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    I2_dsl = evaluate_term(term_I2, polys, n_quad).value

    results = {
        "I2_manual": I2_manual,
        "I2_dsl": I2_dsl,
        "I2_ratio": I2_dsl / I2_manual if abs(I2_manual) > 1e-15 else float('nan'),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("(1,1) PAIR: MANUAL vs DSL COMPARISON")
        print("=" * 70)

        print("\n--- I₂ (Decoupled Term) ---")
        print(f"  Formula: (1/θ) × ∫∫ P₁(u)² Q(t)² exp(2Rt) dudt")
        print(f"  Manual:  {I2_manual:+.12f}")
        print(f"  DSL:     {I2_dsl:+.12f}")
        print(f"  Ratio:   {results['I2_ratio']:.10f}")

        if abs(results['I2_ratio'] - 1.0) < 1e-6:
            print("  ✓ I₂ matches between manual and DSL")
        else:
            print("  ✗ I₂ MISMATCH!")

        print("=" * 70)

    return results


def analyze_11_contribution_to_c(
    R: float = R,
    theta: float = THETA,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze how (1,1) contributes to total c.

    The (1,1) pair is Case B × Case B, so it should be the "cleanest"
    comparison point between our code and PRZZ.

    If (1,1) alone matches PRZZ expectations, the gap is in other pairs.
    If (1,1) already shows a gap, there's a fundamental issue.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Evaluate all (1,1) terms
    terms_11 = make_all_terms_11(theta, R)
    result_11 = evaluate_terms(terms_11, polys, n_quad, return_breakdown=True)

    c_11_raw = result_11.total
    # With factorial normalization: 1/(1!×1!) = 1
    # Symmetry factor: 1 (diagonal)
    c_11_norm = c_11_raw

    # What fraction of c_target is from (1,1)?
    # We don't have PRZZ's per-pair breakdown, but we can estimate
    c_11_fraction = c_11_norm / C_TARGET

    results = {
        "c_11_raw": c_11_raw,
        "c_11_norm": c_11_norm,
        "c_target": C_TARGET,
        "c_11_fraction": c_11_fraction,
        "per_term": result_11.per_term,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("(1,1) CONTRIBUTION TO c")
        print("=" * 70)

        print("\n--- Per-Term Breakdown ---")
        for name, val in result_11.per_term.items():
            print(f"  {name}: {val:+.12f}")

        print(f"\n--- Summary ---")
        print(f"  c_11 (raw):     {c_11_raw:+.12f}")
        print(f"  c_11 (norm):    {c_11_norm:+.12f}")
        print(f"  c_target:       {C_TARGET:.12f}")
        print(f"  c_11/c_target:  {c_11_fraction*100:.2f}%")

        print(f"\n--- Analysis ---")
        print(f"  (1,1) is Case B × Case B (no Case C modification needed)")
        print(f"  If our (1,1) matches PRZZ, the gap is in other pairs.")
        print(f"  (1,1) contributes {c_11_fraction*100:.1f}% of target c.")

        print("=" * 70)

    return results


def compute_what_other_pairs_need(
    R: float = R,
    theta: float = THETA,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute what the non-(1,1) pairs need to contribute to reach target.

    c = c_11 + [other pairs]
    other_needed = c_target - c_11
    other_computed = c_raw - c_11

    If other_computed < other_needed, we're missing positive contributions.
    If other_computed > other_needed, we have excess that shouldn't be there.
    """
    from src.terms_k3_d1 import make_all_terms_k3

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Factorial and symmetry normalization
    factorial_norm = {"11": 1.0, "22": 1.0/4, "33": 1.0/36, "12": 1.0/2, "13": 1.0/6, "23": 1.0/12}
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    all_terms = make_all_terms_k3(theta, R)

    pair_contrib = {}
    total_raw = 0.0

    for pair_key, terms in all_terms.items():
        pair_result = evaluate_terms(terms, polys, n_quad, return_breakdown=False)
        raw = pair_result.total
        norm = factorial_norm[pair_key] * symmetry[pair_key]
        contrib = raw * norm

        pair_contrib[pair_key] = {
            "raw": raw,
            "norm_factor": norm,
            "contribution": contrib,
        }
        total_raw += contrib

    c_11 = pair_contrib["11"]["contribution"]
    c_other_computed = total_raw - c_11
    c_other_needed = C_TARGET - c_11

    results = {
        "c_11": c_11,
        "c_other_computed": c_other_computed,
        "c_other_needed": c_other_needed,
        "c_raw": total_raw,
        "c_target": C_TARGET,
        "pair_contrib": pair_contrib,
        "gap_in_other": c_other_needed - c_other_computed,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("WHERE IS THE GAP?")
        print("=" * 70)

        print("\n--- Per-Pair Contributions (normalized) ---")
        print(f"  {'Pair':<8} {'Raw':>15} {'Factor':>10} {'Contribution':>15}")
        print("  " + "-" * 55)
        for pair in ["11", "22", "33", "12", "13", "23"]:
            p = pair_contrib[pair]
            print(f"  ({pair[0]},{pair[1]})    {p['raw']:>+15.8f} {p['norm_factor']:>10.4f} {p['contribution']:>+15.8f}")

        print(f"\n--- Analysis ---")
        print(f"  c_11 (our):           {c_11:+.10f}")
        print(f"  c_other (our):        {c_other_computed:+.10f}")
        print(f"  c_raw (total):        {total_raw:+.10f}")
        print(f"")
        print(f"  c_target:             {C_TARGET:+.10f}")
        print(f"  c_other needed:       {c_other_needed:+.10f}")
        print(f"")
        print(f"  Gap in 'other' pairs: {results['gap_in_other']:+.10f}")
        print(f"  Gap as % of target:   {results['gap_in_other']/C_TARGET*100:+.2f}%")

        print(f"\n--- Interpretation ---")
        if results['gap_in_other'] > 0:
            print(f"  We need {results['gap_in_other']:.4f} MORE from non-(1,1) pairs")
            print(f"  Case C corrections would DECREASE contributions (wrong direction!)")
            print(f"  → There must be an ADDITIVE term we're missing")
        else:
            print(f"  We have {-results['gap_in_other']:.4f} EXCESS in non-(1,1) pairs")
            print(f"  Case C corrections could reduce this excess")

        print("=" * 70)

    return results


def hypothesis_additive_term(
    R: float = R,
    theta: float = THETA,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test hypothesis: Is there an additive term proportional to I₂?

    PRZZ may have structure: c = [our integrals] + [additive correction]

    The I₅ correction we have is SUBTRACTIVE (-S(0)×...).
    Maybe there's a different additive term?

    Test: What constant addition would make c match?
    """
    from src.terms_k3_d1 import make_all_terms_k3

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    factorial_norm = {"11": 1.0, "22": 1.0/4, "33": 1.0/36, "12": 1.0/2, "13": 1.0/6, "23": 1.0/12}
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    all_terms = make_all_terms_k3(theta, R)

    # Compute raw c and I₂ sum
    c_raw = 0.0
    i2_sum = 0.0

    for pair_key, terms in all_terms.items():
        pair_result = evaluate_terms(terms, polys, n_quad, return_breakdown=True)
        norm = factorial_norm[pair_key] * symmetry[pair_key]
        c_raw += pair_result.total * norm

        # I₂ is index 1 in each pair's term list
        i2_sum += pair_result.per_term.get(f"I2_{pair_key}", 0) * norm

    # What addition is needed?
    addition_needed = C_TARGET - c_raw

    # Express as multiple of I₂
    if abs(i2_sum) > 1e-10:
        addition_as_i2_multiple = addition_needed / i2_sum
    else:
        addition_as_i2_multiple = float('nan')

    # Also check: is it theta-related?
    addition_as_theta_c = addition_needed / (theta * c_raw) if c_raw > 0 else float('nan')

    results = {
        "c_raw": c_raw,
        "c_target": C_TARGET,
        "addition_needed": addition_needed,
        "i2_sum": i2_sum,
        "addition_as_i2_multiple": addition_as_i2_multiple,
        "addition_as_theta_c": addition_as_theta_c,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("HYPOTHESIS: ADDITIVE TERM?")
        print("=" * 70)

        print(f"\n  c_raw:            {c_raw:.10f}")
        print(f"  c_target:         {C_TARGET:.10f}")
        print(f"  Addition needed:  {addition_needed:+.10f}")
        print(f"  As % of raw:      {addition_needed/c_raw*100:+.2f}%")

        print(f"\n--- Test: Addition as Multiple of I₂ ---")
        print(f"  I₂ sum (normalized): {i2_sum:.10f}")
        print(f"  Addition / I₂:       {addition_as_i2_multiple:.6f}")

        print(f"\n--- Test: Addition as θ × c_raw ---")
        print(f"  θ = {theta:.10f}")
        print(f"  Addition / (θ × c_raw): {addition_as_theta_c:.6f}")

        # Check if it's close to known constants
        print(f"\n--- Pattern Matching ---")
        print(f"  Is addition ≈ θ/6 × c_raw? {abs(addition_as_theta_c - 1/6) < 0.01}")
        print(f"  Is addition ≈ I₂ × θ/6?    {abs(addition_as_i2_multiple - theta/6) < 0.1}")

        print("=" * 70)

    return results


if __name__ == "__main__":
    # Test (1,1) manual vs DSL
    compute_przz_style_11_manually(R, verbose=True)

    # Analyze (1,1) contribution
    analyze_11_contribution_to_c(verbose=True)

    # Find where the gap lives
    compute_what_other_pairs_need(verbose=True)

    # Test additive term hypothesis
    hypothesis_additive_term(verbose=True)
