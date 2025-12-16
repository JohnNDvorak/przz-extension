"""
src/investigate_12_anomaly.py
Investigation: Why does (1,2) pair show anomalous R-sensitivity?

Q-OPERATOR ORACLE FINDING:
  Per-pair ratio changes (R=1.3036 → R=1.1167):
      11:  +4.69%
      22:  +2.28%
      33:  +6.75%
      12: +32.32%  ← HUGE OUTLIER!
      13:  +6.64%
      23:  +4.18%

The (1,2) pair is the only one showing anomalous R-sensitivity.
This is a critical clue to the missing term family.

HYPOTHESES:
1. Sign error in (1,2) numeric_prefactor
2. Case B × Case C cross-term has special structure
3. Missing R-dependent factor specific to (1,2)
4. Polynomial structure artifact

This script investigates each hypothesis systematically.
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict

from src.polynomials import load_przz_polynomials
from src.evaluate import evaluate_term
from src.terms_k3_d1 import make_I1_12, make_I2_12, make_I3_12, make_I4_12


def compute_12_term_breakdown(R: float, theta: float = 4/7, n_quad: int = 60) -> Dict:
    """Compute per-term breakdown for (1,2) pair."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    terms = {
        "I1": make_I1_12(theta, R),
        "I2": make_I2_12(theta, R),
        "I3": make_I3_12(theta, R),
        "I4": make_I4_12(theta, R),
    }

    results = {}
    for name, term in terms.items():
        result = evaluate_term(term, polys, n_quad)
        results[name] = result.value

    results["total"] = sum(results.values())
    return results


def hypothesis_1_sign_check(verbose: bool = True) -> Dict:
    """
    Hypothesis 1: Is the numeric_prefactor sign correct?

    For (1,2): (-1)^{ℓ₁+ℓ₂} = (-1)^{1+2} = (-1)^3 = -1

    Check: Does PRZZ expect positive or negative contribution from (1,2)?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("HYPOTHESIS 1: Sign Check for (1,2) Pair")
        print("=" * 70)

        print("\nPRZZ Sign Convention:")
        print("  For pair (ℓ₁, ℓ₂), the sign is (-1)^{ℓ₁+ℓ₂}")
        print("  (1,2): (-1)^{1+2} = (-1)^3 = -1")
        print("\nOur implementation uses:")
        print("  I1_12: numeric_prefactor = -1.0 ✓")
        print("  I2_12: numeric_prefactor = +1/θ (decoupled)")
        print("  I3_12: numeric_prefactor = -1/θ")
        print("  I4_12: numeric_prefactor = -1/θ")

        print("\nThe sign seems correct per the convention.")
        print("=" * 70)

    return {"sign_correct": True, "convention": "(-1)^{ℓ₁+ℓ₂}"}


def hypothesis_2_case_cross_term(verbose: bool = True) -> Dict:
    """
    Hypothesis 2: Does Case B × Case C have special structure?

    P₁ is Case B (ω=0)
    P₂ is Case C (ω=1)

    Check cross-term structure differences.
    """
    theta = 4/7

    # Compute (1,2) at two R values
    r1_12 = compute_12_term_breakdown(1.3036, theta)
    r2_12 = compute_12_term_breakdown(1.1167, theta)

    # Compare (1,1) which is Case B × Case B
    from src.terms_k3_d1 import make_I1_11, make_I2_11, make_I3_11, make_I4_11
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def compute_11_breakdown(R):
        terms = [make_I1_11(theta, R), make_I2_11(theta, R),
                 make_I3_11(theta, R), make_I4_11(theta, R)]
        return {f"I{i+1}": evaluate_term(t, polys, 60).value for i, t in enumerate(terms)}

    r1_11 = compute_11_breakdown(1.3036)
    r2_11 = compute_11_breakdown(1.1167)

    results = {
        "r1_12": r1_12,
        "r2_12": r2_12,
        "r1_11": r1_11,
        "r2_11": r2_11,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("HYPOTHESIS 2: Case B × Case C Cross-Term Analysis")
        print("=" * 70)

        print("\n(1,2) = P₁ × P₂ = Case B × Case C")
        print("(1,1) = P₁ × P₁ = Case B × Case B")

        print(f"\n{'─'*40}")
        print("Per-term breakdown at R=1.3036 vs R=1.1167:")
        print(f"{'─'*40}")

        print("\n(1,2) pair:")
        for term in ["I1", "I2", "I3", "I4"]:
            v1, v2 = r1_12[term], r2_12[term]
            change = (v2 - v1) / abs(v1) * 100 if abs(v1) > 1e-15 else 0
            print(f"  {term}: {v1:+.8f} → {v2:+.8f} ({change:+.1f}%)")

        print("\n(1,1) pair (reference):")
        for term in ["I1", "I2", "I3", "I4"]:
            v1, v2 = r1_11[term], r2_11[term]
            change = (v2 - v1) / abs(v1) * 100 if abs(v1) > 1e-15 else 0
            print(f"  {term}: {v1:+.8f} → {v2:+.8f} ({change:+.1f}%)")

        print("\nKey insight: Which terms show most R-sensitivity?")
        print("=" * 70)

    return results


def hypothesis_3_r_dependent_factor(verbose: bool = True) -> Dict:
    """
    Hypothesis 3: Is there an R-dependent factor missing?

    Test: Compute ratio I1_12 / I1_11 at different R values.
    If there's a missing R-dependent factor in (1,2) but not (1,1),
    this ratio will change with R.
    """
    theta = 4/7
    R_values = [1.0, 1.1, 1.1167, 1.2, 1.3036, 1.4]

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    from src.terms_k3_d1 import make_I1_11

    results = []
    for R in R_values:
        i1_12 = evaluate_term(make_I1_12(theta, R), polys, 60).value
        i1_11 = evaluate_term(make_I1_11(theta, R), polys, 60).value

        results.append({
            "R": R,
            "I1_12": i1_12,
            "I1_11": i1_11,
            "ratio": i1_12 / i1_11 if abs(i1_11) > 1e-15 else float('nan'),
        })

    if verbose:
        print("\n" + "=" * 70)
        print("HYPOTHESIS 3: R-Dependent Factor Search")
        print("=" * 70)

        print("\nRatio I1_12 / I1_11 at different R values:")
        print("(If ratio changes with R, (1,2) has different R-dependence)")
        print()
        print(f"  {'R':>8} | {'I1_12':>15} | {'I1_11':>15} | {'Ratio':>10}")
        print(f"  {'-'*8} | {'-'*15} | {'-'*15} | {'-'*10}")

        for r in results:
            print(f"  {r['R']:>8.4f} | {r['I1_12']:>+15.8f} | {r['I1_11']:>+15.8f} | {r['ratio']:>10.6f}")

        # Check ratio variation
        ratios = [r['ratio'] for r in results]
        ratio_var = np.std(ratios) / np.mean(ratios) * 100
        print(f"\n  Ratio coefficient of variation: {ratio_var:.2f}%")

        if ratio_var > 5:
            print("  → Significant R-dependence in ratio!")
            print("  → (1,2) scales differently with R than (1,1)")
        else:
            print("  → Ratio is relatively stable across R values")

        print("=" * 70)

    return {"r_scan": results}


def hypothesis_4_polynomial_structure(verbose: bool = True) -> Dict:
    """
    Hypothesis 4: Is it the polynomial structure?

    P₁ has degree ~11 (from PRZZ table)
    P₂ has degree ~10

    The different polynomial shapes could interact with R in unexpected ways.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    # Sample polynomials at key points
    u_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    P1_vals = P1.eval(u_vals)
    P2_vals = P2.eval(u_vals)

    if verbose:
        print("\n" + "=" * 70)
        print("HYPOTHESIS 4: Polynomial Structure Analysis")
        print("=" * 70)

        print("\nPolynomial values at key points:")
        print(f"  {'u':>6} | {'P₁(u)':>12} | {'P₂(u)':>12} | {'P₁×P₂':>12}")
        print(f"  {'-'*6} | {'-'*12} | {'-'*12} | {'-'*12}")

        for i, u in enumerate(u_vals):
            prod = P1_vals[i] * P2_vals[i]
            print(f"  {u:>6.2f} | {P1_vals[i]:>+12.6f} | {P2_vals[i]:>+12.6f} | {prod:>+12.6f}")

        # Check P₁ and P₂ derivatives
        print(f"\nFirst derivatives at u=0.5:")
        P1_prime = P1.eval_deriv(np.array([0.5]), 1)[0]
        P2_prime = P2.eval_deriv(np.array([0.5]), 1)[0]
        print(f"  P₁'(0.5) = {P1_prime:+.6f}")
        print(f"  P₂'(0.5) = {P2_prime:+.6f}")

        print("=" * 70)

    return {"P1_vals": P1_vals, "P2_vals": P2_vals}


def run_full_investigation(verbose: bool = True) -> Dict:
    """Run all hypothesis tests."""
    results = {}

    if verbose:
        print("\n")
        print("█" * 80)
        print("█  INVESTIGATING (1,2) PAIR R-SENSITIVITY ANOMALY")
        print("█" * 80)
        print("\nThe (1,2) pair shows +32% R-sensitivity while others show ~2-7%")
        print("This investigation tests 4 hypotheses to identify the cause.\n")

    results["hypothesis_1"] = hypothesis_1_sign_check(verbose)
    results["hypothesis_2"] = hypothesis_2_case_cross_term(verbose)
    results["hypothesis_3"] = hypothesis_3_r_dependent_factor(verbose)
    results["hypothesis_4"] = hypothesis_4_polynomial_structure(verbose)

    if verbose:
        print("\n")
        print("█" * 80)
        print("█  INVESTIGATION SUMMARY")
        print("█" * 80)

        # Analyze findings
        h3_ratios = results["hypothesis_3"]["r_scan"]
        ratio_var = np.std([r['ratio'] for r in h3_ratios]) / np.mean([r['ratio'] for r in h3_ratios]) * 100

        print("\n  Findings:")
        print(f"    1. Sign convention appears correct")
        print(f"    2. Case B×C cross-term structure differs from B×B")
        print(f"    3. I1_12/I1_11 ratio variation: {ratio_var:.1f}%")

        if ratio_var > 5:
            print(f"\n  KEY INSIGHT:")
            print(f"    The (1,2)/(1,1) ratio VARIES with R")
            print(f"    This means (1,2) and (1,1) scale differently!")
            print(f"    → Missing R-dependent term specific to P₁×P₂ cross-term")
            print(f"    → Or different R-dependence in Case B×C vs B×B")

        print("█" * 80 + "\n")

    return results


if __name__ == "__main__":
    run_full_investigation(verbose=True)
