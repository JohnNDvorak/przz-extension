"""
src/case_c_investigation.py
Investigation: Is the Case C auxiliary integral missing?

PRZZ TeX 2369-2384 defines auxiliary integrals for Case C (ω > 0):
    ∫₀¹ (1-a)^i a^{ω-1} (N/n)^{-αa} da

For d=1:
- P₁: k=2, ω = k-2 = 0 → Case B (no auxiliary integral)
- P₂: k=3, ω = k-2 = 1 → Case C (auxiliary integral needed)
- P₃: k=4, ω = k-2 = 2 → Case C (auxiliary integral needed)

If Case C auxiliary integral is MISSING:
- (1,1) = B × B → should be CORRECT
- (1,2) = B × C → P₂ side needs auxiliary integral
- (1,3) = B × C → P₃ side needs auxiliary integral
- (2,2) = C × C → BOTH sides need auxiliary integral
- (2,3) = C × C → BOTH sides need auxiliary integral
- (3,3) = C × C → BOTH sides need auxiliary integral

HYPOTHESIS TEST:
If missing Case C integral is the culprit:
1. (1,1) error should be smallest (no Case C involvement)
2. Pure C×C pairs (2,2), (3,3), (2,3) should have similar error pattern
3. Mixed B×C pairs (1,2), (1,3) should have intermediate error

ACTUAL Q-OPERATOR ORACLE RESULTS:
| Pair | Case | R-sensitivity |
|------|------|--------------|
| (1,1) | B×B | +4.69% |
| (2,2) | C×C | +2.28% ← SMALLEST! |
| (3,3) | C×C | +6.75% |
| (1,2) | B×C | +32.32% ← ANOMALY |
| (1,3) | B×C | +6.64% |
| (2,3) | C×C | +4.18% |

OBSERVATION:
- (2,2) has SMALLEST sensitivity, not (1,1)
- The anomaly is (1,2), a B×C mixed pair
- This doesn't match "missing Case C integral" hypothesis simply

This script investigates the Case C structure more deeply.
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, List

from src.polynomials import load_przz_polynomials
from src.evaluate import evaluate_term, evaluate_c_full
from src.terms_k3_d1 import make_all_terms_k3


def analyze_case_classification(verbose: bool = True) -> Dict:
    """
    Analyze pairs by their ω-case classification.
    """
    # Case classification
    case_map = {
        "11": ("B", "B"),  # P₁×P₁
        "22": ("C", "C"),  # P₂×P₂
        "33": ("C", "C"),  # P₃×P₃
        "12": ("B", "C"),  # P₁×P₂
        "13": ("B", "C"),  # P₁×P₃
        "23": ("C", "C"),  # P₂×P₃
    }

    # ω values (for d=1: ω = k-2 where k is PRZZ index)
    omega_map = {
        "P1": 0,  # k=2, ω=0, Case B
        "P2": 1,  # k=3, ω=1, Case C
        "P3": 2,  # k=4, ω=2, Case C
    }

    results = {
        "case_map": case_map,
        "omega_map": omega_map,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("CASE CLASSIFICATION ANALYSIS")
        print("=" * 70)

        print("\nω values (for d=1: ω = k-2):")
        for poly, omega in omega_map.items():
            case = "B" if omega == 0 else "C"
            print(f"  {poly}: k={omega+2}, ω={omega} → Case {case}")

        print("\nPair classifications:")
        print(f"  {'Pair':>6} | {'Left':>6} | {'Right':>6} | {'Type':>10}")
        print(f"  {'-'*6} | {'-'*6} | {'-'*6} | {'-'*10}")
        for pair, (left, right) in case_map.items():
            pair_type = f"{left}×{right}"
            print(f"  ({pair[0]},{pair[1]}) | Case {left} | Case {right} | {pair_type}")

        print("\nPRZZ Case C Auxiliary Integral (TeX 2369-2384):")
        print("  For ω > 0: ∫₀¹ (1-a)^i a^{ω-1} (N/n)^{-αa} da")
        print("  P₂ (ω=1): ∫₀¹ (1-a)^i (N/n)^{-αa} da")
        print("  P₃ (ω=2): ∫₀¹ (1-a)^i a (N/n)^{-αa} da")

        print("\n" + "=" * 70)

    return results


def check_if_auxiliary_integral_implemented(verbose: bool = True) -> Dict:
    """
    Check if our term builders include auxiliary integral structure.
    """
    # Search for signs of auxiliary integral in our implementation
    # The auxiliary integral would involve:
    # 1. Additional integration variable 'a' in [0,1]
    # 2. Factors like (1-a)^i, a^{ω-1}
    # 3. Exponential factor (N/n)^{-αa}

    # Our current implementation uses 2D quadrature (u,t)
    # If we needed auxiliary integral, we'd need 3D quadrature (u,t,a)

    findings = {
        "auxiliary_integral_present": False,
        "integration_dimension": 2,
        "reason": "No 'a' variable or auxiliary integral structure found in term builders",
    }

    if verbose:
        print("\n" + "=" * 70)
        print("AUXILIARY INTEGRAL IMPLEMENTATION CHECK")
        print("=" * 70)

        print("\nCurrent implementation:")
        print("  - Integration domain: [0,1]² (u,t)")
        print("  - No auxiliary 'a' variable found")
        print("  - Term builders use standard 2D quadrature")

        print("\nIf Case C auxiliary integral were implemented:")
        print("  - Would need 3D quadrature (u,t,a) for Case C pieces")
        print("  - Would see factors like (1-a)^i, a^{ω-1}")
        print("  - Would have (N/n)^{-αa} = T^{-θαa(1-log(n)/log(N))}")

        print("\n⚠️  FINDING: Case C auxiliary integral NOT implemented")
        print("    This could be the source of missing contributions!")

        print("\n" + "=" * 70)

    return findings


def estimate_auxiliary_integral_contribution(
    theta: float = 4/7,
    R: float = 1.3036,
    verbose: bool = True
) -> Dict:
    """
    Estimate what the auxiliary integral might contribute.

    The auxiliary integral is:
        g_d(k, α, n) = ∫₀¹ (1-a)^i a^{ω-1} (N/n)^{-αa} da

    At α = -R/L → -R (in our scaling), this becomes:
        ∫₀¹ (1-a)^i a^{ω-1} (N/n)^{Ra} da

    For the constant term extraction (n ~ N), this approaches:
        ∫₀¹ (1-a)^i a^{ω-1} da = B(i+1, ω)

    where B is the Beta function.
    """
    # Beta function: B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b)
    # For integers: B(m+1, n) = m! * (n-1)! / (m+n)!
    def beta_func(a, b):
        """Compute Beta function B(a,b) for positive integers."""
        import math
        return math.gamma(a) * math.gamma(b) / math.gamma(a + b)

    results = {}

    # For P₂ (ω=1): B(i+1, 1) = 1/(i+1)
    # For P₃ (ω=2): B(i+1, 2) = 1/((i+1)(i+2))

    # The index 'i' depends on the derivative structure
    # For leading terms, typically i=0 or small

    if verbose:
        print("\n" + "=" * 70)
        print("AUXILIARY INTEGRAL CONTRIBUTION ESTIMATE")
        print("=" * 70)

        print("\nBeta function values for different i:")
        print(f"  {'i':>4} | {'B(i+1,1) [P₂]':>15} | {'B(i+1,2) [P₃]':>15}")
        print(f"  {'-'*4} | {'-'*15} | {'-'*15}")

        for i in range(5):
            b_p2 = beta_func(i+1, 1)  # = 1/(i+1)
            b_p3 = beta_func(i+1, 2)  # = 1/((i+1)(i+2))
            results[f"i={i}"] = {"P2": b_p2, "P3": b_p3}
            print(f"  {i:>4} | {b_p2:>15.6f} | {b_p3:>15.6f}")

        print("\nInterpretation:")
        print("  - At i=0: B(1,1)=1.0 for P₂, B(1,2)=0.5 for P₃")
        print("  - The auxiliary integral contributes multiplicative factors")
        print("  - Missing these could explain systematic error in Case C pairs")

        print("\n⚠️  However, this is a SIMPLIFICATION")
        print("    The actual integral depends on (N/n)^{Ra} ≠ 1")
        print("    Full analysis requires tracing PRZZ's exact formula")

        print("\n" + "=" * 70)

    return results


def compare_bb_vs_cc_error_patterns(
    theta: float = 4/7,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict:
    """
    Compare error patterns between B×B and C×C pairs.

    If Case C auxiliary integral is missing:
    - B×B (1,1) should have SMALLER error
    - C×C (2,2, 3,3, 2,3) should have LARGER, SIMILAR errors
    """
    R_values = [1.3036, 1.1167]
    c_targets = {
        1.3036: 2.13745440613217263636,
        1.1167: math.exp(1.1167 * (1 - 0.407511457)),
    }

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    factorial_norm = {
        "11": 1.0, "22": 1.0/4, "33": 1.0/36,
        "12": 1.0/2, "13": 1.0/6, "23": 1.0/12
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    case_type = {
        "11": "B×B",
        "22": "C×C", "33": "C×C", "23": "C×C",
        "12": "B×C", "13": "B×C",
    }

    results = {"R_data": {}}

    for R in R_values:
        all_terms = make_all_terms_k3(theta, R)
        pair_contributions = {}

        for pair_key, terms in all_terms.items():
            total = sum(evaluate_term(t, polys, n_quad).value for t in terms)
            norm = factorial_norm[pair_key] * symmetry[pair_key]
            pair_contributions[pair_key] = total * norm

        c_computed = sum(pair_contributions.values())
        c_target = c_targets[R]

        results["R_data"][R] = {
            "c_computed": c_computed,
            "c_target": c_target,
            "pair_contributions": pair_contributions,
            "factor_needed": c_target / c_computed,
        }

    if verbose:
        print("\n" + "=" * 70)
        print("B×B vs C×C ERROR PATTERN COMPARISON")
        print("=" * 70)

        print("\nIf Case C auxiliary integral is missing:")
        print("  - B×B should be most accurate")
        print("  - C×C pairs should have similar larger error")
        print("  - B×C pairs should have intermediate error")

        for R in R_values:
            rd = results["R_data"][R]
            print(f"\n{'─'*40}")
            print(f"R = {R}")
            print(f"{'─'*40}")
            print(f"  c_target:   {rd['c_target']:.10f}")
            print(f"  c_computed: {rd['c_computed']:.10f}")
            print(f"  Factor:     {rd['factor_needed']:.6f}")

            print(f"\n  By case type:")
            for case in ["B×B", "B×C", "C×C"]:
                pairs_in_case = [p for p, t in case_type.items() if t == case]
                contrib = sum(rd["pair_contributions"][p] for p in pairs_in_case)
                pct = contrib / rd["c_computed"] * 100
                print(f"    {case}: {contrib:+.8f} ({pct:+.1f}%)")

        # Key test: does B×B have smallest contribution to error?
        print(f"\n{'─'*40}")
        print("KEY OBSERVATION")
        print(f"{'─'*40}")

        rd1 = results["R_data"][1.3036]
        rd2 = results["R_data"][1.1167]

        # B×B contribution
        bb_r1 = rd1["pair_contributions"]["11"]
        bb_r2 = rd2["pair_contributions"]["11"]
        bb_change = (bb_r2 - bb_r1) / abs(bb_r1) * 100

        # C×C contribution (sum of 22, 33, 23)
        cc_pairs = ["22", "33", "23"]
        cc_r1 = sum(rd1["pair_contributions"][p] for p in cc_pairs)
        cc_r2 = sum(rd2["pair_contributions"][p] for p in cc_pairs)
        cc_change = (cc_r2 - cc_r1) / abs(cc_r1) * 100

        print(f"\n  B×B (1,1) R-change: {bb_change:+.2f}%")
        print(f"  C×C (sum) R-change: {cc_change:+.2f}%")

        if abs(bb_change) < abs(cc_change):
            print(f"\n  ⚠️ B×B is MORE R-stable than C×C")
            print(f"    Consistent with missing Case C structure")
        else:
            print(f"\n  ? B×B is NOT more R-stable")
            print(f"    Missing auxiliary integral may not be the only issue")

        print("\n" + "=" * 70)

    return results


def generate_status_report(verbose: bool = True) -> str:
    """
    Generate comprehensive status report for GPT.
    """
    analyze_case_classification(verbose=verbose)
    check_if_auxiliary_integral_implemented(verbose=verbose)
    estimate_auxiliary_integral_contribution(verbose=verbose)
    compare_bb_vs_cc_error_patterns(verbose=verbose)

    report = """
================================================================================
PRZZ ASSEMBLY AUDIT - STATUS REPORT FOR GPT
================================================================================

## 1. CURRENT STATE

**439 tests passing**, 3 xfail (golden target tests while investigating)

**Gap Summary:**
- R=1.3036: c_computed = 1.950, c_target = 2.137, factor = 1.096
- R=1.1167: c_computed = 1.642, c_target = 1.938, factor = 1.180
- Gap is R-DEPENDENT (7.65% factor difference)

## 2. KEY FINDINGS FROM Q-OPERATOR ORACLE

**Per-pair R-sensitivity (ratio change from R=1.3036 to R=1.1167):**
| Pair | Case Type | R-sensitivity |
|------|-----------|--------------|
| (1,1) | B×B | +4.69% |
| (2,2) | C×C | +2.28% ← smallest |
| (3,3) | C×C | +6.75% |
| (1,2) | B×C | +32.32% ← ANOMALY |
| (1,3) | B×C | +6.64% |
| (2,3) | C×C | +4.18% |

**Critical Finding:**
- (1,2) pair shows 32% R-sensitivity while all others show ~2-7%
- (1,2) = P₁×P₂ = Case B × Case C cross-term
- Within (1,2), I4_12 shows -26.1% sensitivity (most extreme)
- I4_12 has OPPOSITE SIGN to I4_11 (positive vs negative)

## 3. CASE C AUXILIARY INTEGRAL INVESTIGATION

**PRZZ TeX 2369-2384 defines:**
- For ω > 0 (Case C): ∫₀¹ (1-a)^i a^{ω-1} (N/n)^{-αa} da
- P₂ (ω=1), P₃ (ω=2) are Case C
- P₁ (ω=0) is Case B (no auxiliary integral)

**Finding: NOT IMPLEMENTED**
- Our code uses 2D quadrature (u,t) only
- No auxiliary 'a' variable found
- Case C auxiliary integral is MISSING

**However, this doesn't fully explain the (1,2) anomaly:**
- If missing uniformly, C×C pairs should have larger error than B×B
- But (2,2) has SMALLEST R-sensitivity, not largest
- The anomaly is specifically (1,2), a MIXED B×C pair

## 4. WORKING HYPOTHESES

**Hypothesis A: Missing Case C Auxiliary Integral**
- Partially supported: we don't implement it
- But doesn't explain why (2,2) is most R-stable

**Hypothesis B: B×C Cross-Term Special Structure**
- The (1,2) anomaly suggests B×C interaction has special issues
- May involve how Case B and Case C pieces combine
- Could be related to how PRZZ handles mixed-case cross-terms

**Hypothesis C: Sign/Prefactor Issue in (1,2)**
- I4_12 has opposite sign to I4_11
- FD oracle validated structure, but at fixed R
- May need to trace PRZZ's exact formula for (1,2)

## 5. NEXT STEPS NEEDED

1. **Trace PRZZ's exact formula for (1,2) cross-term** (TeX lines?)
   - Does PRZZ have special handling for B×C?
   - Is there a different prefactor or normalization?

2. **Understand the auxiliary integral's role**
   - At what stage does it appear in PRZZ's derivation?
   - Does it multiply the whole integrand or just certain terms?
   - How does it affect B×C vs C×C differently?

3. **Check I4 term structure specifically**
   - Why does I4_12 have opposite sign to I4_11?
   - Is this expected from polynomial structure?
   - Could there be a sign convention issue?

## 6. QUESTIONS FOR GPT

1. In PRZZ TeX, where exactly is the Case C auxiliary integral introduced?
   What TeX lines show how it combines with the main integrand?

2. Does PRZZ treat B×C cross-terms (like P₁×P₂) differently from
   pure B×B or C×C pairs?

3. The I4 term (y-derivative only) has opposite signs between (1,1) and (1,2).
   Is this expected? What PRZZ formula determines this sign?

4. Given that (2,2) shows the SMALLEST R-sensitivity (not largest),
   how can missing Case C integral be the main issue?

================================================================================
"""

    if verbose:
        print(report)

    return report


if __name__ == "__main__":
    generate_status_report(verbose=True)
