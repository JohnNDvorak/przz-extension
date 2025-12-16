"""
src/gap_attribution.py
Gap Attribution Diagnostic for PRZZ Assembly Audit

This script computes per-pair I1+I2+I3+I4 contributions to identify
WHERE the ~10% gap between computed c and target c is located.

Decision Matrix (from TRUTH_SPEC.md):
- Gap in P3-involving pairs (13, 23, 33) → Case C / ω>0 structure issue
- Gap evenly spread → Global assembly / mirror term issue
- Gap concentrated in I3/I4 → Prefactor or variable scaling issue

IMPORTANT: This uses mode="main" (NO I₅). I₅ is an error term (PRZZ TeX 1626-1628).
"""

from __future__ import annotations
import math
from typing import Dict

from src.terms_k3_d1 import make_all_terms_k3
from src.evaluate import evaluate_term, evaluate_terms
from src.polynomials import load_przz_polynomials


# PRZZ targets
PRZZ_C_TARGET = 2.13745440613217263636
PRZZ_KAPPA_TARGET = 0.417293962

# Standard config
THETA = 4/7
R = 1.3036


def print_gap_attribution(
    n: int = 60,
    theta: float = THETA,
    R: float = R,
    enforce_Q0: bool = True
) -> Dict[str, float]:
    """
    Print detailed gap attribution report.

    For each pair, prints I1, I2, I3, I4 contributions and total.
    Shows where the ~10% deficit is concentrated.

    Args:
        n: Quadrature points per dimension
        theta: θ parameter
        R: R parameter
        enforce_Q0: Q(0)=1 enforcement mode for polynomials

    Returns:
        Dict with pair totals and computed c
    """
    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=enforce_Q0)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Factorial normalization factors (from PRZZ bracket combinatorics)
    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 1/4
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 1/2
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    }

    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0
    }

    # ω values (for Case identification) - from TRUTH_SPEC.md
    # Our poly index i maps to PRZZ k=i+1, and ω=k-2
    # So P1→ω=0 (Case B), P2→ω=1 (Case C), P3→ω=2 (Case C)
    omega_info = {
        "11": "P1×P1: ω=0,0 (B×B)",
        "22": "P2×P2: ω=1,1 (C×C)",
        "33": "P3×P3: ω=2,2 (C×C)",
        "12": "P1×P2: ω=0,1 (B×C)",
        "13": "P1×P3: ω=0,2 (B×C)",
        "23": "P2×P3: ω=1,2 (C×C)",
    }

    # Get all terms
    all_terms = make_all_terms_k3(theta, R)

    print("\n" + "=" * 80)
    print("GAP ATTRIBUTION DIAGNOSTIC")
    print("=" * 80)
    print(f"\nConfig: θ = {theta:.10f}, R = {R}, n = {n}")
    print(f"Mode: main (NO I₅ - PRZZ TeX 1626-1628: I₅ ≪ T/L)")
    print(f"Target: c = {PRZZ_C_TARGET:.15f}")

    # Store results
    results = {
        "pair_raw": {},
        "pair_normalized": {},
        "per_term": {},
    }

    total_normalized = 0.0

    # Groups for analysis
    case_b_only = []  # P1×P1 only
    case_c_involved = []  # P2 or P3 involved

    print("\n" + "-" * 80)
    print("PER-PAIR BREAKDOWN (I1 + I2 + I3 + I4, NO I5)")
    print("-" * 80)

    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms[pair_key]

        # Evaluate each term
        i_values = {}
        for i, term in enumerate(terms, start=1):
            result = evaluate_term(term, polys, n)
            i_values[f"I{i}"] = result.value
            results["per_term"][f"I{i}_{pair_key}"] = result.value

        # Raw pair total
        raw_total = sum(i_values.values())
        results["pair_raw"][pair_key] = raw_total

        # Normalized (with factorial and symmetry)
        norm = factorial_norm[pair_key]
        sym = symmetry_factor[pair_key]
        normalized = raw_total * norm * sym
        results["pair_normalized"][pair_key] = normalized
        total_normalized += normalized

        # Group for analysis
        if pair_key == "11":
            case_b_only.append(pair_key)
        else:
            case_c_involved.append(pair_key)

        # Print
        print(f"\nPair ({pair_key[0]},{pair_key[1]}) [{omega_info[pair_key]}]:")
        print(f"  I1 = {i_values['I1']:+18.12f}")
        print(f"  I2 = {i_values['I2']:+18.12f}")
        print(f"  I3 = {i_values['I3']:+18.12f}")
        print(f"  I4 = {i_values['I4']:+18.12f}")
        print(f"  ─────────────────────────────")
        print(f"  Raw total:   {raw_total:+18.12f}")
        print(f"  Norm factor: {norm * sym:.6f} (1/{int(1/norm)}{'×2' if sym==2 else ''})")
        print(f"  Normalized:  {normalized:+18.12f}")

    # Summary
    c_computed = total_normalized
    kappa_computed = 1 - math.log(c_computed) / R
    delta_c = c_computed - PRZZ_C_TARGET
    delta_c_pct = delta_c / PRZZ_C_TARGET * 100

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n  c_computed:   {c_computed:20.15f}")
    print(f"  c_target:     {PRZZ_C_TARGET:20.15f}")
    print(f"  Δc:           {delta_c:+20.15f}")
    print(f"  Δc/c_target:  {delta_c_pct:+.4f}%")

    print(f"\n  κ_computed:   {kappa_computed:20.15f}")
    print(f"  κ_target:     {PRZZ_KAPPA_TARGET:20.15f}")

    # Gap analysis
    print("\n" + "-" * 80)
    print("GAP ANALYSIS")
    print("-" * 80)

    # Compute contribution percentages
    print("\n  Contribution breakdown (normalized):")
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        contrib = results["pair_normalized"][pair_key]
        pct = contrib / c_computed * 100
        print(f"    {pair_key}: {contrib:+18.12f} ({pct:6.2f}%)")

    # Case B only vs Case C involved
    case_b_sum = sum(results["pair_normalized"][p] for p in case_b_only)
    case_c_sum = sum(results["pair_normalized"][p] for p in case_c_involved)

    print(f"\n  Case B only (P1×P1):    {case_b_sum:+18.12f} ({case_b_sum/c_computed*100:.2f}%)")
    print(f"  Case C involved:        {case_c_sum:+18.12f} ({case_c_sum/c_computed*100:.2f}%)")

    # Diagnosis
    print("\n" + "-" * 80)
    print("DIAGNOSIS")
    print("-" * 80)

    if delta_c_pct < -5:
        print(f"\n  ⚠️  SIGNIFICANT DEFICIT: {delta_c_pct:.2f}%")
        print("  ")
        print("  Check decision matrix:")
        print("  - If P3-involving pairs have larger relative deficit → Case C missing")
        print("  - If deficit evenly spread → Global assembly / mirror term issue")
        print("  - If deficit in I3/I4 specifically → Prefactor or variable scaling")

        # More specific check
        p3_pairs = ["13", "23", "33"]
        p3_sum = sum(results["pair_normalized"][p] for p in p3_pairs)
        non_p3_sum = c_computed - p3_sum

        print(f"\n  P3-involving pairs sum: {p3_sum:.12f}")
        print(f"  Non-P3 pairs sum:       {non_p3_sum:.12f}")

    elif delta_c_pct > 5:
        print(f"\n  ⚠️  SIGNIFICANT EXCESS: {delta_c_pct:.2f}%")
        print("  Something is being over-counted.")
    else:
        print(f"\n  ✓ Within tolerance: {delta_c_pct:.2f}%")

    # I3+I4 vs I1+I2 check
    print("\n  I3+I4 vs I1+I2 ratio check:")
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        i12 = results["per_term"][f"I1_{pair_key}"] + results["per_term"][f"I2_{pair_key}"]
        i34 = results["per_term"][f"I3_{pair_key}"] + results["per_term"][f"I4_{pair_key}"]
        if abs(i12) > 1e-15:
            ratio = i34 / i12
            print(f"    {pair_key}: I3+I4 / I1+I2 = {ratio:+.6f}")
        else:
            print(f"    {pair_key}: I1+I2 ≈ 0, ratio undefined")

    print("\n" + "=" * 80)

    results["c_computed"] = c_computed
    results["c_target"] = PRZZ_C_TARGET
    results["delta_c"] = delta_c
    results["delta_c_pct"] = delta_c_pct

    return results


def quick_gap_check(n: int = 60) -> float:
    """
    Quick gap check - returns delta_c percentage.

    For scripts that just need the number.
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    all_terms = make_all_terms_k3(THETA, R)

    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1/36,
        "12": 0.5, "13": 1/6, "23": 1/12
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0
    }

    total = 0.0
    for pair_key, terms in all_terms.items():
        pair_result = evaluate_terms(terms, polys, n, return_breakdown=False)
        total += pair_result.total * factorial_norm[pair_key] * symmetry_factor[pair_key]

    delta_pct = (total - PRZZ_C_TARGET) / PRZZ_C_TARGET * 100
    return delta_pct


if __name__ == "__main__":
    results = print_gap_attribution(n=60)
