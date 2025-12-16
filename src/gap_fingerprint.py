"""
src/gap_fingerprint.py
Structural Fingerprint of the ~8.77% Gap

Purpose: Determine if the gap is:
  1. A GLOBAL multiplicative factor (all pairs off by same ratio), or
  2. LOCALIZED to certain pair families (ω>0, Case C, etc.)

This diagnostic answers: where should we look next?

Decision Matrix:
- If global factor → mirror combination / analytic extraction mismatch
- If localized to ω>0 pairs → Case C auxiliary integral missing
- If localized to derivative terms → prefactor / variable scaling issue
"""

from __future__ import annotations
import math
from typing import Dict
from dataclasses import dataclass

from src.polynomials import load_przz_polynomials
from src.evaluate import evaluate_term
from src.terms_k3_d1 import make_all_terms_k3


@dataclass
class PairContribution:
    """Contribution from a single pair."""
    pair: str
    ell1: int
    ell2: int
    omega_left: int   # ω for left piece (ℓ₁ - 1)
    omega_right: int  # ω for right piece (ℓ₂ - 1)
    case_left: str    # "B" or "C"
    case_right: str   # "B" or "C"
    i1: float
    i2: float
    i3: float
    i4: float
    raw_total: float
    normalized: float
    weight: float


def compute_gap_fingerprint(
    theta: float = 4/7,
    R: float = 1.3036,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict:
    """
    Compute structural fingerprint of the gap.

    Returns breakdown by:
    - ω-case classification
    - Term type (I1 vs I2 vs I3/I4)
    - Individual pairs
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    all_terms = make_all_terms_k3(theta, R)

    # Factorial normalization
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

    # Pair metadata
    pair_meta = {
        "11": (1, 1),
        "22": (2, 2),
        "33": (3, 3),
        "12": (1, 2),
        "13": (1, 3),
        "23": (2, 3),
    }

    contributions = []

    for pair_key, terms in all_terms.items():
        ell1, ell2 = pair_meta[pair_key]
        omega_left = ell1 - 1   # For d=1: ω = ℓ - 1
        omega_right = ell2 - 1
        case_left = "B" if omega_left == 0 else "C"
        case_right = "B" if omega_right == 0 else "C"

        norm = symmetry_factor[pair_key] * factorial_norm[pair_key]

        i1 = evaluate_term(terms[0], polys, n_quad).value
        i2 = evaluate_term(terms[1], polys, n_quad).value
        i3 = evaluate_term(terms[2], polys, n_quad).value
        i4 = evaluate_term(terms[3], polys, n_quad).value

        raw_total = i1 + i2 + i3 + i4
        normalized = raw_total * norm

        contributions.append(PairContribution(
            pair=pair_key,
            ell1=ell1,
            ell2=ell2,
            omega_left=omega_left,
            omega_right=omega_right,
            case_left=case_left,
            case_right=case_right,
            i1=i1,
            i2=i2,
            i3=i3,
            i4=i4,
            raw_total=raw_total,
            normalized=normalized,
            weight=norm,
        ))

    # Aggregate by category
    c_computed = sum(c.normalized for c in contributions)
    c_target = 2.13745440613217263636

    # Category 1: By ω-case
    case_BB = sum(c.normalized for c in contributions if c.case_left == "B" and c.case_right == "B")
    case_BC = sum(c.normalized for c in contributions if (c.case_left == "B") != (c.case_right == "B"))
    case_CC = sum(c.normalized for c in contributions if c.case_left == "C" and c.case_right == "C")

    # Category 2: By term type
    i1_total = sum(c.i1 * c.weight for c in contributions)
    i2_total = sum(c.i2 * c.weight for c in contributions)
    i3_total = sum(c.i3 * c.weight for c in contributions)
    i4_total = sum(c.i4 * c.weight for c in contributions)

    # Compute what factor would be needed for each category
    global_factor = c_target / c_computed

    results = {
        "c_computed": c_computed,
        "c_target": c_target,
        "gap_percent": (c_target - c_computed) / c_target * 100,
        "global_factor_needed": global_factor,
        "theta_over_6_factor": 1 + theta/6,
        "case_BB": case_BB,
        "case_BC": case_BC,
        "case_CC": case_CC,
        "i1_total": i1_total,
        "i2_total": i2_total,
        "i3_total": i3_total,
        "i4_total": i4_total,
        "contributions": contributions,
    }

    if verbose:
        print("\n" + "=" * 80)
        print("GAP FINGERPRINT: Is the deficit GLOBAL or LOCALIZED?")
        print("=" * 80)

        print(f"\n{'='*40}")
        print("OVERALL GAP")
        print(f"{'='*40}")
        print(f"  c_computed:  {c_computed:.10f}")
        print(f"  c_target:    {c_target:.10f}")
        print(f"  Gap:         {(c_target - c_computed):.10f} ({results['gap_percent']:.2f}%)")
        print(f"  Factor needed: {global_factor:.10f}")
        print(f"  (1 + θ/6):     {1 + theta/6:.10f}")
        print(f"  Match:       {abs(global_factor - (1 + theta/6))/global_factor*100:.4f}% difference")

        print(f"\n{'='*40}")
        print("BY ω-CASE CLASSIFICATION")
        print(f"{'='*40}")
        print(f"  Case B×B (ω=0,0) [P₁×P₁]:  {case_BB:+.10f} ({case_BB/c_computed*100:.1f}%)")
        print(f"  Case B×C (ω=0,>0) [P₁×P₂/₃]: {case_BC:+.10f} ({case_BC/c_computed*100:.1f}%)")
        print(f"  Case C×C (ω>0,>0) [P₂/₃×P₂/₃]: {case_CC:+.10f} ({case_CC/c_computed*100:.1f}%)")

        # Check if factor is uniform across categories
        print(f"\n  If factor {global_factor:.6f} applied uniformly:")
        print(f"    B×B → {case_BB * global_factor:.6f}")
        print(f"    B×C → {case_BC * global_factor:.6f}")
        print(f"    C×C → {case_CC * global_factor:.6f}")

        print(f"\n{'='*40}")
        print("BY TERM TYPE (I₁, I₂, I₃, I₄)")
        print(f"{'='*40}")
        print(f"  I₁ (coupled, all derivs):    {i1_total:+.10f} ({i1_total/c_computed*100:.1f}%)")
        print(f"  I₂ (decoupled, no derivs):   {i2_total:+.10f} ({i2_total/c_computed*100:.1f}%)")
        print(f"  I₃ (x derivs only):          {i3_total:+.10f} ({i3_total/c_computed*100:.1f}%)")
        print(f"  I₄ (y derivs only):          {i4_total:+.10f} ({i4_total/c_computed*100:.1f}%)")

        print(f"\n{'='*40}")
        print("INDIVIDUAL PAIR RATIOS")
        print(f"{'='*40}")
        print(f"  If gap is GLOBAL: all pairs should need ~same factor")
        print(f"  If gap is LOCALIZED: some pairs need larger factors")
        print()

        for c in contributions:
            pair_factor = c_target * (c.normalized / c_computed) / c.normalized if abs(c.normalized) > 1e-10 else float('nan')
            # Actually this doesn't make sense for negative contributions
            # Let's just show the contribution
            case_str = f"{c.case_left}×{c.case_right}"
            print(f"  ({c.ell1},{c.ell2}) [{case_str}]: {c.normalized:+.10f} ({c.normalized/c_computed*100:+6.2f}%)")

        print(f"\n{'='*40}")
        print("DIAGNOSIS")
        print(f"{'='*40}")

        # Check if gap looks global
        # A global factor would scale everything equally
        # So c_target = c_computed * factor means:
        # - positive contributions get bigger
        # - negative contributions get more negative

        # Key test: if we apply factor to each term type, do proportions stay same?
        i1_scaled = i1_total * global_factor
        i2_scaled = i2_total * global_factor
        i3_scaled = i3_total * global_factor
        i4_scaled = i4_total * global_factor

        c_scaled = i1_scaled + i2_scaled + i3_scaled + i4_scaled

        print(f"\n  Test: Apply {global_factor:.6f} uniformly to all term types:")
        print(f"    I₁×factor = {i1_scaled:.6f}")
        print(f"    I₂×factor = {i2_scaled:.6f}")
        print(f"    I₃×factor = {i3_scaled:.6f}")
        print(f"    I₄×factor = {i4_scaled:.6f}")
        print(f"    Sum = {c_scaled:.6f}")
        print(f"    Target = {c_target:.6f}")
        print(f"    Match: {'YES' if abs(c_scaled - c_target) < 0.001 else 'NO'}")

        print(f"\n  CONCLUSION:")
        if abs(global_factor - (1 + theta/6)) / global_factor < 0.01:
            print(f"  ✓ Gap is consistent with GLOBAL factor (1 + θ/6)")
            print(f"    → Look for missing log(N^{{x+y}}T) factor in mirror combination")
            print(f"    → Or missing normalization in overall assembly")
        else:
            print(f"  ? Gap may be LOCALIZED - need more investigation")
            print(f"    → Check Case C auxiliary integral")
            print(f"    → Check ω-dependent terms")

        print("=" * 80)

    return results


if __name__ == "__main__":
    compute_gap_fingerprint(verbose=True)
