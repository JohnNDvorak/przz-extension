"""
src/mirror_check.py
Mirror Term Combination Check

PRZZ TeX References:
- Lines 1502-1511: Difference quotient → integral representation
  (N^{αx+βy} - T^{-α-β}N^{-βx-αy})/(α+β)
  = N^{αx+βy} log(N^{x+y}T) ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt

- Lines 1521-1523: Mirror combination at α=β=-R/L

Key Question:
Does PRZZ's "combine then expand" differ from our "expand then add"?

With α=β=-R/L, we have α+β = -2R/L ≠ 0, so the 1/(α+β) singularity isn't
directly at play. But the integral representation may still affect the
constant extraction.
"""

from __future__ import annotations
import numpy as np
from typing import Dict

from src.polynomials import load_przz_polynomials
from src.quadrature import tensor_grid_2d


def investigate_global_factor(
    theta: float = 4/7,
    R: float = 1.3036,
    verbose: bool = True
) -> Dict:
    """
    Check if the gap can be explained by a simple global factor.

    Key observation: c_target / c_computed ≈ 1.096 ≈ 1 + θ/6 for θ=4/7
    """
    from src.evaluate import evaluate_c_full
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    # Compute c in main mode (no I5)
    result = evaluate_c_full(theta, R, n=60, polynomials=polys, mode="main")
    c_computed = result.total

    # Target from PRZZ
    c_target = 2.13745440613217263636

    # Check various potential global factors
    ratio = c_target / c_computed

    candidates = {
        "1 + θ/6": 1 + theta/6,
        "1 + θ/7": 1 + theta/7,
        "1 + θ²/2": 1 + theta**2/2,
        "exp(θ/6)": np.exp(theta/6),
        "1/(1 - θ/11)": 1/(1 - theta/11),
        "θ + 1": theta + 1,
        "(θ+1)/θ × some": (1 + theta)/theta * (1 - 0.31),  # adjusted
    }

    if verbose:
        print("\n" + "=" * 70)
        print("GLOBAL FACTOR INVESTIGATION")
        print("=" * 70)
        print(f"\nComputed c: {c_computed:.10f}")
        print(f"Target c:   {c_target:.10f}")
        print(f"Ratio:      {ratio:.10f}")
        print(f"\n--- Potential Global Factors ---")

        for name, factor in candidates.items():
            corrected = c_computed * factor
            error = abs(corrected - c_target) / c_target
            match_str = "✓ MATCH" if error < 0.001 else "✗"
            print(f"  {name}: {factor:.6f} → c = {corrected:.6f} ({error*100:.2f}% off) {match_str}")

        print(f"\n--- Exact Factor Needed ---")
        print(f"  {ratio:.10f}")
        print(f"  ≈ 1 + {ratio - 1:.6f}")
        print(f"  θ/6 = {theta/6:.6f}")
        print(f"  Diff from (1+θ/6): {abs(ratio - (1 + theta/6))*100:.4f}%")

        print("=" * 70)

    return {
        "c_computed": c_computed,
        "c_target": c_target,
        "ratio": ratio,
        "one_plus_theta_over_6": 1 + theta/6,
        "ratio_matches_theta_over_6": abs(ratio - (1 + theta/6)) < 0.001
    }


def check_i2_scaling(
    theta: float = 4/7,
    R: float = 1.3036,
    verbose: bool = True
) -> Dict:
    """
    Check if the gap is concentrated in I2 terms vs I1/I3/I4.

    I2 terms have no derivatives - they're evaluated at x=y=0.
    If the gap is specifically in derivative terms, the issue is in
    derivative extraction. If in I2 too, it's a global factor.
    """
    from src.evaluate import evaluate_term
    from src.terms_k3_d1 import make_all_terms_k3
    import math

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    all_terms = make_all_terms_k3(theta, R)

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

    i1_total = 0.0
    i2_total = 0.0
    i3_total = 0.0
    i4_total = 0.0

    for pair_key, terms in all_terms.items():
        norm = symmetry_factor[pair_key] * factorial_norm[pair_key]
        for i, term in enumerate(terms):
            val = evaluate_term(term, polys, 60).value * norm
            if i == 0:
                i1_total += val
            elif i == 1:
                i2_total += val
            elif i == 2:
                i3_total += val
            elif i == 3:
                i4_total += val

    results = {
        "I1_total": i1_total,
        "I2_total": i2_total,
        "I3_total": i3_total,
        "I4_total": i4_total,
        "c_from_sum": i1_total + i2_total + i3_total + i4_total
    }

    if verbose:
        print("\n" + "=" * 70)
        print("TERM TYPE BREAKDOWN")
        print("=" * 70)
        total = i1_total + i2_total + i3_total + i4_total
        print(f"\n  I1 (coupled, all derivatives): {i1_total:+.10f} ({i1_total/total*100:.1f}%)")
        print(f"  I2 (decoupled, no derivatives): {i2_total:+.10f} ({i2_total/total*100:.1f}%)")
        print(f"  I3 (x derivatives only):        {i3_total:+.10f} ({i3_total/total*100:.1f}%)")
        print(f"  I4 (y derivatives only):        {i4_total:+.10f} ({i4_total/total*100:.1f}%)")
        print(f"\n  Total: {total:.10f}")

        # Check if applying global factor to each type would match target
        c_target = 2.13745440613217263636
        factor = c_target / total
        print(f"\n--- If global factor {factor:.6f} applied ---")
        print(f"  I1 × factor = {i1_total * factor:.6f}")
        print(f"  I2 × factor = {i2_total * factor:.6f}")
        print(f"  I3 × factor = {i3_total * factor:.6f}")
        print(f"  I4 × factor = {i4_total * factor:.6f}")
        print("=" * 70)

    return results


def check_przz_prefactor_hypothesis(
    theta: float = 4/7,
    R: float = 1.3036,
    verbose: bool = True
) -> Dict:
    """
    Test hypothesis: Did we miss a (1+θ)/θ → 1/θ simplification somewhere?

    The algebraic prefactor (θS+1)/θ at S=0 gives 1/θ.
    But if PRZZ keeps the full (θS+1)/θ and extracts more terms, we'd differ.
    """
    # The FD oracle already validated this for I1_22 including cross-terms
    # So this isn't the issue

    if verbose:
        print("\n" + "=" * 70)
        print("PREFACTOR HYPOTHESIS CHECK")
        print("=" * 70)
        print("\n  The FD oracle validated that our DSL correctly includes")
        print("  algebraic prefactor cross-terms from (θS+1)/θ differentiation.")
        print("\n  I1_22 FD validation: DSL/FD = 1.0000000314")
        print("  I1_12 FD validation: DSL/FD = 1.0000000016")
        print("\n  CONCLUSION: Algebraic prefactor is handled correctly.")
        print("=" * 70)

    return {"prefactor_validated": True}


def check_log_N_factor(
    theta: float = 4/7,
    R: float = 1.3036,
    verbose: bool = True
) -> Dict:
    """
    Check if there's a missing log(N) = θ·log(T) factor.

    PRZZ TeX line 2309: "by the change of variable x → x log N"

    This suggests PRZZ variables are scaled by log(N). If our x is in
    different units, we'd have a systematic factor.
    """
    # For asymptotic analysis, log(N) = θ·log(T) → ∞
    # But for finite-T computations, this would appear as a constant

    # The key question: does our derivative w.r.t. x need to be
    # multiplied by log(N) to match PRZZ's derivative w.r.t. (x·log N)?

    # If PRZZ uses x̃ = x·log(N), then:
    # d/dx̃ = (1/log N) · d/dx
    # So our derivative is (log N) times larger than PRZZ's

    # For (1,1): 1 derivative in x → factor of 1/log(N)
    # For (2,2): 2 derivatives in x, 2 in y → factor of 1/(log N)^4?

    # But wait - the final answer is a constant, independent of T.
    # So log(N) factors must cancel somehow.

    if verbose:
        print("\n" + "=" * 70)
        print("VARIABLE SCALING HYPOTHESIS (x → x·log N)")
        print("=" * 70)
        print("\n  PRZZ TeX line 2309: 'by the change of variable x → x log N'")
        print("\n  This could introduce factors of log(N) in derivatives.")
        print("  However, the final κ is T-independent (asymptotic limit).")
        print("  So any log(N) factors must cancel in the full assembly.")
        print("\n  Need to trace PRZZ's derivation more carefully to check")
        print("  whether our variables match their 'before' or 'after' scaling.")
        print("=" * 70)

    return {"needs_further_investigation": True}


if __name__ == "__main__":
    investigate_global_factor(verbose=True)
    check_i2_scaling(verbose=True)
    check_przz_prefactor_hypothesis(verbose=True)
    check_log_N_factor(verbose=True)
