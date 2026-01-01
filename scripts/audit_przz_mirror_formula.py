#!/usr/bin/env python3
"""
Phase 59.3: PRZZ Mirror Formula Audit
=====================================

The central paradox:
- PRZZ TeX says: c = S12(+) + exp(2R)×S12(mirror) + S34
- This gives κ = -0.67 (nonsense)
- But PRZZ reports κ = 0.417

Either:
1. Our S12(±R) ≠ PRZZ's integrals
2. The TeX formula doesn't match PRZZ's actual computation
3. There's a normalization we're missing

This script tests various hypotheses.

Created: 2025-12-29
"""

import sys
import math
sys.path.insert(0, "/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension")

from src.kappa_engine import KappaEngine


def audit_przz_mirror_formula():
    """
    Test different mirror formulas against PRZZ target.
    """
    print("=" * 70)
    print("PHASE 59.3: PRZZ MIRROR FORMULA AUDIT")
    print("=" * 70)

    for benchmark, R, c_target, kappa_target, engine_fn in [
        ("kappa", 1.3036, 2.1375, 0.417293962, KappaEngine.from_przz_kappa),
        ("kappa_star", 1.1167, 1.938, 0.327833316, KappaEngine.from_przz_kappa_star),
    ]:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {benchmark.upper()} (R={R})")
        print(f"{'='*60}")

        engine = engine_fn(n_quad=80)
        integrals = engine.compute_integrals()

        S12_plus = integrals.S12_plus
        S12_minus = integrals.S12_minus
        S34_plus = integrals.S34_plus

        print(f"\n  Our computed integrals:")
        print(f"    S12(+R)  = {S12_plus:.10f}")
        print(f"    S12(-R)  = {S12_minus:.10f}")
        print(f"    S34(+R)  = {S34_plus:.10f}")

        # Various mirror weights to test
        exp_R = math.exp(R)
        exp_2R = math.exp(2*R)
        m_ours = exp_R + 5

        formulas = [
            ("PRZZ TeX (exp(2R))", exp_2R),
            ("Our formula (exp(R)+5)", m_ours),
            ("exp(R) alone", exp_R),
            ("5 alone", 5.0),
            ("1/exp(R)", 1/exp_R),
            ("exp(2R) - exp(R)", exp_2R - exp_R),
            ("exp(R) + exp(R)", 2*exp_R),
        ]

        print(f"\n  Testing mirror formulas:")
        print(f"  {'Formula':<25} | {'m value':>10} | {'c':>10} | {'κ':>10} | {'Gap':>10}")
        print(f"  {'-'*25}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}")

        for name, m in formulas:
            c = S12_plus + m * S12_minus + S34_plus
            if c > 0:
                kappa = 1 - math.log(c) / R
                gap = (kappa - kappa_target) / kappa_target * 100
            else:
                kappa = float('nan')
                gap = float('nan')

            print(f"  {name:<25} | {m:>10.4f} | {c:>10.4f} | {kappa:>10.4f} | {gap:>+10.2f}%")

        # What m would be needed?
        m_needed = (c_target - S12_plus - S34_plus) / S12_minus
        c_check = S12_plus + m_needed * S12_minus + S34_plus
        kappa_check = 1 - math.log(c_check) / R

        print(f"\n  Reverse engineering:")
        print(f"    m_needed to hit c_target = {m_needed:.6f}")
        print(f"    c_check = {c_check:.6f} (should be {c_target:.6f})")
        print(f"    κ_check = {kappa_check:.6f} (should be {kappa_target:.6f})")

        print(f"\n  Comparing m values:")
        print(f"    m_needed    = {m_needed:.6f}")
        print(f"    exp(2R)     = {exp_2R:.6f}  (ratio to needed: {exp_2R/m_needed:.4f})")
        print(f"    exp(R)+5    = {m_ours:.6f}  (ratio to needed: {m_ours/m_needed:.4f})")
        print(f"    exp(R)      = {exp_R:.6f}  (ratio to needed: {exp_R/m_needed:.4f})")

        # The critical question: what's the relationship?
        print(f"\n  Relationships:")
        print(f"    exp(2R) / m_needed = {exp_2R/m_needed:.6f}")
        print(f"    exp(2R) - m_needed = {exp_2R - m_needed:.6f}")
        print(f"    If PRZZ uses exp(2R), what correction factor F makes it work?")

        # If PRZZ's formula is: c = S12(+) + exp(2R) × F × S12(mirror) + S34
        # Then F = m_needed / exp(2R)
        F = m_needed / exp_2R
        print(f"    F = m_needed / exp(2R) = {F:.6f}")

        # Is F related to something we know?
        print(f"\n  Is F related to known quantities?")
        print(f"    F × exp(R) = {F * exp_R:.6f}")
        print(f"    F × exp(2R) = {F * exp_2R:.6f}")
        print(f"    1/F = {1/F:.6f}")
        print(f"    exp(-R) / F = {math.exp(-R) / F:.6f}")
        print(f"    F + 1 = {F + 1:.6f}")
        print(f"    1/(1-F) = {1/(1-F) if F < 1 else 'undefined'}")

        # Check if our S12(-R) needs rescaling
        print(f"\n  Does S12(-R) need rescaling?")
        print(f"    If PRZZ's mirror = exp(2R) × S12_przz_mirror")
        print(f"    And ours = m × S12_ours_minus")
        print(f"    Then S12_przz_mirror / S12_ours_minus = m / exp(2R) = {m_needed/exp_2R:.6f}")

        # Hypothesis: Our S12(-R) is exp(R) times PRZZ's mirror integral
        # Then PRZZ's exp(2R) × PRZZ_mirror = exp(2R) × (our_minus / exp(R)) = exp(R) × our_minus
        exp_R_times_minus = exp_R * S12_minus
        c_hypothesis = S12_plus + exp_R_times_minus + S34_plus
        kappa_hypothesis = 1 - math.log(c_hypothesis) / R if c_hypothesis > 0 else float('nan')
        print(f"\n  Hypothesis: Our S12(-R) = exp(R) × PRZZ's mirror")
        print(f"    c = S12(+) + exp(R)×S12(-) + S34 = {c_hypothesis:.6f}")
        print(f"    κ = {kappa_hypothesis:.6f}")
        print(f"    Gap from target: {(kappa_hypothesis - kappa_target)/kappa_target*100:+.2f}%")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  The PRZZ TeX formula c = S12(+) + exp(2R)×S12(-) + S34 gives wrong results.

  Possible explanations:

  1. OUR S12(-R) ≠ PRZZ's MIRROR INTEGRAL
     - If our S12(-R) is scaled differently, the weights would differ
     - The ratio F = m_needed / exp(2R) ≈ 0.65 suggests a factor of ~exp(-R)

  2. PRZZ DOESN'T ACTUALLY USE exp(2R)
     - The TeX shows the algebraic form, but numerical implementation may differ
     - PRZZ might use a different normalization in practice

  3. THE DQ IDENTITY TRANSFORMS THE WEIGHTS
     - The DQ identity: [Direct - exp(2R)×Mirror] / (-2Rθ) = bracket
     - Maybe PRZZ extracts c differently from the bracket

  4. WE'RE MISSING A NORMALIZATION FACTOR
     - Some L, θ, or R-dependent factor in the integral definitions
""")


if __name__ == "__main__":
    audit_przz_mirror_formula()
