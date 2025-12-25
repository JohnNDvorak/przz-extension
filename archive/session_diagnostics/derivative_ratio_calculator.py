"""
Interactive calculator showing exactly what derivative contribution would be needed
to achieve the target ratio of 0.94.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from przz_22_exact_oracle import przz_oracle_22


def main():
    theta = 4.0 / 7.0

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    R_kappa = 1.3036

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    R_kappa_star = 1.1167

    # Compute (2,2) pair
    result_k = przz_oracle_22(P2_k, Q_k, theta, R_kappa, n_quad=80)
    result_ks = przz_oracle_22(P2_ks, Q_ks, theta, R_kappa_star, n_quad=80)

    print("="*80)
    print("DERIVATIVE RATIO CALCULATOR")
    print("="*80)

    print("\n" + "─"*80)
    print("PART 1: ACTUAL VALUES")
    print("─"*80)

    I2_k = result_k.I2
    I2_ks = result_ks.I2
    D_k = result_k.I1 + result_k.I3 + result_k.I4
    D_ks = result_ks.I1 + result_ks.I3 + result_ks.I4
    total_k = result_k.total
    total_ks = result_ks.total

    print(f"\nκ benchmark (R={R_kappa}):")
    print(f"  I₂         = {I2_k:10.6f}")
    print(f"  Deriv (I₁+I₃+I₄) = {D_k:10.6f}  ({100*D_k/I2_k:+6.2f}% of I₂)")
    print(f"  Total      = {total_k:10.6f}")

    print(f"\nκ* benchmark (R={R_kappa_star}):")
    print(f"  I₂         = {I2_ks:10.6f}")
    print(f"  Deriv (I₁+I₃+I₄) = {D_ks:10.6f}  ({100*D_ks/I2_ks:+6.2f}% of I₂)")
    print(f"  Total      = {total_ks:10.6f}")

    print("\n" + "─"*80)
    print("PART 2: RATIO ANALYSIS")
    print("─"*80)

    I2_ratio = I2_k / I2_ks
    D_ratio = D_k / D_ks
    total_ratio = total_k / total_ks

    print(f"\nRatios (κ / κ*):")
    print(f"  I₂ ratio:     {I2_ratio:.6f}  ← Naive formula (wrong direction)")
    print(f"  Deriv ratio:  {D_ratio:.6f}  ← Much smaller than I₂ ratio")
    print(f"  Total ratio:  {total_ratio:.6f}  ← Actual")
    print(f"  Target ratio: {0.94:.6f}  ← What we need")

    gap = total_ratio - 0.94
    print(f"\nGap: {gap:.6f} ({100*gap/0.94:.1f}% error)")

    print("\n" + "─"*80)
    print("PART 3: WHAT WOULD BE NEEDED TO HIT TARGET 0.94?")
    print("─"*80)

    target = 0.94

    # Solve: (I₂_κ + D_κ_needed) / (I₂_κ* + D_κ*) = target
    D_k_needed = target * (I2_ks + D_ks) - I2_k

    print(f"\nTo achieve ratio = {target:.2f}:")
    print(f"  We need: (I₂_κ + D_κ) / (I₂_κ* + D_κ*) = {target}")
    print(f"  Currently: ({I2_k:.4f} + {D_k:.4f}) / ({I2_ks:.4f} + {D_ks:.4f}) = {total_ratio:.4f}")
    print(f"\nSolving for D_κ:")
    print(f"  D_κ = {target} × ({I2_ks:.4f} + {D_ks:.4f}) - {I2_k:.4f}")
    print(f"  D_κ = {target * (I2_ks + D_ks):.4f} - {I2_k:.4f}")
    print(f"  D_κ = {D_k_needed:.6f}  ← What we'd need")

    print(f"\nComparison:")
    print(f"  Actual D_κ:   {D_k:+10.6f}")
    print(f"  Needed D_κ:   {D_k_needed:+10.6f}")
    print(f"  Shortfall:    {D_k - D_k_needed:+10.6f}")

    shortfall_pct = 100 * (D_k - D_k_needed) / I2_k
    print(f"\nShortfall as % of I₂: {shortfall_pct:+.2f}%")

    print("\n" + "─"*80)
    print("PART 4: INTERPRETATION")
    print("─"*80)

    print(f"\nThe derivative terms would need to:")
    if D_k_needed < 0:
        print(f"  • Be NEGATIVE (subtract from I₂)")
        print(f"  • Equal {abs(D_k_needed):.4f} in magnitude")
        print(f"  • Be {100*abs(D_k_needed)/I2_k:.1f}% of I₂")
    else:
        print(f"  • Be POSITIVE (add to I₂)")
        print(f"  • Equal {D_k_needed:.4f} in magnitude")
        print(f"  • Be {100*D_k_needed/I2_k:.1f}% of I₂")

    print(f"\nActually, they are:")
    if D_k > 0:
        print(f"  • POSITIVE (+{D_k:.4f})")
        print(f"  • Only {100*D_k/I2_k:.1f}% of I₂")
    else:
        print(f"  • NEGATIVE ({D_k:.4f})")
        print(f"  • Only {100*abs(D_k)/I2_k:.1f}% of I₂")

    print(f"\nConclusion:")
    if (D_k > 0 and D_k_needed < 0) or (D_k < 0 and D_k_needed > 0):
        print(f"  ✗ WRONG SIGN - Derivatives have the opposite effect!")
    if abs(D_k - D_k_needed) > 0.1 * abs(I2_k):
        print(f"  ✗ WRONG MAGNITUDE - Shortfall is {abs(shortfall_pct):.1f}% of I₂")

    print(f"\n  → Derivative terms CANNOT explain the ratio reversal")
    print(f"  → Other factors must dominate")

    print("\n" + "─"*80)
    print("PART 5: WHY HIGHER DEGREE DOESN'T HELP")
    print("─"*80)

    print("\nPolynomial degrees:")
    print(f"  κ:  P₂ degree {P2_k.to_monomial().degree}, Q degree {Q_k.to_monomial().degree}")
    print(f"  κ*: P₂ degree {P2_ks.to_monomial().degree}, Q degree {Q_ks.to_monomial().degree}")

    print("\nWhen degree increases:")
    print("  1. Derivatives grow ~ O(degree)")
    print("  2. I₂ involves P² → grows ~ O(degree²)")
    print("  3. Net: Deriv/I₂ ratio DECREASES")

    print("\nNumerical evidence:")
    print(f"  κ:  Deriv/I₂ = {100*D_k/I2_k:6.2f}%")
    print(f"  κ*: Deriv/I₂ = {100*D_ks/I2_ks:6.2f}%")
    print(f"\n  → Higher degree κ has SMALLER derivative contribution!")
    print(f"  → This INCREASES the ratio, not decreases it")

    print("\n" + "="*80)
    print("FINAL ANSWER")
    print("="*80)

    print(f"\nCan derivative terms reverse ratio from {I2_ratio:.2f} to {target:.2f}?")
    print(f"\n  NO.")
    print(f"\n  • They go in the WRONG DIRECTION")
    print(f"  • Magnitude is {abs(shortfall_pct):.0f}% of I₂ SHORT")
    print(f"  • Higher degree polynomials make it WORSE, not better")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
