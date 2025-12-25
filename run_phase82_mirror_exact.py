#!/usr/bin/env python3
"""
Phase 8.2 Experiment: TeX Mirror Transform at Term Level

Goal: Compute the mirror contribution EXACTLY using shifted Q operators,
without using a scalar m₁ approximation.

The theory (from TRUTH_SPEC Section 10):
- I₁, I₂ have mirror assembly: I(α,β) + T^{-α-β}·I(-β,-α)
- The operator shift identity: Q(D_α)(T^{-s}F) = T^{-s} × Q(1 + D_α)F

So the mirror term uses Q(1+·) instead of Q(·).

Assembly:
  c = S12_direct + S12_mirror_exact + S34

where:
  S12_direct = Σ(I₁+I₂) at +R with standard operators Q(A_α)Q(A_β)
  S12_mirror_exact = exp(2R) × Σ(I₁+I₂) with shifted Q(1+A_α)Q(1+A_β)
  S34 = Σ(I₃+I₄) at +R (no mirror)

If this works, m₁ becomes UNNECESSARY as a primitive.
"""

import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.mirror_exact import compute_I1_with_shifted_Q
from src.operator_post_identity import compute_I1_operator_post_identity_pair
from src.evaluate import compute_c_paper_ordered


def compute_S12_direct(theta: float, R: float, n: int, polynomials: dict) -> float:
    """
    Compute S12 (I₁+I₂ sum) using standard operators at +R.
    Uses post-identity evaluation (combined identity already applied).
    """
    total = 0.0

    # Triangle pairs with symmetry factor
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    for ell1, ell2 in pairs:
        sym = 2 if ell1 != ell2 else 1

        # Compute I₁ for this pair via post-identity
        result = compute_I1_operator_post_identity_pair(
            theta=theta, R=R, ell1=ell1, ell2=ell2, n=n, polynomials=polynomials
        )
        # Note: I₂ = I₁ for symmetric pairs (same polynomial structure)
        # and the post-identity path gives the combined I₁ contribution

        total += sym * result.I1_value

    return total


def compute_S12_mirror_with_shifted_Q(theta: float, R: float, n: int, polynomials: dict) -> float:
    """
    Compute S12 mirror contribution using shifted Q operators.

    Theory: The mirror term uses Q(1+A_α)Q(1+A_β) instead of Q(A_α)Q(A_β).
    This accounts for the operator shift identity when acting on T^{-s} factor.

    Returns the SUM of shifted-Q I₁ values (before the exp(2R) weight).
    """
    total = 0.0

    # Triangle pairs with symmetry factor
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    for ell1, ell2 in pairs:
        sym = 2 if ell1 != ell2 else 1

        # Compute I₁ with shifted Q(1+·)
        I1_shifted = compute_I1_with_shifted_Q(
            theta=theta, R=R, n=n, polynomials=polynomials,
            ell1=ell1, ell2=ell2, shift=1.0
        )

        total += sym * I1_shifted

    return total


def analyze_phase82(benchmark: str = 'kappa', n: int = 60, verbose: bool = True):
    """
    Analyze Phase 8.2 mirror-exact approach.

    Test: Does S12_direct + exp(2R) × S12_shifted_Q + S34 ≈ c_target?
    """
    # Load appropriate polynomials and targets
    if benchmark == 'kappa':
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        R = 1.3036
        c_target = 2.13745440613217263636
    else:
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=False)
        R = 1.1167
        c_target = 1.9379524124677437

    polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}
    theta = 4.0 / 7.0

    # Get S34 from the ordered evaluator (no mirror)
    result_plus = compute_c_paper_ordered(
        theta=theta, R=R, n=n, polynomials=polynomials, K=3,
        s12_pair_mode='triangle',
    )
    S34 = result_plus.per_term.get('_S34_plus_total', 0.0)

    # Also get S12_direct from the evaluator for comparison
    S12_from_evaluator = result_plus.per_term.get('_S12_plus_total', 0.0)

    # Compute S12_direct using post-identity path
    S12_direct = compute_S12_direct(theta, R, n, polynomials)

    # Compute S12 with shifted Q
    S12_shifted_Q = compute_S12_mirror_with_shifted_Q(theta, R, n, polynomials)

    # The mirror weight from theory is exp(2R) for T^{-α-β} at α=β=-R/L
    exp_2R = np.exp(2 * R)

    # Assemble with exp(2R) as the mirror weight
    S12_mirror_exact = exp_2R * S12_shifted_Q
    c_assembled = S12_direct + S12_mirror_exact + S34

    # Compare to target
    gap = (c_assembled - c_target) / c_target * 100

    # Also compute what the empirical formula gives
    m1_empirical = np.exp(R) + 5

    # Get S12 at -R for empirical comparison
    result_minus = compute_c_paper_ordered(
        theta=theta, R=-R, n=n, polynomials=polynomials, K=3,
        s12_pair_mode='triangle',
    )
    S12_minus = result_minus.per_term.get('_S12_plus_total', 0.0)
    c_empirical = S12_from_evaluator + m1_empirical * S12_minus + S34
    gap_empirical = (c_empirical - c_target) / c_target * 100

    if verbose:
        print("=" * 70)
        print(f"Phase 8.2 Analysis: {benchmark.upper()} Benchmark")
        print("=" * 70)
        print()
        print(f"Parameters: R = {R}, θ = {theta:.6f}, n = {n}")
        print(f"Target c = {c_target:.8f}")
        print()
        print("Component Values:")
        print(f"  S12_direct (post-identity):    {S12_direct:.8f}")
        print(f"  S12_direct (evaluator):        {S12_from_evaluator:.8f}")
        print(f"  S12_shifted_Q:                 {S12_shifted_Q:.8f}")
        print(f"  S34:                           {S34:.8f}")
        print()
        print("Mirror Weights:")
        print(f"  exp(2R) = {exp_2R:.6f}")
        print(f"  m₁_empirical = exp(R)+5 = {m1_empirical:.6f}")
        print()
        print("Assembly Results:")
        print(f"  Phase 8.2 (exp(2R) × shifted_Q):")
        print(f"    S12_mirror_exact = {S12_mirror_exact:.8f}")
        print(f"    c = {c_assembled:.8f} (gap: {gap:+.2f}%)")
        print()
        print(f"  Empirical (m₁ × S12(-R)):")
        print(f"    S12(-R) = {S12_minus:.8f}")
        print(f"    m₁ × S12(-R) = {m1_empirical * S12_minus:.8f}")
        print(f"    c = {c_empirical:.8f} (gap: {gap_empirical:+.2f}%)")
        print()

        # Analyze the relationship between approaches
        if abs(S12_minus) > 1e-10 and abs(S12_shifted_Q) > 1e-10:
            ratio_shifted_to_minus = S12_shifted_Q / S12_minus
            print("Relationship Analysis:")
            print(f"  S12_shifted_Q / S12(-R) = {ratio_shifted_to_minus:.6f}")
            print(f"  exp(2R) × (shifted/minus) = {exp_2R * ratio_shifted_to_minus:.6f}")
            print(f"  This should equal m₁ = {m1_empirical:.6f} if approaches are equivalent")

        print("=" * 70)

    return {
        'benchmark': benchmark,
        'c_target': c_target,
        'c_assembled': c_assembled,
        'gap_percent': gap,
        'S12_direct': S12_direct,
        'S12_shifted_Q': S12_shifted_Q,
        'S12_mirror_exact': S12_mirror_exact,
        'S34': S34,
        'c_empirical': c_empirical,
        'gap_empirical': gap_empirical,
    }


def main():
    print("Phase 8.2: TeX Mirror Transform at Term Level")
    print("=" * 70)
    print()

    # Analyze both benchmarks
    kappa_result = analyze_phase82('kappa', n=60, verbose=True)
    print()
    kappa_star_result = analyze_phase82('kappa_star', n=60, verbose=True)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Phase 8.2 (exp(2R) × shifted_Q):")
    print(f"  κ benchmark:  gap = {kappa_result['gap_percent']:+.2f}%")
    print(f"  κ* benchmark: gap = {kappa_star_result['gap_percent']:+.2f}%")
    print()
    print("Empirical (m₁ × S12(-R)):")
    print(f"  κ benchmark:  gap = {kappa_result['gap_empirical']:+.2f}%")
    print(f"  κ* benchmark: gap = {kappa_star_result['gap_empirical']:+.2f}%")
    print()

    # Gate test
    if abs(kappa_result['gap_percent']) < 2.0 and abs(kappa_star_result['gap_percent']) < 2.0:
        print("✓ PHASE 8.2 GATE PASSED: Both benchmarks within 2% tolerance")
    else:
        print("✗ PHASE 8.2 GATE FAILED: Gap exceeds 2% tolerance")
        print("  Need to investigate the Q-shift relationship further")


if __name__ == "__main__":
    main()
