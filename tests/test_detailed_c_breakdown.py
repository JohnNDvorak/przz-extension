"""
tests/test_detailed_c_breakdown.py
Detailed breakdown of c computation to identify the 1.35% underestimate.

The full c computation gives:
  c = S12(+R) + m×S12(-R) + S34(+R)

where:
  S12 = I₁ + I₂ (summed over all pairs with factorial normalization)
  S34 = I₃ + I₄ (summed over all pairs with factorial normalization)
  m = exp(R) + 5

We compute c = 2.1085 but need c = 2.1375 (gap = 0.0290, or 1.35%)

This test breaks down each component to identify where the gap comes from.
"""

import numpy as np
from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def detailed_breakdown_kappa():
    """Detailed breakdown for κ benchmark."""
    print("\n" + "=" * 70)
    print("DETAILED BREAKDOWN: κ (R=1.3036)")
    print("=" * 70)

    R = 1.3036
    c_target = 2.137454406
    theta = 4.0 / 7.0

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    result = compute_c_paper_with_mirror(
        theta=theta,
        R=R,
        n=60,
        polynomials=polynomials,
        pair_mode="hybrid",
        use_factorial_normalization=True,
        mode="main",
        K=3,
    )

    c = result.total
    per_term = result.per_term

    print(f"\nTarget: c = {c_target:.10f}")
    print(f"Computed: c = {c:.10f}")
    print(f"Gap: {c_target - c:.10f} ({(c_target - c)/c_target*100:+.4f}%)")

    print("\n--- Main Components ---")
    s12_plus = per_term["_S12_plus_total"]
    s12_minus = per_term["_S12_minus_total"]
    s34_plus = per_term["_S34_plus_total"]
    mirror_mult = per_term["_mirror_multiplier"]

    print(f"S12(+R) = {s12_plus:.10f}")
    print(f"S12(-R) = {s12_minus:.10f}")
    print(f"S34(+R) = {s34_plus:.10f}")
    print(f"mirror_mult = {mirror_mult:.10f}")

    # Verify assembly
    c_check = s12_plus + mirror_mult * s12_minus + s34_plus
    print(f"\nc = S12(+R) + m×S12(-R) + S34(+R)")
    print(f"c = {s12_plus:.6f} + {mirror_mult:.6f}×{s12_minus:.6f} + {s34_plus:.6f}")
    print(f"c = {c_check:.10f}")

    # What adjustments would fix the gap?
    gap = c_target - c
    print(f"\n--- What would fix the gap of {gap:.6f}? ---")

    # Option 1: Adjust S12(+R)
    s12_plus_needed = c_target - mirror_mult * s12_minus - s34_plus
    print(f"1. S12(+R) needed: {s12_plus_needed:.10f} (vs {s12_plus:.10f})")
    print(f"   Increase by: {(s12_plus_needed - s12_plus)/s12_plus*100:+.4f}%")

    # Option 2: Adjust S12(-R)
    s12_minus_needed = (c_target - s12_plus - s34_plus) / mirror_mult
    print(f"2. S12(-R) needed: {s12_minus_needed:.10f} (vs {s12_minus:.10f})")
    print(f"   Increase by: {(s12_minus_needed - s12_minus)/s12_minus*100:+.4f}%")

    # Option 3: Adjust S34(+R)
    s34_needed = c_target - s12_plus - mirror_mult * s12_minus
    print(f"3. S34(+R) needed: {s34_needed:.10f} (vs {s34_plus:.10f})")
    print(f"   Increase by: {s34_needed - s34_plus:.6f} (less negative)")

    # Option 4: Adjust mirror_mult
    mirror_needed = (c_target - s12_plus - s34_plus) / s12_minus
    print(f"4. mirror_mult needed: {mirror_needed:.10f} (vs {mirror_mult:.10f})")
    print(f"   Increase by: {(mirror_needed - mirror_mult)/mirror_mult*100:+.4f}%")

    # I1 and I2 breakdown
    print("\n--- I₁ and I₂ Breakdown ---")
    i1_plus = per_term.get("_I1_plus_total", 0)
    i1_minus = per_term.get("_I1_minus_total", 0)
    i2_plus = per_term.get("_I2_plus_total", 0)
    i2_minus = per_term.get("_I2_minus_total", 0)

    print(f"I₁(+R) = {i1_plus:.10f}")
    print(f"I₁(-R) = {i1_minus:.10f}")
    print(f"I₂(+R) = {i2_plus:.10f}")
    print(f"I₂(-R) = {i2_minus:.10f}")
    print(f"S12(+R) check: I₁+I₂ = {i1_plus + i2_plus:.10f} (vs {s12_plus:.10f})")
    print(f"S12(-R) check: I₁+I₂ = {i1_minus + i2_minus:.10f} (vs {s12_minus:.10f})")


def detailed_breakdown_kappa_star():
    """Detailed breakdown for κ* benchmark."""
    print("\n" + "=" * 70)
    print("DETAILED BREAKDOWN: κ* (R=1.1167)")
    print("=" * 70)

    R = 1.1167
    c_target = 1.9379524081
    theta = 4.0 / 7.0

    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    result = compute_c_paper_with_mirror(
        theta=theta,
        R=R,
        n=60,
        polynomials=polynomials,
        pair_mode="hybrid",
        use_factorial_normalization=True,
        mode="main",
        K=3,
    )

    c = result.total
    per_term = result.per_term

    print(f"\nTarget: c = {c_target:.10f}")
    print(f"Computed: c = {c:.10f}")
    print(f"Gap: {c_target - c:.10f} ({(c_target - c)/c_target*100:+.4f}%)")

    print("\n--- Main Components ---")
    s12_plus = per_term["_S12_plus_total"]
    s12_minus = per_term["_S12_minus_total"]
    s34_plus = per_term["_S34_plus_total"]
    mirror_mult = per_term["_mirror_multiplier"]

    print(f"S12(+R) = {s12_plus:.10f}")
    print(f"S12(-R) = {s12_minus:.10f}")
    print(f"S34(+R) = {s34_plus:.10f}")
    print(f"mirror_mult = {mirror_mult:.10f}")

    gap = c_target - c
    print(f"\n--- What would fix the gap of {gap:.6f}? ---")

    s12_plus_needed = c_target - mirror_mult * s12_minus - s34_plus
    print(f"1. S12(+R) needed: {s12_plus_needed:.10f} (vs {s12_plus:.10f})")
    print(f"   Increase by: {(s12_plus_needed - s12_plus)/s12_plus*100:+.4f}%")

    s12_minus_needed = (c_target - s12_plus - s34_plus) / mirror_mult
    print(f"2. S12(-R) needed: {s12_minus_needed:.10f} (vs {s12_minus:.10f})")
    print(f"   Increase by: {(s12_minus_needed - s12_minus)/s12_minus*100:+.4f}%")


def check_mirror_multiplier():
    """Check if mirror multiplier formula is correct."""
    print("\n" + "=" * 70)
    print("MIRROR MULTIPLIER ANALYSIS")
    print("=" * 70)

    for benchmark, R in [("kappa", 1.3036), ("kappa_star", 1.1167)]:
        print(f"\n{benchmark.upper()} (R={R}):")

        K = 3
        mirror_formula = np.exp(R) + (2 * K - 1)
        print(f"  Formula: exp(R) + (2K - 1) = exp({R}) + 5")
        print(f"  exp(R) = {np.exp(R):.10f}")
        print(f"  2K - 1 = 5")
        print(f"  Total = {mirror_formula:.10f}")

        # What if it should be something else?
        print(f"\n  Alternative multipliers:")
        for alt in [np.exp(R) + 4, np.exp(R) + 6, np.exp(R) * 2]:
            print(f"    m = {alt:.4f}: ", end="")
            # Quick calculation of what c would be
            # c ≈ 0.8 + m×0.22 - 0.6 for kappa
            if benchmark == "kappa":
                c_est = 0.797 + alt * 0.220 - 0.600
                c_target = 2.137
                print(f"c ≈ {c_est:.4f} (target {c_target:.3f})")
            else:
                c_est = 0.615 + alt * 0.216 - 0.443
                c_target = 1.938
                print(f"c ≈ {c_est:.4f} (target {c_target:.3f})")


def per_pair_breakdown():
    """Show per-pair contributions."""
    print("\n" + "=" * 70)
    print("PER-PAIR BREAKDOWN")
    print("=" * 70)

    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term

    R = 1.3036
    theta = 4.0 / 7.0

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    terms_plus = make_all_terms_k3(theta, R, kernel_regime="paper")
    terms_minus = make_all_terms_k3(theta, -R, kernel_regime="paper")

    # Factorial normalization factors
    f_norm = {"11": 1.0, "22": 0.25, "33": 1/36, "12": 0.5, "13": 1/6, "23": 1/12}
    sym = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    mirror_mult = np.exp(R) + 5

    print(f"\nκ (R={R}):")
    print("-" * 70)
    print(f"{'Pair':<6} {'I₁+I₂(+R)':<15} {'I₁+I₂(-R)':<15} {'I₃+I₄(+R)':<15} {'Total':<15}")
    print("-" * 70)

    total_s12_plus = 0
    total_s12_minus = 0
    total_s34 = 0

    for pair in ["11", "22", "33", "12", "13", "23"]:
        terms_p = terms_plus[pair]
        terms_m = terms_minus[pair]

        norm = f_norm[pair] * sym[pair]

        # I₁ + I₂
        i12_plus = 0
        i12_minus = 0
        for i in [0, 1]:  # I₁, I₂
            res_p = evaluate_term(terms_p[i], polynomials, n=60, R=R, theta=theta, n_quad_a=40)
            res_m = evaluate_term(terms_m[i], polynomials, n=60, R=-R, theta=theta, n_quad_a=40)
            i12_plus += norm * res_p.value
            i12_minus += norm * res_m.value

        # I₃ + I₄
        i34 = 0
        for i in [2, 3]:  # I₃, I₄
            res_p = evaluate_term(terms_p[i], polynomials, n=60, R=R, theta=theta, n_quad_a=40)
            i34 += norm * res_p.value

        pair_total = i12_plus + mirror_mult * i12_minus + i34
        print(f"{pair:<6} {i12_plus:+14.8f} {i12_minus:+14.8f} {i34:+14.8f} {pair_total:+14.8f}")

        total_s12_plus += i12_plus
        total_s12_minus += i12_minus
        total_s34 += i34

    print("-" * 70)
    c_computed = total_s12_plus + mirror_mult * total_s12_minus + total_s34
    print(f"{'Total':<6} {total_s12_plus:+14.8f} {total_s12_minus:+14.8f} {total_s34:+14.8f} {c_computed:+14.8f}")
    print(f"\nc = S12(+R) + {mirror_mult:.4f}×S12(-R) + S34(+R) = {c_computed:.10f}")
    print(f"Target: c = 2.137454406")
    print(f"Gap: {2.137454406 - c_computed:.10f}")


if __name__ == "__main__":
    detailed_breakdown_kappa()
    detailed_breakdown_kappa_star()
    check_mirror_multiplier()
    per_pair_breakdown()
