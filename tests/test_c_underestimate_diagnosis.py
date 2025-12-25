"""
tests/test_c_underestimate_diagnosis.py
CRITICAL: Diagnose why c is systematically underestimated

Our c is ~1.35% too LOW, causing κ to overshoot by ~1 percentage point.
This is UNACCEPTABLE for a lower bound proof.

PRZZ targets (from paper):
- κ: c = 2.137454, κ ≥ 0.417293962
- κ*: c = ?, κ* ≥ 0.407511457

Our computation:
- κ: c = 2.109, κ = 0.428 (OVERSHOOTS by 1.1 pp)
- κ*: c = 1.915, κ = 0.418 (OVERSHOOTS by 1.1 pp vs 0.4075)

Possible causes:
1. ACTUAL_LOGDERIV fix not integrated into full evaluation
2. Missing positive terms
3. Sign errors
4. Incorrect normalization
"""

import numpy as np
from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def diagnose_c_breakdown():
    """Get detailed breakdown of c computation."""
    print("\n" + "=" * 70)
    print("DIAGNOSING c UNDERESTIMATE")
    print("=" * 70)

    for benchmark, loader, R, c_target, kappa_target in [
        ("kappa", load_przz_polynomials, 1.3036, 2.137454406, 0.417293962),
        ("kappa_star", load_przz_polynomials_kappa_star, 1.1167, None, 0.407511457),
    ]:
        P1, P2, P3, Q = loader() if benchmark == "kappa" else loader()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 60)

        result = compute_c_paper_with_mirror(
            theta=4.0/7.0,
            R=R,
            n=60,
            polynomials=polynomials,
            pair_mode="hybrid",
            use_factorial_normalization=True,
            mode="main",
            K=3,
        )

        c_computed = result.total
        kappa_computed = 1 - np.log(c_computed) / R

        print(f"  c computed: {c_computed:.10f}")
        if c_target:
            print(f"  c target:   {c_target:.10f}")
            print(f"  c error:    {(c_computed - c_target) / c_target * 100:+.4f}%")

        print(f"\n  κ computed: {kappa_computed:.10f}")
        print(f"  κ target:   {kappa_target:.10f}")
        print(f"  κ error:    {(kappa_computed - kappa_target) * 100:+.4f} percentage points")

        # The KEY question: is c too low or κ formula wrong?
        # If c is correct, κ = 1 - log(c)/R should match
        if c_target:
            kappa_from_target_c = 1 - np.log(c_target) / R
            print(f"\n  κ from target c: {kappa_from_target_c:.10f}")
            print(f"  κ target:        {kappa_target:.10f}")
            print(f"  Match: {abs(kappa_from_target_c - kappa_target) < 1e-6}")

        # Show per-term breakdown
        print(f"\n  Per-term breakdown:")
        if result.per_term:
            for key, val in sorted(result.per_term.items()):
                if not key.startswith("_"):
                    print(f"    {key}: {val:+.6f}")

        # What c would we need to hit the target κ?
        c_needed = np.exp(R * (1 - kappa_target))
        print(f"\n  c needed for κ={kappa_target}: {c_needed:.10f}")
        print(f"  Our c: {c_computed:.10f}")
        print(f"  Shortfall: {(c_needed - c_computed) / c_needed * 100:+.4f}%")


def check_evaluation_uses_actual_logderiv():
    """Check if the evaluation is using ACTUAL_LOGDERIV or Laurent approximation."""
    print("\n" + "=" * 70)
    print("CHECKING IF EVALUATION USES ACTUAL_LOGDERIV")
    print("=" * 70)

    # The Phase 15 fix was in j1_euler_maclaurin.py
    # But compute_c_paper_with_mirror uses a different code path

    print("""
The Phase 15 ACTUAL_LOGDERIV fix was applied to:
  - src/ratios/j1_euler_maclaurin.py (j12_as_integral function)

The full c evaluation uses:
  - src/evaluate.py (compute_c_paper_with_mirror)

These might be DIFFERENT code paths!

The full evaluation computes I₁, I₂, I₃, I₄ using term_dsl.py
and evaluate_term(), which may NOT use the ACTUAL_LOGDERIV fix.
""")

    # Check what the I₂ term computation looks like
    from src.terms_k3_d1 import make_all_terms_i2
    from src.term_dsl import Term

    print("\nI₂ term structure:")
    i2_terms = make_all_terms_i2()
    for (l1, l2), term_list in i2_terms.items():
        print(f"\n  ({l1},{l2}): {len(term_list)} terms")
        if (l1, l2) == (1, 1):
            for i, term in enumerate(term_list[:3]):  # Show first 3 terms
                print(f"    Term {i}: {term}")


def compute_target_c_from_kappa():
    """Compute what c should be from the target κ values."""
    print("\n" + "=" * 70)
    print("TARGET c VALUES FROM κ")
    print("=" * 70)

    # From the paper excerpt
    targets = [
        ("κ", 1.3036, 0.417293962),
        ("κ*", 1.1167, 0.407511457),
    ]

    for name, R, kappa in targets:
        c = np.exp(R * (1 - kappa))
        print(f"\n{name} (R={R}, κ={kappa}):")
        print(f"  c = exp(R × (1-κ)) = exp({R} × {1-kappa:.9f})")
        print(f"  c = {c:.10f}")


if __name__ == "__main__":
    compute_target_c_from_kappa()
    diagnose_c_breakdown()
    check_evaluation_uses_actual_logderiv()
