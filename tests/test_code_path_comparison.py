"""
tests/test_code_path_comparison.py
Compare the two code paths for c computation:

Path A: compute_m1_with_mirror_assembly (j1_euler_maclaurin.py)
- Uses ACTUAL_LOGDERIV for J12
- +5 gate validation path
- Computes simplified (1,1) structure

Path B: compute_c_paper_with_mirror (evaluate.py)
- Uses term_dsl with poly/exp factors
- Full c computation path
- Computes all (ℓ₁,ℓ₂) pairs

KEY QUESTION: Why does Path A give ~0.5% gap (with ACTUAL_LOGDERIV)
while Path B gives ~1.35% underestimate?

HYPOTHESIS:
- Path A uses J12/J13/J14 integrals with explicit Laurent/logderiv factors
- Path B uses term_dsl which may be missing these factors
- The ~1.35% gap may be due to missing (ζ'/ζ) factors in Path B
"""

import numpy as np
from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.ratios.j1_euler_maclaurin import (
    LaurentMode,
    compute_m1_with_mirror_assembly,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials


def compare_code_paths():
    """Compare Path A and Path B outputs."""
    print("\n" + "=" * 70)
    print("CODE PATH COMPARISON: ACTUAL_LOGDERIV vs TERM_DSL")
    print("=" * 70)

    targets = {
        "kappa": {"R": 1.3036, "c_target": 2.137454406, "kappa_target": 0.417293962},
        "kappa_star": {"R": 1.1167, "c_target": 1.9379524081, "kappa_target": 0.407511457},
    }

    for benchmark in ["kappa", "kappa_star"]:
        target = targets[benchmark]
        R = target["R"]
        c_target = target["c_target"]
        kappa_target = target["kappa_target"]

        print(f"\n{'=' * 60}")
        print(f"{benchmark.upper()} (R={R}, c_target={c_target:.6f})")
        print("=" * 60)

        # Load polynomials for each path
        polys_a = load_przz_k3_polynomials(benchmark)

        if benchmark == "kappa":
            P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        else:
            P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        polys_b = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        # Path A: compute_m1_with_mirror_assembly
        print("\n--- PATH A: compute_m1_with_mirror_assembly ---")
        result_a = compute_m1_with_mirror_assembly(
            theta=4.0/7.0,
            R=R,
            polys=polys_a,
            K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV,
        )
        A = result_a["exp_coefficient"]
        B = result_a["constant_offset"]
        c_a = A * np.exp(R) + B

        kappa_a = 1 - np.log(c_a) / R
        c_error_a = (c_a - c_target) / c_target * 100
        kappa_error_a = (kappa_a - kappa_target) * 100

        print(f"  exp_coefficient (A): {A:.10f}")
        print(f"  constant_offset (B): {B:.10f}")
        print(f"  B/A ratio: {B/A:.6f}")
        print(f"  c = A×exp(R) + B: {c_a:.10f}")
        print(f"  c error: {c_error_a:+.4f}%")
        print(f"  κ computed: {kappa_a:.10f}")
        print(f"  κ error: {kappa_error_a:+.4f} pp")

        # Path B: compute_c_paper_with_mirror
        print("\n--- PATH B: compute_c_paper_with_mirror ---")
        result_b = compute_c_paper_with_mirror(
            theta=4.0/7.0,
            R=R,
            n=60,
            polynomials=polys_b,
            pair_mode="hybrid",
            use_factorial_normalization=True,
            mode="main",
            K=3,
        )
        c_b = result_b.total

        kappa_b = 1 - np.log(c_b) / R
        c_error_b = (c_b - c_target) / c_target * 100
        kappa_error_b = (kappa_b - kappa_target) * 100

        print(f"  c computed: {c_b:.10f}")
        print(f"  c error: {c_error_b:+.4f}%")
        print(f"  κ computed: {kappa_b:.10f}")
        print(f"  κ error: {kappa_error_b:+.4f} pp")

        # Compare the two paths
        print("\n--- PATH COMPARISON ---")
        c_ratio = c_b / c_a
        print(f"  c(Path B) / c(Path A) = {c_ratio:.6f}")
        print(f"  Path A c error: {c_error_a:+.4f}%")
        print(f"  Path B c error: {c_error_b:+.4f}%")
        print(f"  Difference: {c_error_b - c_error_a:+.4f}%")

        # What factor would make Path B match Path A?
        correction_factor = c_a / c_b
        print(f"\n  To match Path A, multiply Path B by: {correction_factor:.6f}")

        # Breakdown from Path B
        if result_b.per_term:
            print("\n  Path B breakdown:")
            s12_plus = result_b.per_term.get("_S12_plus_total", 0)
            s12_minus = result_b.per_term.get("_S12_minus_total", 0)
            s34_plus = result_b.per_term.get("_S34_plus_total", 0)
            mirror_mult = result_b.per_term.get("_mirror_multiplier", 0)

            print(f"    S12(+R) = {s12_plus:.10f}")
            print(f"    S12(-R) = {s12_minus:.10f}")
            print(f"    S34(+R) = {s34_plus:.10f}")
            print(f"    mirror_mult = {mirror_mult:.6f}")
            print(f"    c = S12(+R) + {mirror_mult:.4f}×S12(-R) + S34(+R)")


def analyze_laurent_factor_effect():
    """
    Analyze how much the Laurent factor affects the result.

    The key insight: J12 uses (ζ'/ζ)(1-R)² factor, but term_dsl doesn't.
    Let's quantify this difference.
    """
    print("\n" + "=" * 70)
    print("LAURENT FACTOR ANALYSIS")
    print("=" * 70)

    from src.ratios.g_product_full import compute_zeta_factors
    from src.ratios.zeta_laurent import EULER_MASCHERONI

    for benchmark, R in [("kappa", 1.3036), ("kappa_star", 1.1167)]:
        print(f"\n{benchmark.upper()} (R={R}):")

        result = compute_zeta_factors(R)

        # Laurent approximation for (ζ'/ζ)(1-R)
        laurent_approx = 1.0 / R + EULER_MASCHERONI
        laurent_squared = laurent_approx ** 2

        # Actual value
        actual_squared = result.logderiv_actual_squared

        ratio = actual_squared / laurent_squared

        print(f"  Laurent (1/R + γ)²: {laurent_squared:.6f}")
        print(f"  Actual (ζ'/ζ)(1-R)²: {actual_squared:.6f}")
        print(f"  Ratio (actual/Laurent): {ratio:.4f}")
        print(f"  This means actual is {(ratio - 1) * 100:+.1f}% different from Laurent")


def check_what_term_dsl_computes():
    """
    Check what factors are actually in the term_dsl computation.

    The I1, I2, I3, I4 terms in term_dsl have:
    - Polynomial factors: P(u), Q(t)
    - Exponential factors: exp(R × args)
    - Algebraic prefactor: (1/θ + x + y)
    - NO explicit ζ'/ζ or Laurent factors!

    The question: where should these factors appear?
    """
    print("\n" + "=" * 70)
    print("WHAT TERM_DSL COMPUTES")
    print("=" * 70)

    print("""
The term_dsl approach computes I₁, I₂, I₃, I₄ integrals directly:

I₁ = d²/dxdy |_{x=y=0} ∫∫ [algebraic_pf × poly_pf × P_factors × Q_factors × exp_factors] du dt

Factors in term_dsl:
  - algebraic_prefactor: (1/θ + x + y) - linear in x, y
  - poly_prefactors: (1-u)^power - depends on pair indices
  - poly_factors: P_ℓ₁(x+u), P_ℓ₂(y+u), Q(Arg_α), Q(Arg_β)
  - exp_factors: exp(R × Arg_α), exp(R × Arg_β)

What's MISSING compared to PRZZ:
  - The zeta quotient products (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)
  - These come from the residue computation at poles
  - J12 in the simplified path uses (ζ'/ζ)(1-R)² = (1/R + γ)² [Laurent]
  - Or with ACTUAL_LOGDERIV: actual numerical (ζ'/ζ)(1-R)²

HYPOTHESIS:
The term_dsl path gives ~1.35% lower c because it's missing the
proper zeta quotient contribution that J12 captures with its Laurent factor.

But wait - if that were true, the gap would be much larger (Laurent is ~66%
different from actual for kappa). So there must be something else going on.
""")


def investigate_i1_vs_j12():
    """
    I1 in term_dsl should correspond to J12 in the simplified path.
    Let's compare them directly.
    """
    print("\n" + "=" * 70)
    print("I1 (TERM_DSL) vs J12 (SIMPLIFIED PATH)")
    print("=" * 70)

    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term
    from src.ratios.j1_euler_maclaurin import j12_as_integral
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    theta = 4.0 / 7.0
    R = 1.3036

    # Get (1,1) terms from term_dsl
    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")
    terms_11 = all_terms["11"]

    # I1 is index 0, I2 is index 1
    i1_term = terms_11[0]
    i2_term = terms_11[1]

    # Evaluate using term_dsl
    i1_result = evaluate_term(i1_term, polys, n=60, R=R, theta=theta, n_quad_a=40)
    i2_result = evaluate_term(i2_term, polys, n=60, R=R, theta=theta, n_quad_a=40)

    print(f"(1,1) term_dsl evaluation at R={R}:")
    print(f"  I₁: {i1_result.value:.10f}")
    print(f"  I₂: {i2_result.value:.10f}")
    print(f"  I₁ + I₂: {i1_result.value + i2_result.value:.10f}")

    # Compare with J12 from simplified path
    # Note: J12 has different structure - uses explicit Laurent factor
    def P1_func(u):
        return P1.eval(np.array([u]))[0]
    def P2_func(u):
        return P2.eval(np.array([u]))[0]

    j12_raw = j12_as_integral(R, theta=theta, P1_func=P1_func, P2_func=P2_func,
                               laurent_mode=LaurentMode.RAW_LOGDERIV)
    j12_actual = j12_as_integral(R, theta=theta, P1_func=P1_func, P2_func=P2_func,
                                  laurent_mode=LaurentMode.ACTUAL_LOGDERIV)

    print(f"\nJ12 from simplified path:")
    print(f"  J12 (RAW_LOGDERIV): {j12_raw:.10f}")
    print(f"  J12 (ACTUAL_LOGDERIV): {j12_actual:.10f}")
    print(f"  Ratio (ACTUAL/RAW): {j12_actual / j12_raw:.6f}")

    print(f"\nComparison:")
    print(f"  (I₁ + I₂) / J12_raw: {(i1_result.value + i2_result.value) / j12_raw:.6f}")
    print(f"  (I₁ + I₂) / J12_actual: {(i1_result.value + i2_result.value) / j12_actual:.6f}")


if __name__ == "__main__":
    compare_code_paths()
    analyze_laurent_factor_effect()
    check_what_term_dsl_computes()
    investigate_i1_vs_j12()
