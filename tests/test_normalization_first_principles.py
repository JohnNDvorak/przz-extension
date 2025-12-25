"""
tests/test_normalization_first_principles.py
CRITICAL: Is the normalization from first principles or curve-fitting?

THE USER'S CONCERN:
==================
If we're curve-fitting the normalization to match PRZZ targets, then any
"improvements" we find through optimization might be illusory artifacts.

To validate our formula, we need to check:
1. Is the scale factor UNIVERSAL (same for all polynomials)?
2. Or is it POLYNOMIAL-DEPENDENT (would invalidate optimization)?

KEY TEST:
=========
For a FIXED R, if we change polynomials, does our computed c/c_target
ratio stay constant? If yes, our formula is structurally correct.
If no, we have a polynomial-dependent normalization problem.
"""

import numpy as np
from src.ratios.j1_euler_maclaurin import (
    LaurentMode,
    compute_m1_with_mirror_assembly,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials


def test_cross_polynomial_scaling():
    """
    Test if the scale factor is polynomial-dependent or R-dependent.

    Method: Use κ polynomials at κ* R value, and vice versa.
    If scale factor follows R, it's R-dependent.
    If scale factor follows polynomials, it's polynomial-dependent.
    """
    print("\n" + "=" * 70)
    print("CRITICAL TEST: IS NORMALIZATION POLYNOMIAL-DEPENDENT?")
    print("=" * 70)

    polys_kappa = load_przz_k3_polynomials("kappa")
    polys_kappa_star = load_przz_k3_polynomials("kappa_star")

    R_kappa = polys_kappa.R  # 1.3036
    R_kappa_star = polys_kappa_star.R  # 1.1167

    print(f"\nR_κ = {R_kappa}, R_κ* = {R_kappa_star}")

    # Compute c for all four combinations
    results = {}

    for poly_name, polys in [("κ_poly", polys_kappa), ("κ*_poly", polys_kappa_star)]:
        for R_name, R in [("R_κ", R_kappa), ("R_κ*", R_kappa_star)]:
            result = compute_m1_with_mirror_assembly(
                theta=4.0/7.0, R=R, polys=polys, K=3,
                laurent_mode=LaurentMode.ACTUAL_LOGDERIV
            )
            A = result['exp_coefficient']
            B = result['constant_offset']
            c = A * np.exp(R) + B

            key = f"{poly_name}@{R_name}"
            results[key] = {
                'c': c,
                'A': A,
                'B': B,
                'B_over_A': B/A,
                'R': R,
            }

    print("\nComputed c values for all polynomial/R combinations:")
    print("-" * 70)
    print(f"{'Combination':<20} {'c':<15} {'A':<12} {'B/A':<10}")
    print("-" * 70)

    for key, data in results.items():
        print(f"{key:<20} {data['c']:<15.6f} {data['A']:<12.6f} {data['B_over_A']:<10.4f}")

    # Now analyze: if we had PRZZ targets for all combinations,
    # we could check if scale factors are consistent

    # We only have targets for the "native" combinations:
    # κ_poly @ R_κ: c_target = 2.137
    # κ*_poly @ R_κ*: c_target = 1.938

    c_kappa_native = results["κ_poly@R_κ"]['c']
    c_kappa_star_native = results["κ*_poly@R_κ*"]['c']

    scale_kappa = 2.137 / c_kappa_native
    scale_kappa_star = 1.938 / c_kappa_star_native

    print(f"\nNative scale factors:")
    print(f"  κ: c_target/c_computed = 2.137/{c_kappa_native:.3f} = {scale_kappa:.6f}")
    print(f"  κ*: c_target/c_computed = 1.938/{c_kappa_star_native:.3f} = {scale_kappa_star:.6f}")
    print(f"  Ratio: {scale_kappa/scale_kappa_star:.4f}")

    # Key question: do the "cross" combinations have consistent scales?
    # If we use κ polynomials at R_κ*, what scale would we expect?

    print("\n" + "=" * 70)
    print("ANALYZING SCALE FACTOR DEPENDENCE")
    print("=" * 70)

    # Hypothesis 1: Scale depends only on R
    print("\nHypothesis 1: Scale = f(R) only")
    print("-" * 50)

    # If scale depends only on R, then:
    # - κ_poly @ R_κ* should have scale = scale_kappa_star
    # - κ*_poly @ R_κ should have scale = scale_kappa

    c_kappa_at_R_star = results["κ_poly@R_κ*"]['c']
    c_kappa_star_at_R = results["κ*_poly@R_κ"]['c']

    print(f"  κ_poly @ R_κ*: c = {c_kappa_at_R_star:.6f}")
    print(f"    If scale = scale_κ*: c_target would be {c_kappa_at_R_star * scale_kappa_star:.6f}")

    print(f"  κ*_poly @ R_κ: c = {c_kappa_star_at_R:.6f}")
    print(f"    If scale = scale_κ: c_target would be {c_kappa_star_at_R * scale_kappa:.6f}")

    # Hypothesis 2: Scale depends only on polynomials
    print("\nHypothesis 2: Scale = f(polynomials) only")
    print("-" * 50)

    print(f"  κ_poly @ R_κ*: c = {c_kappa_at_R_star:.6f}")
    print(f"    If scale = scale_κ: c_target would be {c_kappa_at_R_star * scale_kappa:.6f}")

    print(f"  κ*_poly @ R_κ: c = {c_kappa_star_at_R:.6f}")
    print(f"    If scale = scale_κ*: c_target would be {c_kappa_star_at_R * scale_kappa_star:.6f}")


def analyze_przz_c_ratio():
    """
    Analyze if our c ratio matches PRZZ c ratio.

    PRZZ: c_κ / c_κ* = 2.137 / 1.938 = 1.103
    Our:  c_κ / c_κ* = ?

    If our ratio matches, the structural formula is correct.
    If not, there's a polynomial-dependent error.
    """
    print("\n" + "=" * 70)
    print("C RATIO ANALYSIS: STRUCTURAL CORRECTNESS TEST")
    print("=" * 70)

    polys_kappa = load_przz_k3_polynomials("kappa")
    polys_kappa_star = load_przz_k3_polynomials("kappa_star")

    # Compute at native R values
    result_k = compute_m1_with_mirror_assembly(
        theta=4.0/7.0, R=polys_kappa.R, polys=polys_kappa, K=3,
        laurent_mode=LaurentMode.ACTUAL_LOGDERIV
    )
    result_ks = compute_m1_with_mirror_assembly(
        theta=4.0/7.0, R=polys_kappa_star.R, polys=polys_kappa_star, K=3,
        laurent_mode=LaurentMode.ACTUAL_LOGDERIV
    )

    c_k = result_k['exp_coefficient'] * np.exp(polys_kappa.R) + result_k['constant_offset']
    c_ks = result_ks['exp_coefficient'] * np.exp(polys_kappa_star.R) + result_ks['constant_offset']

    our_ratio = c_k / c_ks
    przz_ratio = 2.137 / 1.938

    print(f"\nPRZZ c ratio: c_κ/c_κ* = 2.137/1.938 = {przz_ratio:.6f}")
    print(f"Our c ratio:  c_κ/c_κ* = {c_k:.3f}/{c_ks:.3f} = {our_ratio:.6f}")
    print(f"\nRatio of ratios: {our_ratio/przz_ratio:.6f}")

    if abs(our_ratio/przz_ratio - 1) < 0.05:
        print("\n✓ Our c ratio matches PRZZ within 5% - formula is structurally correct")
    else:
        print(f"\n✗ Our c ratio differs by {(our_ratio/przz_ratio - 1)*100:.1f}% - POSSIBLE PROBLEM")
        print("  This could indicate polynomial-dependent normalization error")


def investigate_what_przz_normalizes_by():
    """
    Investigate what normalization PRZZ uses.

    From PRZZ paper, the main asymptotic involves:
    M ~ (c/θ) × T/log T

    So c is defined relative to the asymptotic structure.

    The question: what normalization makes our c match PRZZ's c?
    """
    print("\n" + "=" * 70)
    print("INVESTIGATING PRZZ NORMALIZATION")
    print("=" * 70)

    from scipy import integrate

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 50)

        # Compute polynomial integrals
        def P1(u):
            return float(polys.P1.eval(np.array([u]))[0])
        def P2(u):
            return float(polys.P2.eval(np.array([u]))[0])

        int_P1_sq, _ = integrate.quad(lambda u: P1(u)**2, 0, 1)
        int_P2_sq, _ = integrate.quad(lambda u: P2(u)**2, 0, 1)
        int_P1P2, _ = integrate.quad(lambda u: P1(u)*P2(u), 0, 1)

        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        c_computed = result['exp_coefficient'] * np.exp(R) + result['constant_offset']
        c_target = 2.137 if benchmark == "kappa" else 1.938

        scale_needed = c_target / c_computed

        print(f"  c_computed = {c_computed:.6f}")
        print(f"  c_target = {c_target:.6f}")
        print(f"  scale_needed = {scale_needed:.6f}")

        # Try various normalizations
        print(f"\n  Possible normalizations:")
        print(f"    ∫P₁² = {int_P1_sq:.6f}")
        print(f"    ∫P₂² = {int_P2_sq:.6f}")
        print(f"    ∫P₁P₂ = {int_P1P2:.6f}")
        print(f"    ∫P₁²×∫P₂² = {int_P1_sq * int_P2_sq:.6f}")

        # Check if scale_needed matches any polynomial quantity
        print(f"\n  Ratios:")
        print(f"    scale / ∫P₁P₂ = {scale_needed / int_P1P2:.6f}")
        print(f"    scale / ∫P₁² = {scale_needed / int_P1_sq:.6f}")
        print(f"    scale / ∫P₂² = {scale_needed / int_P2_sq:.6f}")
        print(f"    scale / (∫P₁²×∫P₂²) = {scale_needed / (int_P1_sq * int_P2_sq):.6f}")
        print(f"    scale × ∫P₁P₂ = {scale_needed * int_P1P2:.6f}")


def check_ratio_preservation():
    """
    CRITICAL CHECK: If we modify polynomials slightly, does the c ratio
    to PRZZ stay constant?

    This is the acid test for whether optimization would be valid.
    """
    print("\n" + "=" * 70)
    print("RATIO PRESERVATION TEST")
    print("=" * 70)

    print("""
To validate optimization, we need:
  c_true = c_computed × scale

Where 'scale' should be CONSTANT across polynomial variations.

If scale varies with polynomials, optimization is INVALID.
If scale is constant (or only depends on R), optimization is VALID.

Unfortunately, we only have PRZZ targets for two polynomial sets,
so we cannot definitively test this without more reference points.

WHAT WE CAN CHECK:
- The B/A ratio is correct (~5) for both benchmarks ✓
- This suggests STRUCTURAL correctness
- The absolute scale issue might be a UNIVERSAL constant we're missing

WHAT WOULD MAKE OPTIMIZATION VALID:
1. If the missing scale is INDEPENDENT of polynomials (just depends on R, θ, etc.)
2. Then relative improvements would be preserved
3. c_new/c_old = c_computed_new/c_computed_old (scale cancels)
""")

    # Let's check if the scale ratio could be explained by R-dependence
    polys_kappa = load_przz_k3_polynomials("kappa")
    polys_kappa_star = load_przz_k3_polynomials("kappa_star")

    R_k = polys_kappa.R
    R_ks = polys_kappa_star.R

    scale_k = 0.148
    scale_ks = 0.208

    scale_ratio = scale_k / scale_ks
    R_ratio = R_k / R_ks

    print(f"\nScale analysis:")
    print(f"  scale_κ / scale_κ* = {scale_ratio:.4f}")
    print(f"  R_κ / R_κ* = {R_ratio:.4f}")
    print(f"  exp(-R_κ) / exp(-R_κ*) = {np.exp(-R_k) / np.exp(-R_ks):.4f}")
    print(f"  exp(R_κ) / exp(R_κ*) = {np.exp(R_k) / np.exp(R_ks):.4f}")

    # Check if scale ~ exp(-R) or similar
    print(f"\n  scale_κ × exp(R_κ) = {scale_k * np.exp(R_k):.4f}")
    print(f"  scale_κ* × exp(R_κ*) = {scale_ks * np.exp(R_ks):.4f}")

    # If these are equal, then scale = C × exp(-R) for some constant C
    ratio_check = (scale_k * np.exp(R_k)) / (scale_ks * np.exp(R_ks))
    print(f"  Ratio: {ratio_check:.4f}")

    if abs(ratio_check - 1) < 0.1:
        print("\n  ✓ Scale appears to be ~ C×exp(-R), which is R-dependent only")
        print("    This would make optimization VALID")
    else:
        print(f"\n  Scale×exp(R) differs by {(ratio_check-1)*100:.1f}%")
        print("    Scale has polynomial dependence - optimization may be affected")


if __name__ == "__main__":
    test_cross_polynomial_scaling()
    analyze_przz_c_ratio()
    investigate_what_przz_normalizes_by()
    check_ratio_preservation()
