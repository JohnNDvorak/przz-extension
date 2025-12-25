"""
tests/test_normalization_investigation.py
Phase 15 Follow-up: Investigate the missing normalization factor

FINDING: The +5 gate tests B/A ratio (structural), but absolute scale is wrong.
- Our A ≈ 1.66, PRZZ needs A ≈ 0.246
- Ratio: ~6.74x too large

This means there's a normalization factor we're missing.
Candidates:
1. Polynomial normalization (∫P² normalization)
2. Prefactor from Φ̂(0) / log N
3. Other mean-square normalization
"""

import numpy as np
from scipy import integrate
from src.ratios.j1_euler_maclaurin import (
    LaurentMode,
    compute_m1_with_mirror_assembly,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials


def investigate_polynomial_normalization():
    """Check if polynomial normalization explains the factor."""
    print("\n" + "=" * 70)
    print("INVESTIGATING POLYNOMIAL NORMALIZATION")
    print("=" * 70)

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 50)

        # Compute ∫P₁² and ∫P₂² and ∫P₁P₂
        def P1(u):
            return float(polys.P1.eval(np.array([u]))[0])

        def P2(u):
            return float(polys.P2.eval(np.array([u]))[0])

        int_P1_sq, _ = integrate.quad(lambda u: P1(u)**2, 0, 1)
        int_P2_sq, _ = integrate.quad(lambda u: P2(u)**2, 0, 1)
        int_P1P2, _ = integrate.quad(lambda u: P1(u)*P2(u), 0, 1)

        print(f"  ∫P₁² du = {int_P1_sq:.10f}")
        print(f"  ∫P₂² du = {int_P2_sq:.10f}")
        print(f"  ∫P₁P₂ du = {int_P1P2:.10f}")
        print(f"  √(∫P₁²×∫P₂²) = {np.sqrt(int_P1_sq * int_P2_sq):.10f}")

        # The normalization factor might be related to these
        # PRZZ might normalize by ∫P₁² × ∫P₂² or similar


def compute_scale_factor():
    """Compute what scale factor would fix the absolute values."""
    print("\n" + "=" * 70)
    print("COMPUTING REQUIRED SCALE FACTOR")
    print("=" * 70)

    # PRZZ target c values
    targets = {
        'kappa': {'c_target': 2.137454406, 'kappa_target': 0.417293962},
        'kappa_star': {'c_target': 1.938, 'kappa_target': 0.410},  # approximate
    }

    scale_factors = {}

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        A = result['exp_coefficient']
        B = result['constant_offset']
        c_computed = A * np.exp(R) + B

        c_target = targets[benchmark]['c_target']

        # Scale factor to get from c_computed to c_target
        scale = c_target / c_computed

        print(f"\n{benchmark.upper()} (R={R}):")
        print(f"  c computed: {c_computed:.10f}")
        print(f"  c target:   {c_target:.10f}")
        print(f"  Scale factor needed: {scale:.10f}")
        print(f"  1/scale = {1/scale:.10f}")

        scale_factors[benchmark] = scale

    # Are the scale factors related?
    print(f"\nScale factor ratio (κ/κ*): {scale_factors['kappa'] / scale_factors['kappa_star']:.6f}")

    return scale_factors


def test_with_scale_factor():
    """Test κ computation with the scale factor applied."""
    print("\n" + "=" * 70)
    print("κ WITH SCALE FACTOR APPLIED")
    print("=" * 70)

    # Use the average scale factor as a rough normalization
    scale_k = 2.137454406 / 14.483  # approximately
    scale_ks = 1.938 / 9.316  # approximately

    # Or use a single scale factor (average)
    avg_scale = (scale_k + scale_ks) / 2

    print(f"\nUsing per-benchmark scale factors:")

    for benchmark, scale in [("kappa", scale_k), ("kappa_star", scale_ks)]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        A = result['exp_coefficient']
        B = result['constant_offset']
        c_raw = A * np.exp(R) + B
        c_scaled = c_raw * scale

        kappa_scaled = 1 - np.log(c_scaled) / R

        print(f"\n{benchmark.upper()} (R={R}):")
        print(f"  c raw:    {c_raw:.6f}")
        print(f"  scale:    {scale:.6f}")
        print(f"  c scaled: {c_scaled:.6f}")
        print(f"  κ:        {kappa_scaled:.10f}")


def analyze_j12_magnitude():
    """
    The j12 piece dominates. Let's check if its magnitude is correct.

    In PRZZ, the formula involves:
    - 1/(α+β) = 1/(-2R) = -1/(2R)
    - (ζ'/ζ)² factor
    - ∫P₁P₂ du factor

    What is the expected magnitude?
    """
    print("\n" + "=" * 70)
    print("ANALYZING J12 MAGNITUDE")
    print("=" * 70)

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 50)

        # Get components
        from src.ratios.g_product_full import compute_zeta_factors
        zf = compute_zeta_factors(R, precision=100)

        # ∫P₁P₂
        def P1(u):
            return float(polys.P1.eval(np.array([u]))[0])
        def P2(u):
            return float(polys.P2.eval(np.array([u]))[0])

        int_P1P2, _ = integrate.quad(lambda u: P1(u)*P2(u), 0, 1)

        # Components
        divisor = -2 * R
        logderiv_sq = zf.logderiv_actual_squared

        j12_predicted = logderiv_sq * int_P1P2 / divisor

        print(f"  1/(-2R) = {1/divisor:.10f}")
        print(f"  (ζ'/ζ)² = {logderiv_sq:.10f}")
        print(f"  ∫P₁P₂   = {int_P1P2:.10f}")
        print(f"  j12 = (ζ'/ζ)² × ∫P₁P₂ / (-2R) = {j12_predicted:.10f}")

        # What j12 value does PRZZ expect?
        # From PRZZ: c = 2.137, and structure c = A×exp(R) + B with B/A ≈ 5
        # So A ≈ c / (exp(R) + 5) ≈ 0.246
        # If j12(-R) dominates I12(-), then j12(-R) ≈ A
        print(f"\n  PRZZ expected A ≈ {2.137 / (np.exp(R) + 5):.6f}")
        print(f"  Our A (from I12_minus) = 1.66")
        print(f"  Ratio = {1.66 / 0.246:.2f}x")


def check_przz_prefactor():
    """
    Check if PRZZ has a prefactor we're missing.

    From PRZZ formula structure, there should be:
    - T Φ̂(0) / log N factor
    - Normalization from |ζ(1/2+it)|² mean

    These cancel in the ratio κ computation but appear in absolute c.
    """
    print("\n" + "=" * 70)
    print("PRZZ PREFACTOR ANALYSIS")
    print("=" * 70)

    print("""
In PRZZ, the main-term formula for c involves:

  c = ∫∫ [terms] × prefactors

The prefactors include:
1. T Φ̂(0) / log N - from the integral over t
2. Normalization from the mean square structure

For the +5 gate (ratio test), these cancel.
For absolute κ, we need the correct prefactor.

From our analysis:
- Scale factor needed ≈ 0.147 (κ) and 0.208 (κ*)
- These differ by ~40%, suggesting polynomial-dependent normalization

Possible sources:
1. ∫P₁P₂ normalization (we're using raw polynomial products)
2. Cross-term counting (we might be double-counting)
3. Missing 1/(log N) factor that appears in PRZZ
""")


def what_kappa_do_we_achieve():
    """
    Given that we have the right B/A ratio but wrong absolute scale,
    what κ do we get if we use the PRZZ-expected scale?
    """
    print("\n" + "=" * 70)
    print("WHAT κ DO WE ACHIEVE WITH PROPER SCALING?")
    print("=" * 70)

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        BA = result['B_over_A']

        # If we trust B/A ≈ 5.04 (or 4.93 for κ*), and we know
        # c = A × (exp(R) + B/A)
        # Then κ = 1 - log(c)/R = 1 - log(A)/R - log(exp(R) + B/A)/R
        #        = 1 - log(A)/R - (R + log(1 + B/(A×exp(R))))/R
        #        ≈ 1 - log(A)/R - 1 - (B/A)/exp(R)/R  (for small B/(A×exp(R)))

        # The key question: what is log(A)/R?
        # If we don't know A absolutely, but know c_target:
        # A = c_target / (exp(R) + B/A)

        target_kappa = 0.417293962 if benchmark == "kappa" else 0.410
        c_target = np.exp(R * (1 - target_kappa))

        A_expected = c_target / (np.exp(R) + BA)
        c_with_expected_A = A_expected * (np.exp(R) + BA)
        kappa_result = 1 - np.log(c_with_expected_A) / R

        print(f"\n{benchmark.upper()} (R={R}):")
        print(f"  Our B/A = {BA:.6f}")
        print(f"  Target κ = {target_kappa}")
        print(f"  => c_target = {c_target:.6f}")
        print(f"  => A expected = c_target / (exp(R) + B/A) = {A_expected:.6f}")
        print(f"  => c = A × (exp(R) + B/A) = {c_with_expected_A:.6f}")
        print(f"  => κ = 1 - log(c)/R = {kappa_result:.10f}")

        # What κ would we get if B/A were exactly 5?
        BA_target = 5.0
        A_if_BA_5 = c_target / (np.exp(R) + BA_target)
        c_if_BA_5 = A_if_BA_5 * (np.exp(R) + BA_target)
        kappa_if_BA_5 = 1 - np.log(c_if_BA_5) / R

        print(f"\n  If B/A were exactly 5:")
        print(f"    A would be = {A_if_BA_5:.6f}")
        print(f"    κ would be = {kappa_if_BA_5:.10f}")

        # The deviation in κ due to B/A deviation
        delta_BA = BA - BA_target
        print(f"\n  Our B/A deviation: {delta_BA:+.6f}")
        print(f"  Impact on κ: {(kappa_result - kappa_if_BA_5)*100:+.6f} percentage points")


if __name__ == "__main__":
    investigate_polynomial_normalization()
    scale_factors = compute_scale_factor()
    test_with_scale_factor()
    analyze_j12_magnitude()
    check_przz_prefactor()
    what_kappa_do_we_achieve()
