"""
tests/test_r_dependent_correction.py
Phase 15 Follow-up: Test R-dependent correction term

From the investigation:
- At R=1.0: c=+0.142
- At R=1.2: c=+0.009 (near zero)
- At R=1.3: c=-0.040
- At R=1.5: c=-0.111

This suggests a correction of the form:
  c(R) ≈ a × (R - R₀)

where R₀ ≈ 1.12 is the zero-crossing point.

Let's fit this and test.
"""

import numpy as np
from scipy.optimize import curve_fit
from src.ratios.j1_euler_maclaurin import (
    LaurentMode,
    compute_m1_with_mirror_assembly,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials


def fit_r_correction():
    """Fit the R-dependent correction."""
    print("\n" + "=" * 70)
    print("FIT R-DEPENDENT CORRECTION")
    print("=" * 70)

    polys = load_przz_k3_polynomials("kappa")

    # Collect data points
    R_values = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
    corrections = []

    for R in R_values:
        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )
        delta = result['D'] / result['exp_coefficient']
        corrections.append(-delta)  # correction to make B/A = 5

    R_arr = np.array(R_values)
    c_arr = np.array(corrections)

    # Fit linear: c = a * (R - R0)
    def linear_model(R, a, R0):
        return a * (R - R0)

    popt, pcov = curve_fit(linear_model, R_arr, c_arr, p0=[-0.1, 1.2])
    a, R0 = popt

    print(f"\nLinear fit: c(R) = {a:.6f} × (R - {R0:.6f})")
    print(f"\nPredictions vs Actual:")
    print("-" * 60)
    print(f"{'R':<8} {'Actual c':<12} {'Predicted c':<12} {'Error':<12}")
    print("-" * 60)

    for R, c_actual in zip(R_values, corrections):
        c_pred = linear_model(R, a, R0)
        error = c_pred - c_actual
        print(f"{R:<8.4f} {c_actual:+12.6f} {c_pred:+12.6f} {error:+12.6f}")

    return a, R0


def test_corrected_formula(a, R0):
    """Test the corrected formula: target = 5 + c(R) where c = a×(R-R₀)."""
    print("\n" + "=" * 70)
    print("TEST: CORRECTED MIRROR FORMULA")
    print("=" * 70)
    print(f"Using correction: c(R) = {a:.6f} × (R - {R0:.6f})")

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        ba = result['B_over_A']
        correction = a * (R - R0)
        corrected_target = 5 + correction

        # Original error (target = 5)
        delta_original = (ba - 5) / 5 * 100

        # Corrected error
        delta_corrected = (ba - corrected_target) / corrected_target * 100

        print(f"\n{benchmark.upper()} (R={R}):")
        print(f"  B/A = {ba:.6f}")
        print(f"  Correction c(R) = {correction:+.6f}")
        print(f"  Original target = 5.000, δ = {delta_original:+.4f}%")
        print(f"  Corrected target = {corrected_target:.6f}, δ = {delta_corrected:+.4f}%")


def test_quadratic_correction():
    """Also test a quadratic correction: c(R) = a×(R-R₀)²."""
    print("\n" + "=" * 70)
    print("QUADRATIC CORRECTION TEST")
    print("=" * 70)

    polys = load_przz_k3_polynomials("kappa")

    R_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    corrections = []

    for R in R_values:
        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )
        delta = result['D'] / result['exp_coefficient']
        corrections.append(-delta)

    R_arr = np.array(R_values)
    c_arr = np.array(corrections)

    # Fit quadratic: c = a*(R-R0) + b*(R-R0)²
    def quad_model(R, a, R0, b):
        return a * (R - R0) + b * (R - R0)**2

    popt, pcov = curve_fit(quad_model, R_arr, c_arr, p0=[-0.1, 1.2, 0])
    a, R0, b = popt

    print(f"Quadratic fit: c(R) = {a:.6f}×(R-{R0:.4f}) + {b:.6f}×(R-{R0:.4f})²")

    residuals = []
    for R, c_actual in zip(R_values, corrections):
        c_pred = quad_model(R, a, R0, b)
        residuals.append(c_pred - c_actual)

    rms = np.sqrt(np.mean(np.array(residuals)**2))
    print(f"RMS residual: {rms:.6f}")


def analyze_correction_source():
    """
    Analyze where the R-dependent error comes from.

    The formula is: B/A = (i12_plus + i34_plus)/A + 5
                       = D/A + 5
                       = delta + 5

    So delta = D/A = (i12_plus + i34_plus) / i12_minus

    Let's see how each piece scales with R.
    """
    print("\n" + "=" * 70)
    print("ANALYZE: SOURCE OF R-DEPENDENT ERROR")
    print("=" * 70)

    polys = load_przz_k3_polynomials("kappa")

    R_values = [1.0, 1.2, 1.3036, 1.5]

    print(f"\n{'R':<8} {'i12+':<12} {'i34+':<12} {'D':<12} {'A=i12-':<12} {'D/A':<12}")
    print("-" * 70)

    for R in R_values:
        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        i12p = result['i12_plus_total']
        i34p = result['i34_plus_total']
        D = result['D']
        A = result['exp_coefficient']
        delta = D / A

        print(f"{R:<8.4f} {i12p:<12.6f} {i34p:<12.6f} {D:<12.6f} {A:<12.6f} {delta:+12.6f}")

    # Look at individual pieces
    print("\n\nIndividual pieces at R=1.3036:")
    result = compute_m1_with_mirror_assembly(
        theta=4.0/7.0, R=1.3036, polys=polys, K=3,
        laurent_mode=LaurentMode.ACTUAL_LOGDERIV
    )

    print("\nI12+ pieces:")
    for k, v in result['i12_plus_pieces'].items():
        print(f"  {k}: {v:+.6f}")

    print("\nI12- pieces (at -R evaluation):")
    for k, v in result['i12_minus_pieces'].items():
        print(f"  {k}: {v:+.6f}")

    print("\nI34+ pieces:")
    for k, v in result['i34_plus_pieces'].items():
        print(f"  {k}: {v:+.6f}")


if __name__ == "__main__":
    a, R0 = fit_r_correction()
    test_corrected_formula(a, R0)
    test_quadratic_correction()
    analyze_correction_source()
