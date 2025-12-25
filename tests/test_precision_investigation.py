"""
tests/test_precision_investigation.py
Phase 15 Follow-up: Investigate remaining ~1% gap

HYPOTHESIS 1: Numerical precision
- Test with higher mpmath precision
- Test with tighter quadrature tolerance

HYPOTHESIS 2: J13/J14 also use Laurent approximation
- J13 and J14 use (1/R + γ) for their ζ'/ζ factors
- These might also need actual numerical values

HYPOTHESIS 3: Higher-order series terms
- We might be dropping terms in the Laurent expansion
"""

import numpy as np
from src.ratios.j1_euler_maclaurin import (
    LaurentMode,
    compute_m1_with_mirror_assembly,
    j12_as_integral,
    j13_as_integral,
    j14_as_integral,
    EULER_MASCHERONI,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials
from src.ratios.g_product_full import compute_zeta_factors


def test_mpmath_precision_effect():
    """Test if higher mpmath precision changes the results."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1A: MPMATH PRECISION EFFECT")
    print("=" * 70)

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 50)

        for precision in [50, 100, 200, 500]:
            zf = compute_zeta_factors(R, precision=precision)

            # The actual logderiv squared is what we use
            actual_sq = zf.logderiv_actual_squared

            print(f"  precision={precision:3d}: (ζ'/ζ)² = {actual_sq:.15f}")

        # Check convergence
        zf_50 = compute_zeta_factors(R, precision=50)
        zf_500 = compute_zeta_factors(R, precision=500)
        diff = abs(zf_500.logderiv_actual_squared - zf_50.logderiv_actual_squared)
        print(f"  Difference (500 vs 50): {diff:.2e}")


def test_j13_j14_laurent_approximation():
    """
    Check if J13/J14 Laurent approximation is also a source of error.

    Currently j13_as_integral and j14_as_integral use:
        beta_logderiv = 1.0 / R + EULER_MASCHERONI  (Laurent approximation)

    Should they use actual (ζ'/ζ)(1-R)?
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: J13/J14 LAURENT APPROXIMATION")
    print("=" * 70)

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 50)

        # Get actual ζ'/ζ value
        zf = compute_zeta_factors(R, precision=100)
        actual_logderiv = zf.logderiv_actual  # NOT squared, just the single factor
        laurent_logderiv = 1.0 / R + EULER_MASCHERONI

        print(f"  Actual (ζ'/ζ)(1-R) = {actual_logderiv:.10f}")
        print(f"  Laurent (1/R + γ) = {laurent_logderiv:.10f}")
        print(f"  Ratio = {actual_logderiv / laurent_logderiv:.6f}")
        print(f"  Error in Laurent = {(laurent_logderiv - actual_logderiv) / actual_logderiv * 100:+.2f}%")

        # J13 and J14 each use ONE factor of (ζ'/ζ)
        # So the error in J13+J14 from using Laurent is approximately:
        # (actual - laurent) / actual for each term

        # Compute the impact on the full assembly
        print(f"\n  Impact on J13/J14:")
        print(f"    J13/J14 use single (ζ'/ζ) factor")
        print(f"    Laurent underestimates by {(1 - laurent_logderiv/actual_logderiv) * 100:.2f}%")


def compute_with_actual_j13_j14(R: float, theta: float, polys, precision: int = 100):
    """
    Compute mirror assembly with ACTUAL ζ'/ζ values in J13/J14.

    This is a modified version that replaces Laurent with actual values
    in ALL the ζ'/ζ factors, not just J12.
    """
    from scipy import integrate
    from src.ratios.j1_euler_maclaurin import (
        _extract_poly_funcs,
        j11_as_integral,
        j15_as_integral,
        compute_I12_components,
        compute_I34_components,
    )
    from src.ratios.g_product_full import compute_zeta_factors

    P1_func, P2_func = _extract_poly_funcs(polys)

    # Get actual logderiv values
    zf_plus = compute_zeta_factors(R, precision=precision)
    zf_minus = compute_zeta_factors(R, precision=precision)  # For -R, we use same |R|

    actual_logderiv_plus = zf_plus.logderiv_actual  # Single factor
    actual_logderiv_minus = zf_minus.logderiv_actual  # Same value at |R|

    # Modified J13 with actual value
    def j13_actual(R_val, actual_ld):
        def integrand(u):
            return (1.0 - u) * P1_func(u) * P2_func(u)
        poly_integral, _ = integrate.quad(integrand, 0, 1)
        prefactor = -1.0 / theta
        return prefactor * actual_ld * poly_integral

    # Modified J14 with actual value
    def j14_actual(R_val, actual_ld):
        def integrand(u):
            return (1.0 - u) * P1_func(u) * P2_func(u)
        poly_integral, _ = integrate.quad(integrand, 0, 1)
        prefactor = -1.0 / theta
        return prefactor * actual_ld * poly_integral

    # I₁₂ components at +R (using ACTUAL_LOGDERIV for j12)
    i12_plus = compute_I12_components(R, theta, P1_func=P1_func, P2_func=P2_func,
                                       laurent_mode=LaurentMode.ACTUAL_LOGDERIV)

    # I₁₂ components at -R
    i12_minus = compute_I12_components(-R, theta, P1_func=P1_func, P2_func=P2_func,
                                        laurent_mode=LaurentMode.ACTUAL_LOGDERIV)

    # I₃₄ with ACTUAL logderiv (modified)
    j13_plus_actual = j13_actual(R, actual_logderiv_plus)
    j14_plus_actual = j14_actual(R, actual_logderiv_plus)

    # Original I₃₄ for comparison
    i34_plus_original = compute_I34_components(R, theta, P1_func=P1_func, P2_func=P2_func)

    # Totals
    i12_plus_total = sum(i12_plus.values())
    i12_minus_total = sum(i12_minus.values())
    i34_plus_original_total = sum(i34_plus_original.values())
    i34_plus_actual_total = j13_plus_actual + j14_plus_actual

    # Mirror assembly
    K = 3
    m = np.exp(R) + (2 * K - 1)

    # With original I34
    assembled_original = i12_plus_total + m * i12_minus_total + i34_plus_original_total
    A_original = i12_minus_total
    B_original = i12_plus_total + i34_plus_original_total + (2 * K - 1) * i12_minus_total
    BA_original = B_original / A_original

    # With actual I34
    assembled_actual = i12_plus_total + m * i12_minus_total + i34_plus_actual_total
    A_actual = i12_minus_total
    B_actual = i12_plus_total + i34_plus_actual_total + (2 * K - 1) * i12_minus_total
    BA_actual = B_actual / A_actual

    return {
        'original': {
            'B_over_A': BA_original,
            'j13': i34_plus_original['j13'],
            'j14': i34_plus_original['j14'],
        },
        'actual': {
            'B_over_A': BA_actual,
            'j13': j13_plus_actual,
            'j14': j14_plus_actual,
        },
        'j13_j14_change': (i34_plus_actual_total - i34_plus_original_total),
    }


def test_full_actual_logderiv():
    """Test using actual ζ'/ζ in J13/J14 as well as J12."""
    print("\n" + "=" * 70)
    print("TEST: ACTUAL LOGDERIV IN ALL TERMS (J12, J13, J14)")
    print("=" * 70)

    target = 5.0

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 50)

        result = compute_with_actual_j13_j14(R, 4.0/7.0, polys)

        delta_original = (result['original']['B_over_A'] - target) / target * 100
        delta_actual = (result['actual']['B_over_A'] - target) / target * 100

        print(f"  Original (J13/J14 use Laurent):")
        print(f"    B/A = {result['original']['B_over_A']:.6f}, δ = {delta_original:+.2f}%")
        print(f"    j13 = {result['original']['j13']:.6f}")
        print(f"    j14 = {result['original']['j14']:.6f}")

        print(f"\n  With actual ζ'/ζ in J13/J14:")
        print(f"    B/A = {result['actual']['B_over_A']:.6f}, δ = {delta_actual:+.2f}%")
        print(f"    j13 = {result['actual']['j13']:.6f}")
        print(f"    j14 = {result['actual']['j14']:.6f}")

        print(f"\n  Improvement: {delta_original - delta_actual:+.4f} percentage points")


def test_quadrature_tolerance():
    """Test if tighter quadrature tolerance changes results."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1B: QUADRATURE TOLERANCE")
    print("=" * 70)

    from scipy import integrate

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 50)

        # Extract polynomial functions
        from src.ratios.j1_euler_maclaurin import _extract_poly_funcs
        P1_func, P2_func = _extract_poly_funcs(polys)

        def integrand(u):
            return P1_func(u) * P2_func(u)

        for epsabs in [1e-8, 1e-10, 1e-12, 1e-14]:
            result, error = integrate.quad(integrand, 0, 1, epsabs=epsabs)
            print(f"  epsabs={epsabs:.0e}: ∫P₁P₂ = {result:.15f} (±{error:.2e})")


def investigate_asymmetry():
    """
    Investigate why κ overshoots (+0.84%) while κ* undershoots (-1.33%).

    This asymmetry might reveal the source of remaining error.
    """
    print("\n" + "=" * 70)
    print("ASYMMETRY INVESTIGATION: κ vs κ*")
    print("=" * 70)

    results = {}
    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        zf = compute_zeta_factors(R, precision=100)

        results[benchmark] = {
            'R': R,
            'B_over_A': result['B_over_A'],
            'delta': (result['B_over_A'] - 5) / 5 * 100,
            'A': result['exp_coefficient'],
            'B': result['constant_offset'],
            'D': result['D'],
            'i12_plus': result['i12_plus_total'],
            'i12_minus': result['i12_minus_total'],
            'i34_plus': result['i34_plus_total'],
            'actual_logderiv_sq': zf.logderiv_actual_squared,
            'laurent_logderiv_sq': zf.logderiv_laurent_squared,
        }

    print("\nComponent Comparison:")
    print("-" * 70)
    print(f"{'Component':<25} {'κ':<15} {'κ*':<15} {'Ratio κ/κ*':<15}")
    print("-" * 70)

    for key in ['R', 'B_over_A', 'delta', 'A', 'B', 'D',
                'i12_plus', 'i12_minus', 'i34_plus',
                'actual_logderiv_sq', 'laurent_logderiv_sq']:
        k = results['kappa'][key]
        ks = results['kappa_star'][key]
        ratio = k / ks if abs(ks) > 1e-14 else float('nan')

        if key == 'delta':
            print(f"{key:<25} {k:+14.4f}% {ks:+14.4f}% {'':<15}")
        else:
            print(f"{key:<25} {k:<15.6f} {ks:<15.6f} {ratio:<15.4f}")

    print("-" * 70)

    # Check the R-scaling
    print("\nR-Scaling Analysis:")
    R_k = results['kappa']['R']
    R_ks = results['kappa_star']['R']

    # If the error were purely proportional to R, we'd expect:
    # delta_k / delta_ks ≈ some function of R_k / R_ks
    print(f"  R_κ / R_κ* = {R_k / R_ks:.4f}")
    print(f"  δ_κ / δ_κ* = {results['kappa']['delta'] / results['kappa_star']['delta']:.4f}")


if __name__ == "__main__":
    test_mpmath_precision_effect()
    test_j13_j14_laurent_approximation()
    test_full_actual_logderiv()
    test_quadrature_tolerance()
    investigate_asymmetry()
