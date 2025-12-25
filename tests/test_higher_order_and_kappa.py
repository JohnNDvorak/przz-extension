"""
tests/test_higher_order_and_kappa.py
Phase 15 Follow-up: Higher-order corrections and κ computation

1. Experiment with higher-order corrections:
   - Stieltjes constants (γ₁, γ₂, ...)
   - Euler-Maclaurin remainder terms
   - Higher-order Laurent expansion

2. Compute the implied κ bound from our current numbers
"""

import numpy as np
from src.ratios.j1_euler_maclaurin import (
    LaurentMode,
    compute_m1_with_mirror_assembly,
    EULER_MASCHERONI,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials
from src.ratios.g_product_full import compute_zeta_factors

# Stieltjes constants (from literature/mpmath)
# γ_n appears in the Laurent expansion: ζ(s) = 1/(s-1) + Σ ((-1)^n/n!) γ_n (s-1)^n
GAMMA_0 = EULER_MASCHERONI  # γ₀ = γ ≈ 0.5772156649
GAMMA_1 = -0.0728158454836767248605863758749997319  # γ₁
GAMMA_2 = -0.0096903631928723184845303860352125293  # γ₂
GAMMA_3 = 0.0020538344203033458661600465427533842   # γ₃


def compute_logderiv_with_stieltjes(R: float, order: int = 1) -> float:
    """
    Compute (ζ'/ζ)(1-R) using Stieltjes constant expansion.

    The Laurent expansion of ζ'/ζ around s=1:
    (ζ'/ζ)(s) = -1/(s-1) + γ₀ - γ₁(s-1) + (γ₁² - γ₂)(s-1)²/2 + ...

    At s = 1-R (so s-1 = -R):
    (ζ'/ζ)(1-R) = 1/R + γ₀ + γ₁R + O(R²)
    """
    epsilon = -R  # s - 1 = -R

    # Order 0: just the pole
    result = -1/epsilon  # = 1/R

    if order >= 1:
        result += GAMMA_0  # + γ

    if order >= 2:
        result += -GAMMA_1 * epsilon  # - γ₁(-R) = γ₁R

    if order >= 3:
        # The next term involves γ₁² and γ₂
        # From the expansion of log(ζ(s)) and differentiating
        coeff2 = (GAMMA_1**2 - GAMMA_2) / 2
        result += coeff2 * epsilon**2

    if order >= 4:
        # Higher order terms become complex, skip for now
        pass

    return result


def test_stieltjes_correction():
    """Test using Stieltjes constant corrections."""
    print("\n" + "=" * 70)
    print("HIGHER-ORDER CORRECTION: STIELTJES CONSTANTS")
    print("=" * 70)

    print("\nStieltjes constants:")
    print(f"  γ₀ = {GAMMA_0:.10f}")
    print(f"  γ₁ = {GAMMA_1:.10f}")
    print(f"  γ₂ = {GAMMA_2:.10f}")

    for R in [1.3036, 1.1167]:
        print(f"\nR = {R}:")
        print("-" * 50)

        # Get actual value
        zf = compute_zeta_factors(R, precision=100)
        actual = zf.logderiv_actual

        print(f"  Actual (ζ'/ζ)(1-R) = {actual:.10f}")
        print()

        for order in range(1, 5):
            approx = compute_logderiv_with_stieltjes(R, order)
            error = (approx - actual) / actual * 100
            print(f"  Order {order}: {approx:.10f}  (error: {error:+.4f}%)")


def compute_kappa_from_mirror_assembly():
    """
    Compute the implied κ bound from our mirror assembly results.

    PRZZ formula: κ ≥ 1 - log(c)/R

    From mirror assembly: c = A × exp(R) + B
    where A and B come from compute_m1_with_mirror_assembly()
    """
    print("\n" + "=" * 70)
    print("COMPUTING κ BOUND FROM CURRENT NUMBERS")
    print("=" * 70)

    # PRZZ target values
    targets = {
        'kappa': {'R': 1.3036, 'kappa_target': 0.417293962, 'c_target': 2.137454406},
        'kappa_star': {'R': 1.1167, 'kappa_target': 0.410, 'c_target': 1.938},  # approximate
    }

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 60)

        # Get mirror assembly result with ACTUAL_LOGDERIV
        result = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=R, polys=polys, K=3,
            laurent_mode=LaurentMode.ACTUAL_LOGDERIV
        )

        A = result['exp_coefficient']
        B = result['constant_offset']

        # c = A × exp(R) + B
        c_computed = A * np.exp(R) + B

        # κ = 1 - log(c)/R
        kappa_computed = 1 - np.log(c_computed) / R

        # Compare with targets
        target = targets[benchmark]

        print(f"  A (exp coefficient): {A:.10f}")
        print(f"  B (constant offset): {B:.10f}")
        print(f"  B/A: {B/A:.6f} (target: 5)")
        print()
        print(f"  c = A×exp(R) + B: {c_computed:.10f}")
        if 'c_target' in target:
            c_target = target['c_target']
            c_error = (c_computed - c_target) / c_target * 100
            print(f"  c target (PRZZ): {c_target:.10f}")
            print(f"  c error: {c_error:+.4f}%")
        print()
        print(f"  κ = 1 - log(c)/R: {kappa_computed:.10f}")
        if 'kappa_target' in target:
            kappa_target = target['kappa_target']
            kappa_error = (kappa_computed - kappa_target) * 100  # percentage points
            print(f"  κ target (PRZZ): {kappa_target:.10f}")
            print(f"  κ difference: {kappa_error:+.6f} percentage points")


def test_different_laurent_modes_kappa():
    """Compare κ values across different Laurent modes."""
    print("\n" + "=" * 70)
    print("κ COMPARISON ACROSS LAURENT MODES")
    print("=" * 70)

    print(f"\n{'Mode':<20} {'κ (kappa)':<15} {'κ (kappa*)':<15}")
    print("-" * 60)

    for mode in LaurentMode:
        kappas = {}
        for benchmark in ["kappa", "kappa_star"]:
            polys = load_przz_k3_polynomials(benchmark)
            R = polys.R

            result = compute_m1_with_mirror_assembly(
                theta=4.0/7.0, R=R, polys=polys, K=3,
                laurent_mode=mode
            )

            A = result['exp_coefficient']
            B = result['constant_offset']
            c = A * np.exp(R) + B
            kappa_val = 1 - np.log(c) / R

            kappas[benchmark] = kappa_val

        print(f"{mode.value:<20} {kappas['kappa']:<15.10f} {kappas['kappa_star']:<15.10f}")

    print("-" * 60)
    print(f"{'PRZZ Target':<20} {'0.417293962':<15} {'~0.410':<15}")


def experiment_with_j12_stieltjes():
    """
    Try using Stieltjes-corrected values in j12.

    Currently j12 uses actual (ζ'/ζ)² computed numerically.
    What if we use the Stieltjes expansion instead?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: J12 WITH STIELTJES CORRECTION")
    print("=" * 70)

    for benchmark in ["kappa", "kappa_star"]:
        polys = load_przz_k3_polynomials(benchmark)
        R = polys.R

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 50)

        # Get actual values
        zf = compute_zeta_factors(R, precision=100)

        print(f"  (ζ'/ζ)² values:")
        print(f"    Laurent order 1 (1/R+γ)²: {(1/R + GAMMA_0)**2:.10f}")
        print(f"    Laurent order 2 (+γ₁R): {compute_logderiv_with_stieltjes(R, 2)**2:.10f}")
        print(f"    Laurent order 3: {compute_logderiv_with_stieltjes(R, 3)**2:.10f}")
        print(f"    Actual (mpmath): {zf.logderiv_actual_squared:.10f}")

        # The actual value is significantly larger than any Laurent approximation
        # This confirms we need the actual numerical value


def analyze_c_structure():
    """
    Analyze the structure of c more carefully.

    c = A × exp(R) + B

    For PRZZ target κ = 0.417293962:
    c_target = exp(R × (1 - κ)) = exp(1.3036 × 0.582706038) = 2.137454...

    What values of A and B does this imply?
    """
    print("\n" + "=" * 70)
    print("ANALYZING c STRUCTURE")
    print("=" * 70)

    R = 1.3036
    kappa_target = 0.417293962

    # Target c
    c_target = np.exp(R * (1 - kappa_target))
    print(f"\nPRZZ target:")
    print(f"  R = {R}")
    print(f"  κ = {kappa_target}")
    print(f"  c = exp(R(1-κ)) = {c_target:.10f}")

    # Our computed values
    polys = load_przz_k3_polynomials("kappa")
    result = compute_m1_with_mirror_assembly(
        theta=4.0/7.0, R=R, polys=polys, K=3,
        laurent_mode=LaurentMode.ACTUAL_LOGDERIV
    )

    A = result['exp_coefficient']
    B = result['constant_offset']
    c_computed = A * np.exp(R) + B

    print(f"\nOur computation:")
    print(f"  A = {A:.10f}")
    print(f"  B = {B:.10f}")
    print(f"  c = A×exp(R) + B = {c_computed:.10f}")

    # What would A and B need to be to hit the target?
    # If B/A = 5 exactly, then c = A × (exp(R) + 5)
    # So A = c / (exp(R) + 5)
    A_needed = c_target / (np.exp(R) + 5)
    B_needed = 5 * A_needed

    print(f"\nTo hit target with B/A = 5 exactly:")
    print(f"  A needed = {A_needed:.10f}")
    print(f"  B needed = {B_needed:.10f}")
    print(f"  A ratio (ours/needed): {A / A_needed:.6f}")

    # What if B/A isn't exactly 5?
    # Our current B/A
    BA_ours = B / A
    print(f"\nOur B/A = {BA_ours:.6f}")

    # To get c_target with our A:
    B_needed_given_A = c_target - A * np.exp(R)
    BA_needed_given_A = B_needed_given_A / A
    print(f"B/A needed to hit c_target with our A: {BA_needed_given_A:.6f}")


if __name__ == "__main__":
    test_stieltjes_correction()
    compute_kappa_from_mirror_assembly()
    test_different_laurent_modes_kappa()
    experiment_with_j12_stieltjes()
    analyze_c_structure()
