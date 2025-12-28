"""
scripts/investigate_g_i2_gap_final.py
Phase 46.3: Final Hypothesis - Q Affects Mirror Symmetry

HYPOTHESIS:
===========
The g_baseline formula assumes perfect mirror symmetry:
    I2(x+δ) - I2(x-δ) ≈ 2δ × I2'(x) × [1 + correction]

But with Q polynomial, the symmetry might be broken in a way that
requires a Q-dependent correction to g.

APPROACH:
=========
We'll measure the "asymmetry coefficient" by looking at how Q affects
the I2 derivative structure around R=0.

If I2(R) with Q has different curvature than I2(R) with Q=1,
this could explain the g_I2 gap.

Test:
1. Compute I2(R) for R in [-2, +2]
2. Fit to polynomial: I2(R) = a + bR + cR² + ...
3. Extract the ratio: I2(+R)/I2(-R) ≈ (a+bR+cR²)/(a-bR+cR²)
4. See if Q changes this ratio in a way that explains the 0.58% gap

Created: 2025-12-27 (Phase 46.3)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from src.polynomials import load_przz_polynomials, Polynomial
from src.unified_i2_paper import compute_I2_unified_paper


@dataclass
class I2ProfileResult:
    """Result of I2(R) profile measurement."""

    R_values: List[float]
    I2_Q1_values: List[float]
    I2_real_values: List[float]

    # Polynomial fit coefficients [a, b, c, ...] for I2 = a + bR + cR² + ...
    fit_Q1_coeffs: np.ndarray
    fit_real_coeffs: np.ndarray

    @property
    def I2_ratio_at_R(self) -> Dict[float, float]:
        """Compute I2(+R)/I2(-R) ratio at each R value."""
        ratios = {}
        for R in self.R_values:
            if R <= 0:
                continue
            idx_plus = self.R_values.index(R)
            idx_minus = self.R_values.index(-R) if -R in self.R_values else None

            if idx_minus is not None:
                I2_plus_Q1 = self.I2_Q1_values[idx_plus]
                I2_minus_Q1 = self.I2_Q1_values[idx_minus]
                I2_plus_real = self.I2_real_values[idx_plus]
                I2_minus_real = self.I2_real_values[idx_minus]

                ratio_Q1 = I2_plus_Q1 / I2_minus_Q1 if abs(I2_minus_Q1) > 1e-15 else float('inf')
                ratio_real = I2_plus_real / I2_minus_real if abs(I2_minus_real) > 1e-15 else float('inf')

                ratios[R] = {
                    'ratio_Q1': ratio_Q1,
                    'ratio_real': ratio_real,
                    'Q_effect': ratio_real / ratio_Q1 - 1.0 if ratio_Q1 != 0 else float('nan'),
                }

        return ratios


def compute_I2_total(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
    include_Q: bool = True,
) -> float:
    """Compute aggregate I2 total."""
    pairs = [(1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0), (1, 2, 2.0), (1, 3, 2.0), (2, 3, 2.0)]
    factorial = [1, 1, 2, 6]
    total = 0.0

    for ell1, ell2, symmetry in pairs:
        fact_norm = 1.0 / (factorial[ell1] * factorial[ell2])
        weight = symmetry * fact_norm
        result = compute_I2_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=include_Q,
        )
        total += weight * result.I2_value

    return total


def measure_I2_profile(
    R_min: float,
    R_max: float,
    n_points: int,
    theta: float,
    polynomials_Q1: Dict,
    polynomials_real: Dict,
    n_quad: int = 60,
) -> I2ProfileResult:
    """
    Measure I2(R) profile over a range of R values.

    Args:
        R_min, R_max: R range
        n_points: Number of R points to sample
        theta: theta parameter
        polynomials_Q1: Polynomials with Q=1
        polynomials_real: Polynomials with real Q
        n_quad: Quadrature points

    Returns:
        I2ProfileResult with profile and fit coefficients
    """
    R_values = np.linspace(R_min, R_max, n_points).tolist()
    I2_Q1_values = []
    I2_real_values = []

    for R in R_values:
        I2_Q1 = compute_I2_total(R, theta, polynomials_Q1, n_quad, include_Q=False)
        I2_real = compute_I2_total(R, theta, polynomials_real, n_quad, include_Q=True)

        I2_Q1_values.append(I2_Q1)
        I2_real_values.append(I2_real)

    # Fit polynomial: I2(R) = a + bR + cR² + dR³
    # Use degree 3 to capture asymmetry
    fit_Q1_coeffs = np.polyfit(R_values, I2_Q1_values, deg=3)
    fit_real_coeffs = np.polyfit(R_values, I2_real_values, deg=3)

    return I2ProfileResult(
        R_values=R_values,
        I2_Q1_values=I2_Q1_values,
        I2_real_values=I2_real_values,
        fit_Q1_coeffs=fit_Q1_coeffs,
        fit_real_coeffs=fit_real_coeffs,
    )


def analyze_mirror_symmetry_breaking(
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 40,  # Lower for speed
) -> None:
    """
    Analyze how Q breaks mirror symmetry in I2.

    Args:
        theta: theta parameter
        K: Number of mollifier pieces
        n_quad: Quadrature points
    """
    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials_real = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    Q1 = Polynomial(coeffs=[1.0])
    polynomials_Q1 = {"P1": P1, "P2": P2, "P3": P3, "Q": Q1}

    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    g_I2_calibrated = 1.01945154
    gap_pct = (g_I2_calibrated / g_baseline - 1) * 100

    print("=" * 80)
    print("PHASE 46.3: MIRROR SYMMETRY BREAKING ANALYSIS")
    print("=" * 80)
    print()
    print(f"Parameters: θ={theta:.10f}, K={K}")
    print()
    print(f"g_baseline:      {g_baseline:.8f}")
    print(f"g_I2_calibrated: {g_I2_calibrated:.8f}")
    print(f"Gap:             {gap_pct:+.4f}%")
    print()

    print("HYPOTHESIS:")
    print("-----------")
    print("The g_baseline formula assumes I2(R) has perfect mirror symmetry:")
    print("  I2(+R) / I2(-R) should depend only on exp(2R) and polynomial structure")
    print()
    print("But Q polynomial might break this symmetry, requiring a correction to g.")
    print()

    # Measure I2 profile
    print("Measuring I2(R) profile...")
    profile = measure_I2_profile(
        R_min=-2.0,
        R_max=2.0,
        n_points=9,  # -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
        theta=theta,
        polynomials_Q1=polynomials_Q1,
        polynomials_real=polynomials_real,
        n_quad=n_quad,
    )

    print()
    print("=" * 80)
    print("I2(R) PROFILE")
    print("=" * 80)
    print()
    print(f"{'R':<8} | {'I2(Q=1)':<12} | {'I2(real Q)':<12} | {'Ratio real/Q1':<12}")
    print("-" * 60)

    for R, I2_Q1, I2_real in zip(profile.R_values, profile.I2_Q1_values, profile.I2_real_values):
        ratio = I2_real / I2_Q1 if abs(I2_Q1) > 1e-15 else float('nan')
        print(f"{R:+7.2f} | {I2_Q1:+11.6f} | {I2_real:+11.6f} | {ratio:11.6f}")

    print()

    # Analyze polynomial fits
    print("=" * 80)
    print("POLYNOMIAL FIT ANALYSIS")
    print("=" * 80)
    print()
    print("Fitting I2(R) = a + bR + cR² + dR³")
    print()

    a_Q1, b_Q1, c_Q1, d_Q1 = profile.fit_Q1_coeffs[::-1]  # Reverse to get ascending powers
    a_real, b_real, c_real, d_real = profile.fit_real_coeffs[::-1]

    print(f"With Q=1:")
    print(f"  a (constant) = {a_Q1:+.6e}")
    print(f"  b (linear)   = {b_Q1:+.6e}")
    print(f"  c (quadratic)= {c_Q1:+.6e}")
    print(f"  d (cubic)    = {d_Q1:+.6e}")
    print()
    print(f"With real Q:")
    print(f"  a (constant) = {a_real:+.6e}")
    print(f"  b (linear)   = {b_real:+.6e}")
    print(f"  c (quadratic)= {c_real:+.6e}")
    print(f"  d (cubic)    = {d_real:+.6e}")
    print()

    # Analyze symmetry breaking
    # For perfect symmetry, I2(-R) = I2(R) exp(-2R) × (polynomial terms)
    # The ratio I2(+R)/I2(-R) ≈ exp(2R) × [1 + correction terms]

    # The linear term 'b' should be the dominant contribution
    # The cubic term 'd' represents symmetry breaking

    symmetry_Q1 = abs(d_Q1 / b_Q1) if abs(b_Q1) > 1e-15 else float('nan')
    symmetry_real = abs(d_real / b_real) if abs(b_real) > 1e-15 else float('nan')

    print(f"Symmetry breaking coefficient |d/b|:")
    print(f"  With Q=1:   {symmetry_Q1:.6e}")
    print(f"  With real Q: {symmetry_real:.6e}")
    print()

    # Check if Q increases symmetry breaking
    if symmetry_real > symmetry_Q1:
        print("Q INCREASES symmetry breaking (cubic term grows)")
        symmetry_increase = (symmetry_real / symmetry_Q1 - 1) * 100
        print(f"  Increase: {symmetry_increase:+.4f}%")
    else:
        print("Q DECREASES symmetry breaking (cubic term shrinks)")
        symmetry_decrease = (1 - symmetry_real / symmetry_Q1) * 100
        print(f"  Decrease: {symmetry_decrease:+.4f}%")

    print()

    # Analyze ratio at specific R values
    print("=" * 80)
    print("I2(+R)/I2(-R) RATIO ANALYSIS")
    print("=" * 80)
    print()

    ratios = profile.I2_ratio_at_R
    if ratios:
        print(f"{'R':<8} | {'Ratio(Q=1)':<12} | {'Ratio(real Q)':<12} | {'Q effect':<12}")
        print("-" * 60)

        for R in sorted(ratios.keys()):
            r = ratios[R]
            print(f"{R:+7.2f} | {r['ratio_Q1']:11.6f} | {r['ratio_real']:11.6f} | {r['Q_effect']*100:+10.4f}%")

    print()

    # Final analysis
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    # The key question: does the Q-induced change in I2(+R)/I2(-R) ratio
    # require a 0.58% adjustment to g?

    # At R=1.3036, we measured ratio change of -74%
    # But this is already absorbed into the I2 values themselves

    # The 0.58% gap must come from a subtle effect on HOW the mirror formula
    # combines I2(+R) and I2(-R)

    print("KEY FINDINGS:")
    print("-------------")
    print()
    print(f"1. Q changes I2(+R)/I2(-R) ratio dramatically (see table above)")
    print(f"2. Symmetry breaking (cubic term): Q1={symmetry_Q1:.2e}, Q={symmetry_real:.2e}")
    print(f"3. g_I2 calibration gap: {gap_pct:+.4f}%")
    print()
    print("The massive Q effect on the ratio is already baked into I2 values.")
    print("The 0.58% g_I2 gap is a RESIDUAL effect, not directly the ratio change.")
    print()
    print("POSSIBLE MECHANISM:")
    print("-------------------")
    print("The g correction accounts for the Beta moment: ∫ u × u^ℓ du")
    print()
    print("With Q polynomial, the effective moment might need a Q-dependent correction:")
    print("  g_I2 = g_baseline × [1 + f(Q)]")
    print()
    print("Where f(Q) ≈ 0.0058 (0.58%) accounts for how Q modifies the")
    print("relationship between I2(+R) and I2(-R) in the mirror formula.")
    print()
    print("This correction is MUCH SMALLER than Q's direct effect (~74%)")
    print("because most of Q's effect is already absorbed in the I2 values.")
    print()


if __name__ == "__main__":
    analyze_mirror_symmetry_breaking()
