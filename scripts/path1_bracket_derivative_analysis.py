#!/usr/bin/env python3
"""
scripts/path1_bracket_derivative_analysis.py
Path 1: Compute ∂²/∂x∂y B(x,y)|_{0,0} symbolically before collapsing to scalar

HYPOTHESIS:
The exp(R) factor (instead of exp(2R)) emerges when we:
1. Keep the full x,y dependence in the PRZZ bracket
2. Apply ∂²/∂x∂y BEFORE setting x=y=0
3. The derivative structure changes the effective exponential weight

MATHEMATICAL STRUCTURE:
The PRZZ bracket (Lines 1502-1511):
    B(α,β;x,y) = [N^{αx+βy} - T^{-(α+β)}N^{-βx-αy}] / (α+β)

After the DQ identity transformation with α=β=-R/L:
    B = log(N^{x+y}T) × ∫₀¹ exp(2Rt(1+θ(x+y))) dt × prefactors

The question: What is ∂²/∂x∂y of the integrated bracket at x=y=0?
Does it produce an effective exp(R) factor instead of exp(2R)?

Created: 2025-12-29 (Phase 53 - PRZZ Derivation Investigation)
"""

import math
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quadrature import gauss_legendre_01


@dataclass
class BracketDerivativeResult:
    """Results from bracket derivative analysis."""
    # Raw integrals
    scalar_integral: float  # ∫ exp(2Rt) dt = (exp(2R)-1)/(2R)
    xy_coeff_integral: float  # ∫ [xy coefficient] dt

    # Effective multipliers
    effective_exp_factor: float  # What multiplier does xy_coeff give vs scalar?

    # Comparison with targets
    exp_r: float
    exp_2r: float
    production_m_base: float  # exp(R) + 5
    dq_scalar_limit: float  # (exp(2R)-1)/(2R)


def compute_bracket_expansion_coefficients(t: float, R: float, theta: float) -> Dict[str, float]:
    """
    Compute the Taylor expansion coefficients of the bracket integrand at given t.

    The integrand is: (1 + θ(x+y)) × exp(2Rt(1+θ(x+y)))

    We need to extract:
    - constant term (x=y=0)
    - x coefficient
    - y coefficient
    - xy coefficient (mixed derivative)

    Returns dict with coefficients.
    """
    # Let f(x,y) = (1 + θ(x+y)) × exp(2Rt(1+θ(x+y)))
    #            = (1 + θ(x+y)) × exp(2Rt) × exp(2Rtθx) × exp(2Rtθy)

    # Expand exp(2Rtθx) ≈ 1 + 2Rtθx + (2Rtθx)²/2 + ...
    # Expand exp(2Rtθy) ≈ 1 + 2Rtθy + (2Rtθy)²/2 + ...

    exp_2Rt = math.exp(2 * R * t)
    coeff_linear = 2 * R * t * theta  # coefficient of x or y in exp expansion
    coeff_quadratic = (2 * R * t * theta)**2 / 2  # coefficient of x² or y² in exp expansion

    # Product (1 + 2Rtθx + ...)(1 + 2Rtθy + ...)
    # Constant: 1
    # x-coeff: 2Rtθ
    # y-coeff: 2Rtθ
    # xy-coeff: (2Rtθ)² from the cross term

    exp_product_const = 1.0
    exp_product_x = coeff_linear
    exp_product_y = coeff_linear
    exp_product_xy = coeff_linear**2  # = (2Rtθ)²

    # Now multiply by (1 + θx + θy):
    # f = (1 + θx + θy) × exp_factor
    # f_const = 1 × exp_product_const
    # f_x = 1 × exp_product_x + θ × exp_product_const
    # f_y = 1 × exp_product_y + θ × exp_product_const
    # f_xy = 1 × exp_product_xy + θ × exp_product_y + θ × exp_product_x

    f_const = exp_product_const
    f_x = exp_product_x + theta
    f_y = exp_product_y + theta
    f_xy = exp_product_xy + theta * exp_product_y + theta * exp_product_x

    # Multiply everything by exp(2Rt)
    return {
        'const': exp_2Rt * f_const,
        'x': exp_2Rt * f_x,
        'y': exp_2Rt * f_y,
        'xy': exp_2Rt * f_xy,
        't': t,
        'exp_2Rt': exp_2Rt,
    }


def compute_bracket_integrals(R: float, theta: float, n_quad: int = 100) -> BracketDerivativeResult:
    """
    Compute the integrated bracket coefficients over t ∈ [0,1].

    The key quantity is the xy coefficient integral, which gives
    ∂²/∂x∂y [∫₀¹ integrand dt] at x=y=0.
    """
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # Integrate each coefficient
    const_integral = 0.0
    x_integral = 0.0
    y_integral = 0.0
    xy_integral = 0.0

    for t, w in zip(t_nodes, t_weights):
        coeffs = compute_bracket_expansion_coefficients(t, R, theta)
        const_integral += coeffs['const'] * w
        x_integral += coeffs['x'] * w
        y_integral += coeffs['y'] * w
        xy_integral += coeffs['xy'] * w

    # Compare with known values
    exp_r = math.exp(R)
    exp_2r = math.exp(2 * R)
    dq_limit = (exp_2r - 1) / (2 * R)
    production_base = exp_r + 5

    # The scalar integral should equal the DQ limit
    assert abs(const_integral - dq_limit) / dq_limit < 1e-8, f"Scalar mismatch: {const_integral} vs {dq_limit}"

    # What's the effective exponential factor from the xy coefficient?
    # If the structure were exp(factor * R), what factor does xy_integral imply?
    # We can compare xy_integral / const_integral to see the ratio
    effective_ratio = xy_integral / const_integral if const_integral != 0 else 0

    return BracketDerivativeResult(
        scalar_integral=const_integral,
        xy_coeff_integral=xy_integral,
        effective_exp_factor=effective_ratio,
        exp_r=exp_r,
        exp_2r=exp_2r,
        production_m_base=production_base,
        dq_scalar_limit=dq_limit,
    )


def compute_xy_integral_analytic(R: float, theta: float) -> float:
    """
    Compute ∫₀¹ xy_coeff(t) dt analytically.

    xy_coeff(t) = exp(2Rt) × [(2Rtθ)² + 2θ×2Rtθ]
                = exp(2Rt) × [4R²t²θ² + 4Rtθ²]
                = 4Rθ² × exp(2Rt) × t × (Rt + 1)

    So we need:
    ∫₀¹ exp(2Rt) × t × (Rt + 1) dt
    = R × ∫₀¹ t² exp(2Rt) dt + ∫₀¹ t exp(2Rt) dt
    """
    # ∫ t exp(2Rt) dt via integration by parts
    # Let u = t, dv = exp(2Rt)dt
    # du = dt, v = exp(2Rt)/(2R)
    # ∫ t exp(2Rt) dt = t exp(2Rt)/(2R) - ∫ exp(2Rt)/(2R) dt
    #                 = t exp(2Rt)/(2R) - exp(2Rt)/(4R²)
    # From 0 to 1:
    # [exp(2R)/(2R) - exp(2R)/(4R²)] - [0 - 1/(4R²)]
    # = exp(2R)/(2R) - exp(2R)/(4R²) + 1/(4R²)
    # = exp(2R)(2R - 1)/(4R²) + 1/(4R²)
    # = [(2R-1)exp(2R) + 1]/(4R²)

    int_t_exp = ((2*R - 1) * math.exp(2*R) + 1) / (4 * R**2)

    # ∫ t² exp(2Rt) dt via repeated integration by parts
    # ∫ t² exp(2Rt) dt = t² exp(2Rt)/(2R) - (2/2R) ∫ t exp(2Rt) dt
    #                  = t² exp(2Rt)/(2R) - (1/R) × int_t_exp
    # From 0 to 1:
    # exp(2R)/(2R) - (1/R) × int_t_exp

    int_t2_exp = math.exp(2*R)/(2*R) - (1/R) * int_t_exp

    # Full xy coefficient integral:
    # 4Rθ² × [R × int_t2_exp + int_t_exp]
    xy_integral = 4 * R * theta**2 * (R * int_t2_exp + int_t_exp)

    return xy_integral


def analyze_derivative_structure(R: float, theta: float = 4.0/7.0, n_quad: int = 100):
    """
    Full analysis of the bracket derivative structure.
    """
    print("=" * 70)
    print("PATH 1: BRACKET DERIVATIVE ANALYSIS")
    print("=" * 70)
    print(f"\nParameters: R = {R}, θ = {theta}")
    print()

    # Compute via numerical quadrature
    result = compute_bracket_integrals(R, theta, n_quad)

    # Compute analytic value for comparison
    xy_analytic = compute_xy_integral_analytic(R, theta)

    print("INTEGRATED COEFFICIENTS:")
    print(f"  Scalar (const) integral:  {result.scalar_integral:.10f}")
    print(f"  Expected DQ limit:        {result.dq_scalar_limit:.10f}")
    print(f"  xy coefficient integral:  {result.xy_coeff_integral:.10f}")
    print(f"  xy analytic value:        {xy_analytic:.10f}")
    print()

    print("COMPARISON WITH EXPONENTIAL FACTORS:")
    print(f"  exp(R) = {result.exp_r:.6f}")
    print(f"  exp(2R) = {result.exp_2r:.6f}")
    print(f"  Production base = exp(R) + 5 = {result.production_m_base:.6f}")
    print(f"  DQ scalar limit = (exp(2R)-1)/(2R) = {result.dq_scalar_limit:.6f}")
    print()

    # Compute ratios
    ratio_xy_to_scalar = result.xy_coeff_integral / result.scalar_integral

    print("RATIOS:")
    print(f"  xy_coeff / scalar = {ratio_xy_to_scalar:.6f}")
    print(f"  This gives an effective 'correction factor' of {ratio_xy_to_scalar:.6f}")
    print()

    # KEY QUESTION: If I₁ uses xy-coefficient and I₂ uses scalar,
    # what's the effective weight ratio?
    print("KEY INSIGHT:")
    print("  The scalar limit (for I₂-type terms) = DQ_limit = (exp(2R)-1)/(2R)")
    print("  The xy coefficient (for I₁-type terms) has DIFFERENT structure")
    print()

    # Check if xy_coeff / theta² gives something close to exp(R)
    print("HYPOTHESIS TESTS:")

    # Test 1: Does xy_coeff/θ² relate to exp(R)?
    xy_over_theta2 = result.xy_coeff_integral / theta**2
    print(f"  xy_coeff / θ² = {xy_over_theta2:.6f}")
    print(f"  Ratio to exp(R): {xy_over_theta2 / result.exp_r:.6f}")
    print(f"  Ratio to exp(2R): {xy_over_theta2 / result.exp_2r:.6f}")
    print()

    # Test 2: What normalization would make xy_coeff = exp(R)?
    needed_factor = result.exp_r / result.xy_coeff_integral
    print(f"  To get xy_coeff × factor = exp(R), need factor = {needed_factor:.6f}")
    print()

    # Test 3: Compare to production m/DQ_limit ratio
    m_production = result.production_m_base  # exp(R) + 5 (ignoring g corrections)
    gap_ratio = m_production / result.dq_scalar_limit
    print(f"  Production m / DQ_limit = {gap_ratio:.4f} (the '1.8× gap')")
    print()

    return result


def analyze_integrand_structure_at_t(R: float, theta: float = 4.0/7.0):
    """
    Detailed analysis of how the xy coefficient changes with t.

    The integrand xy coefficient is:
    exp(2Rt) × [4R²t²θ² + 4Rtθ²] = 4Rθ² × exp(2Rt) × t × (Rt + 1)

    This has a very specific t-weighting structure.
    """
    print("=" * 70)
    print("INTEGRAND STRUCTURE AT DIFFERENT t VALUES")
    print("=" * 70)
    print()

    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("t      | exp(2Rt)    | xy_coeff(t) | xy/exp(2Rt) | t×(Rt+1)")
    print("-" * 70)

    for t in t_values:
        exp_2Rt = math.exp(2 * R * t)
        xy_coeff = 4 * R * theta**2 * exp_2Rt * t * (R*t + 1) if t > 0 else 0
        ratio = xy_coeff / exp_2Rt if exp_2Rt > 0 and t > 0 else 0
        t_factor = t * (R*t + 1) if t > 0 else 0

        print(f"{t:.2f}   | {exp_2Rt:10.4f} | {xy_coeff:10.4f} | {ratio:10.6f} | {t_factor:.6f}")

    print()
    print("KEY OBSERVATION:")
    print("  At t=0.5: exp(2R×0.5) = exp(R)")
    print(f"  At t=0.5: exp(R) = {math.exp(R):.6f}")
    print()
    print("  The t=0.5 midpoint gives exp(R), but the xy coefficient")
    print("  has extra t×(Rt+1) weighting that shifts the effective center.")
    print()

    # Compute the "effective t" where the integrand has most weight
    # This is related to the average: ∫ t × integrand dt / ∫ integrand dt
    t_nodes, t_weights = gauss_legendre_01(100)

    numerator = 0.0
    denominator = 0.0

    for t, w in zip(t_nodes, t_weights):
        if t > 0:
            xy_coeff = 4 * R * theta**2 * math.exp(2*R*t) * t * (R*t + 1)
        else:
            xy_coeff = 0
        numerator += t * xy_coeff * w
        denominator += xy_coeff * w

    effective_t = numerator / denominator if denominator != 0 else 0
    print(f"  Effective t (weighted average) = {effective_t:.6f}")
    print(f"  exp(2R × effective_t) = {math.exp(2*R*effective_t):.6f}")
    print(f"  Compare to exp(R) = {math.exp(R):.6f}")


def check_combined_identity_structure(R: float, theta: float = 4.0/7.0):
    """
    Check if the COMBINED identity (direct + mirror) has different derivative structure.

    The difference quotient identity COMBINES:
    - Direct: N^{αx+βy}
    - Mirror: -T^{-(α+β)} N^{-βx-αy}

    Into a single t-integral. The mixed derivative of this combined structure
    might have different properties than taking them separately.
    """
    print("=" * 70)
    print("COMBINED IDENTITY ANALYSIS")
    print("=" * 70)
    print()

    # The PRZZ combined identity at α=β=-R/L:
    # B = N^{-R(x+y)/L} × log(N^{x+y}T) × ∫₀¹ [N^{x+y}T]^{t×2R/L} dt
    #
    # In the asymptotic limit:
    # N^{-R(x+y)/L} → exp(-Rθ(x+y))  [outer factor]
    # [N^{x+y}T]^{t×2R/L} → exp(2Rt(1+θ(x+y)))  [inner factor]
    # log(N^{x+y}T) → L(1+θ(x+y))  [log factor, L absorbed asymptotically]

    # So full structure is:
    # B ~ exp(-Rθ(x+y)) × (1+θ(x+y)) × ∫₀¹ exp(2Rt(1+θ(x+y))) dt

    # Let's compute the xy coefficient of the FULL expression
    # including the outer exp(-Rθ(x+y)) factor

    # exp(-Rθ(x+y)) × (1+θ(x+y)) × exp(2Rt(1+θ(x+y)))

    # Let me expand this systematically:
    # f(x,y;t) = exp(-Rθ(x+y)) × (1+θx+θy) × exp(2Rt) × exp(2Rtθ(x+y))
    #          = exp(2Rt) × exp(θ(x+y)(2Rt-R)) × (1+θ(x+y))
    #          = exp(2Rt) × exp(θ(x+y)R(2t-1)) × (1+θ(x+y))

    # Key insight: The combined exp factor is exp(Rθ(x+y)(2t-1))
    # NOT exp(2Rtθ(x+y))!

    print("COMBINED IDENTITY STRUCTURE:")
    print()
    print("Full bracket (after DQ identity):")
    print("  B ~ exp(-Rθ(x+y)) × (1+θ(x+y)) × ∫₀¹ exp(2Rt(1+θ(x+y))) dt")
    print()
    print("Simplifying the exponential product:")
    print("  exp(-Rθ(x+y)) × exp(2Rtθ(x+y)) = exp(Rθ(x+y)(2t-1))")
    print()
    print("So the COMBINED structure has:")
    print("  B ~ exp(2Rt) × exp(Rθ(x+y)(2t-1)) × (1+θ(x+y))")
    print()
    print("At t=0.5: (2t-1) = 0, so exp(Rθ(x+y)(2t-1)) = 1")
    print("The xy-dependence vanishes at t=0.5!")
    print()

    # Compute the xy coefficient of the COMBINED structure
    t_nodes, t_weights = gauss_legendre_01(100)

    xy_coeff_combined = 0.0
    scalar_combined = 0.0

    for t, w in zip(t_nodes, t_weights):
        exp_2Rt = math.exp(2 * R * t)

        # Expansion of exp(Rθ(x+y)(2t-1)) × (1+θ(x+y))
        # Let u = Rθ(2t-1)
        # exp(u(x+y)) ≈ 1 + u(x+y) + u²(x+y)²/2 + ...
        # (1+θ(x+y)) × exp(u(x+y)) ≈ (1+θ(x+y))(1 + u(x+y) + u²xy + ...)
        #                          ≈ 1 + (θ+u)(x+y) + (u² + 2θu)xy + ...

        u = R * theta * (2*t - 1)

        # xy coefficient of combined structure
        xy_coeff_t = u**2 + 2*theta*u  # = u(u + 2θ) = Rθ(2t-1)[Rθ(2t-1) + 2θ]

        xy_coeff_combined += exp_2Rt * xy_coeff_t * w
        scalar_combined += exp_2Rt * w

    print("COMBINED xy coefficient integral:")
    print(f"  xy_coeff_combined = {xy_coeff_combined:.10f}")
    print(f"  scalar_combined = {scalar_combined:.10f}")
    print()

    # Compare with the uncombined version (from earlier)
    result_uncombined = compute_bracket_integrals(R, theta, 100)

    print("COMPARISON:")
    print(f"  Uncombined xy_coeff = {result_uncombined.xy_coeff_integral:.10f}")
    print(f"  Combined xy_coeff = {xy_coeff_combined:.10f}")
    print(f"  Ratio = {xy_coeff_combined / result_uncombined.xy_coeff_integral:.6f}")
    print()

    # The combined version should give DIFFERENT results because of the
    # outer exp(-Rθ(x+y)) factor!

    # Now let's see what effective exponential this gives
    if xy_coeff_combined != 0:
        ratio_to_exp_r = xy_coeff_combined / math.exp(R)
        ratio_to_exp_2r = xy_coeff_combined / math.exp(2*R)
        print(f"  xy_coeff_combined / exp(R) = {ratio_to_exp_r:.6f}")
        print(f"  xy_coeff_combined / exp(2R) = {ratio_to_exp_2r:.6f}")


def main():
    # Test both benchmarks
    for R in [1.3036, 1.1167]:
        print("\n" + "=" * 70)
        print(f"R = {R}")
        print("=" * 70)

        analyze_derivative_structure(R)
        analyze_integrand_structure_at_t(R)
        check_combined_identity_structure(R)

        print()


if __name__ == "__main__":
    main()
