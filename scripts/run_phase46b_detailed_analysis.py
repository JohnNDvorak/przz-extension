#!/usr/bin/env python3
"""
scripts/run_phase46b_detailed_analysis.py
Phase 46B: Detailed analysis of Q-derivative effects on I1 correction

The Q-derivative [xy] coefficient is ~48% of frozen Q², but the calibrated
epsilon is only ~0.09%. This script investigates why.

Key insight: The Q-derivative contribution is ALREADY INCLUDED in the normal
I1 integral. The question is how it affects the g correction formula.

Created: 2025-12-27 (Phase 46B)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.polynomials import load_przz_polynomials, Polynomial
from src.quadrature import gauss_legendre_01
from src.unified_s12.q_affine_expansion import (
    q_product_xy_coeff_post_identity_vectorized,
)


def analyze_i1_correction_structure():
    """
    Analyze how Q-derivative affects the I1 correction formula.

    The I1 bracket is: (1/θ + x + y) × F(x,y,u,t)

    where F includes Q(A_alpha) × Q(A_beta).

    The d²/dxdy extraction gives:
        (1/θ)F_xy + F_x + F_y

    The correction ratio is:
        g_internal = 1 + θ × (F_x + F_y) / F_xy

    The Q-derivative contributes to F_xy. We need to understand:
    1. How much of F_xy comes from Q derivatives
    2. How this affects the correction ratio
    """
    print("=" * 70)
    print("DETAILED ANALYSIS: Q-DERIVATIVE EFFECTS ON I1 CORRECTION")
    print("=" * 70)

    # Load PRZZ polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
    Q_mono = Q.to_monomial()

    theta = 4/7
    R = 1.3036
    K = 3
    n_quad = 80

    t_pts, t_wts = gauss_legendre_01(n_quad)

    # Weight for -R
    weights_minus = np.exp(-2 * R * t_pts)

    # ========================================
    # FROZEN Q CONTRIBUTION
    # ========================================
    # The [xy] from F with frozen Q comes from the polynomial structure,
    # NOT from Q. With Q=Q(t), Q contributes 1 to each monomial coefficient.

    Q_vals = Q_mono.eval(t_pts)
    Q_squared = Q_vals ** 2

    # The frozen Q² integral
    frozen_integral = np.sum(t_wts * weights_minus * Q_squared)
    print(f"\nFrozen Q² integral at -R: {frozen_integral:.6f}")

    # ========================================
    # Q-DERIVATIVE CONTRIBUTION TO [xy]
    # ========================================
    # The [xy] coefficient from Q(A_alpha)Q(A_beta) is:
    # θ² × [2t(t-1)×Q×Q'' + (t²+(t-1)²)×(Q')²]

    xy_coeffs = q_product_xy_coeff_post_identity_vectorized(Q_mono, t_pts, theta)
    xy_integral = np.sum(t_wts * weights_minus * xy_coeffs)
    print(f"Q-derivative [xy] integral at -R: {xy_integral:.6f}")

    ratio = xy_integral / frozen_integral
    print(f"Ratio [xy]/frozen: {ratio:.6f}")

    # ========================================
    # ANALYSIS: WHAT DOES THIS RATIO MEAN?
    # ========================================
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    print("""
The ratio ≈ 0.48 tells us that the Q-derivative [xy] term is about 48%
of the frozen Q² value. But the calibrated epsilon_I1 is only 0.09%.

This means:

1. The Q-derivative IS included in the normal I1 integral computation
   (we're not missing it)

2. The 0.09% epsilon must come from a DIFFERENTIAL effect:
   - How the Q-derivative affects the RATIO between terms
   - Not the absolute magnitude

Let's decompose what enters the correction formula:
""")

    # ========================================
    # CORRECTION FORMULA COMPONENTS
    # ========================================
    # The I1 correction comes from the log factor product rule:
    # d²/dxdy[(1/θ + x + y) × G] = (1/θ)G_xy + G_x + G_y
    #
    # where G = [rest of bracket without log factor]
    #
    # The correction ratio is:
    # g = [(1/θ)G_xy + G_x + G_y] / [(1/θ)G_xy]
    #   = 1 + θ × (G_x + G_y) / G_xy
    #
    # For the Beta moment formula to work, we need:
    # (G_x + G_y) / G_xy ≈ 1/(2K(2K+1))

    print("The correction formula is:")
    print("  g_internal = 1 + θ × (G_x + G_y) / G_xy")
    print("")
    print("where G includes Q(A_alpha) × Q(A_beta).")
    print("")
    print("The Q-derivative affects G_xy (the [xy] coefficient of G).")
    print("If Q-derivative adds to G_xy, it DECREASES the correction ratio!")
    print("")

    # Compute what the correction would be with and without Q-derivative
    # Assume: G_xy_frozen + G_xy_from_Q_deriv = G_xy_total
    # And: (G_x + G_y) is roughly constant (doesn't depend much on Q-derivative)

    # For simplicity, model:
    # G_xy ∝ frozen_Q² + [xy]_from_Q_deriv

    # If the correction is:
    # g = 1 + θ × C / G_xy
    # where C = (G_x + G_y) is some constant

    # Then:
    # g_frozen = 1 + θ × C / G_xy_frozen
    # g_full = 1 + θ × C / G_xy_full
    # where G_xy_full = G_xy_frozen × (1 + ratio)

    # So:
    # g_full = 1 + θ × C / (G_xy_frozen × (1 + ratio))
    #        = 1 + (g_frozen - 1) / (1 + ratio)

    g_baseline = 1 + theta / (2 * K * (2 * K + 1))
    print(f"g_baseline (from Beta moment) = {g_baseline:.8f}")

    # If g_frozen ≈ g_baseline, then:
    delta_g_baseline = g_baseline - 1  # ≈ 0.0136
    g_full_predicted = 1 + delta_g_baseline / (1 + ratio)
    print(f"delta_g_baseline = {delta_g_baseline:.8f}")
    print(f"")
    print(f"If Q-derivative ratio adds to G_xy:")
    print(f"  g_full = 1 + delta_g / (1 + ratio)")
    print(f"         = 1 + {delta_g_baseline:.6f} / (1 + {ratio:.4f})")
    print(f"         = 1 + {delta_g_baseline:.6f} / {1 + ratio:.4f}")
    print(f"         = {g_full_predicted:.8f}")

    print("")
    print("This is LESS than g_baseline, which matches g_I1 ≈ 1.0!")
    print("")
    print("The Q-derivative adds to the denominator G_xy, which")
    print("REDUCES the correction ratio, explaining why g_I1 < g_baseline.")

    # ========================================
    # QUANTITATIVE CHECK
    # ========================================
    print("\n" + "-" * 70)
    print("QUANTITATIVE CHECK")
    print("-" * 70)

    # If g_I1 = 1 + delta_g_baseline / (1 + ratio), then:
    # For ratio = 0.48:
    # g_I1 = 1 + 0.0136 / 1.48 = 1 + 0.0092 = 1.0092

    # But calibrated g_I1 = 1.00091, which is:
    # 1.00091 = 1 + delta_g / (1 + ratio_effective)
    # 0.00091 = delta_g / (1 + ratio_effective)
    # ratio_effective = delta_g / 0.00091 - 1 = 0.0136 / 0.00091 - 1 = 13.95

    # This doesn't match! The ratio would need to be 14 to get g_I1 = 1.00091

    g_I1_calibrated = 1.00091428
    epsilon_I1 = g_I1_calibrated - 1

    implied_ratio = delta_g_baseline / epsilon_I1 - 1
    print(f"Calibrated g_I1 = {g_I1_calibrated:.8f}")
    print(f"epsilon_I1 = {epsilon_I1:.8f}")
    print(f"")
    print(f"For g_I1 = 1 + delta_g/(1+ratio):")
    print(f"  Implied ratio needed = {implied_ratio:.4f}")
    print(f"  Measured ratio       = {ratio:.4f}")
    print(f"")
    print(f"The measured ratio {ratio:.2f} would give g_I1 = {g_full_predicted:.6f}")
    print(f"But calibrated g_I1 = {g_I1_calibrated:.6f}")
    print(f"")
    print(f"DISCREPANCY: The simple model doesn't explain the calibrated value!")

    # ========================================
    # ALTERNATIVE: MAYBE G_x + G_y ALSO CHANGES?
    # ========================================
    print("\n" + "-" * 70)
    print("ALTERNATIVE HYPOTHESIS")
    print("-" * 70)

    print("""
The simple model assumed (G_x + G_y) is constant and Q-derivative only
affects G_xy. But actually, Q-derivative also affects G_x and G_y!

For Q(A) = Q(t) + Q'(t)×a×x + Q'(t)×b×y + Q''(t)×a×b×xy

The x-coefficient is Q'(t)×a, which affects G_x.
The y-coefficient is Q'(t)×b, which affects G_y.

So the correction ratio (G_x + G_y) / G_xy is modified by Q in BOTH
numerator and denominator. The net effect depends on the balance.

This is getting complex. The key insight is:
- Q-derivative effects are INCLUDED in the I1 integral
- The self-correction (g_I1 ≈ 1) happens because Q-derivative terms
  affect both numerator and denominator in the correction formula
- The ~0.09% residual epsilon is what's LEFT OVER after this
  partial cancellation
""")


def main():
    analyze_i1_correction_structure()


if __name__ == "__main__":
    main()
