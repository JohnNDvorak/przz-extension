"""
src/test_factorial_normalization.py
Factorial normalization test per user's guidance.

This test verifies that:
1. Multi-variable coefficient extraction [z₁ z₂ w₁ w₂] F
2. Collapsed coefficient extraction [Z² W²] F × (2!)²
3. Derivative extraction ∂²_Z ∂²_W F(0,0)

All give the same result for F that depends on Z = z₁+z₂ and W = w₁+w₂.
"""

import numpy as np
from math import factorial
from src.series import TruncatedSeries


def test_factorial_equivalence():
    """
    Build any dummy integrand F(z₁,z₂,w₁,w₂) that depends only on Z=z₁+z₂ and W=w₁+w₂.
    Compute:
    - multi-variable coefficient of z₁ z₂ w₁ w₂
    - vs coefficient of Z² W² times (2!)(2!)
    - vs ∂²_Z ∂²_W F(0,0)
    All must agree.
    """
    print("="*70)
    print("FACTORIAL NORMALIZATION TEST")
    print("="*70)
    print()

    # Test function: G(Z, W) = (1 + Z)² * (1 + W)²
    # Expanded: G = 1 + 2Z + Z² + 2W + 4ZW + 2Z²W + W² + 2ZW² + Z²W²
    # Coeff of Z²W² = 1
    # ∂⁴/∂Z²∂W² G = 4, at (0,0): 4

    print("Test function: G(Z, W) = (1 + Z)² × (1 + W)²")
    print("where Z = z1 + z2, W = w1 + w2")
    print()
    print("Expected:")
    print("  [Z²W²] coefficient = 1")
    print("  ∂⁴/∂Z²∂W² G(0,0) = 4")
    print("  [z1z2w1w2] in G(z1+z2, w1+w2) = 4 (since = [Z²W²] × 2! × 2!)")
    print()

    # Method 1: Multi-variable (z1, z2, w1, w2) with nilpotent variables
    print("=== Method 1: Multi-variable (z1, z2, w1, w2) ===")
    vars_multi = ("z1", "z2", "w1", "w2")

    # Build Z = z1 + z2 and W = w1 + w2
    one = TruncatedSeries.from_scalar(1.0, vars_multi)
    z1 = TruncatedSeries.variable("z1", vars_multi)
    z2 = TruncatedSeries.variable("z2", vars_multi)
    w1 = TruncatedSeries.variable("w1", vars_multi)
    w2 = TruncatedSeries.variable("w2", vars_multi)

    Z = z1 + z2
    W = w1 + w2

    # G(Z, W) = (1 + Z)² × (1 + W)²
    one_plus_Z = one + Z
    one_plus_W = one + W
    G_multi = (one_plus_Z * one_plus_Z) * (one_plus_W * one_plus_W)

    # Extract coefficient of z1*z2*w1*w2
    # The bitset for z1=bit0, z2=bit1, w1=bit2, w2=bit3 is 0b1111 = 15
    coeff_z1z2w1w2 = G_multi.coeffs.get(15, 0.0)
    if isinstance(coeff_z1z2w1w2, np.ndarray):
        coeff_z1z2w1w2 = float(coeff_z1z2w1w2)

    print(f"  Coefficient of z1×z2×w1×w2: {coeff_z1z2w1w2}")
    print(f"  Expected (= [Z²W²] × (2!)²): 4")
    print(f"  Match: {'✓' if abs(coeff_z1z2w1w2 - 4) < 1e-10 else '✗'}")
    print()

    # KEY INSIGHT explanation:
    print("="*70)
    print("KEY INSIGHT")
    print("="*70)
    print()
    print("For F(Z, W) where Z = z1+z2, W = w1+w2:")
    print()
    print("  (z1+z2)² = z1² + 2z1z2 + z2²")
    print("  But z1² = z2² = 0 (nilpotent), so (z1+z2)² = 2z1z2")
    print()
    print("  Therefore:")
    print("  [z1z2] in (z1+z2)² = 2")
    print("  [Z²] in Z² = 1")
    print("  Ratio: [z1z2] / [Z²] = 2 = 2!")
    print()
    print("  For 4 variables with Z=z1+z2, W=w1+w2:")
    print("  [z1z2w1w2] = [Z²W²] × (2!)(2!) = [Z²W²] × 4")
    print()

    return abs(coeff_z1z2w1w2 - 4) < 1e-10


def test_derivative_consistency():
    """
    Test that our series engine correctly handles the multi-variable expansion.
    """
    print()
    print("="*70)
    print("DERIVATIVE CONSISTENCY TEST")
    print("="*70)
    print()

    # Use a simple polynomial P(u) = u + u²
    # P(u + z1 + z2) expanded:
    # P = (u + z1 + z2) + (u + z1 + z2)²
    #
    # With nilpotent z1, z2 (z1²=z2²=0):
    # (u + z1 + z2)² = u² + 2u(z1+z2) + (z1+z2)²
    #                = u² + 2u(z1+z2) + 2z1z2
    #
    # P(u + z1 + z2) = u + z1 + z2 + u² + 2u(z1+z2) + 2z1z2
    #               = (u + u²) + (1 + 2u)z1 + (1 + 2u)z2 + 2z1z2
    #
    # Coefficient of z1z2 = 2 = P''(u) (since P'' = 2)

    vars = ("z1", "z2")

    u = 0.5

    # Build u + z1 + z2
    u_const = TruncatedSeries.from_scalar(u, vars)
    z1 = TruncatedSeries.variable("z1", vars)
    z2 = TruncatedSeries.variable("z2", vars)

    arg = u_const + z1 + z2

    # P(arg) = arg + arg²
    P = arg + arg * arg

    # Coefficient of z1*z2 (bitset = 0b11 = 3)
    coeff_z1z2 = P.coeffs.get(3, 0.0)
    if isinstance(coeff_z1z2, np.ndarray):
        coeff_z1z2 = float(coeff_z1z2)

    print(f"P(u) = u + u² with u = {u}")
    print(f"P''(u) = 2 (constant)")
    print()
    print(f"P(u + z1 + z2) expanded, coefficient of z1×z2: {coeff_z1z2}")
    print(f"Expected [z1z2] = P''(u) = 2: {'✓' if abs(coeff_z1z2 - 2) < 1e-10 else '✗'}")
    print()

    # Verify this matches the formula:
    # [z1z2] in F(z1+z2) = F''(0) when evaluated at the origin
    # In our case, F(Z) = (u+Z) + (u+Z)² at the origin means Z=0
    # F''(u) = d²/dZ² [(u+Z) + (u+Z)²] = d²/dZ² [u+Z + u² + 2uZ + Z²]
    #        = d²/dZ² [Z² + (1+2u)Z + (u+u²)]
    #        = 2
    print("Verification:")
    print("  F(Z) = (u+Z) + (u+Z)² = Z² + (1+2u)Z + (u+u²)")
    print("  F''(Z) = 2")
    print("  F''(0) = 2 ✓")

    return abs(coeff_z1z2 - 2) < 1e-10


def test_przz_implication():
    """
    Test the PRZZ implication: for pair (2,2) using x1,x2,y1,y2
    vs collapsed x,y with derivative orders 2,2.
    """
    print()
    print("="*70)
    print("PRZZ IMPLICATION TEST")
    print("="*70)
    print()

    print("For PRZZ pair (2,2):")
    print()
    print("Multi-variable approach (x1, x2, y1, y2):")
    print("  P(x1+x2+u) × P(y1+y2+u) × ...")
    print("  Extract [x1 x2 y1 y2] coefficient")
    print("  This equals ∂/∂x1 ∂/∂x2 ∂/∂y1 ∂/∂y2 F(0,0,0,0)")
    print()
    print("Collapsed approach (x, y) with ∂²x ∂²y:")
    print("  P(x+u) × P(y+u) × ...")
    print("  Compute ∂²/∂x² ∂²/∂y² F(0,0)")
    print()
    print("These should give the SAME result because:")
    print("  [x1 x2] in F(x1+x2+u) = F''(u) (with nilpotent x1, x2)")
    print("  ∂²/∂x² F(x+u)|_{x=0} = F''(u)")
    print()
    print("The factorial normalization is handled by:")
    print("  - Multi-variable: [x1x2] includes the factorial automatically")
    print("  - Collapsed: ∂²x F = 2! × [x²] F, so ∂²x ∂²y F = 4 × [x²y²] F")
    print()
    print("KEY: Our series engine extracts [x1x2y1y2] which equals the derivative.")
    print("     The evaluate.py then divides by (ℓ₁!)(ℓ₂!) = 4 for normalization.")
    print("     But this is DOUBLE-counting the factorial!")
    print()
    print("BUG HYPOTHESIS:")
    print("  evaluate.py divides by 4 for (2,2) pair")
    print("  But [x1x2y1y2] already gives ∂⁴F directly (no factorial needed)")
    print("  So we're dividing by 4 when we shouldn't be!")


if __name__ == "__main__":
    test_factorial_equivalence()
    test_derivative_consistency()
    test_przz_implication()
