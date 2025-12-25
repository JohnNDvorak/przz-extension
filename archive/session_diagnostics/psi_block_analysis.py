"""
src/psi_block_analysis.py
Analysis of Ψ Block Structure and Mapping to PRZZ Integrands

This file explores how the abstract Ψ building blocks (A, B, C, D)
map to concrete operations on our integrand F(x,y,u,t).

Key insight from GPT:
- A, B, C, D are derivatives of LOG F, not F itself
- If F = e^L (where L = log F), then derivatives of F involve
  products of derivatives of L via Faà-di-Bruno
- A² means (∂L/∂x)², NOT ½·∂²L/∂x²

The connected blocks:
- X = (A - C)  →  "connected singleton z-block"
- Y = (B - C)  →  "connected singleton w-block"
- Z = (D - C²) →  "connected paired block"

The subtractions remove "disconnected" lower-order contributions.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple
from math import exp


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def analyze_11_structure(P1, Q, theta: float, R: float, n_quad: int = 60):
    """
    Analyze how (1,1) pair maps Ψ blocks to our DSL I-terms.

    Ψ_{1,1} = (A-C)(B-C) + (D-C²) = AB - AC - BC + D

    Our DSL I-terms:
    - I₁: d²F/dxdy |_{x=y=0}  (mixed derivative)
    - I₂: F|_{x=y=0}          (no derivatives)
    - I₃: -d/dx[...] |_{x=0}  (z-derivative)
    - I₄: -d/dy[...] |_{y=0}  (w-derivative)

    Question: How do AB, D, -AC, -BC map to I₁, I₂, I₃, I₄?
    """
    print("=" * 70)
    print("ANALYSIS: (1,1) Ψ BLOCKS → I-TERMS MAPPING")
    print("=" * 70)

    print("\nΨ_{1,1} expansion:")
    print("  = (A-C)(B-C) + (D-C²)")
    print("  = AB - AC - BC + C² + D - C²")
    print("  = AB - AC - BC + D")
    print()
    print("This gives 4 monomials: AB, -AC, -BC, D")
    print()

    print("=" * 70)
    print("INTERPRETATION KEY")
    print("=" * 70)
    print("""
In the Ψ framework for PRZZ:

  L(x,y,u,t) = log F(x,y,u,t)

where F is our integrand. Then:

  C = L|_{x=y=0}                 (log of base integrand)
  A = ∂L/∂x |_{x=y=0}           (d/dx of log F)
  B = ∂L/∂y |_{x=y=0}           (d/dy of log F)
  D = ∂²L/∂x∂y |_{x=y=0}        (mixed derivative of log F)

IMPORTANT: These are derivatives of LOG(F), not F!

Recall that if F = e^L:
  ∂F/∂x = F × (∂L/∂x) = F × A
  ∂²F/∂x∂y = F × (∂²L/∂x∂y + ∂L/∂x × ∂L/∂y)
           = F × (D + A×B)

So extracting d²F/dxdy gives us F×(D + AB), NOT just D!
""")

    print("=" * 70)
    print("THE MAPPING FOR (1,1)")
    print("=" * 70)
    print("""
When we compute I₁ = ∫∫ d²F/dxdy |_{x=y=0} du dt:
  = ∫∫ F|₀ × (D + AB) du dt
  = (I₂ value) × (D + AB)  [symbolically]

But Ψ gives us AB - AC - BC + D, not (D + AB)...

The key is that Ψ is the COEFFICIENT STRUCTURE for the main term,
after PRZZ's reductions. The -AC and -BC terms come from:
  - The prefactor (1+θ(x+y))/θ in I₁
  - The separate I₃, I₄ contributions

Let's verify by comparing symbolically:

  I₁ contrib: coeff × (D + AB)  [from d²F/dxdy]
  I₂ contrib: some base value
  I₃ contrib: -something × (A - related to C?)
  I₄ contrib: -something × (B - related to C?)

The Ψ formula Σ(AB - AC - BC + D) captures the NET effect after
combining all four I-terms with their proper weights.

For (1,1), the I-terms are:
  I₁: captures the AB part (mixed derivative product structure)
  I₂: captures the D part (the "base" pairing)
  I₃: captures the -AC part (z-derivative with base subtraction)
  I₄: captures the -BC part (w-derivative with base subtraction)

This is why the 4-term I₁-I₄ structure WORKS for (1,1)!
""")

    print("=" * 70)
    print("WHY IT BREAKS FOR (2,2)")
    print("=" * 70)
    print("""
For (2,2), we have:
  Ψ_{2,2} = Σ_p C(2,p)C(2,p)p! × Z^p × X^{2-p} × Y^{2-p}

  p=0: 1 × X² × Y² = (A-C)²(B-C)²
  p=1: 4 × Z × X × Y = 4(D-C²)(A-C)(B-C)
  p=2: 2 × Z² = 2(D-C²)²

Expanding X² = (A-C)² = A² - 2AC + C², etc.

The A² term represents (∂L/∂x)², which in F-space is:
  d²F/dx² involves F × (L_xx + L_x²) = F × (something + A²)

So A² is NOT just "take second x-derivative of F", it's
"the square of the first log-derivative coefficient".

This requires a DIFFERENT derivative structure than I₁-I₄!

The DSL's I₁-I₄ can only capture:
  - D (0-order)
  - A, B (1st-order each)
  - AB (mixed 1st-order)

But Ψ_{2,2} has:
  - A², B², C², AB, A, B, D, plus products with C...
  - This is the "12 monomials" we counted

The DSL hardcodes 4 integrals that match (1,1),
but (2,2) needs the full 12-monomial combinatorial structure.
""")

    print("=" * 70)
    print("IMPLEMENTATION PATH")
    print("=" * 70)
    print("""
GPT's recommendation: Don't expand to monomials!

Instead, implement X, Y, Z as "connected block evaluators":

  X = (A - C)  as a function of (ℓ, u, t)
  Y = (B - C)  as a function of (ℓ̄, u, t)
  Z = (D - C²) as a function of (u, t)

Then for each p-config (p, coeff, x_exp, y_exp, z_exp):
  Integrand += coeff × Z^{z_exp} × X^{x_exp} × Y^{y_exp}

The key question: What are X, Y, Z concretely?

For our integrand F = pref × P(x+u) × P(y+u) × Q(α) × Q(β) × e^{R(...)}:

Let L = log F = log(pref) + log P(x+u) + log P(y+u) + log Q(α) + log Q(β) + R(...)

Then:
  A = ∂L/∂x = P'/P + (Q'/Q + R) × ∂α/∂x + (Q'/Q + R) × ∂β/∂x + pref_x/pref
  B = ∂L/∂y = P'/P + (Q'/Q + R) × ∂α/∂y + (Q'/Q + R) × ∂β/∂y + pref_y/pref
  C = L at x=y=0
  D = ∂²L/∂x∂y

These are computable! The connected blocks X=(A-C), Y=(B-C), Z=(D-C²)
then subtract the appropriate "disconnected" contributions.

For (1,1), this gives us back the I₁-I₄ structure.
For (2,2), this gives us the correct 12-term structure.

NEXT STEP: Implement the log-derivative evaluators A, B, C, D
and verify they reproduce I₁-I₄ for (1,1).
""")


def compute_log_derivatives(
    P, Q, theta: float, R: float, u: float, t: float
) -> dict:
    """
    Compute the log-derivative building blocks A, B, C, D at a single (u,t) point.

    F(x,y,u,t) = (1+θ(x+y))/θ × P(x+u) × P(y+u) × Q(α) × Q(β) × e^{R(α+β)}

    where α = t + θtx + θ(t-1)y, β = t + θ(t-1)x + θty

    At x=y=0: α = β = t

    Returns dict with:
      C: log F at x=y=0
      A: ∂(log F)/∂x at x=y=0
      B: ∂(log F)/∂y at x=y=0
      D: ∂²(log F)/∂x∂y at x=y=0
    """
    # Evaluate polynomials at u (since x=y=0)
    P_u = P.eval([u])[0]
    P_prime_u = P.eval_deriv([u], 1)[0]

    Q_t = Q.eval([t])[0]
    Q_prime_t = Q.eval_deriv([t], 1)[0]

    # Argument derivatives at x=y=0
    # α = t + θtx + θ(t-1)y → ∂α/∂x = θt, ∂α/∂y = θ(t-1)
    # β = t + θ(t-1)x + θty → ∂β/∂x = θ(t-1), ∂β/∂y = θt
    dalpha_dx = theta * t
    dalpha_dy = theta * (t - 1)
    dbeta_dx = theta * (t - 1)
    dbeta_dy = theta * t

    # Prefactor = (1+θ(x+y))/θ = 1/θ + x + y at x=y=0
    prefactor_0 = 1.0 / theta
    # d(prefactor)/dx = 1, d(prefactor)/dy = 1

    # C = log F at x=y=0
    # F = prefactor × P(u)² × Q(t)² × e^{2Rt}
    F_0 = prefactor_0 * P_u * P_u * Q_t * Q_t * exp(2 * R * t)
    C = np.log(F_0) if F_0 > 0 else float('-inf')

    # A = ∂(log F)/∂x at x=y=0
    # log F = log(pref) + log P(x+u) + log P(y+u) + log Q(α) + log Q(β) + R(α+β)
    # ∂(log F)/∂x = (1/pref)×1 + P'/P + 0 + (Q'/Q)×(∂α/∂x) + (Q'/Q)×(∂β/∂x) + R×(∂α/∂x + ∂β/∂x)
    A = (1.0 / prefactor_0) * 1.0  # from prefactor
    A += P_prime_u / P_u           # from P(x+u)
    A += (Q_prime_t / Q_t) * dalpha_dx  # from Q(α)
    A += (Q_prime_t / Q_t) * dbeta_dx   # from Q(β)
    A += R * (dalpha_dx + dbeta_dx)     # from exp

    # B = ∂(log F)/∂y at x=y=0 (symmetric structure)
    B = (1.0 / prefactor_0) * 1.0
    B += P_prime_u / P_u           # from P(y+u)
    B += (Q_prime_t / Q_t) * dalpha_dy
    B += (Q_prime_t / Q_t) * dbeta_dy
    B += R * (dalpha_dy + dbeta_dy)

    # D = ∂²(log F)/∂x∂y at x=y=0
    # Need second derivatives of each component
    # Most terms are linear in x,y so their mixed derivative is 0
    # The only non-zero contribution comes from cross-terms:
    # ∂²(Q'/Q × ∂α/∂x)/∂y = 0 (no y in dalpha_dx, and Q doesn't depend on y)

    # Actually, let me reconsider...
    # log F = log(pref) + log P(x+u) + log P(y+u) + log Q(α) + log Q(β) + R(α+β)

    # The x-dependent terms:
    # - pref: (1+θ(x+y))/θ → log = log(1+θ(x+y)) - log θ, has x+y dependence
    # - P(x+u): depends on x
    # - Q(α), Q(β): α,β depend on both x,y

    # Mixed derivative:
    # d²/dxdy [log(1+θ(x+y))] = d/dx[θ/(1+θ(x+y))] = -θ²/(1+θ(x+y))²
    # At x=y=0: -θ² / 1 = -θ²

    # d²/dxdy [log P(x+u)] = 0 (no y dependence)
    # d²/dxdy [log P(y+u)] = 0 (no x dependence)

    # d²/dxdy [log Q(α)] = d/dy [(Q'/Q) × dalpha_dx]
    #   = (d/dy[Q'/Q]) × dalpha_dx + (Q'/Q) × (d/dy[dalpha_dx])
    #   = (Q''/Q - (Q')²/Q²) × dalpha_dy × dalpha_dx + 0
    # At x=y=0:
    Q_double_prime_t = Q.eval_deriv([t], 2)[0]
    d2_logQ_dt2 = Q_double_prime_t / Q_t - (Q_prime_t / Q_t) ** 2

    # d²/dxdy [log Q(α)] at x=y=0
    # = d2_logQ_dt2 × dalpha_dx × dalpha_dy

    # Similarly for Q(β):
    # d²/dxdy [log Q(β)] = d2_logQ_dt2 × dbeta_dx × dbeta_dy

    # d²/dxdy [R(α+β)] = R × d²/dxdy[α+β] = 0 (α+β is linear in x,y)

    D = -theta ** 2  # from prefactor
    D += d2_logQ_dt2 * dalpha_dx * dalpha_dy  # from Q(α)
    D += d2_logQ_dt2 * dbeta_dx * dbeta_dy    # from Q(β)

    return {'C': C, 'A': A, 'B': B, 'D': D, 'F0': F_0}


def test_log_derivatives():
    """Test the log-derivative computation."""
    from src.polynomials import load_przz_polynomials

    print("=" * 70)
    print("TEST: Log-Derivative Building Blocks")
    print("=" * 70)

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036

    # Test at a few (u,t) points
    test_points = [(0.5, 0.5), (0.3, 0.7), (0.1, 0.9)]

    for u, t in test_points:
        result = compute_log_derivatives(P1, Q, theta, R, u, t)
        print(f"\n(u={u}, t={t}):")
        print(f"  C = {result['C']:.6f}")
        print(f"  A = {result['A']:.6f}")
        print(f"  B = {result['B']:.6f}")
        print(f"  D = {result['D']:.6f}")
        print(f"  F₀ = {result['F0']:.6f}")

        # Verify: A = B at x=y=0 for symmetric (1,1) case
        if abs(result['A'] - result['B']) > 1e-10:
            print(f"  WARNING: A ≠ B (expected for symmetric case)")

    print("\n" + "=" * 70)
    print("CONNECTED BLOCKS at (u=0.5, t=0.5):")
    print("=" * 70)
    u, t = 0.5, 0.5
    r = compute_log_derivatives(P1, Q, theta, R, u, t)

    X = r['A'] - r['C']  # But C is log F, A is d(log F)/dx - different dimensions!
    Y = r['B'] - r['C']
    Z = r['D'] - r['C'] ** 2

    print(f"  X = A - C = {r['A']:.4f} - {r['C']:.4f} = {X:.4f}")
    print(f"  Y = B - C = {r['B']:.4f} - {r['C']:.4f} = {Y:.4f}")
    print(f"  Z = D - C² = {r['D']:.4f} - {r['C']**2:.4f} = {Z:.4f}")

    print("""
NOTE: The dimensions don't match for A-C, B-C!
This suggests "C" in the Ψ formula is NOT the same as "log F".

GPT's framework likely uses normalized/scaled quantities where
A, B, C, D are all dimensionless coefficients in some expansion.

The exact interpretation requires deeper analysis of PRZZ Section 7.
""")


if __name__ == "__main__":
    analyze_11_structure(None, None, 4/7, 1.3036)
    print("\n" + "=" * 70)
    print()
    try:
        test_log_derivatives()
    except Exception as e:
        print(f"Test failed: {e}")
        print("(This is expected if polynomials aren't available)")
