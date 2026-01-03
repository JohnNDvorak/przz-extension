#!/usr/bin/env python3
"""
Path A prototype: exact (symbolic) evaluation of PRZZ integrals for the (1,1) pair.

Goal: show that each I_j^{(1,1)} collapses to the form
    ( A_j(R)·exp(2R) + B_j(R) ) / ( C_j · R^m )
with A_j, B_j in Z[R] when P, Q have rational coefficients.

This establishes the "finite algebra" structure needed for Path A's
algebraic cancellation proof.
"""
import sympy as sp
from sympy import (
    Rational, symbols, exp, diff, integrate, simplify, expand,
    factor, together, fraction, Poly, factorial
)

# ============================================================
# Symbols / parameters
# ============================================================
u, t, x, y, R = symbols('u t x y R', real=True)
z = symbols('z')  # placeholder for exp(R) or exp(2R)

# θ = 4/7 (exact)
theta = Rational(4, 7)

# ============================================================
# Q polynomial (PRZZ baseline; q0 enforced by Q(0) = 1)
# ============================================================
# Note: These are the printed PRZZ values as rationals.
# For true exact computation, we need rational reconstruction.
# For now, use high-precision decimal → rational conversion.
q1 = Rational(636851, 1000000)   # ≈ 0.636851
q3 = Rational(-159327, 1000000)  # ≈ -0.159327
q5 = Rational(32011, 1000000)    # ≈ 0.032011
q0 = 1 - (q1 + q3 + q5)

def Q(expr):
    """Q polynomial in w = 1 - 2·expr, where Q(w) = q0 + q1·w + q3·w³ + q5·w⁵."""
    w = 1 - 2*expr
    return q0 + q1*w + q3*w**3 + q5*w**5

# ============================================================
# P₁ polynomial (universal tilde basis; exact rationals)
# ============================================================
# P̃₁ = a₀ + a₁·(1-x) + a₂·(1-x)² + a₃·(1-x)³
# Then P₁(x) = x + x(1-x)·P̃₁((1-x))
a0 = Rational(-2, 1)    # -2
a1 = Rational(15, 16)   # 0.9375
a2 = Rational(1, 1)     # 1
a3 = Rational(-3, 5)    # -0.6

def tilde_P1(v):
    """P̃₁ evaluated at v = (1-x)."""
    return a0 + a1*v + a2*v**2 + a3*v**3

def P1(xvar):
    """P₁(x) = x + x(1-x)·P̃₁(1-x)."""
    v = 1 - xvar
    return xvar + xvar * v * tilde_P1(v)

# Expand and simplify for efficiency
_x = symbols('_x')
P1_expanded = expand(P1(_x))

def P1_of(expr):
    """Evaluate P₁ at a general expression."""
    return P1_expanded.subs(_x, expr)

# ============================================================
# Helper: split expression into exp(2R) and constant parts
# ============================================================
def split_exp2R(expr):
    """
    Rewrite expr in the form (A₂·e^{2R} + A₀) / D
    where A₂, A₀, D are polynomials in R.

    Returns (A₂, A₀, D).
    """
    expr_s = together(expr)
    num, den = fraction(expr_s)
    num = expand(num)
    den = factor(den)

    # Substitute z = exp(2R)
    num_sub = num.subs({exp(R): z**(Rational(1,2)), exp(2*R): z})
    # Simplify any remaining exp(R) terms
    num_sub = num_sub.subs(exp(R), z**(Rational(1,2)))

    # Try to extract as polynomial in z
    try:
        num_sub = expand(num_sub)
        poly = Poly(num_sub, z)
        coeffs = poly.all_coeffs()
        deg = poly.degree()

        if deg == 1:
            # Linear in z: A₂·z + A₀
            A2, A0 = coeffs
            return factor(A2), factor(A0), den
        elif deg == 0:
            # No z term
            return sp.Integer(0), factor(coeffs[0]), den
        else:
            # Higher degree - return all coefficients
            print(f"Warning: degree {deg} in exp(2R)")
            return coeffs, den
    except Exception as e:
        print(f"split_exp2R failed: {e}")
        return expr, sp.Integer(0), sp.Integer(1)

# ============================================================
# PRZZ Kernel Arguments (for (1,1) pair)
# ============================================================
def Arg_alpha(x_val, y_val, t_val):
    """α argument: t + θt·x + θ(t-1)·y"""
    return t_val + theta*t_val*x_val + theta*(t_val - 1)*y_val

def Arg_beta(x_val, y_val, t_val):
    """β argument: t + θ(t-1)·x + θt·y"""
    return t_val + theta*(t_val - 1)*x_val + theta*t_val*y_val

# ============================================================
# I₁^{(1,1)} - Mixed derivative kernel
# ============================================================
def compute_I1_11():
    """
    Compute I₁^{(1,1)} symbolically.

    Kernel: (1/θ + x + y) · (1-u)² · P₁(u+x) · P₁(u+y) · Q(α) · Q(β) · exp(R(α+β))
    Derivative: d²/dxdy at x=y=0
    Integration: over u ∈ [0,1], t ∈ [0,1]
    """
    print("Computing I₁^{(1,1)}...")

    # Arguments
    alpha = Arg_alpha(x, y, t)
    beta = Arg_beta(x, y, t)

    # Kernel
    K = (1/theta + x + y) * (1-u)**2 * P1_of(u+x) * P1_of(u+y) * Q(alpha) * Q(beta) * exp(R*(alpha + beta))

    # Mixed derivative at x=y=0
    dK_dxdy = diff(diff(K, x), y)
    dK_dxdy_0 = dK_dxdy.subs({x: 0, y: 0})

    # Integrate over u
    I_u = integrate(dK_dxdy_0, (u, 0, 1))
    I_u = simplify(I_u)

    # Integrate over t
    I_t = integrate(I_u, (t, 0, 1))

    # The result may be a Piecewise; extract the main branch
    if hasattr(I_t, 'args') and len(I_t.args) > 0:
        if hasattr(I_t.args[0], '__iter__'):
            I1 = I_t.args[0][0]
        else:
            I1 = I_t
    else:
        I1 = I_t

    return simplify(I1)

# ============================================================
# I₂^{(1,1)} - Separable integral
# ============================================================
def compute_I2_11():
    """
    Compute I₂^{(1,1)} symbolically.

    This is the separable piece: (1/θ) · ∫P₁(u)²du · ∫Q(t)²exp(2Rt)dt
    """
    print("Computing I₂^{(1,1)}...")

    # u-integral: ∫₀¹ P₁(u)² du
    I_u = integrate(P1_of(u)**2, (u, 0, 1))

    # t-integral: ∫₀¹ Q(t)² exp(2Rt) dt
    I_t = integrate(Q(t)**2 * exp(2*R*t), (t, 0, 1))

    # Extract main branch if Piecewise
    if hasattr(I_t, 'args') and len(I_t.args) > 0:
        if hasattr(I_t.args[0], '__iter__'):
            I_t = I_t.args[0][0]

    I2 = simplify((1/theta) * I_u * I_t)
    return I2

# ============================================================
# I₃^{(1,1)} - Boundary derivative (x only)
# ============================================================
def compute_I3_11():
    """
    Compute I₃^{(1,1)} symbolically.

    Kernel: (1/θ + x) · (1-u) · P₁(u+x) · P₁(u) · Q(αₓ) · Q(βₓ) · exp(R(αₓ+βₓ))
    where αₓ = t + θt·x, βₓ = t + θ(t-1)·x
    Derivative: -d/dx at x=0
    """
    print("Computing I₃^{(1,1)}...")

    # Arguments (y=0)
    alpha_x = t + theta*t*x
    beta_x = t + theta*(t-1)*x

    # Kernel
    K = (1/theta + x) * (1-u) * P1_of(u+x) * P1_of(u) * Q(alpha_x) * Q(beta_x) * exp(R*(alpha_x + beta_x))

    # Derivative at x=0
    dK_dx = diff(K, x)
    dK_dx_0 = dK_dx.subs(x, 0)

    # Integrate over u
    I_u = integrate(dK_dx_0, (u, 0, 1))
    I_u = simplify(I_u)

    # Integrate over t
    I_t = integrate(I_u, (t, 0, 1))

    if hasattr(I_t, 'args') and len(I_t.args) > 0:
        if hasattr(I_t.args[0], '__iter__'):
            I_t = I_t.args[0][0]

    # Note: I₃ has a negative sign in PRZZ conventions
    I3 = simplify(-I_t)
    return I3

# ============================================================
# I₄^{(1,1)} - Boundary derivative (y only)
# ============================================================
def compute_I4_11():
    """
    Compute I₄^{(1,1)} symbolically.

    Kernel: (1/θ + y) · (1-u) · P₁(u) · P₁(u+y) · Q(αᵧ) · Q(βᵧ) · exp(R(αᵧ+βᵧ))
    where αᵧ = t + θ(t-1)·y, βᵧ = t + θt·y
    Derivative: -d/dy at y=0
    """
    print("Computing I₄^{(1,1)}...")

    # Arguments (x=0)
    alpha_y = t + theta*(t-1)*y
    beta_y = t + theta*t*y

    # Kernel
    K = (1/theta + y) * (1-u) * P1_of(u) * P1_of(u+y) * Q(alpha_y) * Q(beta_y) * exp(R*(alpha_y + beta_y))

    # Derivative at y=0
    dK_dy = diff(K, y)
    dK_dy_0 = dK_dy.subs(y, 0)

    # Integrate over u
    I_u = integrate(dK_dy_0, (u, 0, 1))
    I_u = simplify(I_u)

    # Integrate over t
    I_t = integrate(I_u, (t, 0, 1))

    if hasattr(I_t, 'args') and len(I_t.args) > 0:
        if hasattr(I_t.args[0], '__iter__'):
            I_t = I_t.args[0][0]

    I4 = simplify(-I_t)
    return I4

# ============================================================
# Main: compute and analyze all four integrals
# ============================================================
def main():
    print("=" * 60)
    print("Path A: Symbolic expansion of (1,1) pair integrals")
    print("=" * 60)
    print(f"\nθ = {theta} = {float(theta):.10f}")
    print(f"\nP̃₁ coefficients: a₀={a0}, a₁={a1}, a₂={a2}, a₃={a3}")
    print(f"\nQ coefficients: q₀={float(q0):.6f}, q₁={float(q1):.6f}, q₃={float(q3):.6f}, q₅={float(q5):.6f}")

    results = {}

    # Compute each integral
    for name, compute_fn in [
        ("I1_11", compute_I1_11),
        ("I2_11", compute_I2_11),
        ("I3_11", compute_I3_11),
        ("I4_11", compute_I4_11),
    ]:
        try:
            expr = compute_fn()
            results[name] = expr

            print(f"\n{'='*60}")
            print(f"{name}:")
            print(f"{'='*60}")

            # Try to split into exp(2R) form
            try:
                A2, A0, den = split_exp2R(expr)
                print(f"  Denominator: {den}")

                if isinstance(A2, sp.Expr):
                    try:
                        poly_A2 = Poly(A2, R)
                        print(f"  A₂ (coeff of e^{{2R}}): degree {poly_A2.degree()}")
                        # Print first few coefficients
                        coeffs = poly_A2.all_coeffs()
                        if len(coeffs) <= 5:
                            print(f"    Coefficients: {coeffs}")
                    except:
                        print(f"  A₂: {A2}")

                if isinstance(A0, sp.Expr):
                    try:
                        poly_A0 = Poly(A0, R)
                        print(f"  A₀ (constant term): degree {poly_A0.degree()}")
                    except:
                        print(f"  A₀: {A0}")

            except Exception as e:
                print(f"  Could not split: {e}")
                print(f"  Raw expression: {expr}")

        except Exception as e:
            print(f"\n{name}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    # Check I₃ = I₄ (should hold by symmetry)
    if "I3_11" in results and "I4_11" in results:
        diff_34 = simplify(results["I3_11"] - results["I4_11"])
        print(f"\n{'='*60}")
        print(f"Symmetry check: I₃ - I₄ = {diff_34}")
        if diff_34 == 0:
            print("  ✓ I₃ = I₄ confirmed")
        else:
            print("  ✗ I₃ ≠ I₄ (unexpected)")

    return results


if __name__ == "__main__":
    results = main()
