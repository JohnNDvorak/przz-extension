#!/usr/bin/env python3
"""
Path A: Symbolic evaluation of all PRZZ pair integrals.

Extends symbolic_11.py to compute I₁, I₂, I₃, I₄ for all 6 pairs:
  (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)

Each integral collapses to the form:
    (A(R)·e^{2R} + B(R)) / (C·R^m)

where A, B are polynomials in R with rational coefficients.

Usage:
    python -m src.path_a.symbolic_pairs [--pair 11] [--quick]
"""
import sympy as sp
from sympy import (
    Rational, symbols, exp, diff, integrate, simplify, expand,
    factor, together, fraction, Poly, factorial, N
)
from typing import Dict, Tuple, Optional
import argparse
import sys

from .optimal_coeffs import (
    theta, R, build_P1, build_P2, build_P3, get_P, build_Q
)

# =============================================================================
# Symbols
# =============================================================================
u, t, x, y = symbols('u t x y', real=True)
z = symbols('z')  # placeholder for exp(2R)


# =============================================================================
# Affine eigenvalue arguments
# =============================================================================
def Arg_alpha(x_val, y_val, t_val):
    """α argument: t + θt·x + θ(t-1)·y"""
    return t_val + theta*t_val*x_val + theta*(t_val - 1)*y_val


def Arg_beta(x_val, y_val, t_val):
    """β argument: t + θ(t-1)·x + θt·y"""
    return t_val + theta*(t_val - 1)*x_val + theta*t_val*y_val


# =============================================================================
# Helper: split expression into exp(2R) form
# =============================================================================
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
    num_sub = num.subs({exp(R): z**(Rational(1, 2)), exp(2*R): z})
    num_sub = num_sub.subs(exp(R), z**(Rational(1, 2)))

    try:
        num_sub = expand(num_sub)
        poly = Poly(num_sub, z)
        coeffs = poly.all_coeffs()
        deg = poly.degree()

        if deg == 1:
            A2, A0 = coeffs
            return factor(A2), factor(A0), den
        elif deg == 0:
            return sp.Integer(0), factor(coeffs[0]), den
        else:
            print(f"Warning: degree {deg} in exp(2R)")
            return coeffs, den
    except Exception as e:
        print(f"split_exp2R failed: {e}")
        return expr, sp.Integer(0), sp.Integer(1)


# =============================================================================
# I₁^{(ℓ₁,ℓ₂)} - Mixed derivative kernel
# =============================================================================
def compute_I1(ell1: int, ell2: int, verbose: bool = True):
    """
    Compute I₁^{(ℓ₁,ℓ₂)} symbolically.

    Kernel: (1/θ + x + y) · (1-u)^{ℓ₁+ℓ₂} · P_{ℓ₁}(u+x) · P_{ℓ₂}(u+y)
            × Q(α) · Q(β) · exp(R(α+β))
    Derivative: d^{ℓ₁+ℓ₂}/dx^{ℓ₁}dy^{ℓ₂} at x=y=0

    For d=1 (K=3), all ℓ values are 1, so derivative is always d²/dxdy.
    """
    if verbose:
        print(f"Computing I₁^{{({ell1},{ell2})}}...")

    # Get polynomials
    P_ell1 = get_P(ell1, u + x)
    P_ell2 = get_P(ell2, u + y)

    # Arguments
    alpha = Arg_alpha(x, y, t)
    beta = Arg_beta(x, y, t)

    # Kernel (prefactor is (θ(x+y)+1)/θ = 1/θ + x + y)
    K = ((1/theta + x + y) * (1-u)**(ell1 + ell2) *
         P_ell1 * P_ell2 *
         build_Q(alpha) * build_Q(beta) *
         exp(R*(alpha + beta)))

    # Mixed derivative d²/dxdy at x=y=0
    dK_dxdy = diff(diff(K, x), y)
    dK_dxdy_0 = dK_dxdy.subs({x: 0, y: 0})

    # Integrate over u
    I_u = integrate(dK_dxdy_0, (u, 0, 1))
    I_u = simplify(I_u)

    # Integrate over t
    I_t = integrate(I_u, (t, 0, 1))

    # Extract main branch from Piecewise
    if hasattr(I_t, 'args') and len(I_t.args) > 0:
        if hasattr(I_t.args[0], '__iter__'):
            I1 = I_t.args[0][0]
        else:
            I1 = I_t
    else:
        I1 = I_t

    return simplify(I1)


# =============================================================================
# I₂^{(ℓ₁,ℓ₂)} - Separable integral
# =============================================================================
def compute_I2(ell1: int, ell2: int, verbose: bool = True):
    """
    Compute I₂^{(ℓ₁,ℓ₂)} symbolically.

    This is the separable piece: (1/θ) · ∫P_{ℓ₁}(u)P_{ℓ₂}(u)du · ∫Q(t)²exp(2Rt)dt
    """
    if verbose:
        print(f"Computing I₂^{{({ell1},{ell2})}}...")

    # u-integral: ∫₀¹ P_{ℓ₁}(u) P_{ℓ₂}(u) du
    P_ell1 = get_P(ell1, u)
    P_ell2 = get_P(ell2, u)
    I_u = integrate(expand(P_ell1 * P_ell2), (u, 0, 1))

    # t-integral: ∫₀¹ Q(t)² exp(2Rt) dt
    Q_t = build_Q(t)
    I_t = integrate(expand(Q_t**2) * exp(2*R*t), (t, 0, 1))

    # Extract main branch from Piecewise
    if hasattr(I_t, 'args') and len(I_t.args) > 0:
        if hasattr(I_t.args[0], '__iter__'):
            I_t = I_t.args[0][0]

    I2 = simplify((1/theta) * I_u * I_t)
    return I2


# =============================================================================
# I₃^{(ℓ₁,ℓ₂)} - Boundary derivative (x only)
# =============================================================================
def compute_I3(ell1: int, ell2: int, verbose: bool = True):
    """
    Compute I₃^{(ℓ₁,ℓ₂)} symbolically.

    Kernel: (1/θ + x) · (1-u)^{ℓ₁} · P_{ℓ₁}(u+x) · P_{ℓ₂}(u)
            × Q(αₓ) · Q(βₓ) · exp(R(αₓ+βₓ))
    where αₓ = t + θt·x, βₓ = t + θ(t-1)·x  (y=0)
    Derivative: -d^{ℓ₁}/dx^{ℓ₁} at x=0
    """
    if verbose:
        print(f"Computing I₃^{{({ell1},{ell2})}}...")

    # Get polynomials (y=0, so P_{ℓ₂}(u) not P_{ℓ₂}(u+y))
    P_ell1_shifted = get_P(ell1, u + x)
    P_ell2 = get_P(ell2, u)

    # Arguments (y=0)
    alpha_x = t + theta*t*x
    beta_x = t + theta*(t-1)*x

    # Kernel
    K = ((1/theta + x) * (1-u)**ell1 *
         P_ell1_shifted * P_ell2 *
         build_Q(alpha_x) * build_Q(beta_x) *
         exp(R*(alpha_x + beta_x)))

    # Derivative d/dx at x=0 (for d=1, all ℓ=1)
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

    # I₃ has negative sign in PRZZ conventions
    I3 = simplify(-I_t)
    return I3


# =============================================================================
# I₄^{(ℓ₁,ℓ₂)} - Boundary derivative (y only)
# =============================================================================
def compute_I4(ell1: int, ell2: int, verbose: bool = True):
    """
    Compute I₄^{(ℓ₁,ℓ₂)} symbolically.

    Kernel: (1/θ + y) · (1-u)^{ℓ₂} · P_{ℓ₁}(u) · P_{ℓ₂}(u+y)
            × Q(αᵧ) · Q(βᵧ) · exp(R(αᵧ+βᵧ))
    where αᵧ = t + θ(t-1)·y, βᵧ = t + θt·y  (x=0)
    Derivative: -d^{ℓ₂}/dy^{ℓ₂} at y=0
    """
    if verbose:
        print(f"Computing I₄^{{({ell1},{ell2})}}...")

    # Get polynomials (x=0, so P_{ℓ₁}(u) not P_{ℓ₁}(u+x))
    P_ell1 = get_P(ell1, u)
    P_ell2_shifted = get_P(ell2, u + y)

    # Arguments (x=0)
    alpha_y = t + theta*(t-1)*y
    beta_y = t + theta*t*y

    # Kernel
    K = ((1/theta + y) * (1-u)**ell2 *
         P_ell1 * P_ell2_shifted *
         build_Q(alpha_y) * build_Q(beta_y) *
         exp(R*(alpha_y + beta_y)))

    # Derivative d/dy at y=0
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


# =============================================================================
# Compute all integrals for a pair
# =============================================================================
def compute_pair(ell1: int, ell2: int, verbose: bool = True) -> Dict:
    """
    Compute all four integrals for pair (ℓ₁, ℓ₂).

    Returns dict with I1, I2, I3, I4 and their exp(2R) decompositions.
    """
    results = {}

    for name, compute_fn in [
        ("I1", lambda: compute_I1(ell1, ell2, verbose)),
        ("I2", lambda: compute_I2(ell1, ell2, verbose)),
        ("I3", lambda: compute_I3(ell1, ell2, verbose)),
        ("I4", lambda: compute_I4(ell1, ell2, verbose)),
    ]:
        try:
            expr = compute_fn()
            A2, A0, den = split_exp2R(expr)
            results[name] = {
                'expr': expr,
                'A2': A2,
                'A0': A0,
                'denominator': den,
            }
        except Exception as e:
            if verbose:
                print(f"  {name}: FAILED - {e}")
            results[name] = {'error': str(e)}

    return results


# =============================================================================
# Main: compute specified pair(s)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Symbolic PRZZ pair computation")
    parser.add_argument('--pair', type=str, default='11',
                        help='Pair to compute: 11, 12, 13, 22, 23, 33, or "all"')
    parser.add_argument('--quick', action='store_true',
                        help='Only compute I2 (fastest)')
    parser.add_argument('--eval-R', type=float, default=None,
                        help='Evaluate at specific R value')
    args = parser.parse_args()

    print("=" * 70)
    print("PATH A: SYMBOLIC PRZZ PAIR COMPUTATION")
    print("=" * 70)
    print(f"\nθ = {theta} = {float(theta):.10f}")

    if args.pair == 'all':
        pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    else:
        ell1 = int(args.pair[0])
        ell2 = int(args.pair[1])
        pairs = [(ell1, ell2)]

    all_results = {}

    for ell1, ell2 in pairs:
        print(f"\n{'='*60}")
        print(f"PAIR ({ell1},{ell2})")
        print(f"{'='*60}")

        if args.quick:
            # Only compute I2
            I2 = compute_I2(ell1, ell2)
            A2, A0, den = split_exp2R(I2)
            print(f"\nI₂^{{({ell1},{ell2})}}:")
            print(f"  Denominator: {den}")
            if isinstance(A2, sp.Expr):
                try:
                    deg = Poly(A2, R).degree()
                    print(f"  A₂ degree: {deg}")
                except:
                    print(f"  A₂: {A2}")

            if args.eval_R is not None:
                val = float(N(I2.subs(R, args.eval_R), 20))
                print(f"  Value at R={args.eval_R}: {val:.12f}")
        else:
            results = compute_pair(ell1, ell2)
            all_results[(ell1, ell2)] = results

            for name in ["I1", "I2", "I3", "I4"]:
                if name in results and 'expr' in results[name]:
                    r = results[name]
                    print(f"\n{name}^{{({ell1},{ell2})}}:")
                    print(f"  Denominator: {r['denominator']}")

                    if isinstance(r['A2'], sp.Expr) and r['A2'] != 0:
                        try:
                            poly = Poly(r['A2'], R)
                            print(f"  A₂ (coeff of e^{{2R}}): degree {poly.degree()}")
                        except:
                            print(f"  A₂: {r['A2']}")

                    if args.eval_R is not None:
                        val = float(N(r['expr'].subs(R, args.eval_R), 20))
                        print(f"  Value at R={args.eval_R}: {val:.12f}")

        # Check I₃ = I₄ symmetry for diagonal pairs
        if ell1 == ell2 and not args.quick:
            if 'I3' in results and 'I4' in results:
                if 'expr' in results['I3'] and 'expr' in results['I4']:
                    diff_34 = simplify(results['I3']['expr'] - results['I4']['expr'])
                    print(f"\nSymmetry check: I₃ - I₄ = {diff_34}")
                    if diff_34 == 0:
                        print("  ✓ I₃ = I₄ confirmed")

    print("\n" + "=" * 70)
    print("COMPUTATION COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
