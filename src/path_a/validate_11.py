#!/usr/bin/env python3
"""
Validation of symbolic (1,1) pair integrals against numeric engine.

Compares the closed-form expressions from symbolic_11.py against
numerical quadrature from przz_exact_i1.py and przz_exact_i2.py.

The symbolic engine expresses each integral as:
    (A(R)·e^{2R} + B(R)) / (C·R^m)

This script evaluates these at specific R values and compares to numeric.
"""
import sympy as sp
from sympy import exp, Rational, N
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.polynomials import load_przz_polynomials
from src.przz_exact_i1 import compute_I1_przz
from src.przz_exact_i2 import compute_I2_przz


def symbolic_I2_11(R_val):
    """
    Compute I₂^{(1,1)} symbolically and evaluate at R=R_val.

    Formula: (1/θ) × ∫₀¹ P₁(u)² du × ∫₀¹ Q(t)² exp(2Rt) dt

    The t-integral has closed form using J_n family.
    """
    from sympy import symbols, integrate, expand, simplify

    R = symbols('R', real=True)
    u, t = symbols('u t', real=True)

    theta = Rational(4, 7)

    # P̃₁ coefficients (universal)
    a0 = Rational(-2, 1)
    a1 = Rational(15, 16)
    a2 = Rational(1, 1)
    a3 = Rational(-3, 5)

    # P₁(u) = u + u(1-u)·P̃₁(1-u)
    def P1(uvar):
        v = 1 - uvar
        tilde = a0 + a1*v + a2*v**2 + a3*v**3
        return uvar + uvar*v*tilde

    # Q coefficients (PRZZ baseline)
    q1 = Rational(636851, 1000000)
    q3 = Rational(-159327, 1000000)
    q5 = Rational(32011, 1000000)
    q0 = 1 - (q1 + q3 + q5)

    def Q(tvar):
        w = 1 - 2*tvar
        return q0 + q1*w + q3*w**3 + q5*w**5

    # u-integral: ∫₀¹ P₁(u)² du
    P1_u = P1(u)
    I_u = integrate(expand(P1_u**2), (u, 0, 1))
    I_u = simplify(I_u)

    # t-integral: ∫₀¹ Q(t)² exp(2Rt) dt
    Q_t = Q(t)
    Q_t_sq = expand(Q_t**2)
    I_t = integrate(Q_t_sq * exp(2*R*t), (t, 0, 1))

    # Handle Piecewise (extract main branch for R ≠ 0)
    if hasattr(I_t, 'args') and len(I_t.args) > 0:
        if hasattr(I_t.args[0], '__iter__'):
            I_t = I_t.args[0][0]

    # I₂ = (1/θ) × I_u × I_t
    I2 = (1/theta) * I_u * I_t
    I2 = simplify(I2)

    # Evaluate at R_val
    return float(N(I2.subs(R, R_val), 50))


def symbolic_I1_11_approx(R_val, n_quad=80):
    """
    Approximate I₁^{(1,1)} using symbolic kernel with numeric quadrature.

    This is a hybrid approach: we build the symbolic kernel at each
    quadrature point and extract the d²/dxdy coefficient.

    For true symbolic, we'd integrate the full expression, but that's
    very slow. This validates the kernel structure.
    """
    import sympy as sp
    from sympy import symbols, diff, exp as sp_exp, Rational

    theta = Rational(4, 7)

    # P̃₁ coefficients
    a0 = Rational(-2, 1)
    a1 = Rational(15, 16)
    a2 = Rational(1, 1)
    a3 = Rational(-3, 5)

    # Q coefficients
    q1 = Rational(636851, 1000000)
    q3 = Rational(-159327, 1000000)
    q5 = Rational(32011, 1000000)
    q0 = 1 - (q1 + q3 + q5)

    x, y, u_sym, t_sym = symbols('x y u t', real=True)
    R = symbols('R', real=True)

    def P1(expr):
        v = 1 - expr
        tilde = a0 + a1*v + a2*v**2 + a3*v**3
        return expr + expr*v*tilde

    def Q(expr):
        w = 1 - 2*expr
        return q0 + q1*w + q3*w**3 + q5*w**5

    # Arguments
    alpha = t_sym + theta*t_sym*x + theta*(t_sym - 1)*y
    beta = t_sym + theta*(t_sym - 1)*x + theta*t_sym*y

    # Kernel
    K = (1/theta + x + y) * (1-u_sym)**2 * P1(u_sym+x) * P1(u_sym+y) * \
        Q(alpha) * Q(beta) * sp_exp(R*(alpha + beta))

    # Mixed derivative at x=y=0
    dK_dxdy = diff(diff(K, x), y)
    dK_dxdy_0 = dK_dxdy.subs({x: 0, y: 0})

    # Compile to numeric function
    kernel_func = sp.lambdify([u_sym, t_sym, R], dK_dxdy_0, modules=['numpy'])

    # Numeric quadrature
    from src.quadrature import gauss_legendre_01
    nodes, weights = gauss_legendre_01(n_quad)

    result = 0.0
    for u, u_w in zip(nodes, weights):
        for t, t_w in zip(nodes, weights):
            val = kernel_func(u, t, R_val)
            if np.isscalar(val):
                result += float(val) * u_w * t_w
            else:
                result += float(val) * u_w * t_w

    return result


def main():
    print("=" * 70)
    print("VALIDATION: Symbolic (1,1) vs Numeric Engine")
    print("=" * 70)

    theta_float = 4.0 / 7.0
    R_values = [1.3036, 1.1167, 0.5, 1.0, 1.5]

    print("\n--- Loading PRZZ polynomials ---")
    P1, P2, P3, Q = load_przz_polynomials()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    print("\n" + "=" * 70)
    print("I₂^{(1,1)} COMPARISON")
    print("=" * 70)
    print(f"\n{'R':>8} {'Symbolic':>16} {'Numeric':>16} {'Rel. Error':>14}")
    print("-" * 56)

    for R_val in R_values:
        # Symbolic evaluation
        I2_symbolic = symbolic_I2_11(R_val)

        # Numeric evaluation
        I2_numeric = compute_I2_przz(theta_float, R_val, 1, 1, polynomials, n_quad=100)

        rel_err = abs(I2_symbolic - I2_numeric.value) / abs(I2_numeric.value) if I2_numeric.value != 0 else 0

        status = "✓" if rel_err < 1e-8 else "✗"
        print(f"{R_val:>8.4f} {I2_symbolic:>16.10f} {I2_numeric.value:>16.10f} {rel_err:>14.2e} {status}")

    print("\n" + "=" * 70)
    print("I₁^{(1,1)} COMPARISON (Symbolic Kernel + Numeric Quadrature)")
    print("=" * 70)
    print(f"\n{'R':>8} {'Symbolic Kern.':>16} {'Numeric':>16} {'Rel. Error':>14}")
    print("-" * 56)

    for R_val in R_values:
        # Symbolic kernel with numeric quadrature
        I1_symbolic = symbolic_I1_11_approx(R_val, n_quad=80)

        # Numeric evaluation
        I1_numeric = compute_I1_przz(theta_float, R_val, 1, 1, polynomials, n_quad=80)

        rel_err = abs(I1_symbolic - I1_numeric.value) / abs(I1_numeric.value) if I1_numeric.value != 0 else 0

        status = "✓" if rel_err < 1e-6 else "✗"
        print(f"{R_val:>8.4f} {I1_symbolic:>16.10f} {I1_numeric.value:>16.10f} {rel_err:>14.2e} {status}")

    print("\n" + "=" * 70)
    print("FULL SYMBOLIC EVALUATION AT R=1.3036")
    print("=" * 70)

    # Run the full symbolic engine once
    print("\nRunning full symbolic computation (this may take a moment)...")

    from src.path_a.symbolic_11 import (
        compute_I1_11, compute_I2_11, compute_I3_11, compute_I4_11
    )

    R_sym = sp.symbols('R', real=True)
    R_test = 1.3036

    # These are the expensive symbolic integrations
    try:
        I2_full = compute_I2_11()
        I2_at_R = float(N(I2_full.subs(R_sym, R_test), 50))
        I2_numeric = compute_I2_przz(theta_float, R_test, 1, 1, polynomials, n_quad=100)
        rel_err = abs(I2_at_R - I2_numeric.value) / abs(I2_numeric.value)
        print(f"\nI₂^{{(1,1)}} at R={R_test}:")
        print(f"  Symbolic: {I2_at_R:.12f}")
        print(f"  Numeric:  {I2_numeric.value:.12f}")
        print(f"  Rel err:  {rel_err:.2e} {'✓' if rel_err < 1e-6 else '✗'}")
    except Exception as e:
        print(f"\nI₂ computation failed: {e}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return


if __name__ == "__main__":
    main()
