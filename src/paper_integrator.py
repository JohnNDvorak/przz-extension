"""
src/paper_integrator.py
Direct implementation of PRZZ equations for diagnostic comparison.

This module implements I1, I2, I3, I4 directly from PRZZ paper equations,
bypassing the DSL entirely. The purpose is to diagnose whether the DSL
has bugs by comparing term-by-term.

PRZZ Reference: RMS_PRZZ.tex lines 1530-1569

Key equations implemented:
- I1: Line 1530-1532 (main coupled term with d²/dxdy)
- I2: Line 1548 (decoupled term, no derivatives)
- I3: Line 1562-1563 (single x derivative)
- I4: Line 1568-1569 (single y derivative, by symmetry)

NOTE: This does NOT include I5 (the arithmetic factor contribution).
Per PRZZ lines 1626-1628, I5 is O(T/L) and absorbed into the error term.
The goal is to match PRZZ's κ using only I1-I4.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Optional
from math import factorial

from src.quadrature import tensor_grid_2d
from src.series import TruncatedSeries
from src.composition import compose_polynomial_on_affine, compose_exp_on_affine


def _compute_I1_11_paper(
    P1, P2, Q,
    theta: float,
    R: float,
    n_quadrature: int = 60
) -> float:
    """
    Compute I1 for (1,1) pair directly from PRZZ equation.

    PRZZ line 1530-1532:
    I₁ = T·Φ̂(0)·d²/dxdy [(θ(x+y)+1)/θ] ×
         ∫∫ (1-u)² P₁(x+u) P₂(y+u)
         × exp(R[θt(x+y)-θy+t]) exp(R[θt(x+y)-θx+t])
         × Q(θt(x+y)-θy+t) Q(θt(x+y)-θx+t) du dt |_{x=y=0}

    We omit T·Φ̂(0) as it cancels in the final c computation.

    The derivative structure:
    Let f(x,y) = (θ(x+y)+1)/θ = 1/θ + x + y
    Let g(x,y) = integral terms

    d²/dxdy [f·g] at x=y=0:
    = f·g_xy + f_x·g_y + f_y·g_x + f_xy·g
    = (1/θ)·g_xy + 1·g_y + 1·g_x + 0·g

    So we need: g(0,0), g_x(0,0), g_y(0,0), g_xy(0,0)
    """
    U, T, W = tensor_grid_2d(n_quadrature)
    var_names = ("x", "y")

    # Accumulate series coefficients
    integral_00 = 0.0  # g(0,0)
    integral_x = 0.0   # g_x(0,0) = coeff of x
    integral_y = 0.0   # g_y(0,0) = coeff of y
    integral_xy = 0.0  # g_xy(0,0) = coeff of xy

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            u = U[i, j]
            t = T[i, j]
            w = W[i, j]

            # Build the integrand as a TruncatedSeries in (x, y)
            # Poly prefactor: (1-u)²
            poly_pref = (1 - u) ** 2

            # P₁(x+u): lin = {"x": 1}
            P1_series = compose_polynomial_on_affine(
                P1, u0=u, lin={"x": 1.0}, var_names=var_names
            )

            # P₂(y+u): lin = {"y": 1}
            P2_series = compose_polynomial_on_affine(
                P2, u0=u, lin={"y": 1.0}, var_names=var_names
            )

            # Arg_α = θt(x+y) - θy + t = t + θt·x + θ(t-1)·y
            # Arg_β = θt(x+y) - θx + t = t + θ(t-1)·x + θt·y
            arg_alpha_base = t
            arg_alpha_lin = {"x": theta * t, "y": theta * (t - 1)}

            arg_beta_base = t
            arg_beta_lin = {"x": theta * (t - 1), "y": theta * t}

            # exp(R·Arg_α)
            exp_alpha = compose_exp_on_affine(R, arg_alpha_base, arg_alpha_lin, var_names)

            # exp(R·Arg_β)
            exp_beta = compose_exp_on_affine(R, arg_beta_base, arg_beta_lin, var_names)

            # Q(Arg_α)
            Q_alpha = compose_polynomial_on_affine(Q, arg_alpha_base, arg_alpha_lin, var_names)

            # Q(Arg_β)
            Q_beta = compose_polynomial_on_affine(Q, arg_beta_base, arg_beta_lin, var_names)

            # Multiply all factors
            product = P1_series * P2_series * exp_alpha * exp_beta * Q_alpha * Q_beta
            product = product * poly_pref

            # Extract coefficients
            integral_00 += w * product.extract(())
            integral_x += w * product.extract(("x",))
            integral_y += w * product.extract(("y",))
            integral_xy += w * product.extract(("x", "y"))

    # Apply derivative structure:
    # d²/dxdy [f·g] = (1/θ)·g_xy + g_y + g_x
    result = (1.0 / theta) * integral_xy + integral_y + integral_x

    return result


def _compute_I2_11_paper(
    P1, P2, Q,
    theta: float,
    R: float,
    n_quadrature: int = 60
) -> float:
    """
    Compute I2 for (1,1) pair directly from PRZZ equation.

    PRZZ line 1548:
    I₂ = T·Φ̂(0)/θ ∫∫ Q(t)² exp(2Rt) P₁(u) P₂(u) dt du

    No formal variables, no derivatives - just a simple 2D integral.
    """
    U, T, W = tensor_grid_2d(n_quadrature)

    result = 0.0
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            u = U[i, j]
            t = T[i, j]
            w = W[i, j]

            # Evaluate each factor
            P1_val = P1.eval_deriv(np.array([u]), 0)[0]
            P2_val = P2.eval_deriv(np.array([u]), 0)[0]
            Q_val = Q.eval_deriv(np.array([t]), 0)[0]
            exp_val = np.exp(2 * R * t)

            result += w * P1_val * P2_val * (Q_val ** 2) * exp_val

    # Prefactor: 1/θ
    return result / theta


def _compute_I3_11_paper(
    P1, P2, Q,
    theta: float,
    R: float,
    n_quadrature: int = 60
) -> float:
    """
    Compute I3 for (1,1) pair directly from PRZZ equation.

    PRZZ line 1562-1563:
    I₃ = -T·Φ̂(0)·(1+θx)/θ · d/dx ∫∫ (1-u) P₁(x+u) P₂(u)
         × exp(R[t+θxt]) exp(R[-θx+t+θxt])
         × Q(t+θxt) Q(-θx+t+θxt) dt du |_{x=0}

    Derivative structure:
    Let f(x) = (1+θx)/θ = 1/θ + x
    Let g(x) = integral terms

    d/dx [f·g] at x=0:
    = f·g_x + f_x·g
    = (1/θ)·g_x + 1·g

    So we need: g(0), g_x(0)
    """
    U, T, W = tensor_grid_2d(n_quadrature)
    var_names = ("x",)

    integral_x = 0.0   # g_x(0) = coeff of x

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            u = U[i, j]
            t = T[i, j]
            w = W[i, j]

            # Poly prefactor: (1-u)
            poly_pref = 1 - u

            # P₁(x+u): lin = {"x": 1}
            P1_series = compose_polynomial_on_affine(
                P1, u0=u, lin={"x": 1.0}, var_names=var_names
            )

            # P₂(u): no x dependence
            P2_val = P2.eval_deriv(np.array([u]), 0)[0]

            # Arguments for Q and exp:
            # Arg₁ = t + θxt = t(1 + θx)
            # Arg₂ = -θx + t + θxt = t + θx(t-1)
            arg1_base = t
            arg1_lin = {"x": theta * t}

            arg2_base = t
            arg2_lin = {"x": theta * (t - 1)}

            # exp(R·Arg₁) and exp(R·Arg₂)
            exp1 = compose_exp_on_affine(R, arg1_base, arg1_lin, var_names)
            exp2 = compose_exp_on_affine(R, arg2_base, arg2_lin, var_names)

            # Q(Arg₁) and Q(Arg₂)
            Q1 = compose_polynomial_on_affine(Q, arg1_base, arg1_lin, var_names)
            Q2 = compose_polynomial_on_affine(Q, arg2_base, arg2_lin, var_names)

            # Multiply all factors
            product = P1_series * exp1 * exp2 * Q1 * Q2 * (poly_pref * P2_val)

            # Extract derivative coefficient
            integral_x += w * product.extract(("x",))

    # PRZZ structure (line 1562-1563):
    # I₃ = -[(1+θx)/θ]|_{x=0} × (d/dx [integral])|_{x=0}
    #    = -(1/θ) × g'(0)
    #
    # The prefactor (1+θx)/θ evaluates at x=0 BEFORE multiplying the derivative.
    # It does NOT participate in Leibniz rule expansion.
    result = -(1.0 / theta) * integral_x

    return result


def _compute_I4_11_paper(
    P1, P2, Q,
    theta: float,
    R: float,
    n_quadrature: int = 60
) -> float:
    """
    Compute I4 for (1,1) pair directly from PRZZ equation.

    PRZZ line 1568-1569:
    I₄ = -T·Φ̂(0)·(1+θy)/θ · d/dy ∫∫ (1-u) P₁(u) P₂(y+u)
         × exp(R[t+θyt]) exp(R[-θy+t+θyt])
         × Q(t+θyt) Q(-θy+t+θyt) dt du |_{y=0}

    This is symmetric to I3, with x→y and P1↔P2 roles swapped.
    """
    U, T, W = tensor_grid_2d(n_quadrature)
    var_names = ("y",)

    integral_0 = 0.0   # g(0)
    integral_y = 0.0   # g_y(0) = coeff of y

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            u = U[i, j]
            t = T[i, j]
            w = W[i, j]

            # Poly prefactor: (1-u)
            poly_pref = 1 - u

            # P₁(u): no y dependence
            P1_val = P1.eval_deriv(np.array([u]), 0)[0]

            # P₂(y+u): lin = {"y": 1}
            P2_series = compose_polynomial_on_affine(
                P2, u0=u, lin={"y": 1.0}, var_names=var_names
            )

            # Arguments for Q and exp:
            # Arg₁ = t + θyt = t(1 + θy)
            # Arg₂ = -θy + t + θyt = t + θy(t-1)
            arg1_base = t
            arg1_lin = {"y": theta * t}

            arg2_base = t
            arg2_lin = {"y": theta * (t - 1)}

            # exp(R·Arg₁) and exp(R·Arg₂)
            exp1 = compose_exp_on_affine(R, arg1_base, arg1_lin, var_names)
            exp2 = compose_exp_on_affine(R, arg2_base, arg2_lin, var_names)

            # Q(Arg₁) and Q(Arg₂)
            Q1 = compose_polynomial_on_affine(Q, arg1_base, arg1_lin, var_names)
            Q2 = compose_polynomial_on_affine(Q, arg2_base, arg2_lin, var_names)

            # Multiply all factors
            product = P2_series * exp1 * exp2 * Q1 * Q2 * (poly_pref * P1_val)

            # Extract derivative coefficient
            integral_y += w * product.extract(("y",))

    # PRZZ structure (line 1568-1569):
    # I₄ = -[(1+θy)/θ]|_{y=0} × (d/dy [integral])|_{y=0}
    #    = -(1/θ) × g'(0)
    #
    # Same as I3: prefactor evaluates at y=0 BEFORE multiplying derivative.
    result = -(1.0 / theta) * integral_y

    return result


def compute_pair_11_paper(
    P1, P2, Q,
    theta: float,
    R: float,
    n_quadrature: int = 60,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute all terms for (1,1) pair using paper equations.

    Returns dict with I1, I2, I3, I4 and total.
    """
    I1 = _compute_I1_11_paper(P1, P2, Q, theta, R, n_quadrature)
    I2 = _compute_I2_11_paper(P1, P2, Q, theta, R, n_quadrature)
    I3 = _compute_I3_11_paper(P1, P2, Q, theta, R, n_quadrature)
    I4 = _compute_I4_11_paper(P1, P2, Q, theta, R, n_quadrature)

    total = I1 + I2 + I3 + I4

    if verbose:
        print(f"Paper (1,1): I1={I1:.10e}, I2={I2:.10e}, I3={I3:.10e}, I4={I4:.10e}")
        print(f"Paper (1,1) total: {total:.10e}")

    return {
        "I1_11": I1,
        "I2_11": I2,
        "I3_11": I3,
        "I4_11": I4,
        "total_11": total
    }


def compute_c_paper_11_only(
    polynomials,
    theta: float,
    R: float,
    n_quadrature: int = 60,
    verbose: bool = False
) -> float:
    """
    Compute c using only (1,1) pair with paper equations.

    This is a minimal diagnostic to compare against DSL.
    For full c computation, need all pairs.

    Args:
        polynomials: Dict or tuple with P1, P2 (=P1 for ℓ=1), Q
        theta: θ parameter
        R: R parameter
        n_quadrature: quadrature points
        verbose: print diagnostics

    Returns:
        c contribution from (1,1) pair only
    """
    # API normalization
    if isinstance(polynomials, tuple) and len(polynomials) >= 3:
        P1, P2, P3, Q = polynomials[0], polynomials[1], polynomials[2], polynomials[3]
    elif isinstance(polynomials, dict):
        P1 = polynomials['P1']
        Q = polynomials['Q']
    else:
        raise TypeError(f"polynomials must be dict or tuple, got {type(polynomials)}")

    # For (1,1) pair, both P_left and P_right are P1
    result = compute_pair_11_paper(P1, P1, Q, theta, R, n_quadrature, verbose)

    # Apply factorial normalization for (1,1): 1/(1!×1!) = 1
    # Symmetry factor for diagonal: 1
    c_11 = result["total_11"]

    if verbose:
        print(f"c contribution from (1,1): {c_11:.10e}")

    return c_11


# ============================================================================
# Extended paper integrator for all K=3 pairs
# ============================================================================

def _compute_I1_pair_paper(
    P_left, P_right, Q,
    l1: int, l2: int,
    theta: float,
    R: float,
    n_quadrature: int = 60
) -> float:
    """
    Compute I1 for arbitrary (l1, l2) pair directly from PRZZ equation.

    Generalization of I1 with:
    - vars = (x1, ..., x_{l1}, y1, ..., y_{l2})
    - deriv_orders all = 1
    - prefactor (1-u)^{l1+l2}

    The derivative structure generalizes:
    d^{l1+l2}/dx1...dxl1 dy1...dyl2 [(θS+1)/θ × integrand]
    where S = Σxi + Σyj
    """
    var_names_x = tuple(f"x{i+1}" for i in range(l1))
    var_names_y = tuple(f"y{j+1}" for j in range(l2))
    var_names = var_names_x + var_names_y

    U, T, W = tensor_grid_2d(n_quadrature)

    # We need to extract the coefficient of x1·x2·...·xl1·y1·y2·...·yl2
    # after applying d/dx1...d/dxl1 d/dy1...d/dyl2 to [(θS+1)/θ × g]

    # The prefactor (θS+1)/θ = 1/θ + x1 + ... + xl1 + y1 + ... + yl2
    # has derivatives that contribute to lower-order terms

    # For the full derivative, we get:
    # coeff(x1...xl1 y1...yl2) in [(1/θ + S) × g]
    # = (1/θ) × coeff(x1...xl1 y1...yl2 in g)
    #   + Σ coeff(x1...xi-1 xi+1...xl1 y1...yl2 in g)  [from each xi term in S]
    #   + Σ coeff(x1...xl1 y1...yj-1 yj+1...yl2 in g)  [from each yj term in S]

    integral = 0.0

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            u = U[i, j]
            t = T[i, j]
            w = W[i, j]

            # Build integrand as series in all variables

            # Poly prefactor: (1-u)^{l1+l2}
            poly_pref = (1 - u) ** (l1 + l2)

            # P_left(Σxi + u): sum of x variables
            P_left_lin = {f"x{i+1}": 1.0 for i in range(l1)}
            P_left_series = compose_polynomial_on_affine(
                P_left, u0=u, lin=P_left_lin, var_names=var_names
            )

            # P_right(Σyj + u): sum of y variables
            P_right_lin = {f"y{j+1}": 1.0 for j in range(l2)}
            P_right_series = compose_polynomial_on_affine(
                P_right, u0=u, lin=P_right_lin, var_names=var_names
            )

            # Arguments (from PRZZ):
            # S = Σxi + Σyj, X = Σxi, Y = Σyj
            # Arg_α = θtS - θY + t = t + θt·X + θ(t-1)·Y
            # Arg_β = θtS - θX + t = t + θ(t-1)·X + θt·Y

            arg_alpha_lin = {}
            for var in var_names_x:
                arg_alpha_lin[var] = theta * t
            for var in var_names_y:
                arg_alpha_lin[var] = theta * (t - 1)

            arg_beta_lin = {}
            for var in var_names_x:
                arg_beta_lin[var] = theta * (t - 1)
            for var in var_names_y:
                arg_beta_lin[var] = theta * t

            # exp and Q factors
            exp_alpha = compose_exp_on_affine(R, t, arg_alpha_lin, var_names)
            exp_beta = compose_exp_on_affine(R, t, arg_beta_lin, var_names)
            Q_alpha = compose_polynomial_on_affine(Q, t, arg_alpha_lin, var_names)
            Q_beta = compose_polynomial_on_affine(Q, t, arg_beta_lin, var_names)

            # Full product
            product = P_left_series * P_right_series * exp_alpha * exp_beta * Q_alpha * Q_beta
            product = product * poly_pref

            # Extract the full derivative coefficient
            full_coeff = product.extract(var_names)

            # Also need lower-order coefficients for prefactor derivatives
            # coeff of (all x vars) × (all but one y var) for each y
            # and coeff of (all but one x var) × (all y vars) for each x

            lower_coeffs = 0.0

            # Contribution from each xi in the prefactor
            for k in range(l1):
                # All x vars except x_{k+1}, all y vars
                subset = tuple(f"x{i+1}" for i in range(l1) if i != k) + var_names_y
                if len(subset) == l1 + l2 - 1:
                    lower_coeffs += product.extract(subset)

            # Contribution from each yj in the prefactor
            for k in range(l2):
                # All x vars, all y vars except y_{k+1}
                subset = var_names_x + tuple(f"y{j+1}" for j in range(l2) if j != k)
                if len(subset) == l1 + l2 - 1:
                    lower_coeffs += product.extract(subset)

            # PRZZ structure: [(θS+1)/θ]|_{S=0} × derivative = (1/θ) × derivative
            # The prefactor evaluates at S=0 BEFORE multiplying the derivative.
            # So we just need the full coefficient multiplied by 1/θ.
            # The lower_coeffs come from the prefactor Leibniz expansion, which is INCORRECT.
            #
            # CORRECT interpretation: (1/θ) × [derivative coefficient]
            contrib = (1.0 / theta) * full_coeff

            integral += w * contrib

    return integral


def _compute_I2_pair_paper(
    P_left, P_right, Q,
    l1: int, l2: int,
    theta: float,
    R: float,
    n_quadrature: int = 60
) -> float:
    """
    Compute I2 for arbitrary (l1, l2) pair.

    I2 has no formal variables - just a 2D integral of:
    (1/θ) × P_left(u) × P_right(u) × Q(t)² × exp(2Rt)

    Note: For l1≠l2 pairs, we need appropriate polynomial combinations.
    """
    U, T, W = tensor_grid_2d(n_quadrature)

    result = 0.0
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            u = U[i, j]
            t = T[i, j]
            w = W[i, j]

            P_left_val = P_left.eval_deriv(np.array([u]), 0)[0]
            P_right_val = P_right.eval_deriv(np.array([u]), 0)[0]
            Q_val = Q.eval_deriv(np.array([t]), 0)[0]
            exp_val = np.exp(2 * R * t)

            result += w * P_left_val * P_right_val * (Q_val ** 2) * exp_val

    return result / theta


def _compute_I3_pair_paper(
    P_left, P_right, Q,
    l1: int, l2: int,
    theta: float,
    R: float,
    n_quadrature: int = 60
) -> float:
    """
    Compute I3 for arbitrary (l1, l2) pair.

    I3 involves derivatives only in the x variables (l1 of them),
    with y variables set to 0.
    """
    var_names = tuple(f"x{i+1}" for i in range(l1))

    U, T, W = tensor_grid_2d(n_quadrature)

    integral = 0.0

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            u = U[i, j]
            t = T[i, j]
            w = W[i, j]

            # Poly prefactor: (1-u)^{l1} for I3-type
            poly_pref = (1 - u) ** l1

            # P_left(Σxi + u)
            P_left_lin = {f"x{i+1}": 1.0 for i in range(l1)}
            P_left_series = compose_polynomial_on_affine(
                P_left, u0=u, lin=P_left_lin, var_names=var_names
            )

            # P_right(u) - no x dependence (y=0)
            P_right_val = P_right.eval_deriv(np.array([u]), 0)[0]

            # Arguments with y=0:
            # Arg₁ = t(1 + θX) where X = Σxi
            # Arg₂ = t + θX(t-1)
            arg1_lin = {f"x{i+1}": theta * t for i in range(l1)}
            arg2_lin = {f"x{i+1}": theta * (t - 1) for i in range(l1)}

            exp1 = compose_exp_on_affine(R, t, arg1_lin, var_names)
            exp2 = compose_exp_on_affine(R, t, arg2_lin, var_names)
            Q1 = compose_polynomial_on_affine(Q, t, arg1_lin, var_names)
            Q2 = compose_polynomial_on_affine(Q, t, arg2_lin, var_names)

            product = P_left_series * exp1 * exp2 * Q1 * Q2 * (poly_pref * P_right_val)

            # Extract full x-derivative coefficient
            full_coeff = product.extract(var_names)

            # PRZZ structure: -[(1+θX)/θ]|_{X=0} × derivative = -(1/θ) × derivative
            contrib = (1.0 / theta) * full_coeff
            integral += w * contrib

    # Negative sign from PRZZ
    return -integral


def _compute_I4_pair_paper(
    P_left, P_right, Q,
    l1: int, l2: int,
    theta: float,
    R: float,
    n_quadrature: int = 60
) -> float:
    """
    Compute I4 for arbitrary (l1, l2) pair.

    I4 involves derivatives only in the y variables (l2 of them),
    with x variables set to 0. Symmetric to I3.
    """
    var_names = tuple(f"y{j+1}" for j in range(l2))

    U, T, W = tensor_grid_2d(n_quadrature)

    integral = 0.0

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            u = U[i, j]
            t = T[i, j]
            w = W[i, j]

            # Poly prefactor: (1-u)^{l2} for I4-type
            poly_pref = (1 - u) ** l2

            # P_left(u) - no y dependence (x=0)
            P_left_val = P_left.eval_deriv(np.array([u]), 0)[0]

            # P_right(Σyj + u)
            P_right_lin = {f"y{j+1}": 1.0 for j in range(l2)}
            P_right_series = compose_polynomial_on_affine(
                P_right, u0=u, lin=P_right_lin, var_names=var_names
            )

            # Arguments with x=0:
            # Arg₁ = t(1 + θY) where Y = Σyj
            # Arg₂ = t + θY(t-1)
            arg1_lin = {f"y{j+1}": theta * t for j in range(l2)}
            arg2_lin = {f"y{j+1}": theta * (t - 1) for j in range(l2)}

            exp1 = compose_exp_on_affine(R, t, arg1_lin, var_names)
            exp2 = compose_exp_on_affine(R, t, arg2_lin, var_names)
            Q1 = compose_polynomial_on_affine(Q, t, arg1_lin, var_names)
            Q2 = compose_polynomial_on_affine(Q, t, arg2_lin, var_names)

            product = P_right_series * exp1 * exp2 * Q1 * Q2 * (poly_pref * P_left_val)

            # Extract full y-derivative coefficient
            full_coeff = product.extract(var_names)

            # PRZZ structure: -[(1+θY)/θ]|_{Y=0} × derivative = -(1/θ) × derivative
            contrib = (1.0 / theta) * full_coeff
            integral += w * contrib

    return -integral


def compute_pair_paper(
    P_left, P_right, Q,
    l1: int, l2: int,
    theta: float,
    R: float,
    n_quadrature: int = 60,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute all terms for (l1, l2) pair using paper equations.
    """
    I1 = _compute_I1_pair_paper(P_left, P_right, Q, l1, l2, theta, R, n_quadrature)
    I2 = _compute_I2_pair_paper(P_left, P_right, Q, l1, l2, theta, R, n_quadrature)
    I3 = _compute_I3_pair_paper(P_left, P_right, Q, l1, l2, theta, R, n_quadrature)
    I4 = _compute_I4_pair_paper(P_left, P_right, Q, l1, l2, theta, R, n_quadrature)

    total = I1 + I2 + I3 + I4

    pair_key = f"{l1}{l2}"
    if verbose:
        print(f"Paper ({l1},{l2}): I1={I1:.10e}, I2={I2:.10e}, I3={I3:.10e}, I4={I4:.10e}")
        print(f"Paper ({l1},{l2}) total: {total:.10e}")

    return {
        f"I1_{pair_key}": I1,
        f"I2_{pair_key}": I2,
        f"I3_{pair_key}": I3,
        f"I4_{pair_key}": I4,
        f"total_{pair_key}": total
    }


def compute_c_paper_k3(
    polynomials,
    theta: float,
    R: float,
    n_quadrature: int = 60,
    verbose: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Compute full c for K=3 using paper equations (no I5).

    Returns (c_total, breakdown_dict).
    """
    # API normalization
    if isinstance(polynomials, tuple) and len(polynomials) >= 4:
        P1, P2, P3, Q = polynomials[0], polynomials[1], polynomials[2], polynomials[3]
    elif isinstance(polynomials, dict):
        P1, P2, P3 = polynomials['P1'], polynomials['P2'], polynomials['P3']
        Q = polynomials['Q']
    else:
        raise TypeError(f"polynomials must be dict or tuple, got {type(polynomials)}")

    P_map = {1: P1, 2: P2, 3: P3}

    # Pairs: (l1, l2, symmetry_factor)
    pairs = [
        (1, 1, 1),
        (2, 2, 1),
        (3, 3, 1),
        (1, 2, 2),  # Off-diagonal counted twice
        (1, 3, 2),
        (2, 3, 2),
    ]

    # Factorial normalization
    def factorial_norm(l1, l2):
        return 1.0 / (factorial(l1) * factorial(l2))

    c_total = 0.0
    breakdown = {}

    for l1, l2, sym in pairs:
        P_left = P_map[l1]
        P_right = P_map[l2]

        result = compute_pair_paper(P_left, P_right, Q, l1, l2, theta, R, n_quadrature, verbose)

        pair_key = f"{l1}{l2}"
        raw_total = result[f"total_{pair_key}"]

        # Apply normalization
        normalized = sym * factorial_norm(l1, l2) * raw_total

        breakdown[f"c_{pair_key}"] = normalized
        c_total += normalized

        if verbose:
            print(f"c_{pair_key}: raw={raw_total:.10e}, norm={normalized:.10e}")

    if verbose:
        print(f"c_total (paper, no I5): {c_total:.10e}")

    return c_total, breakdown
