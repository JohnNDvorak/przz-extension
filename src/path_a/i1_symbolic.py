#!/usr/bin/env python3
"""
Step 3a: Symbolic I₁ in z-Basis

Computes I₁ for all pairs (ℓ₁, ℓ₂) symbolically and expresses in z-basis.

CRITICAL: Using z = e^{R/7} basis (NOT y = e^{2R/7})
  - Reason: Mirror multiplier M₀ = e^R + 5 requires e^R = z⁷ (integer power)
  - The old y = e^{2R/7} gives e^R = y^{7/2} (fractional - FORBIDDEN!)
  - Relationship: y = z², so existing work maps directly

z-Power Structure:
  z⁰  = 1
  z⁴  = exp(4R/7)  [was y²]
  z⁷  = exp(R)     [NEW - from mirror M₀]
  z⁸  = exp(8R/7)  [was y⁴]
  z¹⁴ = exp(2R)    [was y⁷]
  z¹⁸ = exp(18R/7) [was y⁹]
  z²² = exp(22R/7) [was y¹¹]

From PRZZ TeX lines 1530-1532:
    I₁ = (d²/dxdy)|_{x=y=0} × [(θ(x+y)+1)/θ]
         × ∫₀¹ ∫₀¹ (1-u)^{ℓ₁+ℓ₂} P_{ℓ₁}(x+u) P_{ℓ₂}(y+u)
         × exp(R[A_α + A_β]) × Q(A_α) × Q(A_β) du dt

Where:
    A_α = t + θt·x + θ(t-1)·y
    A_β = t + θ(t-1)·x + θt·y

Key insight: Since x², y² vanish, we only need first-order Taylor expansions:
    P(x+u) = P(u) + P'(u)·x
    Q(A_α) = Q(t) + Q'(t)·[θt·x + θ(t-1)·y]
    exp(R(A_α + A_β)) = exp(2Rt)·[1 + Rθ(2t-1)·(x+y) + R²θ²(2t-1)²·xy]

The d²/dxdy derivative extracts the xy coefficient.

Usage:
    python -m src.path_a.i1_symbolic
"""
import sympy as sp
from sympy import (
    Rational, symbols, exp, simplify, expand, together,
    fraction, factorial, N, integrate, Poly, diff
)
from typing import Dict, Tuple, Optional

from src.path_a.j_integral import J_n_closed_form
from src.path_a.optimal_coeffs import (
    R, theta, R_star_approx,
    build_P1, build_P2, build_P3, build_Q
)

# Symbols
u = symbols('u', real=True, positive=True)
t = symbols('t', real=True, positive=True)
x_sym = symbols('x', real=True)
y_sym = symbols('y', real=True)
z_basis = symbols('z', positive=True)  # z = e^{R/7} (integer powers for all exponentials)


def get_polynomial_expr(ell: int, var) -> sp.Expr:
    """Get P_ℓ(var) as a symbolic expression."""
    if ell == 1:
        return build_P1(var)
    elif ell == 2:
        return build_P2(var)
    elif ell == 3:
        return build_P3(var)
    else:
        raise ValueError(f"Invalid piece index: {ell}")


def compute_I1_xy_coefficient(
    ell1: int,
    ell2: int,
    verbose: bool = True
) -> sp.Expr:
    """
    Compute the xy coefficient of the I₁ integrand symbolically.

    The integrand is:
        [(θ(x+y)+1)/θ] × (1-u)^{ℓ₁+ℓ₂} × P_{ℓ₁}(x+u) × P_{ℓ₂}(y+u)
        × exp(R[A_α + A_β]) × Q(A_α) × Q(A_β)

    We expand to first order in x, y and extract the xy coefficient.

    Returns:
        Symbolic expression for the xy coefficient (function of u, t, R)
    """
    if verbose:
        print(f"Computing I₁ xy-coefficient for pair ({ell1},{ell2})...")

    # Build polynomials
    P1_u = get_polynomial_expr(ell1, u)
    P2_u = get_polynomial_expr(ell2, u)
    P1_deriv = diff(P1_u, u)
    P2_deriv = diff(P2_u, u)

    Q_t = build_Q(t)
    Q_deriv = diff(Q_t, t)

    # Affine eigenvalue coefficients
    # A_α = t + θt·x + θ(t-1)·y → x_coeff = θt, y_coeff = θ(t-1)
    # A_β = t + θ(t-1)·x + θt·y → x_coeff = θ(t-1), y_coeff = θt
    A_alpha_x = theta * t
    A_alpha_y = theta * (t - 1)
    A_beta_x = theta * (t - 1)
    A_beta_y = theta * t

    # Exponential factor: exp(R(A_α + A_β)) = exp(2Rt)·exp(Rθ(2t-1)(x+y))
    # Taylor: ≈ exp(2Rt)·[1 + Rθ(2t-1)x + Rθ(2t-1)y + R²θ²(2t-1)²xy + ...]
    exp_0 = exp(2 * R * t)
    exp_x_coeff = R * theta * (2*t - 1)
    exp_y_coeff = R * theta * (2*t - 1)
    exp_xy_coeff = R**2 * theta**2 * (2*t - 1)**2

    # Prefactor: (θ(x+y)+1)/θ = 1/θ + x + y
    # This is: f_0 = 1/θ, f_x = 1, f_y = 1
    pref_0 = 1 / theta
    pref_x = sp.Integer(1)
    pref_y = sp.Integer(1)

    # (1-u)^{ℓ₁+ℓ₂} factor
    one_minus_u_power = (1 - u) ** (ell1 + ell2)

    # P expansions:
    # P₁(x+u) = P₁(u) + P₁'(u)·x
    # P₂(y+u) = P₂(u) + P₂'(u)·y
    # P product: P₁(x+u)·P₂(y+u) = P₁P₂ + P₁'P₂·x + P₁P₂'·y + P₁'P₂'·xy
    P_0 = P1_u * P2_u                    # constant term
    P_x = P1_deriv * P2_u                # x coefficient
    P_y = P1_u * P2_deriv                # y coefficient
    P_xy = P1_deriv * P2_deriv           # xy coefficient

    # Q expansions:
    # Q(A_α) = Q(t) + Q'(t)·(A_α - t) = Q(t) + Q'(t)·[θt·x + θ(t-1)·y]
    # Q(A_β) = Q(t) + Q'(t)·[θ(t-1)·x + θt·y]
    # Q product: Q(A_α)·Q(A_β) = Q² + Q·Q'·(A_α-t + A_β-t) + Q'²·(A_α-t)(A_β-t) + ...
    Q_0 = Q_t * Q_t                      # Q(t)²
    Q_alpha_shift_x = Q_deriv * A_alpha_x  # Q'(t)·θt
    Q_alpha_shift_y = Q_deriv * A_alpha_y  # Q'(t)·θ(t-1)
    Q_beta_shift_x = Q_deriv * A_beta_x    # Q'(t)·θ(t-1)
    Q_beta_shift_y = Q_deriv * A_beta_y    # Q'(t)·θt

    # Q product to first order:
    # Q(A_α)·Q(A_β) ≈ Q² + Q·Q'·[shifts] + Q'²·[cross terms at xy level]
    # = Q² + Q·Q'·[(θt + θ(t-1))x + (θ(t-1) + θt)y] + O(xy from Q'²)
    # Coefficients:
    Q_x = Q_t * (Q_alpha_shift_x + Q_beta_shift_x)  # Q·Q'·[θt + θ(t-1)]·x = Q·Q'·θ(2t-1)·x
    Q_y = Q_t * (Q_alpha_shift_y + Q_beta_shift_y)  # Q·Q'·[θ(t-1) + θt]·y = Q·Q'·θ(2t-1)·y

    # Q xy coefficient from Q'² terms:
    # (A_α - t)(A_β - t) = [θt·x + θ(t-1)·y][θ(t-1)·x + θt·y]
    # xy coefficient: θt·θt + θ(t-1)·θ(t-1) = θ²[t² + (t-1)²]
    Q_prime_sq = Q_deriv * Q_deriv
    QQ_xy_from_shifts = Q_prime_sq * theta**2 * (t**2 + (t-1)**2)

    # Also need Q·Q' × linear × linear contributing to xy
    # Q·Q'·shift_α × Q·Q'·shift_β at xy level is O(x²y) or O(xy²), vanishes
    # The Q xy term is Q_0 at xy level = 0 (Q² has no x,y), plus QQ_xy_from_shifts
    Q_xy = QQ_xy_from_shifts

    # Now combine all factors to get xy coefficient:
    # Integrand = pref × (1-u)^n × P × exp × Q
    #
    # We need xy coefficient of the product:
    #   (pref_0 + pref_x·x + pref_y·y) × (1-u)^n × (P_0 + P_x·x + P_y·y + P_xy·xy)
    #   × exp_0·(1 + exp_x·x + exp_y·y + exp_xy·xy)
    #   × (Q_0 + Q_x·x + Q_y·y + Q_xy·xy)

    # Collect xy contributions from all combinations:
    # xy appears from: const × const × const × xy, or
    #                  const × const × xy × const, or
    #                  const × x × const × y, or etc.

    # Let's denote: A = pref, B = P, C = exp, D = Q
    # A = A_0 + A_x·x + A_y·y
    # B = B_0 + B_x·x + B_y·y + B_xy·xy
    # C = C_0 + C_x·x + C_y·y + C_xy·xy  (actually C_0 = exp_0, others are relative)
    # D = D_0 + D_x·x + D_y·y + D_xy·xy

    # The product ABCD has xy coefficient:
    # A_0 × B_0 × C_0 × D_xy + A_0 × B_0 × C_xy × D_0 + A_0 × B_xy × C_0 × D_0
    # + A_0 × B_x × C_0 × D_y + A_0 × B_x × C_y × D_0 + A_0 × B_y × C_0 × D_x + A_0 × B_y × C_x × D_0
    # + A_0 × B_0 × C_x × D_y + A_0 × B_0 × C_y × D_x
    # + A_x × B_0 × C_0 × D_y + A_x × B_0 × C_y × D_0 + A_x × B_y × C_0 × D_0
    # + A_y × B_0 × C_0 × D_x + A_y × B_0 × C_x × D_0 + A_y × B_x × C_0 × D_0
    # + A_x × B_y × C_0 × D_0 + A_y × B_x × C_0 × D_0 (= 2·A_x·B_x·C_0·D_0 if A_x=A_y, B_x≠B_y)

    # Actually, since A_x = A_y = 1 (pref), this simplifies

    A_0 = pref_0
    A_x = pref_x
    A_y = pref_y
    B_0, B_x, B_y, B_xy = P_0, P_x, P_y, P_xy
    C_0 = exp_0
    C_x, C_y, C_xy = exp_x_coeff, exp_y_coeff, exp_xy_coeff
    D_0, D_x, D_y, D_xy = Q_0, Q_x, Q_y, Q_xy

    # Compute xy coefficient of ABCD
    xy_coeff = sp.Integer(0)

    # Terms with xy from individual factors:
    xy_coeff += A_0 * B_xy * C_0 * D_0  # B has xy
    xy_coeff += A_0 * B_0 * C_xy * C_0 * D_0  # Wait, C_xy is multiplicative!

    # Actually, C is exp_0 × (1 + C_x/C_0 × x + ...), so C_0 factor is common
    # Let me redo: C_full = C_0 × (1 + c_x·x + c_y·y + c_xy·xy) where c_k = C_k/C_0
    # But C_x, C_y, C_xy as I defined them are the linear coefficients in the expansion
    # C_full = C_0 × [1 + (C_x/C_0)·x + (C_y/C_0)·y + (C_xy/C_0)·xy]
    # This is getting confusing. Let me define normalized coefficients:

    c_x = exp_x_coeff  # coefficient of x in exp(...) expansion (before multiplying by exp_0)
    c_y = exp_y_coeff
    c_xy = exp_xy_coeff

    # The full expression for exp(R(A_α+A_β)) = exp(2Rt) × [1 + c_x·x + c_y·y + c_xy·xy + ...]
    # So in the product, we have exp_0 × (1 + c_x·x + c_y·y + c_xy·xy)

    # Similarly for Q product: it's Q_0 × [1 + (Q_x/Q_0)x + (Q_y/Q_0)y + (Q_xy/Q_0)xy]
    # Let's define: q_x = Q_x/Q_0, q_y = Q_y/Q_0, q_xy = Q_xy/Q_0

    # For the prefactor: A = A_0 × [1 + (A_x/A_0)x + (A_y/A_0)y] = (1/θ) × [1 + θx + θy]
    # So a_x = A_x/A_0 = θ, a_y = θ

    # For P: not normalized, it's B_0 + B_x·x + B_y·y + B_xy·xy

    # Let me directly compute the xy coefficient by expanding symbolically:

    # The integrand is:
    # (1/θ + x + y) × (1-u)^n × (P₁P₂ + P₁'P₂·x + P₁P₂'·y + P₁'P₂'·xy)
    # × exp(2Rt)·(1 + c_x·x + c_y·y + c_xy·xy)
    # × (Q² + Q_x·x + Q_y·y + Q_xy·xy)

    # Group constant factor: (1-u)^n × exp(2Rt)
    const_factor = one_minus_u_power * exp_0

    # Now compute xy coefficient of:
    # (1/θ + x + y) × (P₁P₂ + P₁'P₂·x + P₁P₂'·y + P₁'P₂'·xy)
    # × (1 + c_x·x + c_y·y + c_xy·xy)
    # × (Q² + Q_x·x + Q_y·y + Q_xy·xy)

    # Let's denote:
    # F1 = (1/θ + x + y)
    # F2 = (P₁P₂ + P₁'P₂·x + P₁P₂'·y + P₁'P₂'·xy)
    # F3 = (1 + c_x·x + c_y·y + c_xy·xy)
    # F4 = (Q² + Q_x·x + Q_y·y + Q_xy·xy)

    # [F1 × F2 × F3 × F4]_xy = ?

    # This requires systematic enumeration. Let me use a different approach:
    # Create the full product symbolically with x_sym, y_sym and then collect.

    # Full factors with symbolic x, y:
    F1 = 1/theta + x_sym + y_sym
    F2 = P_0 + P_x * x_sym + P_y * y_sym + P_xy * x_sym * y_sym
    F3 = 1 + c_x * x_sym + c_y * y_sym + c_xy * x_sym * y_sym
    F4 = Q_0 + Q_x * x_sym + Q_y * y_sym + Q_xy * x_sym * y_sym

    product = expand(F1 * F2 * F3 * F4)

    # Extract xy coefficient
    xy_coeff = product.coeff(x_sym).coeff(y_sym)

    # Multiply by constant factor
    I1_integrand = const_factor * xy_coeff

    if verbose:
        print(f"  Integrand computed (length: {len(str(I1_integrand))} chars)")

    return I1_integrand


def compute_I1_symbolic(
    ell1: int,
    ell2: int,
    verbose: bool = True
) -> Tuple[sp.Expr, Dict, sp.Expr]:
    """
    Compute I₁ for a pair (ℓ₁, ℓ₂) symbolically.

    Integrates the xy coefficient over (u, t) ∈ [0,1]².

    Returns:
        (I1_expr, z_coeffs, denominator) where
        I1 = (Σ_k A_k(R)·z^k) / D(R)
        and z = e^{R/7}
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"I₁ Symbolic for pair ({ell1},{ell2})")
        print("=" * 60)

    # Get the xy coefficient of the integrand
    xy_coeff = compute_I1_xy_coefficient(ell1, ell2, verbose=verbose)

    # Integrate over u ∈ [0,1]
    if verbose:
        print("  Integrating over u...")
    I1_after_u = integrate(xy_coeff, (u, 0, 1))
    I1_after_u = simplify(I1_after_u)

    if verbose:
        print(f"  After u-integration: {len(str(I1_after_u))} chars")

    # Integrate over t ∈ [0,1]
    if verbose:
        print("  Integrating over t...")
    I1_expr = integrate(I1_after_u, (t, 0, 1))
    I1_expr = simplify(I1_expr)

    if verbose:
        print(f"  After t-integration: {len(str(I1_expr))} chars")

    # Extract z-basis coefficients (z = e^{R/7})
    z_coeffs, denominator = extract_z_basis_coefficients(I1_expr, verbose=verbose)

    return I1_expr, z_coeffs, denominator


def extract_z_basis_coefficients(
    expr: sp.Expr,
    verbose: bool = True
) -> Tuple[Optional[Dict], sp.Expr]:
    """
    Extract coefficients in z-basis where z = e^{R/7}.

    CRITICAL CORRECTION (2026-01-03):
    - I₁ z-powers can range from -8 to +38 (depending on pair derivatives)
    - I₁ contributes 94% of signal (highest priority!)
    - Must handle NEGATIVE z-powers

    z-Power Mapping:
        z⁻⁸ = exp(-8R/7)
        z⁻⁴ = exp(-4R/7)
        z⁰  = 1
        z⁴  = exp(4R/7)
        z⁷  = exp(R)      (for mirror M₀)
        z⁸  = exp(8R/7)
        z¹⁴ = exp(2R)
        z¹⁸ = exp(18R/7)
        z²² = exp(22R/7)
        ...up to z³⁸

    Form: I₁ = (Σ_k A_k(R)·z^k) / D(R)

    Returns (coeffs_dict, denominator) where coeffs_dict = {z_power: coefficient}
    Note: z_power can be NEGATIVE!
    """
    from src.path_a.u_integral_symbolic import _extract_piecewise_main_branch

    # Handle Piecewise
    main_expr = _extract_piecewise_main_branch(expr)

    # Combine into single fraction
    main_expr = together(main_expr)
    num, den = fraction(main_expr)

    # Expand numerator
    num = expand(num)

    z = symbols('z', positive=True)

    # Comprehensive substitution for all z-powers from -8 to +40
    # z^k = exp(kR/7)
    num_sub = num
    for k in range(-2, 11):  # k from -2 to 10 → z^{-8} to z^{40}
        for mult in [4, 7, 1]:  # Handle z^{4k}, z^{7}, z^{k}
            if mult == 4:
                z_power = 4 * k
                exp_val = 4 * k * R / 7
            elif mult == 7 and k >= 0:
                z_power = 7 * (k + 1)
                exp_val = (k + 1) * R
            else:
                continue
            if -10 <= z_power <= 40:
                num_sub = num_sub.subs(exp(exp_val), z**z_power)
                num_sub = num_sub.subs(exp(Rational(4 * k, 7) * R), z**(4*k))

    # Also substitute specific key exponentials
    num_sub = num_sub.subs(exp(2*R), z**14)
    num_sub = num_sub.subs(exp(R), z**7)

    try:
        if num_sub.has(z):
            num_sub = expand(num_sub)
            coeffs = {}

            # Extract coefficients for extended z-power range: -8 to +40
            for p in range(-8, 41):
                if p == 0:
                    coeff = num_sub.subs(z, 0)
                else:
                    coeff = num_sub.coeff(z, p)
                if coeff != 0:
                    coeffs[p] = coeff

            if verbose:
                powers_present = sorted(coeffs.keys())
                print(f"  z-basis: powers present = {powers_present}")

            return coeffs, den
        else:
            # No exp terms, pure polynomial in R
            return {0: num}, den

    except Exception as e:
        if verbose:
            print(f"  Could not extract z-basis: {e}")

    return None, den


def compute_I1_all_pairs(verbose: bool = True) -> Dict:
    """Compute I₁ symbolically for all 6 pairs in z-basis (z = e^{R/7})."""
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    results = {}

    for ell1, ell2 in pairs:
        I1_expr, z_coeffs, denominator = compute_I1_symbolic(ell1, ell2, verbose=verbose)

        # Numeric value at R*
        from src.path_a.u_integral_symbolic import _extract_piecewise_main_branch
        main_expr = _extract_piecewise_main_branch(I1_expr)
        I1_numeric = float(N(main_expr.subs(R, R_star_approx), 20))

        results[(ell1, ell2)] = {
            'I1_expr': I1_expr,
            'z_coeffs': z_coeffs,
            'denominator': denominator,
            'I1_numeric': I1_numeric,
        }

        if verbose:
            print(f"\n  I₁(R*) = {I1_numeric:.10f}")

    return results


def validate_against_numeric(results: Dict, verbose: bool = True) -> None:
    """Validate symbolic I₁ against numeric przz_exact_i1 implementation.

    Uses the SAME optimal polynomials from optimal_coeffs.py for both
    symbolic and numeric computations.
    """
    from src.przz_exact_i1 import compute_I1_all_pairs as compute_I1_numeric
    import numpy as np
    from sympy import N as sympy_N, Float, diff, symbols

    # Create polynomial wrappers using optimal_coeffs (same as symbolic)
    class SymbolicPolyWrapper:
        def __init__(self, build_fn):
            self.build_fn = build_fn

        def eval(self, x_arr):
            result = []
            for x in x_arr:
                val = float(sympy_N(self.build_fn(Float(x)), 20))
                result.append(val)
            return np.array(result)

        def eval_deriv(self, x_arr, k=1):
            u_sym = symbols('u')
            expr = self.build_fn(u_sym)
            deriv_expr = expr
            for _ in range(k):
                deriv_expr = diff(deriv_expr, u_sym)
            result = []
            for x in x_arr:
                val = float(sympy_N(deriv_expr.subs(u_sym, Float(x)), 20))
                result.append(val)
            return np.array(result)

    # Use optimal polynomials for numeric (same as symbolic)
    polynomials = {
        "P1": SymbolicPolyWrapper(build_P1),
        "P2": SymbolicPolyWrapper(build_P2),
        "P3": SymbolicPolyWrapper(build_P3),
        "Q": SymbolicPolyWrapper(build_Q),
    }

    theta_val = float(theta)

    # Compute numeric I₁ at R*
    numeric_results = compute_I1_numeric(theta_val, R_star_approx, polynomials, n_quad=80)

    if verbose:
        print("\n" + "=" * 60)
        print("VALIDATION: Symbolic vs Numeric I₁")
        print("=" * 60)
        print("\n| Pair | I₁_symbolic | I₁_numeric | Ratio |")
        print("|------|-------------|------------|-------|")

    for (ell1, ell2), r in results.items():
        I1_sym = r['I1_numeric']
        key = f"{ell1}{ell2}"
        I1_num = numeric_results[key].value

        ratio = I1_sym / I1_num if abs(I1_num) > 1e-15 else float('inf')

        if verbose:
            print(f"| ({ell1},{ell2}) | {I1_sym:+.8f} | {I1_num:+.8f} | {ratio:.6f} |")


def main():
    print("=" * 70)
    print("STEP 3a: SYMBOLIC I₁ IN z-BASIS (z = e^{R/7})")
    print("=" * 70)
    print(f"\nθ = {theta} = {float(theta):.10f}")
    print(f"R* ≈ {R_star_approx}")
    print("\nz-Power Mapping:")
    print("  z⁰ = 1, z⁴ = e^{4R/7}, z⁷ = e^R (mirror), z⁸ = e^{8R/7}, z¹⁴ = e^{2R}")

    # Compute I₁ for all pairs
    results = compute_I1_all_pairs(verbose=True)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: I₁ z-BASIS DECOMPOSITION")
    print("=" * 60)

    print("\n| Pair | z powers | I₁(R*) |")
    print("|------|----------|--------|")
    for (ell1, ell2), r in results.items():
        z_coeffs = r.get('z_coeffs')
        if z_coeffs:
            z_powers = sorted(z_coeffs.keys())
            z_str = ",".join(str(p) for p in z_powers)
        else:
            z_str = "?"
        I1_val = r['I1_numeric']
        print(f"| ({ell1},{ell2}) | {z_str:^8} | {I1_val:+.8f} |")

    # Validate against numeric
    validate_against_numeric(results, verbose=True)

    print("\n" + "=" * 70)
    print("STEP 3a COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
