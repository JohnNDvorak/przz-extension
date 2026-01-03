#!/usr/bin/env python3
"""
Step 1: Symbolic u-Integral in z-Basis

Computes U(R) = ∫₀¹ K_{ω₁}(u) K_{ω₂}(u) du symbolically
and expresses the result in the z-basis where z = e^{R/7}.

CRITICAL: Using z = e^{R/7} basis (NOT y = e^{2R/7})
  - Reason: Mirror multiplier M₀ = e^R + 5 requires e^R = z⁷ (integer power)
  - Old y-basis had e^R = y^{7/2} (fractional - FORBIDDEN!)
  - Relationship: y = z², so existing work maps directly

Key insight:
- After integrating over u ∈ [0,1], the exp(Rθu) = exp(4Ru/7) terms
  get evaluated at the boundary, giving exp(4R/7) = z⁴ (was y² in old basis)
- The result has form: U(R) = (A(R)·z⁴ + B(R)) / R^p

z-Power Mapping for U integrals:
  z⁰  = 1           (constant)
  z⁴  = exp(4R/7)   (was y² = exp(Rθ))
  z⁸  = exp(8R/7)   (was y⁴ = exp(2Rθ))

For each pair (ℓ₁, ℓ₂):
- ω₁ = ℓ₁ - 1, ω₂ = ℓ₂ - 1
- If ω = 0: K_ω(u) = P_ℓ(u) (raw polynomial)
- If ω ≥ 1: K_ω(u) involves J_n(Rθu) integrals

Usage:
    python -m src.path_a.u_integral_symbolic
"""
import sympy as sp
from sympy import (
    Rational, symbols, exp, simplify, expand, together,
    fraction, factorial, binomial, N, integrate, Poly, collect
)
from typing import Dict, Tuple, List, Optional

from src.path_a.j_integral import J_n_closed_form
from src.path_a.optimal_coeffs import (
    R, theta, R_star_approx,
    build_P1, build_P2, build_P3
)

# Symbols
u = symbols('u', real=True, positive=True)
z = symbols('z', positive=True)  # z = e^{R/7} (integer powers for all exponentials)


def get_polynomial_expr(ell: int, u_sym) -> sp.Expr:
    """Get P_ℓ(u) as a symbolic expression."""
    if ell == 1:
        return build_P1(u_sym)
    elif ell == 2:
        return build_P2(u_sym)
    elif ell == 3:
        return build_P3(u_sym)
    else:
        raise ValueError(f"Invalid piece index: {ell}")


def compute_U_raw(ell1: int, ell2: int) -> Tuple[sp.Expr, str]:
    """
    Compute U(R) for a pair where both pieces use raw polynomials (ω = 0).

    This applies to pair (1,1) only.

    Returns:
        (U_value, description)
    """
    P1_u = get_polynomial_expr(ell1, u)
    P2_u = get_polynomial_expr(ell2, u)

    integrand = P1_u * P2_u
    U = integrate(expand(integrand), (u, 0, 1))
    U = simplify(U)

    return U, f"∫₀¹ P_{ell1}(u)·P_{ell2}(u) du = {U}"


def compute_K_omega_expr(
    ell: int,
    u_sym,
    R_sym,
    theta_val
) -> sp.Expr:
    """
    Compute K_ω(u) symbolically for piece ℓ.

    K_ω(u) = u^ω/(ω-1)! × Σ_k c_k u^k × Σ_j C(k,j)(-1)^j × J_{ω-1+j}(Rθu)

    The result contains exp(Rθu) terms via J_n.
    """
    from src.path_a.case_c_symbolic import get_polynomial_standard_coeffs

    omega = ell - 1

    if omega == 0:
        # Case B: raw polynomial
        return get_polynomial_expr(ell, u_sym)

    # Case C: kernel via J_n integrals
    poly_coeffs = get_polynomial_standard_coeffs(ell)
    lam = R_sym * theta_val * u_sym  # Rθu = 4Ru/7

    result = sp.Integer(0)

    for k, c_k in enumerate(poly_coeffs):
        if c_k == 0:
            continue

        # Binomial expansion of (1-a)^k
        binomial_sum = sp.Integer(0)
        for j in range(k + 1):
            coeff = ((-1)**j) * binomial(k, j)
            J_val = J_n_closed_form(omega - 1 + j, lam)
            binomial_sum += coeff * J_val

        result += c_k * (u_sym**k) * binomial_sum

    # Multiply by u^ω / (ω-1)!
    result = result * (u_sym**omega) / factorial(omega - 1)

    return result


def compute_U_case_c_single(
    ell1: int,
    ell2: int,
    verbose: bool = True
) -> Tuple[sp.Expr, Optional[Dict], sp.Expr]:
    """
    Compute U(R) for a pair involving Case C kernels.

    Returns:
        (U_expr, z_coeffs, denominator) where
        U = (Σ_k A_k(R)·z^k) / D(R)
        z_coeffs = {0: A₀, 4: A₄, 8: A₈, ...}
        and z = e^{R/7}

    z-Power Mapping (from old y-basis):
        z⁰ = y⁰ (constant)
        z⁴ = y² = exp(4R/7)
        z⁸ = y⁴ = exp(8R/7)
    """
    omega1 = ell1 - 1
    omega2 = ell2 - 1

    if verbose:
        print(f"Computing U for pair ({ell1},{ell2}) with ω₁={omega1}, ω₂={omega2}...")

    if omega1 == 0 and omega2 == 0:
        # Both raw polynomials - no exp terms
        U, _ = compute_U_raw(ell1, ell2)
        return U, {0: U}, sp.Integer(1)

    # Get K_ω expressions
    K1 = compute_K_omega_expr(ell1, u, R, theta)
    K2 = compute_K_omega_expr(ell2, u, R, theta)

    if verbose:
        print("  K₁ and K₂ computed")

    # Product K₁ × K₂
    product = K1 * K2
    product = together(product)

    if verbose:
        print("  Product formed, integrating over u...")

    # Symbolic integration
    U_expr = integrate(product, (u, 0, 1))

    if verbose:
        print("  Integration complete, extracting z-basis...")

    # Extract z-basis coefficients (z = e^{R/7})
    # Pass pair indices for pair-dependent z-power computation
    z_coeffs, denominator = extract_z_basis_coefficients(U_expr, ell1=ell1, ell2=ell2, verbose=verbose)

    return U_expr, z_coeffs, denominator


def _extract_piecewise_main_branch(expr: sp.Expr) -> sp.Expr:
    """Recursively find and substitute Piecewise with its main branch (for R ≠ 0)."""
    from sympy import Piecewise

    if isinstance(expr, Piecewise):
        # Return the expression from the first branch (typically for R != 0)
        return expr.args[0][0]

    if expr.is_Atom:
        return expr

    # Recursively handle arguments
    new_args = [_extract_piecewise_main_branch(arg) for arg in expr.args]
    return expr.func(*new_args)


def extract_z_basis_coefficients(
    expr: sp.Expr,
    ell1: int = None,
    ell2: int = None,
    verbose: bool = True
) -> Tuple[Optional[Dict], sp.Expr]:
    """
    Extract coefficients in z-basis where z = e^{R/7}.

    CRITICAL CORRECTION: U integrals have PAIR-DEPENDENT z-powers!

    The U integral formula is:
        U^{(ω₁,ω₂)} = exp(Rθ(ω+2)) × [1 - exp(-4Rθ)] / (4Rθ)
        where ω = ω₁ + ω₂ = (ℓ₁-1) + (ℓ₂-1)

    This expands to (after multiplying by 16R/7):
        U × (16R/7) = 4×z^{4(ω+2)} - 4×z^{4(ω-2)}
                    = 4×z^{4ω+8} - 4×z^{4ω-8}

    z-Powers by Pair:
        (1,1): ω=0 → z⁸, z⁻⁸  (NEGATIVE!)
        (1,2): ω=1 → z¹², z⁻⁴  (NEGATIVE!)
        (1,3): ω=2 → z¹⁶, z⁰
        (2,2): ω=2 → z¹⁶, z⁰
        (2,3): ω=3 → z²⁰, z⁴
        (3,3): ω=4 → z²⁴, z⁸

    Returns (coeffs_dict, denominator) where coeffs_dict = {z_power: coefficient}
    Note: z_power can be NEGATIVE!
    """
    # Recursively extract main branch from any Piecewise (handles nested cases)
    main_expr = _extract_piecewise_main_branch(expr)

    # Combine into single fraction
    main_expr = together(main_expr)
    num, den = fraction(main_expr)

    # Expand numerator
    num = expand(num)

    # Substitution for z-basis: z = exp(R/7)
    # Need to handle a wide range of exponents including negative!
    z_sym = symbols('z', positive=True)

    # Compute expected z-powers based on pair
    if ell1 is not None and ell2 is not None:
        omega = (ell1 - 1) + (ell2 - 1)
        expected_high = 4 * omega + 8   # z^{4ω+8}
        expected_low = 4 * omega - 8    # z^{4ω-8}
        if verbose:
            print(f"  Pair ({ell1},{ell2}): ω={omega}, expected z-powers = {{{expected_high}, {expected_low}}}")
    else:
        expected_high, expected_low = None, None

    # Substitute various exponential forms
    # Build comprehensive list of z-powers: from -8 to +24 (covering all pairs)
    num_sub = num
    for k in range(-2, 7):  # k from -2 to 6 covers z^{-8} to z^{24}
        exp_val = 4 * k * R / 7
        z_power = 4 * k
        num_sub = num_sub.subs(exp(exp_val), z_sym**z_power)
        # Also try with Rational
        num_sub = num_sub.subs(exp(Rational(4 * k, 7) * R), z_sym**z_power)

    try:
        if num_sub.has(z_sym):
            # Expand and collect coefficients of z
            num_sub = expand(num_sub)

            # Extract coefficients for all z-powers from -8 to +24
            coeffs = {}

            for z_power in range(-8, 25):
                if z_power == 0:
                    coeff = num_sub.subs(z_sym, 0)
                else:
                    coeff = num_sub.coeff(z_sym, z_power)
                if coeff != 0:
                    coeffs[z_power] = coeff

            if verbose:
                powers_present = sorted(coeffs.keys())
                print(f"  z-basis: powers present = {powers_present}")
                for zp in powers_present:
                    print(f"    z^{zp}: {len(str(coeffs[zp]))} chars")

            return coeffs, den
        else:
            # No exp term, pure polynomial
            return {0: num}, den

    except Exception as e:
        if verbose:
            print(f"  Could not extract z-basis: {e}")

    return None, den


# Keep old function name as alias for backwards compatibility
def extract_y_basis_coefficients(expr: sp.Expr, verbose: bool = True) -> Tuple[Optional[Dict], sp.Expr]:
    """DEPRECATED: Use extract_z_basis_coefficients instead. This is a compatibility wrapper."""
    z_coeffs, den = extract_z_basis_coefficients(expr, verbose=verbose)
    if z_coeffs is None:
        return None, den
    # Convert z-powers back to y-powers for old code: z^k → y^{k/2}
    y_coeffs = {}
    for z_power, coeff in z_coeffs.items():
        y_power = z_power // 2
        y_coeffs[y_power] = coeff
    return y_coeffs, den


def analyze_u_integral_structure(ell1: int, ell2: int) -> Dict:
    """
    Analyze the structure of U(R) = ∫ K_{ω₁} K_{ω₂} du.

    Determines:
    - Does U contain exp(Rθ) = y² terms?
    - What is the denominator power?
    - What are the polynomial coefficients?
    """
    omega1 = ell1 - 1
    omega2 = ell2 - 1

    result = {
        'ell1': ell1,
        'ell2': ell2,
        'omega1': omega1,
        'omega2': omega2,
    }

    if omega1 == 0 and omega2 == 0:
        # Both raw polynomials
        U, _ = compute_U_raw(ell1, ell2)
        result['has_exp'] = False
        result['U_value'] = U
        result['U_numeric'] = float(U)
        return result

    # Has Case C kernels - compute numerically at R*
    from src.unified_i2_paper import compute_I2_unified_paper
    from src.path_a.unit_test_symbolic import get_optimal_polynomials

    polys = get_optimal_polynomials()
    theta_val = 4/7

    # Get numeric u-integral by extracting from I2 computation
    # I2 = (1/θ) × U × T, so we can get U if we know T

    # Compute T(R*) = t-integral at R*
    from src.path_a.optimal_coeffs import build_Q
    t = symbols('t', real=True)
    Q_t = build_Q(t)
    Q_sq = expand(Q_t**2)

    # T = ∫ exp(2Rt) Q(t)² dt
    T_integrand = exp(2*R*t) * Q_sq
    T_expr = integrate(T_integrand, (t, 0, 1))
    T_numeric = float(N(T_expr.subs(R, R_star_approx), 30))

    # Get I2 from numeric engine
    I2_result = compute_I2_unified_paper(
        R_star_approx, theta_val, ell1, ell2, polys,
        n_quad_u=80, n_quad_t=80, include_Q=True
    )
    I2_numeric = I2_result.I2_value

    # U = I2 × θ / T
    U_numeric = I2_numeric * theta_val / T_numeric

    result['has_exp'] = (omega1 > 0 or omega2 > 0)
    result['U_numeric'] = U_numeric
    result['T_numeric'] = T_numeric
    result['I2_numeric'] = I2_numeric

    return result


def compute_all_u_integrals(verbose: bool = True) -> Dict:
    """Compute U(R) for all 6 pairs."""
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    results = {}

    for ell1, ell2 in pairs:
        if verbose:
            print(f"\n--- Pair ({ell1},{ell2}) ---")

        result = analyze_u_integral_structure(ell1, ell2)
        results[(ell1, ell2)] = result

        if verbose:
            print(f"  ω₁={result['omega1']}, ω₂={result['omega2']}")
            print(f"  Has exp terms: {result['has_exp']}")
            if 'U_value' in result:
                print(f"  U (exact) = {result['U_value']}")
            print(f"  U (numeric at R*) = {result.get('U_numeric', 'N/A'):.10f}")

    return results


def express_in_y_basis(expr: sp.Expr, R_sym=R) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Express expr in the y-basis where y = e^{2R/7}.

    Returns (A, B, D) such that expr = (A·y² + B) / D
    where A, B, D are polynomials in R.
    """
    # Substitute exp(4R/7) = y² and exp(2R) = y⁷
    theta_val = Rational(4, 7)
    exp_Rtheta = exp(R_sym * theta_val)  # exp(4R/7) = y²
    exp_2R = exp(2 * R_sym)  # exp(2R) = y⁷

    # Make substitutions
    expr_sub = expr.subs(exp_2R, y**7)
    expr_sub = expr_sub.subs(exp_Rtheta, y**2)
    expr_sub = expr_sub.subs(exp(R_sym * Rational(4, 7)), y**2)

    # Combine into single fraction
    expr_together = together(expr_sub)
    num, den = fraction(expr_together)

    # Expand numerator and collect by powers of y
    num = expand(num)

    try:
        # Try to express as polynomial in y
        poly_y = Poly(num, y)
        deg = poly_y.degree()

        if deg <= 2:
            # Form: A·y² + B·y + C or simpler
            coeffs = poly_y.all_coeffs()
            while len(coeffs) < 3:
                coeffs = [sp.Integer(0)] + coeffs

            A = coeffs[-3] if len(coeffs) >= 3 else sp.Integer(0)  # y² coeff
            B_lin = coeffs[-2] if len(coeffs) >= 2 else sp.Integer(0)  # y coeff
            C = coeffs[-1] if len(coeffs) >= 1 else sp.Integer(0)  # constant

            return A, B_lin, C, den

    except Exception as e:
        print(f"  Could not express in y-basis: {e}")

    return num, sp.Integer(0), sp.Integer(0), den


def compute_all_symbolic(verbose: bool = True) -> Dict:
    """Compute U(R) symbolically for all 6 pairs in z-basis (z = e^{R/7})."""
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    results = {}

    for ell1, ell2 in pairs:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Pair ({ell1},{ell2})")
            print("=" * 60)

        U_expr, z_coeffs, denominator = compute_U_case_c_single(ell1, ell2, verbose=verbose)

        # Numeric value at R*
        main_expr = _extract_piecewise_main_branch(U_expr)
        U_numeric = float(N(main_expr.subs(R, R_star_approx), 20))

        results[(ell1, ell2)] = {
            'U_expr': U_expr,
            'z_coeffs': z_coeffs,
            'denominator': denominator,
            'U_numeric': U_numeric,
        }

        if verbose:
            print(f"\n  U(R*) = {U_numeric:.10f}")
            print(f"  Denominator: {denominator}")

            if z_coeffs is not None:
                # Verify: U = (Σ A_k·z^k) / D at R*
                # z* = exp(R*/7)
                z_star = float(N(exp(R_star_approx / 7), 20))
                den_val = float(N(denominator.subs(R, R_star_approx), 20))

                U_check = 0
                for z_power, coeff in z_coeffs.items():
                    coeff_val = float(N(coeff.subs(R, R_star_approx), 20))
                    U_check += coeff_val * (z_star ** z_power)
                    if verbose:
                        print(f"    A_{z_power}(R*) = {coeff_val:.6e}")
                U_check /= den_val

                print(f"  z* = {z_star:.10f}")
                print(f"  D(R*) = {den_val:.6e}")
                print(f"  Reconstructed U = {U_check:.10f}")
                diff = abs(U_check - U_numeric)
                print(f"  Verification: {'✓' if diff < 1e-8 else '✗'} (diff = {diff:.2e})")

    return results


def main():
    print("=" * 70)
    print("STEP 1: SYMBOLIC u-INTEGRAL IN z-BASIS (z = e^{R/7})")
    print("=" * 70)
    print(f"\nθ = {theta} = {float(theta):.10f}")
    print(f"R* ≈ {R_star_approx}")
    print("\nz-Power Mapping:")
    print("  z⁰ = 1, z⁴ = e^{4R/7} = e^{Rθ}, z⁷ = e^R (mirror), z⁸ = e^{8R/7}")

    # Compute all u-integrals symbolically
    print("\n" + "=" * 60)
    print("FULL SYMBOLIC COMPUTATION")
    print("=" * 60)

    results = compute_all_symbolic(verbose=True)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: z-BASIS DECOMPOSITION")
    print("=" * 60)
    print(f"\nU(R) = (Σ A_k(R)·z^k) / D(R)  where z = e^{{R/7}}")
    print("Note: z⁴ = exp(Rθ), z⁸ = exp(2Rθ), z⁷ = exp(R)")

    print("\n| Pair | z powers | U(R*) | Verified |")
    print("|------|----------|-------|----------|")
    for (ell1, ell2), r in results.items():
        z_coeffs = r.get('z_coeffs')
        denominator = r.get('denominator')
        if z_coeffs:
            z_powers = sorted(z_coeffs.keys())
            z_str = ",".join(str(p) for p in z_powers)
        else:
            z_str = "?"
        U_val = r['U_numeric']

        # Quick verification
        if z_coeffs and denominator:
            z_star = float(N(exp(R_star_approx / 7), 20))
            den_val = float(N(denominator.subs(R, R_star_approx), 20))
            U_check = sum(
                float(N(c.subs(R, R_star_approx), 20)) * (z_star ** p)
                for p, c in z_coeffs.items()
            ) / den_val
            verified = "✓" if abs(U_check - U_val) < 1e-8 else "✗"
        else:
            verified = "?"

        print(f"| ({ell1},{ell2}) | {z_str:^8} | {U_val:+.8f} | {verified:^8} |")

    # Verify against numeric engine
    print("\n" + "=" * 60)
    print("VALIDATION: Symbolic vs Numeric Engine")
    print("=" * 60)

    from src.unified_i2_paper import compute_I2_unified_paper
    from src.path_a.unit_test_symbolic import get_optimal_polynomials
    from src.path_a.optimal_coeffs import build_Q
    t = symbols('t', real=True)

    polys = get_optimal_polynomials()
    theta_val = 4/7

    # Compute T(R*) = t-integral at R*
    Q_t = build_Q(t)
    Q_sq = expand(Q_t**2)
    T_integrand = exp(2*R*t) * Q_sq
    T_expr = integrate(T_integrand, (t, 0, 1))
    T_numeric = float(N(T_expr.subs(R, R_star_approx), 30))

    print(f"\nT(R*) = ∫₀¹ e^{{2Rt}} Q(t)² dt = {T_numeric:.10f}")

    print("\n| Pair | U_symbolic | U_from_I2 | Ratio |")
    print("|------|------------|-----------|-------|")
    for (ell1, ell2), r in results.items():
        U_sym = r['U_numeric']

        # Get I2 from numeric engine
        I2_result = compute_I2_unified_paper(
            R_star_approx, theta_val, ell1, ell2, polys,
            n_quad_u=80, n_quad_t=80, include_Q=True
        )
        I2_numeric = I2_result.I2_value

        # U = I2 × θ / T
        U_from_I2 = I2_numeric * theta_val / T_numeric

        ratio = U_sym / U_from_I2 if abs(U_from_I2) > 1e-15 else float('inf')
        print(f"| ({ell1},{ell2}) | {U_sym:+.8f} | {U_from_I2:+.8f} | {ratio:.6f} |")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
