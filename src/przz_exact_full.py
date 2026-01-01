"""
src/przz_exact_full.py
Complete PRZZ Exact Implementation with Case C Kernels and Higher-Order Derivatives

This is a full reimplementation of the PRZZ I₁, I₂, I₃, I₄ evaluators using
BivariateSeries for proper higher-order derivatives.

Key Features:
- Case C kernel attenuation for P₂ and P₃ (ω = 1, 2)
- Higher-order derivatives: extracts x^{ℓ₁} y^{ℓ₂} coefficient for pair (ℓ₁, ℓ₂)
- Factorial normalization and sign conventions matching Paper regime

Created: 2025-12-29
Purpose: Independent verification of KappaEngine results
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, List
from dataclasses import dataclass

from src.quadrature import gauss_legendre_01
from src.series_bivariate import (
    BivariateSeries,
    build_exp_bracket,
    build_log_factor,
    build_Q_factor,
)
from src.mollifier_profiles import case_c_taylor_coeffs


def omega_for_ell(ell: int) -> int:
    """Get omega for piece index ℓ. omega = ℓ - 1."""
    return ell - 1


def extract_poly_coeffs(poly) -> List[float]:
    """Extract standard basis polynomial coefficients [c0, c1, c2, ...]."""
    if hasattr(poly, 'to_monomial'):
        mono = poly.to_monomial()
        return list(mono.coeffs)
    if hasattr(poly, '_monomial'):
        return list(poly._monomial.coeffs)
    # Fallback: sample and fit
    u_points = np.linspace(0, 1, 15)
    y_points = poly.eval(u_points)
    coeffs = np.polyfit(u_points, y_points, 10)
    return list(coeffs[::-1])


def build_K_factor(
    P,
    u: float,
    var: str,
    omega: int,
    R: float,
    theta: float,
    max_dx: int,
    max_dy: int,
    n_quad_a: int = 40,
) -> BivariateSeries:
    """
    Build K_ω(u + var) as bivariate series.

    For ω=0 (Case B): just P(u + var)
    For ω>0 (Case C): kernel-attenuated K_ω(u + var)
    """
    if var == "x":
        ax, ay = 1.0, 0.0
        max_order = max_dx
    elif var == "y":
        ax, ay = 0.0, 1.0
        max_order = max_dy
    else:
        raise ValueError(f"var must be 'x' or 'y', got '{var}'")

    if omega == 0:
        # Case B: Standard polynomial composition P(u + var)
        poly_coeffs = extract_poly_coeffs(P)
        dummy = BivariateSeries.zero(max_dx, max_dy)
        return dummy.compose_polynomial(poly_coeffs, a0=u, ax=ax, ay=ay)
    else:
        # Case C: Use kernel-attenuated Taylor coefficients
        # Get K_ω^{(j)}(u) for j=0..max_order (in factorial basis)
        taylor_coeffs = case_c_taylor_coeffs(
            P, u, omega, R, theta, max_order, n_quad_a
        )
        # Build series from Taylor coefficients
        # K_ω(u + delta) = sum_{j=0}^{max_order} K_ω^{(j)}(u)/j! * delta^j
        return _build_series_from_taylor(taylor_coeffs, var, max_dx, max_dy)


def _build_series_from_taylor(
    taylor_coeffs: np.ndarray,
    var: str,
    max_dx: int,
    max_dy: int,
) -> BivariateSeries:
    """
    Build BivariateSeries from Taylor coefficients in factorial basis.

    taylor_coeffs[j] = f^{(j)}(0) (not divided by j!)
    We want: f(delta) = sum_j f^{(j)}(0)/j! * delta^j
    """
    coeffs: Dict = {}

    if var == "x":
        for j, c in enumerate(taylor_coeffs):
            if j > max_dx:
                break
            if c != 0.0:
                coeffs[(j, 0)] = c / math.factorial(j)
    else:  # var == "y"
        for j, c in enumerate(taylor_coeffs):
            if j > max_dy:
                break
            if c != 0.0:
                coeffs[(0, j)] = c / math.factorial(j)

    return BivariateSeries(max_dx=max_dx, max_dy=max_dy, coeffs=coeffs)


def compute_I1_full(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I₁ for pair (ℓ₁, ℓ₂) using full PRZZ method with Case C.

    Extracts x^{ℓ₁} y^{ℓ₂} coefficient from the bracket integrand.
    """
    max_dx = ell1
    max_dy = ell2

    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    Q_coeffs = extract_poly_coeffs(Q)

    nodes, weights = gauss_legendre_01(n_quad)

    # PRZZ (1-u) power
    one_minus_u_power = ell1 + ell2

    total = 0.0
    for u, u_w in zip(nodes, weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power

        for t, t_w in zip(nodes, weights):
            # 1. Exp factor: exp(2Rt + Rθ(2t-1)(x+y))
            a0 = 2 * R * t
            a_xy = R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)

            # 2. Log factor: 1/θ + x + y
            log_factor = build_log_factor(theta, max_dx, max_dy)

            # 3. K factors with Case C kernels
            K_x = build_K_factor(P_ell1, u, "x", omega1, R, theta, max_dx, max_dy, n_quad_a)
            K_y = build_K_factor(P_ell2, u, "y", omega2, R, theta, max_dx, max_dy, n_quad_a)

            # Build bracket
            bracket = exp_factor * log_factor * K_x * K_y

            # 4. Q factors
            Q_alpha = build_Q_factor(
                Q_coeffs,
                a0=t,
                ax=theta * (t - 1),
                ay=theta * t,
                max_dx=max_dx,
                max_dy=max_dy,
            )
            Q_beta = build_Q_factor(
                Q_coeffs,
                a0=t,
                ax=theta * t,
                ay=theta * (t - 1),
                max_dx=max_dx,
                max_dy=max_dy,
            )
            bracket = bracket * Q_alpha * Q_beta

            # 5. Extract x^ℓ₁ y^ℓ₂ coefficient
            coeff = bracket.extract(ell1, ell2)

            # 6. Add to integral
            total += coeff * one_minus_u_factor * u_w * t_w

    # 7. Apply factorial normalization
    total *= math.factorial(ell1) * math.factorial(ell2)

    # 8. Apply sign convention for off-diagonal pairs
    if ell1 != ell2:
        sign = (-1) ** (ell1 + ell2)
        total *= sign

    return total


def compute_I2_full(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I₂ for pair (ℓ₁, ℓ₂) using full PRZZ method with Case C.

    I₂ = (1/θ) × ∫₀¹ ∫₀¹ Q(t)² e^{2Rt} K_{ω₁}(u) K_{ω₂}(u) dt du

    Note: I₂ uses frozen Q(t)² and no derivatives (just K values at u).
    """
    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    nodes, weights = gauss_legendre_01(n_quad)

    # t-integral: ∫₀¹ Q(t)² e^{2Rt} dt
    t_integral = 0.0
    for t, w in zip(nodes, weights):
        Q_t = float(Q.eval(np.array([t]))[0])
        t_integral += Q_t * Q_t * np.exp(2 * R * t) * w

    # u-integral: ∫₀¹ K_{ω₁}(u) K_{ω₂}(u) du
    u_integral = 0.0
    for u, w in zip(nodes, weights):
        # Case C kernel values (just K(u), no derivatives needed)
        K1_taylor = case_c_taylor_coeffs(P_ell1, u, omega1, R, theta, 0, n_quad_a) if omega1 > 0 else None
        K2_taylor = case_c_taylor_coeffs(P_ell2, u, omega2, R, theta, 0, n_quad_a) if omega2 > 0 else None

        K1_u = K1_taylor[0] if K1_taylor is not None else float(P_ell1.eval(np.array([u]))[0])
        K2_u = K2_taylor[0] if K2_taylor is not None else float(P_ell2.eval(np.array([u]))[0])

        u_integral += K1_u * K2_u * w

    return (1.0 / theta) * t_integral * u_integral


def compute_I3_full(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I₃ for pair (ℓ₁, ℓ₂) using full PRZZ method with Case C.

    I₃ uses d^{ℓ₁}/dx^{ℓ₁} derivative, K_{ω₁}(x+u) K_{ω₂}(u), (1-u)^{ℓ₁} factor.

    NOTE: For I₃/I₄, the log factor is a SCALAR prefactor -1/θ, NOT part of the bracket!
    This differs from I₁/I₂ where log factor is (1+θx)/θ inside the bracket.
    """
    max_dx = ell1
    max_dy = 0  # No y dependence in I₃

    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    Q_coeffs = extract_poly_coeffs(Q)

    nodes, weights = gauss_legendre_01(n_quad)

    total = 0.0
    for u, u_w in zip(nodes, weights):
        one_minus_u_factor = (1.0 - u) ** ell1

        # K_{ω₂}(u) - just a scalar
        K2_taylor = case_c_taylor_coeffs(P_ell2, u, omega2, R, theta, 0, n_quad_a) if omega2 > 0 else None
        K2_u = K2_taylor[0] if K2_taylor is not None else float(P_ell2.eval(np.array([u]))[0])

        for t, t_w in zip(nodes, weights):
            # NO log factor inside bracket for I₃/I₄ - it's a scalar prefactor

            # Exp factor: exp(2Rt + Rθ(2t-1)x)
            exp_factor = build_exp_bracket(2 * R * t, R * theta * (2 * t - 1), 0, max_dx, max_dy)

            # K_{ω₁}(x+u)
            K_x = build_K_factor(P_ell1, u, "x", omega1, R, theta, max_dx, max_dy, n_quad_a)

            # Q factors
            Q_alpha = build_Q_factor(Q_coeffs, a0=t, ax=theta * t, ay=0, max_dx=max_dx, max_dy=max_dy)
            Q_beta = build_Q_factor(Q_coeffs, a0=t, ax=-theta * (1 - t), ay=0, max_dx=max_dx, max_dy=max_dy)

            # Build bracket (no log factor!)
            bracket = exp_factor * K_x * Q_alpha * Q_beta

            # Extract x^{ℓ₁} coefficient
            coeff = bracket.extract(ell1, 0)

            # Add to integral (with K2_u scalar)
            total += coeff * K2_u * one_minus_u_factor * u_w * t_w

    # Factorial normalization
    total *= math.factorial(ell1)

    # Scalar prefactor -1/θ for I₃
    return -total / theta


def compute_I4_full(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I₄ for pair (ℓ₁, ℓ₂) using full PRZZ method with Case C.

    I₄ uses d^{ℓ₂}/dy^{ℓ₂} derivative, K_{ω₁}(u) K_{ω₂}(y+u), (1-u)^{ℓ₂} factor.

    NOTE: For I₃/I₄, the log factor is a SCALAR prefactor -1/θ, NOT part of the bracket!
    This differs from I₁/I₂ where log factor is (1+θx)/θ inside the bracket.
    """
    max_dx = 0  # No x dependence in I₄
    max_dy = ell2

    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    Q_coeffs = extract_poly_coeffs(Q)

    nodes, weights = gauss_legendre_01(n_quad)

    total = 0.0
    for u, u_w in zip(nodes, weights):
        one_minus_u_factor = (1.0 - u) ** ell2

        # K_{ω₁}(u) - just a scalar
        K1_taylor = case_c_taylor_coeffs(P_ell1, u, omega1, R, theta, 0, n_quad_a) if omega1 > 0 else None
        K1_u = K1_taylor[0] if K1_taylor is not None else float(P_ell1.eval(np.array([u]))[0])

        for t, t_w in zip(nodes, weights):
            # NO log factor inside bracket for I₃/I₄ - it's a scalar prefactor

            # Exp factor: exp(2Rt + Rθ(2t-1)y)
            exp_factor = build_exp_bracket(2 * R * t, 0, R * theta * (2 * t - 1), max_dx, max_dy)

            # K_{ω₂}(y+u)
            K_y = build_K_factor(P_ell2, u, "y", omega2, R, theta, max_dx, max_dy, n_quad_a)

            # Q factors
            Q_alpha = build_Q_factor(Q_coeffs, a0=t, ax=0, ay=theta * t, max_dx=max_dx, max_dy=max_dy)
            Q_beta = build_Q_factor(Q_coeffs, a0=t, ax=0, ay=-theta * (1 - t), max_dx=max_dx, max_dy=max_dy)

            # Build bracket (no log factor!)
            bracket = exp_factor * K_y * Q_alpha * Q_beta

            # Extract y^{ℓ₂} coefficient
            coeff = bracket.extract(0, ell2)

            # Add to integral (with K1_u scalar)
            total += coeff * K1_u * one_minus_u_factor * u_w * t_w

    # Factorial normalization
    total *= math.factorial(ell2)

    # Scalar prefactor -1/θ for I₄
    return -total / theta


@dataclass
class PRZZExactFullResult:
    """Complete result from PRZZ exact full computation."""
    I1_total: float
    I2_total: float
    I3_total: float
    I4_total: float
    c: float
    kappa: float
    R: float
    theta: float


def compute_przz_exact_full(
    theta: float,
    R: float,
    polynomials: Dict,
    n_quad: int = 60,
    n_quad_a: int = 40,
) -> PRZZExactFullResult:
    """
    Compute c and κ using complete PRZZ method with Case C and higher-order derivatives.
    """
    I1_total = 0.0
    I2_total = 0.0
    I3_total = 0.0
    I4_total = 0.0

    for ell1 in [1, 2, 3]:
        for ell2 in range(ell1, 4):
            sym = 2.0 if ell1 != ell2 else 1.0
            norm = 1.0 / (math.factorial(ell1) * math.factorial(ell2))

            I1 = compute_I1_full(theta, R, ell1, ell2, polynomials, n_quad, n_quad_a)
            I2 = compute_I2_full(theta, R, ell1, ell2, polynomials, n_quad, n_quad_a)
            I3 = compute_I3_full(theta, R, ell1, ell2, polynomials, n_quad, n_quad_a)
            I4 = compute_I4_full(theta, R, ell1, ell2, polynomials, n_quad, n_quad_a)

            # Note: factorial normalization is already applied inside compute_I*_full
            # But the per-pair normalization 1/(ℓ₁!ℓ₂!) needs to cancel with it
            I1_total += sym * norm * I1
            I2_total += sym * norm * I2
            I3_total += sym * norm * I3
            I4_total += sym * norm * I4

    c = I1_total + I2_total + I3_total + I4_total
    kappa = 1 - math.log(c) / R if c > 0 else float('nan')

    return PRZZExactFullResult(
        I1_total=I1_total,
        I2_total=I2_total,
        I3_total=I3_total,
        I4_total=I4_total,
        c=c,
        kappa=kappa,
        R=R,
        theta=theta,
    )


if __name__ == "__main__":
    from src.polynomials import load_przz_polynomials

    print("=" * 70)
    print("PRZZ EXACT FULL: VERIFICATION")
    print("=" * 70)

    theta = 4.0 / 7.0
    R = 1.3036

    P1, P2, P3, Q = load_przz_polynomials()
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Compare per-pair with Paper regime
    from src.unified_i1_paper import compute_I1_unified_paper
    from src.unified_i2_paper import compute_I2_unified_paper

    print("\nPer-pair comparison: PRZZ-exact-full vs Paper regime")
    print("=" * 70)
    print(f"{'Pair':<6} {'Full I1':>12} {'Paper I1':>12} {'Full I2':>12} {'Paper I2':>12}")
    print("-" * 70)

    all_pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    for ell1, ell2 in all_pairs:
        full_I1 = compute_I1_full(theta, R, ell1, ell2, polynomials, n_quad=60)
        full_I2 = compute_I2_full(theta, R, ell1, ell2, polynomials, n_quad=60)

        paper_I1 = compute_I1_unified_paper(
            theta=theta, R=R, ell1=ell1, ell2=ell2,
            polynomials=polynomials,
            n_quad_u=60, n_quad_t=60, n_quad_a=40,
            include_Q=True
        )
        paper_I2 = compute_I2_unified_paper(
            theta=theta, R=R, ell1=ell1, ell2=ell2,
            polynomials=polynomials,
            n_quad_u=60, n_quad_t=60, n_quad_a=40,
            include_Q=True
        )

        I1_diff = abs(full_I1 - paper_I1.I1_value)
        I2_diff = abs(full_I2 - paper_I2.I2_value)
        status = "✓" if I1_diff < 0.001 and I2_diff < 0.001 else "✗"

        print(f"{ell1}{ell2:<5} {full_I1:>12.6f} {paper_I1.I1_value:>12.6f} {full_I2:>12.6f} {paper_I2.I2_value:>12.6f}  {status}")

    # Compare I3/I4 per-pair with KappaEngine
    print("\n" + "=" * 70)
    print("I₃/I₄ PER-PAIR COMPARISON")
    print("=" * 70)

    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term

    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")
    f_norm = {"11": 1.0, "22": 0.25, "33": 1.0/36, "12": 0.5, "13": 1.0/6, "23": 1.0/12}
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    print(f"{'Pair':<6} {'Full I3':>12} {'DSL I3':>12} {'Full I4':>12} {'DSL I4':>12}")
    print("-" * 60)

    full_I3_total = 0.0
    full_I4_total = 0.0
    dsl_I3_total = 0.0
    dsl_I4_total = 0.0

    for ell1, ell2 in all_pairs:
        pair_key = f"{ell1}{ell2}"
        norm = f_norm[pair_key]
        sym = symmetry[pair_key]

        full_I3 = compute_I3_full(theta, R, ell1, ell2, polynomials, n_quad=60)
        full_I4 = compute_I4_full(theta, R, ell1, ell2, polynomials, n_quad=60)

        terms = all_terms[pair_key]
        dsl_I3 = 0.0
        dsl_I4 = 0.0
        if len(terms) > 2:
            I3_result = evaluate_term(terms[2], polynomials, 60, R=R, theta=theta, n_quad_a=40)
            dsl_I3 = I3_result.value
        if len(terms) > 3:
            I4_result = evaluate_term(terms[3], polynomials, 60, R=R, theta=theta, n_quad_a=40)
            dsl_I4 = I4_result.value

        # Accumulate with normalization
        full_I3_total += sym * norm * full_I3
        full_I4_total += sym * norm * full_I4
        dsl_I3_total += sym * norm * dsl_I3
        dsl_I4_total += sym * norm * dsl_I4

        print(f"{ell1}{ell2:<5} {full_I3:>12.6f} {dsl_I3:>12.6f} {full_I4:>12.6f} {dsl_I4:>12.6f}")

    print("-" * 60)
    print(f"{'Total':<6} {full_I3_total:>12.6f} {dsl_I3_total:>12.6f} {full_I4_total:>12.6f} {dsl_I4_total:>12.6f}")
    print(f"\nS34 full = {full_I3_total + full_I4_total:.6f}")
    print(f"S34 DSL  = {dsl_I3_total + dsl_I4_total:.6f}")

    # Full computation with split-channel assembly
    print("\n" + "=" * 70)
    print("SPLIT-CHANNEL ASSEMBLY (m = exp(R) + 5)")
    print("=" * 70)

    # Compute at +R
    result_plus = compute_przz_exact_full(theta, R, polynomials, n_quad=60, n_quad_a=40)
    S12_plus = result_plus.I1_total + result_plus.I2_total
    S34_plus = result_plus.I3_total + result_plus.I4_total

    # Compute at -R
    result_minus = compute_przz_exact_full(theta, -R, polynomials, n_quad=60, n_quad_a=40)
    S12_minus = result_minus.I1_total + result_minus.I2_total

    # Split-channel formula: c = S12(+R) + m*S12(-R) + S34(+R)
    m = math.exp(R) + 5
    c_split = S12_plus + m * S12_minus + S34_plus
    kappa_split = 1 - math.log(c_split) / R if c_split > 0 else float('nan')

    print(f"\nAt R = +{R}:")
    print(f"  I₁(+R) = {result_plus.I1_total:>12.6f}")
    print(f"  I₂(+R) = {result_plus.I2_total:>12.6f}")
    print(f"  I₃(+R) = {result_plus.I3_total:>12.6f}")
    print(f"  I₄(+R) = {result_plus.I4_total:>12.6f}")
    print(f"  S12(+R) = {S12_plus:>11.6f}")
    print(f"  S34(+R) = {S34_plus:>11.6f}")

    print(f"\nAt R = -{R}:")
    print(f"  I₁(-R) = {result_minus.I1_total:>12.6f}")
    print(f"  I₂(-R) = {result_minus.I2_total:>12.6f}")
    print(f"  S12(-R) = {S12_minus:>11.6f}")

    print(f"\nSplit-channel assembly:")
    print(f"  m = exp(R) + 5 = {m:.6f}")
    print(f"  c = S12+ + m*S12- + S34+")
    print(f"  c = {S12_plus:.6f} + {m:.4f}*{S12_minus:.6f} + {S34_plus:.6f}")
    print(f"  c = {c_split:.6f}")
    print(f"  κ = 1 - log(c)/R = {kappa_split:.6f}")

    # Compare to target
    c_target = 2.137
    kappa_target = 0.417293962

    print(f"\nComparison to PRZZ targets:")
    print(f"  c_target = {c_target:.6f}")
    print(f"  c_gap    = {(c_split - c_target)/c_target*100:+.3f}%")
    print(f"  κ_target = {kappa_target:.6f}")
    print(f"  κ_gap    = {(kappa_split - kappa_target)/kappa_target*100:+.3f}%")

    # Compare to KappaEngine
    print("\n" + "=" * 70)
    print("COMPARISON WITH KAPPA ENGINE")
    print("=" * 70)

    from src.kappa_engine import KappaEngine

    engine = KappaEngine.from_przz_kappa(n_quad=60)
    ke_result = engine.compute_kappa()

    print(f"\nKappaEngine:")
    print(f"  S12(+R) = {ke_result.integrals.S12_plus:.6f}")
    print(f"  S12(-R) = {ke_result.integrals.S12_minus:.6f}")
    print(f"  S34(+R) = {ke_result.integrals.S34_plus:.6f}")
    print(f"  m       = {ke_result.corrections.m:.6f}")
    print(f"  c       = {ke_result.c:.6f}")
    print(f"  κ       = {ke_result.kappa:.6f}")

    print(f"\nPRZZ-exact-full:")
    print(f"  S12(+R) = {S12_plus:.6f}")
    print(f"  S12(-R) = {S12_minus:.6f}")
    print(f"  S34(+R) = {S34_plus:.6f}")
    print(f"  m       = {m:.6f}")
    print(f"  c       = {c_split:.6f}")
    print(f"  κ       = {kappa_split:.6f}")

    print(f"\nDifferences (PRZZ-full - KappaEngine):")
    print(f"  S12(+R): {S12_plus - ke_result.integrals.S12_plus:+.6f}")
    print(f"  S12(-R): {S12_minus - ke_result.integrals.S12_minus:+.6f}")
    print(f"  S34(+R): {S34_plus - ke_result.integrals.S34_plus:+.6f}")
    print(f"  c:       {c_split - ke_result.c:+.6f} ({(c_split - ke_result.c)/ke_result.c*100:+.2f}%)")
    print(f"  κ:       {kappa_split - ke_result.kappa:+.6f}")

    # Test with optimal polynomials (κ ≈ 0.52)
    print("\n" + "=" * 70)
    print("VERIFICATION: OPTIMAL POLYNOMIALS (κ ≈ 0.52)")
    print("=" * 70)

    import json
    with open("/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/data/optimal_polynomials_v2.json") as f:
        opt_data = json.load(f)

    from src.polynomials import P1Polynomial, PellPolynomial, QPolynomial

    # Load optimal polynomials
    P1_opt = P1Polynomial(opt_data["P1_tilde"])
    P2_opt = PellPolynomial(opt_data["P2_tilde"])
    P3_opt = PellPolynomial(opt_data["P3_tilde"])
    # Q uses basis coefficients like {1: 0.636851, 3: -0.159327, 5: 0.032011}
    Q_basis = {int(k): v for k, v in opt_data["Q_basis"].items() if k != "note"}
    Q_opt = QPolynomial(Q_basis)

    opt_polys = {"P1": P1_opt, "P2": P2_opt, "P3": P3_opt, "Q": Q_opt}

    print(f"\nExpected: κ = {opt_data['kappa']:.4f}, c = {opt_data['c']:.6f}")

    # Compute with PRZZ-exact-full
    opt_plus = compute_przz_exact_full(theta, R, opt_polys, n_quad=60, n_quad_a=40)
    opt_minus = compute_przz_exact_full(theta, -R, opt_polys, n_quad=60, n_quad_a=40)

    opt_S12_plus = opt_plus.I1_total + opt_plus.I2_total
    opt_S34_plus = opt_plus.I3_total + opt_plus.I4_total
    opt_S12_minus = opt_minus.I1_total + opt_minus.I2_total

    opt_c_split = opt_S12_plus + m * opt_S12_minus + opt_S34_plus
    opt_kappa_split = 1 - math.log(opt_c_split) / R if opt_c_split > 0 else float('nan')

    print(f"\nPRZZ-exact-full (m = exp(R)+5):")
    print(f"  S12(+R) = {opt_S12_plus:.6f}")
    print(f"  S12(-R) = {opt_S12_minus:.6f}")
    print(f"  S34(+R) = {opt_S34_plus:.6f}")
    print(f"  c       = {opt_c_split:.6f}")
    print(f"  κ       = {opt_kappa_split:.6f}")

    # Compare with KappaEngine
    opt_engine = KappaEngine(
        P1_coeffs=opt_data["P1_tilde"],
        P2_coeffs=opt_data["P2_tilde"],
        P3_coeffs=opt_data["P3_tilde"],
        Q_coeffs=opt_data["Q_mono"],
        theta=theta, K=3, R=R, n_quad=60
    )
    opt_ke = opt_engine.compute_kappa()

    print(f"\nKappaEngine (with g-factors):")
    print(f"  S12(+R) = {opt_ke.integrals.S12_plus:.6f}")
    print(f"  S12(-R) = {opt_ke.integrals.S12_minus:.6f}")
    print(f"  S34(+R) = {opt_ke.integrals.S34_plus:.6f}")
    print(f"  m       = {opt_ke.corrections.m:.6f}")
    print(f"  c       = {opt_ke.c:.6f}")
    print(f"  κ       = {opt_ke.kappa:.6f}")

    print(f"\nIntegral differences (PRZZ-full - KappaEngine):")
    print(f"  S12(+R): {opt_S12_plus - opt_ke.integrals.S12_plus:+.6f}")
    print(f"  S12(-R): {opt_S12_minus - opt_ke.integrals.S12_minus:+.6f}")
    print(f"  S34(+R): {opt_S34_plus - opt_ke.integrals.S34_plus:+.6f}")

    print(f"\nVERIFICATION RESULT:")
    if abs(opt_ke.kappa - opt_data['kappa']) < 0.01:
        print(f"  ✓ Optimal polynomials verified: κ = {opt_ke.kappa:.4f} ≈ {opt_data['kappa']:.4f}")
    else:
        print(f"  ✗ Mismatch: κ = {opt_ke.kappa:.4f} vs expected {opt_data['kappa']:.4f}")
