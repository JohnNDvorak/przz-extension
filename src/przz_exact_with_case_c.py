"""
src/przz_exact_with_case_c.py
PRZZ Exact Evaluators WITH Case C Kernel Attenuation

This module implements PRZZ's exact formulas including Case C kernel
attenuation for pieces with ω > 0 (i.e., P₂ and P₃).

From PRZZ TeX lines 2302-2360:
- Case B (ω=0): Direct polynomial P(u)
- Case C (ω>0): Kernel-attenuated K_ω(u) = u^ω/(ω-1)! × ∫₀¹ a^{ω-1} P((1-a)u) exp(Rθua) da

For K=3, d=1:
- P₁ (ℓ=1): ω=0 → Case B (no kernel)
- P₂ (ℓ=2): ω=1 → Case C
- P₃ (ℓ=3): ω=2 → Case C

Created: 2025-12-29
Purpose: Verification that KappaEngine and PRZZ-exact compute the same thing
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, Tuple
from dataclasses import dataclass
from scipy.special import comb, factorial

from src.quadrature import gauss_legendre_01
from src.series import TruncatedSeries


def omega_for_ell(ell: int) -> int:
    """Get omega for piece index ℓ. omega = ℓ - 1."""
    return ell - 1


def case_c_kernel_value(
    P,
    u: float,
    omega: int,
    R: float,
    theta: float,
    n_quad_a: int = 40,
) -> float:
    """
    Compute K_ω(u) = u^ω/(ω-1)! × ∫₀¹ a^{ω-1} P((1-a)u) exp(Rθua) da

    For omega=0, just returns P(u) (Case B).
    """
    if omega == 0:
        return float(P.eval(np.array([u]))[0])

    gamma = R * theta
    a_nodes, a_weights = gauss_legendre_01(n_quad_a)

    # Compute integral: ∫₀¹ a^{ω-1} P((1-a)u) exp(γua) da
    integral = 0.0
    for a, w in zip(a_nodes, a_weights):
        arg = (1 - a) * u
        P_val = float(P.eval(np.array([arg]))[0])
        exp_val = np.exp(gamma * u * a)
        a_power = a ** (omega - 1)
        integral += w * a_power * P_val * exp_val

    # Prefactor: u^ω / (ω-1)!
    prefactor = (u ** omega) / factorial(omega - 1)

    return prefactor * integral


def case_c_kernel_deriv(
    P,
    u: float,
    omega: int,
    R: float,
    theta: float,
    n_quad_a: int = 40,
) -> float:
    """
    Compute d/du K_ω(u) using the product rule and derivative under integral.

    For omega=0, returns P'(u).
    """
    if omega == 0:
        return float(P.eval_deriv(np.array([u]), k=1)[0])

    gamma = R * theta
    a_nodes, a_weights = gauss_legendre_01(n_quad_a)

    # I(u) = ∫₀¹ a^{ω-1} P((1-a)u) exp(γua) da
    # I'(u) = ∫₀¹ a^{ω-1} [(1-a)P'((1-a)u) + γa P((1-a)u)] exp(γua) da

    I_val = 0.0
    I_deriv = 0.0

    for a, w in zip(a_nodes, a_weights):
        arg = (1 - a) * u
        P_val = float(P.eval(np.array([arg]))[0])
        P_deriv = float(P.eval_deriv(np.array([arg]), k=1)[0])
        exp_val = np.exp(gamma * u * a)
        a_power = a ** (omega - 1)

        I_val += w * a_power * P_val * exp_val
        I_deriv += w * a_power * exp_val * ((1 - a) * P_deriv + gamma * a * P_val)

    # K(u) = pref(u) × I(u) where pref(u) = u^ω / (ω-1)!
    # K'(u) = pref'(u) × I(u) + pref(u) × I'(u)
    # pref'(u) = ω × u^{ω-1} / (ω-1)!

    denom = factorial(omega - 1)
    pref = (u ** omega) / denom
    pref_deriv = (omega * u ** (omega - 1)) / denom

    return pref_deriv * I_val + pref * I_deriv


@dataclass
class PRZZExactCaseCResult:
    """Result from PRZZ exact with Case C."""
    I1_total: float
    I2_total: float
    I3_total: float
    I4_total: float
    c: float
    kappa: float
    R: float
    theta: float


def build_series_var(name: str, var_names: Tuple[str, ...]) -> TruncatedSeries:
    """Create a series representing a single variable."""
    idx = var_names.index(name)
    mask = 1 << idx
    return TruncatedSeries(var_names, {0: 0.0, mask: 1.0})


def build_series_constant(value: float, var_names: Tuple[str, ...]) -> TruncatedSeries:
    """Create a constant series."""
    return TruncatedSeries(var_names, {0: value})


def compute_I1_with_case_c(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 80,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I₁ for pair (ℓ₁, ℓ₂) with Case C kernel attenuation.

    Uses K_ω(x+u) instead of P(x+u) for pieces with ω > 0.
    """
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    nodes, weights = gauss_legendre_01(n_quad)
    var_names = ("x", "y")
    xy_mask = 3

    integral = 0.0

    for u, u_w in zip(nodes, weights):
        for t, t_w in zip(nodes, weights):
            x = build_series_var("x", var_names)
            y = build_series_var("y", var_names)
            one = build_series_constant(1.0, var_names)

            # Prefactor: (θ(x+y)+1)/θ = 1/θ + x + y
            prefactor = one * (1.0 / theta) + x + y

            # (1-u)^{ℓ₁+ℓ₂}
            one_minus_u_power = (1.0 - u) ** (ell1 + ell2)

            # K_{ω₁}(x+u) with Case C if needed
            # K(x+u) = K(u) + K'(u)·x (first order Taylor)
            K1_at_u = case_c_kernel_value(P_ell1, u, omega1, R, theta, n_quad_a)
            K1_deriv = case_c_kernel_deriv(P_ell1, u, omega1, R, theta, n_quad_a)
            K1_series = one * K1_at_u + x * K1_deriv

            # K_{ω₂}(y+u) with Case C if needed
            K2_at_u = case_c_kernel_value(P_ell2, u, omega2, R, theta, n_quad_a)
            K2_deriv = case_c_kernel_deriv(P_ell2, u, omega2, R, theta, n_quad_a)
            K2_series = one * K2_at_u + y * K2_deriv

            # Exponential: exp(R[A_α + A_β]) where A_α + A_β = 2t + θ(2t-1)(x+y)
            exp_0 = np.exp(2 * R * t)
            exp_xy_coeff = R * theta * (2 * t - 1)

            exp_series = one * exp_0
            exp_series = exp_series + x * (exp_0 * exp_xy_coeff)
            exp_series = exp_series + y * (exp_0 * exp_xy_coeff)
            # xy term
            current_xy = exp_series.coeffs.get(xy_mask, 0.0)
            exp_series.coeffs[xy_mask] = current_xy + exp_0 * exp_xy_coeff**2

            # Q(A_α) × Q(A_β) with position-dependent eigenvalues
            Q_at_t = float(Q.eval(np.array([t]))[0])
            Q_deriv = float(Q.eval_deriv(np.array([t]), k=1)[0])

            # Q(A_α) = Q(t) + Q'(t)·θt·x + Q'(t)·θ(t-1)·y
            Q_alpha = one * Q_at_t
            Q_alpha = Q_alpha + x * (Q_deriv * theta * t)
            Q_alpha = Q_alpha + y * (Q_deriv * theta * (t - 1))

            # Q(A_β) = Q(t) + Q'(t)·θ(t-1)·x + Q'(t)·θt·y
            Q_beta = one * Q_at_t
            Q_beta = Q_beta + x * (Q_deriv * theta * (t - 1))
            Q_beta = Q_beta + y * (Q_deriv * theta * t)

            # Multiply everything
            integrand = prefactor
            integrand = integrand * one_minus_u_power
            integrand = integrand * K1_series
            integrand = integrand * K2_series
            integrand = integrand * exp_series
            integrand = integrand * Q_alpha
            integrand = integrand * Q_beta

            # Extract xy coefficient
            xy_coeff = integrand.coeffs.get(xy_mask, 0.0)
            if isinstance(xy_coeff, np.ndarray):
                xy_coeff = float(xy_coeff)

            integral += xy_coeff * u_w * t_w

    return integral


def compute_I2_with_case_c(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 80,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I₂ for pair (ℓ₁, ℓ₂) with Case C kernel attenuation.

    I₂ = (1/θ) × ∫₀¹ ∫₀¹ Q(t)² e^{2Rt} K_{ω₁}(u) K_{ω₂}(u) dt du

    Note: I₂ uses frozen Q(t)² (no position dependence).
    """
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    nodes, weights = gauss_legendre_01(n_quad)

    # t-integral: ∫₀¹ Q(t)² e^{2Rt} dt
    t_integral = 0.0
    for t, w in zip(nodes, weights):
        Q_t = float(Q.eval(np.array([t]))[0])
        t_integral += Q_t * Q_t * np.exp(2 * R * t) * w

    # u-integral: ∫₀¹ K_{ω₁}(u) K_{ω₂}(u) du
    u_integral = 0.0
    for u, w in zip(nodes, weights):
        K1_u = case_c_kernel_value(P_ell1, u, omega1, R, theta, n_quad_a)
        K2_u = case_c_kernel_value(P_ell2, u, omega2, R, theta, n_quad_a)
        u_integral += K1_u * K2_u * w

    return (1.0 / theta) * t_integral * u_integral


def compute_I3_with_case_c(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 80,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I₃ for pair (ℓ₁, ℓ₂) with Case C kernel attenuation.

    Uses d/dx derivative, K_{ω₁}(x+u) K_{ω₂}(u), (1-u)^{ℓ₁} factor.
    """
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    nodes, weights = gauss_legendre_01(n_quad)
    var_names = ("x",)
    x_mask = 1

    integral = 0.0

    for u, u_w in zip(nodes, weights):
        for t, t_w in zip(nodes, weights):
            one = TruncatedSeries(var_names, {0: 1.0})
            x = TruncatedSeries(var_names, {0: 0.0, x_mask: 1.0})

            # Prefactor: (1+θx)/θ = 1/θ + x
            prefactor = one * (1.0 / theta) + x

            # (1-u)^{ℓ₁}
            one_minus_u_power = (1.0 - u) ** ell1

            # K_{ω₁}(x+u) = K(u) + K'(u)·x
            K1_at_u = case_c_kernel_value(P_ell1, u, omega1, R, theta, n_quad_a)
            K1_deriv = case_c_kernel_deriv(P_ell1, u, omega1, R, theta, n_quad_a)
            K1_series = one * K1_at_u + x * K1_deriv

            # K_{ω₂}(u) - just a scalar
            K2_at_u = case_c_kernel_value(P_ell2, u, omega2, R, theta, n_quad_a)

            # Exponential: exp(R[A₃_α + A₃_β]) = exp(2Rt + Rθ(2t-1)x)
            exp_0 = np.exp(2 * R * t)
            exp_x_coeff = R * theta * (2 * t - 1)
            exp_series = one * exp_0 + x * (exp_0 * exp_x_coeff)

            # Q(A₃_α) × Q(A₃_β)
            Q_at_t = float(Q.eval(np.array([t]))[0])
            Q_deriv = float(Q.eval_deriv(np.array([t]), k=1)[0])

            Q_alpha = one * Q_at_t + x * (Q_deriv * theta * t)
            Q_beta = one * Q_at_t + x * (Q_deriv * (-theta * (1 - t)))

            # Multiply
            integrand = prefactor * one_minus_u_power * K1_series * K2_at_u
            integrand = integrand * exp_series * Q_alpha * Q_beta

            # Extract x coefficient
            x_coeff = integrand.coeffs.get(x_mask, 0.0)
            if isinstance(x_coeff, np.ndarray):
                x_coeff = float(x_coeff)

            integral += x_coeff * u_w * t_w

    return -integral  # I₃ has negative sign


def compute_I4_with_case_c(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 80,
    n_quad_a: int = 40,
) -> float:
    """
    Compute I₄ for pair (ℓ₁, ℓ₂) with Case C kernel attenuation.

    Uses d/dy derivative, K_{ω₁}(u) K_{ω₂}(y+u), (1-u)^{ℓ₂} factor.
    """
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    nodes, weights = gauss_legendre_01(n_quad)
    var_names = ("y",)
    y_mask = 1

    integral = 0.0

    for u, u_w in zip(nodes, weights):
        for t, t_w in zip(nodes, weights):
            one = TruncatedSeries(var_names, {0: 1.0})
            y = TruncatedSeries(var_names, {0: 0.0, y_mask: 1.0})

            # Prefactor: (1+θy)/θ = 1/θ + y
            prefactor = one * (1.0 / theta) + y

            # (1-u)^{ℓ₂}
            one_minus_u_power = (1.0 - u) ** ell2

            # K_{ω₁}(u) - just a scalar
            K1_at_u = case_c_kernel_value(P_ell1, u, omega1, R, theta, n_quad_a)

            # K_{ω₂}(y+u) = K(u) + K'(u)·y
            K2_at_u = case_c_kernel_value(P_ell2, u, omega2, R, theta, n_quad_a)
            K2_deriv = case_c_kernel_deriv(P_ell2, u, omega2, R, theta, n_quad_a)
            K2_series = one * K2_at_u + y * K2_deriv

            # Exponential
            exp_0 = np.exp(2 * R * t)
            exp_y_coeff = R * theta * (2 * t - 1)
            exp_series = one * exp_0 + y * (exp_0 * exp_y_coeff)

            # Q factors
            Q_at_t = float(Q.eval(np.array([t]))[0])
            Q_deriv = float(Q.eval_deriv(np.array([t]), k=1)[0])

            Q_alpha = one * Q_at_t + y * (Q_deriv * theta * t)
            Q_beta = one * Q_at_t + y * (Q_deriv * (-theta * (1 - t)))

            # Multiply
            integrand = prefactor * one_minus_u_power * K1_at_u * K2_series
            integrand = integrand * exp_series * Q_alpha * Q_beta

            # Extract y coefficient
            y_coeff = integrand.coeffs.get(y_mask, 0.0)
            if isinstance(y_coeff, np.ndarray):
                y_coeff = float(y_coeff)

            integral += y_coeff * u_w * t_w

    return -integral  # I₄ has negative sign


def compute_przz_exact_case_c(
    theta: float,
    R: float,
    polynomials: Dict,
    n_quad: int = 80,
    n_quad_a: int = 40,
) -> PRZZExactCaseCResult:
    """
    Compute c and κ using PRZZ exact formulas WITH Case C kernel attenuation.

    This should match KappaEngine results.
    """
    I1_total = 0.0
    I2_total = 0.0
    I3_total = 0.0
    I4_total = 0.0

    for ell1 in [1, 2, 3]:
        for ell2 in range(ell1, 4):
            sym = 2.0 if ell1 != ell2 else 1.0
            norm = 1.0 / (math.factorial(ell1) * math.factorial(ell2))

            I1 = compute_I1_with_case_c(theta, R, ell1, ell2, polynomials, n_quad, n_quad_a)
            I2 = compute_I2_with_case_c(theta, R, ell1, ell2, polynomials, n_quad, n_quad_a)
            I3 = compute_I3_with_case_c(theta, R, ell1, ell2, polynomials, n_quad, n_quad_a)
            I4 = compute_I4_with_case_c(theta, R, ell1, ell2, polynomials, n_quad, n_quad_a)

            I1_total += sym * norm * I1
            I2_total += sym * norm * I2
            I3_total += sym * norm * I3
            I4_total += sym * norm * I4

    c = I1_total + I2_total + I3_total + I4_total
    kappa = 1 - math.log(c) / R if c > 0 else float('nan')

    return PRZZExactCaseCResult(
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
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

    print("=" * 70)
    print("PRZZ EXACT WITH CASE C: BASELINE VALIDATION")
    print("=" * 70)

    theta = 4.0 / 7.0

    for name, R, c_target, kappa_target, loader in [
        ("kappa", 1.3036, 2.137, 0.417293962, load_przz_polynomials),
        ("kappa_star", 1.1167, 1.938, 0.407511457, load_przz_polynomials_kappa_star),
    ]:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {name.upper()} (R={R})")
        print(f"{'='*60}")

        P1, P2, P3, Q = loader()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_przz_exact_case_c(theta, R, polynomials, n_quad=60)

        print(f"\n  Integral totals:")
        print(f"    I₁ = {result.I1_total:>12.6f}")
        print(f"    I₂ = {result.I2_total:>12.6f}")
        print(f"    I₃ = {result.I3_total:>12.6f}")
        print(f"    I₄ = {result.I4_total:>12.6f}")
        print(f"    ─────────────────────")
        print(f"    c  = {result.c:>12.6f}")
        print(f"    κ  = {result.kappa:>12.9f}")

        print(f"\n  Comparison to PRZZ targets:")
        print(f"    c_target   = {c_target:.6f}")
        print(f"    c gap      = {(result.c - c_target)/c_target*100:+.4f}%")
        print(f"    κ_target   = {kappa_target:.9f}")
        print(f"    κ gap      = {(result.kappa - kappa_target)/kappa_target*100:+.4f}%")
