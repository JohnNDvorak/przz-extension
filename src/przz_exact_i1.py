"""
src/przz_exact_i1.py
PRZZ Exact I₁ Evaluator

From PRZZ TeX lines 1530-1532:
    I₁ = T·Φ̂(0) × (d²/dxdy) × [(θ(x+y)+1)/θ]
         × ∫₀¹ ∫₀¹ (1-u)^{ℓ₁+ℓ₂} P_{ℓ₁}(x+u) P_{ℓ₂}(y+u)
         × exp(R[A_α + A_β])
         × Q(A_α) × Q(A_β)
         |_{x=y=0} du dt + O(T/L)

Where the affine eigenvalue arguments are:
    A_α = θt(x+y) - θy + t = t + θt·x + θ(t-1)·y
    A_β = θt(x+y) - θx + t = t + θ(t-1)·x + θt·y

Key property: I₁ uses POSITION-DEPENDENT Q eigenvalues.
The Q operators act on arguments that depend on x, y, so when we take
d²/dxdy, we get contributions from Q'(t).

For general pair (ℓ₁, ℓ₂), the derivative is d^{ℓ₁+ℓ₂}/dx^{ℓ₁}dy^{ℓ₂}
but for K=3, d=1, all ℓ values are 1, so we always use d²/dxdy.

Created: 2025-12-29
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from src.quadrature import gauss_legendre_01
from src.series import TruncatedSeries


@dataclass
class I1Result:
    """Result of I₁ evaluation."""
    value: float
    ell1: int
    ell2: int
    n_quad: int

    # Diagnostic components
    integral_value: float  # Before 1/θ factor


def build_series_var(name: str, var_names: Tuple[str, ...]) -> TruncatedSeries:
    """Create a series representing a single variable."""
    idx = var_names.index(name)
    mask = 1 << idx
    return TruncatedSeries(var_names, {0: 0.0, mask: 1.0})


def build_series_constant(value: float, var_names: Tuple[str, ...]) -> TruncatedSeries:
    """Create a constant series."""
    return TruncatedSeries(var_names, {0: value})


def compute_I1_integrand_series(
    u: float,
    t: float,
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    P_ell1,
    P_ell2,
    Q,
    var_names: Tuple[str, ...] = ("x", "y"),
) -> TruncatedSeries:
    """
    Build the series for the I₁ integrand.

    The integrand is:
        [(θ(x+y)+1)/θ] × (1-u)^{ℓ₁+ℓ₂} × P_{ℓ₁}(x+u) × P_{ℓ₂}(y+u)
        × exp(R[A_α + A_β]) × Q(A_α) × Q(A_β)

    We build this as a TruncatedSeries and extract the xy coefficient.
    """
    x = build_series_var("x", var_names)
    y = build_series_var("y", var_names)
    one = build_series_constant(1.0, var_names)

    # Affine eigenvalue arguments:
    # A_α = t + θt·x + θ(t-1)·y
    # A_β = t + θ(t-1)·x + θt·y
    A_alpha_0 = t  # constant part
    A_alpha_x = theta * t  # coefficient of x
    A_alpha_y = theta * (t - 1)  # coefficient of y

    A_beta_0 = t  # constant part
    A_beta_x = theta * (t - 1)  # coefficient of x
    A_beta_y = theta * t  # coefficient of y

    # Build A_α and A_β as series
    A_alpha = build_series_constant(A_alpha_0, var_names)
    A_alpha = A_alpha + x * A_alpha_x + y * A_alpha_y

    A_beta = build_series_constant(A_beta_0, var_names)
    A_beta = A_beta + x * A_beta_x + y * A_beta_y

    # Prefactor: (θ(x+y)+1)/θ = 1/θ + x + y
    # This contributes to xy coefficient through interaction with other terms
    prefactor = one * (1.0 / theta) + x + y

    # (1-u)^{ℓ₁+ℓ₂} factor
    one_minus_u_power = (1.0 - u) ** (ell1 + ell2)

    # P_{ℓ₁}(x+u) and P_{ℓ₂}(y+u) as series
    # P(x+u) = P(u) + P'(u)·x + (1/2)P''(u)·x² + ...
    # Since x² = 0, we have P(x+u) = P(u) + P'(u)·x
    P1_at_u = float(P_ell1.eval(np.array([u]))[0])
    P1_deriv_at_u = float(P_ell1.eval_deriv(np.array([u]), k=1)[0])

    P2_at_u = float(P_ell2.eval(np.array([u]))[0])
    P2_deriv_at_u = float(P_ell2.eval_deriv(np.array([u]), k=1)[0])

    # P_{ℓ₁}(x+u) = P1_at_u + P1_deriv_at_u * x
    P1_series = one * P1_at_u + x * P1_deriv_at_u

    # P_{ℓ₂}(y+u) = P2_at_u + P2_deriv_at_u * y
    P2_series = one * P2_at_u + y * P2_deriv_at_u

    # Exponential: exp(R[A_α + A_β])
    # A_α + A_β = 2t + θ(2t-1)·x + θ(2t-1)·y
    # exp(...) ≈ exp(2Rt) × [1 + Rθ(2t-1)·x + Rθ(2t-1)·y + R²θ²(2t-1)²·xy + ...]
    exp_0 = np.exp(2 * R * t)
    exp_x_coeff = R * theta * (2 * t - 1)
    exp_y_coeff = R * theta * (2 * t - 1)
    exp_xy_coeff = R**2 * theta**2 * (2 * t - 1)**2

    exp_series = one * exp_0
    exp_series = exp_series + (x * exp_0 * exp_x_coeff)
    exp_series = exp_series + (y * exp_0 * exp_y_coeff)
    # xy term from exp(R(A_α+A_β))
    xy_mask = 3  # bit 0 | bit 1
    current_xy = exp_series.coeffs.get(xy_mask, 0.0)
    exp_series.coeffs[xy_mask] = current_xy + exp_0 * exp_xy_coeff

    # Q(A_α) × Q(A_β) with position-dependent eigenvalues
    # Q(A_α) = Q(t) + Q'(t)·(A_α - t) + (1/2)Q''(t)·(A_α - t)² + ...
    # A_α - t = θt·x + θ(t-1)·y
    # Since x² = 0, y² = 0, we have:
    # Q(A_α) = Q(t) + Q'(t)·[θt·x + θ(t-1)·y]
    Q_at_t = float(Q.eval(np.array([t]))[0])
    Q_deriv_at_t = float(Q.eval_deriv(np.array([t]), k=1)[0])

    # Q(A_α) = Q(t) + Q'(t)·θt·x + Q'(t)·θ(t-1)·y
    Q_alpha = one * Q_at_t
    Q_alpha = Q_alpha + x * (Q_deriv_at_t * theta * t)
    Q_alpha = Q_alpha + y * (Q_deriv_at_t * theta * (t - 1))

    # Q(A_β) = Q(t) + Q'(t)·θ(t-1)·x + Q'(t)·θt·y
    Q_beta = one * Q_at_t
    Q_beta = Q_beta + x * (Q_deriv_at_t * theta * (t - 1))
    Q_beta = Q_beta + y * (Q_deriv_at_t * theta * t)

    # Multiply everything together
    # integrand = prefactor × one_minus_u_power × P1_series × P2_series × exp_series × Q_alpha × Q_beta
    integrand = prefactor
    integrand = integrand * one_minus_u_power
    integrand = integrand * P1_series
    integrand = integrand * P2_series
    integrand = integrand * exp_series
    integrand = integrand * Q_alpha
    integrand = integrand * Q_beta

    return integrand


def compute_I1_przz(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 80,
) -> I1Result:
    """
    Compute I₁ for pair (ℓ₁, ℓ₂) using PRZZ's exact method.

    Formula (PRZZ lines 1530-1532):
        I₁ = (d²/dxdy)|_{x=y=0} × [(θ(x+y)+1)/θ]
             × ∫₀¹ ∫₀¹ (1-u)^{ℓ₁+ℓ₂} P_{ℓ₁}(x+u) P_{ℓ₂}(y+u)
             × exp(R[A_α + A_β]) × Q(A_α) × Q(A_β) du dt

    The d²/dxdy derivative extracts the xy coefficient from the series.

    Args:
        theta: PRZZ θ parameter (= 4/7)
        R: PRZZ R parameter
        ell1: First piece index (1, 2, or 3)
        ell2: Second piece index (1, 2, or 3)
        polynomials: Dict with P1, P2, P3, Q polynomial objects
        n_quad: Number of quadrature points

    Returns:
        I1Result with value and diagnostics
    """
    # Get polynomials
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    if P_ell1 is None or P_ell2 is None or Q is None:
        raise ValueError(f"Missing polynomials for pair ({ell1}, {ell2})")

    # Quadrature nodes and weights
    nodes, weights = gauss_legendre_01(n_quad)
    var_names = ("x", "y")
    xy_mask = 3  # bit 0 | bit 1

    # Integrate: ∫₀¹ ∫₀¹ [xy coefficient of integrand] du dt
    integral_value = 0.0

    for i, (u, u_w) in enumerate(zip(nodes, weights)):
        for j, (t, t_w) in enumerate(zip(nodes, weights)):
            # Build the series for this (u, t) point
            integrand = compute_I1_integrand_series(
                u, t, theta, R, ell1, ell2, P_ell1, P_ell2, Q, var_names
            )

            # Extract xy coefficient (this is d²/dxdy evaluated at x=y=0)
            xy_coeff = integrand.coeffs.get(xy_mask, 0.0)
            if isinstance(xy_coeff, np.ndarray):
                xy_coeff = float(xy_coeff)

            integral_value += xy_coeff * u_w * t_w

    # The prefactor (θ(x+y)+1)/θ is already included in the series
    # No additional 1/θ factor needed here
    value = integral_value

    return I1Result(
        value=value,
        ell1=ell1,
        ell2=ell2,
        n_quad=n_quad,
        integral_value=integral_value,
    )


def compute_I1_all_pairs(
    theta: float,
    R: float,
    polynomials: Dict,
    n_quad: int = 80,
) -> Dict[str, I1Result]:
    """
    Compute I₁ for all 6 triangle pairs.

    Returns:
        Dict mapping pair key ("11", "12", etc.) to I1Result
    """
    results = {}

    for ell1 in [1, 2, 3]:
        for ell2 in range(ell1, 4):
            key = f"{ell1}{ell2}"
            results[key] = compute_I1_przz(
                theta, R, ell1, ell2, polynomials, n_quad
            )

    return results


if __name__ == "__main__":
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
    import math

    print("=" * 70)
    print("PRZZ EXACT I₁ EVALUATOR TEST")
    print("=" * 70)

    theta = 4.0 / 7.0

    for name, R, loader in [
        ("kappa", 1.3036, load_przz_polynomials),
        ("kappa_star", 1.1167, load_przz_polynomials_kappa_star),
    ]:
        print(f"\n{'='*60}")
        print(f"Benchmark: {name.upper()} (R={R})")
        print(f"{'='*60}")

        P1, P2, P3, Q = loader()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        # Compute all pairs
        results = compute_I1_all_pairs(theta, R, polynomials, n_quad=80)

        # Display results
        print(f"\n  Per-pair I₁ values:")
        print(f"  {'Pair':<6} {'I₁':>14} {'integral':>14}")
        print(f"  {'-'*6} {'-'*14} {'-'*14}")

        I1_total = 0.0
        for key in ["11", "22", "33", "12", "13", "23"]:
            r = results[key]
            # Symmetry factor for off-diagonal
            sym = 2.0 if r.ell1 != r.ell2 else 1.0
            # Factorial normalization
            norm = 1.0 / (math.factorial(r.ell1) * math.factorial(r.ell2))
            contrib = sym * norm * r.value
            I1_total += contrib
            print(f"  {key:<6} {r.value:>14.6f} {r.integral_value:>14.6f}")

        print(f"\n  Total I₁ (with normalization): {I1_total:.6f}")

        # Compare with I₂
        from src.przz_exact_i2 import compute_I2_all_pairs

        I2_results = compute_I2_all_pairs(theta, R, polynomials, n_quad=80)
        I2_total = 0.0
        for key in ["11", "22", "33", "12", "13", "23"]:
            r2 = I2_results[key]
            sym = 2.0 if r2.ell1 != r2.ell2 else 1.0
            norm = 1.0 / (math.factorial(r2.ell1) * math.factorial(r2.ell2))
            I2_total += sym * norm * r2.value

        print(f"\n  Comparison:")
        print(f"    I₁ total = {I1_total:.6f}")
        print(f"    I₂ total = {I2_total:.6f}")
        print(f"    I₁ + I₂  = {I1_total + I2_total:.6f}")
