"""
src/przz_exact_i34.py
PRZZ Exact I₃ and I₄ Evaluators

From PRZZ TeX lines 1562-1563 (I₃):
    I₃ = -T·Φ̂(0) × (1+θx)/θ × (d/dx)
         × ∫₀¹ ∫₀¹ (1-u)^{ℓ₁} P_{ℓ₁}(x+u) P_{ℓ₂}(u)
         × e^{R[t+θxt]} × e^{R[-θx+t+θxt]}
         × Q(t+θxt) × Q(-θx+t+θxt) dt du |_{x=0} + O(T/L)

From PRZZ TeX lines 1568-1569 (I₄):
    I₄ = -T·Φ̂(0) × (1+θy)/θ × (d/dy)
         × ∫₀¹ ∫₀¹ (1-u)^{ℓ₂} P_{ℓ₁}(u) P_{ℓ₂}(y+u)
         × e^{R[t+θyt]} × e^{R[-θy+t+θyt]}
         × Q(t+θyt) × Q(-θy+t+θyt) dt du |_{y=0} + O(T/L)

Key properties:
- Both have NEGATIVE sign
- Single derivatives (d/dx or d/dy)
- (1+θx)/θ or (1+θy)/θ prefactors
- Different (1-u) powers: ℓ₁ for I₃, ℓ₂ for I₄
- Position-dependent Q eigenvalues

Created: 2025-12-29
"""

from __future__ import annotations
import numpy as np
from typing import Dict
from dataclasses import dataclass

from src.quadrature import gauss_legendre_01
from src.series import TruncatedSeries


@dataclass
class I34Result:
    """Result of I₃ or I₄ evaluation."""
    value: float
    ell1: int
    ell2: int
    n_quad: int
    integral_type: str  # "I3" or "I4"
    integral_value: float  # Before sign flip


def compute_I3_przz(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 80,
) -> I34Result:
    """
    Compute I₃ for pair (ℓ₁, ℓ₂) using PRZZ's exact method.

    Formula (PRZZ lines 1562-1563):
        I₃ = -(1+θx)/θ × (d/dx)|_{x=0}
             × ∫₀¹ ∫₀¹ (1-u)^{ℓ₁} P_{ℓ₁}(x+u) P_{ℓ₂}(u)
             × exp(R[A₃_α + A₃_β]) × Q(A₃_α) × Q(A₃_β) du dt

    Where:
        A₃_α = t + θxt = t(1 + θx)
        A₃_β = -θx + t + θxt = t - θx(1-t)

    At x=0: A₃_α = A₃_β = t
    The derivative d/dx picks up terms from x-dependence.
    """
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    if P_ell1 is None or P_ell2 is None or Q is None:
        raise ValueError(f"Missing polynomials for pair ({ell1}, {ell2})")

    nodes, weights = gauss_legendre_01(n_quad)
    var_names = ("x",)  # Single variable for d/dx
    x_mask = 1  # bit 0

    integral_value = 0.0

    for i, (u, u_w) in enumerate(zip(nodes, weights)):
        for j, (t, t_w) in enumerate(zip(nodes, weights)):
            # Build integrand as series in x
            one = TruncatedSeries(var_names, {0: 1.0})
            x = TruncatedSeries(var_names, {0: 0.0, x_mask: 1.0})

            # Prefactor: (1+θx)/θ = 1/θ + x
            prefactor = one * (1.0 / theta) + x

            # (1-u)^{ℓ₁} factor
            one_minus_u_power = (1.0 - u) ** ell1

            # P_{ℓ₁}(x+u) = P1(u) + P1'(u)·x
            P1_at_u = float(P_ell1.eval(np.array([u]))[0])
            P1_deriv = float(P_ell1.eval_deriv(np.array([u]), k=1)[0])
            P1_series = one * P1_at_u + x * P1_deriv

            # P_{ℓ₂}(u) is just a scalar
            P2_at_u = float(P_ell2.eval(np.array([u]))[0])

            # Exponential: exp(R[A₃_α + A₃_β])
            # A₃_α = t(1 + θx) = t + θtx
            # A₃_β = t - θx(1-t) = t - θ(1-t)x
            # Sum: 2t + θ(2t-1)x
            exp_0 = np.exp(2 * R * t)
            exp_x_coeff = R * theta * (2 * t - 1)
            exp_series = one * exp_0 + x * (exp_0 * exp_x_coeff)

            # Q(A₃_α) × Q(A₃_β)
            # A₃_α - t = θtx
            # A₃_β - t = -θ(1-t)x
            Q_at_t = float(Q.eval(np.array([t]))[0])
            Q_deriv = float(Q.eval_deriv(np.array([t]), k=1)[0])

            # Q(A₃_α) = Q(t) + Q'(t)·θt·x
            Q_alpha = one * Q_at_t + x * (Q_deriv * theta * t)

            # Q(A₃_β) = Q(t) + Q'(t)·(-θ(1-t))·x
            Q_beta = one * Q_at_t + x * (Q_deriv * (-theta * (1 - t)))

            # Multiply all terms
            integrand = prefactor
            integrand = integrand * one_minus_u_power
            integrand = integrand * P1_series
            integrand = integrand * P2_at_u
            integrand = integrand * exp_series
            integrand = integrand * Q_alpha
            integrand = integrand * Q_beta

            # Extract x coefficient (d/dx at x=0)
            x_coeff = integrand.coeffs.get(x_mask, 0.0)
            if isinstance(x_coeff, np.ndarray):
                x_coeff = float(x_coeff)

            integral_value += x_coeff * u_w * t_w

    # Apply negative sign (I₃ is negative)
    value = -integral_value

    return I34Result(
        value=value,
        ell1=ell1,
        ell2=ell2,
        n_quad=n_quad,
        integral_type="I3",
        integral_value=integral_value,
    )


def compute_I4_przz(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 80,
) -> I34Result:
    """
    Compute I₄ for pair (ℓ₁, ℓ₂) using PRZZ's exact method.

    Formula (PRZZ lines 1568-1569):
        I₄ = -(1+θy)/θ × (d/dy)|_{y=0}
             × ∫₀¹ ∫₀¹ (1-u)^{ℓ₂} P_{ℓ₁}(u) P_{ℓ₂}(y+u)
             × exp(R[A₄_α + A₄_β]) × Q(A₄_α) × Q(A₄_β) du dt

    Where:
        A₄_α = t + θyt = t(1 + θy)
        A₄_β = -θy + t + θyt = t - θy(1-t)

    At y=0: A₄_α = A₄_β = t
    """
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    if P_ell1 is None or P_ell2 is None or Q is None:
        raise ValueError(f"Missing polynomials for pair ({ell1}, {ell2})")

    nodes, weights = gauss_legendre_01(n_quad)
    var_names = ("y",)  # Single variable for d/dy
    y_mask = 1  # bit 0

    integral_value = 0.0

    for i, (u, u_w) in enumerate(zip(nodes, weights)):
        for j, (t, t_w) in enumerate(zip(nodes, weights)):
            # Build integrand as series in y
            one = TruncatedSeries(var_names, {0: 1.0})
            y = TruncatedSeries(var_names, {0: 0.0, y_mask: 1.0})

            # Prefactor: (1+θy)/θ = 1/θ + y
            prefactor = one * (1.0 / theta) + y

            # (1-u)^{ℓ₂} factor
            one_minus_u_power = (1.0 - u) ** ell2

            # P_{ℓ₁}(u) is just a scalar
            P1_at_u = float(P_ell1.eval(np.array([u]))[0])

            # P_{ℓ₂}(y+u) = P2(u) + P2'(u)·y
            P2_at_u = float(P_ell2.eval(np.array([u]))[0])
            P2_deriv = float(P_ell2.eval_deriv(np.array([u]), k=1)[0])
            P2_series = one * P2_at_u + y * P2_deriv

            # Exponential: exp(R[A₄_α + A₄_β])
            # Same structure as I₃ but with y instead of x
            exp_0 = np.exp(2 * R * t)
            exp_y_coeff = R * theta * (2 * t - 1)
            exp_series = one * exp_0 + y * (exp_0 * exp_y_coeff)

            # Q(A₄_α) × Q(A₄_β)
            Q_at_t = float(Q.eval(np.array([t]))[0])
            Q_deriv = float(Q.eval_deriv(np.array([t]), k=1)[0])

            # Q(A₄_α) = Q(t) + Q'(t)·θt·y
            Q_alpha = one * Q_at_t + y * (Q_deriv * theta * t)

            # Q(A₄_β) = Q(t) + Q'(t)·(-θ(1-t))·y
            Q_beta = one * Q_at_t + y * (Q_deriv * (-theta * (1 - t)))

            # Multiply all terms
            integrand = prefactor
            integrand = integrand * one_minus_u_power
            integrand = integrand * P1_at_u
            integrand = integrand * P2_series
            integrand = integrand * exp_series
            integrand = integrand * Q_alpha
            integrand = integrand * Q_beta

            # Extract y coefficient (d/dy at y=0)
            y_coeff = integrand.coeffs.get(y_mask, 0.0)
            if isinstance(y_coeff, np.ndarray):
                y_coeff = float(y_coeff)

            integral_value += y_coeff * u_w * t_w

    # Apply negative sign (I₄ is negative)
    value = -integral_value

    return I34Result(
        value=value,
        ell1=ell1,
        ell2=ell2,
        n_quad=n_quad,
        integral_type="I4",
        integral_value=integral_value,
    )


def compute_I34_all_pairs(
    theta: float,
    R: float,
    polynomials: Dict,
    n_quad: int = 80,
) -> Dict[str, Dict[str, I34Result]]:
    """
    Compute I₃ and I₄ for all 6 triangle pairs.

    Returns:
        Dict with "I3" and "I4" sub-dicts, each mapping pair key to result
    """
    results = {"I3": {}, "I4": {}}

    for ell1 in [1, 2, 3]:
        for ell2 in range(ell1, 4):
            key = f"{ell1}{ell2}"
            results["I3"][key] = compute_I3_przz(
                theta, R, ell1, ell2, polynomials, n_quad
            )
            results["I4"][key] = compute_I4_przz(
                theta, R, ell1, ell2, polynomials, n_quad
            )

    return results


if __name__ == "__main__":
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
    from src.przz_exact_i1 import compute_I1_all_pairs
    from src.przz_exact_i2 import compute_I2_all_pairs
    import math

    print("=" * 70)
    print("PRZZ EXACT I₃ AND I₄ EVALUATOR TEST")
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

        # Compute all I₃ and I₄
        results = compute_I34_all_pairs(theta, R, polynomials, n_quad=80)

        # Display results
        print(f"\n  Per-pair I₃ and I₄ values:")
        print(f"  {'Pair':<6} {'I₃':>12} {'I₄':>12} {'I₃+I₄':>12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")

        I3_total = 0.0
        I4_total = 0.0
        for key in ["11", "22", "33", "12", "13", "23"]:
            r3 = results["I3"][key]
            r4 = results["I4"][key]
            # Symmetry factor for off-diagonal
            sym = 2.0 if r3.ell1 != r3.ell2 else 1.0
            # Factorial normalization
            norm = 1.0 / (math.factorial(r3.ell1) * math.factorial(r3.ell2))
            I3_total += sym * norm * r3.value
            I4_total += sym * norm * r4.value
            print(f"  {key:<6} {r3.value:>12.6f} {r4.value:>12.6f} {r3.value+r4.value:>12.6f}")

        print(f"\n  Total I₃ (with normalization): {I3_total:.6f}")
        print(f"  Total I₄ (with normalization): {I4_total:.6f}")
        print(f"  Total I₃+I₄: {I3_total + I4_total:.6f}")

        # Compute I₁ and I₂ for full picture
        I1_results = compute_I1_all_pairs(theta, R, polynomials, n_quad=80)
        I2_results = compute_I2_all_pairs(theta, R, polynomials, n_quad=80)

        I1_total = 0.0
        I2_total = 0.0
        for key in ["11", "22", "33", "12", "13", "23"]:
            r1 = I1_results[key]
            r2 = I2_results[key]
            sym = 2.0 if r1.ell1 != r1.ell2 else 1.0
            norm = 1.0 / (math.factorial(r1.ell1) * math.factorial(r1.ell2))
            I1_total += sym * norm * r1.value
            I2_total += sym * norm * r2.value

        print(f"\n  Summary:")
        print(f"    I₁ total = {I1_total:.6f}")
        print(f"    I₂ total = {I2_total:.6f}")
        print(f"    I₃ total = {I3_total:.6f}")
        print(f"    I₄ total = {I4_total:.6f}")
        print(f"    ─────────────────────")
        c_computed = I1_total + I2_total + I3_total + I4_total
        print(f"    c = I₁+I₂+I₃+I₄ = {c_computed:.6f}")

        # PRZZ targets
        if name == "kappa":
            c_target = 2.137
            kappa_target = 0.417293962
        else:
            c_target = 1.938
            kappa_target = 0.407511457

        kappa_computed = 1 - math.log(c_computed) / R
        print(f"\n  Comparison to PRZZ:")
        print(f"    c_computed = {c_computed:.6f}")
        print(f"    c_target   = {c_target:.6f}")
        print(f"    c gap      = {(c_computed - c_target)/c_target*100:+.2f}%")
        print(f"    κ_computed = {kappa_computed:.6f}")
        print(f"    κ_target   = {kappa_target:.6f}")
        print(f"    κ gap      = {(kappa_computed - kappa_target)/kappa_target*100:+.2f}%")
