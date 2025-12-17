"""
src/psi_22_oracle.py
Ψ-Based Oracle for (2,2) Pair Using Correct Derivative Structure

This implements the 12-monomial evaluation for (ℓ=2, ℓ̄=2) using
the log-derivative interpretation of A, B, C, D.

For an integrand F = e^L where L = log F:
  A = ∂L/∂x |_{x=y=0}  (z-derivative of log)
  B = ∂L/∂y |_{x=y=0}  (w-derivative of log)
  D = ∂²L/∂x∂y |_{x=y=0}  (mixed derivative of log)
  C is handled through the connected blocks (A-C), (B-C), (D-C²)

The key insight from GPT:
  A² = (∂L/∂x)² = product of two singleton z-blocks
  B² = (∂L/∂y)² = product of two singleton w-blocks
  D² = (∂²L/∂x∂y)² = product of two paired blocks

For each monomial A^a B^b C^c D^d, we compute:
  ∫∫ F₀ × (A)^a × (B)^b × (some_factor)^c × (D)^d × weights du dt
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Dict, Tuple, NamedTuple
from math import exp, log
from dataclasses import dataclass


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


@dataclass
class LogDerivatives:
    """Log-derivative values A, B, D at a (u, t) point."""
    A: float  # ∂L/∂x
    B: float  # ∂L/∂y
    D: float  # ∂²L/∂x∂y
    F0: float # Base integrand value


def compute_log_derivatives(
    P, Q, theta: float, R: float, u: float, t: float
) -> LogDerivatives:
    """
    Compute log-derivative building blocks at a single (u, t) point.

    For integrand F = (1+θ(x+y))/θ × P(x+u)P(y+u) × Q(α)Q(β) × e^{R(α+β)}
    where α = t + θtx + θ(t-1)y, β = t + θ(t-1)x + θty

    At x=y=0: α = β = t, so F₀ = (1/θ) P(u)² Q(t)² e^{2Rt}

    L = log F = log(pref) + 2 log P(u) + 2 log Q(t) + 2Rt  (at x=y=0)
    """
    # Evaluate at u and t
    P_u = P.eval([u])[0]
    P_prime_u = P.eval_deriv([u], 1)[0]

    Q_t = Q.eval([t])[0]
    Q_prime_t = Q.eval_deriv([t], 1)[0]
    Q_double_prime_t = Q.eval_deriv([t], 2)[0]

    # Skip if near zero
    if abs(P_u) < 1e-15 or abs(Q_t) < 1e-15:
        return LogDerivatives(A=0, B=0, D=0, F0=0)

    # Base integrand F₀
    prefactor_0 = 1.0 / theta
    F0 = prefactor_0 * P_u * P_u * Q_t * Q_t * exp(2 * R * t)

    # Argument derivatives at x=y=0
    darg_alpha_dx = theta * t
    darg_alpha_dy = theta * (t - 1)
    darg_beta_dx = theta * (t - 1)
    darg_beta_dy = theta * t

    # A = ∂L/∂x at x=y=0
    # L = log(pref) + log P(x+u) + log P(y+u) + log Q(α) + log Q(β) + R(α+β)
    # ∂L/∂x = θ + P'/P + (Q'/Q)(∂α/∂x) + (Q'/Q)(∂β/∂x) + R(∂α/∂x + ∂β/∂x)
    A = theta  # from prefactor
    A += P_prime_u / P_u  # from P(x+u)
    A += (Q_prime_t / Q_t) * darg_alpha_dx  # from Q(α)
    A += (Q_prime_t / Q_t) * darg_beta_dx   # from Q(β)
    A += R * (darg_alpha_dx + darg_beta_dx) # from exp

    # B = ∂L/∂y at x=y=0 (same structure)
    B = theta
    B += P_prime_u / P_u  # from P(y+u)
    B += (Q_prime_t / Q_t) * darg_alpha_dy
    B += (Q_prime_t / Q_t) * darg_beta_dy
    B += R * (darg_alpha_dy + darg_beta_dy)

    # D = ∂²L/∂x∂y at x=y=0
    # From prefactor: d²(log pref)/dxdy = -θ²/(1+θ(x+y))² → -θ² at x=y=0
    # From Q terms: involves (Q''/Q - (Q'/Q)²) × (∂α/∂x)(∂α/∂y) + similar for β
    d2_logQ_dt2 = Q_double_prime_t / Q_t - (Q_prime_t / Q_t) ** 2

    D = -theta ** 2  # from prefactor
    D += d2_logQ_dt2 * darg_alpha_dx * darg_alpha_dy  # from Q(α)
    D += d2_logQ_dt2 * darg_beta_dx * darg_beta_dy    # from Q(β)

    return LogDerivatives(A=A, B=B, D=D, F0=F0)


def evaluate_monomial_logderiv(
    a: int, b: int, c: int, d: int,
    P, Q, theta: float, R: float, n_quad: int = 60,
    ell: int = 1, ellbar: int = 1
) -> float:
    """
    Evaluate monomial A^a B^b C^c D^d using log-derivative integration.

    The monomial value is approximately:
    ∫∫ F₀ × A^a × B^b × ... × D^d × (1-u)^{ℓ+ℓ̄} du dt

    Note: This is a simplified approach. The full PRZZ Section 7 machinery
    involves more complex handling of the C factor and case selection.
    """
    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    result = 0.0

    for iu, u in enumerate(u_nodes):
        wu = u_weights[iu]
        one_minus_u_power = (1.0 - u) ** (ell + ellbar)

        for it, t in enumerate(t_nodes):
            wt = t_weights[it]

            ld = compute_log_derivatives(P, Q, theta, R, u, t)

            if ld.F0 < 1e-15:
                continue

            # Monomial contribution: A^a × B^b × D^d
            # The C factor is tricky - for now we use a placeholder
            monomial_value = 1.0
            if a > 0:
                monomial_value *= ld.A ** a
            if b > 0:
                monomial_value *= ld.B ** b
            if d > 0:
                monomial_value *= ld.D ** d

            # For C^c, we need a different approach
            # C represents the "base" or "disconnected" part
            # For now, use a scaling factor
            if c > 0:
                # C is related to the log of the base integrand
                # This is a simplification
                log_F0 = log(ld.F0) if ld.F0 > 0 else 0
                monomial_value *= log_F0 ** c

            # Integrate F₀ × monomial × weights
            integrand = ld.F0 * monomial_value * one_minus_u_power
            result += wu * wt * integrand

    return result


def test_11_logderiv_approach():
    """Test log-derivative approach on (1,1) where we know the answer."""
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22
    from src.psi_monomial_expansion import expand_pair_to_monomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("=" * 60)
    print("TEST: Log-Derivative Approach for (1,1)")
    print("=" * 60)

    # Oracle reference
    oracle = przz_oracle_22(P1, Q, theta, R, n_quad)
    print(f"\nOracle total: {oracle.total:.6f}")

    # Get monomials
    monomials = expand_pair_to_monomials(1, 1)

    print("\nLog-derivative monomial evaluations:")
    total = 0.0
    for (a, b, c, d), coeff in sorted(monomials.items()):
        val = evaluate_monomial_logderiv(a, b, c, d, P1, Q, theta, R, n_quad, ell=1, ellbar=1)
        contrib = coeff * val
        total += contrib

        mono_str = f"A^{a}B^{b}C^{c}D^{d}"
        print(f"  {coeff:+d} × {mono_str:<12} = {coeff:+d} × {val:.4f} = {contrib:+.4f}")

    print(f"\nLog-deriv total: {total:.6f}")
    print(f"Oracle total: {oracle.total:.6f}")
    print(f"Ratio: {total / oracle.total:.4f}" if abs(oracle.total) > 1e-10 else "N/A")


def test_22_logderiv_approach():
    """Test log-derivative approach on (2,2)."""
    from src.polynomials import load_przz_polynomials
    from src.psi_monomial_expansion import expand_pair_to_monomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("\n" + "=" * 60)
    print("TEST: Log-Derivative Approach for (2,2)")
    print("=" * 60)

    # Get monomials
    monomials = expand_pair_to_monomials(2, 2)

    print("\nLog-derivative monomial evaluations:")
    total = 0.0
    for (a, b, c, d), coeff in sorted(monomials.items()):
        try:
            val = evaluate_monomial_logderiv(a, b, c, d, P2, Q, theta, R, n_quad, ell=2, ellbar=2)
            contrib = coeff * val
            total += contrib

            mono_str = f"A^{a}B^{b}C^{c}D^{d}"
            print(f"  {coeff:+d} × {mono_str:<12} = {coeff:+d} × {val:.4f} = {contrib:+.4f}")
        except Exception as e:
            print(f"  ERROR for ({a},{b},{c},{d}): {e}")

    print(f"\nLog-deriv total for (2,2): {total:.6f}")


if __name__ == "__main__":
    test_11_logderiv_approach()
    test_22_logderiv_approach()
