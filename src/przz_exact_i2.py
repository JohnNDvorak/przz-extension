"""
src/przz_exact_i2.py
PRZZ Exact I₂ Evaluator

From PRZZ TeX line 1548:
    I₂ = T·Φ̂(0)/θ × ∫₀¹ ∫₀¹ Q(t)² e^{2Rt} P₁(u) P₂(u) dt du + O(T/L)

Key property: Q(t)² is a FROZEN SCALAR (line 1544):
    Q(D_α) Q(D_β) T^{-tα-tβ} |_{α=β=-R/L} = Q(t)² e^{2Rt}

The Q operators act on T^{-tα-tβ} which has NO x,y dependence,
so Q(t)² is just evaluated at the quadrature point t.

For general pair (ℓ₁, ℓ₂):
    I₂_{ℓ₁,ℓ₂} = (1/θ) × ∫₀¹ ∫₀¹ Q(t)² e^{2Rt} P_{ℓ₁}(u) P_{ℓ₂}(u) dt du

Note: The T·Φ̂(0) factors out and cancels in the asymptotic (1/T)×∫.
We compute the normalized version without T·Φ̂(0).

Created: 2025-12-29
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from src.quadrature import gauss_legendre_01


@dataclass
class I2Result:
    """Result of I₂ evaluation."""
    value: float
    ell1: int
    ell2: int
    n_quad: int

    # Diagnostic components
    t_integral: float  # ∫ Q(t)² e^{2Rt} dt
    u_integral: float  # ∫ P_{ℓ₁}(u) P_{ℓ₂}(u) du


def compute_I2_przz(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 80,
) -> I2Result:
    """
    Compute I₂ for pair (ℓ₁, ℓ₂) using PRZZ's exact method.

    Formula (PRZZ line 1548):
        I₂ = (1/θ) × ∫₀¹ ∫₀¹ Q(t)² e^{2Rt} P_{ℓ₁}(u) P_{ℓ₂}(u) dt du

    The integral separates into:
        I₂ = (1/θ) × [∫₀¹ Q(t)² e^{2Rt} dt] × [∫₀¹ P_{ℓ₁}(u) P_{ℓ₂}(u) du]

    Args:
        theta: PRZZ θ parameter (= 4/7)
        R: PRZZ R parameter
        ell1: First piece index (1, 2, or 3)
        ell2: Second piece index (1, 2, or 3)
        polynomials: Dict with P1, P2, P3, Q polynomial objects
        n_quad: Number of quadrature points

    Returns:
        I2Result with value and diagnostics
    """
    # Get polynomials
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    if P_ell1 is None or P_ell2 is None or Q is None:
        raise ValueError(f"Missing polynomials for pair ({ell1}, {ell2})")

    # Quadrature nodes and weights
    nodes, weights = gauss_legendre_01(n_quad)

    # Compute t-integral: ∫₀¹ Q(t)² e^{2Rt} dt
    t_integral = 0.0
    for t, w in zip(nodes, weights):
        Q_t = float(Q.eval(np.array([t]))[0])
        t_integral += Q_t * Q_t * np.exp(2 * R * t) * w

    # Compute u-integral: ∫₀¹ P_{ℓ₁}(u) P_{ℓ₂}(u) du
    u_integral = 0.0
    for u, w in zip(nodes, weights):
        P1_u = float(P_ell1.eval(np.array([u]))[0])
        P2_u = float(P_ell2.eval(np.array([u]))[0])
        u_integral += P1_u * P2_u * w

    # Combine: I₂ = (1/θ) × t_integral × u_integral
    value = (1.0 / theta) * t_integral * u_integral

    return I2Result(
        value=value,
        ell1=ell1,
        ell2=ell2,
        n_quad=n_quad,
        t_integral=t_integral,
        u_integral=u_integral,
    )


def compute_I2_all_pairs(
    theta: float,
    R: float,
    polynomials: Dict,
    n_quad: int = 80,
) -> Dict[str, I2Result]:
    """
    Compute I₂ for all 6 triangle pairs.

    Returns:
        Dict mapping pair key ("11", "12", etc.) to I2Result
    """
    results = {}

    for ell1 in [1, 2, 3]:
        for ell2 in range(ell1, 4):
            key = f"{ell1}{ell2}"
            results[key] = compute_I2_przz(
                theta, R, ell1, ell2, polynomials, n_quad
            )

    return results


if __name__ == "__main__":
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
    import math

    print("=" * 70)
    print("PRZZ EXACT I₂ EVALUATOR TEST")
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
        results = compute_I2_all_pairs(theta, R, polynomials, n_quad=80)

        # Display results
        print(f"\n  Per-pair I₂ values:")
        print(f"  {'Pair':<6} {'I₂':>12} {'t-int':>12} {'u-int':>12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")

        I2_total = 0.0
        for key in ["11", "22", "33", "12", "13", "23"]:
            r = results[key]
            # Symmetry factor for off-diagonal
            sym = 2.0 if r.ell1 != r.ell2 else 1.0
            # Factorial normalization
            norm = 1.0 / (math.factorial(r.ell1) * math.factorial(r.ell2))
            contrib = sym * norm * r.value
            I2_total += contrib
            print(f"  {key:<6} {r.value:>12.6f} {r.t_integral:>12.6f} {r.u_integral:>12.6f}")

        print(f"\n  Total I₂ (with normalization): {I2_total:.6f}")

        # Expected t-integral for comparison
        F_R = (math.exp(2*R) - 1) / (2*R)
        print(f"\n  Diagnostic:")
        print(f"    Expected ∫e^{{2Rt}}dt = (e^{{2R}}-1)/(2R) = {F_R:.6f}")
        print(f"    Actual t-integral (11) includes Q(t)²")
