"""
src/psi_unified_evaluator.py
Unified Ψ-Based Evaluator for PRZZ κ Computation

This module integrates all Ψ oracles into a single pipeline for computing c and κ.

Key features:
- Evaluates all 6 pairs: (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
- Uses dedicated oracles where available
- Applies factorial normalization: 1/(ℓ₁! × ℓ₂!)
- Applies symmetry factor 2 for off-diagonal pairs
- Returns per-pair breakdown AND total c
- Computes κ = 1 - log(c)/R

Oracle mapping:
- (1,1): przz_22_exact_oracle (validated reference)
- (2,2): psi_22_complete_oracle
- (3,3): psi_33_oracle
- (1,2): psi_12_oracle
- (1,3): TODO - stub using psi_term_generator
- (2,3): TODO - stub using psi_term_generator

Design philosophy:
- Start with validated pairs and build incrementally
- Use consistent polynomial loading from polynomials.py
- Use consistent quadrature from quadrature.py
- Return detailed breakdowns for debugging
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Tuple, NamedTuple
from dataclasses import dataclass

from src.przz_22_exact_oracle import przz_oracle_22
from src.psi_22_complete_oracle import Psi22CompleteOracle
from src.psi_33_oracle import psi_oracle_33
from src.psi_12_oracle import psi_oracle_12


@dataclass
class PairResult:
    """Result for a single (ℓ₁, ℓ₂) pair."""
    ell1: int
    ell2: int
    raw_value: float
    factorial_norm: float
    symmetry_factor: float
    normalized_value: float


class UnifiedResult(NamedTuple):
    """Complete result from unified evaluator."""
    c_total: float
    kappa: float
    R: float
    n_quad: int

    # Per-pair raw values (before normalization)
    c11_raw: float
    c22_raw: float
    c33_raw: float
    c12_raw: float
    c13_raw: float
    c23_raw: float

    # Per-pair normalized values (after factorial + symmetry)
    c11_norm: float
    c22_norm: float
    c33_norm: float
    c12_norm: float
    c13_norm: float
    c23_norm: float


def evaluate_c_psi(
    theta: float,
    R: float,
    n_quad: int,
    polynomials: Dict,
    return_breakdown: bool = True
) -> UnifiedResult:
    """
    Evaluate total c using Ψ oracles for all 6 pairs.

    Computes:
        c = Σ_{ℓ₁ ≤ ℓ₂} multiplier(ℓ₁, ℓ₂) × c_{ℓ₁,ℓ₂}

    where multiplier = (symmetry_factor) × 1/(ℓ₁! × ℓ₂!)

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (1.3036 for κ, 1.1167 for κ*)
        n_quad: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        return_breakdown: If True, include detailed per-pair info

    Returns:
        UnifiedResult with c, κ, and per-pair breakdown
    """
    # Extract polynomials
    P1 = polynomials["P1"]
    P2 = polynomials["P2"]
    P3 = polynomials["P3"]
    Q = polynomials["Q"]

    # Factorial normalization factors
    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1/1 = 1
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 1/4
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 1/2
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    }

    # Symmetry factors
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,  # Diagonal
        "12": 2.0, "13": 2.0, "23": 2.0   # Off-diagonal
    }

    # ==========================================================================
    # (1,1) pair - Use validated PRZZ oracle
    # ==========================================================================
    # NOTE: For (1,1), PRZZ uses P1 for both factors
    oracle_11 = przz_oracle_22(P1, Q, theta, R, n_quad, debug=False)
    c11_raw = oracle_11.total

    # ==========================================================================
    # (2,2) pair - Use Ψ complete oracle
    # ==========================================================================
    # For (2,2), use P2 for both factors
    psi_22 = Psi22CompleteOracle(P2, Q, theta, R, n_quad)
    c22_raw, _ = psi_22.compute_all_monomials(verbose=False)

    # ==========================================================================
    # (3,3) pair - Use Ψ oracle
    # ==========================================================================
    # For (3,3), use P3 for both factors
    result_33 = psi_oracle_33(P3, Q, theta, R, n_quad, debug=False)
    c33_raw = result_33.total

    # ==========================================================================
    # (1,2) pair - Use Ψ oracle
    # ==========================================================================
    result_12 = psi_oracle_12(P1, P2, Q, theta, R, n_quad, debug=False)
    c12_raw = result_12.total

    # ==========================================================================
    # (1,3) pair - TODO: needs full implementation
    # ==========================================================================
    # Placeholder: use simple estimate for now
    c13_raw = _estimate_13_pair(P1, P3, Q, theta, R, n_quad)

    # ==========================================================================
    # (2,3) pair - TODO: needs full implementation
    # ==========================================================================
    # Placeholder: use simple estimate for now
    c23_raw = _estimate_23_pair(P2, P3, Q, theta, R, n_quad)

    # ==========================================================================
    # Apply normalization and compute total
    # ==========================================================================

    c11_norm = factorial_norm["11"] * symmetry_factor["11"] * c11_raw
    c22_norm = factorial_norm["22"] * symmetry_factor["22"] * c22_raw
    c33_norm = factorial_norm["33"] * symmetry_factor["33"] * c33_raw
    c12_norm = factorial_norm["12"] * symmetry_factor["12"] * c12_raw
    c13_norm = factorial_norm["13"] * symmetry_factor["13"] * c13_raw
    c23_norm = factorial_norm["23"] * symmetry_factor["23"] * c23_raw

    c_total = c11_norm + c22_norm + c33_norm + c12_norm + c13_norm + c23_norm

    # Compute κ
    kappa = 1.0 - math.log(c_total) / R

    return UnifiedResult(
        c_total=c_total,
        kappa=kappa,
        R=R,
        n_quad=n_quad,
        c11_raw=c11_raw,
        c22_raw=c22_raw,
        c33_raw=c33_raw,
        c12_raw=c12_raw,
        c13_raw=c13_raw,
        c23_raw=c23_raw,
        c11_norm=c11_norm,
        c22_norm=c22_norm,
        c33_norm=c33_norm,
        c12_norm=c12_norm,
        c13_norm=c13_norm,
        c23_norm=c23_norm
    )


def _estimate_13_pair(P1, P3, Q, theta: float, R: float, n_quad: int) -> float:
    """
    TODO: Implement full (1,3) oracle.

    Placeholder: use simple I₂-type integral as rough estimate.
    This will be replaced with proper Ψ expansion later.
    """
    from src.quadrature import gauss_legendre_01

    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # Evaluate polynomials
    P1_u = P1.eval(u_nodes)
    P3_u = P3.eval(u_nodes)
    Q_t = Q.eval(t_nodes)

    # Simple I₂-type integral: (1/θ) × ∫∫ P₁(u)P₃(u) Q(t)² e^{2Rt} du dt
    u_integral = np.sum(u_weights * P1_u * P3_u)
    t_integral = np.sum(t_weights * Q_t * Q_t * np.exp(2 * R * t_nodes))

    return (1.0 / theta) * u_integral * t_integral


def _estimate_23_pair(P2, P3, Q, theta: float, R: float, n_quad: int) -> float:
    """
    TODO: Implement full (2,3) oracle.

    Placeholder: use simple I₂-type integral as rough estimate.
    This will be replaced with proper Ψ expansion later.
    """
    from src.quadrature import gauss_legendre_01

    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # Evaluate polynomials
    P2_u = P2.eval(u_nodes)
    P3_u = P3.eval(u_nodes)
    Q_t = Q.eval(t_nodes)

    # Simple I₂-type integral: (1/θ) × ∫∫ P₂(u)P₃(u) Q(t)² e^{2Rt} du dt
    u_integral = np.sum(u_weights * P2_u * P3_u)
    t_integral = np.sum(t_weights * Q_t * Q_t * np.exp(2 * R * t_nodes))

    return (1.0 / theta) * u_integral * t_integral


def print_evaluation_report(result: UnifiedResult, polynomial_set: str = "κ") -> None:
    """
    Print detailed evaluation report.

    Args:
        result: UnifiedResult from evaluate_c_psi
        polynomial_set: Label for the polynomial set ("κ" or "κ*")
    """
    print("\n" + "=" * 70)
    print(f"Ψ UNIFIED EVALUATOR REPORT - {polynomial_set} Polynomials")
    print("=" * 70)

    print(f"\nParameters:")
    print(f"  R = {result.R}")
    print(f"  n_quad = {result.n_quad}")

    print(f"\nPer-Pair Raw Values (before normalization):")
    print(f"  c₁₁ (raw): {result.c11_raw:+18.12f}")
    print(f"  c₂₂ (raw): {result.c22_raw:+18.12f}")
    print(f"  c₃₃ (raw): {result.c33_raw:+18.12f}")
    print(f"  c₁₂ (raw): {result.c12_raw:+18.12f}")
    print(f"  c₁₃ (raw): {result.c13_raw:+18.12f} [STUB]")
    print(f"  c₂₃ (raw): {result.c23_raw:+18.12f} [STUB]")

    print(f"\nPer-Pair Normalized Values (after factorial + symmetry):")
    print(f"  c₁₁ (norm): {result.c11_norm:+18.12f} (×1)")
    print(f"  c₂₂ (norm): {result.c22_norm:+18.12f} (×1/4)")
    print(f"  c₃₃ (norm): {result.c33_norm:+18.12f} (×1/36)")
    print(f"  c₁₂ (norm): {result.c12_norm:+18.12f} (×2/2 = ×1)")
    print(f"  c₁₃ (norm): {result.c13_norm:+18.12f} (×2/6 = ×1/3) [STUB]")
    print(f"  c₂₃ (norm): {result.c23_norm:+18.12f} (×2/12 = ×1/6) [STUB]")

    print(f"\nTotals:")
    print(f"  c (total): {result.c_total:20.15f}")
    print(f"  κ:         {result.kappa:20.15f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Quick test with both polynomial sets
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

    theta = 4.0 / 7.0
    n_quad = 60

    print("=" * 70)
    print("UNIFIED Ψ EVALUATOR TEST")
    print("=" * 70)

    # Test with κ polynomials (R=1.3036)
    print("\n--- κ Benchmark (R=1.3036) ---")
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}
    R_kappa = 1.3036

    result_k = evaluate_c_psi(theta, R_kappa, n_quad, polys_k)
    print_evaluation_report(result_k, polynomial_set="κ")

    # Test with κ* polynomials (R=1.1167)
    print("\n\n--- κ* Benchmark (R=1.1167) ---")
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    polys_ks = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}
    R_kappa_star = 1.1167

    result_ks = evaluate_c_psi(theta, R_kappa_star, n_quad, polys_ks)
    print_evaluation_report(result_ks, polynomial_set="κ*")

    # Compare ratios
    print("\n\n--- Comparison ---")
    print(f"c(κ) / c(κ*) ratio:  {result_k.c_total / result_ks.c_total:.4f}")
    print(f"Target ratio:        1.10")
    print(f"\nNote: The stubs for (1,3) and (2,3) are I₂-type estimates.")
    print(f"Full implementation requires proper Ψ expansion for these pairs.")
