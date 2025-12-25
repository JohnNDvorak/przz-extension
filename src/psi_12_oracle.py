"""
Ψ-expansion oracle for the (1,2) cross-pair.

This implements the full Ψ combinatorial expansion for μ × μ⋆Λ (piece 1 × piece 2).

INCREMENTAL IMPLEMENTATION STRATEGY:
    Start with the simplest term (DB ~ I₂-type), verify it works, then add complexity.

The Ψ formula for (ℓ=1, ℓ̄=2):
    Ψ_{1,2}(A,B,C,D) = Σ_{p=0}^{min(1,2)} C(1,p)C(2,p)p! × (D-C²)^p × (A-C)^{1-p} × (B-C)^{2-p}

After expansion and simplification, this gives 7 monomials:
    Ψ_{1,2} = AB² - 2ABC + AC² - B²C + C³ + 2DB - 2DC

INTERPRETATION OF A, B, C, D:
Based on the (1,1) validated mapping (HANDOFF Section 14.3):
    - D corresponds to I₂-type integrals (no derivatives)
    - A, B, C correspond to derivative-weighted integrals
    - The signs and combinatorics encode the PRZZ derivative extraction structure

For the (1,2) pair:
    - P₁ is the μ piece (ℓ=1)
    - P₂ is the μ⋆Λ piece (ℓ=2)

CURRENT APPROACH:
We start with a SIMPLIFIED model where:
    D = (1/θ) × ∫∫ P₁(u) P₂(u) Q(t)² exp(2Rt) du dt    [I₂-type]
    A, B, C = simpler factors based on individual polynomials

This is Version 1 - just to test the monomial structure.
Refinement will come from comparing with DSL and understanding what's missing.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, NamedTuple


class OracleResult12(NamedTuple):
    """Result of (1,2) oracle computation via Ψ expansion."""
    # Individual monomial contributions
    AB2: float
    ABC: float  # coefficient -2
    AC2: float
    B2C: float  # coefficient -1
    C3: float
    DB: float   # coefficient +2
    DC: float   # coefficient -2
    # Total
    total: float


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def psi_oracle_12(
    P1,  # P₁ polynomial (piece 1 = μ)
    P2,  # P₂ polynomial (piece 2 = μ⋆Λ)
    Q,   # Q polynomial
    theta: float,
    R: float,
    n_quad: int = 60,
    debug: bool = False
) -> OracleResult12:
    """
    Compute the (1,2) contribution using full Ψ expansion (7 monomials).

    This computes:
        Ψ_{1,2} = AB² - 2ABC + AC² - B²C + C³ + 2DB - 2DC

    where A, B, C, D are fundamental integrals involving P₁, P₂, Q.

    Args:
        P1: The P₁ polynomial (piece 1 = μ)
        P2: The P₂ polynomial (piece 2 = μ⋆Λ)
        Q: The Q polynomial
        theta: θ = 4/7
        R: R parameter
        n_quad: Quadrature points
        debug: Print debug info

    Returns:
        OracleResult12 with individual monomial values and total
    """
    # Set up quadrature
    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # Precompute polynomial values at u-nodes
    P1_u = P1.eval(u_nodes)
    P2_u = P2.eval(u_nodes)

    # Precompute Q values at t-nodes
    Q_t = Q.eval(t_nodes)

    # Precompute exp(2Rt)
    exp_2Rt = np.exp(2 * R * t_nodes)

    # Common t-integral factor: ∫ Q(t)² exp(2Rt) dt
    t_integral = np.sum(t_weights * Q_t * Q_t * exp_2Rt)

    # =========================================================================
    # Compute the fundamental building blocks A, B, C, D
    #
    # VERSION 1 (Simplified I₂-style baseline):
    # We start by defining D as the I₂-type integral for (1,2):
    #     D = (1/θ) × ∫ P₁(u) P₂(u) du × ∫ Q(t)² exp(2Rt) dt
    #
    # And A, B, C as simpler single-polynomial factors:
    #     A = (1/θ) × ∫ P₁(u) du × ∫ Q(t)² exp(2Rt) dt
    #     B = (1/θ) × ∫ P₂(u) du × ∫ Q(t)² exp(2Rt) dt
    #     C = (1/θ) × ∫ 1 du × ∫ Q(t)² exp(2Rt) dt
    #
    # This is the simplest model. We'll refine based on testing.
    # =========================================================================

    # Prefactor from PRZZ formulas (see przz_22_exact_oracle.py)
    prefactor = 1.0 / theta

    # Common factor: ∫ Q(t)² exp(2Rt) dt
    t_int = t_integral

    # u-integrals
    u_int_P1 = np.sum(u_weights * P1_u)
    u_int_P2 = np.sum(u_weights * P2_u)
    u_int_P1P2 = np.sum(u_weights * P1_u * P2_u)
    u_int_1 = np.sum(u_weights)  # = 1.0

    # A = (1/θ) × ∫ P₁(u) du × ∫ Q(t)² exp(2Rt) dt
    A = prefactor * u_int_P1 * t_int

    # B = (1/θ) × ∫ P₂(u) du × ∫ Q(t)² exp(2Rt) dt
    B = prefactor * u_int_P2 * t_int

    # C = (1/θ) × ∫ 1 du × ∫ Q(t)² exp(2Rt) dt = (1/θ) × t_int
    C = prefactor * u_int_1 * t_int

    # D = (1/θ) × ∫ P₁(u)P₂(u) du × ∫ Q(t)² exp(2Rt) dt
    # This is the I₂-type integral for the (1,2) pair
    D = prefactor * u_int_P1P2 * t_int

    if debug:
        print(f"\nBuilding blocks:")
        print(f"  u-integrals:")
        print(f"    ∫ P₁(u) du:      {u_int_P1:.6f}")
        print(f"    ∫ P₂(u) du:      {u_int_P2:.6f}")
        print(f"    ∫ P₁P₂ du:       {u_int_P1P2:.6f}")
        print(f"  t-integral:")
        print(f"    ∫ Q²e^{{2Rt}} dt:  {t_int:.6f}")
        print(f"  Combined (with 1/θ prefactor):")
        print(f"    A (P₁ factor):   {A:.6f}")
        print(f"    B (P₂ factor):   {B:.6f}")
        print(f"    C (constant):    {C:.6f}")
        print(f"    D (P₁P₂ I₂):     {D:.6f}")

    # =========================================================================
    # Compute the 7 monomials
    # =========================================================================

    # +1 × AB²
    monomial_AB2 = A * B * B

    # -2 × ABC
    monomial_ABC = A * B * C

    # +1 × AC²
    monomial_AC2 = A * C * C

    # -1 × B²C
    monomial_B2C = B * B * C

    # +1 × C³
    monomial_C3 = C * C * C

    # +2 × DB
    monomial_DB = D * B

    # -2 × DC
    monomial_DC = D * C

    if debug:
        print(f"\nMonomials (before coefficients):")
        print(f"  AB²:  {monomial_AB2:.6f} → ×1  = {monomial_AB2:.6f}")
        print(f"  ABC:  {monomial_ABC:.6f} → ×-2 = {-2*monomial_ABC:.6f}")
        print(f"  AC²:  {monomial_AC2:.6f} → ×1  = {monomial_AC2:.6f}")
        print(f"  B²C:  {monomial_B2C:.6f} → ×-1 = {-monomial_B2C:.6f}")
        print(f"  C³:   {monomial_C3:.6f} → ×1  = {monomial_C3:.6f}")
        print(f"  DB:   {monomial_DB:.6f} → ×2  = {2*monomial_DB:.6f}")
        print(f"  DC:   {monomial_DC:.6f} → ×-2 = {-2*monomial_DC:.6f}")

    # Total with correct signs and coefficients
    total = (
        monomial_AB2
        - 2 * monomial_ABC
        + monomial_AC2
        - monomial_B2C
        + monomial_C3
        + 2 * monomial_DB
        - 2 * monomial_DC
    )

    if debug:
        print(f"\nTotal Ψ_{{1,2}}: {total:.6f}")

    return OracleResult12(
        AB2=monomial_AB2,
        ABC=monomial_ABC,
        AC2=monomial_AC2,
        B2C=monomial_B2C,
        C3=monomial_C3,
        DB=monomial_DB,
        DC=monomial_DC,
        total=total
    )


if __name__ == "__main__":
    # Quick test
    from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

    print("="*60)
    print("Testing Ψ (1,2) Oracle")
    print("="*60)

    theta = 4/7

    # Test with κ polynomials (R=1.3036)
    print("\n--- κ Benchmark (R=1.3036) ---")
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    R_kappa = 1.3036
    result_k = psi_oracle_12(P1_k, P2_k, Q_k, theta, R_kappa, n_quad=80, debug=True)

    # Test with κ* polynomials (R=1.1167)
    print("\n--- κ* Benchmark (R=1.1167) ---")
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    R_kappa_star = 1.1167
    result_ks = psi_oracle_12(P1_ks, P2_ks, Q_ks, theta, R_kappa_star, n_quad=80, debug=True)

    # Compare ratios
    print("\n--- Comparison ---")
    print(f"κ / κ* ratio: {result_k.total / result_ks.total:.4f}")
    print(f"\nExpected ratio from DSL: 129× (catastrophic cancellation)")
    print(f"If this Ψ oracle is correct, we should see a MUCH better ratio.")
