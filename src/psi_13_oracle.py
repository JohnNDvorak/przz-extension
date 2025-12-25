"""
Ψ-expansion oracle for the (1,3) cross-pair.

This implements the full Ψ combinatorial expansion for μ × μ⋆Λ⋆Λ (piece 1 × piece 3).

The Ψ formula for (ℓ=1, ℓ̄=3):
    Ψ_{1,3}(A,B,C,D) = Σ_{p=0}^{min(1,3)} C(1,p)C(3,p)p! × (D-C²)^p × (A-C)^{1-p} × (B-C)^{3-p}

After expansion and simplification, this gives 10 monomials:
    p=0: (A-C)(B-C)³ = AB³ - 3AB²C + 3ABC² - AC³ - B³C + 3B²C² - 3BC³ + C⁴
    p=1: 3(D-C²)(B-C)² = 3DB² - 6DBC + 3DC² - 3B²C² + 6BC³ - 3C⁴

Combining like terms (B²C² coefficient: 3-3=0):
    AB³ - 3AB²C + 3ABC² - AC³ - B³C + 3BC³ - 2C⁴ + 3DB² - 6DBC + 3DC²

Total: 10 monomials with nonzero coefficients.

INTERPRETATION OF A, B, C, D:
Based on the (1,1) validated mapping:
    - D corresponds to I₂-type integrals (no derivatives)
    - A, B, C correspond to derivative-weighted integrals
    - The signs and combinatorics encode the PRZZ derivative extraction structure

For the (1,3) pair:
    - P₁ is the μ piece (ℓ=1)
    - P₃ is the μ⋆Λ⋆Λ piece (ℓ=3)

IMPORTANT: P₃ changes sign on [0,1], so ∫P₁P₃ can be negative.
From HANDOFF_SUMMARY: (1,3) ratio was 5.73 in old DSL.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, NamedTuple


class OracleResult13(NamedTuple):
    """Result of (1,3) oracle computation via Ψ expansion."""
    # Individual monomial contributions (raw values, before applying coefficients)
    AB3: float
    AB2C: float
    ABC2: float
    AC3: float
    B3C: float
    BC3: float
    C4: float
    DB2: float
    DBC: float
    DC2: float
    # Total
    total: float


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def psi_oracle_13(
    P1,  # P₁ polynomial (piece 1 = μ)
    P3,  # P₃ polynomial (piece 3 = μ⋆Λ⋆Λ)
    Q,   # Q polynomial
    theta: float,
    R: float,
    n_quad: int = 60,
    debug: bool = False
) -> OracleResult13:
    """
    Compute the (1,3) contribution using full Ψ expansion (8 monomials).

    This computes:
        Ψ_{1,3} = (A-C)(B-C)³ + 3(D-C²)(B-C)²

    where A, B, C, D are fundamental integrals involving P₁, P₃, Q.

    Args:
        P1: The P₁ polynomial (piece 1 = μ)
        P3: The P₃ polynomial (piece 3 = μ⋆Λ⋆Λ)
        Q: The Q polynomial
        theta: θ = 4/7
        R: R parameter
        n_quad: Quadrature points
        debug: Print debug info

    Returns:
        OracleResult13 with individual monomial values and total
    """
    # Set up quadrature
    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # Precompute polynomial values at u-nodes
    P1_u = P1.eval(u_nodes)
    P3_u = P3.eval(u_nodes)

    # Precompute Q values at t-nodes
    Q_t = Q.eval(t_nodes)

    # Precompute exp(2Rt)
    exp_2Rt = np.exp(2 * R * t_nodes)

    # Common t-integral factor: ∫ Q(t)² exp(2Rt) dt
    t_integral = np.sum(t_weights * Q_t * Q_t * exp_2Rt)

    # =========================================================================
    # Compute the fundamental building blocks A, B, C, D
    #
    # Following the (1,2) pattern:
    #     A = (1/θ) × ∫ P₁(u) du × ∫ Q(t)² exp(2Rt) dt
    #     B = (1/θ) × ∫ P₃(u) du × ∫ Q(t)² exp(2Rt) dt
    #     C = (1/θ) × ∫ 1 du × ∫ Q(t)² exp(2Rt) dt
    #     D = (1/θ) × ∫ P₁(u)P₃(u) du × ∫ Q(t)² exp(2Rt) dt
    # =========================================================================

    # Prefactor from PRZZ formulas
    prefactor = 1.0 / theta

    # Common factor: ∫ Q(t)² exp(2Rt) dt
    t_int = t_integral

    # u-integrals
    u_int_P1 = np.sum(u_weights * P1_u)
    u_int_P3 = np.sum(u_weights * P3_u)
    u_int_P1P3 = np.sum(u_weights * P1_u * P3_u)
    u_int_1 = np.sum(u_weights)  # = 1.0

    # A = (1/θ) × ∫ P₁(u) du × ∫ Q(t)² exp(2Rt) dt
    A = prefactor * u_int_P1 * t_int

    # B = (1/θ) × ∫ P₃(u) du × ∫ Q(t)² exp(2Rt) dt
    B = prefactor * u_int_P3 * t_int

    # C = (1/θ) × ∫ 1 du × ∫ Q(t)² exp(2Rt) dt = (1/θ) × t_int
    C = prefactor * u_int_1 * t_int

    # D = (1/θ) × ∫ P₁(u)P₃(u) du × ∫ Q(t)² exp(2Rt) dt
    # This is the I₂-type integral for the (1,3) pair
    D = prefactor * u_int_P1P3 * t_int

    if debug:
        print(f"\nBuilding blocks:")
        print(f"  u-integrals:")
        print(f"    ∫ P₁(u) du:      {u_int_P1:.6f}")
        print(f"    ∫ P₃(u) du:      {u_int_P3:.6f}")
        print(f"    ∫ P₁P₃ du:       {u_int_P1P3:.6f}")
        print(f"  t-integral:")
        print(f"    ∫ Q²e^{{2Rt}} dt:  {t_int:.6f}")
        print(f"  Combined (with 1/θ prefactor):")
        print(f"    A (P₁ factor):   {A:.6f}")
        print(f"    B (P₃ factor):   {B:.6f}")
        print(f"    C (constant):    {C:.6f}")
        print(f"    D (P₁P₃ I₂):     {D:.6f}")

    # =========================================================================
    # Compute the 10 monomials
    # From expansion:
    #   p=0: (A-C)(B-C)³ = AB³ - 3AB²C + 3ABC² - AC³ - B³C + 3B²C² - 3BC³ + C⁴
    #   p=1: 3(D-C²)(B-C)² = 3DB² - 6DBC + 3DC² - 3B²C² + 6BC³ - 3C⁴
    # After combining like terms:
    #   AB³: +1
    #   AB²C: -3
    #   ABC²: +3
    #   AC³: -1
    #   B³C: -1 (note the sign!)
    #   BC³: +3 (from -3 + 6)
    #   C⁴: -2 (from +1 - 3)
    #   DB²: +3
    #   DBC: -6
    #   DC²: +3
    # Note: B²C² has coefficient 3-3=0, so it cancels out
    # =========================================================================

    # Compute raw monomial values
    mono_AB3 = A * B * B * B
    mono_AB2C = A * B * B * C
    mono_ABC2 = A * B * C * C
    mono_AC3 = A * C * C * C
    mono_B3C = B * B * B * C
    mono_BC3 = B * C * C * C
    mono_C4 = C * C * C * C
    mono_DB2 = D * B * B
    mono_DBC = D * B * C
    mono_DC2 = D * C * C

    if debug:
        print(f"\nMonomials (before coefficients):")
        print(f"  AB³:   {mono_AB3:.6f} → ×1  = {mono_AB3:.6f}")
        print(f"  AB²C:  {mono_AB2C:.6f} → ×-3 = {-3*mono_AB2C:.6f}")
        print(f"  ABC²:  {mono_ABC2:.6f} → ×3  = {3*mono_ABC2:.6f}")
        print(f"  AC³:   {mono_AC3:.6f} → ×-1 = {-mono_AC3:.6f}")
        print(f"  B³C:   {mono_B3C:.6f} → ×-1 = {-mono_B3C:.6f}")
        print(f"  BC³:   {mono_BC3:.6f} → ×3  = {3*mono_BC3:.6f}")
        print(f"  C⁴:    {mono_C4:.6f} → ×-2 = {-2*mono_C4:.6f}")
        print(f"  DB²:   {mono_DB2:.6f} → ×3  = {3*mono_DB2:.6f}")
        print(f"  DBC:   {mono_DBC:.6f} → ×-6 = {-6*mono_DBC:.6f}")
        print(f"  DC²:   {mono_DC2:.6f} → ×3  = {3*mono_DC2:.6f}")

    # Total with correct signs and coefficients
    total = (
        mono_AB3
        - 3 * mono_AB2C
        + 3 * mono_ABC2
        - mono_AC3
        - mono_B3C
        + 3 * mono_BC3
        - 2 * mono_C4
        + 3 * mono_DB2
        - 6 * mono_DBC
        + 3 * mono_DC2
    )

    if debug:
        print(f"\nTotal Ψ_{{1,3}}: {total:.6f}")

    return OracleResult13(
        AB3=mono_AB3,
        AB2C=mono_AB2C,
        ABC2=mono_ABC2,
        AC3=mono_AC3,
        B3C=mono_B3C,
        BC3=mono_BC3,
        C4=mono_C4,
        DB2=mono_DB2,
        DBC=mono_DBC,
        DC2=mono_DC2,
        total=total
    )


if __name__ == "__main__":
    # Quick test
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

    print("="*60)
    print("Testing Ψ (1,3) Oracle")
    print("="*60)

    theta = 4/7

    # Test with κ polynomials (R=1.3036)
    print("\n--- κ Benchmark (R=1.3036) ---")
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    R_kappa = 1.3036
    result_k = psi_oracle_13(P1_k, P3_k, Q_k, theta, R_kappa, n_quad=80, debug=True)

    # Test with κ* polynomials (R=1.1167)
    print("\n--- κ* Benchmark (R=1.1167) ---")
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    R_kappa_star = 1.1167
    result_ks = psi_oracle_13(P1_ks, P3_ks, Q_ks, theta, R_kappa_star, n_quad=80, debug=True)

    # Compare ratios
    print("\n--- Comparison ---")
    print(f"κ / κ* ratio: {result_k.total / result_ks.total:.4f}")
    print(f"\nExpected ratio from DSL: 5.73")
    print(f"If this Ψ oracle is correct, we should see a MUCH better ratio.")
