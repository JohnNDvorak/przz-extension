"""
Ψ-expansion oracle for the (2,3) cross-pair.

This implements the full Ψ combinatorial expansion for μ⋆Λ × μ⋆Λ⋆Λ (piece 2 × piece 3).

The Ψ formula for (ℓ=2, ℓ̄=3):
    Ψ_{2,3}(A,B,C,D) = Σ_{p=0}^{min(2,3)} C(2,p)C(3,p)p! × (D-C²)^p × (A-C)^{2-p} × (B-C)^{3-p}

After expansion and simplification, this gives 18 monomials:
    p=0: (A-C)²(B-C)³ → expands to 8 monomials
    p=1: 6(D-C²)(A-C)(B-C)² → expands to 8 monomials
    p=2: 6(D-C²)²(B-C) → expands to 4 monomials

Total: 18 unique monomials after combining like terms.

INTERPRETATION OF A, B, C, D:
Based on the (1,1) validated mapping:
    - D corresponds to I₂-type integrals (no derivatives)
    - A, B, C correspond to derivative-weighted integrals
    - The signs and combinatorics encode the PRZZ derivative extraction structure

For the (2,3) pair:
    - P₂ is the μ⋆Λ piece (ℓ=2)
    - P₃ is the μ⋆Λ⋆Λ piece (ℓ=3)

IMPORTANT: P₃ changes sign on [0,1], so ∫P₂P₃ can be negative.
From HANDOFF_SUMMARY: (2,3) ratio was 9.04 in old DSL.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, NamedTuple


class OracleResult23(NamedTuple):
    """Result of (2,3) oracle computation via Ψ expansion."""
    # Individual monomial contributions (raw values)
    A2B3: float
    A2B2C: float
    A2BC2: float
    A2C3: float
    AB3C: float
    AB2C2: float
    ABC3: float
    AC4: float
    B3C2: float
    B2C3: float
    BC4: float
    C5: float
    DAB2: float
    DABC: float
    DAC2: float
    DB2C: float
    DBC2: float
    DC3: float
    D2B: float
    D2C: float
    # Total
    total: float


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def psi_oracle_23(
    P2,  # P₂ polynomial (piece 2 = μ⋆Λ)
    P3,  # P₃ polynomial (piece 3 = μ⋆Λ⋆Λ)
    Q,   # Q polynomial
    theta: float,
    R: float,
    n_quad: int = 60,
    debug: bool = False
) -> OracleResult23:
    """
    Compute the (2,3) contribution using full Ψ expansion (18 monomials).

    This computes:
        Ψ_{2,3} = (A-C)²(B-C)³ + 6(D-C²)(A-C)(B-C)² + 6(D-C²)²(B-C)

    where A, B, C, D are fundamental integrals involving P₂, P₃, Q.

    Args:
        P2: The P₂ polynomial (piece 2 = μ⋆Λ)
        P3: The P₃ polynomial (piece 3 = μ⋆Λ⋆Λ)
        Q: The Q polynomial
        theta: θ = 4/7
        R: R parameter
        n_quad: Quadrature points
        debug: Print debug info

    Returns:
        OracleResult23 with individual monomial values and total
    """
    # Set up quadrature
    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # Precompute polynomial values at u-nodes
    P2_u = P2.eval(u_nodes)
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
    #     A = (1/θ) × ∫ P₂(u) du × ∫ Q(t)² exp(2Rt) dt
    #     B = (1/θ) × ∫ P₃(u) du × ∫ Q(t)² exp(2Rt) dt
    #     C = (1/θ) × ∫ 1 du × ∫ Q(t)² exp(2Rt) dt
    #     D = (1/θ) × ∫ P₂(u)P₃(u) du × ∫ Q(t)² exp(2Rt) dt
    # =========================================================================

    # Prefactor from PRZZ formulas
    prefactor = 1.0 / theta

    # Common factor: ∫ Q(t)² exp(2Rt) dt
    t_int = t_integral

    # u-integrals
    u_int_P2 = np.sum(u_weights * P2_u)
    u_int_P3 = np.sum(u_weights * P3_u)
    u_int_P2P3 = np.sum(u_weights * P2_u * P3_u)
    u_int_1 = np.sum(u_weights)  # = 1.0

    # A = (1/θ) × ∫ P₂(u) du × ∫ Q(t)² exp(2Rt) dt
    A = prefactor * u_int_P2 * t_int

    # B = (1/θ) × ∫ P₃(u) du × ∫ Q(t)² exp(2Rt) dt
    B = prefactor * u_int_P3 * t_int

    # C = (1/θ) × ∫ 1 du × ∫ Q(t)² exp(2Rt) dt = (1/θ) × t_int
    C = prefactor * u_int_1 * t_int

    # D = (1/θ) × ∫ P₂(u)P₃(u) du × ∫ Q(t)² exp(2Rt) dt
    # This is the I₂-type integral for the (2,3) pair
    D = prefactor * u_int_P2P3 * t_int

    if debug:
        print(f"\nBuilding blocks:")
        print(f"  u-integrals:")
        print(f"    ∫ P₂(u) du:      {u_int_P2:.6f}")
        print(f"    ∫ P₃(u) du:      {u_int_P3:.6f}")
        print(f"    ∫ P₂P₃ du:       {u_int_P2P3:.6f}")
        print(f"  t-integral:")
        print(f"    ∫ Q²e^{{2Rt}} dt:  {t_int:.6f}")
        print(f"  Combined (with 1/θ prefactor):")
        print(f"    A (P₂ factor):   {A:.6f}")
        print(f"    B (P₃ factor):   {B:.6f}")
        print(f"    C (constant):    {C:.6f}")
        print(f"    D (P₂P₃ I₂):     {D:.6f}")

    # =========================================================================
    # Compute the 18 monomials
    #
    # From the monomial expansion, we get these 18 unique monomials:
    # (using the coefficients from expand_pair_to_monomials(2, 3))
    # =========================================================================

    # Compute raw monomial values
    mono_A2B3 = A * A * B * B * B
    mono_A2B2C = A * A * B * B * C
    mono_A2BC2 = A * A * B * C * C
    mono_A2C3 = A * A * C * C * C
    mono_AB3C = A * B * B * B * C
    mono_AB2C2 = A * B * B * C * C
    mono_ABC3 = A * B * C * C * C
    mono_AC4 = A * C * C * C * C
    mono_B3C2 = B * B * B * C * C
    mono_B2C3 = B * B * C * C * C
    mono_BC4 = B * C * C * C * C
    mono_C5 = C * C * C * C * C
    mono_DAB2 = D * A * B * B
    mono_DABC = D * A * B * C
    mono_DAC2 = D * A * C * C
    mono_DB2C = D * B * B * C
    mono_DBC2 = D * B * C * C
    mono_DC3 = D * C * C * C
    mono_D2B = D * D * B
    mono_D2C = D * D * C

    # Get the actual coefficients from the monomial expansion
    # These match the pattern from expand_pair_to_monomials(2, 3)
    # p=0: (A-C)²(B-C)³
    # p=1: 6(D-C²)(A-C)(B-C)²
    # p=2: 6(D-C²)²(B-C)

    # The total is the sum with correct coefficients
    total = (
        # From p=0: (A-C)²(B-C)³
        1 * mono_A2B3 +
        -3 * mono_A2B2C +
        3 * mono_A2BC2 +
        -1 * mono_A2C3 +
        -2 * mono_AB3C +
        6 * mono_AB2C2 +
        -6 * mono_ABC3 +
        2 * mono_AC4 +
        1 * mono_B3C2 +
        -3 * mono_B2C3 +
        3 * mono_BC4 +
        -1 * mono_C5 +
        # From p=1: 6(D-C²)(A-C)(B-C)²
        6 * mono_DAB2 +
        -12 * mono_DABC +
        6 * mono_DAC2 +
        -6 * mono_DB2C +
        12 * mono_DBC2 +
        -6 * mono_DC3 +
        # From p=2: 6(D-C²)²(B-C)
        6 * mono_D2B +
        -6 * mono_D2C
    )

    if debug:
        print(f"\nMonomials (with coefficients applied):")
        print(f"  A²B³:   {mono_A2B3:.6f} → ×1  = {1*mono_A2B3:.6f}")
        print(f"  A²B²C:  {mono_A2B2C:.6f} → ×-3 = {-3*mono_A2B2C:.6f}")
        print(f"  A²BC²:  {mono_A2BC2:.6f} → ×3  = {3*mono_A2BC2:.6f}")
        print(f"  A²C³:   {mono_A2C3:.6f} → ×-1 = {-1*mono_A2C3:.6f}")
        print(f"  AB³C:   {mono_AB3C:.6f} → ×-2 = {-2*mono_AB3C:.6f}")
        print(f"  AB²C²:  {mono_AB2C2:.6f} → ×6  = {6*mono_AB2C2:.6f}")
        print(f"  ABC³:   {mono_ABC3:.6f} → ×-6 = {-6*mono_ABC3:.6f}")
        print(f"  AC⁴:    {mono_AC4:.6f} → ×2  = {2*mono_AC4:.6f}")
        print(f"  B³C²:   {mono_B3C2:.6f} → ×1  = {1*mono_B3C2:.6f}")
        print(f"  B²C³:   {mono_B2C3:.6f} → ×-3 = {-3*mono_B2C3:.6f}")
        print(f"  BC⁴:    {mono_BC4:.6f} → ×3  = {3*mono_BC4:.6f}")
        print(f"  C⁵:     {mono_C5:.6f} → ×-1 = {-1*mono_C5:.6f}")
        print(f"  DAB²:   {mono_DAB2:.6f} → ×6  = {6*mono_DAB2:.6f}")
        print(f"  DABC:   {mono_DABC:.6f} → ×-12 = {-12*mono_DABC:.6f}")
        print(f"  DAC²:   {mono_DAC2:.6f} → ×6  = {6*mono_DAC2:.6f}")
        print(f"  DB²C:   {mono_DB2C:.6f} → ×-6 = {-6*mono_DB2C:.6f}")
        print(f"  DBC²:   {mono_DBC2:.6f} → ×12 = {12*mono_DBC2:.6f}")
        print(f"  DC³:    {mono_DC3:.6f} → ×-6 = {-6*mono_DC3:.6f}")
        print(f"  D²B:    {mono_D2B:.6f} → ×6  = {6*mono_D2B:.6f}")
        print(f"  D²C:    {mono_D2C:.6f} → ×-6 = {-6*mono_D2C:.6f}")
        print(f"\nTotal Ψ_{{2,3}}: {total:.6f}")

    return OracleResult23(
        A2B3=mono_A2B3,
        A2B2C=mono_A2B2C,
        A2BC2=mono_A2BC2,
        A2C3=mono_A2C3,
        AB3C=mono_AB3C,
        AB2C2=mono_AB2C2,
        ABC3=mono_ABC3,
        AC4=mono_AC4,
        B3C2=mono_B3C2,
        B2C3=mono_B2C3,
        BC4=mono_BC4,
        C5=mono_C5,
        DAB2=mono_DAB2,
        DABC=mono_DABC,
        DAC2=mono_DAC2,
        DB2C=mono_DB2C,
        DBC2=mono_DBC2,
        DC3=mono_DC3,
        D2B=mono_D2B,
        D2C=mono_D2C,
        total=total
    )


if __name__ == "__main__":
    # Quick test
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

    print("="*60)
    print("Testing Ψ (2,3) Oracle")
    print("="*60)

    theta = 4/7

    # Test with κ polynomials (R=1.3036)
    print("\n--- κ Benchmark (R=1.3036) ---")
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    R_kappa = 1.3036
    result_k = psi_oracle_23(P2_k, P3_k, Q_k, theta, R_kappa, n_quad=80, debug=True)

    # Test with κ* polynomials (R=1.1167)
    print("\n--- κ* Benchmark (R=1.1167) ---")
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    R_kappa_star = 1.1167
    result_ks = psi_oracle_23(P2_ks, P3_ks, Q_ks, theta, R_kappa_star, n_quad=80, debug=True)

    # Compare ratios
    print("\n--- Comparison ---")
    print(f"κ / κ* ratio: {result_k.total / result_ks.total:.4f}")
    print(f"\nExpected ratio from DSL: 9.04")
    print(f"If this Ψ oracle is correct, we should see a MUCH better ratio.")
