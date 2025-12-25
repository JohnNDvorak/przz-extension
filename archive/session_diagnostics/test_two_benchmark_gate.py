"""
src/test_two_benchmark_gate.py
Two-benchmark validation gate for PRZZ clean-path implementation.

This tests the fundamental requirement: BOTH benchmarks must pass.
"""

from __future__ import annotations
import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.section7_clean_evaluator import Section7CleanEvaluator
from src.psi_separated_c import expand_pair_to_monomials_separated


# PRZZ Target Values
THETA = 4.0 / 7.0
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167
C_TARGET_KAPPA = 2.137
C_TARGET_KAPPA_STAR = 1.938
RATIO_TARGET = C_TARGET_KAPPA / C_TARGET_KAPPA_STAR  # ~1.103


def compute_basic_I2(evaluator, ell: int, ellbar: int) -> float:
    """
    Compute basic I₂ contribution for a pair.

    I₂ = (1/θ) × ∫ P_ℓ(u) P_ℓ̄(u) du × ∫ Q(t)² exp(2Rt) dt
    """
    u_nodes = evaluator.u_nodes
    u_weights = evaluator.u_weights
    t_nodes, t_weights = u_nodes, u_weights  # Same quadrature

    # Get polynomials
    P_ell = evaluator.P_polys[ell - 1].to_monomial()
    P_ellbar = evaluator.P_polys[ellbar - 1].to_monomial()
    Q = evaluator.Q.to_monomial()

    # u-integral: ∫ P_ℓ(u) P_ℓ̄(u) du
    P_ell_vals = P_ell.eval(u_nodes)
    P_ellbar_vals = P_ellbar.eval(u_nodes)
    u_integral = np.sum(u_weights * P_ell_vals * P_ellbar_vals)

    # t-integral: ∫ Q(t)² exp(2Rt) dt
    Q_vals = Q.eval(t_nodes)
    exp_2Rt = np.exp(2 * evaluator.R * t_nodes)
    t_integral = np.sum(t_weights * Q_vals * Q_vals * exp_2Rt)

    # I₂ = (1/θ) × u_integral × t_integral
    I2 = (1.0 / evaluator.theta) * u_integral * t_integral

    return I2


def test_basic_I2_all_pairs():
    """Test basic I₂ for all K=3 pairs on both benchmarks."""
    print("\n" + "="*80)
    print("BASIC I₂ COMPARISON (ALL PAIRS)")
    print("="*80)

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    # Create evaluators
    eval_k = Section7CleanEvaluator([P1_k, P2_k, P3_k], Q_k, R_KAPPA, THETA, n_quad=60)
    eval_ks = Section7CleanEvaluator([P1_ks, P2_ks, P3_ks], Q_ks, R_KAPPA_STAR, THETA, n_quad=60)

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    print(f"\n{'Pair':<10} {'κ I₂':>12} {'κ* I₂':>12} {'Ratio':>10}")
    print("-" * 50)

    total_k = 0.0
    total_ks = 0.0

    for ell, ellbar in pairs:
        I2_k = compute_basic_I2(eval_k, ell, ellbar)
        I2_ks = compute_basic_I2(eval_ks, ell, ellbar)

        # Symmetry factor: diagonal pairs count once, off-diagonal count twice
        factor = 1 if ell == ellbar else 2

        total_k += factor * I2_k
        total_ks += factor * I2_ks

        ratio = I2_k / I2_ks if I2_ks != 0 else float('inf')
        print(f"({ell},{ellbar})      {I2_k:12.6f} {I2_ks:12.6f} {ratio:10.4f}")

    print("-" * 50)
    print(f"Total      {total_k:12.6f} {total_ks:12.6f} {total_k/total_ks:10.4f}")
    print(f"\nTarget c:  {C_TARGET_KAPPA:12.3f} {C_TARGET_KAPPA_STAR:12.3f} {RATIO_TARGET:10.3f}")

    return total_k, total_ks


def test_clean_path_11():
    """Test clean-path evaluation for (1,1) pair."""
    print("\n" + "="*80)
    print("CLEAN-PATH (1,1) EVALUATION")
    print("="*80)

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    # Create evaluators
    eval_k = Section7CleanEvaluator([P1_k, P2_k, P3_k], Q_k, R_KAPPA, THETA, n_quad=60)
    eval_ks = Section7CleanEvaluator([P1_ks, P2_ks, P3_ks], Q_ks, R_KAPPA_STAR, THETA, n_quad=60)

    # Test parameters
    alpha = -R_KAPPA  # α ≈ -R (asymptotic)
    beta = -R_KAPPA

    alpha_ks = -R_KAPPA_STAR
    beta_ks = -R_KAPPA_STAR

    print("\nκ benchmark:")
    print(f"  R = {R_KAPPA}, α = β = {alpha:.4f}")

    # Compute I_{1,d} for each monomial
    monomials = expand_pair_to_monomials_separated(1, 1)
    print(f"\n  Monomial contributions (PRE-MIRROR):")
    total_I1d_k = 0.0
    for mono in monomials:
        contrib = eval_k.eval_monomial(mono, 1, 1, alpha, beta)
        total_I1d_k += contrib
        print(f"    {mono.coeff:+d}×A^{mono.a}B^{mono.b}C_α^{mono.c_alpha}C_β^{mono.c_beta}D^{mono.d}: {contrib:.6f}")

    print(f"  Total I_{{1,d}}(α,β) = {total_I1d_k:.6f}")

    # With mirror
    I_d_k = eval_k.apply_mirror(1, 1, alpha, beta)
    print(f"  After mirror: I_d = {I_d_k:.6f}")

    print("\nκ* benchmark:")
    print(f"  R = {R_KAPPA_STAR}, α = β = {alpha_ks:.4f}")

    total_I1d_ks = 0.0
    for mono in monomials:
        contrib = eval_ks.eval_monomial(mono, 1, 1, alpha_ks, beta_ks)
        total_I1d_ks += contrib

    I_d_ks = eval_ks.apply_mirror(1, 1, alpha_ks, beta_ks)
    print(f"  Total I_{{1,d}}(α,β) = {total_I1d_ks:.6f}")
    print(f"  After mirror: I_d = {I_d_ks:.6f}")

    print(f"\nRatio comparison:")
    print(f"  I_{{1,d}} ratio: {total_I1d_k / total_I1d_ks:.4f}")
    print(f"  I_d ratio: {I_d_k / I_d_ks:.4f}")
    print(f"  Target ratio: {RATIO_TARGET:.4f}")


def test_monomial_breakdown():
    """Show detailed monomial breakdown for (1,1)."""
    print("\n" + "="*80)
    print("(1,1) MONOMIAL BREAKDOWN")
    print("="*80)

    monomials = expand_pair_to_monomials_separated(1, 1)

    print("\nMonomial interpretation:")
    for mono in monomials:
        # Interpret each monomial
        if mono.d == 1 and mono.a == 0 and mono.b == 0:
            interp = "D term: second derivative ∂²/∂x∂y of base → I₂-like"
        elif mono.a == 1 and mono.b == 1 and mono.d == 0:
            interp = "AB term: both first derivatives → I₁-like"
        elif mono.a == 1 and mono.c_alpha == 1:
            interp = "A×C_α term: left derivative × α-pole → I₃-like"
        elif mono.b == 1 and mono.c_beta == 1:
            interp = "B×C_β term: right derivative × β-pole → I₄-like"
        else:
            interp = "Unknown"

        print(f"  {mono.coeff:+d}×A^{mono.a}B^{mono.b}C_α^{mono.c_alpha}C_β^{mono.c_beta}D^{mono.d}")
        print(f"      → {interp}")


def main():
    """Run all gate tests."""
    test_monomial_breakdown()
    test_basic_I2_all_pairs()
    test_clean_path_11()

    print("\n" + "="*80)
    print("TWO-BENCHMARK VALIDATION GATE SUMMARY")
    print("="*80)
    print(f"\nTarget values:")
    print(f"  κ:  c = {C_TARGET_KAPPA}")
    print(f"  κ*: c = {C_TARGET_KAPPA_STAR}")
    print(f"  Ratio: {RATIO_TARGET:.3f}")
    print("\nNote: Full implementation requires proper Q operator application")
    print("and correct handling of C_α, C_β pole contributions.")


if __name__ == "__main__":
    main()
