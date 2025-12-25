"""
src/debug_mirror_term.py
Debug the mirror term sign issue in the clean-path implementation.
"""

from __future__ import annotations
import numpy as np
from src.polynomials import load_przz_polynomials
from src.section7_clean_evaluator import Section7CleanEvaluator
from src.psi_separated_c import expand_pair_to_monomials_separated


def debug_alpha_sweep():
    """Sweep α values to understand the mirror term behavior."""
    print("\n" + "="*80)
    print("α PARAMETER SWEEP FOR (1,1)")
    print("="*80)

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036

    evaluator = Section7CleanEvaluator([P1, P2, P3], Q, R, theta, n_quad=60)

    # Test different α values
    alphas = [0.0, -0.1, -0.5, -1.0, -R]

    print(f"\nα = β sweep (R={R}):")
    print(f"Mirror factor = exp(2R) = {np.exp(2*R):.4f}")
    print()
    print(f"{'α':>8} {'I_{1,d}(α,β)':>15} {'I_{1,d}(-β,-α)':>15} {'Mirror contrib':>15} {'I_d (total)':>15}")
    print("-" * 75)

    for alpha in alphas:
        beta = alpha  # Symmetric case

        # PRE-MIRROR I_{1,d}(α, β)
        I1d_direct = evaluator.compute_I1d_pair(1, 1, alpha, beta)

        # PRE-MIRROR I_{1,d}(-β, -α)
        I1d_mirror = evaluator.compute_I1d_pair(1, 1, -beta, -alpha)

        # Mirror contribution
        mirror_factor = np.exp(2 * R)
        mirror_contrib = mirror_factor * I1d_mirror

        # Total
        I_d = I1d_direct + mirror_contrib

        print(f"{alpha:8.4f} {I1d_direct:15.6f} {I1d_mirror:15.6f} {mirror_contrib:15.6f} {I_d:15.6f}")


def debug_monomial_contributions():
    """Debug individual monomial contributions at different α."""
    print("\n" + "="*80)
    print("MONOMIAL CONTRIBUTIONS AT α = 0 vs α = -R")
    print("="*80)

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036

    evaluator = Section7CleanEvaluator([P1, P2, P3], Q, R, theta, n_quad=60)

    monomials = expand_pair_to_monomials_separated(1, 1)

    for alpha, label in [(0.0, "α=0"), (-R, "α=-R")]:
        beta = alpha
        print(f"\n{label} (β={beta:.4f}):")
        total = 0.0
        for mono in monomials:
            contrib = evaluator.eval_monomial(mono, 1, 1, alpha, beta)
            total += contrib
            print(f"  {mono.coeff:+d}×A^{mono.a}B^{mono.b}C_α^{mono.c_alpha}C_β^{mono.c_beta}D^{mono.d}: {contrib:.6f}")
        print(f"  Total: {total:.6f}")


def compare_to_oracle():
    """Compare clean-path I₂ against oracle values."""
    print("\n" + "="*80)
    print("COMPARISON TO PRZZ ORACLE")
    print("="*80)

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036

    evaluator = Section7CleanEvaluator([P1, P2, P3], Q, R, theta, n_quad=60)

    # Oracle I₂ formula (POST-MIRROR):
    # I₂ = (1/θ) × ∫ P₁² du × ∫ Q² exp(2Rt) dt

    u_nodes = evaluator.u_nodes
    u_weights = evaluator.u_weights
    t_nodes, t_weights = u_nodes, u_weights

    P1_mono = P1.to_monomial()
    Q_mono = Q.to_monomial()

    # u-integral
    P1_vals = P1_mono.eval(u_nodes)
    u_integral = np.sum(u_weights * P1_vals * P1_vals)

    # t-integral
    Q_vals = Q_mono.eval(t_nodes)
    exp_2Rt = np.exp(2 * R * t_nodes)
    t_integral = np.sum(t_weights * Q_vals * Q_vals * exp_2Rt)

    # Oracle I₂
    I2_oracle = (1.0 / theta) * u_integral * t_integral

    print(f"\nOracle (POST-MIRROR) computation:")
    print(f"  ∫ P₁² du = {u_integral:.6f}")
    print(f"  ∫ Q² exp(2Rt) dt = {t_integral:.6f}")
    print(f"  I₂ = (1/θ) × {u_integral:.4f} × {t_integral:.4f} = {I2_oracle:.6f}")

    # Clean-path at α = 0
    print(f"\nClean-path at α = β = 0:")
    I1d_0 = evaluator.compute_I1d_pair(1, 1, 0.0, 0.0)
    print(f"  I_{{1,d}}(0, 0) = {I1d_0:.6f}")

    # What about the mirror at α = 0?
    # I_{1,d}(-0, -0) = I_{1,d}(0, 0) (same!)
    I1d_mirror_0 = evaluator.compute_I1d_pair(1, 1, 0.0, 0.0)
    mirror_factor = np.exp(2 * R)
    I_d_0 = I1d_0 + mirror_factor * I1d_mirror_0

    print(f"  Mirror: I_{{1,d}}(0, 0) = {I1d_mirror_0:.6f}")
    print(f"  Mirror factor: exp(2R) = {mirror_factor:.4f}")
    print(f"  I_d = I_{{1,d}} + exp(2R)×I_{{1,d}} = (1 + exp(2R))×I_{{1,d}} = {I_d_0:.6f}")

    print(f"\nComparison:")
    print(f"  Oracle I₂: {I2_oracle:.6f}")
    print(f"  Clean-path I_d (α=0): {I_d_0:.6f}")
    print(f"  Ratio: {I_d_0/I2_oracle:.4f}")

    # What factor would make them equal?
    needed_factor = I2_oracle / I_d_0
    print(f"\nNeeded scaling factor: {needed_factor:.6f}")


def analyze_case_b_structure():
    """Analyze Case B structure for (1,1)."""
    print("\n" + "="*80)
    print("CASE B STRUCTURE ANALYSIS")
    print("="*80)

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036

    evaluator = Section7CleanEvaluator([P1, P2, P3], Q, R, theta, n_quad=60)

    u = evaluator.u_nodes
    w = evaluator.u_weights

    P1_mono = P1.to_monomial()

    # For (1,1), all monomials have (l1=0 or 1) and (m1=0 or 1)
    # Case B (l=1): F = P(u)
    # Case A (l=0): F = α × P(u)

    # Let's trace each monomial:
    # D (a=0,b=0,d=1): l1=1, m1=1 → Case B,B
    # AB (a=1,b=1,d=0): l1=1, m1=1 → Case B,B
    # AC_α (a=1,c_α=1): l1=1, m1=0 → Case B,A
    # BC_β (b=1,c_β=1): l1=0, m1=1 → Case A,B

    print("\nMonomial → Case mapping for (1,1):")
    monomials = expand_pair_to_monomials_separated(1, 1)
    for mono in monomials:
        l1 = mono.a + mono.d
        m1 = mono.b + mono.d
        case_l = "B" if l1 == 1 else "A"
        case_r = "B" if m1 == 1 else "A"
        print(f"  {mono.coeff:+d}×A^{mono.a}B^{mono.b}C_α^{mono.c_alpha}C_β^{mono.c_beta}D^{mono.d}")
        print(f"      l1={l1}, m1={m1} → ({case_l},{case_r})")

    print("\nF_d evaluations at α=0:")
    # Case B: F = P(u)
    F_B = evaluator.Fd_case_B(u, P1_mono)
    print(f"  Case B: F = P(u), ∫F² du = {np.sum(w * F_B * F_B):.6f}")

    # Case A at α=0: F = 0
    F_A = evaluator.Fd_case_A(u, 0.0, P1_mono)
    print(f"  Case A: F = 0×P(u) = 0, ∫F² du = {np.sum(w * F_A * F_A):.6f}")

    # So at α=0:
    # D term: F_B × F_B, contributes ∫P² du
    # AB term: F_B × F_B, contributes ∫P² du
    # AC_α term: F_B × F_A = 0
    # BC_β term: F_A × F_B = 0

    print("\nInterpretation at α=β=0:")
    print("  D and AB monomials: non-zero (Case B × Case B)")
    print("  AC_α and BC_β monomials: zero (Case A gives 0)")
    print("  Total I_{1,d} = 2 × ∫P₁²du = {:.6f}".format(2 * np.sum(w * F_B * F_B)))


def main():
    debug_alpha_sweep()
    debug_monomial_contributions()
    compare_to_oracle()
    analyze_case_b_structure()


if __name__ == "__main__":
    main()
