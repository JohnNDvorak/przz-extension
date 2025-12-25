"""
src/debug_structure.py
Debug the relationship between clean-path and POST-MIRROR I-terms.
"""

from __future__ import annotations
import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.quadrature import gauss_legendre_01


def analyze_I2_structure():
    """Analyze the I₂ structure and what the clean-path should compute."""
    print("\n" + "="*80)
    print("I₂ STRUCTURE ANALYSIS")
    print("="*80)

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036

    u_nodes, u_weights = gauss_legendre_01(60)
    t_nodes, t_weights = gauss_legendre_01(60)

    P1_mono = P1.to_monomial()
    Q_mono = Q.to_monomial()

    # Basic integrals
    P1_vals = P1_mono.eval(u_nodes)
    Q_vals = Q_mono.eval(t_nodes)
    exp_2Rt = np.exp(2 * R * t_nodes)

    u_integral = np.sum(u_weights * P1_vals * P1_vals)
    t_integral = np.sum(t_weights * Q_vals * Q_vals * exp_2Rt)

    print("\n1. PRZZ POST-MIRROR I₂ formula:")
    print("   I₂ = (1/θ) × ∫ P_ℓ(u)P_ℓ̄(u) du × ∫ Q(t)² exp(2Rt) dt")
    print()
    print(f"   For (1,1) pair:")
    print(f"   ∫ P₁²du = {u_integral:.6f}")
    print(f"   ∫ Q²exp(2Rt)dt = {t_integral:.6f}")
    print(f"   θ = {theta:.6f}")
    print(f"   I₂ = (1/{theta:.4f}) × {u_integral:.4f} × {t_integral:.4f}")
    print(f"      = {1/theta:.4f} × {u_integral * t_integral:.6f}")
    print(f"      = {(1/theta) * u_integral * t_integral:.6f}")

    # Compare to oracle value
    I2_oracle = (1.0 / theta) * u_integral * t_integral
    print(f"\n   Oracle I₂ = {I2_oracle:.6f}")

    print("\n2. Ψ EXPANSION for (1,1):")
    print("   Ψ_{1,1} = AB - A×C_α - B×C_β + D")
    print()
    print("   This gives 4 monomials, NOT 4 separate integrals!")
    print("   The D and AB monomials contribute to the SAME integral structure.")

    print("\n3. CORRECT INTERPRETATION:")
    print("   For (1,1) at α=β=0:")
    print("   - D monomial (coeff +1): gives ∫P₁²du")
    print("   - AB monomial (coeff +1): gives ∫P₁²du")
    print("   - AC_α monomial (coeff -1): gives 0 (Case A at α=0)")
    print("   - BC_β monomial (coeff -1): gives 0 (Case A at β=0)")
    print()
    print("   BUT: D and AB don't ADD - they represent the SAME derivative structure!")
    print("   The Ψ expansion tracks coefficients, not separate integrals.")

    print("\n4. KEY INSIGHT:")
    print("   The D term represents d²/dxdy at x=y=0")
    print("   The AB term represents d/dx × d/dy at x=y=0")
    print("   These are different contributions to the Taylor coefficient!")
    print()
    print("   d²(fg)/dxdy = f × ∂²g/∂x∂y + ∂f/∂x × ∂g/∂y + ∂f/∂y × ∂g/∂x + ∂²f/∂x∂y × g")
    print()
    print("   For P(x+u)P(y+u) at x=y=0:")
    print("   - ∂²(PP)/∂x∂y = P(u) × 0 + P'(u) × P'(u) + P'(u) × P'(u) + 0 × P(u)")
    print("                 = 2 × (P'(u))²")

    # Compute P'(u)² integral
    P1_prime_vals = P1_mono.eval_deriv(u_nodes, 1)
    u_integral_prime_sq = np.sum(u_weights * P1_prime_vals * P1_prime_vals)
    print(f"\n   ∫ (P₁'(u))² du = {u_integral_prime_sq:.6f}")
    print(f"   2 × ∫ (P₁'(u))² du = {2 * u_integral_prime_sq:.6f}")


def compare_benchmarks():
    """Compare the integral structure between κ and κ* benchmarks."""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)

    # Load both benchmarks
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    theta = 4.0 / 7.0
    R_k = 1.3036
    R_ks = 1.1167

    u_nodes, u_weights = gauss_legendre_01(60)
    t_nodes, t_weights = gauss_legendre_01(60)

    def compute_components(P, Q, R, label):
        P_mono = P.to_monomial()
        Q_mono = Q.to_monomial()

        P_vals = P_mono.eval(u_nodes)
        Q_vals = Q_mono.eval(t_nodes)
        exp_2Rt = np.exp(2 * R * t_nodes)

        u_int = np.sum(u_weights * P_vals * P_vals)
        t_int = np.sum(t_weights * Q_vals * Q_vals * exp_2Rt)

        return u_int, t_int, (1/theta) * u_int * t_int

    print("\n(1,1) pair components:")
    print(f"{'Component':<20} {'κ':>12} {'κ*':>12} {'Ratio':>10}")
    print("-" * 56)

    u_k, t_k, I2_k = compute_components(P1_k, Q_k, R_k, "κ")
    u_ks, t_ks, I2_ks = compute_components(P1_ks, Q_ks, R_ks, "κ*")

    print(f"∫P₁²du             {u_k:12.6f} {u_ks:12.6f} {u_k/u_ks:10.4f}")
    print(f"∫Q²exp(2Rt)dt      {t_k:12.6f} {t_ks:12.6f} {t_k/t_ks:10.4f}")
    print(f"I₂ = (1/θ)×prod    {I2_k:12.6f} {I2_ks:12.6f} {I2_k/I2_ks:10.4f}")

    print("\nTarget ratio: 1.103")
    print(f"I₂ ratio: {I2_k/I2_ks:.4f}")
    print()
    print("NOTE: I₂ ratio (1.20) > target ratio (1.10)")
    print("The derivative terms I₁, I₃, I₄ should bring this DOWN to 1.10")


def analyze_derivative_structure():
    """Analyze how derivatives affect the ratio."""
    print("\n" + "="*80)
    print("DERIVATIVE STRUCTURE")
    print("="*80)

    # Load both benchmarks
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    theta = 4.0 / 7.0
    R_k = 1.3036
    R_ks = 1.1167

    u_nodes, u_weights = gauss_legendre_01(60)
    t_nodes, t_weights = gauss_legendre_01(60)

    def analyze_P(P, Q, R, label):
        P_mono = P.to_monomial()
        Q_mono = Q.to_monomial()

        P_vals = P_mono.eval(u_nodes)
        P_prime = P_mono.eval_deriv(u_nodes, 1)
        Q_vals = Q_mono.eval(t_nodes)
        Q_prime = Q_mono.eval_deriv(t_nodes, 1)

        print(f"\n{label}:")
        print(f"  ∫P² = {np.sum(u_weights * P_vals * P_vals):.6f}")
        print(f"  ∫(P')² = {np.sum(u_weights * P_prime * P_prime):.6f}")
        print(f"  ∫Q² = {np.sum(t_weights * Q_vals * Q_vals):.6f}")
        print(f"  ∫(Q')² = {np.sum(t_weights * Q_prime * Q_prime):.6f}")

        # I₁ structure: d²/dxdy of integrand
        # This involves Q'(t) terms
        exp_2Rt = np.exp(2 * R * t_nodes)

        # Simplified I₁ approximation
        # I₁ ~ (terms with Q derivatives)
        print(f"  ∫Q'² × exp(2Rt) = {np.sum(t_weights * Q_prime**2 * exp_2Rt):.6f}")

    analyze_P(P1_k.to_monomial(), Q_k.to_monomial(), R_k, "κ benchmark")
    analyze_P(P1_ks.to_monomial(), Q_ks.to_monomial(), R_ks, "κ* benchmark")


def main():
    analyze_I2_structure()
    compare_benchmarks()
    analyze_derivative_structure()


if __name__ == "__main__":
    main()
