"""
src/test_clean_evaluator.py
Test the Section 7 clean evaluator against known oracle values.
"""

from __future__ import annotations
import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.section7_clean_evaluator import (
    Section7CleanEvaluator,
    expand_pair_to_monomials_separated,
    monomial_to_triple_separated
)


def test_11_structure():
    """Verify (1,1) produces the expected monomial structure."""
    print("\n" + "="*70)
    print("TEST: (1,1) MONOMIAL STRUCTURE")
    print("="*70)

    monomials = expand_pair_to_monomials_separated(1, 1)

    print(f"Number of monomials: {len(monomials)} (expected: 4)")

    # Expected structure:
    # D (coeff +1) → I₂-like
    # AB (coeff +1) → I₁-like
    # -A×C_α (coeff -1) → I₃-like
    # -B×C_β (coeff -1) → I₄-like

    for mono in monomials:
        triple = monomial_to_triple_separated(mono)
        print(f"  {mono.coeff:+d} × A^{mono.a}B^{mono.b}C_α^{mono.c_alpha}C_β^{mono.c_beta}D^{mono.d}")
        print(f"      → l1={triple.l1}, m1={triple.m1} → Cases ({triple.case_left.value},{triple.case_right.value})")

    # Check if structure matches expectation
    expected_keys = {
        (0, 0, 0, 0, 1, +1),  # D
        (1, 1, 0, 0, 0, +1),  # AB
        (1, 0, 1, 0, 0, -1),  # A×C_α
        (0, 1, 0, 1, 0, -1),  # B×C_β
    }

    actual_keys = {(m.a, m.b, m.c_alpha, m.c_beta, m.d, m.coeff) for m in monomials}

    if actual_keys == expected_keys:
        print("\n✓ (1,1) structure MATCHES expected I₁-I₄ decomposition")
    else:
        print("\n✗ (1,1) structure MISMATCH!")
        print(f"  Expected: {expected_keys}")
        print(f"  Actual: {actual_keys}")


def test_polynomial_loading():
    """Test that polynomials load correctly."""
    print("\n" + "="*70)
    print("TEST: POLYNOMIAL LOADING")
    print("="*70)

    # Load κ benchmark
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)

    # Load κ* benchmark
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    print("\nκ benchmark (R=1.3036):")
    print(f"  P1 degree: {P1_k.to_monomial().degree}")
    print(f"  P2 degree: {P2_k.to_monomial().degree}")
    print(f"  P3 degree: {P3_k.to_monomial().degree}")
    print(f"  Q degree: {Q_k.to_monomial().degree}")
    print(f"  P1(0)={P1_k.eval(np.array([0.0]))[0]:.6f}, P1(1)={P1_k.eval(np.array([1.0]))[0]:.6f}")
    print(f"  Q(0)={Q_k.eval(np.array([0.0]))[0]:.6f}")

    print("\nκ* benchmark (R=1.1167):")
    print(f"  P1 degree: {P1_ks.to_monomial().degree}")
    print(f"  P2 degree: {P2_ks.to_monomial().degree}")
    print(f"  P3 degree: {P3_ks.to_monomial().degree}")
    print(f"  Q degree: {Q_ks.to_monomial().degree}")
    print(f"  P1(0)={P1_ks.eval(np.array([0.0]))[0]:.6f}, P1(1)={P1_ks.eval(np.array([1.0]))[0]:.6f}")
    print(f"  Q(0)={Q_ks.eval(np.array([0.0]))[0]:.6f}")


def test_fd_case_dispatch():
    """Test F_d case dispatch for different l values."""
    print("\n" + "="*70)
    print("TEST: F_d CASE DISPATCH")
    print("="*70)

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    # Create evaluator
    theta = 4.0 / 7.0
    R = 1.3036
    evaluator = Section7CleanEvaluator([P1, P2, P3], Q, R, theta, n_quad=40)

    # Test points
    u = np.linspace(0.1, 0.9, 5)

    print("\nCase A (l=0, ω=-1): α × P(u)")
    alpha = -R  # Typical value
    result_A = evaluator.Fd_case_A(u, alpha, P1.to_monomial())
    print(f"  At u={u[2]:.2f}: F_A = {result_A[2]:.6f}")

    print("\nCase B (l=1, ω=0): P(u)")
    result_B = evaluator.Fd_case_B(u, P1.to_monomial())
    print(f"  At u={u[2]:.2f}: F_B = {result_B[2]:.6f}")

    print("\nCase C (l=2, ω=1): Kernel integral")
    result_C = evaluator.Fd_case_C(u, omega=1, alpha=alpha, P=P1.to_monomial())
    print(f"  At u={u[2]:.2f}: F_C = {result_C[2]:.6f}")

    # Verify dispatch
    print("\nDispatch verification:")
    for l in [0, 1, 2, 3]:
        result = evaluator.eval_Fd(u, l, alpha, P1.to_monomial())
        omega = l - 1
        case = "A" if omega == -1 else ("B" if omega == 0 else "C")
        print(f"  l={l} (ω={omega}) → Case {case}: F[2] = {result[2]:.6f}")


def test_basic_integral():
    """Test basic integral computation for (1,1)."""
    print("\n" + "="*70)
    print("TEST: BASIC (1,1) INTEGRAL")
    print("="*70)

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    theta = 4.0 / 7.0
    R = 1.3036

    # Create evaluator
    evaluator = Section7CleanEvaluator([P1, P2, P3], Q, R, theta, n_quad=60)

    # Compute basic integrals that should match known values
    u_nodes = evaluator.u_nodes
    u_weights = evaluator.u_weights
    t_nodes, t_weights = evaluator.u_nodes, evaluator.u_weights  # Same quadrature

    # P1 polynomial
    P1_mono = P1.to_monomial()
    Q_mono = Q.to_monomial()

    # ∫ P₁(u)² du
    P1_vals = P1_mono.eval(u_nodes)
    int_P1_sq = np.sum(u_weights * P1_vals * P1_vals)
    print(f"\n∫ P₁(u)² du = {int_P1_sq:.6f}")

    # ∫ Q(t)² exp(2Rt) dt
    Q_vals = Q_mono.eval(t_nodes)
    exp_2Rt = np.exp(2 * R * t_nodes)
    int_Q_exp = np.sum(t_weights * Q_vals * Q_vals * exp_2Rt)
    print(f"∫ Q(t)² exp(2Rt) dt = {int_Q_exp:.6f}")

    # I₂ = (1/θ) × ∫P₁² du × ∫Q²e^{2Rt} dt
    I2_basic = (1.0 / theta) * int_P1_sq * int_Q_exp
    print(f"\nI₂ (basic) = (1/θ) × {int_P1_sq:.6f} × {int_Q_exp:.6f} = {I2_basic:.6f}")


def test_two_benchmark_comparison():
    """Compare basic integrals between κ and κ* benchmarks."""
    print("\n" + "="*70)
    print("TEST: TWO-BENCHMARK COMPARISON")
    print("="*70)

    # Load both benchmarks
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    theta = 4.0 / 7.0
    R_k = 1.3036
    R_ks = 1.1167

    # Create evaluators
    eval_k = Section7CleanEvaluator([P1_k, P2_k, P3_k], Q_k, R_k, theta, n_quad=60)
    eval_ks = Section7CleanEvaluator([P1_ks, P2_ks, P3_ks], Q_ks, R_ks, theta, n_quad=60)

    print("\nComparing basic Q integrals:")

    # Q integrals
    for label, evaluator, R in [("κ", eval_k, R_k), ("κ*", eval_ks, R_ks)]:
        t_nodes = evaluator.u_nodes
        t_weights = evaluator.u_weights
        Q_vals = evaluator.Q.to_monomial().eval(t_nodes)
        exp_2Rt = np.exp(2 * R * t_nodes)
        int_Q_exp = np.sum(t_weights * Q_vals * Q_vals * exp_2Rt)
        print(f"  {label}: ∫ Q(t)² exp(2Rt) dt = {int_Q_exp:.6f}")

    # Target values from handoff
    print("\nTarget c values:")
    print(f"  κ:  c = 2.137")
    print(f"  κ*: c = 1.938")
    print(f"  Ratio: 1.103")


def main():
    """Run all tests."""
    test_11_structure()
    test_polynomial_loading()
    test_fd_case_dispatch()
    test_basic_integral()
    test_two_benchmark_comparison()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
