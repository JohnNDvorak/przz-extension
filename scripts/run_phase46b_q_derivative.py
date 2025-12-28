#!/usr/bin/env python3
"""
scripts/run_phase46b_q_derivative.py
Phase 46B: Analyze Q-derivative contributions to epsilon

This script computes the Q-derivative epsilon corrections from first principles
and compares them to the calibrated values.

Created: 2025-12-27 (Phase 46B)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.g_q_functional import (
    compute_q_derivative_epsilon,
    compute_full_q_derivative_analysis,
    compare_to_calibrated,
)


def main():
    print("=" * 70)
    print("PHASE 46B: Q-DERIVATIVE EPSILON DERIVATION")
    print("=" * 70)

    theta = 4/7
    K = 3
    g_baseline = 1 + theta / (2 * K * (2 * K + 1))

    print(f"\nParameters:")
    print(f"  theta = {theta:.10f}")
    print(f"  K = {K}")
    print(f"  g_baseline = {g_baseline:.10f}")

    # Calibrated values (from Phase 45)
    g_I1_calibrated = 1.00091428
    g_I2_calibrated = 1.01945154

    print(f"\nCalibrated g values (for comparison):")
    print(f"  g_I1_calibrated = {g_I1_calibrated:.8f}")
    print(f"  g_I2_calibrated = {g_I2_calibrated:.8f}")

    print("\n" + "=" * 70)
    print("KAPPA BENCHMARK (R = 1.3036)")
    print("=" * 70)

    _, _, _, Q_kappa = load_przz_polynomials(enforce_Q0=False)
    Q_mono = Q_kappa.to_monomial()
    R_kappa = 1.3036

    # Compute epsilon from Q-derivative
    epsilon_result = compute_q_derivative_epsilon(Q_mono, theta, R_kappa, K)

    print(f"\nQ-derivative analysis:")
    print(f"  frozen_Q2_minus = {epsilon_result.frozen_Q2_minus:.6f}")
    print(f"  xy_deriv_minus  = {epsilon_result.xy_deriv_minus:.6f}")
    print(f"  raw ratio       = {epsilon_result.xy_deriv_minus / epsilon_result.frozen_Q2_minus:.6f}")

    print(f"\nDerived epsilon values:")
    print(f"  epsilon_I1_Q = {epsilon_result.epsilon_I1_Q:.8f}")
    print(f"  epsilon_I2_Q = {epsilon_result.epsilon_I2_Q:.8f}")

    print(f"\nDerived g values:")
    print(f"  g_I1_corrected = {epsilon_result.g_I1_corrected:.8f}")
    print(f"  g_I2_corrected = {epsilon_result.g_I2_corrected:.8f}")

    # Compare to calibrated
    comparison = compare_to_calibrated(epsilon_result)

    print(f"\nComparison to calibrated:")
    print(f"  epsilon_I1: derived={comparison['epsilon_I1_derived']:.6f}, "
          f"calibrated={comparison['epsilon_I1_calibrated']:.6f}, "
          f"gap={comparison['epsilon_I1_gap']:.6f}")
    print(f"  epsilon_I2: derived={comparison['epsilon_I2_derived']:.6f}, "
          f"calibrated={comparison['epsilon_I2_calibrated']:.6f}, "
          f"gap={comparison['epsilon_I2_gap']:.6f}")

    # Full analysis
    full_analysis = compute_full_q_derivative_analysis(Q_mono, theta, R_kappa, K)
    print(f"\nDetailed Q moments:")
    print(f"  At +R: Q*Q'' = {full_analysis['Q_Qpp_plus']:.6f}, (Q')² = {full_analysis['Qp2_plus']:.6f}")
    print(f"  At -R: Q*Q'' = {full_analysis['Q_Qpp_minus']:.6f}, (Q')² = {full_analysis['Qp2_minus']:.6f}")

    print("\n" + "=" * 70)
    print("KAPPA* BENCHMARK (R = 1.1167)")
    print("=" * 70)

    _, _, _, Q_kappa_star = load_przz_polynomials_kappa_star(enforce_Q0=False)
    Q_star_mono = Q_kappa_star.to_monomial()
    R_star = 1.1167

    # Compute epsilon from Q-derivative
    epsilon_star = compute_q_derivative_epsilon(Q_star_mono, theta, R_star, K)

    print(f"\nQ-derivative analysis:")
    print(f"  frozen_Q2_minus = {epsilon_star.frozen_Q2_minus:.6f}")
    print(f"  xy_deriv_minus  = {epsilon_star.xy_deriv_minus:.6f}")
    print(f"  raw ratio       = {epsilon_star.xy_deriv_minus / epsilon_star.frozen_Q2_minus:.6f}")

    print(f"\nDerived epsilon values:")
    print(f"  epsilon_I1_Q = {epsilon_star.epsilon_I1_Q:.8f}")
    print(f"  epsilon_I2_Q = {epsilon_star.epsilon_I2_Q:.8f}")

    print(f"\nDerived g values:")
    print(f"  g_I1_corrected = {epsilon_star.g_I1_corrected:.8f}")
    print(f"  g_I2_corrected = {epsilon_star.g_I2_corrected:.8f}")

    # Compare to calibrated
    comparison_star = compare_to_calibrated(epsilon_star)

    print(f"\nComparison to calibrated:")
    print(f"  epsilon_I1: derived={comparison_star['epsilon_I1_derived']:.6f}, "
          f"calibrated={comparison_star['epsilon_I1_calibrated']:.6f}, "
          f"gap={comparison_star['epsilon_I1_gap']:.6f}")
    print(f"  epsilon_I2: derived={comparison_star['epsilon_I2_derived']:.6f}, "
          f"calibrated={comparison_star['epsilon_I2_calibrated']:.6f}, "
          f"gap={comparison_star['epsilon_I2_gap']:.6f}")

    # Full analysis
    full_star = compute_full_q_derivative_analysis(Q_star_mono, theta, R_star, K)
    print(f"\nDetailed Q moments:")
    print(f"  At +R: Q*Q'' = {full_star['Q_Qpp_plus']:.6f}, (Q')² = {full_star['Qp2_plus']:.6f}")
    print(f"  At -R: Q*Q'' = {full_star['Q_Qpp_minus']:.6f}, (Q')² = {full_star['Qp2_minus']:.6f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nThe Q-derivative raw ratio measures the relative size of Q' and Q''")
    print("contributions to the [xy] coefficient extraction.")
    print("")
    print("Key finding:")
    print(f"  kappa:  raw ratio = {epsilon_result.xy_deriv_minus / epsilon_result.frozen_Q2_minus:.6f}")
    print(f"  kappa*: raw ratio = {epsilon_star.xy_deriv_minus / epsilon_star.frozen_Q2_minus:.6f}")
    print("")
    print("These are the Q-derivative epsilon contributions to g_I1.")
    print("For g_I2, since Q is frozen at Q(t)², epsilon_I2_Q = 0 by construction.")
    print("")
    print("The calibrated epsilon_I2 ≈ 0.006 must come from something ELSE -")
    print("likely the interaction between log factor and Beta moment in I2.")


if __name__ == "__main__":
    main()
