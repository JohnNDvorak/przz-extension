#!/usr/bin/env python3
"""
scripts/run_phase12_integrand_probe.py
Phase 12.2: Integrand-Level Diagnosis

Compares empirical vs Phase 10 vs Phase 12 mirror integrands at specific (u,t) nodes.
If the ratio is roughly constant → hunting a missing scalar normalization.
If the ratio varies strongly → hunting missing functional structure.
"""

import numpy as np
from src.polynomials import load_przz_polynomials
from src.series import TruncatedSeries
from src.composition import compose_polynomial_on_affine, compose_exp_on_affine
from src.mirror_operator_exact import (
    apply_QQexp_mirror_composition,
    get_mirror_exp_affine_coeffs,
    get_mirror_eigenvalues_with_swap,
    get_mirror_eigenvalues_complement_t,
)
from src.operator_post_identity import get_A_alpha_affine_coeffs, get_A_beta_affine_coeffs


def compute_empirical_basis_integrand(
    u: float, t: float,
    theta: float, R: float,
    P1_poly, Q_poly
):
    """
    Compute the empirical basis integrand for I1 at (u, t).
    This is I1 evaluated at -R (the basis for the empirical mirror formula).
    """
    var_names = ("x", "y")

    # Profile at u
    P1_series = compose_polynomial_on_affine(P1_poly, u, {"x": 1.0}, var_names)
    P2_series = compose_polynomial_on_affine(P1_poly, u, {"y": 1.0}, var_names)

    # Direct eigenvalues at -R
    u0_a, x_a, y_a = get_A_alpha_affine_coeffs(t, theta)
    u0_b, x_b, y_b = get_A_beta_affine_coeffs(t, theta)

    # Q series (at -R we use same eigenvalues but R -> -R in exp)
    Q_alpha = compose_polynomial_on_affine(Q_poly, u0_a, {"x": x_a, "y": y_a}, var_names)
    Q_beta = compose_polynomial_on_affine(Q_poly, u0_b, {"x": x_b, "y": y_b}, var_names)

    # Exp at -R: the exponential factor changes sign
    R_minus = -R
    exp_u0 = 2 * R_minus * t  # This gives negative exponent
    exp_x = theta * R_minus * (2*t - 1)
    exp_y = theta * R_minus * (2*t - 1)
    exp_series = compose_exp_on_affine(1.0, exp_u0, {"x": exp_x, "y": exp_y}, var_names)

    # Algebraic prefactor
    alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

    # Product
    product = P1_series * P2_series * Q_alpha * Q_beta * exp_series * alg_prefactor

    # Extract xy coefficient (1-u)^2 prefactor for (1,1) pair
    scalar_prefactor = (1 - u) ** 2
    xy_coeff = product.extract(("x", "y"))

    return scalar_prefactor * xy_coeff


def compute_phase10_mirror_integrand(
    u: float, t: float,
    theta: float, R: float,
    P1_poly, Q_poly
):
    """
    Compute Phase 10 (static eigenvalues) mirror integrand for I1 at (u, t).
    """
    var_names = ("x", "y")

    # Profile at u
    P1_series = compose_polynomial_on_affine(P1_poly, u, {"x": 1.0}, var_names)
    P2_series = compose_polynomial_on_affine(P1_poly, u, {"y": 1.0}, var_names)

    # Phase 10: Static mirror eigenvalues (no t-dependence)
    core = apply_QQexp_mirror_composition(Q_poly, t, theta, R, var_names, use_t_dependent=False)

    # Algebraic prefactor
    alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

    # Product
    product = P1_series * P2_series * core * alg_prefactor

    # Extract xy coefficient
    scalar_prefactor = (1 - u) ** 2
    xy_coeff = product.extract(("x", "y"))

    # T weight (exp(2R))
    T_weight = np.exp(2 * R)

    return scalar_prefactor * xy_coeff * T_weight


def compute_phase12_mirror_integrand(
    u: float, t: float,
    theta: float, R: float,
    P1_poly, Q_poly
):
    """
    Compute Phase 12 (t-dependent complement eigenvalues) mirror integrand for I1 at (u, t).
    """
    var_names = ("x", "y")

    # Profile at u
    P1_series = compose_polynomial_on_affine(P1_poly, u, {"x": 1.0}, var_names)
    P2_series = compose_polynomial_on_affine(P1_poly, u, {"y": 1.0}, var_names)

    # Phase 12: T-dependent complement eigenvalues
    core = apply_QQexp_mirror_composition(Q_poly, t, theta, R, var_names, use_t_dependent=True)

    # Algebraic prefactor
    alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

    # Product
    product = P1_series * P2_series * core * alg_prefactor

    # Extract xy coefficient
    scalar_prefactor = (1 - u) ** 2
    xy_coeff = product.extract(("x", "y"))

    # T weight (exp(2R))
    T_weight = np.exp(2 * R)

    return scalar_prefactor * xy_coeff * T_weight


def main():
    print("=" * 70)
    print("Phase 12.2: Integrand-Level Diagnosis")
    print("=" * 70)

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    theta = 4.0 / 7.0
    R = 1.3036

    # Test nodes
    test_nodes = [
        (0.2, 0.5),
        (0.8, 0.5),
        (0.5, 0.2),
        (0.5, 0.8),
        (0.3, 0.3),
        (0.7, 0.7),
    ]

    print(f"\nParameters: theta={theta:.4f}, R={R:.4f}")
    print(f"Empirical m1 = exp(R) + 5 = {np.exp(R) + 5:.4f}")
    print()

    # Display eigenvalue comparison
    print("Eigenvalue Structure Comparison at t=0.5:")
    t = 0.5
    u0_a, x_a, y_a = get_A_alpha_affine_coeffs(t, theta)
    u0_b, x_b, y_b = get_A_beta_affine_coeffs(t, theta)
    eig_p10 = get_mirror_eigenvalues_with_swap(theta)
    eig_p12 = get_mirror_eigenvalues_complement_t(t, theta)

    print(f"  Direct A_α = {u0_a:.3f} + {x_a:.3f}x + {y_a:.3f}y")
    print(f"  Direct A_β = {u0_b:.3f} + {x_b:.3f}x + {y_b:.3f}y")
    print(f"  Phase 10 A_α^mirror = {eig_p10.u0_alpha:.3f} + {eig_p10.x_alpha:.3f}x + {eig_p10.y_alpha:.3f}y")
    print(f"  Phase 12 A_α^mirror = {eig_p12.u0_alpha:.3f} + {eig_p12.x_alpha:.3f}x + {eig_p12.y_alpha:.3f}y")
    print()

    print("-" * 70)
    print(f"{'(u, t)':<12} {'Empirical':>12} {'Phase 10':>12} {'Phase 12':>12} {'P10/Emp':>10} {'P12/Emp':>10}")
    print("-" * 70)

    ratios_p10 = []
    ratios_p12 = []

    for u, t in test_nodes:
        emp = compute_empirical_basis_integrand(u, t, theta, R, P1, Q)
        p10 = compute_phase10_mirror_integrand(u, t, theta, R, P1, Q)
        p12 = compute_phase12_mirror_integrand(u, t, theta, R, P1, Q)

        ratio_p10 = p10 / emp if abs(emp) > 1e-15 else float('inf')
        ratio_p12 = p12 / emp if abs(emp) > 1e-15 else float('inf')

        ratios_p10.append(ratio_p10)
        ratios_p12.append(ratio_p12)

        print(f"({u:.1f}, {t:.1f})    {emp:>12.6f} {p10:>12.6f} {p12:>12.6f} {ratio_p10:>10.2f} {ratio_p12:>10.2f}")

    print("-" * 70)

    # Statistics
    print("\n=== Ratio Analysis ===")
    print(f"Phase 10 / Empirical:")
    print(f"  Mean: {np.mean(ratios_p10):.2f}")
    print(f"  Std:  {np.std(ratios_p10):.2f}")
    print(f"  Range: [{min(ratios_p10):.2f}, {max(ratios_p10):.2f}]")

    print(f"\nPhase 12 / Empirical:")
    print(f"  Mean: {np.mean(ratios_p12):.2f}")
    print(f"  Std:  {np.std(ratios_p12):.2f}")
    print(f"  Range: [{min(ratios_p12):.2f}, {max(ratios_p12):.2f}]")

    print("\n=== Diagnosis ===")
    cv_p10 = np.std(ratios_p10) / np.mean(ratios_p10) if np.mean(ratios_p10) != 0 else float('inf')
    cv_p12 = np.std(ratios_p12) / np.mean(ratios_p12) if np.mean(ratios_p12) != 0 else float('inf')

    if cv_p10 < 0.1:
        print("Phase 10: Ratio is roughly constant → missing SCALAR normalization")
    else:
        print("Phase 10: Ratio varies → missing FUNCTIONAL structure")

    if cv_p12 < 0.1:
        print("Phase 12: Ratio is roughly constant → missing SCALAR normalization")
    else:
        print("Phase 12: Ratio varies → missing FUNCTIONAL structure")

    print(f"\nTarget m1: ~8.68")
    print(f"Phase 10 implied m1 ratio: ~{np.mean(ratios_p10) * np.exp(2*R):.1f}")
    print(f"Phase 12 implied m1 ratio: ~{np.mean(ratios_p12) * np.exp(2*R):.1f}")


if __name__ == "__main__":
    main()
