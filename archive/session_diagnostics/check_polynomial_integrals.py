"""
Check polynomial integral magnitudes to understand the 2.4x scaling factor.

Specifically, we want to compute:
1. ∫₀¹ P₂(u)² du for both κ and κ*
2. ∫₀¹ P₂(u)P₂'(u) du
3. ∫₀¹ (1-u)² P₂(u)² du
4. Other relevant polynomial moments

This will help determine if polynomial magnitude differences contribute
to the observed 2.4x ratio.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def gauss_legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def compute_polynomial_integrals(P2, label: str, n_quad: int = 100):
    """Compute various polynomial integrals."""
    u_nodes, u_weights = gauss_legendre_01(n_quad)

    # Evaluate polynomial and derivatives
    P = P2.eval(u_nodes)
    Pp = P2.eval_deriv(u_nodes, 1)
    Ppp = P2.eval_deriv(u_nodes, 2)

    # Various integrals
    int_P2 = np.sum(u_weights * P * P)
    int_P = np.sum(u_weights * P)
    int_PPp = np.sum(u_weights * P * Pp)
    int_Pp2 = np.sum(u_weights * Pp * Pp)
    int_1mu_P2 = np.sum(u_weights * (1 - u_nodes) * P * P)
    int_1mu2_P2 = np.sum(u_weights * (1 - u_nodes)**2 * P * P)
    int_u_P2 = np.sum(u_weights * u_nodes * P * P)

    print(f"--- {label} ---")
    print(f"Tilde coeffs: {len(P2.tilde_coeffs)} terms")
    print(f"∫ P² du:              {int_P2:>12.6f}")
    print(f"∫ P du:               {int_P:>12.6f}")
    print(f"∫ P·P' du:            {int_PPp:>12.6f}")
    print(f"∫ (P')² du:           {int_Pp2:>12.6f}")
    print(f"∫ (1-u)·P² du:        {int_1mu_P2:>12.6f}")
    print(f"∫ (1-u)²·P² du:       {int_1mu2_P2:>12.6f}")
    print(f"∫ u·P² du:            {int_u_P2:>12.6f}")
    print()

    return {
        'P2': int_P2,
        'P': int_P,
        'PPp': int_PPp,
        'Pp2': int_Pp2,
        '1mu_P2': int_1mu_P2,
        '1mu2_P2': int_1mu2_P2,
        'u_P2': int_u_P2,
    }


def main():
    print("="*80)
    print("POLYNOMIAL INTEGRAL ANALYSIS")
    print("="*80)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    # Compute integrals
    k_ints = compute_polynomial_integrals(P2_k, "κ benchmark P₂")
    ks_ints = compute_polynomial_integrals(P2_ks, "κ* benchmark P₂")

    # Compute ratios
    print("="*80)
    print("RATIO ANALYSIS: κ / κ*")
    print("="*80)
    print()

    for key in k_ints.keys():
        ratio = k_ints[key] / ks_ints[key] if ks_ints[key] != 0 else float('inf')
        print(f"Ratio for {key:12s}: {ratio:>10.4f}")
    print()

    # Special focus on the main integrals that appear in I₂
    print("="*80)
    print("KEY INTEGRALS FOR I₂")
    print("="*80)
    print()
    print("I₂ contains: (1/θ) × [∫ P²(u) du] × [∫ Q²(t) e^{2Rt} dt]")
    print()
    print(f"∫ P²(u) du ratio: {k_ints['P2'] / ks_ints['P2']:.4f}")
    print()

    # Now check Q integrals
    print("="*80)
    print("Q POLYNOMIAL INTEGRALS")
    print("="*80)
    print()

    # Compute Q integrals for different R values
    R_kappa = 1.3036
    R_kappa_star = 1.1167
    n_quad = 100

    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # κ benchmark
    Q_k_vals = Q_k.eval(t_nodes)
    Qp_k_vals = Q_k.eval_deriv(t_nodes, 1)
    exp_2Rt_k = np.exp(2 * R_kappa * t_nodes)

    int_Q2_k = np.sum(t_weights * Q_k_vals * Q_k_vals)
    int_Q2_e2Rt_k = np.sum(t_weights * Q_k_vals * Q_k_vals * exp_2Rt_k)

    print(f"--- κ benchmark Q (R={R_kappa}) ---")
    print(f"∫ Q² dt:              {int_Q2_k:>12.6f}")
    print(f"∫ Q²·e^{{2Rt}} dt:      {int_Q2_e2Rt_k:>12.6f}")
    print()

    # κ* benchmark
    Q_ks_vals = Q_ks.eval(t_nodes)
    Qp_ks_vals = Q_ks.eval_deriv(t_nodes, 1)
    exp_2Rt_ks = np.exp(2 * R_kappa_star * t_nodes)

    int_Q2_ks = np.sum(t_weights * Q_ks_vals * Q_ks_vals)
    int_Q2_e2Rt_ks = np.sum(t_weights * Q_ks_vals * Q_ks_vals * exp_2Rt_ks)

    print(f"--- κ* benchmark Q (R={R_kappa_star}) ---")
    print(f"∫ Q² dt:              {int_Q2_ks:>12.6f}")
    print(f"∫ Q²·e^{{2Rt}} dt:      {int_Q2_e2Rt_ks:>12.6f}")
    print()

    # Ratios
    print("="*80)
    print("Q INTEGRAL RATIOS")
    print("="*80)
    print()
    print(f"∫ Q² dt ratio:         {int_Q2_k / int_Q2_ks:>10.4f}")
    print(f"∫ Q²·e^{{2Rt}} dt ratio: {int_Q2_e2Rt_k / int_Q2_e2Rt_ks:>10.4f}")
    print()

    # Now compute the full I₂ ratio from components
    theta = 4/7
    u_integral_k = k_ints['P2']
    u_integral_ks = ks_ints['P2']
    t_integral_k = int_Q2_e2Rt_k
    t_integral_ks = int_Q2_e2Rt_ks

    I2_k = (1.0 / theta) * u_integral_k * t_integral_k
    I2_ks = (1.0 / theta) * u_integral_ks * t_integral_ks

    print("="*80)
    print("I₂ RECONSTRUCTION FROM COMPONENTS")
    print("="*80)
    print()
    print(f"I₂(κ) = (1/θ) × {u_integral_k:.6f} × {t_integral_k:.6f} = {I2_k:.6f}")
    print(f"I₂(κ*) = (1/θ) × {u_integral_ks:.6f} × {t_integral_ks:.6f} = {I2_ks:.6f}")
    print(f"Ratio: {I2_k / I2_ks:.4f}")
    print()
    print("From oracle: I₂(κ)/I₂(κ*) = 2.6685")
    print()

    # Break down the ratio
    print("="*80)
    print("RATIO DECOMPOSITION")
    print("="*80)
    print()
    print(f"∫ P² ratio:           {u_integral_k / u_integral_ks:>10.4f}")
    print(f"∫ Q²·e^{{2Rt}} ratio:   {t_integral_k / t_integral_ks:>10.4f}")
    print(f"Product:              {(u_integral_k / u_integral_ks) * (t_integral_k / t_integral_ks):>10.4f}")
    print()

    # What fraction comes from each?
    p_ratio = u_integral_k / u_integral_ks
    q_ratio = t_integral_k / t_integral_ks
    total_ratio = p_ratio * q_ratio

    print(f"P² contributes: {p_ratio:.4f} (factor of {p_ratio:6.2f}x)")
    print(f"Q²·e^{{2Rt}} contributes: {q_ratio:.4f} (factor of {q_ratio:6.2f}x)")
    print(f"Combined: {total_ratio:.4f}")
    print()

    # Check if these are consistent
    print("="*80)
    print("CONSISTENCY CHECK")
    print("="*80)
    print()
    print(f"Expected I₂ ratio from oracle: 2.6685")
    print(f"Computed I₂ ratio from parts: {total_ratio:.4f}")
    print(f"Match: {abs(total_ratio - 2.6685) < 0.01}")
    print()


if __name__ == "__main__":
    main()
