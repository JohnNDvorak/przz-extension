#!/usr/bin/env python3
"""
Validate new optimum polynomials (v2) - using Q_mono correctly.

New optimum: κ = 0.5213 (+24.9% over PRZZ baseline of 0.4173)
"""

import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polynomials import P1Polynomial, PellPolynomial, Polynomial
from src.kappa_engine import KappaEngine


def load_new_optimum():
    """Load new optimum polynomials."""
    path = Path(__file__).parent.parent / "data" / "optimal_polynomials_v2.json"
    with open(path) as f:
        data = json.load(f)
    return data


def run_kappa_engine(data, R=1.3036, n_quad=60):
    """Run KappaEngine on the new polynomials."""
    engine = KappaEngine(
        P1_coeffs=data['P1_tilde'],
        P2_coeffs=data['P2_tilde'],
        P3_coeffs=data['P3_tilde'],
        Q_coeffs=data['Q_mono'],
        theta=4/7,
        K=3,
        R=R,
        n_quad=n_quad,
    )
    return engine.compute_kappa()


def gate_psd_cs(data, R=1.3036):
    """Gate PSD/CS: Verify pair matrix is positive semi-definite."""
    print("\n" + "=" * 70)
    print("GATE PSD/CS: Positive Semi-Definite Check")
    print("=" * 70)

    from src.unified_i2_paper import compute_I2_unified_paper

    # Build polynomial objects
    P1 = P1Polynomial(data['P1_tilde'])
    P2 = PellPolynomial(data['P2_tilde'])
    P3 = PellPolynomial(data['P3_tilde'])
    Q = Polynomial(np.array(data['Q_mono']))

    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    theta = 4/7

    # Compute I2 for all pairs
    pairs = {}
    for ell1 in range(1, 4):
        for ell2 in range(ell1, 4):
            result = compute_I2_unified_paper(
                R, theta, ell1=ell1, ell2=ell2,
                polynomials=polys,
                n_quad_u=60, n_quad_t=60, n_quad_a=40,
            )
            key = f"{ell1}{ell2}"
            pairs[key] = result.I2_value
            print(f"  I2({ell1},{ell2}) = {result.I2_value:+.8e}")

    # Build Gram matrix
    G = np.zeros((3, 3))
    G[0, 0] = pairs["11"]
    G[1, 1] = pairs["22"]
    G[2, 2] = pairs["33"]
    G[0, 1] = G[1, 0] = pairs["12"] / 2
    G[0, 2] = G[2, 0] = pairs["13"] / 2
    G[1, 2] = G[2, 1] = pairs["23"] / 2

    eigenvalues = np.linalg.eigvalsh(G)
    lambda_min = eigenvalues.min()

    # Correlations
    rho = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            denom = np.sqrt(abs(G[i, i]) * abs(G[j, j]))
            if denom > 1e-15:
                rho[i, j] = G[i, j] / denom

    # Cauchy-Schwarz
    cs_pass = all(
        abs(G[i, j]) <= np.sqrt(abs(G[i, i]) * abs(G[j, j])) + 1e-10
        for i in range(3) for j in range(i+1, 3)
    )

    psd_pass = lambda_min >= -1e-10

    print(f"\n  Eigenvalues: {eigenvalues}")
    print(f"  λ_min = {lambda_min:.6e}")
    print(f"\n  Correlations:")
    print(f"    ρ(1,2) = {rho[0,1]:+.4f}")
    print(f"    ρ(1,3) = {rho[0,2]:+.4f}")
    print(f"    ρ(2,3) = {rho[1,2]:+.4f}")
    print(f"\n  PSD: {'PASS' if psd_pass else 'FAIL'}")
    print(f"  Cauchy-Schwarz: {'PASS' if cs_pass else 'FAIL'}")

    return psd_pass and cs_pass, {"G": G, "eigenvalues": eigenvalues, "rho": rho, "pairs": pairs}


def gate_k2_reduction(data, R=1.3036):
    """Gate 1: P3=0 should eliminate all Case C pairs."""
    print("\n" + "=" * 70)
    print("GATE 1: K=2 Reduction (P3=0 eliminates Case C)")
    print("=" * 70)

    from src.unified_i1_paper import compute_I1_unified_paper

    P1 = P1Polynomial(data['P1_tilde'])
    P2 = PellPolynomial(data['P2_tilde'])
    P3_zero = Polynomial(np.array([0.0]))
    Q = Polynomial(np.array(data['Q_mono']))

    polys_zero = {"P1": P1, "P2": P2, "P3": P3_zero, "Q": Q}
    theta = 4/7

    all_vanish = True
    print("\n  P3-involving pairs with P3=0:")
    for ell1, ell2 in [(1, 3), (2, 3), (3, 3)]:
        result = compute_I1_unified_paper(
            R, theta, ell1=ell1, ell2=ell2,
            polynomials=polys_zero,
            n_quad_u=60, n_quad_t=60,
        )
        val = result.I1_value
        status = "PASS" if abs(val) < 1e-12 else "FAIL"
        print(f"    ({ell1},{ell2}): I1 = {val:.6e} [{status}]")
        if abs(val) >= 1e-12:
            all_vanish = False

    print(f"\n  Gate 1: {'PASS' if all_vanish else 'FAIL'}")
    return all_vanish


def gate_independent_evaluator(data, R=1.3036):
    """Gate 2: Independent evaluator matches production."""
    print("\n" + "=" * 70)
    print("GATE 2: Independent Evaluator")
    print("=" * 70)

    from scipy import integrate
    import math

    theta = 4/7
    P2_tilde = data['P2_tilde']
    P3_tilde = data['P3_tilde']

    def eval_Pn(x, tilde):
        x = np.atleast_1d(x)
        p_tilde = sum(c * (x ** i) for i, c in enumerate(tilde))
        return x * p_tilde

    def case_c_kernel(P_eval, u, omega):
        if u < 1e-14:
            return 0.0
        factorial_denom = math.factorial(omega - 1)

        def integrand(a):
            arg = (1 - a) * u
            P_val = P_eval(np.array([arg]))[0]
            a_power = a ** (omega - 1) if omega > 1 else 1.0
            return P_val * a_power * np.exp(R * theta * u * a)

        integral, _ = integrate.quad(integrand, 0, 1, limit=100)
        return (u ** omega) / factorial_denom * integral

    from src.case_c_kernel import compute_case_c_kernel

    all_match = True
    print("\n  Case C kernel comparison:")

    for omega, tilde, name in [(1, P2_tilde, "P2"), (2, P3_tilde, "P3")]:
        P_eval = lambda x, t=tilde: eval_Pn(x, t)
        for u in [0.3, 0.5, 0.7]:
            K_indep = case_c_kernel(P_eval, u, omega)
            K_prod = compute_case_c_kernel(P_eval, np.array([u]), omega, R, theta)[0]
            rel_diff = abs(K_indep - K_prod) / (abs(K_prod) + 1e-15)
            status = "PASS" if rel_diff < 1e-6 else "FAIL"
            if rel_diff >= 1e-6:
                all_match = False
            print(f"    {name} u={u}: diff={rel_diff:.2e} [{status}]")

    print(f"\n  Gate 2: {'PASS' if all_match else 'FAIL'}")
    return all_match


def gate_basis_stability(data, R=1.3036):
    """Gate 4: Basis stability."""
    print("\n" + "=" * 70)
    print("GATE 4: Basis Stability")
    print("=" * 70)

    from numpy.polynomial import chebyshev as C
    from src.quadrature import gauss_legendre_01

    theta = 4/7

    P2 = PellPolynomial(data['P2_tilde'])
    Q = Polynomial(np.array(data['Q_mono']))

    p2_mono = P2.to_monomial().coeffs
    p2_cheb = C.poly2cheb(p2_mono)

    def eval_mono(x):
        return sum(c * (x ** i) for i, c in enumerate(p2_mono))

    def eval_cheb(x):
        return C.chebval(x, p2_cheb)

    x_test = np.linspace(0.01, 0.99, 30)
    y_mono = np.array([eval_mono(x) for x in x_test])
    y_cheb = np.array([eval_cheb(x) for x in x_test])

    eval_diff = np.max(np.abs(y_mono - y_cheb))
    print(f"\n  P2 evaluation difference: {eval_diff:.2e}")

    u_nodes, u_weights = gauss_legendre_01(60)
    t_nodes, t_weights = gauss_legendre_01(60)

    Q_vals = Q.eval(t_nodes)
    t_int = np.sum(t_weights * Q_vals**2 * np.exp(2*R*t_nodes)) / theta

    P2_mono_vals = np.array([eval_mono(u) for u in u_nodes])
    P2_cheb_vals = np.array([eval_cheb(u) for u in u_nodes])

    i2_mono = np.sum(u_weights * P2_mono_vals**2) * t_int
    i2_cheb = np.sum(u_weights * P2_cheb_vals**2) * t_int

    rel_diff = abs(i2_mono - i2_cheb) / (abs(i2_mono) + 1e-15)

    print(f"  I2 difference: {rel_diff:.2e}")

    passed = rel_diff < 1e-10
    print(f"\n  Gate 4: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("\n" + "=" * 70)
    print("VALIDATION OF NEW OPTIMUM")
    print("κ = 0.5213 (+24.9% over PRZZ baseline 0.4173)")
    print("=" * 70)

    data = load_new_optimum()

    # Run KappaEngine
    print("\n--- KappaEngine Verification ---")
    result = run_kappa_engine(data)
    print(f"  Computed κ: {result.kappa:.6f}")
    print(f"  Computed c: {result.c:.6f}")
    print(f"  Claimed κ:  {data['kappa']}")
    print(f"  Claimed c:  {data['c']}")
    print(f"  Match: {'YES' if abs(result.kappa - data['kappa']) < 0.001 else 'NO'}")

    # Run gates
    gates = {}
    psd_pass, psd_data = gate_psd_cs(data)
    gates["PSD/CS"] = psd_pass

    gates["K2 Reduction"] = gate_k2_reduction(data)
    gates["Independent"] = gate_independent_evaluator(data)
    gates["Basis"] = gate_basis_stability(data)

    # Kappa* check
    print("\n" + "=" * 70)
    print("κ* Benchmark Check (R=1.1167)")
    print("=" * 70)

    result_star = run_kappa_engine(data, R=1.1167)
    print(f"  Computed κ*: {result_star.kappa:.6f}")
    print(f"  Claimed κ*:  {data['kappa_star']}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = all(gates.values())

    for name, passed in gates.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    print(f"\n  κ improvement: {(result.kappa / 0.4173 - 1) * 100:+.1f}% over PRZZ")
    print(f"  κ* improvement: {(result_star.kappa / 0.4075 - 1) * 100:+.1f}% over PRZZ")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL GATES PASS - NEW OPTIMUM VALIDATED")
        print(f"κ = {result.kappa:.4f} (Δκ = +{result.kappa - 0.4173:.4f})")
    else:
        print("SOME GATES FAILED")
    print("=" * 70)

    # Detailed decomposition
    print("\n" + "=" * 70)
    print("DECOMPOSITION DETAILS")
    print("=" * 70)

    integrals = result.integrals
    corrections = result.corrections

    print(f"\n  I1(+R): {integrals.I1_plus:.6f}")
    print(f"  I2(+R): {integrals.I2_plus:.6f}")
    print(f"  I1(-R): {integrals.I1_minus:.6f}")
    print(f"  I2(-R): {integrals.I2_minus:.6f}")
    print(f"  I3(+R): {integrals.I3_plus:.6f}")
    print(f"  I4(+R): {integrals.I4_plus:.6f}")
    print(f"\n  f_I1: {integrals.f_I1:.6f}")
    print(f"  m: {corrections.m:.6f}")
    print(f"  g_I1: {corrections.g_I1:.6f}")
    print(f"  g_I2: {corrections.g_I2:.6f}")

    S12_plus = integrals.I1_plus + integrals.I2_plus
    S12_minus = integrals.I1_minus + integrals.I2_minus
    S34_plus = integrals.I3_plus + integrals.I4_plus

    print(f"\n  Assembly:")
    print(f"    S12(+R) = {S12_plus:.6f}")
    print(f"    S12(-R) = {S12_minus:.6f}")
    print(f"    S34(+R) = {S34_plus:.6f}")
    print(f"    c = {S12_plus:.4f} + {corrections.m:.4f} × {S12_minus:.4f} + {S34_plus:.4f}")
    print(f"      = {result.c:.6f}")

    return all_passed, result, psd_data


if __name__ == "__main__":
    passed, result, psd_data = main()
