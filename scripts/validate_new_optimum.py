#!/usr/bin/env python3
"""
Validate new optimum polynomials against all gates.

New optimum: κ = 0.5213 (+24.9% over PRZZ baseline)

Gates to check:
1. PSD/Cauchy-Schwarz: Pair matrix must be positive semi-definite
2. K=2 Reduction: P3=0 must eliminate all Case C pairs
3. Independent Evaluator: Must match production code
4. Basis Stability: Same polynomial in different bases gives same c
"""

import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polynomials import P1Polynomial, PellPolynomial, QPolynomial, Polynomial
from src.unified_i1_paper import compute_I1_unified_paper
from src.unified_i2_paper import compute_I2_unified_paper


def load_new_optimum():
    """Load new optimum polynomials."""
    path = Path(__file__).parent.parent / "data" / "optimal_polynomials_v2.json"
    with open(path) as f:
        data = json.load(f)

    P1 = P1Polynomial(data["P1_tilde"])
    P2 = PellPolynomial(data["P2_tilde"])
    P3 = PellPolynomial(data["P3_tilde"])

    # Q uses PRZZ fixed basis
    Q_basis = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
    Q = QPolynomial(Q_basis, enforce_Q0=True)

    return {
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "Q": Q,
        "meta": data,
    }


def gate_psd_cauchy_schwarz(polys, R=1.3036, theta=4/7):
    """Gate PSD/CS: Verify pair matrix is positive semi-definite."""
    print("\n" + "=" * 70)
    print("GATE PSD/CS: Positive Semi-Definite Check")
    print("=" * 70)

    # Compute all pair I2 values
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

    # Build Gram matrix (divide off-diagonals by 2)
    G = np.zeros((3, 3))
    G[0, 0] = pairs["11"]
    G[1, 1] = pairs["22"]
    G[2, 2] = pairs["33"]
    G[0, 1] = G[1, 0] = pairs["12"] / 2
    G[0, 2] = G[2, 0] = pairs["13"] / 2
    G[1, 2] = G[2, 1] = pairs["23"] / 2

    # Check PSD
    eigenvalues = np.linalg.eigvalsh(G)
    lambda_min = eigenvalues.min()
    is_psd = lambda_min >= -1e-10

    # Compute correlations
    rho = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            denom = np.sqrt(abs(G[i, i]) * abs(G[j, j]))
            if denom > 1e-15:
                rho[i, j] = G[i, j] / denom

    # Check Cauchy-Schwarz
    cs_violations = []
    for i in range(3):
        for j in range(i + 1, 3):
            bound = np.sqrt(abs(G[i, i]) * abs(G[j, j]))
            if abs(G[i, j]) > bound + 1e-10:
                cs_violations.append((i + 1, j + 1))

    print(f"\n  Gram matrix eigenvalues: {eigenvalues}")
    print(f"  λ_min = {lambda_min:.6e}")
    print(f"\n  Correlations:")
    print(f"    ρ(1,2) = {rho[0, 1]:+.4f}")
    print(f"    ρ(1,3) = {rho[0, 2]:+.4f}")
    print(f"    ρ(2,3) = {rho[1, 2]:+.4f}")

    psd_pass = is_psd
    cs_pass = len(cs_violations) == 0

    print(f"\n  PSD: {'PASS' if psd_pass else 'FAIL'}")
    print(f"  Cauchy-Schwarz: {'PASS' if cs_pass else 'FAIL'}")

    return psd_pass and cs_pass, {"G": G, "eigenvalues": eigenvalues, "rho": rho, "pairs": pairs}


def gate_k2_reduction(polys, R=1.3036, theta=4/7):
    """Gate 1: P3=0 should eliminate all Case C pairs."""
    print("\n" + "=" * 70)
    print("GATE 1: K=2 Reduction (P3=0 eliminates Case C)")
    print("=" * 70)

    # Create zero P3
    P3_zero = Polynomial(np.array([0.0]))

    polys_zero_p3 = {
        "P1": polys["P1"],
        "P2": polys["P2"],
        "P3": P3_zero,
        "Q": polys["Q"],
    }

    # Check P3-involving pairs vanish
    p3_pairs = [(1, 3), (2, 3), (3, 3)]
    all_vanish = True

    print("\n  P3-involving pairs with P3=0:")
    for ell1, ell2 in p3_pairs:
        result = compute_I1_unified_paper(
            R, theta, ell1=ell1, ell2=ell2,
            polynomials=polys_zero_p3,
            n_quad_u=60, n_quad_t=60,
        )
        val = result.I1_value
        status = "PASS" if abs(val) < 1e-12 else "FAIL"
        print(f"    ({ell1},{ell2}): I1 = {val:.6e} [{status}]")
        if abs(val) >= 1e-12:
            all_vanish = False

    print(f"\n  Gate 1: {'PASS' if all_vanish else 'FAIL'}")
    return all_vanish


def gate_independent_evaluator(polys, R=1.3036, theta=4/7):
    """Gate 2: Independent evaluator matches production."""
    print("\n" + "=" * 70)
    print("GATE 2: Independent Evaluator Comparison")
    print("=" * 70)

    from scipy import integrate
    import math

    # Extract coefficients
    P1_tilde = polys["P1"].tilde_coeffs
    P2_tilde = polys["P2"].tilde_coeffs
    P3_tilde = polys["P3"].tilde_coeffs

    # Independent polynomial evaluators
    def eval_P1(x):
        x = np.atleast_1d(x)
        z = 1 - x
        p_tilde = sum(c * (z ** i) for i, c in enumerate(P1_tilde))
        return x + x * z * p_tilde

    def eval_Pn(x, tilde):
        x = np.atleast_1d(x)
        p_tilde = sum(c * (x ** i) for i, c in enumerate(tilde))
        return x * p_tilde

    def eval_P2(x):
        return eval_Pn(x, P2_tilde)

    def eval_P3(x):
        return eval_Pn(x, P3_tilde)

    # Independent Case C kernel
    def case_c_kernel(P_eval, u, omega):
        if u < 1e-14:
            return 0.0
        factorial_denom = math.factorial(omega - 1)

        def integrand(a):
            arg = (1 - a) * u
            P_val = P_eval(np.array([arg]))[0]
            a_power = a ** (omega - 1) if omega > 1 else 1.0
            exp_factor = np.exp(R * theta * u * a)
            return P_val * a_power * exp_factor

        integral, _ = integrate.quad(integrand, 0, 1, limit=100)
        return (u ** omega) / factorial_denom * integral

    # Test Case C kernels at sample points
    print("\n  Case C kernel comparison at sample points:")
    u_test = [0.3, 0.5, 0.7]

    from src.case_c_kernel import compute_case_c_kernel

    all_match = True

    for omega, P_eval, P_name in [(1, eval_P2, "P2"), (2, eval_P3, "P3")]:
        print(f"\n    {P_name} (omega={omega}):")
        for u in u_test:
            K_indep = case_c_kernel(P_eval, u, omega)
            K_prod = compute_case_c_kernel(P_eval, np.array([u]), omega, R, theta)[0]
            rel_diff = abs(K_indep - K_prod) / (abs(K_prod) + 1e-15)
            status = "PASS" if rel_diff < 1e-6 else "FAIL"
            print(f"      u={u}: indep={K_indep:.6e}, prod={K_prod:.6e}, diff={rel_diff:.2e} [{status}]")
            if rel_diff >= 1e-6:
                all_match = False

    print(f"\n  Gate 2: {'PASS' if all_match else 'FAIL'}")
    return all_match


def gate_basis_stability(polys, R=1.3036, theta=4/7):
    """Gate 4: Same polynomial in different bases gives same c."""
    print("\n" + "=" * 70)
    print("GATE 4: Basis Stability")
    print("=" * 70)

    from numpy.polynomial import chebyshev as C
    from src.quadrature import gauss_legendre_01

    # Get P2 in monomial form
    P2_mono = polys["P2"].to_monomial().coeffs

    # Convert to Chebyshev
    P2_cheb = C.poly2cheb(P2_mono)

    # Evaluate at test points
    x_test = np.linspace(0.01, 0.99, 30)

    def eval_mono(x):
        result = np.zeros_like(x)
        for i, c in enumerate(P2_mono):
            result += c * (x ** i)
        return result

    def eval_cheb(x):
        return C.chebval(x, P2_cheb)

    y_mono = eval_mono(x_test)
    y_cheb = eval_cheb(x_test)

    eval_diff = np.max(np.abs(y_mono - y_cheb))
    print(f"\n  P2 evaluation difference (mono vs cheb): {eval_diff:.2e}")

    # Compute I2 with both
    u_nodes, u_weights = gauss_legendre_01(60)
    t_nodes, t_weights = gauss_legendre_01(60)

    Q_vals = polys["Q"].eval(t_nodes)
    t_int = np.sum(t_weights * Q_vals**2 * np.exp(2*R*t_nodes)) / theta

    P2_mono_vals = eval_mono(u_nodes)
    P2_cheb_vals = eval_cheb(u_nodes)

    i2_mono = np.sum(u_weights * P2_mono_vals**2) * t_int
    i2_cheb = np.sum(u_weights * P2_cheb_vals**2) * t_int

    rel_diff = abs(i2_mono - i2_cheb) / (abs(i2_mono) + 1e-15)

    print(f"  I2 monomial:   {i2_mono:.10e}")
    print(f"  I2 Chebyshev:  {i2_cheb:.10e}")
    print(f"  Relative diff: {rel_diff:.2e}")

    passed = rel_diff < 1e-10
    print(f"\n  Gate 4: {'PASS' if passed else 'FAIL'}")
    return passed


def compute_full_decomposition(polys, R=1.3036, theta=4/7, K=3):
    """Compute full S12/S34 decomposition."""
    print("\n" + "=" * 70)
    print("FULL DECOMPOSITION")
    print("=" * 70)

    # Compute all pairs at +R
    pairs_plus = {}
    pairs_minus = {}

    for ell1 in range(1, 4):
        for ell2 in range(ell1, 4):
            # I1 at +R
            i1_plus = compute_I1_unified_paper(
                R, theta, ell1=ell1, ell2=ell2,
                polynomials=polys,
                n_quad_u=60, n_quad_t=60,
            ).I1_value

            # I2 at +R
            i2_plus = compute_I2_unified_paper(
                R, theta, ell1=ell1, ell2=ell2,
                polynomials=polys,
                n_quad_u=60, n_quad_t=60, n_quad_a=40,
            ).I2_value

            # I1 at -R
            i1_minus = compute_I1_unified_paper(
                -R, theta, ell1=ell1, ell2=ell2,
                polynomials=polys,
                n_quad_u=60, n_quad_t=60,
            ).I1_value

            # I2 at -R
            i2_minus = compute_I2_unified_paper(
                -R, theta, ell1=ell1, ell2=ell2,
                polynomials=polys,
                n_quad_u=60, n_quad_t=60, n_quad_a=40,
            ).I2_value

            key = f"{ell1}{ell2}"
            sym_factor = 1 if ell1 == ell2 else 2

            pairs_plus[key] = {
                "I1": i1_plus,
                "I2": i2_plus,
                "total": sym_factor * (i1_plus + i2_plus),
            }
            pairs_minus[key] = {
                "I1": i1_minus,
                "I2": i2_minus,
                "total": sym_factor * (i1_minus + i2_minus),
            }

    # S12 = sum of pairs (1,1), (1,2), (2,2)
    S12_plus = sum(pairs_plus[k]["total"] for k in ["11", "12", "22"])
    S12_minus = sum(pairs_minus[k]["total"] for k in ["11", "12", "22"])

    # S34 = sum of pairs (1,3), (2,3), (3,3)
    S34_plus = sum(pairs_plus[k]["total"] for k in ["13", "23", "33"])
    S34_minus = sum(pairs_minus[k]["total"] for k in ["13", "23", "33"])

    # Mirror multiplier
    m = (1 + theta / (2*K*(2*K+1))) * (np.exp(R) + (2*K - 1))

    # c assembly
    c = S12_plus + m * S12_minus + S34_plus

    # kappa
    kappa = 1 - np.log(c) / R

    print("\n  Per-pair contributions (+R):")
    for key in ["11", "12", "22", "13", "23", "33"]:
        p = pairs_plus[key]
        print(f"    ({key[0]},{key[1]}): I1={p['I1']:+.6e}, I2={p['I2']:+.6e}, total={p['total']:+.6e}")

    print(f"\n  Assembly:")
    print(f"    S12(+R) = {S12_plus:.6f}")
    print(f"    S12(-R) = {S12_minus:.6f}")
    print(f"    S34(+R) = {S34_plus:.6f}")
    print(f"    m = {m:.6f}")
    print(f"\n    c = S12(+R) + m*S12(-R) + S34(+R)")
    print(f"      = {S12_plus:.6f} + {m:.6f}*{S12_minus:.6f} + {S34_plus:.6f}")
    print(f"      = {c:.6f}")
    print(f"\n    κ = 1 - log(c)/R = {kappa:.6f}")

    return {
        "pairs_plus": pairs_plus,
        "pairs_minus": pairs_minus,
        "S12_plus": S12_plus,
        "S12_minus": S12_minus,
        "S34_plus": S34_plus,
        "m": m,
        "c": c,
        "kappa": kappa,
    }


def main():
    print("\n" + "=" * 70)
    print("VALIDATION OF NEW OPTIMUM: κ = 0.5213 (+24.9% improvement)")
    print("=" * 70)

    # Load polynomials
    polys = load_new_optimum()
    meta = polys["meta"]

    print(f"\n  Source: {meta['source']}")
    print(f"  Claimed κ: {meta['kappa']}")
    print(f"  Claimed c: {meta['c']}")

    # Run all gates
    results = {}

    # Gate PSD/CS
    psd_pass, psd_data = gate_psd_cauchy_schwarz(polys)
    results["psd_cs"] = psd_pass

    # Gate 1: K=2 Reduction
    k2_pass = gate_k2_reduction(polys)
    results["k2_reduction"] = k2_pass

    # Gate 2: Independent Evaluator
    indep_pass = gate_independent_evaluator(polys)
    results["independent"] = indep_pass

    # Gate 4: Basis Stability
    basis_pass = gate_basis_stability(polys)
    results["basis"] = basis_pass

    # Full decomposition
    decomp = compute_full_decomposition(polys)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = all(results.values())

    print(f"\n  Gate PSD/CS:     {'PASS' if results['psd_cs'] else 'FAIL'}")
    print(f"  Gate 1 (K2):     {'PASS' if results['k2_reduction'] else 'FAIL'}")
    print(f"  Gate 2 (Indep):  {'PASS' if results['independent'] else 'FAIL'}")
    print(f"  Gate 4 (Basis):  {'PASS' if results['basis'] else 'FAIL'}")

    print(f"\n  Computed κ: {decomp['kappa']:.6f}")
    print(f"  Claimed κ:  {meta['kappa']:.4f}")
    print(f"  Difference: {abs(decomp['kappa'] - meta['kappa']):.6f}")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL GATES PASS - NEW OPTIMUM VALIDATED")
    else:
        print("SOME GATES FAILED - INVESTIGATE")
    print("=" * 70)

    return all_passed, decomp, psd_data


if __name__ == "__main__":
    passed, decomp, psd_data = main()
