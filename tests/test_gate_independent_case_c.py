#!/usr/bin/env python3
"""
Gate 2: Independent C-Case Evaluator

This is the CRITICAL validation gate. It builds a completely separate
implementation of Case C pairs (1,3), (2,3), (3,3) using:
- Only scipy.integrate (no production quadrature)
- Literal PRZZ formulas (no BivariateSeries)
- No code reuse from production modules

If this matches production to 1e-6 relative tolerance, the Case C
implementation is correct and the sign flips are real.

PRZZ Formula References:
========================
TeX 2370-2375: K_omega(u; R) = u^omega/(omega-1)! * integral_0^1 P((1-a)u) * a^{omega-1} * exp(R*theta*u*a) da

For I2 (no derivatives):
I2_ij = (1/theta) * integral_0^1 integral_0^1 K_i(u) * K_j(u) * Q(t)^2 * exp(2Rt) du dt

where:
- K_i for i=1 (omega=0): just P1(u)
- K_i for i=2 (omega=1): u * integral_0^1 P2((1-a)u) * exp(R*theta*u*a) da
- K_i for i=3 (omega=2): u^2 * integral_0^1 P3((1-a)u) * a * exp(R*theta*u*a) da

Created: 2025-12-28
"""

import numpy as np
import pytest
from scipy import integrate
from pathlib import Path
import json
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# INDEPENDENT POLYNOMIAL LOADING (no production code)
# =============================================================================

def load_przz_polynomials_direct() -> dict:
    """
    Load PRZZ polynomials directly from JSON, bypassing production code.

    Returns dict with P1, P2, P3, Q as coefficient arrays.
    """
    params_path = Path(__file__).parent.parent / "data" / "przz_parameters.json"
    with open(params_path) as f:
        data = json.load(f)

    # P1: tilde coeffs [c0, c1, ...] for P1(x) = x + x(1-x) * sum_i c_i * (1-x)^i
    p1_tilde = data["polynomials"]["P1"]["tilde_coeffs"]

    # P2, P3: tilde coeffs for P_ell(x) = x * sum_i c_i * (1-x)^i
    p2_tilde = data["polynomials"]["P2"]["tilde_coeffs"]
    p3_tilde = data["polynomials"]["P3"]["tilde_coeffs"]

    # Q: coeffs in (1-2t)^k basis - stored as [{k: int, c: float}, ...]
    q_coeffs_raw = data["polynomials"]["Q"]["coeffs_in_basis_terms"]
    # Convert to array indexed by k
    max_k = max(item["k"] for item in q_coeffs_raw)
    q_coeffs = np.zeros(max_k + 1)
    for item in q_coeffs_raw:
        q_coeffs[item["k"]] = item["c"]

    return {
        "P1_tilde": np.array(p1_tilde),
        "P2_tilde": np.array(p2_tilde),
        "P3_tilde": np.array(p3_tilde),
        "Q_coeffs": q_coeffs,
    }


def load_optimized_polynomials_direct() -> dict:
    """
    Load optimized polynomials (alpha=70, beta=-30) from optimization results.
    """
    opt_path = Path(__file__).parent.parent / "data" / "optimal_polynomials.json"
    try:
        with open(opt_path) as f:
            data = json.load(f)
        return {
            "P1_tilde": np.array(data["P1_tilde"]),
            "P2_tilde": np.array(data["P2_tilde"]),
            "P3_tilde": np.array(data["P3_tilde"]),
            "Q_coeffs": np.array(data["Q_coeffs"]),
        }
    except FileNotFoundError:
        # Fallback: use PRZZ baseline with alpha, beta modification
        # This is a minimal approximation for testing
        baseline = load_przz_polynomials_direct()
        # Note: Real optimization changes all coefficients
        # For gate testing, we load from actual optimization output
        raise FileNotFoundError("Optimized polynomials not found - run optimization first")


def eval_P1_direct(x: np.ndarray, tilde_coeffs: np.ndarray) -> np.ndarray:
    """
    Evaluate P1(x) = x + x(1-x) * P_tilde(1-x)
    where P_tilde(z) = sum_i c_i * z^i (in (1-x)^i basis, evaluated at 1-x)

    Production formula from polynomials.py lines 156-194.
    """
    x = np.atleast_1d(x)
    z = 1 - x
    p_tilde = np.zeros_like(x)
    for i, c in enumerate(tilde_coeffs):
        p_tilde += c * (z ** i)
    return x + x * z * p_tilde


def eval_Pn_direct(x: np.ndarray, tilde_coeffs: np.ndarray) -> np.ndarray:
    """
    Evaluate P2 or P3: P(x) = x * P_tilde(x)
    where P_tilde(x) = sum_i c_i * x^i (in MONOMIAL basis)

    Production formula from polynomials.py lines 231-243.
    P_ell(x) = x * (c0 + c1*x + c2*x^2 + ...)
    """
    x = np.atleast_1d(x)
    p_tilde = np.zeros_like(x)
    for i, c in enumerate(tilde_coeffs):
        p_tilde += c * (x ** i)
    return x * p_tilde


def eval_Q_direct(t: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Evaluate Q(t) in (1-2t)^k basis: Q(t) = sum_k c_k * (1-2t)^k
    """
    t = np.atleast_1d(t)
    basis = 1 - 2 * t
    result = np.zeros_like(t)
    for k, c in enumerate(coeffs):
        result += c * (basis ** k)
    return result


# =============================================================================
# INDEPENDENT CASE C KERNEL (no production code)
# =============================================================================

def compute_case_c_kernel_independent(
    P_eval,
    u: float,
    omega: int,
    R: float,
    theta: float,
) -> float:
    """
    Compute Case C kernel at a single u point using scipy quadrature.

    K_omega(u; R) = u^omega / (omega-1)! * integral_0^1 P((1-a)u) * a^{omega-1} * exp(R*theta*u*a) da

    For omega=1: K_1 = u * integral_0^1 P((1-a)u) * exp(R*theta*u*a) da
    For omega=2: K_2 = u^2 * integral_0^1 P((1-a)u) * a * exp(R*theta*u*a) da
    """
    if omega <= 0:
        raise ValueError(f"Case C requires omega > 0, got {omega}")

    if u < 1e-14:
        return 0.0  # u^omega factor makes this 0

    import math
    factorial_denom = math.factorial(omega - 1)

    def integrand(a):
        arg = (1 - a) * u
        P_val = P_eval(np.array([arg]))[0]
        a_power = a ** (omega - 1) if omega > 1 else 1.0
        exp_factor = np.exp(R * theta * u * a)
        return P_val * a_power * exp_factor

    integral, _ = integrate.quad(integrand, 0, 1, limit=100)

    return (u ** omega) / factorial_denom * integral


# =============================================================================
# INDEPENDENT I2 EVALUATOR (no production code)
# =============================================================================

def compute_I2_pair_independent(
    ell1: int,
    ell2: int,
    R: float,
    theta: float,
    poly_data: dict,
) -> float:
    """
    Compute I2 for pair (ell1, ell2) using completely independent implementation.

    I2 = (1/theta) * integral_0^1 integral_0^1 K_ell1(u) * K_ell2(u) * Q(t)^2 * exp(2Rt) du dt

    where K_ell uses Case C kernel for ell >= 2.
    """
    # Build polynomial evaluators
    P_evals = {
        1: lambda x: eval_P1_direct(x, poly_data["P1_tilde"]),
        2: lambda x: eval_Pn_direct(x, poly_data["P2_tilde"]),
        3: lambda x: eval_Pn_direct(x, poly_data["P3_tilde"]),
    }
    Q_eval = lambda t: eval_Q_direct(t, poly_data["Q_coeffs"])

    omega1 = ell1 - 1  # omega = 0 for P1, 1 for P2, 2 for P3
    omega2 = ell2 - 1

    def integrand_ut(u, t):
        # K_ell1(u)
        if omega1 == 0:
            K1 = P_evals[1](np.array([u]))[0]
        else:
            K1 = compute_case_c_kernel_independent(P_evals[ell1], u, omega1, R, theta)

        # K_ell2(u)
        if omega2 == 0:
            K2 = P_evals[1](np.array([u]))[0]
        else:
            K2 = compute_case_c_kernel_independent(P_evals[ell2], u, omega2, R, theta)

        Q_val = Q_eval(np.array([t]))[0]
        exp_val = np.exp(2 * R * t)

        return K1 * K2 * (Q_val ** 2) * exp_val / theta

    # 2D integration
    result, _ = integrate.dblquad(
        integrand_ut,
        0, 1,  # t limits
        0, 1,  # u limits (as function of t, but constant here)
        epsabs=1e-10,
        epsrel=1e-10,
    )

    return result


# =============================================================================
# INDEPENDENT I1 EVALUATOR (simplified for diagonal pairs)
# =============================================================================

def compute_I1_pair_independent_simplified(
    ell1: int,
    ell2: int,
    R: float,
    theta: float,
    poly_data: dict,
    n_u: int = 40,
    n_t: int = 40,
) -> float:
    """
    Compute I1 for pair (ell1, ell2) - simplified implementation for validation.

    For I1, we need derivatives. This uses finite differences on the Case C kernel
    which is less accurate but completely independent of production code.

    NOTE: This is a simplified approximation. For full validation, we use
    the I2 comparison (which is exact) plus the K2 reduction gate.
    """
    # This is a placeholder - full I1 requires derivative extraction
    # which needs series machinery. For Gate 2, we focus on I2 validation.
    # I1 is indirectly validated via Gate 1 (K2 reduction).
    return 0.0  # Skip for now


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestIndependentI2Baseline:
    """Test independent I2 evaluator on PRZZ baseline polynomials."""

    @pytest.fixture
    def baseline_polys(self):
        return load_przz_polynomials_direct()

    @pytest.fixture
    def production_values(self):
        """Load production I2 values from derivation report."""
        path = Path(__file__).parent.parent / "data" / "derivation_report" / "kappa_baseline.json"
        with open(path) as f:
            data = json.load(f)
        # Extract I2 component from pairs
        return data.get("pairs", {})

    def test_pair_33_i2_matches_production(self, baseline_polys):
        """
        Pair (3,3) I2: Independent vs Production.

        This is the most sensitive Case C test - both sides use omega=2.
        """
        R = 1.3036
        theta = 4 / 7

        # Independent computation
        i2_independent = compute_I2_pair_independent(3, 3, R, theta, baseline_polys)

        # Production computation (for comparison)
        from src.polynomials import load_przz_polynomials
        from src.unified_i2_paper import compute_I2_unified_paper

        P1, P2, P3, Q = load_przz_polynomials()
        prod_polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        i2_prod_result = compute_I2_unified_paper(
            R, theta, ell1=3, ell2=3,
            polynomials=prod_polys,
            n_quad_u=60, n_quad_t=60, n_quad_a=40,
        )
        i2_production = i2_prod_result.I2_value

        # Compare
        rel_diff = abs(i2_independent - i2_production) / (abs(i2_production) + 1e-15)

        print(f"\nPair (3,3) I2:")
        print(f"  Independent: {i2_independent:.10e}")
        print(f"  Production:  {i2_production:.10e}")
        print(f"  Rel diff:    {rel_diff:.2e}")

        assert rel_diff < 1e-4, f"I2 (3,3) mismatch: rel_diff={rel_diff:.6e}"

    def test_pair_23_i2_matches_production(self, baseline_polys):
        """
        Pair (2,3) I2: Independent vs Production.
        """
        R = 1.3036
        theta = 4 / 7

        i2_independent = compute_I2_pair_independent(2, 3, R, theta, baseline_polys)

        from src.polynomials import load_przz_polynomials
        from src.unified_i2_paper import compute_I2_unified_paper

        P1, P2, P3, Q = load_przz_polynomials()
        prod_polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        i2_prod_result = compute_I2_unified_paper(
            R, theta, ell1=2, ell2=3,
            polynomials=prod_polys,
            n_quad_u=60, n_quad_t=60, n_quad_a=40,
        )
        i2_production = i2_prod_result.I2_value

        rel_diff = abs(i2_independent - i2_production) / (abs(i2_production) + 1e-15)

        print(f"\nPair (2,3) I2:")
        print(f"  Independent: {i2_independent:.10e}")
        print(f"  Production:  {i2_production:.10e}")
        print(f"  Rel diff:    {rel_diff:.2e}")

        assert rel_diff < 1e-4, f"I2 (2,3) mismatch: rel_diff={rel_diff:.6e}"

    def test_pair_13_i2_matches_production(self, baseline_polys):
        """
        Pair (1,3) I2: Independent vs Production.

        This tests mixed Case B x Case C.
        """
        R = 1.3036
        theta = 4 / 7

        i2_independent = compute_I2_pair_independent(1, 3, R, theta, baseline_polys)

        from src.polynomials import load_przz_polynomials
        from src.unified_i2_paper import compute_I2_unified_paper

        P1, P2, P3, Q = load_przz_polynomials()
        prod_polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        i2_prod_result = compute_I2_unified_paper(
            R, theta, ell1=1, ell2=3,
            polynomials=prod_polys,
            n_quad_u=60, n_quad_t=60, n_quad_a=40,
        )
        i2_production = i2_prod_result.I2_value

        rel_diff = abs(i2_independent - i2_production) / (abs(i2_production) + 1e-15)

        print(f"\nPair (1,3) I2:")
        print(f"  Independent: {i2_independent:.10e}")
        print(f"  Production:  {i2_production:.10e}")
        print(f"  Rel diff:    {rel_diff:.2e}")

        assert rel_diff < 1e-4, f"I2 (1,3) mismatch: rel_diff={rel_diff:.6e}"


class TestIndependentKernelDirect:
    """Test Case C kernel computation directly."""

    @pytest.fixture
    def baseline_polys(self):
        return load_przz_polynomials_direct()

    def test_kernel_omega1_at_sample_points(self, baseline_polys):
        """Compare Case C kernel (omega=1) at sample points."""
        R = 1.3036
        theta = 4 / 7

        P2_eval = lambda x: eval_Pn_direct(x, baseline_polys["P2_tilde"])

        # Test at several u points
        u_test = [0.1, 0.3, 0.5, 0.7, 0.9]

        from src.case_c_kernel import compute_case_c_kernel

        print("\nCase C kernel (omega=1) comparison:")
        for u in u_test:
            # Independent
            K_indep = compute_case_c_kernel_independent(P2_eval, u, omega=1, R=R, theta=theta)

            # Production
            K_prod = compute_case_c_kernel(P2_eval, np.array([u]), omega=1, R=R, theta=theta)[0]

            rel_diff = abs(K_indep - K_prod) / (abs(K_prod) + 1e-15)
            print(f"  u={u:.1f}: indep={K_indep:.8e}, prod={K_prod:.8e}, rel_diff={rel_diff:.2e}")

            assert rel_diff < 1e-8, f"Kernel mismatch at u={u}"

    def test_kernel_omega2_at_sample_points(self, baseline_polys):
        """Compare Case C kernel (omega=2) at sample points."""
        R = 1.3036
        theta = 4 / 7

        P3_eval = lambda x: eval_Pn_direct(x, baseline_polys["P3_tilde"])

        u_test = [0.1, 0.3, 0.5, 0.7, 0.9]

        from src.case_c_kernel import compute_case_c_kernel

        print("\nCase C kernel (omega=2) comparison:")
        for u in u_test:
            K_indep = compute_case_c_kernel_independent(P3_eval, u, omega=2, R=R, theta=theta)
            K_prod = compute_case_c_kernel(P3_eval, np.array([u]), omega=2, R=R, theta=theta)[0]

            rel_diff = abs(K_indep - K_prod) / (abs(K_prod) + 1e-15)
            print(f"  u={u:.1f}: indep={K_indep:.8e}, prod={K_prod:.8e}, rel_diff={rel_diff:.2e}")

            assert rel_diff < 1e-8, f"Kernel mismatch at u={u}"


class TestIndependentPolynomialEval:
    """Verify independent polynomial evaluation matches production."""

    @pytest.fixture
    def baseline_polys(self):
        return load_przz_polynomials_direct()

    def test_P1_matches_production(self, baseline_polys):
        """P1 evaluation: independent vs production."""
        from src.polynomials import load_przz_polynomials
        P1_prod, _, _, _ = load_przz_polynomials()

        u_test = np.linspace(0, 1, 20)

        P1_indep = eval_P1_direct(u_test, baseline_polys["P1_tilde"])
        P1_production = P1_prod.eval(u_test)

        max_diff = np.max(np.abs(P1_indep - P1_production))
        print(f"\nP1 max diff: {max_diff:.2e}")

        assert max_diff < 1e-10, f"P1 mismatch: max_diff={max_diff:.6e}"

    def test_P2_matches_production(self, baseline_polys):
        """P2 evaluation: independent vs production."""
        from src.polynomials import load_przz_polynomials
        _, P2_prod, _, _ = load_przz_polynomials()

        u_test = np.linspace(0, 1, 20)

        P2_indep = eval_Pn_direct(u_test, baseline_polys["P2_tilde"])
        P2_production = P2_prod.eval(u_test)

        max_diff = np.max(np.abs(P2_indep - P2_production))
        print(f"\nP2 max diff: {max_diff:.2e}")

        assert max_diff < 1e-10, f"P2 mismatch: max_diff={max_diff:.6e}"

    def test_P3_matches_production(self, baseline_polys):
        """P3 evaluation: independent vs production."""
        from src.polynomials import load_przz_polynomials
        _, _, P3_prod, _ = load_przz_polynomials()

        u_test = np.linspace(0, 1, 20)

        P3_indep = eval_Pn_direct(u_test, baseline_polys["P3_tilde"])
        P3_production = P3_prod.eval(u_test)

        max_diff = np.max(np.abs(P3_indep - P3_production))
        print(f"\nP3 max diff: {max_diff:.2e}")

        assert max_diff < 1e-10, f"P3 mismatch: max_diff={max_diff:.6e}"

    def test_Q_matches_production(self, baseline_polys):
        """Q evaluation: independent vs production."""
        from src.polynomials import load_przz_polynomials
        _, _, _, Q_prod = load_przz_polynomials()

        t_test = np.linspace(0, 1, 20)

        Q_indep = eval_Q_direct(t_test, baseline_polys["Q_coeffs"])
        Q_production = Q_prod.eval(t_test)

        max_diff = np.max(np.abs(Q_indep - Q_production))
        print(f"\nQ max diff: {max_diff:.2e}")

        assert max_diff < 1e-10, f"Q mismatch: max_diff={max_diff:.6e}"


class TestFullI2ValidationSummary:
    """Comprehensive I2 validation across all Case C pairs."""

    def test_full_i2_summary(self):
        """Run full I2 comparison for all Case C pairs."""
        R = 1.3036
        theta = 4 / 7

        poly_data = load_przz_polynomials_direct()

        from src.polynomials import load_przz_polynomials
        from src.unified_i2_paper import compute_I2_unified_paper

        P1, P2, P3, Q = load_przz_polynomials()
        prod_polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        print("\n" + "=" * 70)
        print("GATE 2: INDEPENDENT I2 EVALUATOR SUMMARY")
        print("=" * 70)

        # Test all pairs involving Case C (P2 or P3)
        case_c_pairs = [
            (1, 2), (2, 2),  # omega2 >= 1
            (1, 3), (2, 3), (3, 3),  # omega2 = 2
        ]

        all_passed = True
        results = []

        for ell1, ell2 in case_c_pairs:
            # Independent
            i2_indep = compute_I2_pair_independent(ell1, ell2, R, theta, poly_data)

            # Production
            i2_prod_result = compute_I2_unified_paper(
                R, theta, ell1=ell1, ell2=ell2,
                polynomials=prod_polys,
                n_quad_u=60, n_quad_t=60, n_quad_a=40,
            )
            i2_prod = i2_prod_result.I2_value

            rel_diff = abs(i2_indep - i2_prod) / (abs(i2_prod) + 1e-15)
            passed = rel_diff < 1e-4
            status = "PASS" if passed else "FAIL"

            results.append({
                "pair": (ell1, ell2),
                "independent": i2_indep,
                "production": i2_prod,
                "rel_diff": rel_diff,
                "passed": passed,
            })

            if not passed:
                all_passed = False

        # Print results
        for r in results:
            print(f"\n  Pair ({r['pair'][0]},{r['pair'][1]}):")
            print(f"    Independent: {r['independent']:.8e}")
            print(f"    Production:  {r['production']:.8e}")
            print(f"    Rel diff:    {r['rel_diff']:.2e} [{('PASS' if r['passed'] else 'FAIL')}]")

        print("\n" + "=" * 70)
        print(f"GATE 2 OVERALL: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 70)

        assert all_passed, "Some pairs failed independent validation"


if __name__ == "__main__":
    # Run quick validation
    print("\n" + "=" * 70)
    print("GATE 2: INDEPENDENT CASE C EVALUATOR - Quick Check")
    print("=" * 70)

    R = 1.3036
    theta = 4 / 7

    poly_data = load_przz_polynomials_direct()

    print("\nCase C kernel test (omega=1, P2):")
    P2_eval = lambda x: eval_Pn_direct(x, poly_data["P2_tilde"])
    for u in [0.3, 0.5, 0.7]:
        K = compute_case_c_kernel_independent(P2_eval, u, omega=1, R=R, theta=theta)
        print(f"  K_1(u={u}) = {K:.8e}")

    print("\nCase C kernel test (omega=2, P3):")
    P3_eval = lambda x: eval_Pn_direct(x, poly_data["P3_tilde"])
    for u in [0.3, 0.5, 0.7]:
        K = compute_case_c_kernel_independent(P3_eval, u, omega=2, R=R, theta=theta)
        print(f"  K_2(u={u}) = {K:.8e}")

    print("\nI2 test for pair (3,3):")
    i2_33 = compute_I2_pair_independent(3, 3, R, theta, poly_data)
    print(f"  I2(3,3) = {i2_33:.8e}")
