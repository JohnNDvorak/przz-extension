"""
src/test_case_c_correction.py
Test Case C correction on both PRZZ benchmarks.

This tests whether adding the Case C auxiliary integral
fixes the R-dependent gap observed in (1,2) and other pairs.

PRZZ Benchmarks:
- Benchmark 1: R = 1.3036, kappa = 0.417293962, c = 2.13745440613217
- Benchmark 2: R* = 1.1167, kappa* = 0.408 (derived from R*/R = kappa*/kappa)

Key findings:
- (1,2) shows +32.32% R-sensitivity anomaly
- Other pairs show ~2-7% R-sensitivity
- Case C auxiliary integral is MISSING from our implementation
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, Tuple, Any

from src.polynomials import load_przz_polynomials
from src.evaluate import evaluate_term, evaluate_c_full, compute_kappa
from src.terms_k3_d1 import make_I1_12, make_I2_12, make_I3_12, make_I4_12
from src.quadrature import gauss_legendre_01


# PRZZ benchmark targets
THETA = 4/7
R1 = 1.3036
C1_TARGET = 2.13745440613217263636
KAPPA1_TARGET = 0.417293962

# Derived second benchmark (from R-sensitivity analysis)
R2 = 1.1167  # R* where gap is different


def compute_case_c_integral_factor(
    R: float,
    theta: float,
    omega: int,
    n_quad: int = 60
) -> float:
    """
    Compute Case C integral factor for a single piece.

    integral_0^1 (1-a)^i a^{omega-1} exp(R * theta * f(a)) da

    where f(a) encodes the R-dependence from (N/n)^{-alpha*a}.

    For leading order (i=0):
    - omega=1 (P_2): integral_0^1 exp(R*theta*a) da
    - omega=2 (P_3): integral_0^1 a * exp(R*theta*a) da

    Args:
        R: Shift parameter
        theta: Mollifier exponent
        omega: Case C parameter (omega = k-2)
        n_quad: Quadrature points

    Returns:
        Case C integral factor
    """
    nodes, weights = gauss_legendre_01(n_quad)

    i = 0  # Leading term power for (1-a)

    # Weight function: (1-a)^i * a^{omega-1}
    if omega == 1:
        # a^0 = 1
        weight = (1 - nodes) ** i  # = 1 for i=0
    elif omega == 2:
        # a^1 = a
        weight = (1 - nodes) ** i * nodes
    else:
        weight = (1 - nodes) ** i * (nodes ** (omega - 1))

    # R-dependent exponential factor
    # Key: this is the factor from (N/n)^{-alpha*a} at alpha = -R/L
    # Simplified model: exp(R * theta * a)
    exp_factor = np.exp(R * theta * nodes)

    integrand = weight * exp_factor
    return np.sum(weights * integrand)


def compute_12_with_case_c(
    R: float,
    theta: float = THETA,
    n_quad: int = 60,
    apply_correction: bool = True
) -> Dict[str, float]:
    """
    Compute (1,2) pair with optional Case C correction.

    The (1,2) pair is BxC:
    - P_1 (ell=1) is Case B (omega=0): no correction
    - P_2 (ell=2) is Case C (omega=1): needs correction

    Args:
        R: Shift parameter
        theta: Mollifier exponent
        n_quad: Quadrature points
        apply_correction: If True, apply Case C correction factor

    Returns:
        Dict with term values and correction info
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Compute standard terms
    terms = {
        "I1": make_I1_12(theta, R),
        "I2": make_I2_12(theta, R),
        "I3": make_I3_12(theta, R),
        "I4": make_I4_12(theta, R),
    }

    results = {}
    for name, term in terms.items():
        result = evaluate_term(term, polys, n_quad)
        results[f"{name}_raw"] = result.value

    # Case C correction factor for P_2 (omega=1)
    if apply_correction:
        omega_2 = 1  # For P_2
        case_c_factor = compute_case_c_integral_factor(R, theta, omega_2, n_quad)

        # The Case C factor multiplies the P_2 contributions
        # Since (1,2) is BxC, only the C piece gets the factor
        for name in ["I1", "I2", "I3", "I4"]:
            results[f"{name}_corrected"] = results[f"{name}_raw"] * case_c_factor

        results["case_c_factor"] = case_c_factor
    else:
        for name in ["I1", "I2", "I3", "I4"]:
            results[f"{name}_corrected"] = results[f"{name}_raw"]
        results["case_c_factor"] = 1.0

    results["total_raw"] = sum(results[f"{name}_raw"] for name in ["I1", "I2", "I3", "I4"])
    results["total_corrected"] = sum(results[f"{name}_corrected"] for name in ["I1", "I2", "I3", "I4"])

    return results


def test_case_c_on_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    """
    Test Case C correction on both PRZZ benchmarks.

    This is the key diagnostic: if Case C correction is the missing piece,
    it should make the R-sensitivity consistent across both benchmarks.

    Returns:
        Dict with benchmark comparison results
    """
    n_quad = 60

    # Compute at both R values, with and without correction
    r1_no_corr = compute_12_with_case_c(R1, THETA, n_quad, apply_correction=False)
    r1_with_corr = compute_12_with_case_c(R1, THETA, n_quad, apply_correction=True)

    r2_no_corr = compute_12_with_case_c(R2, THETA, n_quad, apply_correction=False)
    r2_with_corr = compute_12_with_case_c(R2, THETA, n_quad, apply_correction=True)

    # Compute R-sensitivity (change from R2 to R1)
    def sensitivity(val1, val2):
        if abs(val2) > 1e-15:
            return (val1 - val2) / abs(val2) * 100
        return 0.0

    results = {
        "R1": R1,
        "R2": R2,
        "r1_no_corr": r1_no_corr,
        "r1_with_corr": r1_with_corr,
        "r2_no_corr": r2_no_corr,
        "r2_with_corr": r2_with_corr,
        "sensitivity_raw": sensitivity(r1_no_corr["total_raw"], r2_no_corr["total_raw"]),
        "sensitivity_corrected": sensitivity(r1_with_corr["total_corrected"], r2_with_corr["total_corrected"]),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("CASE C CORRECTION TEST ON PRZZ BENCHMARKS")
        print("=" * 70)

        print(f"\nBenchmark 1: R = {R1}")
        print(f"Benchmark 2: R = {R2}")
        print(f"theta = {THETA:.6f}")

        print("\n--- Case C Correction Factors ---")
        print(f"  R = {R2:.4f}: factor = {r2_with_corr['case_c_factor']:.10f}")
        print(f"  R = {R1:.4f}: factor = {r1_with_corr['case_c_factor']:.10f}")
        factor_change = (r1_with_corr['case_c_factor'] - r2_with_corr['case_c_factor']) / r2_with_corr['case_c_factor'] * 100
        print(f"  Factor change: {factor_change:+.2f}%")

        print("\n--- (1,2) Pair Total Values ---")
        print(f"  WITHOUT correction:")
        print(f"    R={R2:.4f}: {r2_no_corr['total_raw']:+.10f}")
        print(f"    R={R1:.4f}: {r1_no_corr['total_raw']:+.10f}")
        print(f"    R-sensitivity: {results['sensitivity_raw']:+.2f}%")

        print(f"\n  WITH correction:")
        print(f"    R={R2:.4f}: {r2_with_corr['total_corrected']:+.10f}")
        print(f"    R={R1:.4f}: {r1_with_corr['total_corrected']:+.10f}")
        print(f"    R-sensitivity: {results['sensitivity_corrected']:+.2f}%")

        print("\n--- Analysis ---")
        print(f"  Original anomaly: +32.32% R-sensitivity")
        print(f"  After correction: {results['sensitivity_corrected']:+.2f}% R-sensitivity")

        if abs(results['sensitivity_corrected']) < abs(results['sensitivity_raw']) / 2:
            print("\n  *** CORRECTION REDUCES R-SENSITIVITY ***")
            print("  Case C integral is likely the missing piece!")
        elif abs(results['sensitivity_corrected']) > abs(results['sensitivity_raw']):
            print("\n  WARNING: Correction INCREASES R-sensitivity")
            print("  Simple model is wrong direction - need to revisit formula")
        else:
            print("\n  Correction has some effect but doesn't fully explain gap")
            print("  May need more sophisticated Case C model")

        print("=" * 70)

    return results


def test_full_c_with_case_c(verbose: bool = True) -> Dict[str, Any]:
    """
    Test full c computation with Case C corrections on all affected pairs.

    This applies the Case C correction to:
    - (1,2): 1 a-integral (omega=1)
    - (1,3): 1 a-integral (omega=2)
    - (2,2): 2 a-integrals (omega=1, omega=1)
    - (2,3): 2 a-integrals (omega=1, omega=2)
    - (3,3): 2 a-integrals (omega=2, omega=2)

    Returns:
        Dict with c values with and without corrections
    """
    n_quad = 60

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Compute correction factors at both R values
    def get_correction_factors(R):
        omega_1 = compute_case_c_integral_factor(R, THETA, omega=1, n_quad=n_quad)  # For P_2
        omega_2 = compute_case_c_integral_factor(R, THETA, omega=2, n_quad=n_quad)  # For P_3
        return {
            "omega_1": omega_1,
            "omega_2": omega_2,
            # Pair correction factors
            "12": omega_1,            # BxC(1)
            "13": omega_2,            # BxC(2)
            "22": omega_1 * omega_1,  # C(1)xC(1)
            "23": omega_1 * omega_2,  # C(1)xC(2)
            "33": omega_2 * omega_2,  # C(2)xC(2)
        }

    factors_r1 = get_correction_factors(R1)
    factors_r2 = get_correction_factors(R2)

    # Compute standard c at both R values
    result_r1_raw = evaluate_c_full(THETA, R1, n_quad, polys, mode="main")
    result_r2_raw = evaluate_c_full(THETA, R2, n_quad, polys, mode="main")

    if verbose:
        print("\n" + "=" * 70)
        print("FULL c WITH CASE C CORRECTIONS (PRELIMINARY)")
        print("=" * 70)

        print("\n--- Case C Correction Factors ---")
        print(f"  At R={R2:.4f}:")
        print(f"    omega=1 (P_2): {factors_r2['omega_1']:.8f}")
        print(f"    omega=2 (P_3): {factors_r2['omega_2']:.8f}")

        print(f"\n  At R={R1:.4f}:")
        print(f"    omega=1 (P_2): {factors_r1['omega_1']:.8f}")
        print(f"    omega=2 (P_3): {factors_r1['omega_2']:.8f}")

        print("\n--- Pair Correction Factors ---")
        pairs = ["12", "13", "22", "23", "33"]
        for pair in pairs:
            f1 = factors_r1[pair]
            f2 = factors_r2[pair]
            change = (f1 - f2) / f2 * 100
            print(f"  ({pair[0]},{pair[1]}): R={R2:.4f}→{R1:.4f}: {f2:.6f}→{f1:.6f} ({change:+.2f}%)")

        print("\n--- Raw c Values (before Case C correction) ---")
        print(f"  R={R2:.4f}: c = {result_r2_raw.total:.10f}")
        print(f"  R={R1:.4f}: c = {result_r1_raw.total:.10f}")
        print(f"  Target:   c = {C1_TARGET:.10f}")

        gap_r1 = (result_r1_raw.total - C1_TARGET) / C1_TARGET * 100
        print(f"\n  Gap at R={R1:.4f}: {gap_r1:+.2f}%")

        print("\n--- NOTE ---")
        print("  Full corrected c requires reimplementing pair integrands")
        print("  to include the auxiliary a-integral in the quadrature.")
        print("  This diagnostic shows the correction factors only.")
        print("=" * 70)

    return {
        "factors_r1": factors_r1,
        "factors_r2": factors_r2,
        "c_r1_raw": result_r1_raw.total,
        "c_r2_raw": result_r2_raw.total,
        "c_target": C1_TARGET,
    }


if __name__ == "__main__":
    # Test Case C correction on (1,2) pair
    test_case_c_on_benchmarks(verbose=True)

    # Test full c correction factors
    test_full_c_with_case_c(verbose=True)
