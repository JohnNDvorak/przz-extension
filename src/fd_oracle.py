"""
src/fd_oracle.py
Finite-Difference Oracle for PRZZ Prefactor Validation

This module implements independent validation of DSL derivative extraction
using finite differences with Richardson extrapolation for stability.

Purpose: Validate that our DSL prefactor (-1/θ for I₃/I₄) is correct
at the variable stage we're using.

PRZZ Reference: Lines 1562-1563
- I₃ = -TΦ̂(0) × (1+θx)/θ × d/dx[...]|_{x=0}
- At x=0: (1+θx)/θ = 1/θ, so prefactor is -1/θ

Decision Matrix:
- DSL ≈ FD × (-1/θ) → Paper prefactor correct
- DSL ≈ FD × (-1) → Our variables scaled relative to PRZZ
- DSL ≠ either → Bug in DSL derivative extraction

HARD GATE: Do NOT proceed with audit until this resolves.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Callable, Tuple

from src.quadrature import tensor_grid_2d, gauss_legendre_01
from src.polynomials import load_przz_polynomials


# =============================================================================
# Core Oracle Functions
# =============================================================================

def compute_F_I3_11(
    x: float,
    P1,  # P₁ polynomial
    Q,   # Q polynomial
    theta: float,
    R: float,
    n_quad: int
) -> float:
    """
    Compute I3 integrand as function of formal variable x.

    PRZZ equation reference: Lines 1562-1563
    Variable stage: x is raw (unscaled) per PRZZ display

    F(x) = ∫∫ (1-u) P₁(x+u) P₁(u) exp(R[t+θtx]) exp(R[-θx+t+θtx])
           × Q(t+θtx) Q(-θx+t+θtx) dt du

    WITHOUT derivative extraction or prefactor.

    Args:
        x: Formal variable value
        P1: P₁ polynomial
        Q: Q polynomial
        theta: θ parameter
        R: R parameter
        n_quad: Number of quadrature points per dimension

    Returns:
        Value of F(x)
    """
    # Build 2D quadrature grid
    U, T, W = tensor_grid_2d(n_quad)

    # Arguments (from PRZZ lines 1562-1563)
    # Arg_α at y=0: t + θt·x = t(1 + θx)
    # Arg_β at y=0: -θx + t + θtx = t - θx(1-t) = t + θx(t-1)
    Arg_alpha = T + theta * T * x  # t(1 + θx)
    Arg_beta = T + theta * x * (T - 1)  # t + θx(t-1)

    # Polynomial factors
    P1_shifted = P1.eval(x + U)  # P₁(x + u)
    P1_unshifted = P1.eval(U)    # P₁(u)
    Q_alpha = Q.eval(Arg_alpha)
    Q_beta = Q.eval(Arg_beta)

    # Exponential factors
    exp_alpha = np.exp(R * Arg_alpha)
    exp_beta = np.exp(R * Arg_beta)

    # Poly prefactor: (1-u)
    poly_pref = 1 - U

    # Full integrand
    integrand = poly_pref * P1_shifted * P1_unshifted * Q_alpha * Q_beta * exp_alpha * exp_beta

    # Integrate
    return float(np.sum(W * integrand))


def richardson_derivative(
    F: Callable[[float], float],
    h0: float = 1e-3,
    order: int = 4
) -> Tuple[float, np.ndarray]:
    """
    Compute d/dx F(0) using Richardson extrapolation.

    Uses central differences with successive halving and extrapolation
    for stability and accuracy.

    Args:
        F: Function to differentiate
        h0: Initial step size
        order: Number of extrapolation levels

    Returns:
        Tuple of (best estimate, array of raw estimates)
    """
    estimates = []

    for i in range(order):
        h = h0 / (2 ** i)
        # Central difference
        fd = (F(h) - F(-h)) / (2 * h)
        estimates.append(fd)

    estimates = np.array(estimates)

    # Richardson extrapolation (for central differences, error is O(h²))
    # R_{i,j} = R_{i,j-1} + (R_{i,j-1} - R_{i-1,j-1}) / (4^j - 1)
    R = np.zeros((order, order))
    R[:, 0] = estimates

    for j in range(1, order):
        for i in range(j, order):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

    # Best estimate is R[order-1, order-1]
    return R[order-1, order-1], estimates


def five_point_derivative(
    F: Callable[[float], float],
    h: float = 1e-4
) -> float:
    """
    Compute d/dx F(0) using 5-point stencil.

    f'(0) ≈ (-f(2h) + 8f(h) - 8f(-h) + f(-2h)) / (12h)

    Error: O(h⁴)

    Args:
        F: Function to differentiate
        h: Step size

    Returns:
        Derivative estimate
    """
    return (-F(2*h) + 8*F(h) - 8*F(-h) + F(-2*h)) / (12 * h)


# =============================================================================
# Validation Tests
# =============================================================================

def validate_I3_prefactor(
    theta: float = 4/7,
    R: float = 1.3036,
    n_quad: int = 80,
    verbose: bool = True
) -> Dict:
    """
    Validate I₃ prefactor using FD oracle.

    Compares DSL value to FD × (-1/θ) and FD × (-1).

    Args:
        theta: θ parameter
        R: R parameter
        n_quad: Quadrature points (fix high to reduce quad error)
        verbose: Print detailed output

    Returns:
        Dict with validation results
    """
    from src.terms_k3_d1 import make_I3_11
    from src.evaluate import evaluate_term

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Get DSL value
    term = make_I3_11(theta, R)
    dsl_result = evaluate_term(term, polys, n_quad)
    dsl_value = dsl_result.value

    # Build F(x) function
    def F(x):
        return compute_F_I3_11(x, P1, Q, theta, R, n_quad)

    # Compute FD derivatives
    richardson_deriv, raw_estimates = richardson_derivative(F)
    five_pt_deriv = five_point_derivative(F)

    # Expected values under different prefactors
    fd_with_paper_prefactor = (-1/theta) * richardson_deriv  # -1/θ
    fd_with_alt_prefactor = (-1) * richardson_deriv          # -1

    # Compute relative errors
    if abs(dsl_value) > 1e-15:
        err_paper = abs(dsl_value - fd_with_paper_prefactor) / abs(dsl_value)
        err_alt = abs(dsl_value - fd_with_alt_prefactor) / abs(dsl_value)
    else:
        err_paper = abs(dsl_value - fd_with_paper_prefactor)
        err_alt = abs(dsl_value - fd_with_alt_prefactor)

    results = {
        "dsl_value": dsl_value,
        "fd_derivative": richardson_deriv,
        "fd_paper_prefactor": fd_with_paper_prefactor,
        "fd_alt_prefactor": fd_with_alt_prefactor,
        "rel_err_paper": err_paper,
        "rel_err_alt": err_alt,
        "five_pt_deriv": five_pt_deriv,
        "raw_estimates": raw_estimates,
        "theta": theta,
        "R": R,
        "n_quad": n_quad,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("FD ORACLE: I₃ PREFACTOR VALIDATION")
        print("=" * 70)
        print(f"\nConfig: θ = {theta:.10f}, R = {R}, n = {n_quad}")
        print(f"\nPRZZ Reference: Lines 1562-1563")
        print(f"  I₃ = -TΦ̂(0) × (1+θx)/θ × d/dx[...]|_(x=0)")
        print(f"  At x=0: (1+θx)/θ = 1/θ = {1/theta:.10f}")

        print(f"\n--- Derivative Estimates ---")
        print(f"  Raw FD estimates (h halving):")
        for i, est in enumerate(raw_estimates):
            h = 1e-3 / (2**i)
            print(f"    h = {h:.0e}: {est:+.12f}")
        print(f"  Richardson extrapolated: {richardson_deriv:+.12f}")
        print(f"  5-point stencil:         {five_pt_deriv:+.12f}")

        print(f"\n--- DSL vs FD Comparison ---")
        print(f"  DSL value (with prefactor):  {dsl_value:+.12f}")
        print(f"  FD × (-1/θ):                 {fd_with_paper_prefactor:+.12f}")
        print(f"  FD × (-1):                   {fd_with_alt_prefactor:+.12f}")

        print(f"\n--- Relative Errors ---")
        print(f"  |DSL - FD×(-1/θ)| / |DSL| = {err_paper:.6e}")
        print(f"  |DSL - FD×(-1)|   / |DSL| = {err_alt:.6e}")

        print(f"\n--- Decision ---")
        if err_paper < 1e-4:
            print(f"  ✓ DSL ≈ FD × (-1/θ): Paper prefactor VALIDATED")
            print(f"    The -1/θ prefactor is correct at our variable stage.")
            results["verdict"] = "paper_correct"
        elif err_alt < 1e-4:
            print(f"  ⚠️  DSL ≈ FD × (-1): Alternative prefactor matches")
            print(f"    Our variables may be scaled relative to PRZZ.")
            print(f"    Investigate x → x·log(N) rescaling.")
            results["verdict"] = "alt_correct"
        else:
            print(f"  ❌ DSL ≠ either FD estimate: Bug in DSL or oracle")
            print(f"    Check derivative extraction or oracle implementation.")
            results["verdict"] = "neither"

        print("\n" + "=" * 70)

    return results


def convergence_study(
    theta: float = 4/7,
    R: float = 1.3036,
    n_values: list = None,
    h_values: list = None
) -> Dict:
    """
    Study convergence in both quadrature n and step size h.

    Args:
        theta: θ parameter
        R: R parameter
        n_values: List of quadrature points to test
        h_values: List of step sizes to test

    Returns:
        Dict with convergence data
    """
    if n_values is None:
        n_values = [40, 60, 80, 100]
    if h_values is None:
        h_values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

    from src.terms_k3_d1 import make_I3_11
    from src.evaluate import evaluate_term

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    print("\n" + "=" * 70)
    print("FD ORACLE: CONVERGENCE STUDY")
    print("=" * 70)

    # Convergence in n (fix h)
    print("\n--- Convergence in quadrature n (h = 1e-4 fixed) ---")
    h_fixed = 1e-4
    n_results = {}
    for n in n_values:
        def F(x):
            return compute_F_I3_11(x, P1, Q, theta, R, n)
        fd_deriv = five_point_derivative(F, h_fixed)

        term = make_I3_11(theta, R)
        dsl_val = evaluate_term(term, polys, n).value
        fd_paper = (-1/theta) * fd_deriv

        n_results[n] = {
            "dsl": dsl_val,
            "fd_paper": fd_paper,
            "diff": dsl_val - fd_paper,
        }
        print(f"  n={n:3d}: DSL={dsl_val:+.10f}, FD×(-1/θ)={fd_paper:+.10f}, diff={dsl_val-fd_paper:+.6e}")

    # Convergence in h (fix n)
    print("\n--- Convergence in step h (n = 80 fixed) ---")
    n_fixed = 80
    def F(x):
        return compute_F_I3_11(x, P1, Q, theta, R, n_fixed)

    term = make_I3_11(theta, R)
    dsl_val = evaluate_term(term, polys, n_fixed).value

    h_results = {}
    for h in h_values:
        fd_deriv = five_point_derivative(F, h)
        fd_paper = (-1/theta) * fd_deriv
        h_results[h] = {
            "fd_deriv": fd_deriv,
            "fd_paper": fd_paper,
            "diff": dsl_val - fd_paper,
        }
        print(f"  h={h:.0e}: FD×(-1/θ)={fd_paper:+.10f}, diff={dsl_val-fd_paper:+.6e}")

    print(f"\n  DSL value (n=80): {dsl_val:+.10f}")
    print("\n" + "=" * 70)

    return {"n_results": n_results, "h_results": h_results, "dsl_n80": dsl_val}


if __name__ == "__main__":
    # Run validation
    results = validate_I3_prefactor(verbose=True)

    # Run convergence study
    conv_results = convergence_study()
