"""
src/diagnostics/q_correction_formula.py
Phase 40: Analytical Q Correction Formula

This module derives the δ_Q correction to the mirror multiplier formula:

    m = [1 + θ/(2K(2K+1)) + δ_Q] × [exp(R) + (2K-1)]

The δ_Q correction accounts for Q polynomial derivative effects that create
the ±0.15% residual in the base derived formula.

DERIVATION:
===========
The residual arises because:
1. I1 uses Q(Arg_α) × Q(Arg_β) with x,y-dependent arguments
2. I2 uses Q(t)² (frozen, no derivatives)
3. The d²/dxdy extraction produces Q' terms: Q'(t)² × 2θ²t(t-1)
4. This integrates to ~-0.137 (raw), but is heavily diluted

The dilution factor f ≈ 0.007 (from 22.8% raw to 0.15% final).

FORMULA:
========
δ_Q = λ × θ² × ⟨Q'² × t(t-1) × exp(2Rt)⟩ / ⟨Q² × exp(2Rt)⟩ × f(K)

where:
- λ is a structure constant (~-0.03)
- f(K) is the K-dependent dilution factor

Created: 2025-12-27 (Phase 40)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


@dataclass
class QMoments:
    """Q polynomial moments for δ_Q computation."""

    # Basic moments (unweighted)
    Q_sq_avg: float           # ⟨Q²⟩
    Q_prime_sq_avg: float     # ⟨Q'²⟩
    Q_prime_t_t1_avg: float   # ⟨Q'² × t(t-1)⟩

    # R-weighted moments
    Q_sq_exp_avg: float           # ⟨Q² × exp(2Rt)⟩
    Q_prime_t_t1_exp_avg: float   # ⟨Q'² × t(t-1) × exp(2Rt)⟩

    # Derived ratios
    @property
    def unweighted_moment_ratio(self) -> float:
        """⟨Q'² × t(t-1)⟩ / ⟨Q²⟩"""
        return self.Q_prime_t_t1_avg / self.Q_sq_avg if self.Q_sq_avg > 1e-15 else 0.0

    @property
    def weighted_moment_ratio(self) -> float:
        """⟨Q'² × t(t-1) × exp(2Rt)⟩ / ⟨Q² × exp(2Rt)⟩"""
        return self.Q_prime_t_t1_exp_avg / self.Q_sq_exp_avg if self.Q_sq_exp_avg > 1e-15 else 0.0


def compute_q_moments(
    Q_coeffs: List[float],
    R: float,
    n_quad: int = 100,
) -> QMoments:
    """
    Compute Q polynomial moments needed for δ_Q formula.

    Args:
        Q_coeffs: Q polynomial coefficients in PRZZ basis (1-2x)^k
        R: R parameter for exp(2Rt) weighting
        n_quad: Quadrature points

    Returns:
        QMoments with all computed values
    """
    from src.quadrature import gauss_legendre_01

    # Setup quadrature on [0, 1]
    t_vals, w_vals = gauss_legendre_01(n_quad)

    # Evaluate Q(t) and Q'(t) at all quadrature points
    Q_vals = np.zeros(n_quad)
    Q_prime_vals = np.zeros(n_quad)

    for i, t in enumerate(t_vals):
        Q_vals[i] = _eval_Q(Q_coeffs, t)
        Q_prime_vals[i] = _eval_Q_prime(Q_coeffs, t)

    # Compute moments
    Q_sq = Q_vals ** 2
    Q_prime_sq = Q_prime_vals ** 2
    t_t1 = t_vals * (t_vals - 1)  # t(t-1), always negative on (0,1)
    exp_2Rt = np.exp(2 * R * t_vals)

    # Basic moments
    Q_sq_avg = np.sum(w_vals * Q_sq)
    Q_prime_sq_avg = np.sum(w_vals * Q_prime_sq)
    Q_prime_t_t1_avg = np.sum(w_vals * Q_prime_sq * t_t1)

    # R-weighted moments
    Q_sq_exp_avg = np.sum(w_vals * Q_sq * exp_2Rt)
    Q_prime_t_t1_exp_avg = np.sum(w_vals * Q_prime_sq * t_t1 * exp_2Rt)

    return QMoments(
        Q_sq_avg=Q_sq_avg,
        Q_prime_sq_avg=Q_prime_sq_avg,
        Q_prime_t_t1_avg=Q_prime_t_t1_avg,
        Q_sq_exp_avg=Q_sq_exp_avg,
        Q_prime_t_t1_exp_avg=Q_prime_t_t1_exp_avg,
    )


def _eval_Q(Q_coeffs: List[float], t: float) -> float:
    """Evaluate Q(t) from PRZZ basis coefficients."""
    # Q(t) = Σ c_k × (1-2t)^k
    z = 1 - 2 * t
    result = 0.0
    z_power = 1.0
    for c in Q_coeffs:
        result += c * z_power
        z_power *= z
    return result


def _eval_Q_prime(Q_coeffs: List[float], t: float) -> float:
    """Evaluate Q'(t) from PRZZ basis coefficients."""
    # Q(t) = Σ c_k × (1-2t)^k
    # Q'(t) = Σ c_k × k × (1-2t)^(k-1) × (-2)
    z = 1 - 2 * t
    result = 0.0
    for k, c in enumerate(Q_coeffs):
        if k > 0:
            result += c * k * (z ** (k - 1)) * (-2)
    return result


def compute_effective_delta_Q(
    R: float,
    theta: float,
    K: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> float:
    """
    Compute effective δ_Q from frozen-Q experiment.

    This measures the actual Q derivative effect on the correction factor
    by comparing frozen-Q (Q(t)²) vs normal-Q (Q(Arg)) computations.

    Args:
        R: R parameter
        theta: θ parameter
        K: Number of mollifier pieces
        polynomials: Dict with P1, P2, P3, Q
        n_quad: Quadrature points

    Returns:
        Effective δ_Q (should be negative and small, ~-0.0015)
    """
    from src.diagnostics.q_residual import compute_q_residual_diagnostic

    diag = compute_q_residual_diagnostic(
        R=R, theta=theta, K=K, polynomials=polynomials,
        benchmark_name="effective_delta_Q", n_quad=n_quad,
    )

    # The Q effect on correction ratio is already computed
    ratio_effect = diag.Q_effect_on_correction_pct / 100

    # Scale to the Beta correction
    # The ratio effect modifies the correction factor, so:
    # m_actual = m_frozen × (1 + ratio_effect)
    # We want δ_Q such that:
    # [1 + β + δ_Q] = [1 + β] × (1 + ratio_effect)
    # δ_Q = [1 + β] × ratio_effect
    beta = theta / (2 * K * (2 * K + 1))
    delta_Q = (1 + beta) * ratio_effect

    return delta_Q


def fit_structure_constant(
    benchmarks: List[Tuple[float, float, int, Dict, float]],
    theta: float = 4/7,
) -> float:
    """
    Fit the structure constant λ from benchmark data.

    The formula is:
        δ_Q = λ × θ² × weighted_moment_ratio

    We solve for λ using least squares on the benchmarks.

    Args:
        benchmarks: List of (R, theta, K, polynomials, measured_delta_Q)
        theta: θ parameter

    Returns:
        Fitted λ value
    """
    # Collect data points
    moment_ratios = []
    delta_Qs = []

    for R, _, K, polynomials, measured_delta_Q in benchmarks:
        Q = polynomials["Q"]
        Q_coeffs = _get_Q_coeffs(Q)
        moments = compute_q_moments(Q_coeffs, R)
        moment_ratios.append(moments.weighted_moment_ratio)
        delta_Qs.append(measured_delta_Q)

    # Solve: δ_Q = λ × θ² × moment_ratio
    # λ = δ_Q / (θ² × moment_ratio)
    theta_sq = theta ** 2
    lambdas = []
    for mr, dq in zip(moment_ratios, delta_Qs):
        if abs(mr) > 1e-15:
            lambdas.append(dq / (theta_sq * mr))

    if lambdas:
        return np.mean(lambdas)
    return 0.0


def compute_delta_Q_analytical(
    R: float,
    theta: float,
    K: int,
    Q_coeffs: List[float],
    lambda_: float = -0.03,  # Default from fitting
) -> float:
    """
    Compute δ_Q analytically from Q polynomial structure.

    FORMULA:
        δ_Q = λ × θ² × ⟨Q'² × t(t-1) × exp(2Rt)⟩ / ⟨Q² × exp(2Rt)⟩

    Args:
        R: R parameter
        theta: θ parameter
        K: Number of mollifier pieces (currently unused, for future K-scaling)
        Q_coeffs: Q polynomial coefficients in PRZZ basis
        lambda_: Structure constant (default fitted from κ/κ* benchmarks)

    Returns:
        Analytical δ_Q correction
    """
    moments = compute_q_moments(Q_coeffs, R)
    delta_Q = lambda_ * (theta ** 2) * moments.weighted_moment_ratio
    return delta_Q


def _get_Q_coeffs(Q) -> List[float]:
    """Extract Q coefficients from polynomial object in PRZZ basis (1-2x)^k."""
    # Handle QPolynomial type with basis_coeffs dict
    if hasattr(Q, 'basis_coeffs'):
        # QPolynomial: basis_coeffs is {k: c_k} for (1-2x)^k
        basis = Q.basis_coeffs
        if not basis:
            return [0.0]
        max_k = max(basis.keys())
        coeffs = [basis.get(k, 0.0) for k in range(max_k + 1)]
        return coeffs
    # Handle numpy Polynomial
    elif hasattr(Q, 'coef'):
        return list(Q.coef)
    elif hasattr(Q, 'coefficients'):
        return list(Q.coefficients)
    else:
        raise ValueError(f"Unknown Q polynomial type: {type(Q)}")


@dataclass
class DeltaQResult:
    """Result of δ_Q computation with diagnostics."""

    delta_Q: float              # The δ_Q correction value
    method: str                 # "empirical" or "analytical"

    # Q moments (if analytical)
    moments: Optional[QMoments] = None

    # Effective values (if empirical)
    ratio_effect_pct: Optional[float] = None

    # Structure constant (if analytical)
    lambda_: Optional[float] = None


def compute_delta_Q_with_diagnostics(
    R: float,
    theta: float,
    K: int,
    polynomials: Dict,
    method: str = "empirical",
    lambda_: float = -0.03,
    n_quad: int = 60,
) -> DeltaQResult:
    """
    Compute δ_Q with full diagnostics.

    Args:
        R: R parameter
        theta: θ parameter
        K: Number of mollifier pieces
        polynomials: Dict with P1, P2, P3, Q
        method: "empirical" (from frozen-Q experiment) or "analytical" (from formula)
        lambda_: Structure constant for analytical method
        n_quad: Quadrature points

    Returns:
        DeltaQResult with δ_Q and diagnostics
    """
    Q = polynomials["Q"]
    Q_coeffs = _get_Q_coeffs(Q)

    if method == "empirical":
        delta_Q = compute_effective_delta_Q(R, theta, K, polynomials, n_quad)

        from src.diagnostics.q_residual import compute_q_residual_diagnostic
        diag = compute_q_residual_diagnostic(
            R=R, theta=theta, K=K, polynomials=polynomials,
            benchmark_name="diagnostics", n_quad=n_quad,
        )

        return DeltaQResult(
            delta_Q=delta_Q,
            method="empirical",
            ratio_effect_pct=diag.Q_effect_on_correction_pct,
        )

    elif method == "analytical":
        moments = compute_q_moments(Q_coeffs, R)
        delta_Q = compute_delta_Q_analytical(R, theta, K, Q_coeffs, lambda_)

        return DeltaQResult(
            delta_Q=delta_Q,
            method="analytical",
            moments=moments,
            lambda_=lambda_,
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def print_delta_Q_report(result: DeltaQResult, benchmark_name: str = "") -> None:
    """Print a formatted δ_Q diagnostic report."""
    print()
    print("=" * 60)
    print(f"δ_Q CORRECTION REPORT{': ' + benchmark_name if benchmark_name else ''}")
    print("=" * 60)
    print(f"Method: {result.method}")
    print(f"δ_Q: {result.delta_Q:+.6f}")
    print()

    if result.method == "empirical" and result.ratio_effect_pct is not None:
        print("EMPIRICAL MEASUREMENT")
        print("-" * 40)
        print(f"  Q effect on ratio: {result.ratio_effect_pct:+.4f}%")

    elif result.method == "analytical" and result.moments is not None:
        print("ANALYTICAL COMPUTATION")
        print("-" * 40)
        print(f"  λ (structure constant): {result.lambda_:+.4f}")
        print(f"  θ²: {(4/7)**2:.4f}")
        print(f"  ⟨Q²⟩: {result.moments.Q_sq_avg:.6f}")
        print(f"  ⟨Q'²⟩: {result.moments.Q_prime_sq_avg:.6f}")
        print(f"  ⟨Q'² × t(t-1)⟩: {result.moments.Q_prime_t_t1_avg:.6f}")
        print(f"  ⟨Q² × exp(2Rt)⟩: {result.moments.Q_sq_exp_avg:.6f}")
        print(f"  ⟨Q'² × t(t-1) × exp(2Rt)⟩: {result.moments.Q_prime_t_t1_exp_avg:.6f}")
        print()
        print(f"  Unweighted moment ratio: {result.moments.unweighted_moment_ratio:.6f}")
        print(f"  Weighted moment ratio: {result.moments.weighted_moment_ratio:.6f}")

    print()
    print(f"CORRECTION FACTOR EFFECT")
    print("-" * 40)
    print(f"  δ_Q adds {result.delta_Q * 100:+.4f}% to correction factor")
    print()


if __name__ == "__main__":
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

    theta = 4/7
    K = 3

    print("=" * 70)
    print("PHASE 40: δ_Q CORRECTION ANALYSIS")
    print("=" * 70)

    # κ benchmark
    P1, P2, P3, Q = load_przz_polynomials()
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    result_kappa_emp = compute_delta_Q_with_diagnostics(
        R=1.3036, theta=theta, K=K, polynomials=polys_kappa,
        method="empirical",
    )
    print_delta_Q_report(result_kappa_emp, "κ (empirical)")

    result_kappa_ana = compute_delta_Q_with_diagnostics(
        R=1.3036, theta=theta, K=K, polynomials=polys_kappa,
        method="analytical", lambda_=-0.03,
    )
    print_delta_Q_report(result_kappa_ana, "κ (analytical)")

    # κ* benchmark
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    polys_star = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    result_star_emp = compute_delta_Q_with_diagnostics(
        R=1.1167, theta=theta, K=K, polynomials=polys_star,
        method="empirical",
    )
    print_delta_Q_report(result_star_emp, "κ* (empirical)")

    result_star_ana = compute_delta_Q_with_diagnostics(
        R=1.1167, theta=theta, K=K, polynomials=polys_star,
        method="analytical", lambda_=-0.03,
    )
    print_delta_Q_report(result_star_ana, "κ* (analytical)")

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"κ benchmark:")
    print(f"  Empirical δ_Q: {result_kappa_emp.delta_Q:+.6f}")
    print(f"  Analytical δ_Q: {result_kappa_ana.delta_Q:+.6f}")
    print()
    print(f"κ* benchmark:")
    print(f"  Empirical δ_Q: {result_star_emp.delta_Q:+.6f}")
    print(f"  Analytical δ_Q: {result_star_ana.delta_Q:+.6f}")
    print()
