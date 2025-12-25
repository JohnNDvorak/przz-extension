"""
src/combined_identity_unified_t.py
Unified-t Combined Identity Implementation

GPT Phase 3: Correct TeX-faithful combined identity with ONE integration variable t.

KEY INSIGHT (from GPT deep review):
==================================
The TeX combined identity (lines 1508-1511) uses a SINGLE t∈[0,1] parameter that:
1. Regularizes the 1/(α+β) singularity via t-integral
2. IS the same t appearing in Q affine arguments: A_α = t + θ(t-1)x + θty

Run 20's error: Introduced separate s-integral, treating s and t as independent.
Option A's error: Finite-L evaluation diverges because 1/(α+β) = -L/(2R) grows with L.

MATHEMATICAL STRUCTURE:
======================
The TeX combined identity (before Q application):
    [N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
    = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt

At α=β=-R/L (asymptotic):
    = exp(-Rθ(x+y)) × L(1+θ(x+y)) × ∫₀¹ exp(2Rt(1+θ(x+y))) dt

After asymptotic absorption and Q application:
    Q(A_α)Q(A_β) × exp(2Rt + Rθ(2t-1)(x+y))

Note: exp(-Rθ(x+y)) × exp(2Rt(1+θ(x+y))) = exp(2Rt + Rθ(2t-1)(x+y))

KEY QUESTION (Gap A):
====================
Does the unified-t kernel have BOTH:
- (1+θ(x+y)) from log(N^{x+y}T)
- (1/θ + x + y) from algebraic prefactor

Or is one already accounted for? Note: (1/θ + x + y) = (1+θ(x+y))/θ

The sanity checks will determine this.

See docs/K_SAFE_BASELINE_LOCKDOWN.md for context.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from src.series import TruncatedSeries
from src.composition import compose_polynomial_on_affine, compose_exp_on_affine


# =============================================================================
# Sanity Check 1: Scalar Combined Identity Test
# =============================================================================

def scalar_combined_identity_lhs(z: float, s: float) -> float:
    """
    Compute (1 - z^{-s}) / s for the combined identity LHS.

    This is the difference quotient form of the combined identity.

    Args:
        z: Base (z > 0)
        s: Exponent (s != 0)

    Returns:
        (1 - z^{-s}) / s
    """
    if abs(s) < 1e-15:
        raise ValueError("s cannot be zero")
    return (1 - z**(-s)) / s


def scalar_combined_identity_rhs(z: float, s: float, n_quad: int = 50) -> float:
    """
    Compute log(z) × ∫₀¹ z^{-ts} dt for the combined identity RHS.

    This is the regularized integral form that avoids the 1/s singularity.

    Args:
        z: Base (z > 0)
        s: Exponent (s != 0)
        n_quad: Number of quadrature points

    Returns:
        log(z) × ∫₀¹ z^{-ts} dt
    """
    from src.quadrature import gauss_legendre_01

    t_nodes, t_weights = gauss_legendre_01(n_quad)

    integral = 0.0
    for t, w in zip(t_nodes, t_weights):
        integral += z**(-t * s) * w

    return np.log(z) * integral


def verify_scalar_combined_identity(z: float, s: float, n_quad: int = 50) -> Tuple[float, float, float]:
    """
    Verify the scalar combined identity:
        (1 - z^{-s})/s == log(z) × ∫₀¹ z^{-ts} dt

    Args:
        z: Base (z > 0)
        s: Exponent (s != 0)
        n_quad: Number of quadrature points

    Returns:
        (lhs, rhs, relative_error)
    """
    lhs = scalar_combined_identity_lhs(z, s)
    rhs = scalar_combined_identity_rhs(z, s, n_quad)

    rel_error = abs(lhs - rhs) / abs(lhs) if abs(lhs) > 1e-15 else abs(lhs - rhs)

    return lhs, rhs, rel_error


# =============================================================================
# Sanity Check 2: Q=1, P=1 Reduced Kernel Test
# =============================================================================

def compute_I1_reduced_kernel(
    theta: float,
    R: float,
    n: int,
    include_log_factor: bool = True,
    include_alg_prefactor: bool = True,
    verbose: bool = False
) -> float:
    """
    Compute I1 with Q=1 and P=1 to isolate the combined structure.

    This reduced kernel helps identify normalization issues:
    - If Q=1: Q(A_α)Q(A_β) = 1
    - If P=1: P(x+u)P(y+u) = 1

    The integrand becomes:
        [exp structure] × [log factor?] × [alg prefactor?]

    The exp structure is always included:
        exp(2Rt + Rθ(2t-1)(x+y))

    Toggles:
        include_log_factor: (1+θ(x+y)) from log(N^{x+y}T)
        include_alg_prefactor: (1/θ + x + y) from algebraic prefactor

    Since (1/θ + x + y) = (1+θ(x+y))/θ, having BOTH is:
        (1+θ(x+y)) × (1+θ(x+y))/θ = (1+θ(x+y))²/θ

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        n: Number of quadrature points
        include_log_factor: Include (1+θ(x+y)) factor
        include_alg_prefactor: Include (1/θ + x + y) factor
        verbose: Print diagnostic output

    Returns:
        I1 value from reduced kernel
    """
    from src.quadrature import gauss_legendre_01

    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    var_names = ("x", "y")
    I1_total = 0.0

    for i_u, (u, w_u) in enumerate(zip(u_nodes, u_weights)):
        for i_t, (t, w_t) in enumerate(zip(t_nodes, t_weights)):
            # Scalar prefactor: (1-u)² for (1,1) pair
            scalar_prefactor = (1 - u) ** 2

            # Build exp series: exp(2Rt + Rθ(2t-1)(x+y))
            # = exp(2Rt) × exp(Rθ(2t-1)(x+y))
            exp_u0 = 2 * R * t
            exp_lin_coeff = R * theta * (2 * t - 1)
            exp_lin = {"x": exp_lin_coeff, "y": exp_lin_coeff}
            exp_series = compose_exp_on_affine(1.0, exp_u0, exp_lin, var_names)

            integrand = exp_series

            # Optional: log factor (1+θ(x+y)) = 1 + θx + θy
            if include_log_factor:
                log_factor = TruncatedSeries.from_scalar(1.0, var_names)
                log_factor = log_factor + TruncatedSeries.variable("x", var_names) * theta
                log_factor = log_factor + TruncatedSeries.variable("y", var_names) * theta
                integrand = integrand * log_factor

            # Optional: algebraic prefactor (1/θ + x + y)
            if include_alg_prefactor:
                alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
                alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
                alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)
                integrand = integrand * alg_prefactor

            # Extract xy coefficient
            xy_coeff = integrand.extract(("x", "y"))

            # Integrate
            contribution = xy_coeff * scalar_prefactor * w_u * w_t
            I1_total += contribution

            if verbose and i_u == 0 and i_t < 3:
                print(f"  (u={u:.3f}, t={t:.3f}): xy_coeff={xy_coeff:.6f}")

    return I1_total


# =============================================================================
# Unified-t Combined Identity Kernel
# =============================================================================

@dataclass
class UnifiedTI1Result:
    """Result from unified-t combined identity I1 computation."""
    I1_value: float
    R: float
    theta: float
    n_quad: int
    pair: Tuple[int, int]
    details: Optional[Dict] = None


def get_unified_t_affine_coeffs(
    t: float, theta: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Get affine coefficients for A_α and A_β in the unified-t approach.

    A_α = t + θ(t-1)·x + θt·y
    A_β = t + θt·x + θ(t-1)·y

    These are the SAME affine forms as post-identity operator, because
    the t in the combined identity IS the same t in the Q arguments.

    Args:
        t: Integration variable, t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)

    Returns:
        ((u0_α, x_α, y_α), (u0_β, x_β, y_β))
    """
    u0 = t

    # A_α coefficients
    x_alpha = theta * (t - 1)
    y_alpha = theta * t

    # A_β coefficients
    x_beta = theta * t
    y_beta = theta * (t - 1)

    return (u0, x_alpha, y_alpha), (u0, x_beta, y_beta)


def apply_Q_unified_t(
    Q_poly,
    t: float,
    theta: float,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Compute Q(A_α) × Q(A_β) using unified-t affine forms.

    This is structurally identical to the post-identity operator's
    apply_Q_post_identity_composition, but reimplemented here for
    clarity and potential future divergence.

    Args:
        Q_poly: Q polynomial with eval_deriv(x, k) method
        t: Integration variable (BOTH combined identity AND Q argument)
        theta: PRZZ θ parameter (4/7)
        var_names: Variable names tuple

    Returns:
        TruncatedSeries representing Q(A_α) × Q(A_β)
    """
    (u0, x_alpha, y_alpha), (_, x_beta, y_beta) = get_unified_t_affine_coeffs(t, theta)

    lin_alpha = {"x": x_alpha, "y": y_alpha}
    lin_beta = {"x": x_beta, "y": y_beta}

    Q_alpha = compose_polynomial_on_affine(Q_poly, u0, lin_alpha, var_names)
    Q_beta = compose_polynomial_on_affine(Q_poly, u0, lin_beta, var_names)

    return Q_alpha * Q_beta


def build_unified_t_exp_series(
    t: float,
    theta: float,
    R: float,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Build the exponential factor for unified-t combined identity.

    The combined identity exp structure after asymptotic simplification:
        exp(-Rθ(x+y)) × exp(2Rt(1+θ(x+y)))
        = exp(-Rθ(x+y) + 2Rt + 2Rtθ(x+y))
        = exp(2Rt + Rθ(2t-1)(x+y))

    This is exactly exp(R*(Arg_α + Arg_β)) where Arg_α + Arg_β = 2t + θ(2t-1)(x+y).

    Args:
        t: Integration variable
        theta: PRZZ θ parameter
        R: PRZZ R parameter
        var_names: Variable names tuple

    Returns:
        TruncatedSeries for the exp factor
    """
    u0 = 2 * R * t
    lin_coeff = R * theta * (2 * t - 1)
    lin = {"x": lin_coeff, "y": lin_coeff}

    return compose_exp_on_affine(1.0, u0, lin, var_names)


def build_log_factor_series(
    theta: float,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Build the log factor (1+θ(x+y)) from log(N^{x+y}T).

    In the combined identity:
        log(N^{x+y}T) = L × (1 + θ(x+y))

    The L factor is absorbed asymptotically. This function returns
    just the (1+θ(x+y)) part.

    Note: This may duplicate the algebraic prefactor (1/θ + x + y)
    since (1+θ(x+y))/θ = 1/θ + x + y.

    Args:
        theta: PRZZ θ parameter
        var_names: Variable names tuple

    Returns:
        TruncatedSeries for (1+θ(x+y))
    """
    series = TruncatedSeries.from_scalar(1.0, var_names)
    series = series + TruncatedSeries.variable("x", var_names) * theta
    series = series + TruncatedSeries.variable("y", var_names) * theta
    return series


def compute_I1_tex_combined_unified_t_pair(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    n: int,
    polynomials: Dict,
    include_log_factor: bool = False,
    verbose: bool = False
) -> UnifiedTI1Result:
    """
    Compute I1(ℓ₁,ℓ₂) using unified-t combined identity.

    This is the CORRECT TeX-faithful implementation where the combined
    identity's t parameter IS the same t in the Q affine arguments.

    Structure at each (u,t) point:
    1. Q(A_α)×Q(A_β) with t-dependent affine forms
    2. exp(2Rt + Rθ(2t-1)(x+y)) - combined exp factor
    3. Profile factors P_ℓ₁(x+u) × P_ℓ₂(y+u)
    4. Algebraic prefactor (1/θ + x + y)
    5. Optional: log factor (1+θ(x+y)) - may be double-counting!

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        ell1: First piece index (1, 2, or 3)
        ell2: Second piece index (1, 2, or 3)
        n: Number of quadrature points
        polynomials: Dict with 'P1', 'P2', 'P3', 'Q' keys
        include_log_factor: Include (1+θ(x+y)) factor (may cause 2x overshoot)
        verbose: Print diagnostic output

    Returns:
        UnifiedTI1Result with I1 value and metadata
    """
    from src.quadrature import gauss_legendre_01
    from src.mollifier_profiles import case_c_taylor_coeffs
    from src.composition import compose_profile_on_affine

    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    Q_poly = polynomials['Q']
    P_polys = {
        1: polynomials['P1'],
        2: polynomials.get('P2'),
        3: polynomials.get('P3'),
    }

    var_names = ("x", "y")

    # Case B/C selection
    omega1 = ell1 - 1
    omega2 = ell2 - 1

    # Sign factor
    sign_factor = (-1) ** (ell1 + ell2)

    # (1-u) power
    if ell1 == 1 and ell2 == 1:
        one_minus_u_power = 2
    else:
        one_minus_u_power = max(0, (ell1 - 1) + (ell2 - 1))

    I1_total = 0.0

    for i_u, (u, w_u) in enumerate(zip(u_nodes, u_weights)):
        for i_t, (t, w_t) in enumerate(zip(t_nodes, t_weights)):
            # Scalar prefactor
            scalar_prefactor = (1 - u) ** one_minus_u_power * sign_factor

            # Step 1: Q×Q with unified-t affine forms
            QQ_series = apply_Q_unified_t(Q_poly, t, theta, var_names)

            # Step 2: Exp factor from combined identity
            exp_series = build_unified_t_exp_series(t, theta, R, var_names)

            # Core = Q×Q × exp
            core_series = QQ_series * exp_series

            # Optional: log factor (1+θ(x+y))
            if include_log_factor:
                log_series = build_log_factor_series(theta, var_names)
                core_series = core_series * log_series

            # Step 3: Profile series for ℓ₁ on x
            if omega1 == 0:
                P1_series = compose_polynomial_on_affine(
                    P_polys[ell1], u, {"x": 1.0}, var_names
                )
            else:
                taylor_coeffs = case_c_taylor_coeffs(
                    P_polys[ell1], u, omega1, R, theta, max_order=1
                )
                P1_series = compose_profile_on_affine(taylor_coeffs, {"x": 1.0}, var_names)

            # Step 4: Profile series for ℓ₂ on y
            if omega2 == 0:
                P2_series = compose_polynomial_on_affine(
                    P_polys[ell2], u, {"y": 1.0}, var_names
                )
            else:
                taylor_coeffs = case_c_taylor_coeffs(
                    P_polys[ell2], u, omega2, R, theta, max_order=1
                )
                P2_series = compose_profile_on_affine(taylor_coeffs, {"y": 1.0}, var_names)

            # Step 5: Algebraic prefactor (1/θ + x + y)
            alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

            # Step 6: Multiply all
            integrand = core_series * P1_series * P2_series * alg_prefactor

            # Step 7: Extract xy coefficient
            xy_coeff = integrand.extract(("x", "y"))

            # Step 8: Integrate
            contribution = xy_coeff * scalar_prefactor * w_u * w_t
            I1_total += contribution

            if verbose and i_u == 0 and i_t < 3:
                print(f"  (u={u:.3f}, t={t:.3f}): xy_coeff={xy_coeff:.6f}")

    if verbose:
        print(f"I1_unified_t_{ell1}{ell2} = {I1_total:.8f}")

    return UnifiedTI1Result(
        I1_value=I1_total,
        R=R,
        theta=theta,
        n_quad=n,
        pair=(ell1, ell2),
        details={
            "method": "unified_t_combined_identity",
            "include_log_factor": include_log_factor,
            "omega1": omega1,
            "omega2": omega2,
            "one_minus_u_power": one_minus_u_power,
            "sign_factor": sign_factor,
        }
    )


# =============================================================================
# Sanity Check 3: Global Factor vs Structural Comparison
# =============================================================================

def compare_unified_t_to_post_identity(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    include_log_factor: bool = False
) -> Dict:
    """
    Compare unified-t kernel to post-identity operator for all pairs.

    This is Sanity Check 3: Identify global vs structural mismatches.
    - Same constant factor across all pairs: missing normalization
    - Varying ratios by pair: structural issue

    Args:
        theta: PRZZ θ parameter
        R: PRZZ R parameter
        n: Number of quadrature points
        polynomials: Dict with polynomials
        include_log_factor: Include (1+θ(x+y)) in unified-t

    Returns:
        Dict with per-pair comparison results
    """
    from src.operator_post_identity import compute_I1_operator_post_identity_pair

    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    results = {}

    for ell1, ell2 in pairs:
        # Post-identity operator
        post_result = compute_I1_operator_post_identity_pair(
            theta, R, ell1, ell2, n, polynomials
        )
        I1_post = post_result.I1_value

        # Unified-t
        unified_result = compute_I1_tex_combined_unified_t_pair(
            theta, R, ell1, ell2, n, polynomials,
            include_log_factor=include_log_factor
        )
        I1_unified = unified_result.I1_value

        # Ratio
        ratio = I1_unified / I1_post if abs(I1_post) > 1e-15 else float('inf')

        results[(ell1, ell2)] = {
            "I1_post": I1_post,
            "I1_unified": I1_unified,
            "ratio": ratio,
        }

    # Check if all ratios are similar (global factor) or varying (structural)
    ratios = [r["ratio"] for r in results.values() if np.isfinite(r["ratio"])]
    if ratios:
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        cv = std_ratio / abs(mean_ratio) if abs(mean_ratio) > 1e-15 else float('inf')

        if cv < 0.05:
            diagnosis = "GLOBAL_FACTOR"
        else:
            diagnosis = "STRUCTURAL_MISMATCH"
    else:
        mean_ratio = float('nan')
        cv = float('nan')
        diagnosis = "UNKNOWN"

    return {
        "pairs": results,
        "mean_ratio": mean_ratio,
        "cv": cv,
        "diagnosis": diagnosis,
    }


# =============================================================================
# m1 Derivation from Unified-t
# =============================================================================

def derive_m1_from_unified_t(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    include_log_factor: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Attempt to derive m1 from unified-t combined identity.

    If unified-t gives the combined (plus + mirror) value, then:
        m1_eff = (I1_unified - I1_plus) / I1_minus_base

    Compare to:
        - Empirical: m1 = exp(R) + 5
        - Naive: m1 = exp(2R)

    Args:
        theta: PRZZ θ parameter
        R: PRZZ R parameter
        n: Number of quadrature points
        polynomials: Dict with polynomials
        include_log_factor: Include (1+θ(x+y)) in unified-t
        verbose: Print diagnostic output

    Returns:
        Dict with m1 derivation results
    """
    from src.operator_post_identity import compute_I1_operator_post_identity_pair

    # Get post-identity values (L→∞ limit)
    result_plus = compute_I1_operator_post_identity_pair(
        theta, R, 1, 1, n, polynomials
    )
    I1_plus = result_plus.I1_value

    result_minus = compute_I1_operator_post_identity_pair(
        theta, -R, 1, 1, n, polynomials
    )
    I1_minus_base = result_minus.I1_value

    # Get unified-t value
    unified_result = compute_I1_tex_combined_unified_t_pair(
        theta, R, 1, 1, n, polynomials,
        include_log_factor=include_log_factor
    )
    I1_unified = unified_result.I1_value

    # Derive m1_eff
    if abs(I1_minus_base) > 1e-15:
        m1_eff = (I1_unified - I1_plus) / I1_minus_base
    else:
        m1_eff = float('nan')

    m1_empirical = np.exp(R) + 5
    m1_naive = np.exp(2 * R)

    if verbose:
        print(f"\nm1 Derivation (include_log_factor={include_log_factor}):")
        print(f"  I1_plus = {I1_plus:.8f}")
        print(f"  I1_minus_base = {I1_minus_base:.8f}")
        print(f"  I1_unified = {I1_unified:.8f}")
        print(f"  m1_eff = {m1_eff:.4f}")
        print(f"  m1_empirical = {m1_empirical:.4f}")
        print(f"  m1_naive = {m1_naive:.4f}")
        print(f"  Ratio m1_eff/m1_empirical = {m1_eff/m1_empirical:.4f}")

    return {
        "I1_plus": I1_plus,
        "I1_minus_base": I1_minus_base,
        "I1_unified": I1_unified,
        "m1_eff": m1_eff,
        "m1_empirical": m1_empirical,
        "m1_naive": m1_naive,
        "ratio_to_empirical": m1_eff / m1_empirical if np.isfinite(m1_eff) else float('nan'),
        "ratio_to_naive": m1_eff / m1_naive if np.isfinite(m1_eff) else float('nan'),
    }
