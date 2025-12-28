"""
src/unified_s12_evaluator_v3.py
Phase 21C: Unified S12 Evaluator via True Integrand-Level Bracket

PURPOSE:
========
This module computes S12 using the CORRECT unified bracket structure where
the difference quotient identity is applied INSIDE the integrand, not by
computing at +R and -R separately.

KEY INSIGHT (from GPT 2025-12-25):
==================================
The Phase 21B approach of computing I1(+R) and I1(-R) separately and using
the symmetry I1(+R) = exp(2R)*I1(-R) is WRONG for the full polynomial case.

The correct approach is:
1. Build the unified bracket as a SINGLE OBJECT at each (u,t)
2. The bracket structure exp(2Rt + Rθ(2t-1)(x+y)) already combines direct+mirror
3. Multiply by P factors, Q factors, log factor, algebraic prefactor
4. Extract xy coefficient and integrate over (u,t)
5. D=0 should emerge from the structure, not from subtraction

MATHEMATICAL STRUCTURE:
=======================
The PRZZ difference quotient identity (Lines 1502-1511):
    [N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
    = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt

At α = β = -R/L, this becomes:
    exp(2Rt + Rθ(2t-1)(x+y)) × (1 + θ(x+y))

This is the "unified bracket" - a single object that combines both direct
and mirror contributions via the t-integral structure.

ANTI-CHEAT:
===========
This implementation does NOT:
- Compute at +R and -R separately
- Set S12_plus = 0 artificially
- Use the empirical m = exp(R) + 5 formula

REFERENCES:
===========
- src/difference_quotient.py: Core identity implementation
- src/abd_diagnostics.py: ABD decomposition
- tests/test_phase22_symmetry_ladder.py: Ladder tests
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import math
import numpy as np

from src.series import TruncatedSeries
from src.quadrature import gauss_legendre_01
from src.difference_quotient import build_bracket_exp_series, build_log_factor_series
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.abd_diagnostics import ABDDecomposition, compute_abd_decomposition


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass
class UnifiedI1ResultV3:
    """Result of unified I1 evaluation for a single pair."""
    ell1: int
    ell2: int
    I1_value: float

    # Diagnostics
    n_quad_u: int
    n_quad_t: int
    include_Q: bool


@dataclass
class UnifiedS12ResultV3:
    """Result of unified S12 evaluation using the true integrand-level bracket."""

    # Per-pair breakdown
    per_pair: Dict[str, UnifiedI1ResultV3]

    # Aggregated S12 value (sum over pairs with normalization)
    S12_total: float

    # Individual pair contributions (for diagnostics)
    pair_contributions: Dict[str, float]

    # Metadata
    R: float
    theta: float
    benchmark: str
    n_quad_u: int
    n_quad_t: int
    include_Q: bool

    # Normalization info
    normalize_scalar_baseline: bool = False  # Deprecated: use normalization_mode
    normalization_mode: str = "none"  # "none", "scalar", "diagnostic_corrected"
    scalar_baseline_factor: float = 1.0
    normalization_factor: float = 1.0  # Actual factor used for normalization
    S12_unnormalized: float = 0.0


# =============================================================================
# NORMALIZATION FACTOR
# =============================================================================


def compute_t_integral_factor(R: float) -> float:
    """
    Compute the t-integral factor F(R) = (exp(2R) - 1) / (2R).

    This is the scalar limit of the t-integral at x=y=0.

    Args:
        R: PRZZ R parameter

    Returns:
        F(R) = (exp(2R) - 1) / (2R)
    """
    if abs(R) < 1e-15:
        return 1.0  # L'Hôpital limit
    return (math.exp(2 * R) - 1) / (2 * R)


def compute_scalar_baseline_factor(R: float) -> float:
    """
    Compute the scalar baseline factor for normalizing the unified bracket.

    From first principles (PRZZ TeX Lines 1502-1511):
        The difference quotient identity is:
            [N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β) = bracket

        At α = β = -Rθ, we have (α+β) = -2Rθ.

        The bracket includes a t-integral with scalar factor:
            F(R) = ∫₀¹ exp(2Rt) dt = (exp(2R)-1)/(2R)

        The empirical approach computes:
            c = Direct + m × Mirror

        While the identity relates:
            [Direct - exp(2R)×Mirror] / (-2Rθ) = bracket

        The factor of 2 in the denominator (-2Rθ) means the unified bracket
        is inflated by F(R)/2 relative to the empirical S12_combined, not F(R).

    Args:
        R: PRZZ R parameter

    Returns:
        F(R)/2 = (exp(2R) - 1) / (4R)
    """
    return compute_t_integral_factor(R) / 2.0


def compute_xy_baseline_factor(
    R: float,
    theta: float,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
    n_quad: int = 40,
    include_Q: bool = True,
) -> float:
    """
    Compute the xy coefficient baseline normalization factor (Phase 23).

    This is F_xy(R, θ) = ∫₀¹∫₀¹ [xy coeff of full bracket] du dt

    This captures the full xy-level contribution including:
    - Log factor (1/θ + x + y) mixing with exp linear terms
    - Q eigenvalue t-dependence
    - P polynomial cross-terms

    The xy coefficient of the bracket product includes:
    - exp_xy × (1/θ) — from exp's second-order term
    - exp_x × 1 + exp_y × 1 — from exp's linear terms × log's linear terms
    - Q eigenvalue contributions through Q'(t) × eigenvalue linear coefficients

    Dividing S12 by this instead of F(R)/2 gives correct xy-level normalization,
    accounting for the log factor and Q eigenvalue effects that the scalar
    baseline normalization misses.

    Args:
        R: PRZZ R parameter
        theta: PRZZ θ parameter
        polynomials: Dictionary with P1, P2, P3, Q polynomials
        ell1: First piece index (default 1)
        ell2: Second piece index (default 1)
        n_quad: Number of quadrature points (default 40)
        include_Q: Whether to include Q polynomial factors (default True)

    Returns:
        F_xy(R, θ) = ∫₀¹∫₀¹ [xy coeff of full bracket] du dt
    """
    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    total = 0.0
    xy_mask = (1 << 0) | (1 << 1)  # binary mask for xy term

    for u, u_w in zip(u_nodes, u_weights):
        for t, t_w in zip(t_nodes, t_weights):
            series = build_unified_bracket_series(
                u, t, theta, R, ell1, ell2, polynomials,
                var_names=("x", "y"), include_Q=include_Q
            )
            xy_coeff = series.coeffs.get(xy_mask, 0.0)
            if isinstance(xy_coeff, np.ndarray):
                xy_coeff = float(xy_coeff)
            total += xy_coeff * u_w * t_w

    return total


def compute_diagnostic_correction_factor_linear_fit(R: float) -> float:
    """
    DIAGNOSTIC ONLY: Empirical correction factor from linear fit.

    WARNING: This function is QUARANTINED (Phase 24). It was derived by comparing
    unified bracket S12 to empirical S12 across both benchmarks, violating the
    "derived > tuned" discipline. It should NOT be used in production code.

    The correction factor adjusts F(R)/2 to account for non-scalar effects
    that the simple scalar baseline normalization misses:
    - Log factor (1/θ + x + y) contribution to xy coefficient
    - Q eigenvalue t-dependence effects
    - Polynomial structure interactions

    Derived from (Phase 23):
        correction(1.3036) = 0.9688  [κ benchmark]
        correction(1.1167) = 0.9545  [κ* benchmark]

    Linear fit: correction(R) = 0.8691 + 0.0765 × R

    Args:
        R: PRZZ R parameter

    Returns:
        Correction factor (multiply F(R)/2 by this to get correct normalization)

    See Also:
        - Phase 24 aims to DERIVE this correction from first principles
        - Use normalization_mode="diagnostic_corrected" with allow_diagnostic_correction=True
    """
    # Linear coefficients from empirical fit (PHASE 23 - QUARANTINED)
    a = 0.869060
    b = 0.076512
    return a + b * R


# Legacy alias for backwards compatibility during transition
compute_empirical_correction_factor = compute_diagnostic_correction_factor_linear_fit


def compute_diagnostic_corrected_baseline_factor(R: float) -> float:
    """
    DIAGNOSTIC ONLY: Compute F(R)/2 × correction(R).

    WARNING: This function is QUARANTINED (Phase 24). It uses the empirically-fitted
    correction factor, violating the "derived > tuned" discipline.

    Args:
        R: PRZZ R parameter

    Returns:
        F(R)/2 × correction(R) = diagnostic corrected normalization factor
    """
    F_scalar = compute_scalar_baseline_factor(R)
    correction = compute_diagnostic_correction_factor_linear_fit(R)
    return F_scalar * correction


# Legacy alias for backwards compatibility during transition
compute_corrected_baseline_factor = compute_diagnostic_corrected_baseline_factor


# =============================================================================
# UNIFIED BRACKET BUILDER
# =============================================================================


def build_unified_bracket_series(
    u: float,
    t: float,
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    var_names: Tuple[str, ...] = ("x", "y"),
    include_Q: bool = True,
) -> TruncatedSeries:
    """
    Build the unified bracket series with full polynomial factors at (u, t).

    PRZZ DIFFERENCE QUOTIENT IDENTITY (TeX Lines 1502-1511):
    =========================================================
    [N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
        = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt

    At α = β = -R/L = -Rθ, the RHS becomes:
        exp(-Rθ(x+y)) × L(1+θ(x+y)) × ∫₀¹ exp(2Rt(1+θ(x+y))) dt

    KEY DERIVATION (from first principles):
    =======================================
    1. N^{αx+βy} = exp(-Rθ(x+y)) at α=β=-Rθ

    2. log(N^{x+y}T) = log(T^{θ(x+y)+1}) = L(1+θ(x+y)) = (1+θ(x+y))/θ = 1/θ + x + y
       (Since L = log T = 1/θ in our normalization where θL = 1)

    3. (N^{x+y}T)^{-t(α+β)} = exp(2Rt(1+θ(x+y)))
       (Since -t(α+β) = 2Rtθ and θL = 1)

    4. Combining the exp factors:
       exp(-Rθ(x+y)) × exp(2Rt(1+θ(x+y))) = exp(2Rt + Rθ(2t-1)(x+y))

    Structure:
        exp(2Rt + Rθ(2t-1)(x+y))     [combined exp factor]
        × (1/θ + x + y)              [log factor: L(1+θ(x+y)) = 1/θ + x + y]
        × P_ell1(x+u) × P_ell2(y+u)  [P factors]
        × Q(A_α) × Q(A_β)            [Q factors with t-dependent eigenvalues]

    For Q factors, the unified bracket eigenvalues are:
        A_α = t + θ(t-1)x + θt·y
        A_β = t + θt·x + θ(t-1)·y

    CRITICAL: This bracket already combines direct and mirror contributions
    via the exp(2Rt + Rθ(2t-1)(x+y)) structure. We do NOT compute at +R and -R
    separately.
    """
    # 1. Exp factor: exp(2Rt + Rθ(2t-1)(x+y))
    #    This combines the outer exp(-Rθ(x+y)) with the t-integral's exp(2Rt(1+θ(x+y)))
    series = build_bracket_exp_series(t, theta, R, var_names)

    # 2. Log factor: log(N^{x+y}T) = L(1+θ(x+y)) = 1/θ + x + y
    #    NOTE: This is a SINGLE factor, not (1+θ(x+y)) × (1/θ + x + y)!
    #    The L = 1/θ multiplies (1+θ(x+y)) to give 1/θ + x + y.
    log_series = TruncatedSeries.from_scalar(1.0 / theta, var_names)
    log_series = log_series + TruncatedSeries.variable("x", var_names)
    log_series = log_series + TruncatedSeries.variable("y", var_names)
    series = series * log_series

    # 4. P factors: P_ell1(x+u) × P_ell2(y+u)
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")

    if P_ell1 is not None:
        # P(x+u) = P(u) + P'(u)*x + ...
        P1_val = float(P_ell1.eval(np.array([u]))[0])
        P1_deriv = float(P_ell1.eval_deriv(np.array([u]), 1)[0])
        P1_series = TruncatedSeries.from_scalar(P1_val, var_names)
        P1_series = P1_series + TruncatedSeries.variable("x", var_names) * P1_deriv
        series = series * P1_series

    if P_ell2 is not None:
        P2_val = float(P_ell2.eval(np.array([u]))[0])
        P2_deriv = float(P_ell2.eval_deriv(np.array([u]), 1)[0])
        P2_series = TruncatedSeries.from_scalar(P2_val, var_names)
        P2_series = P2_series + TruncatedSeries.variable("y", var_names) * P2_deriv
        series = series * P2_series

    # 5. Q factors: Q(A_α) × Q(A_β)
    if include_Q:
        Q = polynomials.get("Q")
        if Q is not None:
            # Unified bracket eigenvalues (t-dependent):
            # A_α = t + θ(t-1)x + θt·y
            # A_β = t + θt·x + θ(t-1)·y
            #
            # At x=y=0: A_α = A_β = t
            # Q(A_α) = Q(t) + Q'(t)[θ(t-1)x + θt·y] + ...

            Q_val = float(Q.eval(np.array([t]))[0])       # Q(t)
            Q_deriv = float(Q.eval_deriv(np.array([t]), 1)[0])  # Q'(t)

            # Eigenvalue linear coefficients for A_α
            eig_alpha_x = theta * (t - 1)  # coefficient of x in A_α
            eig_alpha_y = theta * t        # coefficient of y in A_α

            Q_alpha_series = TruncatedSeries.from_scalar(Q_val, var_names)
            Q_alpha_series = Q_alpha_series + TruncatedSeries.variable("x", var_names) * (Q_deriv * eig_alpha_x)
            Q_alpha_series = Q_alpha_series + TruncatedSeries.variable("y", var_names) * (Q_deriv * eig_alpha_y)

            # Eigenvalue linear coefficients for A_β (swapped)
            eig_beta_x = theta * t
            eig_beta_y = theta * (t - 1)

            Q_beta_series = TruncatedSeries.from_scalar(Q_val, var_names)
            Q_beta_series = Q_beta_series + TruncatedSeries.variable("x", var_names) * (Q_deriv * eig_beta_x)
            Q_beta_series = Q_beta_series + TruncatedSeries.variable("y", var_names) * (Q_deriv * eig_beta_y)

            series = series * Q_alpha_series * Q_beta_series

    return series


# =============================================================================
# UNIFIED I1 EVALUATOR
# =============================================================================


def compute_I1_unified_v3(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad_u: int = 40,
    n_quad_t: int = 40,
    include_Q: bool = True,
) -> UnifiedI1ResultV3:
    """
    Compute I1 for pair (ell1, ell2) using the unified bracket.

    This builds the unified bracket at each (u,t), extracts the xy coefficient,
    and integrates. The result is a SINGLE value that represents the combined
    direct+mirror contribution.

    CRITICAL: We do NOT compute at +R and -R separately. The unified bracket
    already incorporates both via the exp(2Rt + Rθ(2t-1)(x+y)) structure.

    PRZZ PREFACTOR (Phase 25 fix):
    The integral includes (1-u)^{ℓ₁+ℓ₂} from the Euler-Maclaurin summation.
    See PRZZ TeX line 1435 for I₁, lines 2391-2396 for derivation.
    """
    var_names = ("x", "y")
    xy_mask = (1 << 0) | (1 << 1)

    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    # PRZZ (1-u) power from Euler-Maclaurin: (1-u)^{ℓ₁+ℓ₂} for I₁
    one_minus_u_power = ell1 + ell2

    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        # Compute (1-u)^{ℓ₁+ℓ₂} prefactor
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power

        for t, t_w in zip(t_nodes, t_weights):
            series = build_unified_bracket_series(
                u, t, theta, R, ell1, ell2, polynomials, var_names, include_Q
            )
            xy_coeff = series.coeffs.get(xy_mask, 0.0)
            if isinstance(xy_coeff, np.ndarray):
                xy_coeff = float(xy_coeff)
            # Apply (1-u)^{ℓ₁+ℓ₂} prefactor
            total += xy_coeff * one_minus_u_factor * u_w * t_w

    return UnifiedI1ResultV3(
        ell1=ell1,
        ell2=ell2,
        I1_value=total,
        n_quad_u=n_quad_u,
        n_quad_t=n_quad_t,
        include_Q=include_Q,
    )


# =============================================================================
# UNIFIED S12 EVALUATOR
# =============================================================================


# Triangle pairs with symmetry factors
TRIANGLE_PAIRS: List[Tuple[str, int, int, float]] = [
    ("11", 1, 1, 1.0),   # (key, ell1, ell2, symmetry_factor)
    ("22", 2, 2, 1.0),
    ("33", 3, 3, 1.0),
    ("12", 1, 2, 2.0),   # 2x for off-diagonal
    ("13", 1, 3, 2.0),
    ("23", 2, 3, 2.0),
]


def factorial_norm(ell1: int, ell2: int) -> float:
    """Compute factorial normalization 1/(ell1! × ell2!)."""
    return 1.0 / (math.factorial(ell1) * math.factorial(ell2))


def compute_S12_unified_v3(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad_u: int = 40,
    n_quad_t: int = 40,
    include_Q: bool = True,
    use_factorial_normalization: bool = True,
    benchmark: str = "unified_v3",
    normalize_scalar_baseline: bool = False,
    normalization_mode: str = "auto",
    allow_diagnostic_correction: bool = False,
) -> UnifiedS12ResultV3:
    """
    Compute S12 using the unified bracket for ALL 6 triangle pairs.

    This is the main entry point for Phase 21C/22/23/24. It:
    1. Computes I1 for each pair using the unified bracket
    2. Applies factorial normalization and symmetry factors
    3. Normalizes according to normalization_mode
    4. Returns the aggregated S12 value

    CRITICAL: This does NOT compute at +R and -R separately. The unified
    bracket already incorporates both contributions via the t-integral.

    NORMALIZATION MODES (Phase 24):
    ===============================
    - "none": No normalization (raw unified bracket)
    - "scalar": Divide by F(R)/2 = (exp(2R)-1)/(4R) [Phase 22]
    - "diagnostic_corrected": QUARANTINED. Divide by F(R)/2 × correction(R).
      Requires allow_diagnostic_correction=True. [Phase 23 - empirically fitted]
    - "auto": Use "diagnostic_corrected" if normalize_scalar_baseline=True AND
      allow_diagnostic_correction=True, else "scalar" if normalize_scalar_baseline=True,
      else "none"

    LEGACY MODE (deprecated):
    - "corrected": Alias for "diagnostic_corrected" (for backwards compatibility)

    Phase 24 aims to DERIVE the correction factor from first principles,
    replacing the empirically-fitted correction.

    Args:
        R: PRZZ R parameter
        theta: PRZZ θ parameter
        polynomials: Dictionary of P₁, P₂, P₃, Q polynomials
        n_quad_u: Number of quadrature points for u-integration
        n_quad_t: Number of quadrature points for t-integration
        include_Q: Whether to include Q polynomial factors
        use_factorial_normalization: Whether to apply 1/(ell1! * ell2!)
        benchmark: Benchmark name for labeling
        normalize_scalar_baseline: Deprecated, use normalization_mode
        normalization_mode: One of "none", "scalar", "diagnostic_corrected", "auto"
        allow_diagnostic_correction: Must be True to use "diagnostic_corrected" mode

    Returns:
        UnifiedS12ResultV3 with S12 value and diagnostics

    Raises:
        ValueError: If diagnostic_corrected is used without allow_diagnostic_correction=True
    """
    per_pair = {}
    pair_contributions = {}
    S12_total = 0.0

    for pair_key, ell1, ell2, symmetry in TRIANGLE_PAIRS:
        # Compute I1 for this pair
        I1_result = compute_I1_unified_v3(
            R=R,
            theta=theta,
            ell1=ell1,
            ell2=ell2,
            polynomials=polynomials,
            n_quad_u=n_quad_u,
            n_quad_t=n_quad_t,
            include_Q=include_Q,
        )

        # Apply normalization
        if use_factorial_normalization:
            norm = factorial_norm(ell1, ell2)
        else:
            norm = 1.0

        contrib = norm * symmetry * I1_result.I1_value

        per_pair[pair_key] = I1_result
        pair_contributions[pair_key] = contrib
        S12_total += contrib

    # Store unnormalized value
    S12_unnormalized = S12_total

    # Handle legacy "corrected" mode name → "diagnostic_corrected"
    if normalization_mode == "corrected":
        normalization_mode = "diagnostic_corrected"

    # Determine effective normalization mode
    if normalization_mode == "auto":
        if normalize_scalar_baseline:
            if allow_diagnostic_correction:
                effective_mode = "diagnostic_corrected"
            else:
                effective_mode = "scalar"
        else:
            effective_mode = "none"
    else:
        effective_mode = normalization_mode

    # Guard: diagnostic_corrected requires explicit opt-in
    if effective_mode == "diagnostic_corrected" and not allow_diagnostic_correction:
        raise ValueError(
            "normalization_mode='diagnostic_corrected' requires allow_diagnostic_correction=True. "
            "This mode uses empirically-fitted correction (Phase 23) which violates "
            "'derived > tuned' discipline. Use 'scalar' for first-principles normalization, "
            "or set allow_diagnostic_correction=True for diagnostic purposes only."
        )

    # Compute normalization factors
    scalar_baseline_factor = compute_scalar_baseline_factor(R)

    # Determine normalization factor based on mode
    if effective_mode == "none":
        normalization_factor = 1.0
    elif effective_mode == "scalar":
        normalization_factor = scalar_baseline_factor
    elif effective_mode == "diagnostic_corrected":
        normalization_factor = compute_diagnostic_corrected_baseline_factor(R)
    else:
        raise ValueError(f"Unknown normalization_mode: {normalization_mode}")

    # Apply normalization
    if effective_mode != "none":
        S12_total = S12_total / normalization_factor
        # Also normalize per-pair contributions for consistency
        for key in pair_contributions:
            pair_contributions[key] = pair_contributions[key] / normalization_factor

    return UnifiedS12ResultV3(
        per_pair=per_pair,
        S12_total=S12_total,
        pair_contributions=pair_contributions,
        R=R,
        theta=theta,
        benchmark=benchmark,
        n_quad_u=n_quad_u,
        n_quad_t=n_quad_t,
        include_Q=include_Q,
        normalize_scalar_baseline=normalize_scalar_baseline,
        normalization_mode=effective_mode,
        scalar_baseline_factor=scalar_baseline_factor,
        normalization_factor=normalization_factor,
        S12_unnormalized=S12_unnormalized,
    )


# =============================================================================
# DUAL BENCHMARK RUNNER
# =============================================================================


def run_dual_benchmark_v3(
    n_quad_u: int = 40,
    n_quad_t: int = 40,
    include_Q: bool = True,
    normalize_scalar_baseline: bool = False,
    normalization_mode: str = "auto",
    allow_diagnostic_correction: bool = False,
) -> Tuple[UnifiedS12ResultV3, UnifiedS12ResultV3]:
    """
    Run unified S12 computation on both benchmarks.

    Args:
        n_quad_u: Number of quadrature points for u-integration
        n_quad_t: Number of quadrature points for t-integration
        include_Q: Whether to include Q polynomial factors
        normalize_scalar_baseline: Deprecated, use normalization_mode
        normalization_mode: One of "none", "scalar", "diagnostic_corrected", "auto"
        allow_diagnostic_correction: Must be True to use "diagnostic_corrected" mode

    Returns:
        (kappa_result, kappa_star_result)
    """
    theta = 4.0 / 7.0

    # Load polynomials for each benchmark
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials()
    kappa_polys = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    kappa_star_polys = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    kappa = compute_S12_unified_v3(
        R=1.3036,
        theta=theta,
        polynomials=kappa_polys,
        n_quad_u=n_quad_u,
        n_quad_t=n_quad_t,
        include_Q=include_Q,
        benchmark="kappa_v3",
        normalize_scalar_baseline=normalize_scalar_baseline,
        normalization_mode=normalization_mode,
        allow_diagnostic_correction=allow_diagnostic_correction,
    )

    kappa_star = compute_S12_unified_v3(
        R=1.1167,
        theta=theta,
        polynomials=kappa_star_polys,
        n_quad_u=n_quad_u,
        n_quad_t=n_quad_t,
        include_Q=include_Q,
        benchmark="kappa_star_v3",
        normalize_scalar_baseline=normalize_scalar_baseline,
        normalization_mode=normalization_mode,
        allow_diagnostic_correction=allow_diagnostic_correction,
    )

    return kappa, kappa_star


# =============================================================================
# MAIN EXECUTION
# =============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED S12 EVALUATOR V3 - TRUE INTEGRAND-LEVEL BRACKET")
    print("Phase 21C/22: No +R/-R splitting, with scalar baseline normalization")
    print("=" * 70)

    # Run with normalization to compare
    print("\n*** UNNORMALIZED (raw unified bracket) ***")
    kappa_raw, kappa_star_raw = run_dual_benchmark_v3(
        include_Q=True, normalize_scalar_baseline=False
    )

    print("\n*** NORMALIZED (divided by F(R)) ***")
    kappa_norm, kappa_star_norm = run_dual_benchmark_v3(
        include_Q=True, normalize_scalar_baseline=True
    )

    for (raw, norm, benchmark_name, R, c_target) in [
        (kappa_raw, kappa_norm, "kappa", 1.3036, 2.137),
        (kappa_star_raw, kappa_star_norm, "kappa_star", 1.1167, 1.938),
    ]:
        print(f"\n{'='*70}")
        print(f"Benchmark: {benchmark_name.upper()} (R={R})")
        print("=" * 70)

        print("\n--- Normalization Info ---")
        print(f"  F(R) = (exp(2R)-1)/(2R): {norm.scalar_baseline_factor:.6f}")

        print("\n--- S12 Values ---")
        print(f"  S12 (unnormalized): {raw.S12_total:.6f}")
        print(f"  S12 (normalized):   {norm.S12_total:.6f}")
        print(f"  Ratio (should be F(R)): {raw.S12_total / norm.S12_total:.6f}")

        print("\n--- Per-Pair Breakdown (Normalized) ---")
        for pair_key in ["11", "22", "33", "12", "13", "23"]:
            contrib = norm.pair_contributions[pair_key]
            print(f"  {pair_key}: {contrib:.6e}")

        print(f"\n--- c Estimate (S12 alone, no I₃₄) ---")
        print(f"  c target: {c_target:.6f}")
        print(f"  Note: Full c requires I₃₄ contribution")

        print(f"\n--- Settings ---")
        print(f"  n_quad_u: {norm.n_quad_u}")
        print(f"  n_quad_t: {norm.n_quad_t}")
        print(f"  include_Q: {norm.include_Q}")
        print(f"  normalize_scalar_baseline: {norm.normalize_scalar_baseline}")

    print("\n" + "=" * 70)
    print("PHASE 22 NOTES:")
    print("- Normalization divides by F(R) = (exp(2R)-1)/(2R)")
    print("- This removes the t-integral scalar inflation factor")
    print("- D=0 and B/A=5 should still hold after normalization")
    print("=" * 70)
