"""
src/operator_post_identity.py
Post-Identity Operator Approach for PRZZ Q Application

This module implements GPT's "post-identity" operator approach, which applies
Q(D_α)Q(D_β) AFTER the PRZZ combined identity has introduced t-dependence.

KEY INSIGHT (from GPT Step 2 analysis):
---------------------------------------
The pre-identity bracket B(α,β,x,y) lacks the integration variable t.
This t comes from PRZZ's combined identity s-integral and enters the
affine forms as:
    Q_α = t + θt·x + (θt-θ)·y
    Q_β = t + (θt-θ)·x + θt·y

The (θt-θ) terms create asymmetry between x and y coefficients,
which is essential for the correct xy coefficient in nilpotent algebra.

MATHEMATICAL STRUCTURE:
-----------------------
We define the post-identity exponential core:
    E(α,β;x,y,t) = exp(θL(αx+βy)) · exp(-t(α+β)L(1+θ(x+y)))

This is the exact shape that, under D_α, D_β (where D = -1/L × d/d...),
yields affine forms with the -θ shifts.

KEY PROPERTY (eigenvalue form):
    D_α^n E = A_α^n E  where A_α = D_α log E
    D_β^n E = A_β^n E  where A_β = D_β log E

So: Q(D_α)Q(D_β)E = Q(A_α)Q(A_β)E

The A_α, A_β are affine in x, y with the crucial (θt-θ) cross-terms.

NOTE: This module keeps src/operator_level_mirror.py intact as a
"pre-identity divergence diagnostic" for Step 2's original experiment.

See docs/OPERATOR_VS_COMPOSITION.md for full mathematical derivation.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass

from src.composition import compose_polynomial_on_affine
from src.series import TruncatedSeries


# =============================================================================
# Step 1: Post-Identity Exponential Core E(α,β;x,y,t)
# =============================================================================

def compute_post_identity_core_E(
    alpha: float,
    beta: float,
    x: float,
    y: float,
    t: float,
    theta: float,
    L: float
) -> float:
    """
    Compute the post-identity exponential core.

    E(α,β;x,y,t) = exp(θL(αx+βy)) · exp(-t(α+β)L(1+θ(x+y)))

    This is the minimal α,β-dependent factor from the PRZZ combined identity
    that, under D_α, D_β operators, yields the correct affine forms with
    (θt-θ) shifts.

    IMPORTANT: This does NOT include the 1/(α+β) factor that was Step 2's trap.

    Args:
        alpha: First Mellin variable, typically -R/L
        beta: Second Mellin variable, typically -R/L
        x, y: Formal series variables (for numerical verification)
        t: Integration variable from combined identity, t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)
        L: log T (asymptotic parameter)

    Returns:
        E(α,β;x,y,t) as a float
    """
    term1 = theta * L * (alpha * x + beta * y)
    term2 = -t * (alpha + beta) * L * (1 + theta * (x + y))
    return np.exp(term1 + term2)


# =============================================================================
# Step 2: Operator Multipliers A_α, A_β (Closed-Form Eigenvalues)
# =============================================================================

def compute_A_alpha(x: float, y: float, t: float, theta: float) -> float:
    """
    Compute A_α = D_α log E (the "eigenvalue" for D_α acting on E).

    Mathematical derivation:
        log E = θL(αx+βy) - t(α+β)L(1+θ(x+y))

        D_α log E = -1/L × d/dα [log E]
                  = -1/L × [θLx - tL(1+θ(x+y))]
                  = -θx + t(1+θ(x+y))
                  = t(1+θ(x+y)) - θx

    Expanded to first order in nilpotent x, y:
        A_α = t + θ(t-1)x + θt·y

    This matches tex_mirror's arg_β structure (note the swap).

    Args:
        x, y: Variables (numeric for verification, affine coefficients for composition)
        t: Integration variable, t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)

    Returns:
        A_α(x, y, t) as a float
    """
    return t * (1 + theta * (x + y)) - theta * x


def compute_A_beta(x: float, y: float, t: float, theta: float) -> float:
    """
    Compute A_β = D_β log E (the "eigenvalue" for D_β acting on E).

    Mathematical derivation:
        log E = θL(αx+βy) - t(α+β)L(1+θ(x+y))

        D_β log E = -1/L × d/dβ [log E]
                  = -1/L × [θLy - tL(1+θ(x+y))]
                  = -θy + t(1+θ(x+y))
                  = t(1+θ(x+y)) - θy

    Expanded to first order in nilpotent x, y:
        A_β = t + θt·x + θ(t-1)·y

    This matches tex_mirror's arg_α structure (note the swap).

    Args:
        x, y: Variables (numeric for verification, affine coefficients for composition)
        t: Integration variable, t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)

    Returns:
        A_β(x, y, t) as a float
    """
    return t * (1 + theta * (x + y)) - theta * y


def get_A_alpha_affine_coeffs(t: float, theta: float) -> Tuple[float, float, float]:
    """
    Extract the affine coefficients for A_α.

    A_α = t + θ(t-1)·x + θt·y

    Returns:
        (u0, x_coeff, y_coeff) where A_α = u0 + x_coeff·x + y_coeff·y
    """
    u0 = t
    x_coeff = theta * (t - 1)
    y_coeff = theta * t
    return u0, x_coeff, y_coeff


def get_A_beta_affine_coeffs(t: float, theta: float) -> Tuple[float, float, float]:
    """
    Extract the affine coefficients for A_β.

    A_β = t + θt·x + θ(t-1)·y

    Returns:
        (u0, x_coeff, y_coeff) where A_β = u0 + x_coeff·x + y_coeff·y
    """
    u0 = t
    x_coeff = theta * t
    y_coeff = theta * (t - 1)
    return u0, x_coeff, y_coeff


# =============================================================================
# Step 3: Operator Application (Two Paths)
# =============================================================================

# Path A: Closed-Form Composition (Main Implementation)

def apply_Q_post_identity_composition(
    Q_poly,
    t: float,
    theta: float,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Compute Q(A_α) × Q(A_β) via composition on affine forms.

    NOTE: This returns Q×Q ONLY (no exp factor).
    For the full core including exp series, use apply_QQexp_post_identity_composition().

    Args:
        Q_poly: Q polynomial with eval_deriv(x, k) method
        t: Integration variable, t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)
        var_names: Variable names tuple, default ("x", "y")

    Returns:
        TruncatedSeries representing Q(A_α) × Q(A_β)
    """
    # Build A_α affine form
    u0_alpha, x_coeff_alpha, y_coeff_alpha = get_A_alpha_affine_coeffs(t, theta)
    lin_alpha = {"x": x_coeff_alpha, "y": y_coeff_alpha}

    # Build A_β affine form
    u0_beta, x_coeff_beta, y_coeff_beta = get_A_beta_affine_coeffs(t, theta)
    lin_beta = {"x": x_coeff_beta, "y": y_coeff_beta}

    # Compose Q on each affine form
    Q_alpha_series = compose_polynomial_on_affine(Q_poly, u0_alpha, lin_alpha, var_names)
    Q_beta_series = compose_polynomial_on_affine(Q_poly, u0_beta, lin_beta, var_names)

    # Multiply series
    return Q_alpha_series * Q_beta_series


def get_exp_affine_coeffs(t: float, theta: float, R: float) -> Tuple[float, float, float]:
    """
    Get the affine coefficients for exp(R*(Arg_α + Arg_β)).

    The DSL exp factor has:
        u0 = 2*R*t
        lin_x = R*(2*theta*t - theta)
        lin_y = R*(2*theta*t - theta)

    This is because Arg_α + Arg_β = 2t + θ(2t-1)*(x+y) at the symmetric point,
    and we multiply by R.

    Returns:
        (u0, x_coeff, y_coeff) for the exp series
    """
    u0 = 2 * R * t
    lin_coeff = R * (2 * theta * t - theta)  # = R*θ*(2t-1)
    return u0, lin_coeff, lin_coeff


def apply_QQexp_post_identity_composition(
    Q_poly,
    t: float,
    theta: float,
    R: float,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Compute Q(A_α) × Q(A_β) × exp(R*(Arg_α + Arg_β)) via composition.

    This is the COMPLETE post-identity core including the exp factor.
    The exp factor is NOT a scalar prefactor - it has x/y dependence
    and contributes to the xy coefficient.

    The exp series uses:
        u0 = 2*R*t
        lin = {"x": R*(2θt-θ), "y": R*(2θt-θ)}

    Args:
        Q_poly: Q polynomial with eval_deriv(x, k) method
        t: Integration variable, t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        var_names: Variable names tuple, default ("x", "y")

    Returns:
        TruncatedSeries representing Q(A_α) × Q(A_β) × exp(R*(Arg_α+Arg_β))
    """
    from src.composition import compose_exp_on_affine

    # Build Q×Q series
    QQ_series = apply_Q_post_identity_composition(Q_poly, t, theta, var_names)

    # Build exp series with proper affine coefficients
    exp_u0, exp_x_coeff, exp_y_coeff = get_exp_affine_coeffs(t, theta, R)
    exp_lin = {"x": exp_x_coeff, "y": exp_y_coeff}

    # compose_exp_on_affine computes exp(R * (u0/R + δ)) = exp(u0 + R*δ)
    # But our u0 is already 2*R*t and lin is already R*(2θt-θ)
    # So we use R=1.0 and pass the pre-scaled u0 and lin
    exp_series = compose_exp_on_affine(1.0, exp_u0, exp_lin, var_names)

    # Multiply Q×Q by exp series
    return QQ_series * exp_series


# Path B: Definition-Level Operator Sum (Verification Only)

def apply_Q_post_identity_operator_sum(
    Q_mono: list,
    x_val: float,
    y_val: float,
    t: float,
    theta: float
) -> float:
    """
    Compute Q(D_α)Q(D_β)E via explicit operator sum (for verification).

    Uses the eigenvalue property: D_α^n E = A_α^n E

    So: Q(D_α)Q(D_β)E = Σᵢ Σⱼ qᵢ qⱼ A_α^i A_β^j × E

    For numeric x, y (NOT nilpotent), this computes the scalar result.

    Args:
        Q_mono: Q polynomial in monomial form [q_0, q_1, q_2, ..., q_n]
        x_val, y_val: Numeric x, y values for evaluation
        t: Integration variable, t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)

    Returns:
        Σᵢ Σⱼ qᵢ qⱼ A_α^i A_β^j as a float
    """
    A_alpha = compute_A_alpha(x_val, y_val, t, theta)
    A_beta = compute_A_beta(x_val, y_val, t, theta)

    result = 0.0
    for i, qi in enumerate(Q_mono):
        if abs(qi) < 1e-15:
            continue
        for j, qj in enumerate(Q_mono):
            if abs(qj) < 1e-15:
                continue
            result += qi * qj * (A_alpha ** i) * (A_beta ** j)

    return result


def evaluate_Q_product_at_affine(
    Q_poly,
    x_val: float,
    y_val: float,
    t: float,
    theta: float
) -> float:
    """
    Evaluate Q(A_α) × Q(A_β) at numeric x, y values.

    This is a convenience function for testing the composition path
    against the operator sum path.

    Args:
        Q_poly: Q polynomial with eval(x) method
        x_val, y_val: Numeric x, y values
        t: Integration variable
        theta: PRZZ θ parameter

    Returns:
        Q(A_α(x,y,t)) × Q(A_β(x,y,t)) as a float
    """
    A_alpha = compute_A_alpha(x_val, y_val, t, theta)
    A_beta = compute_A_beta(x_val, y_val, t, theta)

    Q_alpha = Q_poly.eval(np.array([A_alpha]))[0]
    Q_beta = Q_poly.eval(np.array([A_beta]))[0]

    return Q_alpha * Q_beta


# =============================================================================
# L-Stability Verification
# =============================================================================

def evaluate_operator_applied_core(
    alpha: float,
    beta: float,
    x_val: float,
    y_val: float,
    t: float,
    theta: float,
    L: float,
    Q_mono: list
) -> float:
    """
    Evaluate Q(D_α)Q(D_β)E at given parameters.

    Uses the eigenvalue property:
        Q(D_α)Q(D_β)E = Q(A_α)Q(A_β) × E

    This should be STABLE in L (no L-divergence).

    Args:
        alpha, beta: Mellin variables, typically -R/L
        x_val, y_val: Numeric x, y values
        t: Integration variable
        theta: PRZZ θ parameter
        L: log T
        Q_mono: Q polynomial in monomial form

    Returns:
        Q(A_α)Q(A_β) × E as a float
    """
    # Compute E
    E = compute_post_identity_core_E(alpha, beta, x_val, y_val, t, theta, L)

    # Compute Q(A_α)Q(A_β)
    Q_product = apply_Q_post_identity_operator_sum(Q_mono, x_val, y_val, t, theta)

    return Q_product * E


# =============================================================================
# Helper: Convert Q from basis to monomial form
# =============================================================================

def convert_Q_basis_to_monomial(basis_coeffs: Dict[int, float]) -> list:
    """
    Convert Q from (1-2x)^k basis to monomial form.

    Q(x) = Σₖ cₖ(1-2x)^k → Σⱼ qⱼx^j

    Uses binomial expansion: (1-2x)^k = Σⱼ C(k,j)(-2)^j x^j

    Args:
        basis_coeffs: Dict mapping power k to coefficient c_k

    Returns:
        List [q_0, q_1, ..., q_max_k] of monomial coefficients
    """
    if not basis_coeffs:
        return [0.0]

    max_k = max(basis_coeffs.keys())
    mono = [0.0] * (max_k + 1)

    for k, c_k in basis_coeffs.items():
        for j in range(k + 1):
            binom = _binomial(k, j)
            mono[j] += c_k * binom * ((-2) ** j)

    return mono


def _binomial(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


# =============================================================================
# Exp Composition Helper
# =============================================================================

def compose_exp_on_affine_post_identity(
    R: float,
    t: float,
    theta: float,
    which: str,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Compute exp(R × A_α) or exp(R × A_β) as a TruncatedSeries.

    Args:
        R: PRZZ R parameter
        t: Integration variable
        theta: PRZZ θ parameter
        which: "alpha" or "beta"
        var_names: Variable names tuple

    Returns:
        TruncatedSeries for exp(R × A)
    """
    from src.composition import compose_exp_on_affine

    if which == "alpha":
        u0, x_coeff, y_coeff = get_A_alpha_affine_coeffs(t, theta)
    elif which == "beta":
        u0, x_coeff, y_coeff = get_A_beta_affine_coeffs(t, theta)
    else:
        raise ValueError(f"which must be 'alpha' or 'beta', got {which}")

    lin = {"x": x_coeff, "y": y_coeff}
    return compose_exp_on_affine(R, u0, lin, var_names)


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class PostIdentityI1Result:
    """Result from post-identity operator I1 computation."""
    I1_value: float
    R: float
    theta: float
    n_quad: int
    details: Optional[Dict] = None


# =============================================================================
# Step 6: Full I1(1,1) Computation
# =============================================================================

def compute_I1_operator_post_identity_11(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    verbose: bool = False
) -> PostIdentityI1Result:
    """
    Compute I1(1,1) using the post-identity operator approach.

    This uses the EXACT SAME series-multiplication approach as the DSL:
    1. Build Q×Q×exp core series (with proper exp x/y dependence)
    2. Build P₁(x+u) and P₁(y+u) series
    3. Build algebraic prefactor series (1/θ + x + y)
    4. Multiply all series
    5. Extract ("x", "y") coefficient
    6. Multiply by scalar (1-u)² and integrate

    The exp factor is NOT a scalar prefactor - it has x/y dependence
    and contributes to the xy coefficient via:
        u0 = 2*R*t
        lin = {"x": R*(2θt-θ), "y": R*(2θt-θ)}

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter (1.3036)
        n: Number of quadrature points
        polynomials: Dict with 'P1', 'Q' keys
        verbose: Print diagnostic output

    Returns:
        PostIdentityI1Result with I1 value and metadata
    """
    from src.quadrature import gauss_legendre_01

    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    Q_poly = polynomials['Q']
    P1 = polynomials['P1']

    var_names = ("x", "y")
    I1_total = 0.0

    for i_u, (u, w_u) in enumerate(zip(u_nodes, u_weights)):
        for i_t, (t, w_t) in enumerate(zip(t_nodes, t_weights)):
            # Scalar prefactor: (1-u)² only
            # NO exp prefactor here - the exp is in the series
            scalar_prefactor = (1 - u) ** 2

            # Step 1: Build Q(A_α)Q(A_β) × exp(R*(Arg_α+Arg_β)) core series
            # This is the complete core with exp x/y dependence
            core_series = apply_QQexp_post_identity_composition(
                Q_poly, t, theta, R, var_names
            )

            # Step 2: Build P₁(x+u) series
            P1_x_series = compose_polynomial_on_affine(
                P1, u, {"x": 1.0}, var_names
            )

            # Step 3: Build P₁(y+u) series
            P1_y_series = compose_polynomial_on_affine(
                P1, u, {"y": 1.0}, var_names
            )

            # Step 4: Build algebraic prefactor series (1/θ + x + y)
            alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

            # Step 5: Multiply all series
            integrand = core_series * P1_x_series * P1_y_series * alg_prefactor

            # Step 6: Extract ("x", "y") coefficient
            xy_coeff = integrand.extract(("x", "y"))

            # Step 7: Multiply by scalar prefactor and integrate
            contribution = xy_coeff * scalar_prefactor * w_u * w_t
            I1_total += contribution

            if verbose and i_u == 0 and i_t < 3:
                print(f"  (u={u:.3f}, t={t:.3f}): xy_coeff={xy_coeff:.6f}, scalar_pref={scalar_prefactor:.6f}")

    if verbose:
        print(f"I1_operator_post_identity_11 = {I1_total:.8f}")

    return PostIdentityI1Result(
        I1_value=I1_total,
        R=R,
        theta=theta,
        n_quad=n,
        details={
            "method": "operator_post_identity",
            "var_names": var_names,
        }
    )


# =============================================================================
# Step 7: Analytic Coefficient Computation (GPT Phase 1, Step 2)
# =============================================================================

def compute_analytic_QQexp_coeffs(
    Q_poly,
    t: float,
    theta: float,
    R: float
) -> Tuple[float, float, float, float]:
    """
    Compute {c00, cx, cy, cxy} for Q(A_α)Q(A_β)exp(...) ANALYTICALLY.

    This is the SKEPTIC-KILLER function: computes the nilpotent series coefficients
    using closed-form analytic formulas, NOT the series engine.

    Mathematical derivation:
    -----------------------
    For nilpotent A = t + a_x·x + a_y·y:
        Q(A) = Q(t + δ) where δ = a_x·x + a_y·y
        Q(A) = Q(t) + Q'(t)·δ + Q''(t)/2!·δ² + ...

    Under nilpotent rules (x² = y² = 0):
        δ² = 2·a_x·a_y·xy

    So Q(A) has coefficients:
        const: Q(t)
        x:     Q'(t)·a_x
        y:     Q'(t)·a_y
        xy:    Q''(t)·a_x·a_y  (the /2 from Taylor cancels the 2 from δ²)

    For the product Q(A_α)×Q(A_β), use nilpotent algebra.

    For exp(u0 + b·x + b·y) under nilpotent rules:
        const: exp(u0)
        x:     exp(u0)·b
        y:     exp(u0)·b
        xy:    exp(u0)·b²

    Args:
        Q_poly: Q polynomial with eval() and eval_deriv() methods
        t: Integration variable, t ∈ [0, 1]
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter

    Returns:
        (C00, Cx, Cy, Cxy) - the four nilpotent series coefficients
    """
    t_arr = np.array([t])
    Q0 = Q_poly.eval(t_arr)[0]               # Q(t)
    Q1 = Q_poly.eval_deriv(t_arr, 1)[0]      # Q'(t)
    Q2 = Q_poly.eval_deriv(t_arr, 2)[0]      # Q''(t)

    # A_α = t + θ(t-1)x + θt·y
    ax_alpha = theta * (t - 1)
    ay_alpha = theta * t

    # A_β = t + θt·x + θ(t-1)·y
    ax_beta = theta * t
    ay_beta = theta * (t - 1)

    # Q(A_α) series coefficients
    q00_alpha = Q0
    qx_alpha = Q1 * ax_alpha
    qy_alpha = Q1 * ay_alpha
    qxy_alpha = Q2 * ax_alpha * ay_alpha  # /2 from Taylor × 2 from δ² = 1

    # Q(A_β) series coefficients
    q00_beta = Q0
    qx_beta = Q1 * ax_beta
    qy_beta = Q1 * ay_beta
    qxy_beta = Q2 * ax_beta * ay_beta

    # Q×Q nilpotent product: (a + bx + cy + dxy)(e + fx + gy + hxy)
    # = ae + (af+be)x + (ag+ce)y + (ah+bg+cf+de)xy
    QQ00 = q00_alpha * q00_beta
    QQx = q00_alpha * qx_beta + qx_alpha * q00_beta
    QQy = q00_alpha * qy_beta + qy_alpha * q00_beta
    QQxy = (q00_alpha * qxy_beta + qx_alpha * qy_beta +
            qy_alpha * qx_beta + qxy_alpha * q00_beta)

    # Exp series: exp(u0 + b·x + b·y)
    # where u0 = 2Rt, b = R(2θt - θ) = Rθ(2t-1)
    b = R * (2 * theta * t - theta)
    e0 = np.exp(2 * R * t)
    ex = e0 * b
    ey = e0 * b
    exy = e0 * b * b

    # Final: QQ × exp (nilpotent product)
    C00 = QQ00 * e0
    Cx = QQ00 * ex + QQx * e0
    Cy = QQ00 * ey + QQy * e0
    Cxy = QQ00 * exy + QQx * ey + QQy * ex + QQxy * e0

    return C00, Cx, Cy, Cxy


# =============================================================================
# Step 8: Generalized I1 Computation for Any (ℓ₁, ℓ₂) Pair
# =============================================================================

def _build_profile_series(
    poly,
    omega: int,
    u: float,
    R: float,
    theta: float,
    var_name: str,
    var_names: Tuple[str, ...]
) -> TruncatedSeries:
    """
    Build profile series K(u+δ) for Case B or Case C.

    For Case B (ω=0): P(u+δ) via compose_polynomial_on_affine
    For Case C (ω>0): K_ω(u+δ;R,θ) via case_c_taylor_coeffs + compose_profile_on_affine

    Args:
        poly: Polynomial object (P₁, P₂, or P₃)
        omega: Case indicator (0=Case B, >0=Case C)
        u: Base point
        R: PRZZ R parameter
        theta: PRZZ θ parameter
        var_name: Which variable to use ("x" or "y")
        var_names: Full variable names tuple

    Returns:
        TruncatedSeries representing the profile series
    """
    from src.mollifier_profiles import case_c_taylor_coeffs
    from src.composition import compose_polynomial_on_affine, compose_profile_on_affine

    lin = {var_name: 1.0}  # δ = 1·var_name

    if omega == 0:
        # Case B: direct polynomial composition
        return compose_polynomial_on_affine(poly, u, lin, var_names)
    else:
        # Case C: get K_ω Taylor coefficients and compose
        # max_order=1 because we only need {const, var} terms
        taylor_coeffs = case_c_taylor_coeffs(
            poly, u, omega, R, theta, max_order=1
        )
        return compose_profile_on_affine(taylor_coeffs, lin, var_names)


def compute_I1_operator_post_identity_pair(
    theta: float,
    R: float,
    ell1: int,
    ell2: int,
    n: int,
    polynomials: Dict,
    verbose: bool = False
) -> PostIdentityI1Result:
    """
    Compute I1(ℓ₁,ℓ₂) using the post-identity operator approach for any pair.

    This generalizes I1(1,1) to Case-C pairs with ℓ ∈ {2, 3}.

    The structure follows the DSL's _make_I1_generic_v2:
    - Variables: (x, y) with derivative d²/dxdy
    - Sign: (-1)^{ℓ₁+ℓ₂}
    - Power: (1-u)^{max(0, (ℓ₁-1) + (ℓ₂-1))}
    - Profile factors:
      - ℓ=1 (Case B): P₁(x+u), P₁(y+u) via compose_polynomial_on_affine
      - ℓ>1 (Case C): K_{ℓ-1}(x+u;R,θ), K_{ℓ-1}(y+u;R,θ) via case_c_taylor_coeffs

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        ell1: First piece index (1, 2, or 3)
        ell2: Second piece index (1, 2, or 3)
        n: Number of quadrature points
        polynomials: Dict with 'P1', 'P2', 'P3', 'Q' keys
        verbose: Print diagnostic output

    Returns:
        PostIdentityI1Result with I1 value and metadata
    """
    from src.quadrature import gauss_legendre_01

    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    Q_poly = polynomials['Q']
    P_polys = {
        1: polynomials['P1'],
        2: polynomials['P2'],
        3: polynomials['P3'],
    }

    # Omega values for Case B/C selection
    omega1 = ell1 - 1  # ℓ=1 → ω=0 (Case B), ℓ=2 → ω=1, ℓ=3 → ω=2
    omega2 = ell2 - 1

    var_names = ("x", "y")

    # Sign factor: (-1)^{ℓ₁+ℓ₂}
    sign_factor = (-1) ** (ell1 + ell2)

    # (1-u) power for I₁
    # NOTE: The (1,1) case is SPECIAL in the DSL - it uses power=2 explicitly,
    # while other pairs use max(0, (ℓ₁-1) + (ℓ₂-1)).
    # This matches the DSL's make_I1_11_v2 vs _make_I1_generic_v2 distinction.
    if ell1 == 1 and ell2 == 1:
        one_minus_u_power = 2
    else:
        one_minus_u_power = max(0, (ell1 - 1) + (ell2 - 1))

    I1_total = 0.0

    for i_u, (u, w_u) in enumerate(zip(u_nodes, u_weights)):
        for i_t, (t, w_t) in enumerate(zip(t_nodes, t_weights)):
            # Scalar prefactor: (1-u)^power × sign
            scalar_prefactor = (1 - u) ** one_minus_u_power * sign_factor

            # Step 1: Build Q(A_α)Q(A_β) × exp(R*(Arg_α+Arg_β)) core series
            core_series = apply_QQexp_post_identity_composition(
                Q_poly, t, theta, R, var_names
            )

            # Step 2: Build profile series for ℓ₁ on x
            P1_series = _build_profile_series(
                P_polys[ell1], omega1, u, R, theta, "x", var_names
            )

            # Step 3: Build profile series for ℓ₂ on y
            P2_series = _build_profile_series(
                P_polys[ell2], omega2, u, R, theta, "y", var_names
            )

            # Step 4: Build algebraic prefactor series (1/θ + x + y)
            alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
            alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

            # Step 5: Multiply all series
            integrand = core_series * P1_series * P2_series * alg_prefactor

            # Step 6: Extract ("x", "y") coefficient
            xy_coeff = integrand.extract(("x", "y"))

            # Step 7: Multiply by scalar prefactor and integrate
            contribution = xy_coeff * scalar_prefactor * w_u * w_t
            I1_total += contribution

            if verbose and i_u == 0 and i_t < 3:
                print(f"  (u={u:.3f}, t={t:.3f}): xy_coeff={xy_coeff:.6f}, "
                      f"scalar_pref={scalar_prefactor:.6f}")

    if verbose:
        print(f"I1_operator_post_identity_{ell1}{ell2} = {I1_total:.8f}")

    return PostIdentityI1Result(
        I1_value=I1_total,
        R=R,
        theta=theta,
        n_quad=n,
        details={
            "method": "operator_post_identity",
            "pair": (ell1, ell2),
            "omega1": omega1,
            "omega2": omega2,
            "one_minus_u_power": one_minus_u_power,
            "sign_factor": sign_factor,
            "var_names": var_names,
        }
    )
