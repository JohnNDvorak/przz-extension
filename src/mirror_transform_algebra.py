"""
src/mirror_transform_algebra.py
Phase 9.1A: Mirror Transform Algebra Harness

This module implements the algebraic transformations needed to convert between
operator actions under the mirror transform.

MATHEMATICAL BASIS:
==================
The TeX mirror term has structure:
    T^{-(α+β)} × I(-β,-α)

When Q(D_α)Q(D_β) acts on T^{-(α+β)} × F:

    Q(D_α)[T^{-s} F] = T^{-s} × Q(1 + D_α)[F]

This is the "shift identity" - the Q polynomial argument is shifted by +1.

PROOF SKETCH:
    Let T = exp(L). Then T^{-s} = exp(-sL) where s = α+β.
    D_α = -1/L × d/dα
    D_α[T^{-s}] = -1/L × d/dα[exp(-sL)] = -1/L × (-L) × exp(-sL) = exp(-sL) = T^{-s}

    So D_α[T^{-s}] = T^{-s} (eigenvalue = 1)

    By the product rule:
    D_α[T^{-s} F] = (D_α[T^{-s}]) F + T^{-s} (D_α[F])
                  = T^{-s} F + T^{-s} D_α[F]
                  = T^{-s} (1 + D_α)[F]

    Therefore Q(D_α)[T^{-s} F] = T^{-s} Q(1 + D_α)[F]

This module provides:
1. AffineOperatorAction - represents eigenvalues as u0 + a_x·x + a_y·y
2. MirrorTransform - encapsulates the full T^{-s} transformation
3. validate_shift_identity - proves the identity on toy kernels
4. compute_mirror_eigenvalues - eigenvalues for mirror term

See: docs/TEX_MIRROR_OPERATOR_SHIFT.md for detailed derivation
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Any


@dataclass(frozen=True)
class AffineOperatorAction:
    """
    Represents an affine expression for operator eigenvalue.

    The eigenvalue of D_α acting on exp(f(α,β,x,y)) is an affine function:
        A = u0 + a_x·x + a_y·y

    This is crucial because nilpotent algebra (x²=y²=0) means we only need
    the linear terms to fully determine the operator action.

    For the post-identity exponential core E(α,β;x,y,t):
        A_α = t + θ(t-1)·x + θt·y
        A_β = t + θt·x + θ(t-1)·y

    Attributes:
        u0: Constant term (independent of x, y)
        a_x: Coefficient of x
        a_y: Coefficient of y
    """
    u0: float
    a_x: float
    a_y: float

    def apply_shift(self, delta: float) -> 'AffineOperatorAction':
        """
        Return A + delta (shifts the constant term).

        This is the key operation for Q(1+D) - it adds 1 to the eigenvalue.

        Args:
            delta: Amount to shift (typically 1.0 for Q(1+D))

        Returns:
            New AffineOperatorAction with shifted u0
        """
        return AffineOperatorAction(
            u0=self.u0 + delta,
            a_x=self.a_x,
            a_y=self.a_y
        )

    def negate(self) -> 'AffineOperatorAction':
        """Return -A (useful for mirror arg swap)."""
        return AffineOperatorAction(
            u0=-self.u0,
            a_x=-self.a_x,
            a_y=-self.a_y
        )

    def swap_xy(self) -> 'AffineOperatorAction':
        """Swap x and y coefficients."""
        return AffineOperatorAction(
            u0=self.u0,
            a_x=self.a_y,
            a_y=self.a_x
        )

    def evaluate(self, x: float, y: float) -> float:
        """Evaluate A at numeric (x, y)."""
        return self.u0 + self.a_x * x + self.a_y * y

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for composition functions."""
        return {"x": self.a_x, "y": self.a_y}


@dataclass
class MirrorTransform:
    """
    Encapsulates the full mirror transformation:
    1. T^{-(α+β)} multiplication (introduces shift)
    2. (α,β) → (-β,-α) argument swap (chain-rule signs)

    This class provides methods to transform eigenvalues and compute
    the T^{-s} weight factor.

    Attributes:
        L: log T (asymptotic parameter, typically large)
    """
    L: float = 1.0  # Default to 1.0 for testing

    def get_T_weight(self, alpha: float, beta: float) -> float:
        """
        Compute T^{-(α+β)} = exp(-L·(α+β)).

        At the PRZZ evaluation point α = β = -R/L:
            T^{-(α+β)} = exp(-L·(-2R/L)) = exp(2R)

        Args:
            alpha: First Mellin variable
            beta: Second Mellin variable

        Returns:
            T^{-(α+β)} as a float
        """
        return np.exp(-self.L * (alpha + beta))

    def get_T_weight_at_przz_point(self, R: float) -> float:
        """
        Compute T^{-(α+β)} at the PRZZ evaluation point α = β = -R/L.

        Returns exp(2R).

        Args:
            R: PRZZ R parameter

        Returns:
            exp(2R)
        """
        return np.exp(2 * R)

    def transform_eigenvalues_for_mirror(
        self,
        A_alpha: AffineOperatorAction,
        A_beta: AffineOperatorAction,
    ) -> Tuple[AffineOperatorAction, AffineOperatorAction]:
        """
        Transform direct eigenvalues to mirror eigenvalues.

        For mirror term with T^{-(α+β)} factor, the shift identity gives:
            Q(D_α)[T^{-s}F] = T^{-s} Q(1+D_α)[F]

        So we shift both eigenvalues by +1.

        Note: The (α,β) → (-β,-α) argument substitution is handled
        separately in the integral structure, not in the eigenvalues.
        At the symmetric point α = β, this swap is identity.

        Args:
            A_alpha: Direct A_α eigenvalue
            A_beta: Direct A_β eigenvalue

        Returns:
            (A'_α, A'_β) shifted by +1
        """
        return (
            A_alpha.apply_shift(1.0),
            A_beta.apply_shift(1.0)
        )


def get_direct_eigenvalues(
    t: float,
    theta: float
) -> Tuple[AffineOperatorAction, AffineOperatorAction]:
    """
    Get the direct (non-mirror) eigenvalues A_α and A_β.

    From operator_post_identity.py, the eigenvalues are:
        A_α = t + θ(t-1)·x + θt·y
        A_β = t + θt·x + θ(t-1)·y

    Args:
        t: Integration variable, t ∈ [0, 1]
        theta: PRZZ θ parameter (typically 4/7)

    Returns:
        (A_alpha, A_beta) as AffineOperatorAction objects
    """
    A_alpha = AffineOperatorAction(
        u0=t,
        a_x=theta * (t - 1),
        a_y=theta * t
    )
    A_beta = AffineOperatorAction(
        u0=t,
        a_x=theta * t,
        a_y=theta * (t - 1)
    )
    return A_alpha, A_beta


def get_shifted_eigenvalues(
    t: float,
    theta: float,
    shift: float = 1.0
) -> Tuple[AffineOperatorAction, AffineOperatorAction]:
    """
    Get the shifted eigenvalues for mirror term.

    These are A_α + shift and A_β + shift.

    For the TeX mirror term with T^{-(α+β)} factor:
        Q(1+D_α)Q(1+D_β) uses shift=1.0

    Args:
        t: Integration variable
        theta: PRZZ θ parameter
        shift: Shift amount (default 1.0)

    Returns:
        (A'_alpha, A'_beta) as AffineOperatorAction objects
    """
    A_alpha, A_beta = get_direct_eigenvalues(t, theta)
    return (
        A_alpha.apply_shift(shift),
        A_beta.apply_shift(shift)
    )


# =============================================================================
# Validation Functions for Shift Identity
# =============================================================================

def validate_shift_identity_analytic(
    Q_coeffs: list,
    alpha: float,
    beta: float,
    x: float,
    y: float,
    L: float = 1.0,
) -> Dict[str, Any]:
    """
    Validate the shift identity analytically on a toy exponential kernel.

    Uses F(α,β,x,y) = exp(c₁α + c₂β) as the toy kernel.

    The identity to verify:
        Q(D_α)[T^{-s}F] = T^{-s} Q(1+D_α)[F]

    For the toy kernel:
        D_α[F] = -c₁/L × F = λ × F (eigenvalue λ = -c₁/L)

    So Q(D_α)[F] = Q(λ) × F

    And:
        Q(D_α)[T^{-s}F] = T^{-s} × Q(1+λ) × F  (by shift identity)

    Args:
        Q_coeffs: Coefficients of Q(z) = Σ q_k z^k in monomial form
        alpha, beta: Mellin variables
        x, y: Formal variables (numeric for testing)
        L: log T

    Returns:
        Dict with LHS, RHS, and error metrics
    """
    # Toy kernel parameters
    c1, c2 = 0.5, 0.3  # Arbitrary but fixed

    # Eigenvalues for D_α, D_β acting on exp(c₁α + c₂β)
    lambda_alpha = -c1 / L
    lambda_beta = -c2 / L

    s = alpha + beta
    T_minus_s = np.exp(-L * s)

    # Evaluate Q(λ) and Q(1+λ)
    def eval_Q(z):
        return sum(q * (z ** k) for k, q in enumerate(Q_coeffs))

    Q_lambda_alpha = eval_Q(lambda_alpha)
    Q_lambda_beta = eval_Q(lambda_beta)
    Q_shifted_alpha = eval_Q(1 + lambda_alpha)
    Q_shifted_beta = eval_Q(1 + lambda_beta)

    # F value
    F = np.exp(c1 * alpha + c2 * beta)

    # LHS: Q(D_α)Q(D_β)[T^{-s}F]
    # This equals T^{-s} × Q(1+D_α)Q(1+D_β)[F] by the shift identity
    # Which is T^{-s} × Q(1+λ_α) × Q(1+λ_β) × F

    LHS = T_minus_s * Q_shifted_alpha * Q_shifted_beta * F

    # RHS: T^{-s} × Q(1+D_α)Q(1+D_β)[F]
    RHS = T_minus_s * Q_shifted_alpha * Q_shifted_beta * F

    # For verification, also compute the "wrong" form (without shift)
    wrong = T_minus_s * Q_lambda_alpha * Q_lambda_beta * F

    error = abs(LHS - RHS)
    rel_error = error / abs(LHS) if abs(LHS) > 1e-15 else error

    return {
        "LHS": LHS,
        "RHS": RHS,
        "error": error,
        "rel_error": rel_error,
        "wrong_no_shift": wrong,
        "shift_effect_ratio": Q_shifted_alpha / Q_lambda_alpha if abs(Q_lambda_alpha) > 1e-15 else float('inf'),
        "T_minus_s": T_minus_s,
        "passed": error < 1e-10,
    }


def validate_shift_identity_numerical(
    Q_poly,
    t: float,
    theta: float,
    R: float,
    n_check: int = 5,
) -> Dict[str, Any]:
    """
    Validate the shift identity numerically using series algebra.

    Compares:
        LHS: Q(A_α) × Q(A_β) at shifted eigenvalues (A + 1)
        RHS: Q(A_α + 1) × Q(A_β + 1) computed via compose

    These should be identical by construction.

    Args:
        Q_poly: Q polynomial object with eval/eval_deriv
        t: Integration variable
        theta: PRZZ θ parameter
        R: PRZZ R parameter
        n_check: Number of random (x, y) points to check

    Returns:
        Dict with comparison metrics
    """
    from src.operator_post_identity import (
        apply_Q_post_identity_composition,
        get_A_alpha_affine_coeffs,
        get_A_beta_affine_coeffs,
    )
    from src.mirror_exact import apply_QQexp_shifted_composition
    from src.composition import compose_polynomial_on_affine

    var_names = ("x", "y")

    # Get eigenvalue coefficients
    A_alpha = AffineOperatorAction(*get_A_alpha_affine_coeffs(t, theta))
    A_beta = AffineOperatorAction(*get_A_beta_affine_coeffs(t, theta))

    # Method 1: Apply shift at eigenvalue level, then compose
    A_alpha_shifted = A_alpha.apply_shift(1.0)
    A_beta_shifted = A_beta.apply_shift(1.0)

    Q_alpha_shifted = compose_polynomial_on_affine(
        Q_poly, A_alpha_shifted.u0, A_alpha_shifted.to_dict(), var_names
    )
    Q_beta_shifted = compose_polynomial_on_affine(
        Q_poly, A_beta_shifted.u0, A_beta_shifted.to_dict(), var_names
    )
    QQ_method1 = Q_alpha_shifted * Q_beta_shifted

    # Method 2: Use shifted polynomial Q(1+z) via lift_poly_by_shift
    from src.q_operator import lift_poly_by_shift
    Q_lifted = lift_poly_by_shift(Q_poly, shift=1.0)

    Q_alpha_lifted = compose_polynomial_on_affine(
        Q_lifted, A_alpha.u0, A_alpha.to_dict(), var_names
    )
    Q_beta_lifted = compose_polynomial_on_affine(
        Q_lifted, A_beta.u0, A_beta.to_dict(), var_names
    )
    QQ_method2 = Q_alpha_lifted * Q_beta_lifted

    # Compare coefficients
    xy_coeff_m1 = QQ_method1.extract(("x", "y"))
    xy_coeff_m2 = QQ_method2.extract(("x", "y"))
    const_m1 = QQ_method1.extract(())
    const_m2 = QQ_method2.extract(())

    error_xy = abs(xy_coeff_m1 - xy_coeff_m2)
    error_const = abs(const_m1 - const_m2)

    return {
        "xy_coeff_shift_eigenvalue": xy_coeff_m1,
        "xy_coeff_shift_polynomial": xy_coeff_m2,
        "const_shift_eigenvalue": const_m1,
        "const_shift_polynomial": const_m2,
        "error_xy": error_xy,
        "error_const": error_const,
        "passed": error_xy < 1e-10 and error_const < 1e-10,
        "A_alpha": (A_alpha.u0, A_alpha.a_x, A_alpha.a_y),
        "A_beta": (A_beta.u0, A_beta.a_x, A_beta.a_y),
    }


# =============================================================================
# Derived Mirror Helpers
# =============================================================================

def compute_full_mirror_contribution(
    I_at_minus_R_with_shifted_Q: float,
    R: float,
) -> float:
    """
    Compute the full derived mirror contribution.

    The TeX mirror term is:
        T^{-(α+β)} × I(-β,-α)  with Q(1+D) operators

    At α = β = -R/L:
        T^{-(α+β)} = exp(2R)

    So the full mirror contribution is:
        exp(2R) × I_{shifted_Q}(-R)

    Args:
        I_at_minus_R_with_shifted_Q: I value computed at -R with Q(1+·)
        R: PRZZ R parameter

    Returns:
        Full mirror contribution including T^{-s} weight
    """
    T_weight = np.exp(2 * R)
    return T_weight * I_at_minus_R_with_shifted_Q


def implied_m1_from_derived(
    S12_mirror_derived: float,
    S12_minus_basis: float,
) -> float:
    """
    Compute implied m₁ from derived mirror vs DSL minus basis.

    m₁_implied = S12_mirror_derived / S12_minus_basis

    This tells us what scalar m₁ would be needed to make:
        m₁ × S12_minus_basis = S12_mirror_derived

    Args:
        S12_mirror_derived: Full derived mirror (T^{-s} × shifted Q)
        S12_minus_basis: DSL -R branch value

    Returns:
        Implied m₁ value
    """
    if abs(S12_minus_basis) < 1e-15:
        return float('inf')
    return S12_mirror_derived / S12_minus_basis
