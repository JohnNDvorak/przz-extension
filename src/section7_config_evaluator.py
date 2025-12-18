"""
src/section7_config_evaluator.py
Section 7 Configuration Evaluator

This module connects the p-config enumeration (psi_block_configs.py) to the
Section 7 primitives (przz_section7_oracle.py) to compute pair contributions
WITHOUT using the I₁-I₄ template.

The key insight from GPT's guidance:
- The I₁-I₄ template is ONLY valid for (1,1)
- Higher pairs need the full Ψ combinatorial expansion
- But instead of expanding to 78 monomials, we use 16 p-configs

Block interpretation:
- X = (A - C) = singleton z-derivative contribution
- Y = (B - C) = singleton w-derivative contribution
- Z = (D - C²) = paired zw-structure contribution

For each BlockConfig, we evaluate:
    coeff × X^{ℓ-p} × Y^{ℓ̄-p} × Z^p

Each block evaluation uses the Section 7 F_d primitives + Euler-Maclaurin.
"""

from __future__ import annotations
import numpy as np
from math import factorial, exp, log
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from src.psi_block_configs import BlockConfig, psi_p_configs
from src.przz_section7_oracle import (
    gauss_legendre_01,
    compute_Fd_case_B,
    euler_maclaurin_exponent,
    euler_maclaurin_coefficient,
    PolynomialEvaluator
)


@dataclass
class BlockEvalResult:
    """Result of evaluating a single p-config."""
    config: BlockConfig
    value: float
    debug: Optional[Dict] = None


def eval_block_X(
    P: PolynomialEvaluator,
    Q: PolynomialEvaluator,
    u: float, t: float,
    theta: float, R: float
) -> float:
    """
    Evaluate the X = (A - C) block at a quadrature point.

    X represents the "connected singleton z-derivative" contribution.

    In PRZZ's framework for ℓ=1 (Case B):
    - A is related to d/dz[integrand]|_{z=0}
    - C is the base log-integrand

    For the I₁-I₄ mapping:
    - (A - C) corresponds to the I₃-type contribution (z-derivative only)

    Using the fused t-integral form, at a point (u,t):
    X ≈ d/dx[P(x+u) × Q(α) × Q(β) × exp(R(α+β))]|_{x=0} / [P(u) × Q(t)² × exp(2Rt)]

    Since α = t + θt×x + θ(t-1)×y evaluated at x=y=0 gives α = t,
    the derivative structure at x=0 is:

    X_contrib = (P'/P + θt×Q'/Q + θt×Q'/Q + θt×R + θt×R) at (u,t)

    But more simply, for the (1,1) case in the fused form:
    X = -1/θ × [algebraic structure of I₃]
    """
    u_arr = np.array([u])
    t_arr = np.array([t])

    P_u = float(P(u_arr)[0])
    Pp_u = float(P.deriv(u_arr)[0])
    Q_t = float(Q(t_arr)[0])
    Qp_t = float(Q.deriv(t_arr)[0])

    # For (1,1), the X block contribution at (u,t) involves:
    # d/dx[P(x+u)]|_{x=0} = P'(u)
    # and the Q/exp derivatives w.r.t. x

    # The full derivative structure:
    # d/dx[P(x+u) × Q(α) × Q(β) × exp(R(α+β))]|_{x=0}
    # = P'(u) × Q(t)² × exp(2Rt)
    #   + P(u) × Q'(t) × θt × Q(t) × exp(2Rt)
    #   + P(u) × Q(t) × Q'(t) × θ(t-1) × exp(2Rt)
    #   + P(u) × Q(t)² × exp(2Rt) × R × (θt + θ(t-1))

    # Normalized by the base integrand P(u)Q(t)²exp(2Rt):
    E = exp(R * t)
    base = P_u * Q_t**2 * E**2

    if abs(base) < 1e-15:
        return 0.0

    # x-derivative terms:
    # 1. From P(x+u): P'(u)/P(u)
    # 2. From Q(α): Q'(t) × θt / Q(t)  (since dα/dx = θt at x=y=0)
    # 3. From Q(β): Q'(t) × θ(t-1) / Q(t)  (since dβ/dx = θ(t-1) at x=y=0)
    # 4. From exp(R×α): R × θt
    # 5. From exp(R×β): R × θ(t-1)

    term1 = Pp_u / P_u if abs(P_u) > 1e-15 else 0.0
    term2 = Qp_t * theta * t / Q_t if abs(Q_t) > 1e-15 else 0.0
    term3 = Qp_t * theta * (t - 1) / Q_t if abs(Q_t) > 1e-15 else 0.0
    term4 = R * theta * t
    term5 = R * theta * (t - 1)

    X_val = term1 + term2 + term3 + term4 + term5

    return X_val


def eval_block_Y(
    P: PolynomialEvaluator,
    Q: PolynomialEvaluator,
    u: float, t: float,
    theta: float, R: float
) -> float:
    """
    Evaluate the Y = (B - C) block at a quadrature point.

    Y represents the "connected singleton w-derivative" contribution.
    By symmetry with X, but using y-derivatives instead of x-derivatives.

    The y-derivative structure:
    dα/dy = θ(t-1) at x=y=0
    dβ/dy = θt at x=y=0
    """
    u_arr = np.array([u])
    t_arr = np.array([t])

    P_u = float(P(u_arr)[0])
    Pp_u = float(P.deriv(u_arr)[0])
    Q_t = float(Q(t_arr)[0])
    Qp_t = float(Q.deriv(t_arr)[0])

    if abs(P_u) < 1e-15 or abs(Q_t) < 1e-15:
        return 0.0

    # y-derivative terms (swapped coefficients compared to X):
    # 1. From P(y+u): P'(u)/P(u)
    # 2. From Q(α): Q'(t) × θ(t-1) / Q(t)
    # 3. From Q(β): Q'(t) × θt / Q(t)
    # 4. From exp(R×α): R × θ(t-1)
    # 5. From exp(R×β): R × θt

    term1 = Pp_u / P_u
    term2 = Qp_t * theta * (t - 1) / Q_t
    term3 = Qp_t * theta * t / Q_t
    term4 = R * theta * (t - 1)
    term5 = R * theta * t

    Y_val = term1 + term2 + term3 + term4 + term5

    return Y_val


def eval_block_Z(
    P: PolynomialEvaluator,
    Q: PolynomialEvaluator,
    u: float, t: float,
    theta: float, R: float
) -> float:
    """
    Evaluate the Z = (D - C²) block at a quadrature point.

    Z represents the "connected paired zw-structure" contribution.
    This is the mixed ∂²/∂z∂w derivative, minus the disconnected C² term.

    For (1,1), this corresponds to the I₂-type contribution.
    """
    u_arr = np.array([u])
    t_arr = np.array([t])

    P_u = float(P(u_arr)[0])
    Pp_u = float(P.deriv(u_arr)[0])
    Ppp_u = float(P.deriv(u_arr, 2)[0])
    Q_t = float(Q(t_arr)[0])
    Qp_t = float(Q.deriv(t_arr)[0])
    Qpp_t = float(Q.deriv(t_arr, 2)[0])

    if abs(P_u) < 1e-15 or abs(Q_t) < 1e-15:
        return 0.0

    # The D block is d²/dxdy[integrand]|_{x=y=0}
    # This includes cross-derivative terms that don't factorize as X×Y

    # For the polynomial factor P(x+u)P(y+u):
    # d²/dxdy[P(x+u)P(y+u)] = P'(u)² (at x=y=0)
    # This is the "connected" part - derivatives on different P factors

    # The Z = D - C² removes the "disconnected" part
    # For (1,1), Z corresponds to the base integral structure

    # Simple approximation: Z ≈ 1/θ (the I₂ coefficient structure)
    # More precisely, Z captures the "paired" part after subtracting X×Y contribution

    # From the (1,1) validated formula:
    # D = I₂ contribution ≈ (1/θ) × P²Q²e^{2Rt}
    # C = base log ≈ log(PQ²e^{Rt}) structure

    # For numerical stability, we use the known (1,1) relation:
    # Z contributes the "base" value without derivatives
    # Z_val ≈ 1/θ at the normalized level

    return 1.0 / theta


def eval_config_11(
    config: BlockConfig,
    P: PolynomialEvaluator,
    Q: PolynomialEvaluator,
    theta: float,
    R: float,
    n_quad: int = 60
) -> BlockEvalResult:
    """
    Evaluate a BlockConfig for the (1,1) pair.

    For (1,1), we have 2 p-configs:
    - p=0: coeff=1, X¹Y¹ (mixed derivative contribution)
    - p=1: coeff=1, Z¹ (base contribution)

    This should match the sum I₁ + I₂ - I₃ - I₄ from the DSL.
    """
    assert config.ell == 1 and config.ellbar == 1, "This function is for (1,1) only"

    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # Euler-Maclaurin weight for (1,1): L₁ = 0, L₂ = 0
    # Combined exponent = 0, so (1-u)^0 = 1
    one_minus_u_exp = euler_maclaurin_exponent(0) + euler_maclaurin_exponent(0)
    em_coeff = euler_maclaurin_coefficient(0) * euler_maclaurin_coefficient(0)

    total = 0.0

    for iu, u in enumerate(u_nodes):
        for it, t in enumerate(t_nodes):
            wu = u_weights[iu]
            wt = t_weights[it]

            # Base integrand: P(u)² × Q(t)² × exp(2Rt) × (1/θ)
            u_arr = np.array([u])
            t_arr = np.array([t])

            P_u = float(P(u_arr)[0])
            Q_t = float(Q(t_arr)[0])
            E = exp(R * t)

            base = (1.0 / theta) * P_u**2 * Q_t**2 * E**2

            # Weight factor
            weight = (1.0 - u) ** one_minus_u_exp

            # Evaluate blocks based on config
            if config.p == 0:
                # X¹Y¹ term
                X_val = eval_block_X(P, Q, u, t, theta, R)
                Y_val = eval_block_Y(P, Q, u, t, theta, R)
                block_val = X_val * Y_val
            else:  # p == 1
                # Z¹ term
                block_val = eval_block_Z(P, Q, u, t, theta, R)

            # Contribution to integral
            contrib = base * weight * block_val * em_coeff
            total += wu * wt * contrib

    return BlockEvalResult(
        config=config,
        value=config.coeff * total,
        debug={"em_exp": one_minus_u_exp, "em_coeff": em_coeff}
    )


def eval_pair_via_configs(
    ell: int, ellbar: int,
    polynomials: Dict,
    theta: float, R: float,
    n_quad: int = 60
) -> Tuple[float, List[BlockEvalResult]]:
    """
    Evaluate a pair (ℓ, ℓ̄) contribution using p-config summation.

    This is the main entry point for the Section 7 config evaluator.

    Args:
        ell, ellbar: Pair indices
        polynomials: Dict with "P1", "P2", "P3", "Q"
        theta, R: Parameters
        n_quad: Quadrature points

    Returns:
        (total_value, list_of_config_results)
    """
    # Get the polynomial for this piece
    P_poly = polynomials[f"P{ell}"]
    Q_poly = polynomials["Q"]

    P = PolynomialEvaluator(P_poly)
    Q = PolynomialEvaluator(Q_poly)

    # Generate p-configs for this pair
    configs = psi_p_configs(ell, ellbar)

    results = []
    total = 0.0

    for config in configs:
        if ell == 1 and ellbar == 1:
            result = eval_config_11(config, P, Q, theta, R, n_quad)
        else:
            # For now, stub for other pairs
            result = BlockEvalResult(config=config, value=0.0, debug={"note": "not implemented"})

        results.append(result)
        total += result.value

    return total, results


def test_11_validation():
    """
    Validate (1,1) config evaluation against existing DSL.

    The p-config approach should reproduce the I₁+I₂-I₃-I₄ sum.
    """
    from src.polynomials import load_przz_polynomials
    from src.evaluate import evaluate_c11

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("=" * 70)
    print("(1,1) CONFIG EVALUATOR VALIDATION")
    print("=" * 70)

    # DSL result
    dsl_result = evaluate_c11(theta, R, n_quad, polynomials)
    dsl_c11 = dsl_result.total

    # Config result
    config_c11, config_results = eval_pair_via_configs(1, 1, polynomials, theta, R, n_quad)

    print(f"\nDSL I₁+I₂+I₃+I₄ sum: {dsl_c11:.6f}")
    print(f"\nConfig evaluation:")
    for result in config_results:
        print(f"  {result.config}: {result.value:.6f}")
    print(f"\nConfig total: {config_c11:.6f}")

    print(f"\nRatio (Config/DSL): {config_c11 / dsl_c11:.4f}")

    if abs(config_c11 / dsl_c11 - 1.0) < 0.1:
        print("✓ Within 10% - good starting point")
    else:
        print("✗ Large discrepancy - need to debug block evaluations")


if __name__ == "__main__":
    test_11_validation()
