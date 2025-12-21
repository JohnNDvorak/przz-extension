#!/usr/bin/env python3
"""
GPT Run 8B1: Direct I1(1,1) Validation (Case B Only)

This script validates I1 for pair (1,1) only - Case B (no kernel derivatives).
The goal is to verify the DSL derivative extraction matches a direct computation.

For pair (1,1), P₁ is Case B (ω=0), so no Case C kernel is needed.
This is the simplest possible I1 validation.

MATHEMATICAL STRUCTURE of I1(1,1):

I₁ = d²/dxdy |_{x=y=0} [∫∫ F(x, y, u, t) du dt]

where the integrand F has structure:
    F = (1/θ + x + y)(1-u)² × P₁(x+u)P₁(y+u)
        × Q(Arg_α)Q(Arg_β) × exp(R×Arg_α + R×Arg_β)

with:
    Arg_α = t + θt·x + θ(t-1)·y
    Arg_β = t + θ(t-1)·x + θt·y

Since we need d²F/dxdy |_{x=y=0}, we extract the coefficient of x¹y¹.

For Case B (P1), the polynomials are not kernel-transformed:
    P₁(x+u) = P₁(u) + P₁'(u)·x + O(x²)

Usage:
    python run_gpt_run8b1_direct_i1_11.py
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.quadrature import gauss_legendre_01
from src.evaluate import compute_operator_implied_weights
from src.series import TruncatedSeries


THETA = 4.0 / 7.0


@dataclass
class I1DirectResult:
    """Result of direct I1(1,1) evaluation."""
    i1_value: float
    u_integral: float
    t_integral: float
    R: float


def compute_i1_11_direct_numerical(
    theta: float,
    R: float,
    polynomials: Dict,
    n: int = 100
) -> I1DirectResult:
    """
    Compute I1 for pair (1,1) using numerical derivative approximation.

    This is a validation approach - we use small epsilon to approximate
    the mixed derivative d²F/dxdy.

    For pair (1,1), this should match the DSL evaluation.
    """
    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    eps = 1e-6  # Small epsilon for numerical derivative

    def F(x: float, y: float) -> float:
        """Compute the full integrand at fixed (x, y)."""
        total = 0.0

        for i, u in enumerate(u_nodes):
            for j, t in enumerate(t_nodes):
                weight = u_weights[i] * t_weights[j]

                # Algebraic prefactor: (1/θ + x + y)
                alg_pf = 1.0 / theta + x + y

                # Poly prefactor: (1-u)²
                poly_pf = (1 - u) ** 2

                # P₁(x+u) and P₁(y+u)
                P1_x = P1.eval(np.array([x + u]))[0]
                P1_y = P1.eval(np.array([y + u]))[0]

                # Arguments
                Arg_alpha = t + theta * t * x + theta * (t - 1) * y
                Arg_beta = t + theta * (t - 1) * x + theta * t * y

                # Q factors
                Q_alpha = Q.eval(np.array([Arg_alpha]))[0]
                Q_beta = Q.eval(np.array([Arg_beta]))[0]

                # Exp factors
                exp_factor = np.exp(R * (Arg_alpha + Arg_beta))

                # Full integrand
                integrand = alg_pf * poly_pf * P1_x * P1_y * Q_alpha * Q_beta * exp_factor

                total += weight * integrand

        return total

    # Compute mixed derivative using finite differences
    # d²F/dxdy ≈ [F(ε,ε) - F(ε,-ε) - F(-ε,ε) + F(-ε,-ε)] / (4ε²)
    F_pp = F(eps, eps)
    F_pm = F(eps, -eps)
    F_mp = F(-eps, eps)
    F_mm = F(-eps, -eps)

    mixed_deriv = (F_pp - F_pm - F_mp + F_mm) / (4 * eps * eps)

    return I1DirectResult(
        i1_value=mixed_deriv,
        u_integral=0.0,  # Not separated in this approach
        t_integral=0.0,
        R=R,
    )


def compute_i1_11_direct_series(
    theta: float,
    R: float,
    polynomials: Dict,
    n: int = 60
) -> I1DirectResult:
    """
    Compute I1 for pair (1,1) using the series engine directly.

    This mirrors what the DSL does but constructs the series manually
    to validate the approach.
    """
    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    # For pair (1,1), we need d²/dxdy at x=y=0
    # The series has vars ("x", "y") with max orders (1, 1)
    # The coefficient of x^1 y^1 is what we need (bitset = 0b11 = 3)

    total = 0.0

    for i, u in enumerate(u_nodes):
        for j, t in enumerate(t_nodes):
            weight = u_weights[i] * t_weights[j]

            # Build the series at this (u, t) point
            # We track coefficients: 1, x, y, xy

            # Initialize coefficients dict
            # key = bitset (0=const, 1=x, 2=y, 3=xy)
            coeffs = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}

            # 1. Algebraic prefactor: (1/θ + x + y)
            # = 1/θ + 1·x + 1·y + 0·xy
            alg_const = 1.0 / theta
            alg_x = 1.0
            alg_y = 1.0

            # 2. Poly prefactor: (1-u)² - pure scalar
            poly_pf = (1 - u) ** 2

            # 3. P₁(x+u) = P₁(u) + P₁'(u)·x + O(x²)
            P1_u = P1.eval(np.array([u]))[0]
            P1_deriv_u = P1.eval_deriv(np.array([u]), 1)[0]

            # P₁(y+u) = P₁(u) + P₁'(u)·y + O(y²)
            # Same derivative since P1 is the same polynomial

            # P₁(x+u)·P₁(y+u) = [P₁(u) + P₁'(u)·x][P₁(u) + P₁'(u)·y]
            #                 = P₁(u)² + P₁(u)P₁'(u)·x + P₁(u)P₁'(u)·y + P₁'(u)²·xy
            P_const = P1_u * P1_u
            P_x = P1_u * P1_deriv_u
            P_y = P1_u * P1_deriv_u
            P_xy = P1_deriv_u * P1_deriv_u

            # 4. Q factors: Q(Arg_α)·Q(Arg_β)
            # Arg_α = t + θt·x + θ(t-1)·y
            # Arg_β = t + θ(t-1)·x + θt·y

            # At x=y=0: Arg_α = Arg_β = t
            Q_t = Q.eval(np.array([t]))[0]
            Q_deriv_t = Q.eval_deriv(np.array([t]), 1)[0]

            # Expand Q(Arg_α) around x=y=0:
            # Q(Arg_α) = Q(t) + Q'(t)·[θt·x + θ(t-1)·y] + O(x²,y²,xy)
            # The O(xy) term from first order is 0.
            # Second order terms: Q''(t)·[θt·x + θ(t-1)·y]²/2
            # The xy coefficient from this is: Q''(t)·θ²t(t-1)
            Q_deriv2_t = Q.eval_deriv(np.array([t]), 2)[0]

            # Coefficients for Q(Arg_α):
            # const: Q(t)
            # x: Q'(t)·θt
            # y: Q'(t)·θ(t-1)
            # xy: Q''(t)·θ²t(t-1)/2 (from (θt·x + θ(t-1)·y)² expansion)
            #     = Q''(t)·θ²·t·(t-1) · 2/2 = Q''(t)·θ²·t·(t-1) (cross term has factor 2)
            Q_alpha_const = Q_t
            Q_alpha_x = Q_deriv_t * theta * t
            Q_alpha_y = Q_deriv_t * theta * (t - 1)
            Q_alpha_xy = Q_deriv2_t * theta * theta * t * (t - 1)  # Cross term coefficient

            # Similarly for Q(Arg_β):
            Q_beta_const = Q_t
            Q_beta_x = Q_deriv_t * theta * (t - 1)
            Q_beta_y = Q_deriv_t * theta * t
            Q_beta_xy = Q_deriv2_t * theta * theta * (t - 1) * t

            # Q(Arg_α)·Q(Arg_β) product:
            # We need the xy coefficient
            # (a + bx + cy + dxy)(e + fx + gy + hxy)
            # xy coeff = ah + bg·0 + cf·0 + de + (cross of b,g and c,f)
            # Actually: xy coeff = a·h + e·d + b·g + c·f
            QQ_xy = (Q_alpha_const * Q_beta_xy +
                     Q_beta_const * Q_alpha_xy +
                     Q_alpha_x * Q_beta_y +
                     Q_alpha_y * Q_beta_x)
            QQ_const = Q_alpha_const * Q_beta_const
            QQ_x = Q_alpha_const * Q_beta_x + Q_alpha_x * Q_beta_const
            QQ_y = Q_alpha_const * Q_beta_y + Q_alpha_y * Q_beta_const

            # 5. Exp factor: exp(R·Arg_α + R·Arg_β) = exp(R·(Arg_α + Arg_β))
            # Arg_α + Arg_β = 2t + θt·x + θ(t-1)·y + θ(t-1)·x + θt·y
            #               = 2t + θ(2t-1)·x + θ(2t-1)·y
            # At x=y=0: exp(2Rt)
            # Expansion: exp(2Rt + R·θ(2t-1)·x + R·θ(2t-1)·y)
            #          = exp(2Rt) · exp(R·θ(2t-1)·(x+y))
            #          = exp(2Rt) · [1 + R·θ(2t-1)·(x+y) + [R·θ(2t-1)]²·(x+y)²/2 + ...]
            exp_2Rt = np.exp(2 * R * t)
            exp_coeff = R * theta * (2 * t - 1)

            # exp(2Rt) · [1 + c(x+y) + c²(x+y)²/2]
            # = exp(2Rt) · [1 + cx + cy + c²x²/2 + c²xy + c²y²/2]
            # xy coeff: exp(2Rt) · c²
            E_const = exp_2Rt
            E_x = exp_2Rt * exp_coeff
            E_y = exp_2Rt * exp_coeff
            E_xy = exp_2Rt * exp_coeff * exp_coeff

            # Now multiply everything together to get the xy coefficient
            # F = alg_pf · poly_pf · P_prod · Q_prod · E

            # First combine alg_pf and poly_pf (scalar times series)
            # alg: (1/θ + x + y) times poly_pf (scalar)
            # = poly_pf/θ + poly_pf·x + poly_pf·y + 0·xy
            AP_const = poly_pf * alg_const
            AP_x = poly_pf * alg_x
            AP_y = poly_pf * alg_y
            AP_xy = 0.0

            # Then combine with P product
            # (a + bx + cy + dxy)(e + fx + gy + hxy)
            # xy = ah + de + bg + cf
            APP_xy = (AP_const * P_xy + AP_xy * P_const +
                      AP_x * P_y + AP_y * P_x)
            APP_const = AP_const * P_const
            APP_x = AP_const * P_x + AP_x * P_const
            APP_y = AP_const * P_y + AP_y * P_const

            # Then combine with QQ
            APPQ_xy = (APP_const * QQ_xy + APP_xy * QQ_const +
                       APP_x * QQ_y + APP_y * QQ_x)
            APPQ_const = APP_const * QQ_const
            APPQ_x = APP_const * QQ_x + APP_x * QQ_const
            APPQ_y = APP_const * QQ_y + APP_y * QQ_const

            # Finally combine with E
            full_xy = (APPQ_const * E_xy + APPQ_xy * E_const +
                       APPQ_x * E_y + APPQ_y * E_x)

            total += weight * full_xy

    return I1DirectResult(
        i1_value=total,
        u_integral=0.0,
        t_integral=0.0,
        R=R,
    )


def main():
    print("=" * 80)
    print("GPT Run 8B1: Direct I1(1,1) Validation (Case B Only)")
    print("=" * 80)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    benchmarks = [
        ("κ", polys_kappa, 1.3036),
        ("κ*", polys_kappa_star, 1.1167),
    ]

    for bench_name, polys, R_val in benchmarks:
        print(f"\n{'='*70}")
        print(f"Benchmark: {bench_name} (R={R_val})")
        print(f"{'='*70}")

        # Get DSL result
        implied = compute_operator_implied_weights(
            theta=THETA,
            R=R_val,
            polynomials=polys,
            sigma=5/32,
            normalization="grid",
            lift_scope="i1_only",
            n=60,
            n_quad_a=40,
        )

        # Get I1(1,1) from pair breakdown (with factorial normalization)
        i1_11_dsl = implied.pair_breakdown["11"]["I1_plus"]
        # Raw value without normalization
        i1_11_dsl_raw = implied.pair_breakdown["11"]["I1_plus_raw"]

        # Compute direct using series approach
        direct_result = compute_i1_11_direct_series(THETA, R_val, polys, n=60)

        # The DSL applies factorial normalization 1/(1!×1!) = 1
        # So normalized and raw should be the same for (1,1)
        print(f"\n--- I1(1,1) Comparison ---")
        print(f"DSL I1(1,1) raw:       {i1_11_dsl_raw:+.8f}")
        print(f"DSL I1(1,1) normalized: {i1_11_dsl:+.8f}")
        print(f"Direct I1(1,1):        {direct_result.i1_value:+.8f}")

        if abs(i1_11_dsl_raw) > 1e-10:
            ratio = direct_result.i1_value / i1_11_dsl_raw
            print(f"Ratio (direct/DSL):    {ratio:.6f}")

            if abs(ratio - 1.0) < 0.02:
                print("STATUS: ALIGNED (within 2%)")
            else:
                print("STATUS: NOT ALIGNED")
        else:
            print("Ratio: N/A (DSL ≈ 0)")

        # Also try numerical derivative for comparison
        print(f"\n--- Numerical Derivative Check ---")
        numerical_result = compute_i1_11_direct_numerical(THETA, R_val, polys, n=40)
        print(f"Numerical I1(1,1):     {numerical_result.i1_value:+.8f}")
        if abs(i1_11_dsl_raw) > 1e-10:
            ratio_num = numerical_result.i1_value / i1_11_dsl_raw
            print(f"Ratio (numerical/DSL): {ratio_num:.6f}")

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Run 8B1 validates I1(1,1) for Case B (P₁ with ω=0).

The direct series expansion computes d²F/dxdy |_{x=y=0} by:
1. Expanding each factor to first order in x and y
2. Computing the product's xy coefficient
3. Integrating over (u, t)

If direct matches DSL, the derivative extraction machinery is validated
for the simplest case (no Case C kernel derivatives).

NEXT STEPS:
1. If aligned: Create pytest gate test for I1(1,1)
2. Extend to Case C pairs (requires kernel derivatives)
""")


if __name__ == "__main__":
    main()
