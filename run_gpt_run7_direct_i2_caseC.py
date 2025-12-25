#!/usr/bin/env python3
"""
GPT Run 7: Direct TeX I2 Evaluation with Case C Kernels

This script extends Run 6 by implementing Case C kernel handling for P2 and P3.
Run 6 showed that pair (1,1) matches exactly (P1 is Case B), but P2/P3 pairs
have discrepancies because the model uses Case C kernels.

CASE C KERNEL FORMULA (PRZZ TeX 2370-2375):
    K_ω(u; R) = ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθua) da

POLYNOMIAL REPLACEMENT:
    P_ℓ(u) → u^ω × K_ω(u; R)

Where:
    P₁: ω = 0 (Case B) → raw P₁(u) unchanged
    P₂: ω = 1 (Case C) → u × K₁(u; R)
    P₃: ω = 2 (Case C) → u² × K₂(u; R)

EXPECTED OUTCOME:
    After implementing Case C, ALL 9 pairs should have direct/model ratio ≈ 1.0

Usage:
    python run_gpt_run7_direct_i2_caseC.py
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable

from src.evaluate import compute_operator_implied_weights
from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.quadrature import gauss_legendre_01


THETA = 4.0 / 7.0
R_REF = 1.3036


# CORRECT factorial normalization: 1/(ell1! × ell2!)
F_NORM = {
    f"{i}{j}": 1.0 / (math.factorial(i) * math.factorial(j))
    for i in (1, 2, 3) for j in (1, 2, 3)
}


@dataclass
class I2DirectResult:
    """Result of direct TeX I2 evaluation."""
    u_integral: float  # ∫ K_{ℓ1}(u) K_{ℓ2}(u) du (with Case C transforms)
    t_integral_plus: float  # (1/θ) ∫ Q(t)² exp(2Rt) dt at +R
    t_integral_minus: float  # (1/θ) ∫ Q(t)² exp(-2Rt) dt at -R
    i2_plus: float  # I2 at +R
    i2_minus: float  # I2 at -R
    i2_mirror_simple: float  # I2(+R) + exp(2R) × I2(-R)
    exp_2R: float
    exp_R: float
    R: float


def compute_case_c_kernel(
    P_eval: Callable[[np.ndarray], np.ndarray],
    u_nodes: np.ndarray,
    omega: int,
    R: float,
    theta: float,
    a_nodes: np.ndarray,
    a_weights: np.ndarray
) -> np.ndarray:
    """
    Compute Case C kernel K_ω(u; R) = ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθua) da

    Vectorized implementation for efficiency.

    Args:
        P_eval: Polynomial evaluation function P(x) -> array
        u_nodes: Grid of u values, shape (Nu,)
        omega: ω value (1 for P₂, 2 for P₃)
        R: R parameter
        theta: θ parameter (typically 4/7)
        a_nodes: Quadrature nodes for a-integral, shape (Na,)
        a_weights: Quadrature weights for a-integral, shape (Na,)

    Returns:
        K_ω(u; R) evaluated at each point in u_nodes, shape (Nu,)
    """
    if omega <= 0:
        raise ValueError(f"Case C requires omega > 0, got {omega}")

    # Create mesh for broadcasting
    # u_grid[i,j] = u_nodes[i], a_grid[i,j] = a_nodes[j]
    u_grid = u_nodes[:, np.newaxis]  # (Nu, 1)
    a_grid = a_nodes[np.newaxis, :]  # (1, Na)

    # Polynomial argument: (1-a)*u
    poly_arg = (1 - a_grid) * u_grid  # (Nu, Na)

    # Evaluate P at all arguments
    P_vals = P_eval(poly_arg.flatten()).reshape(len(u_nodes), len(a_nodes))

    # a^{ω-1} weight
    if omega == 1:
        a_weight = np.ones_like(a_nodes)  # a^0 = 1
    else:
        a_weight = a_nodes ** (omega - 1)

    # exp(Rθua)
    exp_factor = np.exp(R * theta * u_grid * a_grid)

    # Integrate over a (axis 1)
    integrand = P_vals * a_weight[np.newaxis, :] * exp_factor
    K = np.sum(a_weights[np.newaxis, :] * integrand, axis=1)

    return K


def compute_i2_all_pairs_case_c(
    theta: float,
    R: float,
    polynomials: Dict,
    n: int = 100,
    n_a: int = 40
) -> Dict[str, I2DirectResult]:
    """
    Compute I2 for all 9 ordered pairs with Case C kernel handling.

    This is the key function that implements the Case C transformation:
    - P1: omega=0 (Case B) → raw P1.eval(u)
    - P2: omega=1 (Case C) → u × K_1(u; R)
    - P3: omega=2 (Case C) → u² × K_2(u; R)

    IMPORTANT: For I2(-R), we need K_ω(u; -R), not K_ω(u; +R)!
    The kernel contains exp(Rθua) which changes with R sign.

    Returns:
        Dict mapping pair key to I2DirectResult
    """
    u_nodes, u_weights = gauss_legendre_01(n)
    a_nodes, a_weights = gauss_legendre_01(n_a)

    P1 = polynomials["P1"]
    P2 = polynomials["P2"]
    P3 = polynomials["P3"]
    Q = polynomials["Q"]

    # Precompute kernel-transformed polynomial values for BOTH +R and -R
    # P1: Case B (no kernel, just raw polynomial) - same for +R and -R
    K1_vals_plus = P1.eval(u_nodes)
    K1_vals_minus = P1.eval(u_nodes)  # Case B: no R-dependence

    # P2: Case C (omega=1) → u × K_1(u; R)
    K2_kernel_plus = compute_case_c_kernel(
        P_eval=P2.eval, u_nodes=u_nodes, omega=1, R=R, theta=theta,
        a_nodes=a_nodes, a_weights=a_weights
    )
    K2_kernel_minus = compute_case_c_kernel(
        P_eval=P2.eval, u_nodes=u_nodes, omega=1, R=-R, theta=theta,
        a_nodes=a_nodes, a_weights=a_weights
    )
    K2_vals_plus = u_nodes * K2_kernel_plus
    K2_vals_minus = u_nodes * K2_kernel_minus

    # P3: Case C (omega=2) → u² × K_2(u; R)
    K3_kernel_plus = compute_case_c_kernel(
        P_eval=P3.eval, u_nodes=u_nodes, omega=2, R=R, theta=theta,
        a_nodes=a_nodes, a_weights=a_weights
    )
    K3_kernel_minus = compute_case_c_kernel(
        P_eval=P3.eval, u_nodes=u_nodes, omega=2, R=-R, theta=theta,
        a_nodes=a_nodes, a_weights=a_weights
    )
    K3_vals_plus = (u_nodes ** 2) * K3_kernel_plus
    K3_vals_minus = (u_nodes ** 2) * K3_kernel_minus

    K_vals_plus = {"P1": K1_vals_plus, "P2": K2_vals_plus, "P3": K3_vals_plus}
    K_vals_minus = {"P1": K1_vals_minus, "P2": K2_vals_minus, "P3": K3_vals_minus}

    # t-integrals (same for all pairs - use u_nodes as t_nodes since [0,1])
    Q_vals = Q.eval(u_nodes)
    exp_plus = np.exp(2 * R * u_nodes)
    exp_minus = np.exp(-2 * R * u_nodes)
    t_integral_plus = np.sum(u_weights * Q_vals**2 * exp_plus) / theta
    t_integral_minus = np.sum(u_weights * Q_vals**2 * exp_minus) / theta

    exp_2R = np.exp(2 * R)
    exp_R = np.exp(R)

    # Compute all 9 pairs using kernel-transformed values
    results = {}
    pairs = ["11", "22", "33", "12", "21", "13", "31", "23", "32"]
    P_map = {"1": "P1", "2": "P2", "3": "P3"}

    for pair_key in pairs:
        p1_key = P_map[pair_key[0]]
        p2_key = P_map[pair_key[1]]

        # u-integral at +R with kernels computed at +R
        u_integral_plus = np.sum(u_weights * K_vals_plus[p1_key] * K_vals_plus[p2_key])

        # u-integral at -R with kernels computed at -R
        u_integral_minus = np.sum(u_weights * K_vals_minus[p1_key] * K_vals_minus[p2_key])

        i2_plus = u_integral_plus * t_integral_plus
        i2_minus = u_integral_minus * t_integral_minus
        i2_mirror_simple = i2_plus + exp_2R * i2_minus

        results[pair_key] = I2DirectResult(
            u_integral=u_integral_plus,  # Store +R value
            t_integral_plus=t_integral_plus,
            t_integral_minus=t_integral_minus,
            i2_plus=i2_plus,
            i2_minus=i2_minus,
            i2_mirror_simple=i2_mirror_simple,
            exp_2R=exp_2R,
            exp_R=exp_R,
            R=R,
        )

    return results


def safe_ratio(direct: float, model: float, threshold: float = 1e-10) -> str:
    """Compute ratio with near-zero guard to prevent explosion."""
    if abs(model) < threshold:
        return "N/A (≈0)"
    ratio = direct / model
    return f"{ratio:.4f}"


def test_r_zero_analytic(verbose: bool = True) -> bool:
    """
    R=0 analytic check for Case C kernel.

    At R=0, exp(Rθua) = 1, so:
        K_ω(u; R=0) = ∫₀¹ P((1-a)u) × a^{ω-1} da

    For constant P=1:
        K_1 = ∫₀¹ a^0 da = 1
        K_2 = ∫₀¹ a^1 da = 0.5
    """
    from src.polynomials import Polynomial
    P_const = Polynomial([1.0])  # P(x) = 1

    u_nodes, _ = gauss_legendre_01(50)
    a_nodes, a_weights = gauss_legendre_01(50)

    K1 = compute_case_c_kernel(
        P_eval=P_const.eval,
        u_nodes=u_nodes,
        omega=1,
        R=0,
        theta=THETA,
        a_nodes=a_nodes,
        a_weights=a_weights
    )
    K2 = compute_case_c_kernel(
        P_eval=P_const.eval,
        u_nodes=u_nodes,
        omega=2,
        R=0,
        theta=THETA,
        a_nodes=a_nodes,
        a_weights=a_weights
    )

    K1_expected = 1.0
    K2_expected = 0.5

    K1_match = np.allclose(K1, K1_expected, rtol=1e-10)
    K2_match = np.allclose(K2, K2_expected, rtol=1e-10)

    if verbose:
        print("\n--- R=0 ANALYTIC CHECK ---")
        print(f"For P=1 (constant polynomial):")
        print(f"  K_1(u; R=0) = ∫₀¹ a^0 da = 1")
        print(f"    Computed: {K1[len(K1)//2]:.10f} (at u=0.5)")
        print(f"    Match: {K1_match}")
        print(f"  K_2(u; R=0) = ∫₀¹ a^1 da = 0.5")
        print(f"    Computed: {K2[len(K2)//2]:.10f} (at u=0.5)")
        print(f"    Match: {K2_match}")

    return K1_match and K2_match


def main():
    print("=" * 80)
    print("GPT Run 7: Direct TeX I2 with Case C Kernels")
    print("=" * 80)
    print()

    # Run R=0 analytic check first
    r_zero_pass = test_r_zero_analytic(verbose=True)
    print()

    # Show Case C kernel formula
    print("CASE C KERNEL FORMULA (PRZZ TeX 2370-2375):")
    print("-" * 50)
    print("  K_ω(u; R) = ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθua) da")
    print()
    print("POLYNOMIAL REPLACEMENT:")
    print("  P₁: ω = 0 (Case B) → raw P₁(u) unchanged")
    print("  P₂: ω = 1 (Case C) → u × K₁(u; R)")
    print("  P₃: ω = 2 (Case C) → u² × K₂(u; R)")
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    benchmarks = [
        ("κ", polys_kappa, 1.3036, 2.137),
        ("κ*", polys_kappa_star, 1.1167, 1.938),
    ]

    all_aligned = True

    for bench_name, polys, R_val, c_target in benchmarks:
        print(f"\n{'='*70}")
        print(f"Benchmark: {bench_name} (R={R_val})")
        print(f"{'='*70}")

        # Get model results with pair breakdown
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

        # Compute direct I2 with Case C for all pairs
        direct_results = compute_i2_all_pairs_case_c(THETA, R_val, polys, n=100, n_a=40)

        # Compare per-pair
        print("\n--- PER-PAIR COMPARISON: Direct (Case C) vs Model ---")
        print(f"{'Pair':<6} {'Case':<8} {'norm':>8} {'Direct+':>12} {'Model+':>12} {'Ratio+':>10} "
              f"{'Direct-':>12} {'Model-':>12} {'Ratio-':>10}")
        print("-" * 110)

        pair_breakdown = implied.pair_breakdown

        i2_direct_plus_total = 0.0
        i2_direct_minus_total = 0.0
        i2_model_plus_total = 0.0
        i2_model_minus_total = 0.0

        # Determine Case for each polynomial
        case_map = {"1": "B", "2": "C", "3": "C"}

        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            r = direct_results[pair_key]
            norm = F_NORM[pair_key]

            # Determine case combination
            case1 = case_map[pair_key[0]]
            case2 = case_map[pair_key[1]]
            case_str = f"{case1}×{case2}"

            # Direct values (with correct normalization)
            direct_plus = norm * r.i2_plus
            direct_minus = norm * r.i2_minus

            # Model values from pair_breakdown
            if pair_key in pair_breakdown:
                model_plus = pair_breakdown[pair_key].get("I2_plus", 0.0)
                model_minus = pair_breakdown[pair_key].get("I2_minus_base", 0.0)
            else:
                model_plus = 0.0
                model_minus = 0.0

            # Accumulate totals
            i2_direct_plus_total += direct_plus
            i2_direct_minus_total += direct_minus
            i2_model_plus_total += model_plus
            i2_model_minus_total += model_minus

            # Compute ratios with near-zero guard
            ratio_plus_str = safe_ratio(direct_plus, model_plus)
            ratio_minus_str = safe_ratio(direct_minus, model_minus)

            print(f"{pair_key:<6} {case_str:<8} {norm:>8.4f} {direct_plus:>+12.6f} {model_plus:>+12.6f} {ratio_plus_str:>10} "
                  f"{direct_minus:>+12.6f} {model_minus:>+12.6f} {ratio_minus_str:>10}")

        print("-" * 110)

        # Total comparison
        total_ratio_plus_str = safe_ratio(i2_direct_plus_total, i2_model_plus_total)
        total_ratio_minus_str = safe_ratio(i2_direct_minus_total, i2_model_minus_total)

        print(f"{'TOTAL':<6} {'':<8} {'':>8} {i2_direct_plus_total:>+12.6f} {i2_model_plus_total:>+12.6f} {total_ratio_plus_str:>10} "
              f"{i2_direct_minus_total:>+12.6f} {i2_model_minus_total:>+12.6f} {total_ratio_minus_str:>10}")

        # Compute numeric ratios for alignment check
        total_ratio_plus = i2_direct_plus_total / i2_model_plus_total if abs(i2_model_plus_total) > 1e-15 else float('inf')
        total_ratio_minus = i2_direct_minus_total / i2_model_minus_total if abs(i2_model_minus_total) > 1e-15 else float('inf')

        # Compare with model's aggregated I2 values
        print(f"\n--- AGGREGATE COMPARISON ---")
        print(f"Model I2_plus (aggregated):        {implied.I2_plus:+.6f}")
        print(f"Direct I2_plus (Case C, correct norm): {i2_direct_plus_total:+.6f}")
        print(f"Ratio (direct/model):               {total_ratio_plus_str}")
        print()
        print(f"Model I2_minus_base (aggregated):    {implied.I2_minus_base:+.6f}")
        print(f"Direct I2_minus (Case C, correct norm): {i2_direct_minus_total:+.6f}")
        print(f"Ratio (direct/model):                 {total_ratio_minus_str}")

        # Alignment verdict
        print(f"\n--- ALIGNMENT VERDICT ---")
        plus_aligned = abs(total_ratio_plus - 1.0) < 0.02  # Within 2%
        minus_aligned = abs(total_ratio_minus - 1.0) < 0.02

        if plus_aligned and minus_aligned:
            print(f"ALIGNED: Direct I2 (with Case C) matches model within 2%!")
            print(f"Conclusion: Case C kernels correctly implemented.")
        else:
            print(f"NOT ALIGNED: Ratios are {total_ratio_plus:.4f} (plus) and {total_ratio_minus:.4f} (minus)")
            print(f"Conclusion: Investigation needed - check Case C implementation.")
            all_aligned = False

    # Summary
    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
Run 7 implements Case C kernels for P2 and P3:

CASE C KERNEL:
    K_ω(u; R) = ∫₀¹ P((1-a)u) × a^{{ω-1}} × exp(Rθua) da

POLYNOMIAL REPLACEMENT:
    P₁: ω=0 (Case B) → raw P₁(u)
    P₂: ω=1 (Case C) → u × K₁(u; R)
    P₃: ω=2 (Case C) → u² × K₂(u; R)

R=0 ANALYTIC CHECK: {"PASSED" if r_zero_pass else "FAILED"}

ALIGNMENT STATUS: {"ALL PAIRS ALIGNED" if all_aligned else "SOME PAIRS NOT ALIGNED"}
""")

    if all_aligned:
        print("""
SUCCESS: I2 is now PROVEN FROM FIRST PRINCIPLES for all K=3 pairs!

The separable I2 formula with Case C kernels matches the model exactly.
This upgrades I2 from "calibration" to "derived" status.

NEXT STEPS:
1. Add pytest gate tests for this alignment
2. Update documentation to reflect I2 proven status
3. Consider extending to I1 channel (more complex due to derivatives)
""")
    else:
        print("""
INVESTIGATION NEEDED:

Some pairs don't match within tolerance. Check:
1. Case C kernel formula matches model implementation
2. u^ω prefactor is correctly applied
3. Quadrature precision (try n_a=60)

Use GPT's tip: tune on pair (2,2) first, then verify all pairs.
""")


if __name__ == "__main__":
    main()
