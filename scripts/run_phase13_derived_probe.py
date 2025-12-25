#!/usr/bin/env python3
"""
scripts/run_phase13_derived_probe.py
Phase 13.0: Derived-vs-Derived Integrand Probe

GPT's key insight: Phase 12 compared operator-mirror to empirical(-R), which is
apples-to-oranges. We should compare derived mirror to direct(t→1−t) reference
to detect internal implementation bugs.

If the complement eigenvalues satisfy A_mirror(t) = A_direct(1−t), then the
mirror integrand should match the direct integrand evaluated at t'=1−t
(with appropriate prefactors).

KEY HYPOTHESIS TO TEST:
-----------------------
Phase 12's complement eigenvalues:
    A_α^mirror(t) = 1 - A_β(t) = (1-t) - θt·x + θ(1-t)·y
    A_β^mirror(t) = 1 - A_α(t) = (1-t) + θ(1-t)·x - θt·y

Direct eigenvalues at t' = 1-t:
    A_α(1-t) = (1-t) + θ((1-t)-1)·x + θ(1-t)·y = (1-t) - θt·x + θ(1-t)·y
    A_β(1-t) = (1-t) + θ(1-t)·x + θ((1-t)-1)·y = (1-t) + θ(1-t)·x - θt·y

These ARE IDENTICAL! So Q(A_α^mirror(t)) = Q(A_α(1-t)).

BUT the exp factors differ:
    Mirror exp:      exp(2Rt - θR(x+y))           [lines 218-221 of mirror_operator_exact.py]
    Direct exp(1-t): exp(2R(1-t) + θR(1-2t)(x+y)) [from get_exp_affine_coeffs with t→1-t]

If ratios vary wildly → exp kernel mismatch (bug)
If ratios are stable → derived mirror is internally consistent
"""

import numpy as np
from src.polynomials import load_przz_polynomials
from src.series import TruncatedSeries
from src.composition import compose_polynomial_on_affine, compose_exp_on_affine
from src.operator_post_identity import (
    get_A_alpha_affine_coeffs,
    get_A_beta_affine_coeffs,
    get_exp_affine_coeffs,
    apply_QQexp_post_identity_composition,
)
from src.mirror_operator_exact import (
    apply_QQexp_mirror_composition,
    get_mirror_eigenvalues_complement_t,
    get_mirror_exp_affine_coeffs,
)


def compute_direct_integrand(
    u: float, t: float,
    theta: float, R: float,
    P1_poly, Q_poly
) -> float:
    """
    Compute the DIRECT integrand for I1 at (u, t).

    Uses: Q(A_α(t))Q(A_β(t)) × exp(2Rt + θR(2t-1)(x+y))
    """
    var_names = ("x", "y")

    # Profile at u
    P1_series = compose_polynomial_on_affine(P1_poly, u, {"x": 1.0}, var_names)
    P2_series = compose_polynomial_on_affine(P1_poly, u, {"y": 1.0}, var_names)

    # Direct Q×Q×exp composition
    core = apply_QQexp_post_identity_composition(Q_poly, t, theta, R, var_names)

    # Algebraic prefactor: 1/θ + x + y
    alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

    # Product
    product = P1_series * P2_series * core * alg_prefactor

    # Extract xy coefficient with (1-u)² prefactor
    scalar_prefactor = (1 - u) ** 2
    xy_coeff = product.extract(("x", "y"))

    return scalar_prefactor * xy_coeff


def compute_direct_integrand_t_flip(
    u: float, t: float,
    theta: float, R: float,
    P1_poly, Q_poly
) -> float:
    """
    Compute the DIRECT integrand at t' = 1-t (the t-flip reference).

    This is what the mirror integrand SHOULD match if eigenvalues and exp
    are both transformed consistently under t → 1-t.
    """
    t_prime = 1.0 - t
    return compute_direct_integrand(u, t_prime, theta, R, P1_poly, Q_poly)


def compute_phase12_mirror_integrand(
    u: float, t: float,
    theta: float, R: float,
    P1_poly, Q_poly
) -> float:
    """
    Compute Phase 12 (t-dependent complement) mirror integrand for I1 at (u, t).

    Uses: Q(A_α^mirror(t))Q(A_β^mirror(t)) × exp(2Rt - θR(x+y)) × exp(2R)

    Note: exp(2R) is the T^{-(α+β)} weight factor.
    BUGGY: Uses static exp coefficients!
    """
    var_names = ("x", "y")

    # Profile at u
    P1_series = compose_polynomial_on_affine(P1_poly, u, {"x": 1.0}, var_names)
    P2_series = compose_polynomial_on_affine(P1_poly, u, {"y": 1.0}, var_names)

    # Phase 12: T-dependent complement eigenvalues with STATIC mirror exp (BUGGY!)
    core = apply_QQexp_mirror_composition(Q_poly, t, theta, R, var_names,
                                          use_t_dependent=True, use_t_flip_exp=False)

    # Algebraic prefactor
    alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

    # Product
    product = P1_series * P2_series * core * alg_prefactor

    # Extract xy coefficient
    scalar_prefactor = (1 - u) ** 2
    xy_coeff = product.extract(("x", "y"))

    # T weight: exp(2R)
    T_weight = np.exp(2 * R)

    return scalar_prefactor * xy_coeff * T_weight


def compute_phase13_mirror_integrand(
    u: float, t: float,
    theta: float, R: float,
    P1_poly, Q_poly
) -> float:
    """
    Compute Phase 13 (t-dependent complement + t-flip exp) mirror integrand.

    Uses: Q(A_α^mirror(t))Q(A_β^mirror(t)) × exp(2Rt + θR(1-2t)(x+y)) × exp(2R)

    This is the CORRECTED version where exp coefficients match direct at t' = 1-t.
    """
    var_names = ("x", "y")

    # Profile at u
    P1_series = compose_polynomial_on_affine(P1_poly, u, {"x": 1.0}, var_names)
    P2_series = compose_polynomial_on_affine(P1_poly, u, {"y": 1.0}, var_names)

    # Phase 13: T-dependent complement eigenvalues with T-FLIP exp (FIXED!)
    core = apply_QQexp_mirror_composition(Q_poly, t, theta, R, var_names,
                                          use_t_dependent=True, use_t_flip_exp=True)

    # Algebraic prefactor
    alg_prefactor = TruncatedSeries.from_scalar(1.0 / theta, var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("x", var_names)
    alg_prefactor = alg_prefactor + TruncatedSeries.variable("y", var_names)

    # Product
    product = P1_series * P2_series * core * alg_prefactor

    # Extract xy coefficient
    scalar_prefactor = (1 - u) ** 2
    xy_coeff = product.extract(("x", "y"))

    # T weight: exp(2R)
    T_weight = np.exp(2 * R)

    return scalar_prefactor * xy_coeff * T_weight


def analyze_exp_factor_mismatch(t: float, theta: float, R: float):
    """
    Analyze the exp factor mismatch between mirror and direct(t→1-t).

    Mirror exp:      exp(2Rt - θR(x+y))
    Direct exp(1-t): exp(2R(1-t) + θR(1-2t)(x+y))

    At different t values, the (x+y) coefficients differ!
    """
    # Mirror exp coefficients
    mir_u0, mir_x, mir_y = get_mirror_exp_affine_coeffs(t, theta, R)

    # Direct exp coefficients at t' = 1-t
    t_prime = 1.0 - t
    dir_u0, dir_x, dir_y = get_exp_affine_coeffs(t_prime, theta, R)

    return {
        "mirror": {"u0": mir_u0, "x": mir_x, "y": mir_y},
        "direct_t_flip": {"u0": dir_u0, "x": dir_x, "y": dir_y},
        "u0_diff": mir_u0 - dir_u0,
        "x_diff": mir_x - dir_x,
        "y_diff": mir_y - dir_y,
    }


def main():
    print("=" * 80)
    print("Phase 13.0: Derived-vs-Derived Integrand Probe")
    print("=" * 80)
    print()
    print("GPT's Key Insight:")
    print("  Phase 12 compared mirror to empirical(-R) → apples-to-oranges!")
    print("  Compare mirror to direct(t→1−t) → detect implementation bugs.")
    print()

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    theta = 4.0 / 7.0
    R = 1.3036

    # Test nodes
    test_nodes = [
        (0.2, 0.5),
        (0.8, 0.5),
        (0.5, 0.2),
        (0.5, 0.8),
        (0.3, 0.3),
        (0.7, 0.7),
    ]

    print(f"Parameters: theta={theta:.4f}, R={R:.4f}")
    print()

    # ==========================================================================
    # Part 1: Analyze exp factor mismatch
    # ==========================================================================
    print("=" * 80)
    print("Part 1: Exp Factor Mismatch Analysis")
    print("=" * 80)
    print()
    print("  Mirror exp:      exp(2Rt - θR(x+y))")
    print("  Direct exp(1-t): exp(2R(1-t) + θR(1-2t)(x+y))")
    print()
    print(f"{'t':>6} {'mir_u0':>10} {'dir_u0':>10} {'diff_u0':>10} | "
          f"{'mir_x':>10} {'dir_x':>10} {'diff_x':>10}")
    print("-" * 80)

    for t in [0.0, 0.2, 0.5, 0.8, 1.0]:
        exp_data = analyze_exp_factor_mismatch(t, theta, R)
        mir = exp_data["mirror"]
        dir_tf = exp_data["direct_t_flip"]

        print(f"{t:>6.2f} {mir['u0']:>10.4f} {dir_tf['u0']:>10.4f} {exp_data['u0_diff']:>10.4f} | "
              f"{mir['x']:>10.4f} {dir_tf['x']:>10.4f} {exp_data['x_diff']:>10.4f}")

    print()
    print("CRITICAL OBSERVATION:")
    print("  - u0: Mirror has 2Rt, Direct(1-t) has 2R(1-t)")
    print("    Sum: mir_u0 + dir_u0(at t) = 2Rt + 2R(1-t) = 2R (constant)")
    print("  - x,y: Mirror has -θR, Direct(1-t) has θR(1-2t)")
    print("    At t=0.5: Mirror = -θR, Direct = 0 → MISMATCH!")
    print()

    # ==========================================================================
    # Part 2: Eigenvalue comparison (should be identical under complement)
    # ==========================================================================
    print("=" * 80)
    print("Part 2: Eigenvalue Verification (A_mirror(t) == A_direct(1-t)?)")
    print("=" * 80)
    print()

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t_prime = 1.0 - t

        # Mirror eigenvalues at t
        eig_mir = get_mirror_eigenvalues_complement_t(t, theta)

        # Direct eigenvalues at 1-t
        u0_a_dir, x_a_dir, y_a_dir = get_A_alpha_affine_coeffs(t_prime, theta)
        u0_b_dir, x_b_dir, y_b_dir = get_A_beta_affine_coeffs(t_prime, theta)

        # Check if they match
        alpha_match = (
            abs(eig_mir.u0_alpha - u0_a_dir) < 1e-10 and
            abs(eig_mir.x_alpha - x_a_dir) < 1e-10 and
            abs(eig_mir.y_alpha - y_a_dir) < 1e-10
        )

        beta_match = (
            abs(eig_mir.u0_beta - u0_b_dir) < 1e-10 and
            abs(eig_mir.x_beta - x_b_dir) < 1e-10 and
            abs(eig_mir.y_beta - y_b_dir) < 1e-10
        )

        status_a = "✓ MATCH" if alpha_match else "✗ MISMATCH"
        status_b = "✓ MATCH" if beta_match else "✗ MISMATCH"

        print(f"t={t:.2f}: A_α^mirror vs A_α(1-t): {status_a}, A_β^mirror vs A_β(1-t): {status_b}")

    print()
    print("CONCLUSION: Eigenvalues ARE correctly transformed under complement.")
    print("But the exp factors are NOT transformed consistently!")
    print()

    # ==========================================================================
    # Part 3: Integrand comparison (Phase 12 vs Direct(1-t))
    # ==========================================================================
    print("=" * 80)
    print("Part 3: Phase 12 Integrand Comparison (Mirror vs Direct(t→1-t))")
    print("=" * 80)
    print()
    print("-" * 80)
    print(f"{'(u, t)':<12} {'Direct(t)':>14} {'P12 Mirror':>14} {'Direct(1-t)':>14} "
          f"{'P12/Dir(1-t)':>12}")
    print("-" * 80)

    ratios_p12 = []

    for u, t in test_nodes:
        direct_t = compute_direct_integrand(u, t, theta, R, P1, Q)
        mirror_p12 = compute_phase12_mirror_integrand(u, t, theta, R, P1, Q)
        direct_t_flip = compute_direct_integrand_t_flip(u, t, theta, R, P1, Q)

        if abs(direct_t_flip) > 1e-15:
            ratio = mirror_p12 / direct_t_flip
        else:
            ratio = float('inf')

        ratios_p12.append(ratio)

        print(f"({u:.1f}, {t:.1f})    {direct_t:>14.6f} {mirror_p12:>14.6f} "
              f"{direct_t_flip:>14.6f} {ratio:>12.4f}")

    print("-" * 80)

    # ==========================================================================
    # Part 4: Phase 13 with t-flip exp (FIXED!)
    # ==========================================================================
    print()
    print("=" * 80)
    print("Part 4: Phase 13 Integrand Comparison (FIXED t-flip exp)")
    print("=" * 80)
    print()
    print("-" * 80)
    print(f"{'(u, t)':<12} {'Direct(1-t)':>14} {'P13 Mirror':>14} "
          f"{'P13/Dir(1-t)':>12}")
    print("-" * 80)

    ratios_p13 = []

    for u, t in test_nodes:
        mirror_p13 = compute_phase13_mirror_integrand(u, t, theta, R, P1, Q)
        direct_t_flip = compute_direct_integrand_t_flip(u, t, theta, R, P1, Q)

        if abs(direct_t_flip) > 1e-15:
            ratio = mirror_p13 / direct_t_flip
        else:
            ratio = float('inf')

        ratios_p13.append(ratio)

        print(f"({u:.1f}, {t:.1f})    {direct_t_flip:>14.6f} {mirror_p13:>14.6f} "
              f"{ratio:>12.4f}")

    print("-" * 80)

    # Statistics for both
    finite_p12 = [r for r in ratios_p12 if r != float('inf') and not np.isnan(r)]
    finite_p13 = [r for r in ratios_p13 if r != float('inf') and not np.isnan(r)]

    print()
    print("=== Ratio Analysis: Phase 12 (buggy exp) ===")
    if finite_p12:
        print(f"  Mean:  {np.mean(finite_p12):.4f}")
        print(f"  Std:   {np.std(finite_p12):.4f}")
        cv_p12 = np.std(finite_p12) / abs(np.mean(finite_p12)) if np.mean(finite_p12) != 0 else float('inf')
        print(f"  CV:    {cv_p12:.4f}")

    print()
    print("=== Ratio Analysis: Phase 13 (fixed t-flip exp) ===")
    if finite_p13:
        print(f"  Mean:  {np.mean(finite_p13):.4f}")
        print(f"  Std:   {np.std(finite_p13):.4f}")
        cv_p13 = np.std(finite_p13) / abs(np.mean(finite_p13)) if np.mean(finite_p13) != 0 else float('inf')
        print(f"  CV:    {cv_p13:.4f}")

        if cv_p13 < 0.1:
            print()
            print("SUCCESS! Phase 13 ratios are stable (CV < 0.1)")
            print(f"  Implied T-weight factor: ~{np.mean(finite_p13):.4f}")
            print(f"  Expected exp(2R) = {np.exp(2*R):.4f}")
        else:
            print()
            print("Phase 13 still shows variation - investigate further")

    print()
    print("=" * 80)
    print("Phase 13.0 Conclusion")
    print("=" * 80)
    print()
    print("The derived mirror implementation transforms eigenvalues correctly:")
    print("  A_α^mirror(t) = A_α(1-t)  ✓")
    print("  A_β^mirror(t) = A_β(1-t)  ✓")
    print()
    print("Phase 12 (buggy) exp kernel:")
    print("  Mirror exp: exp(2Rt - θR(x+y))  ← STATIC coefficient")
    print()
    print("Phase 13 (fixed) exp kernel:")
    print("  Mirror exp: exp(2Rt + θR(1-2t)(x+y))  ← t-DEPENDENT coefficient")
    print()
    if finite_p13 and cv_p13 < 0.1:
        print("PHASE 13 FIX SUCCESSFUL!")
        print(f"  Ratios now stable with CV = {cv_p13:.4f}")
        print(f"  All ratios ≈ {np.mean(finite_p13):.4f}")
    else:
        print("Phase 13 fix applied but ratios still vary.")


if __name__ == "__main__":
    main()
