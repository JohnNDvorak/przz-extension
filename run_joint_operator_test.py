"""
run_joint_operator_test.py
Test the joint Q-shift operator mode (Codex Run 3)

Compare standard operator (Q → Q_lift) vs joint operator (Q → Q_lift_norm).
"""

from __future__ import annotations

from src.evaluate import (
    compute_c_paper_operator_q_shift,
    compute_c_paper_operator_q_shift_joint,
    solve_two_weight_operator,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_C_TARGET = 2.137
KAPPA_STAR_R = 1.1167
KAPPA_STAR_C_TARGET = 1.938


def main():
    print("=" * 80)
    print("JOINT OPERATOR MODE TEST (Codex Run 3)")
    print("=" * 80)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_s, "P2": P2_s, "P3": P3_s, "Q": Q_s}

    n_quad = 40
    n_quad_a = 30

    # =========================================================================
    # STANDARD OPERATOR MODE
    # =========================================================================
    print("STANDARD OPERATOR MODE (Q → Q_lift):")
    print("-" * 60)

    result_k_std = compute_c_paper_operator_q_shift(
        theta=THETA, R=KAPPA_R, n=n_quad, polynomials=polys_kappa,
        n_quad_a=n_quad_a, verbose=False,
    )
    result_ks_std = compute_c_paper_operator_q_shift(
        theta=THETA, R=KAPPA_STAR_R, n=n_quad, polynomials=polys_kappa_star,
        n_quad_a=n_quad_a, verbose=False,
    )

    print(f"  κ:  m1_implied={result_k_std.per_term['_m1_implied']:.4f}, "
          f"m2_implied={result_k_std.per_term['_m2_implied']:.4f}")
    print(f"  κ*: m1_implied={result_ks_std.per_term['_m1_implied']:.4f}, "
          f"m2_implied={result_ks_std.per_term['_m2_implied']:.4f}")
    print()

    # =========================================================================
    # JOINT OPERATOR MODE
    # =========================================================================
    print("JOINT OPERATOR MODE (Q → Q_lift * Q(0)/Q(1)):")
    print("-" * 60)

    result_k_joint = compute_c_paper_operator_q_shift_joint(
        theta=THETA, R=KAPPA_R, n=n_quad, polynomials=polys_kappa,
        n_quad_a=n_quad_a, verbose=True,
    )
    result_ks_joint = compute_c_paper_operator_q_shift_joint(
        theta=THETA, R=KAPPA_STAR_R, n=n_quad, polynomials=polys_kappa_star,
        n_quad_a=n_quad_a, verbose=True,
    )

    print()
    print(f"  κ:  m1_implied={result_k_joint.per_term['_m1_implied']:.4f}, "
          f"m2_implied={result_k_joint.per_term['_m2_implied']:.4f}")
    print(f"  κ*: m1_implied={result_ks_joint.per_term['_m1_implied']:.4f}, "
          f"m2_implied={result_ks_joint.per_term['_m2_implied']:.4f}")
    print()

    # =========================================================================
    # SOLVE 2×2 WITH JOINT MODE
    # =========================================================================
    print("=" * 80)
    print("2×2 SOLVE COMPARISON")
    print("=" * 80)
    print()

    # Standard operator 2×2 solve
    std_solve = solve_two_weight_operator(
        result_k_std, result_ks_std,
        c_target_k=KAPPA_C_TARGET, c_target_k_star=KAPPA_STAR_C_TARGET,
        use_operator_channels=True,
    )

    # Joint operator 2×2 solve
    joint_solve = solve_two_weight_operator(
        result_k_joint, result_ks_joint,
        c_target_k=KAPPA_C_TARGET, c_target_k_star=KAPPA_STAR_C_TARGET,
        use_operator_channels=True,
    )

    # Base solve (for reference)
    base_solve = solve_two_weight_operator(
        result_k_std, result_ks_std,
        c_target_k=KAPPA_C_TARGET, c_target_k_star=KAPPA_STAR_C_TARGET,
        use_operator_channels=False,
    )

    print(f"{'Mode':<25} {'m1':>12} {'m2':>12} {'cond':>10}")
    print("-" * 60)
    print(f"{'BASE (I_minus_base)':<25} {base_solve['m1']:>12.4f} {base_solve['m2']:>12.4f} {base_solve['cond']:>10.2f}")
    print(f"{'STANDARD OP (I_minus_op)':<25} {std_solve['m1']:>12.4f} {std_solve['m2']:>12.4f} {std_solve['cond']:>10.2f}")
    print(f"{'JOINT OP (I_minus_op)':<25} {joint_solve['m1']:>12.4f} {joint_solve['m2']:>12.4f} {joint_solve['cond']:>10.2f}")
    print()

    # =========================================================================
    # VERDICT
    # =========================================================================
    print("=" * 80)
    if joint_solve['cond'] < std_solve['cond'] * 0.5:
        print("JOINT MODE: Better conditioning than standard operator")
        if joint_solve['cond'] < base_solve['cond'] * 2:
            print("         → Joint mode is viable for 2×2 solve")
    else:
        print("JOINT MODE: Similar or worse conditioning than standard operator")
        print("         → Normalization did not fix the benchmark divergence")
    print("=" * 80)


if __name__ == "__main__":
    main()
