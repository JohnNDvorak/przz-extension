#!/usr/bin/env python3
"""
scripts/run_phase27_meff_report.py
Phase 27: Mirror m_eff Diagnostic Report

This script computes the derived mirror transform for all 6 pairs and
compares against the -R proxy approach to determine the effective
multiplier m_eff.

The goal is to understand how the derived mirror relates to the
empirical formula m = exp(R) + 5.

Created: 2025-12-26 (Phase 27)
"""

import sys
import math

# Add project root to path
sys.path.insert(0, ".")

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.mirror_transform_derived import (
    compute_I1_mirror_derived,
    compute_I1_mirror_P1Q1,
)
from src.unified_i1_general import (
    compute_I1_unified_general,
    compute_I1_unified_general_P1Q1,
)


def run_meff_diagnostic_single_mode(
    R: float, theta: float, polynomials: dict, benchmark_name: str,
    T_prefactor_mode: str, n_quad: int = 60
) -> dict:
    """Run m_eff diagnostic for a single T_prefactor mode."""

    print(f"\n{'=' * 80}")
    print(f"M_EFF DIAGNOSTIC - {benchmark_name} (R={R})")
    print(f"T_prefactor_mode: {T_prefactor_mode}")
    print(f"{'=' * 80}")

    m_empirical = math.exp(R) + 5
    print(f"\nEmpirical m = exp(R) + 5 = {m_empirical:.6f}")
    print(f"exp(2R) = {math.exp(2*R):.6f}")
    print(f"exp(2R/theta) = {math.exp(2*R/theta):.6f}")

    pairs = ["11", "22", "33", "12", "13", "23"]
    results = {}

    print(f"\n{'-' * 100}")
    print(f"{'Pair':<6} {'Direct(+R)':<14} {'Mirror':<14} {'Proxy(-R)':<14} "
          f"{'m_eff':<12} {'m_emp ratio':<12}")
    print(f"{'-' * 100}")

    for pair in pairs:
        ell1, ell2 = int(pair[0]), int(pair[1])

        # Direct at +R
        direct_result = compute_I1_unified_general(
            R=R, theta=theta, ell1=ell1, ell2=ell2,
            polynomials=polynomials, n_quad_u=n_quad, n_quad_t=n_quad,
        )
        direct = direct_result.I1_value

        # Derived mirror with specified mode
        mirror_result = compute_I1_mirror_derived(
            R=R, theta=theta, ell1=ell1, ell2=ell2,
            polynomials=polynomials, n_quad_u=n_quad, n_quad_t=n_quad,
            T_prefactor_mode=T_prefactor_mode,
        )
        mirror = mirror_result.I1_mirror_value

        # Proxy using -R
        proxy_result = compute_I1_unified_general(
            R=-R, theta=theta, ell1=ell1, ell2=ell2,
            polynomials=polynomials, n_quad_u=n_quad, n_quad_t=n_quad,
        )
        proxy = proxy_result.I1_value

        # Effective multiplier
        if abs(proxy) > 1e-15:
            m_eff = mirror / proxy
            m_ratio = m_eff / m_empirical
        else:
            m_eff = None
            m_ratio = None

        results[pair] = {
            "direct": direct,
            "mirror": mirror,
            "proxy": proxy,
            "m_eff": m_eff,
        }

        m_eff_str = f"{m_eff:.4f}" if m_eff is not None else "N/A"
        m_ratio_str = f"{m_ratio:.4f}" if m_ratio is not None else "N/A"
        print(f"({ell1},{ell2})  {direct:>12.6e}  {mirror:>12.6e}  "
              f"{proxy:>12.6e}  {m_eff_str:>10}  {m_ratio_str:>10}")

    print(f"{'-' * 100}")

    # Compute total S12 using both approaches
    S12_direct = sum(
        (1 if k[0] == k[1] else 2) * results[k]["direct"] /
        (math.factorial(int(k[0])) * math.factorial(int(k[1])))
        for k in pairs
    )
    S12_mirror = sum(
        (1 if k[0] == k[1] else 2) * results[k]["mirror"] /
        (math.factorial(int(k[0])) * math.factorial(int(k[1])))
        for k in pairs
    )
    S12_proxy = sum(
        (1 if k[0] == k[1] else 2) * results[k]["proxy"] /
        (math.factorial(int(k[0])) * math.factorial(int(k[1])))
        for k in pairs
    )

    print(f"\nTotal S12 (with factorial normalization and symmetry):")
    print(f"  S12_direct(+R) = {S12_direct:.8f}")
    print(f"  S12_mirror     = {S12_mirror:.8f}")
    print(f"  S12_proxy(-R)  = {S12_proxy:.8f}")

    if abs(S12_proxy) > 1e-15:
        S12_m_eff = S12_mirror / S12_proxy
        print(f"\nTotal m_eff (S12_mirror / S12_proxy) = {S12_m_eff:.6f}")
        print(f"  vs empirical m = {m_empirical:.6f}")
        print(f"  ratio = {S12_m_eff / m_empirical:.6f}")

    return results


def run_p1q1_diagnostic(R: float, theta: float):
    """Run P=Q=1 microcase diagnostic."""

    print(f"\n{'=' * 80}")
    print(f"P=Q=1 MICROCASE DIAGNOSTIC (R={R}, theta={theta:.6f})")
    print(f"{'=' * 80}")

    for ell1, ell2 in [(1, 1), (2, 2), (1, 2)]:
        direct = compute_I1_unified_general_P1Q1(
            R=R, theta=theta, ell1=ell1, ell2=ell2,
            n_quad_u=60, n_quad_t=60,
        )

        # Test different modes
        print(f"\n({ell1},{ell2}):")
        print(f"  Direct: {direct:.8e}")

        for mode in ["absorbed", "none", "exp_2R"]:
            mirror = compute_I1_mirror_P1Q1(
                R=R, theta=theta, ell1=ell1, ell2=ell2,
                n_quad_u=60, n_quad_t=60,
            )
            # For P=Q=1, manually compute with different interpretations
            # since the function doesn't take mode parameter
            print(f"  Mirror (raw): {mirror:.8e}")
            print(f"  Mirror × exp(2R): {mirror * math.exp(2*R):.8e}")
            break


def main():
    """Main diagnostic routine."""

    print("Phase 27: Mirror m_eff Diagnostic Report")
    print("=" * 80)

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    kappa_polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    kappa_star_polys = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Parameters
    R_kappa = 1.3036
    R_kappa_star = 1.1167
    theta = 4 / 7

    # P=Q=1 diagnostic
    run_p1q1_diagnostic(R_kappa, theta)

    # Test different T_prefactor modes for kappa benchmark
    for mode in ["absorbed", "none", "exp_2R"]:
        run_meff_diagnostic_single_mode(
            R_kappa, theta, kappa_polys, "KAPPA", mode, n_quad=60
        )

    # Best mode for kappa* benchmark
    run_meff_diagnostic_single_mode(
        R_kappa_star, theta, kappa_star_polys, "KAPPA*", "exp_2R", n_quad=60
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings:
1. The derived mirror with exp(2R) prefactor gives m_eff values
2. Compare m_eff to empirical m = exp(R) + 5
3. If they match, we have correctly derived the mirror transform
4. If not, investigate the structure further
""")


if __name__ == "__main__":
    main()
