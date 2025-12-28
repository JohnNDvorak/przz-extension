#!/usr/bin/env python3
"""
scripts/run_phase28_s12_matrix_compare.py
Phase 28: S12 Backend Matrix Comparison

Prints the full 3×3 ordered I1/I2 matrices from both backends:
- unified_general (Phase 26B)
- term_dsl (evaluate.py)

This reveals whether the backends compute the same mathematical object
under different conventions, or fundamentally different objects.

Created: 2025-12-26 (Phase 28)
"""

import sys
import math

sys.path.insert(0, ".")

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluator.s12_spec import (
    S12CanonicalValue,
    S12FullMatrix,
    FactorialMode,
    SignMode,
    get_ordered_pairs,
    parse_pair_key,
)


def compute_unified_general_matrix(
    R: float,
    theta: float,
    polynomials: dict,
    term_type: str,  # "I1" or "I2"
    n_quad: int = 60,
) -> S12FullMatrix:
    """
    Compute full 3×3 I1 or I2 matrix using unified_general backend.

    Note: unified_general applies factorial normalization (ℓ₁!×ℓ₂!) and
    off-diagonal sign (-1)^{ℓ₁+ℓ₂}.
    """
    from src.unified_i1_general import compute_I1_unified_general
    from src.unified_i2_general import compute_I2_unified_general

    values = {}

    for ell1 in [1, 2, 3]:
        for ell2 in [1, 2, 3]:
            if term_type == "I1":
                result = compute_I1_unified_general(
                    R=R, theta=theta, ell1=ell1, ell2=ell2,
                    polynomials=polynomials, n_quad_u=n_quad, n_quad_t=n_quad,
                )
                value = result.I1_value
            else:  # I2
                result = compute_I2_unified_general(
                    R=R, theta=theta, ell1=ell1, ell2=ell2,
                    polynomials=polynomials, n_quad_u=n_quad, n_quad_t=n_quad,
                )
                value = result.I2_value

            values[(ell1, ell2)] = S12CanonicalValue(
                ell1=ell1,
                ell2=ell2,
                value=value,
                factorial_mode=FactorialMode.DERIVATIVE,  # unified_general includes ℓ!
                sign_mode=SignMode.OFFDIAG_ALTERNATING,   # includes (-1)^{ℓ₁+ℓ₂}
                backend="unified_general",
                term_type=term_type,
            )

    return S12FullMatrix(
        term_type=term_type,
        backend="unified_general",
        R=R,
        theta=theta,
        values=values,
    )


def compute_term_dsl_matrix(
    R: float,
    theta: float,
    polynomials: dict,
    term_type: str,  # "I1" or "I2"
    n_quad: int = 60,
) -> S12FullMatrix:
    """
    Compute full 3×3 I1 or I2 matrix using term_dsl backend (evaluate.py).

    Note: term_dsl returns raw coefficient values without factorial
    normalization applied.
    """
    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term

    # Build terms for all pairs
    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")

    values = {}

    # term_dsl uses triangle keys, so we need to handle ordered pairs
    for ell1 in [1, 2, 3]:
        for ell2 in [1, 2, 3]:
            # Get triangle key (smaller first)
            if ell1 <= ell2:
                triangle_key = f"{ell1}{ell2}"
                is_swapped = False
            else:
                triangle_key = f"{ell2}{ell1}"
                is_swapped = True

            terms = all_terms.get(triangle_key, [])

            # Find the correct term (I1 is index 0, I2 is index 1)
            term_idx = 0 if term_type == "I1" else 1

            if term_idx >= len(terms):
                # Term not available
                values[(ell1, ell2)] = S12CanonicalValue(
                    ell1=ell1,
                    ell2=ell2,
                    value=0.0,
                    factorial_mode=FactorialMode.COEFFICIENT,
                    sign_mode=SignMode.NONE,
                    backend="term_dsl",
                    term_type=term_type,
                )
                continue

            term = terms[term_idx]
            result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)

            # For swapped pairs, the term might need sign adjustment
            # depending on symmetry properties
            value = result.value

            values[(ell1, ell2)] = S12CanonicalValue(
                ell1=ell1,
                ell2=ell2,
                value=value,
                factorial_mode=FactorialMode.COEFFICIENT,  # term_dsl raw coefficient
                sign_mode=SignMode.NONE,                   # no sign adjustment
                backend="term_dsl",
                term_type=term_type,
            )

    return S12FullMatrix(
        term_type=term_type,
        backend="term_dsl",
        R=R,
        theta=theta,
        values=values,
    )


def print_comparison_table(
    matrix1: S12FullMatrix,
    matrix2: S12FullMatrix,
    normalize_to_coefficient: bool = True,
    remove_sign: bool = True,
):
    """Print side-by-side comparison of two matrices."""

    print(f"\n{'=' * 100}")
    print(f"COMPARISON: {matrix1.backend} vs {matrix2.backend}")
    print(f"Term: {matrix1.term_type}, R={matrix1.R}")
    print(f"{'=' * 100}")

    if normalize_to_coefficient:
        print("(Values normalized to coefficient mode: dividing by ℓ₁!×ℓ₂!)")
    if remove_sign:
        print("(Off-diagonal sign convention removed)")

    print(f"\n{'-' * 100}")
    print(f"{'Pair':<8} {'Backend1':<14} {'Backend2':<14} {'Ratio':<12} {'Match?':<8} Notes")
    print(f"{'-' * 100}")

    for ell1 in [1, 2, 3]:
        for ell2 in [1, 2, 3]:
            v1 = matrix1.get(ell1, ell2)
            v2 = matrix2.get(ell1, ell2)

            if v1 is None or v2 is None:
                print(f"({ell1},{ell2})  N/A")
                continue

            # Normalize
            if normalize_to_coefficient:
                v1 = v1.to_coefficient_mode()
                v2 = v2.to_coefficient_mode()

            if remove_sign:
                v1 = v1.with_sign_mode(SignMode.NONE)
                v2 = v2.with_sign_mode(SignMode.NONE)

            val1 = v1.value
            val2 = v2.value

            # Compute ratio and match
            if abs(val2) > 1e-15:
                ratio = val1 / val2
            else:
                ratio = float('inf') if val1 != 0 else 1.0

            rel_err = abs(val1 - val2) / max(abs(val1), abs(val2), 1e-15)
            match = "✓" if rel_err < 1e-4 else "✗"

            # Notes
            notes = []
            if abs(ratio - 1.0) < 0.01:
                notes.append("same")
            elif abs(ratio + 1.0) < 0.01:
                notes.append("SIGN FLIP")
            elif abs(ratio) > 1e-15:
                expected_factorial = math.factorial(ell1) * math.factorial(ell2)
                if abs(ratio - expected_factorial) / expected_factorial < 0.1:
                    notes.append(f"factorial ({ell1}!×{ell2}!={expected_factorial})")
                elif abs(ratio - 1/expected_factorial) * expected_factorial < 0.1:
                    notes.append(f"1/factorial")

            notes_str = ", ".join(notes) if notes else ""

            print(f"({ell1},{ell2})  {val1:>12.6e}  {val2:>12.6e}  {ratio:>10.4f}  {match:>6}  {notes_str}")

    print(f"{'-' * 100}")


def print_raw_matrices(
    matrix1: S12FullMatrix,
    matrix2: S12FullMatrix,
):
    """Print raw values from both matrices."""

    print(f"\n{'=' * 80}")
    print(f"RAW VALUES: {matrix1.term_type}")
    print(f"{'=' * 80}")

    print(f"\n{matrix1.backend} (factorial_mode={matrix1.values[(1,1)].factorial_mode.value}, "
          f"sign_mode={matrix1.values[(1,1)].sign_mode.value}):")
    print("     ℓ₂=1          ℓ₂=2          ℓ₂=3")
    for ell1 in [1, 2, 3]:
        print(f"ℓ₁={ell1}", end="")
        for ell2 in [1, 2, 3]:
            v = matrix1.get(ell1, ell2)
            print(f"  {v.value:>12.6e}" if v else "  N/A", end="")
        print()

    print(f"\n{matrix2.backend} (factorial_mode={matrix2.values[(1,1)].factorial_mode.value}, "
          f"sign_mode={matrix2.values[(1,1)].sign_mode.value}):")
    print("     ℓ₂=1          ℓ₂=2          ℓ₂=3")
    for ell1 in [1, 2, 3]:
        print(f"ℓ₁={ell1}", end="")
        for ell2 in [1, 2, 3]:
            v = matrix2.get(ell1, ell2)
            print(f"  {v.value:>12.6e}" if v else "  N/A", end="")
        print()


def run_benchmark_comparison(benchmark_name: str, R: float, theta: float, polynomials: dict):
    """Run full comparison for one benchmark."""

    print(f"\n{'#' * 100}")
    print(f"# BENCHMARK: {benchmark_name} (R={R})")
    print(f"{'#' * 100}")

    # Compute matrices for I1
    print("\nComputing I1 matrices...")
    I1_unified = compute_unified_general_matrix(R, theta, polynomials, "I1", n_quad=60)
    I1_dsl = compute_term_dsl_matrix(R, theta, polynomials, "I1", n_quad=60)

    print_raw_matrices(I1_unified, I1_dsl)
    print_comparison_table(I1_unified, I1_dsl, normalize_to_coefficient=True, remove_sign=True)

    # Compute matrices for I2
    print("\nComputing I2 matrices...")
    I2_unified = compute_unified_general_matrix(R, theta, polynomials, "I2", n_quad=60)
    I2_dsl = compute_term_dsl_matrix(R, theta, polynomials, "I2", n_quad=60)

    print_raw_matrices(I2_unified, I2_dsl)
    print_comparison_table(I2_unified, I2_dsl, normalize_to_coefficient=True, remove_sign=True)

    return {
        "I1_unified": I1_unified,
        "I1_dsl": I1_dsl,
        "I2_unified": I2_unified,
        "I2_dsl": I2_dsl,
    }


def main():
    """Main entry point."""

    print("=" * 100)
    print("PHASE 28: S12 BACKEND MATRIX COMPARISON")
    print("=" * 100)
    print("""
Purpose: Determine whether unified_general and term_dsl backends compute
the same mathematical object under different conventions.

Key questions:
1. Are differences due to factorial normalization (ℓ₁!×ℓ₂!)?
2. Are differences due to off-diagonal sign convention (-1)^{ℓ₁+ℓ₂}?
3. Are there fundamental mathematical differences?
""")

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials()
    kappa_polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    kappa_star_polys = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    theta = 4 / 7

    # Run comparison for kappa benchmark
    kappa_results = run_benchmark_comparison("KAPPA", R=1.3036, theta=theta, polynomials=kappa_polys)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("""
If all pairs show "Match: ✓" after normalization, the backends are equivalent.
If some pairs show "SIGN FLIP", check if applying (-1)^{ℓ₁+ℓ₂} fixes it.
If ratios are close to factorial values, factorial convention differs.
If fundamental differences remain, backends compute different objects.
""")


if __name__ == "__main__":
    main()
