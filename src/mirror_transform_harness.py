"""
src/mirror_transform_harness.py
Phase 10.1: Mirror Transform Diagnostic Harness

This module provides diagnostic infrastructure for comparing different mirror
computation approaches within the evaluator's semantic framework.

PURPOSE:
========
Enable dual-route comparison between:
1. S12_direct_pair: Per-pair I1+I2 at +R (standard)
2. S12_mirror_candidate_pair: Per-pair derived mirror (operator approach)
3. S12_mirror_basis_pair: Per-pair DSL -R basis (empirical)
4. S34_pair: Per-pair I3+I4 (no mirror, for completeness)

This harness ensures all comparisons use the SAME evaluator semantics,
avoiding the "semantic difference" confusion from Phase 9.

USAGE:
======
    harness = MirrorTransformHarness(theta, R, n, polynomials)
    result = harness.run()

    # Compare approaches
    print(f"S12 direct: {result.S12_direct_total}")
    print(f"S12 operator mirror: {result.S12_operator_mirror_total}")
    print(f"S12 empirical mirror: {result.S12_empirical_mirror_total}")
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MirrorHarnessResult:
    """
    Result from running the mirror transform harness.

    All values are computed using the evaluator's semantics,
    ensuring apples-to-apples comparison.
    """
    # Per-pair values
    S12_direct_pair: Dict[str, float] = field(default_factory=dict)
    """I1(+R) + I2(+R) per pair."""

    S12_operator_mirror_pair: Dict[str, float] = field(default_factory=dict)
    """Derived mirror via operator approach, per pair."""

    S12_basis_pair: Dict[str, float] = field(default_factory=dict)
    """I1(-R) + I2(-R) per pair (DSL minus basis)."""

    S34_pair: Dict[str, float] = field(default_factory=dict)
    """I3(+R) + I4(+R) per pair (no mirror)."""

    # Totals
    S12_direct_total: float = 0.0
    S12_operator_mirror_total: float = 0.0
    S12_basis_total: float = 0.0
    S34_total: float = 0.0

    # Derived quantities
    m1_implied: float = 0.0
    """Implied m₁ from operator mirror / basis."""

    c_direct_only: float = 0.0
    """c computed with direct S12 only (no mirror)."""

    c_with_empirical: float = 0.0
    """c computed with empirical m₁ × S12_basis."""

    c_with_operator: float = 0.0
    """c computed with operator-derived mirror."""

    # Parameters
    R: float = 0.0
    theta: float = 0.0
    n: int = 0


class MirrorTransformHarness:
    """
    Diagnostic harness for comparing mirror computation approaches.

    This harness uses the evaluator's exact term evaluation to compute
    S12 components, enabling fair comparison between different mirror
    strategies.
    """

    # Factorial normalization (matches evaluate.py)
    FACTORIAL_NORM = {
        "11": 1.0,
        "22": 0.25,
        "33": 1.0 / 36.0,
        "12": 0.5,
        "13": 1.0 / 6.0,
        "23": 1.0 / 12.0,
    }

    # Symmetry factors (triangle×2)
    SYMMETRY = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    PAIRS = ["11", "22", "33", "12", "13", "23"]

    def __init__(
        self,
        theta: float,
        R: float,
        n: int,
        polynomials: Dict,
        K: int = 3,
        use_t_dependent: bool = False,
        use_t_flip_exp: bool = False
    ):
        """
        Initialize the harness.

        Args:
            theta: PRZZ θ parameter (4/7)
            R: PRZZ R parameter
            n: Quadrature points
            polynomials: Dict with P1, P2, P3, Q
            K: Number of mollifier pieces
            use_t_dependent: If True, use Phase 12 complement eigenvalues.
                            If False (default), use Phase 10 static eigenvalues.
            use_t_flip_exp: If True, use Phase 13 t-flip consistent exp coefficients.
                           If False (default), use static exp coefficients.
        """
        self.theta = theta
        self.R = R
        self.n = n
        self.polynomials = polynomials
        self.K = K
        self.use_t_dependent = use_t_dependent
        self.use_t_flip_exp = use_t_flip_exp

    def run(self, verbose: bool = False) -> MirrorHarnessResult:
        """
        Run the full harness computation.

        Returns:
            MirrorHarnessResult with all per-pair and total values
        """
        result = MirrorHarnessResult(R=self.R, theta=self.theta, n=self.n)

        # Compute per-pair values
        for pair_key in self.PAIRS:
            ell1 = int(pair_key[0])
            ell2 = int(pair_key[1])
            norm = self.FACTORIAL_NORM[pair_key] * self.SYMMETRY[pair_key]

            # S12 direct (+R)
            direct = self._compute_S12_direct_pair(ell1, ell2)
            result.S12_direct_pair[pair_key] = norm * direct
            result.S12_direct_total += norm * direct

            # S12 operator mirror (swap approach)
            operator_mirror = self._compute_S12_operator_mirror_pair(ell1, ell2)
            result.S12_operator_mirror_pair[pair_key] = norm * operator_mirror
            result.S12_operator_mirror_total += norm * operator_mirror

            # S12 basis (-R)
            basis = self._compute_S12_basis_pair(ell1, ell2)
            result.S12_basis_pair[pair_key] = norm * basis
            result.S12_basis_total += norm * basis

            # S34 (no mirror)
            s34 = self._compute_S34_pair(ell1, ell2)
            result.S34_pair[pair_key] = norm * s34
            result.S34_total += norm * s34

            if verbose:
                print(f"Pair {pair_key}: direct={norm*direct:.6f}, "
                      f"op_mirror={norm*operator_mirror:.6f}, "
                      f"basis={norm*basis:.6f}, s34={norm*s34:.6f}")

        # Compute derived quantities
        if abs(result.S12_basis_total) > 1e-15:
            result.m1_implied = result.S12_operator_mirror_total / result.S12_basis_total
        else:
            result.m1_implied = float('inf')

        # c computations
        result.c_direct_only = result.S12_direct_total + result.S34_total

        # Empirical: c = S12(+R) + m₁ × S12(-R) + S34
        m1_empirical = np.exp(self.R) + 5  # K=3 formula
        result.c_with_empirical = (
            result.S12_direct_total +
            m1_empirical * result.S12_basis_total +
            result.S34_total
        )

        # Operator: c = S12(+R) + S12_operator_mirror + S34
        result.c_with_operator = (
            result.S12_direct_total +
            result.S12_operator_mirror_total +
            result.S34_total
        )

        if verbose:
            print(f"\n=== Harness Summary ===")
            print(f"S12_direct_total: {result.S12_direct_total:.6f}")
            print(f"S12_operator_mirror_total: {result.S12_operator_mirror_total:.6f}")
            print(f"S12_basis_total: {result.S12_basis_total:.6f}")
            print(f"S34_total: {result.S34_total:.6f}")
            print(f"m1_implied: {result.m1_implied:.4f}")
            print(f"m1_empirical: {m1_empirical:.4f}")
            print(f"c_direct_only: {result.c_direct_only:.6f}")
            print(f"c_with_empirical: {result.c_with_empirical:.6f}")
            print(f"c_with_operator: {result.c_with_operator:.6f}")

        return result

    def _compute_S12_direct_pair(self, ell1: int, ell2: int) -> float:
        """Compute I1(+R) + I2(+R) for a pair."""
        from src.mirror_exact import compute_I1_standard, _compute_I2_with_shifted_Q

        I1 = compute_I1_standard(
            theta=self.theta, R=self.R, n=self.n,
            polynomials=self.polynomials, ell1=ell1, ell2=ell2
        )

        I2 = _compute_I2_with_shifted_Q(
            theta=self.theta, R=self.R, n=self.n,
            polynomials=self.polynomials, ell1=ell1, ell2=ell2, shift=0.0
        )

        return I1 + I2

    def _compute_S12_operator_mirror_pair(self, ell1: int, ell2: int) -> float:
        """Compute derived mirror via operator approach for a pair."""
        from src.mirror_operator_exact import (
            compute_I1_mirror_operator_exact,
            compute_I2_mirror_operator_exact
        )

        I1_result = compute_I1_mirror_operator_exact(
            theta=self.theta, R=self.R, n=self.n,
            polynomials=self.polynomials, ell1=ell1, ell2=ell2,
            use_t_dependent=self.use_t_dependent,
            use_t_flip_exp=self.use_t_flip_exp
        )

        I2_result = compute_I2_mirror_operator_exact(
            theta=self.theta, R=self.R, n=self.n,
            polynomials=self.polynomials, ell1=ell1, ell2=ell2,
            use_t_dependent=self.use_t_dependent,
            use_t_flip_exp=self.use_t_flip_exp
        )

        return I1_result.value + I2_result.value

    def _compute_S12_basis_pair(self, ell1: int, ell2: int) -> float:
        """Compute I1(-R) + I2(-R) for a pair (DSL minus basis)."""
        from src.mirror_exact import compute_I1_at_minus_R, _compute_I2_with_shifted_Q

        I1 = compute_I1_at_minus_R(
            theta=self.theta, R=self.R, n=self.n,
            polynomials=self.polynomials, ell1=ell1, ell2=ell2,
            use_shifted_Q=False
        )

        I2 = _compute_I2_with_shifted_Q(
            theta=self.theta, R=-self.R, n=self.n,
            polynomials=self.polynomials, ell1=ell1, ell2=ell2, shift=0.0
        )

        return I1 + I2

    def _compute_S34_pair(self, ell1: int, ell2: int) -> float:
        """
        Compute I3(+R) + I4(+R) for a pair (no mirror).

        Uses the same DSL-based evaluation path as the canonical evaluator
        to ensure apples-to-apples comparison.

        I3/I4 have single-derivative structure:
        - I3: d/dx only (y=0)
        - I4: d/dy only (x=0)

        Per TRUTH_SPEC Section 10: I3/I4 are NEVER mirrored.
        """
        from src.terms_k3_d1 import make_all_terms_k3
        from src.evaluate import evaluate_term

        # Build terms with paper regime (same as canonical evaluator)
        pair_key = f"{ell1}{ell2}"
        all_terms = make_all_terms_k3(self.theta, self.R, kernel_regime="paper")

        if pair_key not in all_terms:
            return 0.0

        terms = all_terms[pair_key]

        # I₃ is at index 2, I₄ is at index 3
        total = 0.0
        for term in terms[2:4]:  # I₃, I₄
            result = evaluate_term(
                term, self.polynomials, self.n,
                R=self.R, theta=self.theta, n_quad_a=40
            )
            total += result.value

        return total


def run_harness_comparison(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    verbose: bool = True
) -> MirrorHarnessResult:
    """
    Convenience function to run harness comparison.

    Args:
        theta: PRZZ θ parameter
        R: PRZZ R parameter
        n: Quadrature points
        polynomials: Dict with polynomial objects
        verbose: Print diagnostics

    Returns:
        MirrorHarnessResult
    """
    harness = MirrorTransformHarness(theta, R, n, polynomials)
    return harness.run(verbose=verbose)


def compare_approaches_at_benchmarks(
    polynomials_kappa: Dict,
    polynomials_kappa_star: Dict,
    n: int = 40,
    verbose: bool = True
) -> Tuple[MirrorHarnessResult, MirrorHarnessResult]:
    """
    Compare approaches at both PRZZ benchmarks.

    Args:
        polynomials_kappa: Polynomials for κ benchmark
        polynomials_kappa_star: Polynomials for κ* benchmark
        n: Quadrature points
        verbose: Print diagnostics

    Returns:
        Tuple of (kappa_result, kappa_star_result)
    """
    theta = 4.0 / 7.0
    R_kappa = 1.3036
    R_kappa_star = 1.1167

    if verbose:
        print("="*60)
        print("κ BENCHMARK (R=1.3036)")
        print("="*60)

    kappa_result = run_harness_comparison(
        theta, R_kappa, n, polynomials_kappa, verbose=verbose
    )

    if verbose:
        print("\n" + "="*60)
        print("κ* BENCHMARK (R=1.1167)")
        print("="*60)

    kappa_star_result = run_harness_comparison(
        theta, R_kappa_star, n, polynomials_kappa_star, verbose=verbose
    )

    if verbose:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"κ  c_with_empirical: {kappa_result.c_with_empirical:.6f}")
        print(f"κ* c_with_empirical: {kappa_star_result.c_with_empirical:.6f}")
        print(f"κ  c_with_operator:  {kappa_result.c_with_operator:.6f}")
        print(f"κ* c_with_operator:  {kappa_star_result.c_with_operator:.6f}")

    return kappa_result, kappa_star_result
