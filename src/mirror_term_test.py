"""
src/mirror_term_test.py
Test different mirror term combinations to identify the correct PRZZ formula.

The PRZZ mirror combination (TeX lines 1502-1504):
    H(α,β) = (N^{αx+βy} - T^{-α-β}·N^{-βx-αy}) / (α+β)

At evaluation point α=β=-R, this involves both exp(+R×...) and exp(-R×...) terms.

This script tests different combinations to find which matches PRZZ targets:
- κ benchmark: R=1.3036, c_target=2.137
- κ* benchmark: R=1.1167, c_target=1.939
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, Tuple
from dataclasses import dataclass

from src.terms_k3_d1 import make_all_terms_k3
from src.term_dsl import Term, ExpFactor
from src.evaluate import evaluate_terms
from src.polynomials import load_przz_polynomials


@dataclass
class MirrorTestResult:
    """Result of testing a mirror combination."""
    name: str
    c_kappa: float
    c_kappa_star: float
    target_kappa: float
    target_kappa_star: float
    gap_kappa: float
    gap_kappa_star: float
    gap_ratio: float  # gap_kappa_star / gap_kappa


def evaluate_with_exp_sign(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    exp_sign: float = 1.0
) -> float:
    """
    Evaluate c with modified exp factor sign.

    Args:
        theta: θ parameter
        R: R parameter
        n: Quadrature points
        polynomials: Polynomial dict
        exp_sign: +1.0 for normal, -1.0 for mirror

    Returns:
        Total c value
    """
    # Get all terms
    all_terms = make_all_terms_k3(theta, R)

    # Factorial normalization factors
    factorial_norm = {
        "11": 1.0, "22": 1.0/4, "33": 1.0/36,
        "12": 1.0/2, "13": 1.0/6, "23": 1.0/12
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    total = 0.0

    for pair_key, terms in all_terms.items():
        # Modify exp factors if exp_sign != 1.0
        if exp_sign != 1.0:
            modified_terms = []
            for term in terms:
                # Create new term with modified exp factors
                new_exp_factors = []
                for ef in term.exp_factors:
                    # Scale becomes exp_sign * original_scale
                    new_exp_factors.append(ExpFactor(exp_sign * ef.scale, ef.argument))

                modified_terms.append(Term(
                    name=term.name,
                    pair=term.pair,
                    przz_reference=term.przz_reference,
                    vars=term.vars,
                    deriv_orders=term.deriv_orders,
                    domain=term.domain,
                    numeric_prefactor=term.numeric_prefactor,
                    algebraic_prefactor=term.algebraic_prefactor,
                    poly_prefactors=term.poly_prefactors,
                    poly_factors=term.poly_factors,
                    exp_factors=new_exp_factors
                ))
            terms_to_eval = modified_terms
        else:
            terms_to_eval = terms

        pair_result = evaluate_terms(terms_to_eval, polynomials, n, return_breakdown=False, R=R, theta=theta)

        norm = factorial_norm[pair_key] * symmetry[pair_key]
        total += norm * pair_result.total

    return total


def test_mirror_combinations(
    n: int = 60,
    verbose: bool = True
) -> Dict[str, MirrorTestResult]:
    """
    Test various mirror term combinations.

    Combinations to test:
    1. Direct only (current): I_direct
    2. Mirror only: I_mirror (exp_sign=-1)
    3. Sum: I_direct + I_mirror
    4. Difference normalized: (I_direct - I_mirror) / (-2R)
    5. Average: (I_direct + I_mirror) / 2
    6. Sinh-like: (I_direct - I_mirror) / 2
    7. Weighted sum: I_direct + c_factor * I_mirror (various c_factors)
    """
    theta = 4.0 / 7.0

    # Load both benchmark polynomial sets
    from src.polynomials import load_przz_polynomials_kappa_star

    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    # Target values
    R_kappa = 1.3036
    R_kappa_star = 1.1167
    c_target_kappa = 2.13745440613217263636
    c_target_kappa_star = math.exp(R_kappa_star * (1 - 0.407511457))  # 1.9385...

    results = {}

    if verbose:
        print("\n" + "=" * 80)
        print("MIRROR TERM COMBINATION TEST")
        print("=" * 80)
        print(f"\nTargets:")
        print(f"  κ benchmark (R={R_kappa}): c = {c_target_kappa:.6f}")
        print(f"  κ* benchmark (R={R_kappa_star}): c = {c_target_kappa_star:.6f}")
        print(f"  Target ratio: {c_target_kappa / c_target_kappa_star:.4f}")

    # Compute direct and mirror for both benchmarks
    I_direct_k = evaluate_with_exp_sign(theta, R_kappa, n, polys_kappa, exp_sign=1.0)
    I_mirror_k = evaluate_with_exp_sign(theta, R_kappa, n, polys_kappa, exp_sign=-1.0)

    I_direct_ks = evaluate_with_exp_sign(theta, R_kappa_star, n, polys_kappa_star, exp_sign=1.0)
    I_mirror_ks = evaluate_with_exp_sign(theta, R_kappa_star, n, polys_kappa_star, exp_sign=-1.0)

    if verbose:
        print(f"\nBase values (κ benchmark, R={R_kappa}):")
        print(f"  I_direct = {I_direct_k:.6f}")
        print(f"  I_mirror = {I_mirror_k:.6f}")
        print(f"\nBase values (κ* benchmark, R={R_kappa_star}):")
        print(f"  I_direct = {I_direct_ks:.6f}")
        print(f"  I_mirror = {I_mirror_ks:.6f}")

    # Test combinations
    combinations = [
        ("1. Direct only (current)", lambda d, m, R: d),
        ("2. Mirror only", lambda d, m, R: m),
        ("3. Sum (d + m)", lambda d, m, R: d + m),
        ("4. (d - m)/(-2R)", lambda d, m, R: (d - m) / (-2 * R)),
        ("5. Average (d + m)/2", lambda d, m, R: (d + m) / 2),
        ("6. Sinh-like (d - m)/2", lambda d, m, R: (d - m) / 2),
        ("7. d + m/2", lambda d, m, R: d + m / 2),
        ("8. (d + m) × (1 + 1/(2R))", lambda d, m, R: (d + m) * (1 + 1/(2*R))),
        ("9. d × (1 + 1/R)", lambda d, m, R: d * (1 + 1/R)),
        ("10. d × exp(R)/cosh(R)", lambda d, m, R: d * math.exp(R) / math.cosh(R)),
    ]

    if verbose:
        print("\n" + "-" * 80)
        print(f"{'Combination':<30} | {'c(κ)':>10} | {'c(κ*)':>10} | {'Gap κ':>8} | {'Gap κ*':>8} | {'Ratio':>8}")
        print("-" * 80)

    for name, formula in combinations:
        c_k = formula(I_direct_k, I_mirror_k, R_kappa)
        c_ks = formula(I_direct_ks, I_mirror_ks, R_kappa_star)

        gap_k = (c_k - c_target_kappa) / c_target_kappa * 100
        gap_ks = (c_ks - c_target_kappa_star) / c_target_kappa_star * 100

        if abs(gap_k) > 1e-6:
            gap_ratio = gap_ks / gap_k
        else:
            gap_ratio = float('inf')

        results[name] = MirrorTestResult(
            name=name,
            c_kappa=c_k,
            c_kappa_star=c_ks,
            target_kappa=c_target_kappa,
            target_kappa_star=c_target_kappa_star,
            gap_kappa=gap_k,
            gap_kappa_star=gap_ks,
            gap_ratio=gap_ratio
        )

        if verbose:
            print(f"{name:<30} | {c_k:>10.4f} | {c_ks:>10.4f} | {gap_k:>+7.1f}% | {gap_ks:>+7.1f}% | {gap_ratio:>8.2f}")

    # Find best combination (smallest max absolute gap)
    best = min(results.values(), key=lambda r: max(abs(r.gap_kappa), abs(r.gap_kappa_star)))

    if verbose:
        print("-" * 80)
        print(f"\nBest combination: {best.name}")
        print(f"  Gap κ:  {best.gap_kappa:+.2f}%")
        print(f"  Gap κ*: {best.gap_kappa_star:+.2f}%")

        # Check two-benchmark gate
        if abs(best.gap_ratio - 1.0) < 0.2:  # Within 20% of 1:1 ratio
            print(f"  Two-benchmark gate: PASS (ratio {best.gap_ratio:.2f})")
        else:
            print(f"  Two-benchmark gate: FAIL (ratio {best.gap_ratio:.2f}, need ~1.0)")

    return results


def explore_weighted_combinations(
    n: int = 60,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Explore weighted combinations of direct and mirror terms.

    Returns the optimal weight for each benchmark.
    """
    theta = 4.0 / 7.0

    # Load polynomials
    from src.polynomials import load_przz_polynomials_kappa_star

    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    # Parameters
    R_kappa = 1.3036
    R_kappa_star = 1.1167
    c_target_kappa = 2.13745440613217263636
    c_target_kappa_star = math.exp(R_kappa_star * (1 - 0.407511457))

    # Compute base values
    I_direct_k = evaluate_with_exp_sign(theta, R_kappa, n, polys_kappa, exp_sign=1.0)
    I_mirror_k = evaluate_with_exp_sign(theta, R_kappa, n, polys_kappa, exp_sign=-1.0)

    I_direct_ks = evaluate_with_exp_sign(theta, R_kappa_star, n, polys_kappa_star, exp_sign=1.0)
    I_mirror_ks = evaluate_with_exp_sign(theta, R_kappa_star, n, polys_kappa_star, exp_sign=-1.0)

    # For c = I_direct + w × I_mirror, find optimal w
    # c_target = I_direct + w × I_mirror
    # w = (c_target - I_direct) / I_mirror

    w_opt_k = (c_target_kappa - I_direct_k) / I_mirror_k if abs(I_mirror_k) > 1e-10 else 0
    w_opt_ks = (c_target_kappa_star - I_direct_ks) / I_mirror_ks if abs(I_mirror_ks) > 1e-10 else 0

    if verbose:
        print("\n" + "=" * 70)
        print("OPTIMAL WEIGHT EXPLORATION")
        print("=" * 70)

        print(f"\nκ benchmark (R={R_kappa}):")
        print(f"  I_direct = {I_direct_k:.6f}")
        print(f"  I_mirror = {I_mirror_k:.6f}")
        print(f"  c_target = {c_target_kappa:.6f}")
        print(f"  Optimal w = {w_opt_k:.6f}")
        print(f"  Verification: I_d + w×I_m = {I_direct_k + w_opt_k * I_mirror_k:.6f}")

        print(f"\nκ* benchmark (R={R_kappa_star}):")
        print(f"  I_direct = {I_direct_ks:.6f}")
        print(f"  I_mirror = {I_mirror_ks:.6f}")
        print(f"  c_target = {c_target_kappa_star:.6f}")
        print(f"  Optimal w = {w_opt_ks:.6f}")
        print(f"  Verification: I_d + w×I_m = {I_direct_ks + w_opt_ks * I_mirror_ks:.6f}")

        print(f"\n--- Analysis ---")
        print(f"  Weight ratio (κ*/κ): {w_opt_ks / w_opt_k if abs(w_opt_k) > 1e-10 else float('inf'):.4f}")

        # Check if weight could be R-dependent
        print(f"\n--- Testing R-dependent weight hypotheses ---")
        hypotheses = [
            ("w = 1", lambda R: 1.0),
            ("w = -1", lambda R: -1.0),
            ("w = R", lambda R: R),
            ("w = -R", lambda R: -R),
            ("w = 1/R", lambda R: 1/R),
            ("w = -1/R", lambda R: -1/R),
            ("w = 1/(2R)", lambda R: 1/(2*R)),
            ("w = -1/(2R)", lambda R: -1/(2*R)),
            ("w = exp(R)-1", lambda R: math.exp(R) - 1),
            ("w = sinh(R)/R", lambda R: math.sinh(R)/R),
            ("w = (exp(R)-1)/R", lambda R: (math.exp(R)-1)/R),
            ("w = (exp(2R)-1)/(2R)", lambda R: (math.exp(2*R)-1)/(2*R)),
        ]

        print(f"\n  {'Hypothesis':<25} | {'w(κ)':>10} | {'w(κ*)':>10} | {'Gap κ':>8} | {'Gap κ*':>8}")
        print(f"  {'-'*25} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8}")

        for name, w_func in hypotheses:
            w_k = w_func(R_kappa)
            w_ks = w_func(R_kappa_star)

            c_k = I_direct_k + w_k * I_mirror_k
            c_ks = I_direct_ks + w_ks * I_mirror_ks

            gap_k = (c_k - c_target_kappa) / c_target_kappa * 100
            gap_ks = (c_ks - c_target_kappa_star) / c_target_kappa_star * 100

            print(f"  {name:<25} | {w_k:>10.4f} | {w_ks:>10.4f} | {gap_k:>+7.1f}% | {gap_ks:>+7.1f}%")

        print(f"\n  Optimal weights needed:")
        print(f"    κ:  w = {w_opt_k:.6f}")
        print(f"    κ*: w = {w_opt_ks:.6f}")

    return w_opt_k, w_opt_ks


if __name__ == "__main__":
    print("\n" + "█" * 80)
    print("█ MIRROR TERM INVESTIGATION")
    print("█" * 80)

    # Test various combinations
    results = test_mirror_combinations(n=60, verbose=True)

    # Explore weighted combinations
    explore_weighted_combinations(n=60, verbose=True)

    print("\n" + "█" * 80)
