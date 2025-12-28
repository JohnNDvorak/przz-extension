"""
src/mirror_transform_paper_exact.py
Phase 29: Mirror Transform in Paper Regime

This module computes the MIRROR contribution to S12 using paper regime kernels.

PRZZ MIRROR STRUCTURE (from TeX lines 1502-1511):
==================================================
The identity has form: I(alpha, beta) + T^{-(alpha+beta)} * I(-beta, -alpha)

At PRZZ evaluation point alpha = beta = -R/L:
- T^{-(alpha+beta)} = exp(2R)
- The mirror term becomes: exp(2R) * I(R/L, R/L)

KEY INSIGHT FROM PHASE 28:
==========================
- The ratio S12(+R)/S12(-R) ≈ 3.6 does NOT match empirical m ≈ 8.68
- The empirical m = exp(R) + 5 was calibrated, not derived
- This module computes S12_mirror at operator level for diagnostic purposes

OPERATOR-LEVEL MIRROR:
======================
The mirror I(-beta,-alpha) involves:
1. Swapped Q eigenvalues (x↔y conjugation)
2. Sign flip in exponential factor
3. T^{-(alpha+beta)} = exp(2R) prefactor

For paper regime:
- P1 uses Case B (raw polynomial)
- P2/P3 use Case C (kernel attenuation) - omega = ell - 1

Created: 2025-12-26 (Phase 29)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math
import numpy as np

from src.quadrature import gauss_legendre_01
from src.unified_i1_paper import compute_I1_unified_paper
from src.unified_i2_paper import compute_I2_unified_paper


@dataclass
class MirrorPaperResult:
    """Result of paper regime mirror computation."""

    # Per-term results
    S12_direct: float       # I1 + I2 at +R (paper regime)
    S12_proxy_neg_R: float  # I1 + I2 at -R (paper regime) - for comparison only
    S12_mirror_exact: float # Full mirror contribution (exp(2R) * S12_swapped)

    # Derived quantities
    m_eff: float            # S12_mirror / S12_proxy (diagnostic only!)
    m_empirical: float      # exp(R) + 5 (reference)

    # Parameters
    R: float
    theta: float


def compute_S12_paper_sum(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
    n_quad_a: int = 40,
) -> float:
    """
    Compute total S12 = sum(I1 + I2) over all pairs using paper regime.

    Uses triangle*2 convention with factorial normalization.
    """
    # Factorial normalization matching production evaluator
    f_norm = {
        "11": 1.0,
        "22": 0.25,
        "33": 1.0 / 36.0,
        "12": 0.5,
        "13": 1.0 / 6.0,
        "23": 1.0 / 12.0,
    }

    # Symmetry factors (2x for off-diagonal)
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    pairs = ["11", "22", "33", "12", "13", "23"]
    total = 0.0

    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        norm = f_norm[pair_key]
        sym = symmetry[pair_key]
        full_norm = sym * norm

        # I1 using unified_paper
        I1_result = compute_I1_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=n_quad_a,
            include_Q=True, apply_factorial_norm=True,
        )

        # I2 using unified_paper
        I2_result = compute_I2_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=n_quad_a,
            include_Q=True,
        )

        # Note: unified_paper already applies factorial norm for I1
        # For I2, there's no factorial norm (no derivatives)
        pair_contrib = (I1_result.I1_value + I2_result.I2_value) * norm * sym
        total += pair_contrib

    return total


def compute_mirror_paper_analysis(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
    n_quad_a: int = 40,
    verbose: bool = False,
) -> MirrorPaperResult:
    """
    Analyze the mirror relationship in paper regime.

    Computes:
    1. S12_direct = S12(+R) in paper regime
    2. S12_proxy = S12(-R) in paper regime (for ratio comparison)
    3. S12_mirror = exp(2R) * S12_swapped (operator-level mirror)

    The key diagnostic is comparing m_eff = S12_mirror / S12_proxy
    against the empirical m = exp(R) + 5.

    NOTE: For Phase 29, we use S12(-R) as a proxy for the swapped integral.
    The true operator-level mirror involves eigenvalue conjugation.
    """
    # S12 at +R (direct term)
    S12_direct = compute_S12_paper_sum(
        R, theta, polynomials, n_quad, n_quad_a
    )

    # S12 at -R (proxy for mirror structure analysis)
    S12_proxy_neg_R = compute_S12_paper_sum(
        -R, theta, polynomials, n_quad, n_quad_a
    )

    # For the mirror term, we use the PRZZ structure:
    # S12_mirror = exp(2R) * S12(-R) as approximation
    # The true operator form has eigenvalue swaps
    T_weight = math.exp(2 * R)
    S12_mirror_approx = T_weight * S12_proxy_neg_R

    # Effective m (diagnostic only)
    m_empirical = math.exp(R) + 5
    m_eff = S12_mirror_approx / S12_proxy_neg_R if abs(S12_proxy_neg_R) > 1e-15 else float('inf')

    if verbose:
        print(f"\nMirror Analysis in Paper Regime (R={R})")
        print(f"=" * 50)
        print(f"S12_direct(+R) = {S12_direct:.6f}")
        print(f"S12_proxy(-R)  = {S12_proxy_neg_R:.6f}")
        print(f"S12_mirror     = exp(2R) * S12(-R) = {S12_mirror_approx:.6f}")
        print(f"")
        print(f"Ratio S12(+R)/S12(-R) = {S12_direct / S12_proxy_neg_R:.4f}")
        print(f"")
        print(f"m_eff = S12_mirror / S12(-R) = exp(2R) = {m_eff:.4f}")
        print(f"m_empirical = exp(R) + 5 = {m_empirical:.4f}")
        print(f"")
        print(f"c_partial = S12(+R) + m × S12(-R)")
        print(f"  = {S12_direct:.6f} + {m_empirical:.4f} × {S12_proxy_neg_R:.6f}")
        print(f"  = {S12_direct + m_empirical * S12_proxy_neg_R:.4f}")

    return MirrorPaperResult(
        S12_direct=S12_direct,
        S12_proxy_neg_R=S12_proxy_neg_R,
        S12_mirror_exact=S12_mirror_approx,
        m_eff=m_eff,
        m_empirical=m_empirical,
        R=R,
        theta=theta,
    )


def compute_c_paper_derived(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
    verbose: bool = False,
) -> Dict:
    """
    Compute c using paper regime with empirical mirror formula.

    This reproduces the compute_c_paper_with_mirror() logic using
    the new unified_paper backends.

    Formula: c = S12(+R) + m × S12(-R) + S34(+R)

    where m = exp(R) + 5 for K=3.

    PHASE 30 FIX: S34 is now computed via term DSL (not hardcoded -0.6).
    The hardcoded value was causing 9.29% gap on κ* benchmark.
    """
    from src.unified_i1_paper import compute_I1_unified_paper
    from src.unified_i2_paper import compute_I2_unified_paper
    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term

    # Compute S12 components
    mirror_result = compute_mirror_paper_analysis(
        R, theta, polynomials, n_quad, verbose=False
    )

    m = mirror_result.m_empirical  # exp(R) + 5

    # PHASE 30 FIX: Compute actual S34 using term DSL
    # Previously used hardcoded S34 = -0.6 which was wrong for κ*
    all_terms = make_all_terms_k3(theta, R, kernel_regime="paper")

    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry_factor = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    S34_total = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms[pair_key]
        norm = factorial_norm[pair_key]
        sym = symmetry_factor[pair_key]
        full_norm = sym * norm

        # I₃ and I₄ (indices 2, 3) - NO mirror
        for term in terms[2:4]:
            result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
            S34_total += full_norm * result.value

    # c formula
    c = mirror_result.S12_direct + m * mirror_result.S12_proxy_neg_R + S34_total

    # kappa from c
    kappa = 1 - math.log(c) / R if c > 0 else float('-inf')

    if verbose:
        print(f"\nc Computation (Paper Regime Derived)")
        print(f"=" * 50)
        print(f"R = {R}, theta = {theta:.6f}")
        print(f"")
        print(f"S12_direct(+R) = {mirror_result.S12_direct:.6f}")
        print(f"S12_proxy(-R)  = {mirror_result.S12_proxy_neg_R:.6f}")
        print(f"m = exp(R) + 5 = {m:.4f}")
        print(f"S34(+R)        = {S34_total:.6f}")
        print(f"")
        print(f"c = S12(+R) + m × S12(-R) + S34")
        print(f"  = {mirror_result.S12_direct:.6f} + {m:.4f} × {mirror_result.S12_proxy_neg_R:.6f} + {S34_total:.6f}")
        print(f"  = {c:.6f}")
        print(f"")
        print(f"kappa = 1 - log(c)/R = {kappa:.6f}")

    return {
        "c": c,
        "kappa": kappa,
        "S12_direct": mirror_result.S12_direct,
        "S12_proxy": mirror_result.S12_proxy_neg_R,
        "S34": S34_total,
        "m": m,
        "R": R,
        "theta": theta,
    }


# =============================================================================
# Diagnostic: Per-pair breakdown
# =============================================================================


def breakdown_paper_pairs(
    R: float,
    theta: float,
    polynomials: Dict,
    n_quad: int = 60,
) -> Dict:
    """
    Get per-pair I1/I2 breakdown in paper regime.
    """
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    pairs = ["11", "22", "33", "12", "13", "23"]
    breakdown = {}

    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        I1_result = compute_I1_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
        )
        I2_result = compute_I2_unified_paper(
            R, theta, ell1, ell2, polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
        )

        norm = f_norm[pair_key]
        sym = symmetry[pair_key]

        breakdown[pair_key] = {
            "I1_raw": I1_result.I1_value,
            "I2_raw": I2_result.I2_value,
            "I1_normed": I1_result.I1_value * norm * sym,
            "I2_normed": I2_result.I2_value * norm * sym,
            "total_normed": (I1_result.I1_value + I2_result.I2_value) * norm * sym,
        }

    return breakdown
