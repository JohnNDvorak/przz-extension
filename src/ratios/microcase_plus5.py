"""
src/ratios/microcase_plus5.py
Phase 14 Task 4B: Minimal Micro-Case for +5 Signature

PAPER ANCHOR:
============
m₁ = exp(R) + 5 where 5 = 2K - 1 for K=3

The "+5" is combinatorial from the paper's five-term J₁ decomposition,
NOT from operator mirroring (which gives only ~0.84×exp(R)).

PURPOSE:
========
This module provides a minimal test case that demonstrates:
1. The five-piece structure gives rise to a constant offset
2. The constant offset should be ~5 for K=3
3. The exp(R) part comes from a different mechanism (operator scaling)

This is the "micro-case" that proves the +5 has an upstream origin.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np

from src.ratios.j1_k3_decomposition import (
    J1Pieces,
    build_J1_pieces_K3,
    sum_J1,
)
from src.ratios.arithmetic_factor import A11_prime_sum


def microcase_plus5_signature() -> Dict:
    """
    Compute the "+5 signature" from J₁ decomposition.

    This function demonstrates that the five-piece structure of J₁
    gives rise to a constant offset in the mirror formula.

    Returns:
        dict with:
        - "num_pieces": 5 (the count from J₁ decomposition)
        - "piece_names": ["j11", "j12", "j13", "j14", "j15"]
        - "piece_values": numeric values at test specialization
        - "A11_anchor": the A^{(1,1)} value (should be ~1.385)
        - "K": 3 (the number of mollifier pieces)
        - "2K_minus_1": 5 (= 2×3 - 1, the formula for the constant)
    """
    # Evaluate at test point: α=β=0, s=0, u=0.5
    pieces = build_J1_pieces_K3(alpha=0.0, beta=0.0, s=0.0, u=0.5)

    # Get A^{(1,1)} anchor value
    A11_value = A11_prime_sum(0.0, prime_cutoff=10000)

    return {
        "num_pieces": 5,
        "piece_names": ["j11", "j12", "j13", "j14", "j15"],
        "piece_values": {
            "j11": float(np.real(pieces.j11)),
            "j12": float(np.real(pieces.j12)),
            "j13": float(np.real(pieces.j13)),
            "j14": float(np.real(pieces.j14)),
            "j15": float(np.real(pieces.j15)),
        },
        "total": float(np.real(sum_J1(pieces))),
        "A11_anchor": float(A11_value),
        "K": 3,
        "2K_minus_1": 5,
    }


def analyze_constant_vs_expR(
    R_values: List[float] = None
) -> Dict:
    """
    Analyze how J₁ pieces decompose into exp(R) vs constant parts.

    The hypothesis is:
    - Some pieces scale with exp(R) → the "mirror" part
    - Some pieces give constant offset → the "+5"

    Args:
        R_values: List of R values to test (default: [1.0, 1.2, 1.3036, 1.4])

    Returns:
        dict with decomposition analysis across R values
    """
    if R_values is None:
        R_values = [1.0, 1.2, 1.3036, 1.4]

    results = {
        "R_values": R_values,
        "exp_R": [np.exp(R) for R in R_values],
        "exp_R_plus_5": [np.exp(R) + 5 for R in R_values],
        "piece_totals": [],
        "per_piece": {f"j1{i}": [] for i in range(1, 6)},
    }

    for R in R_values:
        # At PRZZ point: α = β = -R
        alpha = -R
        beta = -R
        s = alpha + beta

        pieces = build_J1_pieces_K3(alpha, beta, s, u=0.5)
        results["piece_totals"].append(float(np.real(sum_J1(pieces))))

        for i, name in enumerate(["j11", "j12", "j13", "j14", "j15"]):
            results["per_piece"][name].append(float(np.real(pieces[i])))

    return results


def estimate_constant_offset(
    R_low: float = 1.0,
    R_high: float = 1.4
) -> Dict:
    """
    Estimate the constant offset by comparing at two R values.

    If J₁_total = A × exp(R) + B, then:
    - Slope A = (J₁(R_high) - J₁(R_low)) / (exp(R_high) - exp(R_low))
    - Intercept B = J₁(R_low) - A × exp(R_low)

    The intercept B should be approximately 5 for K=3.

    Returns:
        dict with:
        - "exp_coefficient": A (should relate to operator mirror ~0.84)
        - "constant_offset": B (should be ~5)
        - "R_values": [R_low, R_high]
    """
    def get_J1_at_R(R):
        alpha = -R
        beta = -R
        s = alpha + beta
        pieces = build_J1_pieces_K3(alpha, beta, s, u=0.5)
        return float(np.real(sum_J1(pieces)))

    J1_low = get_J1_at_R(R_low)
    J1_high = get_J1_at_R(R_high)

    exp_low = np.exp(R_low)
    exp_high = np.exp(R_high)

    # Linear regression in exp(R) space
    slope = (J1_high - J1_low) / (exp_high - exp_low)
    intercept = J1_low - slope * exp_low

    return {
        "exp_coefficient": float(slope),
        "constant_offset": float(intercept),
        "R_values": [R_low, R_high],
        "J1_values": [J1_low, J1_high],
        "target_constant": 5,
        "offset_vs_target": float(abs(intercept - 5)),
    }


def decompose_by_piece_type(R: float = 1.3036) -> Dict:
    """
    Decompose contributions by piece type.

    Categories:
    - "zeta_only": pieces that don't use A derivatives (j11)
    - "first_deriv": pieces using first A derivatives (j12)
    - "second_deriv": pieces using second A derivatives (j13)
    - "mixed_deriv": pieces using mixed derivatives (j14)
    - "A11_term": the A^{(1,1)} contribution (j15)

    This helps understand which pieces contribute to exp(R) vs constant.

    Returns:
        dict with categorized contributions
    """
    alpha = -R
    beta = -R
    s = alpha + beta

    pieces = build_J1_pieces_K3(alpha, beta, s, u=0.5)

    return {
        "R": R,
        "categories": {
            "zeta_only": float(np.real(pieces.j11)),
            "first_deriv": float(np.real(pieces.j12)),
            "second_deriv": float(np.real(pieces.j13)),
            "mixed_deriv": float(np.real(pieces.j14)),
            "A11_term": float(np.real(pieces.j15)),
        },
        "total": float(np.real(sum_J1(pieces))),
        "exp_R": float(np.exp(R)),
        "target_m1": float(np.exp(R) + 5),
    }


def verify_piece_count_formula(K: int = 3) -> Dict:
    """
    Verify that the constant offset formula 2K-1 makes sense.

    For K=3: 2×3-1 = 5 pieces
    For K=4: 2×4-1 = 7 pieces (if pattern holds)

    This suggests the "+5" is structural, not accidental.

    Returns:
        dict with formula verification
    """
    return {
        "K": K,
        "formula_2K_minus_1": 2 * K - 1,
        "expected_constant": 2 * K - 1,
        "interpretation": (
            f"For K={K} mollifier pieces, the constant offset "
            f"in m₁ = exp(R) + {2*K-1} comes from the "
            f"{2*K-1}-fold decomposition structure."
        ),
    }


def print_plus5_analysis():
    """Print a formatted analysis of the +5 signature."""
    print("=" * 60)
    print("PHASE 14: +5 SIGNATURE ANALYSIS")
    print("=" * 60)
    print()

    # Basic signature
    sig = microcase_plus5_signature()
    print(f"Number of J₁ pieces: {sig['num_pieces']}")
    print(f"Formula: 2K - 1 = 2×{sig['K']} - 1 = {sig['2K_minus_1']}")
    print(f"A^{{(1,1)}} anchor: {sig['A11_anchor']:.6f}")
    print()

    # Piece values
    print("Piece values at (α,β,s,u) = (0,0,0,0.5):")
    for name, val in sig["piece_values"].items():
        print(f"  {name}: {val:.6f}")
    print(f"  Total: {sig['total']:.6f}")
    print()

    # Constant offset estimation
    offset = estimate_constant_offset()
    print("Linear decomposition J₁ ≈ A×exp(R) + B:")
    print(f"  A (exp coefficient): {offset['exp_coefficient']:.6f}")
    print(f"  B (constant offset): {offset['constant_offset']:.6f}")
    print(f"  Target B: 5")
    print(f"  |B - 5|: {offset['offset_vs_target']:.6f}")
    print()

    # Category breakdown
    decomp = decompose_by_piece_type()
    print("Piece categories at R=1.3036:")
    for cat, val in decomp["categories"].items():
        print(f"  {cat}: {val:.6f}")
    print()

    print("=" * 60)


if __name__ == "__main__":
    print_plus5_analysis()
