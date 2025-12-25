"""
src/ratios/bridge_to_S12.py
Phase 14 Task 5: Bridge J₁ Pieces to S12 Mirror Deficit

PAPER ANCHOR:
============
Phase 13 showed: operator mirror ≈ 0.84×exp(R), missing the +5.

This module connects the five-piece J₁ structure to the S12 mirror deficit:
- Operator mirror captures ONE piece (the exp(R) scaling)
- The "+5" comes from the OTHER J₁ components

HYPOTHESIS:
==========
m₁ = exp(R) + 5 decomposes as:
- exp(R) part: comes from the T^{-α-β} operator mirroring
- "+5" part: comes from the five-piece combinatorial structure

This module provides tools to analyze this decomposition.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from src.ratios.j1_k3_decomposition import (
    J1Pieces,
    build_J1_pieces_K3,
    sum_J1,
)
from src.ratios.arithmetic_factor import A11_prime_sum


def compute_S12_from_J1_pieces_micro(
    theta: float,
    R: float,
    *,
    pair: str = "11",
    Q_trivial: bool = True
) -> Dict:
    """
    Compute S12 mirror contribution from J₁ pieces.

    This provides a micro-case analysis of how J₁ pieces contribute
    to the S12 channel. Start with (1,1) pair and trivial Q.

    IMPORTANT: The contour variables s, u are SMALL (near 0) for residue
    extraction. They are DIFFERENT from the shift parameters α, β.
    The paper's bracket terms are functions of (s,u) that get integrated,
    with residues extracted at s=u=0.

    Args:
        theta: The θ parameter (typically 4/7)
        R: The PRZZ R parameter (e.g., 1.3036)
        pair: Which mollifier pair (default "11")
        Q_trivial: If True, use Q=1 (trivial polynomial)

    Returns:
        dict with:
        - "exp_R_coefficient": coefficient of exp(R) term
        - "constant_offset": additive constant (should be ~5)
        - "total": full mirror contribution
        - "per_piece": breakdown by J₁ piece
    """
    # At PRZZ point: α = β = -R
    alpha = -R
    beta = -R

    # Contour variables s, u are SMALL - we're extracting residues at s=u=0
    # Use small positive values to avoid poles
    s_values = [0.05, 0.1, 0.15]
    u_values = [0.05, 0.1, 0.15]

    per_piece_totals = {"j11": 0.0, "j12": 0.0, "j13": 0.0, "j14": 0.0, "j15": 0.0}
    total = 0.0
    n_points = 0

    for s in s_values:
        for u in u_values:
            pieces = build_J1_pieces_K3(alpha, beta, complex(s), complex(u))
            piece_sum = sum_J1(pieces)

            # Accumulate per-piece contributions
            for i, name in enumerate(["j11", "j12", "j13", "j14", "j15"]):
                per_piece_totals[name] += float(np.real(pieces[i]))

            total += float(np.real(piece_sum))
            n_points += 1

    # Normalize
    for name in per_piece_totals:
        per_piece_totals[name] /= n_points
    total /= n_points

    # Estimate exp(R) coefficient and constant offset via linear regression
    # Compare at R and R+0.1
    R2 = R + 0.1
    total_at_R2 = _compute_total_at_R(R2, s_values, u_values)

    exp_R = np.exp(R)
    exp_R2 = np.exp(R2)

    # Linear regression: total = A * exp(R) + B
    slope = (total_at_R2 - total) / (exp_R2 - exp_R)
    intercept = total - slope * exp_R

    return {
        "exp_R_coefficient": float(slope),
        "constant_offset": float(intercept),
        "total": float(total),
        "per_piece": per_piece_totals,
        "R": R,
        "theta": theta,
    }


def _compute_total_at_R(
    R: float,
    s_values: List[float],
    u_values: List[float]
) -> float:
    """Helper to compute total at a given R using small contour variables."""
    alpha = -R
    beta = -R

    total = 0.0
    n_points = 0
    for s in s_values:
        for u in u_values:
            pieces = build_J1_pieces_K3(alpha, beta, complex(s), complex(u))
            total += float(np.real(sum_J1(pieces)))
            n_points += 1

    return total / n_points


def decompose_m1_from_pieces(
    theta: float,
    R: float
) -> Dict:
    """
    Decompose m₁ = exp(R) + 5 into piece contributions.

    Shows which J₁ pieces contribute to exp(R) vs constant.

    Args:
        theta: The θ parameter
        R: The PRZZ R parameter

    Returns:
        dict with:
        - "exp_coefficient": coefficient of exp(R) part
        - "constant_offset": additive constant part
        - "target_constant": 2K-1 = 5 for K=3
        - "per_piece_contribution": how each piece contributes
    """
    # Compute at two R values to separate exp(R) from constant
    R1, R2 = R, R + 0.2

    s12_R1 = compute_S12_from_J1_pieces_micro(theta, R1)
    s12_R2 = compute_S12_from_J1_pieces_micro(theta, R2)

    total_R1 = s12_R1["total"]
    total_R2 = s12_R2["total"]

    exp_R1 = np.exp(R1)
    exp_R2 = np.exp(R2)

    # Solve: total = A * exp(R) + B
    A = (total_R2 - total_R1) / (exp_R2 - exp_R1)
    B = total_R1 - A * exp_R1

    # Per-piece contribution analysis
    per_piece_contribution = {}
    for name in ["j11", "j12", "j13", "j14", "j15"]:
        val_R1 = s12_R1["per_piece"][name]
        val_R2 = s12_R2["per_piece"][name]

        # Estimate scaling for this piece
        piece_slope = (val_R2 - val_R1) / (exp_R2 - exp_R1)
        piece_intercept = val_R1 - piece_slope * exp_R1

        per_piece_contribution[name] = {
            "exp_coefficient": float(piece_slope),
            "constant": float(piece_intercept),
        }

    return {
        "exp_coefficient": float(A),
        "constant_offset": float(B),
        "target_constant": 5,  # = 2K - 1 for K=3
        "per_piece_contribution": per_piece_contribution,
        "R_values": [R1, R2],
        "total_values": [total_R1, total_R2],
    }


def analyze_piece_exp_vs_constant(R: float = 1.3036) -> Dict:
    """
    Analyze which pieces contribute to exp(R) vs constant.

    This is the key analysis for understanding the +5 origin.

    Returns:
        dict with:
        - "piece_scaling": for each piece, whether it scales with exp(R)
        - "exp_scaling_pieces": list of pieces that scale with exp(R)
        - "constant_pieces": list of pieces that contribute constant offset
    """
    # Use decomposition from above
    decomp = decompose_m1_from_pieces(theta=4.0 / 7.0, R=R)

    piece_scaling = {}
    exp_scaling_pieces = []
    constant_pieces = []

    for name, contrib in decomp["per_piece_contribution"].items():
        exp_coef = contrib["exp_coefficient"]
        constant = contrib["constant"]

        # Determine dominant contribution
        # A piece "scales with exp(R)" if |exp_coef * exp(R)| > |constant|
        exp_contribution = abs(exp_coef * np.exp(R))
        const_contribution = abs(constant)

        if exp_contribution > const_contribution and abs(exp_coef) > 0.01:
            scaling_type = "exp"
            exp_scaling_pieces.append(name)
        else:
            scaling_type = "constant"
            constant_pieces.append(name)

        piece_scaling[name] = {
            "type": scaling_type,
            "exp_coefficient": exp_coef,
            "constant": constant,
            "exp_contribution": float(exp_contribution),
            "const_contribution": float(const_contribution),
        }

    return {
        "R": R,
        "piece_scaling": piece_scaling,
        "exp_scaling_pieces": exp_scaling_pieces,
        "constant_pieces": constant_pieces,
        "total_exp_coefficient": decomp["exp_coefficient"],
        "total_constant_offset": decomp["constant_offset"],
    }


def get_operator_mirror_piece(R: float = 1.3036) -> Dict:
    """
    Identify which piece(s) match the operator mirror behavior.

    Phase 13 found: operator mirror ≈ 0.84 × exp(R).

    This function identifies which J₁ piece(s) exhibit similar scaling.

    Returns:
        dict with:
        - "matching_pieces": pieces that scale like operator mirror
        - "operator_value": the operator mirror value (~0.84×exp(R))
        - "piece_values": values of each piece
    """
    # Phase 13 operator mirror coefficient
    OPERATOR_COEFFICIENT = 0.84

    exp_R = np.exp(R)
    operator_value = OPERATOR_COEFFICIENT * exp_R

    analysis = analyze_piece_exp_vs_constant(R)

    matching_pieces = []
    piece_values = {}

    for name, scaling in analysis["piece_scaling"].items():
        piece_val = scaling["exp_coefficient"] * exp_R + scaling["constant"]
        piece_values[name] = float(piece_val)

        # Check if this piece matches operator mirror behavior
        if scaling["type"] == "exp" and abs(scaling["exp_coefficient"]) > 0.1:
            matching_pieces.append(name)

    return {
        "R": R,
        "matching_pieces": matching_pieces,
        "operator_value": float(operator_value),
        "operator_coefficient": OPERATOR_COEFFICIENT,
        "piece_values": piece_values,
        "exp_R": float(exp_R),
    }


def print_bridge_analysis(R: float = 1.3036):
    """Print a formatted analysis of the bridge to S12."""
    print("=" * 60)
    print(f"PHASE 14: BRIDGE TO S12 ANALYSIS (R={R})")
    print("=" * 60)
    print()

    # Decomposition
    decomp = decompose_m1_from_pieces(theta=4.0 / 7.0, R=R)
    print("m₁ decomposition:")
    print(f"  m₁ ≈ A × exp(R) + B")
    print(f"  A (exp coefficient): {decomp['exp_coefficient']:.6f}")
    print(f"  B (constant offset): {decomp['constant_offset']:.6f}")
    print(f"  Target B: {decomp['target_constant']} (= 2K-1 for K=3)")
    print()

    print("Per-piece contributions:")
    for name, contrib in decomp["per_piece_contribution"].items():
        print(f"  {name}:")
        print(f"    exp coefficient: {contrib['exp_coefficient']:.6f}")
        print(f"    constant: {contrib['constant']:.6f}")
    print()

    # Operator mirror comparison
    op_result = get_operator_mirror_piece(R)
    print("Operator mirror comparison:")
    print(f"  Phase 13 operator: {op_result['operator_coefficient']:.2f} × exp(R)")
    print(f"  Operator value: {op_result['operator_value']:.4f}")
    print(f"  exp(R): {op_result['exp_R']:.4f}")
    print(f"  Matching pieces: {op_result['matching_pieces']}")
    print()

    # Piece scaling analysis
    scaling = analyze_piece_exp_vs_constant(R)
    print("Piece scaling analysis:")
    print(f"  exp(R)-scaling pieces: {scaling['exp_scaling_pieces']}")
    print(f"  Constant pieces: {scaling['constant_pieces']}")
    print()

    print("=" * 60)


if __name__ == "__main__":
    print_bridge_analysis()
