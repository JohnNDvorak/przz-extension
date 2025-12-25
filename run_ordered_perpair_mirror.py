"""
run_ordered_perpair_mirror.py
Per-pair mirror diagnostic with ORDERED PAIRS (Priority B - GPT Guidance 2025-12-19)

KEY INSIGHT: Triangle×2 symmetry is BROKEN for I3/I4 terms.
- (I3+I4)_12 ≠ (I3+I4)_21 (272-307% difference)
- (I3+I4)_13 ≠ (I3+I4)_31 (71% difference)
- (I3+I4)_23 ≠ (I3+I4)_32 (276% difference)

This script uses all 9 ordered pairs directly to:
1. Report A,B,C,D for each ordered pair separately
2. Compare 12 vs 21 mirror values directly
3. Identify if asymmetry is the source of pair-12 pathology

Computes for each pair:
- A = dir_I12 = (I1+I2)_direct
- B = mir_I12 = (I1+I2)_mirror
- C = dir_I34 = (I3+I4)_direct
- D = mir_I34 = (I3+I4)_mirror
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from src.evaluate import (
    EvaluationResult,
    evaluate_c_ordered,
    evaluate_c_ordered_with_exp_transform,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

# All 9 ordered pairs
ORDERED_PAIR_KEYS: Tuple[str, ...] = (
    "11", "22", "33",  # Diagonal
    "12", "21",        # Cross 1-2
    "13", "31",        # Cross 1-3
    "23", "32",        # Cross 2-3
)

# Factorial normalization weights: 1/(ℓ₁! × ℓ₂!)
FACTORIAL_WEIGHTS: Dict[str, float] = {
    "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1.0
    "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 0.25
    "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
    "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 0.5
    "21": 1.0 / (math.factorial(2) * math.factorial(1)),  # 0.5
    "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
    "31": 1.0 / (math.factorial(3) * math.factorial(1)),  # 1/6
    "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    "32": 1.0 / (math.factorial(3) * math.factorial(2)),  # 1/12
}

# Constants
THETA = 4.0 / 7.0
N_QUAD = 60
N_QUAD_A = 40


@dataclass
class OrderedPairDiagnostic:
    """Per-pair mirror diagnostic data for ordered pairs."""
    pair: str
    A: float  # dir_I12 = (I1+I2)_direct (RAW)
    B: float  # mir_I12 = (I1+I2)_mirror (RAW)
    C: float  # dir_I34 = (I3+I4)_direct (RAW)
    D: float  # mir_I34 = (I3+I4)_mirror (RAW)

    @property
    def weight(self) -> float:
        """Factorial weight for this pair."""
        return FACTORIAL_WEIGHTS[self.pair]

    @property
    def A_weighted(self) -> float:
        """Weighted dir_I12."""
        return self.weight * self.A

    @property
    def B_weighted(self) -> float:
        """Weighted mir_I12."""
        return self.weight * self.B

    @property
    def C_weighted(self) -> float:
        """Weighted dir_I34."""
        return self.weight * self.C

    @property
    def D_weighted(self) -> float:
        """Weighted mir_I34."""
        return self.weight * self.D

    @property
    def m_zero(self) -> float:
        """Scalar m that zeros this pair's WEIGHTED contribution."""
        if abs(self.B_weighted) < 1e-15:
            return float('inf')
        return -(self.A_weighted + self.C_weighted) / self.B_weighted

    @property
    def sensitivity(self) -> float:
        """Absolute sensitivity to mirror multiplier (WEIGHTED)."""
        return abs(self.B_weighted)

    def contrib_at_m(self, m: float) -> float:
        """WEIGHTED contribution to total c at given m."""
        return self.A_weighted + self.C_weighted + m * self.B_weighted

    @property
    def scalar_mirror_possible(self) -> bool:
        """Can a positive scalar m rescue this pair?"""
        direct_sum = self.A_weighted + self.C_weighted
        if self.B_weighted < 0 and direct_sum < 0:
            return False
        if self.B_weighted > 0 and direct_sum > 0:
            return True
        return True


def extract_i_values(
    res: EvaluationResult,
    pair: str,
) -> Tuple[float, float, float, float]:
    """Extract I1, I2, I3, I4 values for a given pair from ordered result.

    The per_term keys are like "12_I1_12", "12_I2_12", etc.
    """
    i1 = float(res.per_term.get(f"{pair}_I1_{pair}", 0.0))
    i2 = float(res.per_term.get(f"{pair}_I2_{pair}", 0.0))
    i3 = float(res.per_term.get(f"{pair}_I3_{pair}", 0.0))
    i4 = float(res.per_term.get(f"{pair}_I4_{pair}", 0.0))
    return i1, i2, i3, i4


def run_ordered_analysis(
    name: str,
    *,
    theta: float,
    R: float,
    c_target: float,
    polynomials: Dict[str, object],
    n_quad: int,
    n_quad_a: int,
    mirror_mode: str,
    q_a0_shift: float,
):
    """Run full per-pair analysis using ordered pairs."""
    print()
    print("=" * 100)
    print(f"ORDERED PAIR MIRROR ANALYSIS: {name} (R={R}, c_target={c_target})")
    print("=" * 100)

    # Direct evaluation
    direct = evaluate_c_ordered(
        theta=theta,
        R=R,
        n=n_quad,
        polynomials=polynomials,
        use_factorial_normalization=True,
        kernel_regime="paper",
        n_quad_a=n_quad_a,
    )

    # Mirror evaluation
    if mirror_mode == "r_flip":
        mirror = evaluate_c_ordered_with_exp_transform(
            theta=theta,
            R=-R,  # Flip R
            n=n_quad,
            polynomials=polynomials,
            kernel_regime="paper",
            exp_scale_multiplier=1.0,
            exp_t_flip=False,
            q_a0_shift=q_a0_shift,
            use_factorial_normalization=True,
            n_quad_a=n_quad_a,
        )
    elif mirror_mode == "exp_sign":
        mirror = evaluate_c_ordered_with_exp_transform(
            theta=theta,
            R=R,
            n=n_quad,
            polynomials=polynomials,
            kernel_regime="paper",
            exp_scale_multiplier=-1.0,
            exp_t_flip=False,
            q_a0_shift=q_a0_shift,
            use_factorial_normalization=True,
            n_quad_a=n_quad_a,
        )
    elif mirror_mode == "exp_sign_tflip":
        mirror = evaluate_c_ordered_with_exp_transform(
            theta=theta,
            R=R,
            n=n_quad,
            polynomials=polynomials,
            kernel_regime="paper",
            exp_scale_multiplier=-1.0,
            exp_t_flip=True,
            q_a0_shift=q_a0_shift,
            use_factorial_normalization=True,
            n_quad_a=n_quad_a,
        )
    else:
        raise ValueError(f"Unknown mirror_mode: {mirror_mode!r}")

    # Collect diagnostics for all 9 ordered pairs
    diagnostics: List[OrderedPairDiagnostic] = []

    for pair in ORDERED_PAIR_KEYS:
        i1_d, i2_d, i3_d, i4_d = extract_i_values(direct, pair)
        i1_m, i2_m, i3_m, i4_m = extract_i_values(mirror, pair)

        A = i1_d + i2_d  # direct I1+I2
        B = i1_m + i2_m  # mirror I1+I2
        C = i3_d + i4_d  # direct I3+I4
        D = i3_m + i4_m  # mirror I3+I4

        diagnostics.append(OrderedPairDiagnostic(pair=pair, A=A, B=B, C=C, D=D))

    # Print raw values
    print()
    print("Per-pair RAW values (9 ordered pairs, NOT weighted):")
    print("-" * 110)
    print(f"{'Pair':>6}  {'weight':>8}  {'A (dir_I12)':>14}  {'B (mir_I12)':>14}  {'C (dir_I34)':>14}  {'D (mir_I34)':>14}")
    print("-" * 110)

    for d in diagnostics:
        print(f"  {d.pair:>4}  {d.weight:8.4f}  {d.A:+14.8f}  {d.B:+14.8f}  {d.C:+14.8f}  {d.D:+14.8f}")

    # Print weighted values
    print()
    print("Per-pair WEIGHTED values (9 ordered pairs, weight × raw):")
    print("-" * 110)
    print(f"{'Pair':>6}  {'weight':>8}  {'A_w':>14}  {'B_w':>14}  {'C_w':>14}  {'D_w':>14}")
    print("-" * 110)

    total_A = total_B = total_C = total_D = 0.0
    for d in diagnostics:
        print(f"  {d.pair:>4}  {d.weight:8.4f}  {d.A_weighted:+14.8f}  {d.B_weighted:+14.8f}  {d.C_weighted:+14.8f}  {d.D_weighted:+14.8f}")
        total_A += d.A_weighted
        total_B += d.B_weighted
        total_C += d.C_weighted
        total_D += d.D_weighted

    print("-" * 110)
    print(f" TOTAL  {'':>8}  {total_A:+14.8f}  {total_B:+14.8f}  {total_C:+14.8f}  {total_D:+14.8f}")

    # Global m_needed (using weighted totals to match weighted c_target)
    m_global = (c_target - total_A - total_C) / total_B if abs(total_B) > 1e-15 else float('inf')

    print()
    print(f"Mirror mode: {mirror_mode}  (q_a0_shift={q_a0_shift:+g})")
    print(f"Global m_needed (for I1+I2 mirror only): {m_global:.6f}")
    print(f"exp(R) + 5 = {math.exp(R) + 5:.6f}")

    # CRITICAL COMPARISON: 12 vs 21
    print()
    print("=" * 100)
    print("CROSS-PAIR ASYMMETRY ANALYSIS (Triangle×2 Failure Diagnosis)")
    print("=" * 100)

    swap_pairs = [("12", "21"), ("13", "31"), ("23", "32")]
    for pair_a, pair_b in swap_pairs:
        d_a = next(d for d in diagnostics if d.pair == pair_a)
        d_b = next(d for d in diagnostics if d.pair == pair_b)

        # Note: pairs pq and qp have the SAME weight (symmetric in ℓ₁, ℓ₂)
        print(f"\n{pair_a} vs {pair_b}  (weight={d_a.weight:.4f}):")
        print(f"  A_w (dir I1+I2): {d_a.A_weighted:+14.8f} vs {d_b.A_weighted:+14.8f}  Δ={d_a.A_weighted - d_b.A_weighted:+.6e}")
        print(f"  B_w (mir I1+I2): {d_a.B_weighted:+14.8f} vs {d_b.B_weighted:+14.8f}  Δ={d_a.B_weighted - d_b.B_weighted:+.6e}")
        print(f"  C_w (dir I3+I4): {d_a.C_weighted:+14.8f} vs {d_b.C_weighted:+14.8f}  Δ={d_a.C_weighted - d_b.C_weighted:+.6e}")
        print(f"  D_w (mir I3+I4): {d_a.D_weighted:+14.8f} vs {d_b.D_weighted:+14.8f}  Δ={d_a.D_weighted - d_b.D_weighted:+.6e}")

        # Check if asymmetry explains pair-12 pathology
        if d_a.B_weighted < 0 and d_b.B_weighted >= 0:
            print(f"  ** {pair_a} has B_w<0 but {pair_b} has B_w>=0 - asymmetry matters! **")
        elif d_a.B_weighted >= 0 and d_b.B_weighted < 0:
            print(f"  ** {pair_b} has B_w<0 but {pair_a} has B_w>=0 - asymmetry matters! **")

        # Check if sum could be rescued (using weighted values)
        combined_A = d_a.A_weighted + d_b.A_weighted
        combined_B = d_a.B_weighted + d_b.B_weighted
        combined_C = d_a.C_weighted + d_b.C_weighted
        if combined_B != 0:
            m_combined = -(combined_A + combined_C) / combined_B
            print(f"  Combined m_zero (weighted): {m_combined:+.4f}")

    # Derived quantities (all weighted)
    print()
    print("Per-pair derived quantities (WEIGHTED):")
    print("-" * 100)
    print(f"{'Pair':>6}  {'m_zero':>12}  {'|B_w|':>12}  {'A_w+C_w':>12}  {'contrib@m':>14}  {'OK?':>6}")
    print("-" * 100)

    for d in diagnostics:
        m_z = d.m_zero
        m_z_str = f"{m_z:+12.4f}" if abs(m_z) < 1e6 else f"{'inf':>12}"
        contrib = d.contrib_at_m(m_global)
        ok = "YES" if d.scalar_mirror_possible else "NO"
        print(f"  {d.pair:>4}  {m_z_str}  {d.sensitivity:12.6f}  {d.A_weighted + d.C_weighted:+12.6f}  {contrib:+14.8f}  {ok:>6}")

    # Final c at global m
    c_at_m = sum(d.contrib_at_m(m_global) for d in diagnostics)
    print()
    print(f"Total c at m_global={m_global:.4f}: {c_at_m:.6f} (target: {c_target})")

    return diagnostics, m_global


def main():
    parser = argparse.ArgumentParser(description="Ordered-pair mirror diagnostics.")
    parser.add_argument("--n", type=int, default=N_QUAD, help="u/t quadrature points")
    parser.add_argument("--n-quad-a", type=int, default=N_QUAD_A, help="a quadrature points")
    parser.add_argument("--theta", type=float, default=THETA, help="theta")
    parser.add_argument(
        "--mirror-mode",
        choices=("r_flip", "exp_sign", "exp_sign_tflip"),
        default="r_flip",
        help="Mirror diagnostic mode",
    )
    parser.add_argument("--mirror-q-a0-shift", type=float, default=0.0)
    args = parser.parse_args()

    print("=" * 100)
    print("ORDERED PAIR MIRROR DIAGNOSTIC ANALYSIS")
    print("All 9 pairs: 11, 22, 33, 12, 21, 13, 31, 23, 32")
    print("=" * 100)
    print(f"Mode: {args.mirror_mode}")
    print()

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Run analysis
    diag_k, m_k = run_ordered_analysis(
        "κ",
        theta=args.theta,
        R=1.3036,
        c_target=2.137,
        polynomials=polys_k,
        n_quad=args.n,
        n_quad_a=args.n_quad_a,
        mirror_mode=args.mirror_mode,
        q_a0_shift=args.mirror_q_a0_shift,
    )

    diag_s, m_s = run_ordered_analysis(
        "κ*",
        theta=args.theta,
        R=1.1167,
        c_target=1.938,
        polynomials=polys_s,
        n_quad=args.n,
        n_quad_a=args.n_quad_a,
        mirror_mode=args.mirror_mode,
        q_a0_shift=args.mirror_q_a0_shift,
    )

    # Summary
    print()
    print("=" * 100)
    print("SUMMARY: m_needed comparison")
    print("=" * 100)
    print(f"κ  benchmark: m_needed = {m_k:.6f}, exp(R)+5 = {math.exp(1.3036)+5:.6f}")
    print(f"κ* benchmark: m_needed = {m_s:.6f}, exp(R)+5 = {math.exp(1.1167)+5:.6f}")
    print()

    # Key question: does ordering matter for the pathology? (Use WEIGHTED B values)
    d12_k = next(d for d in diag_k if d.pair == "12")
    d21_k = next(d for d in diag_k if d.pair == "21")
    b12_k = d12_k.B_weighted
    b21_k = d21_k.B_weighted

    print("KEY FINDING (κ benchmark, weighted B values):")
    print("-" * 60)
    if b12_k < 0 and b21_k < 0:
        print("  BOTH 12 and 21 have B_w < 0 in κ benchmark")
        print("  → Asymmetry is NOT the sole cause of pathology")
    elif b12_k * b21_k < 0:
        print(f"  12 has B_w = {b12_k:+.6f}, 21 has B_w = {b21_k:+.6f}")
        print("  → Signs DIFFER - asymmetry IS contributing to pathology!")
    else:
        print("  Both 12 and 21 have B_w >= 0")
        print("  → No pathology from these pairs")


if __name__ == "__main__":
    main()
