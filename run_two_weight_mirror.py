"""
run_two_weight_mirror.py
Two-Weight Mirror Model Analysis (GPT Guidance 2025-12-19)

This script computes:
1. Split I₁/I₂ channels at +R and -R
2. S12_plus, S12_minus, S34_plus (ordered sum)
3. Single-m model: c = S12_plus + m*S12_minus + S34_plus
4. Two-weight model: c = I1_plus + m1*I1_minus + I2_plus + m2*I2_minus + S34_plus

Solves for (m1, m2) using both κ and κ* targets.
"""

from __future__ import annotations

import argparse
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from src.evaluate import (
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

THETA = 4.0 / 7.0
N_QUAD = 60
N_QUAD_A = 40


@dataclass
class SplitChannels:
    """Split I₁/I₂ channels for a benchmark."""
    name: str
    R: float
    c_target: float

    # Individual channels (weighted sums over all 9 ordered pairs)
    I1_plus: float   # Σ w_pq * I₁(pq) at +R
    I2_plus: float   # Σ w_pq * I₂(pq) at +R
    I1_minus: float  # Σ w_pq * I₁(pq) at -R
    I2_minus: float  # Σ w_pq * I₂(pq) at -R
    I3_plus: float   # Σ w_pq * I₃(pq) at +R
    I4_plus: float   # Σ w_pq * I₄(pq) at +R

    @property
    def S12_plus(self) -> float:
        return self.I1_plus + self.I2_plus

    @property
    def S12_minus(self) -> float:
        return self.I1_minus + self.I2_minus

    @property
    def S34_plus(self) -> float:
        return self.I3_plus + self.I4_plus


def extract_split_i_values(res, pair: str) -> Tuple[float, float, float, float]:
    """Extract individual I1, I2, I3, I4 values (raw) for a given pair."""
    i1 = float(res.per_term.get(f"{pair}_I1_{pair}", 0.0))
    i2 = float(res.per_term.get(f"{pair}_I2_{pair}", 0.0))
    i3 = float(res.per_term.get(f"{pair}_I3_{pair}", 0.0))
    i4 = float(res.per_term.get(f"{pair}_I4_{pair}", 0.0))
    return i1, i2, i3, i4


def compute_split_channels(
    name: str,
    *,
    theta: float,
    R: float,
    c_target: float,
    polynomials: Dict,
    n_quad: int,
    n_quad_a: int,
    mirror_q_poly_shift: float = 0.0,
) -> SplitChannels:
    """Compute split I₁/I₂ channels for a benchmark."""

    # Direct evaluation at +R
    direct = evaluate_c_ordered(
        theta=theta,
        R=R,
        n=n_quad,
        polynomials=polynomials,
        use_factorial_normalization=True,
        kernel_regime="paper",
        n_quad_a=n_quad_a,
    )

    polynomials_mirror = polynomials
    if mirror_q_poly_shift != 0.0:
        from src.q_operator import lift_poly_by_shift

        polynomials_mirror = dict(polynomials)
        polynomials_mirror["Q"] = lift_poly_by_shift(polynomials["Q"], shift=mirror_q_poly_shift)

    # Mirror evaluation at -R
    mirror = evaluate_c_ordered_with_exp_transform(
        theta=theta,
        R=-R,  # Flip R
        n=n_quad,
        polynomials=polynomials_mirror,
        kernel_regime="paper",
        exp_scale_multiplier=1.0,
        exp_t_flip=False,
        q_a0_shift=0.0,
        use_factorial_normalization=True,
        n_quad_a=n_quad_a,
    )

    # Accumulate weighted channel sums
    I1_plus = I2_plus = I1_minus = I2_minus = 0.0
    I3_plus = I4_plus = 0.0

    for pair in ORDERED_PAIR_KEYS:
        w = FACTORIAL_WEIGHTS[pair]

        i1_d, i2_d, i3_d, i4_d = extract_split_i_values(direct, pair)
        i1_m, i2_m, i3_m, i4_m = extract_split_i_values(mirror, pair)

        I1_plus += w * i1_d
        I2_plus += w * i2_d
        I1_minus += w * i1_m
        I2_minus += w * i2_m
        I3_plus += w * i3_d
        I4_plus += w * i4_d

    return SplitChannels(
        name=name,
        R=R,
        c_target=c_target,
        I1_plus=I1_plus,
        I2_plus=I2_plus,
        I1_minus=I1_minus,
        I2_minus=I2_minus,
        I3_plus=I3_plus,
        I4_plus=I4_plus,
    )


def solve_single_m(ch: SplitChannels) -> float:
    """Solve for single m in: c = S12_plus + m*S12_minus + S34_plus."""
    rhs = ch.c_target - ch.S12_plus - ch.S34_plus
    if abs(ch.S12_minus) < 1e-15:
        return float('inf')
    return rhs / ch.S12_minus


def solve_two_weight(ch_kappa: SplitChannels, ch_kappa_star: SplitChannels) -> Optional[Tuple[float, float]]:
    """
    Solve for (m1, m2) using both benchmarks:

    c_κ  = I1⁺_κ  + m1*I1⁻_κ  + I2⁺_κ  + m2*I2⁻_κ  + S34⁺_κ
    c_κ* = I1⁺_κ* + m1*I1⁻_κ* + I2⁺_κ* + m2*I2⁻_κ* + S34⁺_κ*

    Rewritten as:
    m1*I1⁻_κ  + m2*I2⁻_κ  = c_κ  - I1⁺_κ  - I2⁺_κ  - S34⁺_κ
    m1*I1⁻_κ* + m2*I2⁻_κ* = c_κ* - I1⁺_κ* - I2⁺_κ* - S34⁺_κ*

    Matrix form: A @ [m1, m2]^T = b
    """
    # Coefficient matrix
    A = np.array([
        [ch_kappa.I1_minus, ch_kappa.I2_minus],
        [ch_kappa_star.I1_minus, ch_kappa_star.I2_minus],
    ])

    # RHS
    b = np.array([
        ch_kappa.c_target - ch_kappa.I1_plus - ch_kappa.I2_plus - ch_kappa.S34_plus,
        ch_kappa_star.c_target - ch_kappa_star.I1_plus - ch_kappa_star.I2_plus - ch_kappa_star.S34_plus,
    ])

    # Check for singular matrix
    det = np.linalg.det(A)
    if abs(det) < 1e-15:
        return None

    # Solve
    m = np.linalg.solve(A, b)
    return float(m[0]), float(m[1])


def compute_c_at_weights(ch: SplitChannels, m1: float, m2: float) -> float:
    """Compute c with two-weight model."""
    return ch.I1_plus + m1 * ch.I1_minus + ch.I2_plus + m2 * ch.I2_minus + ch.S34_plus


def main():
    parser = argparse.ArgumentParser(description="Two-weight mirror model analysis.")
    parser.add_argument("--n", type=int, default=N_QUAD, help="u/t quadrature points")
    parser.add_argument("--n-quad-a", type=int, default=N_QUAD_A, help="a quadrature points")
    parser.add_argument("--theta", type=float, default=THETA, help="theta")
    parser.add_argument(
        "--mirror-mode",
        choices=["baseline", "operator_q_shift"],
        default="baseline",
        help=(
            "Mirror (-R) branch mode: 'baseline' uses Q as-is; "
            "'operator_q_shift' applies Q(1+D) by replacing Q(x) with Q(x+1) in the mirror branch."
        ),
    )
    args = parser.parse_args()

    mirror_q_poly_shift = 1.0 if args.mirror_mode == "operator_q_shift" else 0.0

    print("=" * 100)
    print("TWO-WEIGHT MIRROR MODEL ANALYSIS")
    print("=" * 100)
    print(f"mirror_mode: {args.mirror_mode} (mirror_q_poly_shift={mirror_q_poly_shift})")
    print()

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Compute split channels for both benchmarks
    print("Computing split channels for κ benchmark (R=1.3036)...")
    ch_kappa = compute_split_channels(
        "κ",
        theta=args.theta,
        R=1.3036,
        c_target=2.137,
        polynomials=polys_kappa,
        n_quad=args.n,
        n_quad_a=args.n_quad_a,
        mirror_q_poly_shift=mirror_q_poly_shift,
    )

    print("Computing split channels for κ* benchmark (R=1.1167)...")
    ch_kappa_star = compute_split_channels(
        "κ*",
        theta=args.theta,
        R=1.1167,
        c_target=1.938,
        polynomials=polys_kappa_star,
        n_quad=args.n,
        n_quad_a=args.n_quad_a,
        mirror_q_poly_shift=mirror_q_poly_shift,
    )

    # Report split channels
    print()
    print("=" * 100)
    print("SPLIT CHANNEL VALUES (weighted sums over ordered pairs)")
    print("=" * 100)

    for ch in [ch_kappa, ch_kappa_star]:
        print()
        print(f"--- {ch.name} benchmark (R={ch.R}, c_target={ch.c_target}) ---")
        print(f"  I1_plus  (+R): {ch.I1_plus:+.8f}")
        print(f"  I2_plus  (+R): {ch.I2_plus:+.8f}")
        print(f"  I1_minus (-R): {ch.I1_minus:+.8f}")
        print(f"  I2_minus (-R): {ch.I2_minus:+.8f}")
        print(f"  I3_plus  (+R): {ch.I3_plus:+.8f}")
        print(f"  I4_plus  (+R): {ch.I4_plus:+.8f}")
        print()
        print(f"  S12_plus  = I1+I2 at +R: {ch.S12_plus:+.8f}")
        print(f"  S12_minus = I1+I2 at -R: {ch.S12_minus:+.8f}")
        print(f"  S34_plus  = I3+I4 at +R: {ch.S34_plus:+.8f}")

    # Single-m model
    print()
    print("=" * 100)
    print("SINGLE-M MODEL: c = S12_plus + m*S12_minus + S34_plus")
    print("=" * 100)

    m_kappa = solve_single_m(ch_kappa)
    m_kappa_star = solve_single_m(ch_kappa_star)

    print()
    print(f"κ  benchmark: m_needed = {m_kappa:.6f}")
    print(f"              exp(R)+5 = {math.exp(1.3036)+5:.6f}")
    print(f"              gap      = {m_kappa - (math.exp(1.3036)+5):.6f}")
    print()
    print(f"κ* benchmark: m_needed = {m_kappa_star:.6f}")
    print(f"              exp(R)+5 = {math.exp(1.1167)+5:.6f}")
    print(f"              gap      = {m_kappa_star - (math.exp(1.1167)+5):.6f}")

    # Check cross-validation: use κ's m on κ*, and vice versa
    print()
    print("Cross-validation:")
    c_kappa_at_mstar = ch_kappa.S12_plus + m_kappa_star * ch_kappa.S12_minus + ch_kappa.S34_plus
    c_kstar_at_mkappa = ch_kappa_star.S12_plus + m_kappa * ch_kappa_star.S12_minus + ch_kappa_star.S34_plus

    print(f"  κ at m_κ*: c={c_kappa_at_mstar:.6f} (target={ch_kappa.c_target}, gap={100*(c_kappa_at_mstar/ch_kappa.c_target - 1):+.2f}%)")
    print(f"  κ* at m_κ: c={c_kstar_at_mkappa:.6f} (target={ch_kappa_star.c_target}, gap={100*(c_kstar_at_mkappa/ch_kappa_star.c_target - 1):+.2f}%)")

    # Two-weight model
    print()
    print("=" * 100)
    print("TWO-WEIGHT MODEL: c = I1_plus + m1*I1_minus + I2_plus + m2*I2_minus + S34_plus")
    print("=" * 100)

    result = solve_two_weight(ch_kappa, ch_kappa_star)

    if result is None:
        print("\nSingular matrix - cannot solve for (m1, m2)")
    else:
        m1, m2 = result

        print()
        print(f"Solved weights: m1 = {m1:.6f}, m2 = {m2:.6f}")
        print()

        # Verify both targets are hit
        c_kappa_2w = compute_c_at_weights(ch_kappa, m1, m2)
        c_kstar_2w = compute_c_at_weights(ch_kappa_star, m1, m2)

        print("Verification (should hit targets exactly):")
        print(f"  κ:  c_computed = {c_kappa_2w:.8f}, target = {ch_kappa.c_target}, Δ = {c_kappa_2w - ch_kappa.c_target:.2e}")
        print(f"  κ*: c_computed = {c_kstar_2w:.8f}, target = {ch_kappa_star.c_target}, Δ = {c_kstar_2w - ch_kappa_star.c_target:.2e}")

        # Compare to single-m
        print()
        print("Comparison to single-m model:")
        m_avg_single = (m_kappa + m_kappa_star) / 2
        print(f"  Average single-m: {m_avg_single:.6f}")
        print(f"  Two-weight m1:    {m1:.6f} (Δ from avg: {m1 - m_avg_single:+.6f})")
        print(f"  Two-weight m2:    {m2:.6f} (Δ from avg: {m2 - m_avg_single:+.6f})")

        # Stability check: are m1 and m2 close to each other?
        m_ratio = m1 / m2 if m2 != 0 else float('inf')
        print()
        print(f"Stability check:")
        print(f"  m1/m2 ratio: {m_ratio:.4f}")
        print(f"  |m1-m2|:     {abs(m1-m2):.6f}")

        if abs(m_ratio - 1) < 0.1:
            print("  → Weights are similar - single-m model may suffice")
        else:
            print("  → Weights differ significantly - two-weight model is needed!")

        # Report exp(R) + const structure
        print()
        print("Structure analysis (m = exp(R) + const):")
        exp_R_kappa = math.exp(1.3036)
        exp_R_kstar = math.exp(1.1167)

        # If m1 for κ = exp(R_κ) + k1, and m1 for κ* = exp(R_κ*) + k1...
        # But we have a single m1 that works for BOTH - so check what exp(R)+k gives
        print(f"  exp(R_κ)  = {exp_R_kappa:.6f}")
        print(f"  exp(R_κ*) = {exp_R_kstar:.6f}")

        # For m1, what constant k would make m1 = exp(R) + k match?
        # That's not directly answerable since m1 is a single value for both R's
        # But we can check if m1 looks like either exp(R) + something
        k1_at_kappa = m1 - exp_R_kappa
        k1_at_kstar = m1 - exp_R_kstar
        k2_at_kappa = m2 - exp_R_kappa
        k2_at_kstar = m2 - exp_R_kstar

        print()
        print(f"  m1 - exp(R_κ)  = {k1_at_kappa:+.6f}")
        print(f"  m1 - exp(R_κ*) = {k1_at_kstar:+.6f}")
        print(f"  m2 - exp(R_κ)  = {k2_at_kappa:+.6f}")
        print(f"  m2 - exp(R_κ*) = {k2_at_kstar:+.6f}")

    # Coefficient matrix analysis
    print()
    print("=" * 100)
    print("COEFFICIENT MATRIX ANALYSIS")
    print("=" * 100)

    A = np.array([
        [ch_kappa.I1_minus, ch_kappa.I2_minus],
        [ch_kappa_star.I1_minus, ch_kappa_star.I2_minus],
    ])

    det = np.linalg.det(A)
    cond = np.linalg.cond(A)

    print()
    print("Matrix A (I-minus coefficients):")
    print(f"  [ {A[0,0]:+.8f}  {A[0,1]:+.8f} ]  (κ)")
    print(f"  [ {A[1,0]:+.8f}  {A[1,1]:+.8f} ]  (κ*)")
    print()
    print(f"  det(A) = {det:.8e}")
    print(f"  cond(A) = {cond:.2f}")

    if cond > 100:
        print("  → HIGH condition number - solution may be unstable!")
    elif cond > 10:
        print("  → Moderate condition number - exercise caution")
    else:
        print("  → Well-conditioned system")

    # Channel ratios
    print()
    print("=" * 100)
    print("CHANNEL RATIOS (κ / κ*)")
    print("=" * 100)

    print()
    print(f"  I1_plus  ratio: {ch_kappa.I1_plus / ch_kappa_star.I1_plus:.4f}" if ch_kappa_star.I1_plus != 0 else "  I1_plus ratio: undefined")
    print(f"  I2_plus  ratio: {ch_kappa.I2_plus / ch_kappa_star.I2_plus:.4f}" if ch_kappa_star.I2_plus != 0 else "  I2_plus ratio: undefined")
    print(f"  I1_minus ratio: {ch_kappa.I1_minus / ch_kappa_star.I1_minus:.4f}" if ch_kappa_star.I1_minus != 0 else "  I1_minus ratio: undefined")
    print(f"  I2_minus ratio: {ch_kappa.I2_minus / ch_kappa_star.I2_minus:.4f}" if ch_kappa_star.I2_minus != 0 else "  I2_minus ratio: undefined")
    print(f"  S34_plus ratio: {ch_kappa.S34_plus / ch_kappa_star.S34_plus:.4f}" if ch_kappa_star.S34_plus != 0 else "  S34_plus ratio: undefined")
    print()
    print(f"  c_target ratio: {ch_kappa.c_target / ch_kappa_star.c_target:.4f}")

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
