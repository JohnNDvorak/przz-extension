"""
run_perpair_mirror_analysis.py
Per-pair mirror diagnostic analysis (GPT recommendation 2025-12-19)

NOTE (sync with DSL truth path)
-------------------------------
This script is intended to diagnose the **paper regime** (ω-driven Case B/C
selection for P₂/P₃). Term generation must therefore pass
`kernel_regime="paper"`; otherwise the `_v2` builders default to raw.

Computes for each pair:
- A = dir_I12 = (I1+I2)_direct
- B = mir_I12 = (I1+I2)_mirror
- C = dir_I34 = (I3+I4)_direct
- D = mir_I34 = (I3+I4)_mirror

Derived quantities:
- m_zero = -(A + C) / B  (scalar multiplier to zero out pair's contribution)
- sensitivity = |B|  (how much pair affects mirror recombination)
- contrib@m = A + C + m * B  (contribution at given m)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.evaluate import EvaluationResult, evaluate_c_full, evaluate_c_full_with_exp_transform
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

PAIR_KEYS: Tuple[str, ...] = ("11", "22", "33", "12", "13", "23")

# Constants
THETA = 4.0 / 7.0
N_QUAD = 60
N_QUAD_A = 40

# Factorial normalizations (matches scale audit per-pair breakdown)
# NOTE: symmetry factor is NOT applied here - it's applied when summing to total c
FACTORIAL_NORM = {
    "11": 1.0,
    "22": 1.0 / 4.0,
    "33": 1.0 / 36.0,
    "12": 1.0 / 2.0,
    "13": 1.0 / 6.0,
    "23": 1.0 / 12.0,
}

# Symmetry factors for total c computation (2 for off-diagonal)
# Applied only when computing aggregate totals, NOT per-pair values
SYMMETRY = {
    "11": 1.0,
    "22": 1.0,
    "33": 1.0,
    "12": 2.0,
    "13": 2.0,
    "23": 2.0,
}


@dataclass
class PairDiagnostic:
    """Per-pair mirror diagnostic data."""
    pair: str
    A: float  # dir_I12
    B: float  # mir_I12
    C: float  # dir_I34
    D: float  # mir_I34

    @property
    def m_zero(self) -> float:
        """Scalar m that zeros this pair's total contribution."""
        if abs(self.B) < 1e-15:
            return float('inf')
        return -(self.A + self.C) / self.B

    @property
    def sensitivity(self) -> float:
        """Absolute sensitivity to mirror multiplier."""
        return abs(self.B)

    def contrib_at_m(self, m: float) -> float:
        """Contribution to total c at given m."""
        return self.A + self.C + m * self.B

    @property
    def scalar_mirror_possible(self) -> bool:
        """Can a positive scalar m rescue this pair?

        Impossible if B < 0 and A + C < 0 (need negative m).
        """
        direct_sum = self.A + self.C
        if self.B < 0 and direct_sum < 0:
            return False  # Need m < 0 to make positive
        if self.B > 0 and direct_sum > 0:
            return True  # Any positive m works
        return True  # Edge cases


def _norm_factor(pair: str, *, use_factorial_normalization: bool) -> float:
    """Return the (symmetry × factorial) normalization factor for a pair."""
    base = SYMMETRY[pair]
    if use_factorial_normalization:
        base *= FACTORIAL_NORM[pair]
    return base


def _pair_groups(
    res: EvaluationResult,
    pair: str,
    *,
    use_factorial_normalization: bool,
) -> Tuple[float, float]:
    """Return (I1+I2, I3+I4) normalized contributions for a single pair."""
    nf = _norm_factor(pair, use_factorial_normalization=use_factorial_normalization)
    i12_raw = float(res.per_term.get(f"I1_{pair}", 0.0)) + float(res.per_term.get(f"I2_{pair}", 0.0))
    i34_raw = float(res.per_term.get(f"I3_{pair}", 0.0)) + float(res.per_term.get(f"I4_{pair}", 0.0))
    return nf * i12_raw, nf * i34_raw


def evaluate_pair_groups(
    pair: str,
    direct: EvaluationResult,
    mirror: EvaluationResult,
    *,
    use_factorial_normalization: bool,
) -> PairDiagnostic:
    """Evaluate A, B, C, D for a single pair from two EvaluationResult objects."""
    A, C = _pair_groups(direct, pair, use_factorial_normalization=use_factorial_normalization)
    B, D = _pair_groups(mirror, pair, use_factorial_normalization=use_factorial_normalization)
    return PairDiagnostic(pair=pair, A=A, B=B, C=C, D=D)


def run_analysis(
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
    use_factorial_normalization: bool,
):
    """Run full per-pair analysis for one benchmark."""
    print()
    print("=" * 90)
    print(f"PER-PAIR MIRROR ANALYSIS: {name} (R={R}, c_target={c_target})")
    print("=" * 90)

    # Direct evaluation (paper regime).
    direct = evaluate_c_full(
        theta=theta,
        R=R,
        n=n_quad,
        polynomials=polynomials,
        return_breakdown=True,
        use_factorial_normalization=use_factorial_normalization,
        mode="main",
        kernel_regime="paper",
        n_quad_a=n_quad_a,
    )

    # Mirror evaluation (diagnostic only).
    if mirror_mode == "r_flip":
        mirror = evaluate_c_full_with_exp_transform(
            theta=theta,
            R=-R,
            n=n_quad,
            polynomials=polynomials,
            kernel_regime="paper",
            exp_scale_multiplier=1.0,
            exp_t_flip=False,
            q_a0_shift=q_a0_shift,
            return_breakdown=True,
            use_factorial_normalization=use_factorial_normalization,
            mode="main",
            n_quad_a=n_quad_a,
        )
    elif mirror_mode == "exp_sign":
        mirror = evaluate_c_full_with_exp_transform(
            theta=theta,
            R=R,
            n=n_quad,
            polynomials=polynomials,
            kernel_regime="paper",
            exp_scale_multiplier=-1.0,
            exp_t_flip=False,
            q_a0_shift=q_a0_shift,
            return_breakdown=True,
            use_factorial_normalization=use_factorial_normalization,
            mode="main",
            n_quad_a=n_quad_a,
        )
    elif mirror_mode == "exp_sign_tflip":
        mirror = evaluate_c_full_with_exp_transform(
            theta=theta,
            R=R,
            n=n_quad,
            polynomials=polynomials,
            kernel_regime="paper",
            exp_scale_multiplier=-1.0,
            exp_t_flip=True,
            q_a0_shift=q_a0_shift,
            return_breakdown=True,
            use_factorial_normalization=use_factorial_normalization,
            mode="main",
            n_quad_a=n_quad_a,
        )
    else:
        raise ValueError(f"Unknown mirror_mode: {mirror_mode!r}")

    # Collect diagnostics for all pairs.
    diagnostics: List[PairDiagnostic] = []
    for pair in PAIR_KEYS:
        diag = evaluate_pair_groups(pair, direct, mirror, use_factorial_normalization=use_factorial_normalization)
        diagnostics.append(diag)

    # Print raw values
    print()
    print("Per-pair raw values (normalized):")
    print("-" * 90)
    print(f"{'Pair':>6}  {'A (dir_I12)':>14}  {'B (mir_I12)':>14}  {'C (dir_I34)':>14}  {'D (mir_I34)':>14}")
    print("-" * 90)

    total_A = total_B = total_C = total_D = 0.0
    for d in diagnostics:
        print(f"  {d.pair:>4}  {d.A:+14.8f}  {d.B:+14.8f}  {d.C:+14.8f}  {d.D:+14.8f}")
        total_A += d.A
        total_B += d.B
        total_C += d.C
        total_D += d.D

    print("-" * 90)
    print(f" TOTAL  {total_A:+14.8f}  {total_B:+14.8f}  {total_C:+14.8f}  {total_D:+14.8f}")

    # Compute global m_needed (for I1+I2 mirror only)
    m_global = (c_target - total_A - total_C) / total_B if abs(total_B) > 1e-15 else float('inf')

    print()
    print(f"Mirror mode: {mirror_mode}  (q_a0_shift={q_a0_shift:+g})")
    print(f"Global m_needed (for I1+I2 mirror only): {m_global:.6f}")
    print(f"exp(R) + 5 = {math.exp(R) + 5:.6f}")
    print()

    # Derived quantities
    print("Per-pair derived quantities:")
    print("-" * 90)
    print(f"{'Pair':>6}  {'m_zero':>12}  {'sensitivity':>12}  {'A+C':>12}  {'contrib@m':>14}  {'Scalar OK?':>10}")
    print("-" * 90)

    for d in diagnostics:
        m_z = d.m_zero
        m_z_str = f"{m_z:+12.4f}" if abs(m_z) < 1e6 else f"{'inf':>12}"
        contrib = d.contrib_at_m(m_global)
        ok = "YES" if d.scalar_mirror_possible else "**NO**"
        print(f"  {d.pair:>4}  {m_z_str}  {d.sensitivity:12.6f}  {d.A + d.C:+12.6f}  {contrib:+14.8f}  {ok:>10}")

    # Sort by sensitivity
    print()
    print("Pairs sorted by |B| (sensitivity to mirror multiplier):")
    sorted_by_sens = sorted(diagnostics, key=lambda x: x.sensitivity, reverse=True)
    for d in sorted_by_sens:
        sign = "+" if d.B >= 0 else "-"
        print(f"  {d.pair}: |B|={d.sensitivity:.6f} ({sign})")

    # Identify problematic pairs
    print()
    print("Pairs where scalar mirror is IMPOSSIBLE (B<0 and A+C<0):")
    problematic = [d for d in diagnostics if not d.scalar_mirror_possible]
    if problematic:
        for d in problematic:
            print(f"  {d.pair}: B={d.B:+.6f}, A+C={d.A+d.C:+.6f} → need m<0")
    else:
        print("  None - all pairs allow positive m")

    # Final c at global m
    c_at_m = sum(d.contrib_at_m(m_global) for d in diagnostics)
    print()
    print(f"Total c at m_global={m_global:.4f}: {c_at_m:.6f} (target: {c_target})")

    return diagnostics, m_global


def main():
    parser = argparse.ArgumentParser(description="Per-pair mirror diagnostics (paper regime).")
    parser.add_argument("--n", type=int, default=N_QUAD, help="u/t quadrature points (default: 60)")
    parser.add_argument("--n-quad-a", type=int, default=N_QUAD_A, help="a quadrature points for Case C (default: 40)")
    parser.add_argument("--theta", type=float, default=THETA, help="theta (default: 4/7)")
    parser.add_argument(
        "--mirror-mode",
        choices=("r_flip", "exp_sign", "exp_sign_tflip"),
        default="r_flip",
        help=(
            "Mirror diagnostic mode: "
            "'r_flip' evaluates at -R (also flips Case C internal exponent); "
            "'exp_sign' flips ExpFactor signs (keeps Case C at +R); "
            "'exp_sign_tflip' also maps t -> 1-t inside ExpFactor arguments."
        ),
    )
    parser.add_argument(
        "--mirror-q-a0-shift",
        type=float,
        default=0.0,
        help="Diagnostic: shift the Q(...) AffineExpr constant term by this amount in the mirror run.",
    )
    parser.add_argument(
        "--no-factorial-normalization",
        action="store_true",
        help="Disable 1/(ℓ1!ℓ2!) normalization (diagnostic only).",
    )
    args = parser.parse_args()

    theta = float(args.theta)
    n_quad = int(args.n)
    n_quad_a = int(args.n_quad_a)
    mirror_mode = str(args.mirror_mode)
    q_a0_shift = float(args.mirror_q_a0_shift)
    use_factorial_normalization = not bool(args.no_factorial_normalization)

    print("=" * 90)
    print("PER-PAIR MIRROR DIAGNOSTIC ANALYSIS")
    print("Paper regime, K=3, d=1")
    print("=" * 90)
    print(f"Mode: {mirror_mode}  (q_a0_shift={q_a0_shift:+g})")
    print(f"theta={theta:.12f}, n={n_quad}, n_quad_a={n_quad_a}, factorial_norm={use_factorial_normalization}")
    print()

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Run analysis
    diag_k, _ = run_analysis(
        "κ",
        theta=theta,
        R=1.3036,
        c_target=2.137,
        polynomials=polys_k,
        n_quad=n_quad,
        n_quad_a=n_quad_a,
        mirror_mode=mirror_mode,
        q_a0_shift=q_a0_shift,
        use_factorial_normalization=use_factorial_normalization,
    )
    diag_s, _ = run_analysis(
        "κ*",
        theta=theta,
        R=1.1167,
        c_target=1.938,
        polynomials=polys_s,
        n_quad=n_quad,
        n_quad_a=n_quad_a,
        mirror_mode=mirror_mode,
        q_a0_shift=q_a0_shift,
        use_factorial_normalization=use_factorial_normalization,
    )

    # Cross-benchmark comparison
    print()
    print("=" * 90)
    print("CROSS-BENCHMARK COMPARISON")
    print("=" * 90)
    print()
    print(f"{'Pair':>6}  {'B_κ':>12}  {'B_κ*':>12}  {'ratio':>10}  {'Both B<0?':>10}")
    print("-" * 60)

    for dk, ds in zip(diag_k, diag_s):
        ratio = dk.B / ds.B if abs(ds.B) > 1e-15 else float('inf')
        both_neg = "YES" if dk.B < 0 and ds.B < 0 else "no"
        print(f"  {dk.pair:>4}  {dk.B:+12.6f}  {ds.B:+12.6f}  {ratio:10.4f}  {both_neg:>10}")

    print()
    print("KEY FINDING:")
    print("-" * 60)

    # Check if 12 pair has consistent negative B
    b12_k = next(d for d in diag_k if d.pair == "12").B
    b12_s = next(d for d in diag_s if d.pair == "12").B

    if b12_k < 0 and b12_s < 0:
        print("  Pair 12 has B<0 in BOTH benchmarks")
        print("  → The 12 pathology is NOT an R-specific artifact")
        print("  → Structural transform needed (swap and/or operator shift)")
    else:
        print("  Pair 12 B sign differs between benchmarks")
        print("  → R-dependent behavior")


if __name__ == "__main__":
    main()
