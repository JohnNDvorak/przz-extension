"""
run_scale_audit.py
Scale audit for PRZZ K=3 (d=1) evaluation.

Purpose
-------
We currently have:
- kernel_regime="raw"   : legacy/diagnostic (P(u±x) everywhere)
- kernel_regime="paper" : ω-driven Case B/C (P2/P3 via Case C kernels)

The key symptom is:
- "paper" fixes κ/κ* ratio directionally,
- but absolute c values are ~10× too small vs PRZZ targets.

This script prints per-pair and per-I-term normalized contributions under both
regimes for κ and κ* benchmarks to localize where the scale gap lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import argparse
import math

from src.evaluate import EvaluationResult, evaluate_c_full
from src.evaluate import evaluate_c_full_with_exp_transform
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


PAIR_KEYS: Tuple[str, ...] = ("11", "22", "33", "12", "13", "23")

# Normalization factors used by evaluate_c_full (when use_factorial_normalization=True)
FACTORIAL_NORM: Dict[str, float] = {
    "11": 1.0,
    "22": 1.0 / 4.0,
    "33": 1.0 / 36.0,
    "12": 1.0 / 2.0,
    "13": 1.0 / 6.0,
    "23": 1.0 / 12.0,
}
SYMMETRY_FACTOR: Dict[str, float] = {
    "11": 1.0,
    "22": 1.0,
    "33": 1.0,
    "12": 2.0,
    "13": 2.0,
    "23": 2.0,
}


@dataclass(frozen=True)
class Bench:
    name: str
    R: float
    c_target: float


KAPPA = Bench(name="κ", R=1.3036, c_target=2.137)
KAPPA_STAR = Bench(name="κ*", R=1.1167, c_target=1.938)


def _norm_factor(pair: str, *, use_factorial_normalization: bool) -> float:
    base = SYMMETRY_FACTOR[pair]
    if use_factorial_normalization:
        base *= FACTORIAL_NORM[pair]
    return base


def _evaluate(
    *,
    theta: float,
    bench: Bench,
    polynomials: Dict[str, object],
    n_quad: int,
    n_quad_a: int,
    kernel_regime: str,
    use_factorial_normalization: bool,
) -> EvaluationResult:
    return evaluate_c_full(
        theta=theta,
        R=bench.R,
        n=n_quad,
        polynomials=polynomials,
        return_breakdown=True,
        use_factorial_normalization=use_factorial_normalization,
        mode="main",
        kernel_regime=kernel_regime,  # "raw" or "paper"
        n_quad_a=n_quad_a,
    )


def _pair_norm(res: EvaluationResult, pair: str, *, use_factorial_normalization: bool) -> float:
    raw_key = f"_c{pair}_raw"
    raw = float(res.per_term.get(raw_key, 0.0))
    return _norm_factor(pair, use_factorial_normalization=use_factorial_normalization) * raw


def _iterm_norm(res: EvaluationResult, pair: str, iterm: int, *, use_factorial_normalization: bool) -> float:
    key = f"I{iterm}_{pair}"
    raw = float(res.per_term.get(key, 0.0))
    return _norm_factor(pair, use_factorial_normalization=use_factorial_normalization) * raw


def _sum_iterms(res: EvaluationResult, *, use_factorial_normalization: bool) -> Dict[str, float]:
    totals: Dict[str, float] = {f"I{i}": 0.0 for i in range(1, 5)}
    for pair in PAIR_KEYS:
        for i in range(1, 5):
            totals[f"I{i}"] += _iterm_norm(res, pair, i, use_factorial_normalization=use_factorial_normalization)
    return totals


def _top_terms(
    res_raw: EvaluationResult,
    res_paper: EvaluationResult,
    *,
    use_factorial_normalization: bool,
    top_n: int,
) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str, float, float]] = []
    for pair in PAIR_KEYS:
        for i in range(1, 5):
            key = f"I{i}_{pair}"
            v_raw = _iterm_norm(res_raw, pair, i, use_factorial_normalization=use_factorial_normalization)
            v_paper = _iterm_norm(res_paper, pair, i, use_factorial_normalization=use_factorial_normalization)
            rows.append((key, v_raw, v_paper))
    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    return rows[:top_n]


def _print_benchmark(
    *,
    bench: Bench,
    theta: float,
    polynomials: Dict[str, object],
    n_quad: int,
    n_quad_a: int,
    use_factorial_normalization: bool,
    mirror_mode: str,
    mirror_q_a0_shift: float,
) -> Dict[str, float]:
    print()
    print("=" * 78)
    print(f"SCALE AUDIT: {bench.name} (R={bench.R})")
    print("=" * 78)

    res_raw = _evaluate(
        theta=theta,
        bench=bench,
        polynomials=polynomials,
        n_quad=n_quad,
        n_quad_a=n_quad_a,
        kernel_regime="raw",
        use_factorial_normalization=use_factorial_normalization,
    )
    res_paper = _evaluate(
        theta=theta,
        bench=bench,
        polynomials=polynomials,
        n_quad=n_quad,
        n_quad_a=n_quad_a,
        kernel_regime="paper",
        use_factorial_normalization=use_factorial_normalization,
    )

    c_raw = float(res_raw.total)
    c_paper = float(res_paper.total)
    print(f"c_target: {bench.c_target:.12f}")
    print(f"c_raw:    {c_raw:.12f}    (scale_needed={bench.c_target / c_raw:.6f})")
    print(f"c_paper:  {c_paper:.12f}    (scale_needed={bench.c_target / c_paper:.6f})")
    print(f"raw/paper: {c_raw / c_paper:.6f}")

    # Optional: mirror diagnostics
    if mirror_mode != "none":
        print()
        print("Paper-regime mirror component (diagnostic):")

        if mirror_mode == "r_flip":
            res_m = evaluate_c_full_with_exp_transform(
                theta=theta,
                R=-bench.R,
                n=n_quad,
                polynomials=polynomials,
                kernel_regime="paper",
                exp_scale_multiplier=1.0,
                exp_t_flip=False,
                q_a0_shift=mirror_q_a0_shift,
                return_breakdown=True,
                use_factorial_normalization=use_factorial_normalization,
                mode="main",
                n_quad_a=n_quad_a,
            )
            c_mirror = float(res_m.total)
            mult_default = math.exp(2.0 * bench.R)
            extra = f", q_a0_shift={mirror_q_a0_shift:+g}" if mirror_q_a0_shift != 0.0 else ""
            print(f"  mode: r_flip{extra}  (paper at -R; flips Case C internal exponent too)")
        elif mirror_mode in ("exp_sign", "exp_sign_tflip"):
            exp_t_flip = mirror_mode == "exp_sign_tflip"
            res_m = evaluate_c_full_with_exp_transform(
                theta=theta,
                R=bench.R,
                n=n_quad,
                polynomials=polynomials,
                kernel_regime="paper",
                exp_scale_multiplier=-1.0,
                exp_t_flip=exp_t_flip,
                q_a0_shift=mirror_q_a0_shift,
                return_breakdown=True,
                use_factorial_normalization=use_factorial_normalization,
                mode="main",
                n_quad_a=n_quad_a,
            )
            c_mirror = float(res_m.total)
            mult_default = math.exp(2.0 * bench.R)
            extra = f", q_a0_shift={mirror_q_a0_shift:+g}" if mirror_q_a0_shift != 0.0 else ""
            print(f"  mode: {mirror_mode}{extra}  (negates ExpFactor scales; Case C sees +R)")
        else:
            raise ValueError(f"Unknown mirror_mode: {mirror_mode!r}")

        # Solve for multiplier needed to hit target: c_target ≈ c_paper + m * c_mirror
        if c_mirror != 0.0:
            m_needed = (bench.c_target - c_paper) / c_mirror
        else:
            m_needed = float("inf")

        print(f"  c_direct:  {c_paper:+.12f}")
        print(f"  c_mirror:  {c_mirror:+.12f}")
        print(f"  exp(2R):   {mult_default:+.12f}")
        print(f"  m_needed:  {m_needed:+.12f}")
        print(f"  recomb(exp(2R)): {c_paper + mult_default * c_mirror:+.12f}")

        # Group-level audit: I12 vs I34 contributions under direct and mirror.
        direct_i = _sum_iterms(res_paper, use_factorial_normalization=use_factorial_normalization)
        mirror_i = _sum_iterms(res_m, use_factorial_normalization=use_factorial_normalization)
        direct_12 = direct_i["I1"] + direct_i["I2"]
        direct_34 = direct_i["I3"] + direct_i["I4"]
        mirror_12 = mirror_i["I1"] + mirror_i["I2"]
        mirror_34 = mirror_i["I3"] + mirror_i["I4"]

        print()
        print("  Group audit (normalized totals):")
        print("    group     direct             mirror             direct + exp(2R)*mirror")
        print(f"    I1+I2: {direct_12:+18.12f}  {mirror_12:+18.12f}  {direct_12 + mult_default * mirror_12:+18.12f}")
        print(f"    I3+I4: {direct_34:+18.12f}  {mirror_34:+18.12f}  {direct_34 + mult_default * mirror_34:+18.12f}")
        print(f"    total: {c_paper:+18.12f}  {c_mirror:+18.12f}  {c_paper + mult_default * c_mirror:+18.12f}")

        # “If-only” multipliers: what m would be needed if mirroring applied only to one group?
        if mirror_12 != 0.0:
            m_only_12 = (bench.c_target - c_paper) / mirror_12
        else:
            m_only_12 = float("inf")
        if mirror_34 != 0.0:
            m_only_34 = (bench.c_target - c_paper) / mirror_34
        else:
            m_only_34 = float("inf")
        print("    m_needed if mirror applies only to:")
        print(f"      (I1+I2): {m_only_12:+.12f}")
        print(f"      (I3+I4): {m_only_34:+.12f}")

        print()
        print("  Per-pair group audit (normalized contributions to total c):")
        print("    pair        dir(I1+I2)         dir(I3+I4)      mir(I1+I2)        mir(I3+I4)")
        for pair in PAIR_KEYS:
            d12 = _iterm_norm(res_paper, pair, 1, use_factorial_normalization=use_factorial_normalization) + _iterm_norm(
                res_paper, pair, 2, use_factorial_normalization=use_factorial_normalization
            )
            d34 = _iterm_norm(res_paper, pair, 3, use_factorial_normalization=use_factorial_normalization) + _iterm_norm(
                res_paper, pair, 4, use_factorial_normalization=use_factorial_normalization
            )
            m12 = _iterm_norm(res_m, pair, 1, use_factorial_normalization=use_factorial_normalization) + _iterm_norm(
                res_m, pair, 2, use_factorial_normalization=use_factorial_normalization
            )
            m34 = _iterm_norm(res_m, pair, 3, use_factorial_normalization=use_factorial_normalization) + _iterm_norm(
                res_m, pair, 4, use_factorial_normalization=use_factorial_normalization
            )
            print(f"    {pair}: {d12:+18.12f}  {d34:+18.12f}  {m12:+18.12f}  {m34:+18.12f}")

        print()
        print("  Per-pair recombination with exp(2R) (diagnostic):")
        print("    pair         (I1+I2) combined        (I3+I4) combined            total combined")
        for pair in PAIR_KEYS:
            d12 = _iterm_norm(res_paper, pair, 1, use_factorial_normalization=use_factorial_normalization) + _iterm_norm(
                res_paper, pair, 2, use_factorial_normalization=use_factorial_normalization
            )
            d34 = _iterm_norm(res_paper, pair, 3, use_factorial_normalization=use_factorial_normalization) + _iterm_norm(
                res_paper, pair, 4, use_factorial_normalization=use_factorial_normalization
            )
            m12 = _iterm_norm(res_m, pair, 1, use_factorial_normalization=use_factorial_normalization) + _iterm_norm(
                res_m, pair, 2, use_factorial_normalization=use_factorial_normalization
            )
            m34 = _iterm_norm(res_m, pair, 3, use_factorial_normalization=use_factorial_normalization) + _iterm_norm(
                res_m, pair, 4, use_factorial_normalization=use_factorial_normalization
            )
            c12 = d12 + mult_default * m12
            c34 = d34 + mult_default * m34
            print(f"    {pair}: {c12:+22.12f}  {c34:+22.12f}  {c12 + c34:+22.12f}")

    # Per-pair totals (normalized)
    print()
    print("Per-pair normalized contributions (to total c):")
    print("  pair        raw               paper             paper/raw")
    for pair in PAIR_KEYS:
        pr = _pair_norm(res_raw, pair, use_factorial_normalization=use_factorial_normalization)
        pp = _pair_norm(res_paper, pair, use_factorial_normalization=use_factorial_normalization)
        ratio = (pp / pr) if pr != 0.0 else float("inf")
        print(f"  {pair}: {pr:+18.12f}  {pp:+18.12f}  {ratio:9.6f}")

    # Per-I-term totals (normalized)
    print()
    print("Per-I-term normalized totals (summed over all pairs):")
    raw_i = _sum_iterms(res_raw, use_factorial_normalization=use_factorial_normalization)
    pap_i = _sum_iterms(res_paper, use_factorial_normalization=use_factorial_normalization)
    print("  term        raw               paper             paper/raw")
    for i in range(1, 5):
        key = f"I{i}"
        r = raw_i[key]
        p = pap_i[key]
        ratio = (p / r) if r != 0.0 else float("inf")
        print(f"  {key}: {r:+18.12f}  {p:+18.12f}  {ratio:9.6f}")

    # Sanity check: I1..I4 sums should reproduce totals (within fp tolerance)
    sum_raw = sum(raw_i.values())
    sum_paper = sum(pap_i.values())
    print()
    print("Sanity (I1..I4 sum vs c):")
    print(f"  raw   sum(I1..I4)={sum_raw:.12f}  c_raw={c_raw:.12f}  diff={sum_raw - c_raw:+.3e}")
    print(f"  paper sum(I1..I4)={sum_paper:.12f}  c_paper={c_paper:.12f}  diff={sum_paper - c_paper:+.3e}")

    # Top terms (by |paper contribution|)
    print()
    print("Top term contributions by |paper| (normalized):")
    print("  term            raw                paper              paper/raw")
    for key, v_raw, v_paper in _top_terms(
        res_raw,
        res_paper,
        use_factorial_normalization=use_factorial_normalization,
        top_n=12,
    ):
        ratio = (v_paper / v_raw) if v_raw != 0.0 else float("inf")
        print(f"  {key:8s} {v_raw:+18.12f}  {v_paper:+18.12f}  {ratio:9.6f}")

    return {
        "c_raw": c_raw,
        "c_paper": c_paper,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PRZZ scale audit (raw vs paper regimes).")
    parser.add_argument("--n", type=int, default=60, help="u/t quadrature points (default: 60)")
    parser.add_argument("--n-quad-a", type=int, default=40, help="a quadrature points for Case C (default: 40)")
    parser.add_argument("--theta", type=float, default=4.0 / 7.0, help="theta (default: 4/7)")
    parser.add_argument(
        "--no-factorial-normalization",
        action="store_true",
        help="Disable 1/(ℓ1!ℓ2!) normalization (diagnostic only).",
    )
    parser.add_argument(
        "--mirror-mode",
        choices=("none", "exp_sign", "exp_sign_tflip", "r_flip"),
        default="none",
        help=(
            "Mirror diagnostic mode: "
            "'exp_sign' flips ExpFactor signs (keeps Case C at +R); "
            "'r_flip' evaluates at -R (also flips Case C internal exponent)."
        ),
    )
    parser.add_argument(
        "--mirror-q-a0-shift",
        type=float,
        default=0.0,
        help=(
            "Diagnostic: shift the Q(...) AffineExpr constant term by this amount "
            "when computing the mirror component (tests Q(D)->Q(D+shift) hypotheses)."
        ),
    )
    args = parser.parse_args()

    theta = float(args.theta)
    n_quad = int(args.n)
    n_quad_a = int(args.n_quad_a)
    use_factorial_normalization = not bool(args.no_factorial_normalization)

    # κ benchmark polys
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # κ* benchmark polys
    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    out_k = _print_benchmark(
        bench=KAPPA,
        theta=theta,
        polynomials=polys_k,
        n_quad=n_quad,
        n_quad_a=n_quad_a,
        use_factorial_normalization=use_factorial_normalization,
        mirror_mode=str(args.mirror_mode),
        mirror_q_a0_shift=float(args.mirror_q_a0_shift),
    )
    out_s = _print_benchmark(
        bench=KAPPA_STAR,
        theta=theta,
        polynomials=polys_s,
        n_quad=n_quad,
        n_quad_a=n_quad_a,
        use_factorial_normalization=use_factorial_normalization,
        mirror_mode=str(args.mirror_mode),
        mirror_q_a0_shift=float(args.mirror_q_a0_shift),
    )

    ratio_raw = out_k["c_raw"] / out_s["c_raw"]
    ratio_paper = out_k["c_paper"] / out_s["c_paper"]
    target_ratio = KAPPA.c_target / KAPPA_STAR.c_target

    print()
    print("=" * 78)
    print("RATIO SUMMARY")
    print("=" * 78)
    print(f"target ratio: {target_ratio:.12f}")
    print(f"raw ratio:    {ratio_raw:.12f}")
    print(f"paper ratio:  {ratio_paper:.12f}")
    print()


if __name__ == "__main__":
    main()
