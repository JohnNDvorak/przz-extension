"""
run_convergence_sweep.py
Convergence & Stability Sweep for Two-Weight Mirror Model (GPT Guidance 2025-12-20)

Grid sweep over n and n_quad_a to verify (m1, m2) stability.
Stop condition: if m1/m2 drift > 2-3% between (60,40) and (80,80), not converged.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from src.evaluate import (
    evaluate_c_ordered,
    evaluate_c_ordered_with_exp_transform,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

THETA = 4.0 / 7.0

# Factorial weights
ORDERED_PAIR_KEYS = ("11", "22", "33", "12", "21", "13", "31", "23", "32")
FACTORIAL_WEIGHTS = {
    "11": 1.0,
    "22": 0.25,
    "33": 1.0/36,
    "12": 0.5,
    "21": 0.5,
    "13": 1.0/6,
    "31": 1.0/6,
    "23": 1.0/12,
    "32": 1.0/12,
}


@dataclass
class SweepPoint:
    """Result at one (n, n_quad_a) grid point."""
    n: int
    n_quad_a: int
    # κ benchmark
    I1_plus_k: float
    I2_plus_k: float
    I1_minus_k: float
    I2_minus_k: float
    S34_plus_k: float
    m_single_k: float
    # κ* benchmark
    I1_plus_s: float
    I2_plus_s: float
    I1_minus_s: float
    I2_minus_s: float
    S34_plus_s: float
    m_single_s: float
    # Two-weight solve
    m1: float
    m2: float
    det: float


def extract_split_channels(res, pairs=ORDERED_PAIR_KEYS):
    """Extract I1, I2, I3, I4 channels (weighted sums)."""
    I1 = I2 = I3 = I4 = 0.0
    for pair in pairs:
        w = FACTORIAL_WEIGHTS[pair]
        I1 += w * float(res.per_term.get(f"{pair}_I1_{pair}", 0.0))
        I2 += w * float(res.per_term.get(f"{pair}_I2_{pair}", 0.0))
        I3 += w * float(res.per_term.get(f"{pair}_I3_{pair}", 0.0))
        I4 += w * float(res.per_term.get(f"{pair}_I4_{pair}", 0.0))
    return I1, I2, I3, I4


def run_sweep_point(n: int, n_quad_a: int, polys_k: Dict, polys_s: Dict) -> SweepPoint:
    """Compute all metrics at one grid point."""

    # κ benchmark (R=1.3036)
    R_k = 1.3036
    c_target_k = 2.137

    direct_k = evaluate_c_ordered(
        theta=THETA, R=R_k, n=n, polynomials=polys_k,
        use_factorial_normalization=True, kernel_regime="paper", n_quad_a=n_quad_a,
    )
    mirror_k = evaluate_c_ordered_with_exp_transform(
        theta=THETA, R=-R_k, n=n, polynomials=polys_k,
        kernel_regime="paper", exp_scale_multiplier=1.0, exp_t_flip=False,
        q_a0_shift=0.0, use_factorial_normalization=True, n_quad_a=n_quad_a,
    )

    i1d_k, i2d_k, i3d_k, i4d_k = extract_split_channels(direct_k)
    i1m_k, i2m_k, _, _ = extract_split_channels(mirror_k)

    S12_plus_k = i1d_k + i2d_k
    S12_minus_k = i1m_k + i2m_k
    S34_plus_k = i3d_k + i4d_k

    m_single_k = (c_target_k - S12_plus_k - S34_plus_k) / S12_minus_k if abs(S12_minus_k) > 1e-15 else float('inf')

    # κ* benchmark (R=1.1167)
    R_s = 1.1167
    c_target_s = 1.938

    direct_s = evaluate_c_ordered(
        theta=THETA, R=R_s, n=n, polynomials=polys_s,
        use_factorial_normalization=True, kernel_regime="paper", n_quad_a=n_quad_a,
    )
    mirror_s = evaluate_c_ordered_with_exp_transform(
        theta=THETA, R=-R_s, n=n, polynomials=polys_s,
        kernel_regime="paper", exp_scale_multiplier=1.0, exp_t_flip=False,
        q_a0_shift=0.0, use_factorial_normalization=True, n_quad_a=n_quad_a,
    )

    i1d_s, i2d_s, i3d_s, i4d_s = extract_split_channels(direct_s)
    i1m_s, i2m_s, _, _ = extract_split_channels(mirror_s)

    S12_plus_s = i1d_s + i2d_s
    S12_minus_s = i1m_s + i2m_s
    S34_plus_s = i3d_s + i4d_s

    m_single_s = (c_target_s - S12_plus_s - S34_plus_s) / S12_minus_s if abs(S12_minus_s) > 1e-15 else float('inf')

    # Two-weight solve
    A = np.array([
        [i1m_k, i2m_k],
        [i1m_s, i2m_s],
    ])
    b = np.array([
        c_target_k - i1d_k - i2d_k - S34_plus_k,
        c_target_s - i1d_s - i2d_s - S34_plus_s,
    ])

    det = np.linalg.det(A)
    if abs(det) < 1e-15:
        m1, m2 = float('inf'), float('inf')
    else:
        m_vec = np.linalg.solve(A, b)
        m1, m2 = float(m_vec[0]), float(m_vec[1])

    return SweepPoint(
        n=n, n_quad_a=n_quad_a,
        I1_plus_k=i1d_k, I2_plus_k=i2d_k,
        I1_minus_k=i1m_k, I2_minus_k=i2m_k,
        S34_plus_k=S34_plus_k, m_single_k=m_single_k,
        I1_plus_s=i1d_s, I2_plus_s=i2d_s,
        I1_minus_s=i1m_s, I2_minus_s=i2m_s,
        S34_plus_s=S34_plus_s, m_single_s=m_single_s,
        m1=m1, m2=m2, det=det,
    )


def main():
    print("=" * 120)
    print("CONVERGENCE & STABILITY SWEEP FOR TWO-WEIGHT MIRROR MODEL")
    print("=" * 120)
    print()

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    polys_s = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Grid
    ns = [20, 30, 40, 60, 80]
    n_quad_as = [20, 40, 80]

    results: List[SweepPoint] = []

    total_points = len(ns) * len(n_quad_as)
    point_idx = 0

    for n in ns:
        for n_qa in n_quad_as:
            point_idx += 1
            print(f"Computing ({n}, {n_qa})... [{point_idx}/{total_points}]", flush=True)
            pt = run_sweep_point(n, n_qa, polys_k, polys_s)
            results.append(pt)

    # Print table
    print()
    print("=" * 120)
    print("STABILITY SWEEP RESULTS")
    print("=" * 120)
    print()
    print(f"{'n':>4} {'n_qa':>5} | {'m_single_κ':>12} {'m_single_κ*':>12} | {'m1':>10} {'m2':>10} {'m1/m2':>8} | {'det':>12} | notes")
    print("-" * 120)

    for pt in results:
        ratio = pt.m1 / pt.m2 if pt.m2 != 0 else float('inf')
        notes = ""
        if abs(pt.det) < 1e-8:
            notes = "SINGULAR"
        print(f"{pt.n:>4} {pt.n_quad_a:>5} | {pt.m_single_k:>12.6f} {pt.m_single_s:>12.6f} | {pt.m1:>10.4f} {pt.m2:>10.4f} {ratio:>8.4f} | {pt.det:>12.6e} | {notes}")

    # Stability check: compare (60,40) vs (80,80)
    print()
    print("=" * 120)
    print("STABILITY CHECK: (60,40) vs (80,80)")
    print("=" * 120)

    pt_60_40 = next(p for p in results if p.n == 60 and p.n_quad_a == 40)
    pt_80_80 = next(p for p in results if p.n == 80 and p.n_quad_a == 80)

    m1_drift = abs(pt_80_80.m1 - pt_60_40.m1) / abs(pt_60_40.m1) * 100 if pt_60_40.m1 != 0 else float('inf')
    m2_drift = abs(pt_80_80.m2 - pt_60_40.m2) / abs(pt_60_40.m2) * 100 if pt_60_40.m2 != 0 else float('inf')
    ms_k_drift = abs(pt_80_80.m_single_k - pt_60_40.m_single_k) / abs(pt_60_40.m_single_k) * 100 if pt_60_40.m_single_k != 0 else float('inf')
    ms_s_drift = abs(pt_80_80.m_single_s - pt_60_40.m_single_s) / abs(pt_60_40.m_single_s) * 100 if pt_60_40.m_single_s != 0 else float('inf')

    print()
    print(f"  m1:          (60,40)={pt_60_40.m1:.6f}  (80,80)={pt_80_80.m1:.6f}  drift={m1_drift:.2f}%")
    print(f"  m2:          (60,40)={pt_60_40.m2:.6f}  (80,80)={pt_80_80.m2:.6f}  drift={m2_drift:.2f}%")
    print(f"  m_single_κ:  (60,40)={pt_60_40.m_single_k:.6f}  (80,80)={pt_80_80.m_single_k:.6f}  drift={ms_k_drift:.2f}%")
    print(f"  m_single_κ*: (60,40)={pt_60_40.m_single_s:.6f}  (80,80)={pt_80_80.m_single_s:.6f}  drift={ms_s_drift:.2f}%")

    print()
    if max(m1_drift, m2_drift) < 3:
        print("RESULT: (m1, m2) are STABLE under quadrature refinement (<3% drift)")
        print("        → PROCEED with operator implementation")
    else:
        print("WARNING: (m1, m2) drift > 3% - NOT CONVERGED YET")
        print("        → INCREASE quadrature before operator work")

    # Channel values at (60,40)
    print()
    print("=" * 120)
    print("CHANNEL VALUES AT (n=60, n_quad_a=40)")
    print("=" * 120)
    pt = pt_60_40
    print()
    print("κ benchmark (R=1.3036):")
    print(f"  I1_plus  = {pt.I1_plus_k:+.8f}")
    print(f"  I2_plus  = {pt.I2_plus_k:+.8f}")
    print(f"  I1_minus = {pt.I1_minus_k:+.8f}")
    print(f"  I2_minus = {pt.I2_minus_k:+.8f}")
    print(f"  S34_plus = {pt.S34_plus_k:+.8f}")
    print()
    print("κ* benchmark (R=1.1167):")
    print(f"  I1_plus  = {pt.I1_plus_s:+.8f}")
    print(f"  I2_plus  = {pt.I2_plus_s:+.8f}")
    print(f"  I1_minus = {pt.I1_minus_s:+.8f}")
    print(f"  I2_minus = {pt.I2_minus_s:+.8f}")
    print(f"  S34_plus = {pt.S34_plus_s:+.8f}")


if __name__ == "__main__":
    main()
