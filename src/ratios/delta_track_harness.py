"""
src/ratios/delta_track_harness.py
Phase 14G/14I: Delta tracking diagnostic harness.

Provides:
1. DeltaMetrics dataclass with (A, D, delta, B_over_A)
2. DeltaMetricsExtended with attribution (I12_plus, I12_minus, I34_plus, delta_s12, delta_s34)
3. Mode comparison (raw vs pole_cancelled)
4. Mode sweep for single benchmark
5. Swap matrix (2×2 R, poly experiment from Phase 14F)
6. JSON output to artifacts/
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json
import os

from src.ratios.przz_polynomials import load_przz_k3_polynomials
from src.ratios.j1_euler_maclaurin import (
    compute_m1_with_mirror_assembly, LaurentMode
)


@dataclass(frozen=True)
class DeltaMetrics:
    """Delta metrics for a single computation (Phase 14G)."""
    benchmark: str
    laurent_mode: str
    R: float
    A: float
    D: float
    delta: float
    B_over_A: float


@dataclass(frozen=True)
class DeltaMetricsExtended:
    """Extended delta metrics with attribution (Phase 14I).

    Provides breakdown of which component (S12 or S34) dominates delta.

    Phase 17B: Added per-piece breakdown for J11/J12/J13/J14/J15.
    """
    benchmark: str
    laurent_mode: str
    R: float
    # Core metrics
    A: float              # = I12_minus (mirror coefficient)
    D: float              # = I12_plus + I34_plus
    delta: float          # = D / A
    B_over_A: float       # = (2K-1) + delta
    # Attribution breakdown
    I12_plus: float       # Sum of j11, j12, j15 at +R
    I12_minus: float      # Sum of j11, j12, j15 at -R (= A)
    I34_plus: float       # Sum of j13, j14 at +R
    delta_s12: float      # I12_plus / I12_minus
    delta_s34: float      # I34_plus / I12_minus
    # Phase 17B: Per-piece breakdown
    j11_plus: float = 0.0
    j12_plus: float = 0.0
    j15_plus: float = 0.0
    j13_plus: float = 0.0
    j14_plus: float = 0.0
    j11_minus: float = 0.0
    j12_minus: float = 0.0
    j15_minus: float = 0.0


def compute_delta_metrics(
    benchmark: str,
    laurent_mode: LaurentMode,
    theta: float = 4.0 / 7.0,
    K: int = 3,
) -> DeltaMetrics:
    """Compute delta metrics for given benchmark and mode (Phase 14G API)."""
    polys = load_przz_k3_polynomials(benchmark)
    decomp = compute_m1_with_mirror_assembly(
        theta=theta, R=polys.R, polys=polys, K=K,
        laurent_mode=laurent_mode,
    )
    return DeltaMetrics(
        benchmark=benchmark,
        laurent_mode=laurent_mode.value,
        R=polys.R,
        A=decomp["exp_coefficient"],
        D=decomp["D"],
        delta=decomp["delta"],
        B_over_A=decomp["B_over_A"],
    )


def compute_delta_metrics_extended(
    benchmark: str,
    laurent_mode: LaurentMode,
    theta: float = 4.0 / 7.0,
    K: int = 3,
) -> DeltaMetricsExtended:
    """Compute extended delta metrics with attribution (Phase 14I API).

    Phase 17B: Now includes per-piece breakdown (j11, j12, j13, j14, j15).

    Args:
        benchmark: Either "kappa" or "kappa_star"
        laurent_mode: LaurentMode enum value
        theta: θ parameter (default 4/7)
        K: Number of mollifier pieces (default 3)

    Returns:
        DeltaMetricsExtended with full attribution breakdown including per-piece
    """
    polys = load_przz_k3_polynomials(benchmark)
    decomp = compute_m1_with_mirror_assembly(
        theta=theta, R=polys.R, polys=polys, K=K,
        laurent_mode=laurent_mode,
    )

    I12_plus = decomp["i12_plus_total"]
    I12_minus = decomp["i12_minus_total"]
    I34_plus = decomp["i34_plus_total"]

    # Compute attribution ratios (delta = delta_s12 + delta_s34)
    delta_s12 = I12_plus / I12_minus if abs(I12_minus) > 1e-14 else float('inf')
    delta_s34 = I34_plus / I12_minus if abs(I12_minus) > 1e-14 else float('inf')

    # Phase 17B: Extract per-piece values
    i12_plus_pieces = decomp["i12_plus_pieces"]
    i12_minus_pieces = decomp["i12_minus_pieces"]
    i34_plus_pieces = decomp["i34_plus_pieces"]

    return DeltaMetricsExtended(
        benchmark=benchmark,
        laurent_mode=laurent_mode.value,
        R=polys.R,
        A=decomp["exp_coefficient"],
        D=decomp["D"],
        delta=decomp["delta"],
        B_over_A=decomp["B_over_A"],
        I12_plus=I12_plus,
        I12_minus=I12_minus,
        I34_plus=I34_plus,
        delta_s12=delta_s12,
        delta_s34=delta_s34,
        # Phase 17B: Per-piece breakdown
        j11_plus=i12_plus_pieces.get("j11", 0.0),
        j12_plus=i12_plus_pieces.get("j12", 0.0),
        j15_plus=i12_plus_pieces.get("j15", 0.0),
        j13_plus=i34_plus_pieces.get("j13", 0.0),
        j14_plus=i34_plus_pieces.get("j14", 0.0),
        j11_minus=i12_minus_pieces.get("j11", 0.0),
        j12_minus=i12_minus_pieces.get("j12", 0.0),
        j15_minus=i12_minus_pieces.get("j15", 0.0),
    )


def run_mode_comparison(verbose: bool = True) -> Dict:
    """Compare raw_logderiv vs pole_cancelled for both benchmarks.

    Args:
        verbose: If True, print formatted comparison table

    Returns:
        Dictionary mapping "{benchmark}_{mode}" to DeltaMetrics
    """
    results = {}

    for benchmark in ["kappa", "kappa_star"]:
        for mode in [LaurentMode.RAW_LOGDERIV, LaurentMode.POLE_CANCELLED]:
            key = f"{benchmark}_{mode.value}"
            results[key] = compute_delta_metrics(benchmark, mode)

    if verbose:
        print("=" * 80)
        print("PHASE 14G: Laurent Mode Comparison")
        print("=" * 80)
        print()
        print(f"{'Benchmark':<12} {'Mode':<18} {'delta':<10} {'B/A':<10} {'Gap':<10}")
        print("-" * 60)

        for key, m in results.items():
            gap = (m.B_over_A - 5.0) / 5.0 * 100
            print(f"{m.benchmark:<12} {m.laurent_mode:<18} {m.delta:<10.4f} "
                  f"{m.B_over_A:<10.4f} {gap:+.2f}%")

        print("-" * 60)
        print()

        # Summary
        raw_k = results["kappa_raw_logderiv"]
        raw_ks = results["kappa_star_raw_logderiv"]
        pc_k = results["kappa_pole_cancelled"]
        pc_ks = results["kappa_star_pole_cancelled"]

        print("SUMMARY:")
        print(f"  Raw mode:          κ delta={raw_k.delta:.4f}, κ* delta={raw_ks.delta:.4f}")
        print(f"  Pole-cancelled:    κ delta={pc_k.delta:.4f}, κ* delta={pc_ks.delta:.4f}")
        print()

        # Did pole_cancelled reduce delta?
        k_improved = pc_k.delta < raw_k.delta
        ks_improved = pc_ks.delta < raw_ks.delta

        if k_improved and ks_improved:
            print("  ✓ Pole-cancelled mode reduces delta for BOTH benchmarks")
        elif k_improved:
            print("  ⚠ Pole-cancelled reduces delta for κ only")
        elif ks_improved:
            print("  ⚠ Pole-cancelled reduces delta for κ* only")
        else:
            print("  ✗ Pole-cancelled mode does NOT reduce delta")

        print("=" * 80)

    return results


def run_mode_sweep(benchmark: str, verbose: bool = True) -> Dict[str, DeltaMetricsExtended]:
    """Sweep both Laurent modes for a single benchmark (Phase 14I).

    Args:
        benchmark: Either "kappa" or "kappa_star"
        verbose: If True, print comparison table

    Returns:
        Dictionary mapping mode name to DeltaMetricsExtended
    """
    results = {}

    for mode in [LaurentMode.RAW_LOGDERIV, LaurentMode.POLE_CANCELLED]:
        results[mode.value] = compute_delta_metrics_extended(benchmark, mode)

    if verbose:
        print("=" * 80)
        print(f"MODE SWEEP: {benchmark.upper()}")
        print("=" * 80)
        print()
        print(f"{'Mode':<18} {'I12+':<10} {'I12-':<10} {'I34+':<10} {'δ_s12':<10} {'δ_s34':<10} {'delta':<10}")
        print("-" * 78)

        for mode_name, m in results.items():
            print(f"{mode_name:<18} {m.I12_plus:<10.4f} {m.I12_minus:<10.4f} "
                  f"{m.I34_plus:<10.4f} {m.delta_s12:<10.4f} {m.delta_s34:<10.4f} "
                  f"{m.delta:<10.4f}")

        print("=" * 80)

    return results


def run_swap_matrix(verbose: bool = True) -> Dict[str, Dict]:
    """Run 2×2 (R, polynomial) swap experiment (Phase 14F/14I).

    Tests whether delta is R-driven or polynomial-driven by swapping
    R and polynomial configurations.

    Returns:
        Dictionary with experiment results
    """
    polys_k = load_przz_k3_polynomials("kappa")
    polys_ks = load_przz_k3_polynomials("kappa_star")

    R_k = polys_k.R
    R_ks = polys_ks.R

    configs = [
        {"R": R_k, "R_name": "κ R", "polys": polys_k, "poly_name": "κ polys"},
        {"R": R_k, "R_name": "κ R", "polys": polys_ks, "poly_name": "κ* polys"},
        {"R": R_ks, "R_name": "κ* R", "polys": polys_k, "poly_name": "κ polys"},
        {"R": R_ks, "R_name": "κ* R", "polys": polys_ks, "poly_name": "κ* polys"},
    ]

    results = []
    for cfg in configs:
        decomp = compute_m1_with_mirror_assembly(
            theta=4.0/7.0, R=cfg["R"], polys=cfg["polys"], K=3,
            laurent_mode=LaurentMode.RAW_LOGDERIV,
        )
        results.append({
            "R_name": cfg["R_name"],
            "R": cfg["R"],
            "poly_name": cfg["poly_name"],
            "A": decomp["exp_coefficient"],
            "D": decomp["D"],
            "delta": decomp["delta"],
            "B_over_A": decomp["B_over_A"],
            "I12_plus": decomp["i12_plus_total"],
            "I12_minus": decomp["i12_minus_total"],
            "I34_plus": decomp["i34_plus_total"],
        })

    if verbose:
        print("=" * 80)
        print("SWAP MATRIX: 2×2 (R, Polynomial) Experiment")
        print("=" * 80)
        print()
        print(f"{'R':<10} {'Polys':<12} {'delta':<10} {'B/A':<10} {'I12+':<10} {'I12-':<10} {'I34+':<10}")
        print("-" * 72)

        for r in results:
            print(f"{r['R_name']:<10} {r['poly_name']:<12} {r['delta']:<10.4f} "
                  f"{r['B_over_A']:<10.4f} {r['I12_plus']:<10.4f} "
                  f"{r['I12_minus']:<10.4f} {r['I34_plus']:<10.4f}")

        print("-" * 72)
        print()

        # Compute R vs poly effects
        poly_effect = (abs(results[1]["delta"] - results[0]["delta"]) +
                       abs(results[3]["delta"] - results[2]["delta"])) / 2
        R_effect = (abs(results[2]["delta"] - results[0]["delta"]) +
                    abs(results[3]["delta"] - results[1]["delta"])) / 2

        print(f"Polynomial effect on delta: {poly_effect:.4f}")
        print(f"R effect on delta:          {R_effect:.4f}")
        print(f"R/Poly ratio:               {R_effect/poly_effect:.1f}x" if poly_effect > 0.001 else "R/Poly ratio: inf")
        print("=" * 80)

    return {"results": results, "R_k": R_k, "R_ks": R_ks}


def save_delta_report(filename: str = None) -> str:
    """Save comprehensive delta report to JSON (Phase 14I).

    Args:
        filename: Output filename (default: artifacts/delta_report.json)

    Returns:
        Path to saved file
    """
    if filename is None:
        os.makedirs("artifacts", exist_ok=True)
        filename = "artifacts/delta_report.json"

    report = {
        "phase": "14I",
        "description": "Delta tracking diagnostic report",
    }

    # Mode comparison with extended metrics
    mode_results = {}
    for benchmark in ["kappa", "kappa_star"]:
        mode_results[benchmark] = {}
        for mode in [LaurentMode.RAW_LOGDERIV, LaurentMode.POLE_CANCELLED]:
            metrics = compute_delta_metrics_extended(benchmark, mode)
            mode_results[benchmark][mode.value] = asdict(metrics)
    report["mode_comparison"] = mode_results

    # Swap matrix
    swap_result = run_swap_matrix(verbose=False)
    report["swap_matrix"] = swap_result["results"]

    with open(filename, "w") as f:
        json.dump(report, f, indent=2)

    return filename


def print_extended_comparison(verbose: bool = True) -> Dict:
    """Print extended comparison with attribution (Phase 14I main output)."""
    print("=" * 90)
    print("PHASE 14I: Extended Delta Comparison with Attribution")
    print("=" * 90)
    print()

    results = {}
    for benchmark in ["kappa", "kappa_star"]:
        print(f"\n--- {benchmark.upper()} ---")
        metrics = compute_delta_metrics_extended(benchmark, LaurentMode.RAW_LOGDERIV)
        results[benchmark] = metrics

        print(f"  R = {metrics.R:.4f}")
        print(f"  A (I12_minus) = {metrics.A:.6f}")
        print(f"  D = {metrics.D:.6f}")
        print(f"  delta = D/A = {metrics.delta:.6f}")
        print(f"  B/A = {metrics.B_over_A:.6f} (gap: {(metrics.B_over_A - 5)/5*100:+.2f}%)")
        print()
        print("  Attribution:")
        print(f"    I12_plus  = {metrics.I12_plus:.6f}  → δ_s12 = {metrics.delta_s12:.6f}")
        print(f"    I34_plus  = {metrics.I34_plus:.6f}  → δ_s34 = {metrics.delta_s34:.6f}")
        print(f"    I12_minus = {metrics.I12_minus:.6f}")
        print(f"    Check: δ_s12 + δ_s34 = {metrics.delta_s12 + metrics.delta_s34:.6f} (should ≈ {metrics.delta:.6f})")

    print("\n" + "=" * 90)
    return results


def compare_laurent_modes_per_piece(
    benchmark: str = None,
    verbose: bool = True
) -> Dict:
    """Phase 17B: Compare RAW vs ACTUAL laurent mode per-piece to identify asymmetry source.

    This diagnostic identifies which J-piece (j11, j12, j13, j14, j15) is responsible
    for the κ vs κ* asymmetry observed in Phase 16.

    Args:
        benchmark: If None, runs both "kappa" and "kappa_star"
        verbose: If True, print comparison table

    Returns:
        Dictionary with per-piece differences between RAW and ACTUAL modes
    """
    benchmarks = [benchmark] if benchmark else ["kappa", "kappa_star"]
    results = {}

    for bm in benchmarks:
        raw = compute_delta_metrics_extended(bm, LaurentMode.RAW_LOGDERIV)
        actual = compute_delta_metrics_extended(bm, LaurentMode.ACTUAL_LOGDERIV)

        # Compute per-piece deltas (ACTUAL - RAW)
        piece_deltas = {
            "j11_plus": actual.j11_plus - raw.j11_plus,
            "j12_plus": actual.j12_plus - raw.j12_plus,
            "j15_plus": actual.j15_plus - raw.j15_plus,
            "j13_plus": actual.j13_plus - raw.j13_plus,
            "j14_plus": actual.j14_plus - raw.j14_plus,
            "j11_minus": actual.j11_minus - raw.j11_minus,
            "j12_minus": actual.j12_minus - raw.j12_minus,
            "j15_minus": actual.j15_minus - raw.j15_minus,
        }

        # Sum deltas for attribution
        delta_i12_plus = (piece_deltas["j11_plus"] +
                         piece_deltas["j12_plus"] +
                         piece_deltas["j15_plus"])
        delta_i12_minus = (piece_deltas["j11_minus"] +
                          piece_deltas["j12_minus"] +
                          piece_deltas["j15_minus"])
        delta_i34_plus = piece_deltas["j13_plus"] + piece_deltas["j14_plus"]

        results[bm] = {
            "raw": raw,
            "actual": actual,
            "piece_deltas": piece_deltas,
            "delta_i12_plus": delta_i12_plus,
            "delta_i12_minus": delta_i12_minus,
            "delta_i34_plus": delta_i34_plus,
            "delta_B_over_A": actual.B_over_A - raw.B_over_A,
        }

    if verbose:
        print("=" * 100)
        print("PHASE 17B: Per-Piece Laurent Mode Comparison (RAW vs ACTUAL)")
        print("=" * 100)
        print()
        print("GOAL: Identify which J-piece causes the κ vs κ* asymmetry in Phase 16")
        print()

        for bm in benchmarks:
            res = results[bm]
            raw = res["raw"]
            actual = res["actual"]
            pd = res["piece_deltas"]

            print(f"--- {bm.upper()} (R={raw.R:.4f}) ---")
            print()
            print(f"{'Piece':<12} {'RAW':<12} {'ACTUAL':<12} {'Δ (ACT-RAW)':<14} {'% Change':<10}")
            print("-" * 60)

            # +R pieces
            for piece in ["j11_plus", "j12_plus", "j15_plus", "j13_plus", "j14_plus"]:
                raw_val = getattr(raw, piece)
                act_val = getattr(actual, piece)
                delta = pd[piece]
                pct = (delta / raw_val * 100) if abs(raw_val) > 1e-10 else 0.0
                print(f"{piece:<12} {raw_val:<12.6f} {act_val:<12.6f} {delta:<+14.6f} {pct:>+8.1f}%")

            print()
            print(f"  I12(+R) Δ: {res['delta_i12_plus']:+.6f}")
            print(f"  I34(+R) Δ: {res['delta_i34_plus']:+.6f}")
            print()

            # -R pieces (mirror)
            print(f"{'Piece':<12} {'RAW':<12} {'ACTUAL':<12} {'Δ (ACT-RAW)':<14} {'% Change':<10}")
            print("-" * 60)
            for piece in ["j11_minus", "j12_minus", "j15_minus"]:
                raw_val = getattr(raw, piece)
                act_val = getattr(actual, piece)
                delta = pd[piece]
                pct = (delta / raw_val * 100) if abs(raw_val) > 1e-10 else 0.0
                print(f"{piece:<12} {raw_val:<12.6f} {act_val:<12.6f} {delta:<+14.6f} {pct:>+8.1f}%")

            print()
            print(f"  I12(-R) Δ: {res['delta_i12_minus']:+.6f}")
            print()

            # B/A summary
            print(f"  B/A change: {raw.B_over_A:.4f} → {actual.B_over_A:.4f} "
                  f"(Δ = {res['delta_B_over_A']:+.4f})")
            gap_raw = (raw.B_over_A - 5.0) / 5.0 * 100
            gap_actual = (actual.B_over_A - 5.0) / 5.0 * 100
            print(f"  Gap from 5: {gap_raw:+.2f}% → {gap_actual:+.2f}%")
            print()

        # Cross-benchmark comparison
        if len(benchmarks) == 2:
            print("=" * 60)
            print("CROSS-BENCHMARK ASYMMETRY ANALYSIS")
            print("=" * 60)
            k = results["kappa"]
            ks = results["kappa_star"]

            print(f"\n{'Piece':<12} {'κ Δ':<14} {'κ* Δ':<14} {'Ratio κ/κ*':<12}")
            print("-" * 54)
            for piece in ["j11_plus", "j12_plus", "j15_plus", "j13_plus", "j14_plus"]:
                dk = k["piece_deltas"][piece]
                dks = ks["piece_deltas"][piece]
                ratio = dk / dks if abs(dks) > 1e-10 else float('inf')
                print(f"{piece:<12} {dk:<+14.6f} {dks:<+14.6f} {ratio:<12.2f}")

            print()
            print("Interpretation:")
            print("  - If ratio ≈ 1.0: Piece affects both benchmarks similarly")
            print("  - If ratio >> 1 or << 1: Piece causes asymmetry")
            print()

            # Find the most asymmetric piece
            max_asym = 0
            asym_piece = None
            for piece in ["j11_plus", "j12_plus", "j15_plus", "j13_plus", "j14_plus"]:
                dk = abs(k["piece_deltas"][piece])
                dks = abs(ks["piece_deltas"][piece])
                if dks > 1e-10 and dk > 1e-10:
                    ratio = dk / dks
                    asym = abs(ratio - 1.0)
                    if asym > max_asym:
                        max_asym = asym
                        asym_piece = piece

            if asym_piece:
                print(f"  Most asymmetric piece: {asym_piece} (ratio deviation: {max_asym:.2f})")
            print()

        print("=" * 100)

    return results


if __name__ == "__main__":
    # Run extended comparison
    print_extended_comparison()
    print()

    # Run swap matrix
    run_swap_matrix()
    print()

    # Save report
    path = save_delta_report()
    print(f"Report saved to: {path}")
