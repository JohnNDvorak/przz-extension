"""
src/ratios/delta_track.py
Phase 18.2: Authoritative Delta Tracking with Convergence Sweeps

PURPOSE:
========
Extend the delta tracking harness to be the authoritative diagnostic for:
1. Structured DeltaRecord with all metrics and convergence parameters
2. ConvergenceTable for parameter sweeps
3. One-shot debugging lens for investigating the +5 signature

This module provides the "single source of truth" for delta metrics.

USAGE:
======
>>> from src.ratios.delta_track import compute_delta_record, run_convergence_sweep
>>> record = compute_delta_record("kappa", LaurentMode.ACTUAL_LOGDERIV)
>>> table = run_convergence_sweep("kappa", "quadrature_n", [40, 60, 80, 100])
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from src.ratios.przz_polynomials import load_przz_k3_polynomials
from src.ratios.j1_euler_maclaurin import (
    compute_m1_with_mirror_assembly,
    LaurentMode,
    DEFAULT_LAURENT_MODE,
)


@dataclass(frozen=True)
class DeltaRecord:
    """
    Structured record for a single delta computation.

    This is the authoritative representation of delta metrics,
    including both core values and convergence parameters.
    """

    # Identification
    benchmark: str
    laurent_mode: str
    R: float
    theta: float
    K: int

    # Core metrics
    A: float
    """exp(R) coefficient = I12_minus"""

    B: float
    """Constant offset"""

    delta: float
    """D/A contamination ratio"""

    B_over_A: float
    """Should be ~5 for K=3"""

    # Channel breakdown
    I12_plus: float
    """Sum of j11+j12+j15 at +R"""

    I12_minus: float
    """Sum of j11+j12+j15 at -R"""

    I34_plus: float
    """Sum of j13+j14 at +R"""

    # Per-piece breakdown (Phase 17B)
    j11_plus: float
    j12_plus: float
    j15_plus: float
    j13_plus: float
    j14_plus: float
    j11_minus: float
    j12_minus: float
    j15_minus: float

    # Convergence parameters (optional, for sweep tracking)
    quadrature_n: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass(frozen=True)
class ConvergenceTable:
    """Results from a convergence sweep."""

    benchmark: str
    """Benchmark name: 'kappa' or 'kappa_star'"""

    sweep_parameter: str
    """Parameter being swept: 'quadrature_n'"""

    parameter_values: Tuple[int, ...]
    """Values tested"""

    records: Tuple[DeltaRecord, ...]
    """DeltaRecord for each parameter value"""

    # Convergence summary
    is_converged: bool
    """True if final values have stabilized"""

    relative_change_final: float
    """Relative change between last two values (for B/A)"""

    def summary_table(self) -> str:
        """Return a formatted summary table."""
        lines = []
        lines.append(f"\nConvergence Sweep: {self.sweep_parameter}")
        lines.append(f"Benchmark: {self.benchmark}")
        lines.append("-" * 60)
        lines.append(f"{'Value':<10} {'B/A':<12} {'delta':<12} {'A':<12}")
        lines.append("-" * 60)

        for i, record in enumerate(self.records):
            param = self.parameter_values[i]
            lines.append(
                f"{param:<10} {record.B_over_A:<12.6f} {record.delta:<12.6f} "
                f"{record.A:<12.6f}"
            )

        lines.append("-" * 60)
        status = "CONVERGED" if self.is_converged else "NOT CONVERGED"
        lines.append(f"Status: {status} (final change: {self.relative_change_final:.4%})")
        return "\n".join(lines)


def compute_delta_record(
    benchmark: str,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
    theta: float = 4.0 / 7.0,
    K: int = 3,
    quadrature_n: Optional[int] = None,
) -> DeltaRecord:
    """
    Compute a single DeltaRecord with specified parameters.

    Args:
        benchmark: 'kappa' or 'kappa_star'
        laurent_mode: Laurent mode for evaluation
        theta: theta parameter
        K: Number of pieces
        quadrature_n: Optional quadrature resolution (for tracking)

    Returns:
        DeltaRecord with all metrics
    """
    polys = load_przz_k3_polynomials(benchmark)

    # Compute mirror assembly
    decomp = compute_m1_with_mirror_assembly(
        theta=theta,
        R=polys.R,
        polys=polys,
        K=K,
        laurent_mode=laurent_mode,
    )

    # Extract values
    I12_plus = decomp["i12_plus_total"]
    I12_minus = decomp["i12_minus_total"]
    I34_plus = decomp["i34_plus_total"]

    i12_plus_pieces = decomp["i12_plus_pieces"]
    i12_minus_pieces = decomp["i12_minus_pieces"]
    i34_plus_pieces = decomp["i34_plus_pieces"]

    return DeltaRecord(
        benchmark=benchmark,
        laurent_mode=laurent_mode.value,
        R=polys.R,
        theta=theta,
        K=K,
        A=decomp["exp_coefficient"],
        B=decomp["constant_offset"],
        delta=decomp["delta"],
        B_over_A=decomp["B_over_A"],
        I12_plus=I12_plus,
        I12_minus=I12_minus,
        I34_plus=I34_plus,
        j11_plus=i12_plus_pieces.get("j11", 0.0),
        j12_plus=i12_plus_pieces.get("j12", 0.0),
        j15_plus=i12_plus_pieces.get("j15", 0.0),
        j13_plus=i34_plus_pieces.get("j13", 0.0),
        j14_plus=i34_plus_pieces.get("j14", 0.0),
        j11_minus=i12_minus_pieces.get("j11", 0.0),
        j12_minus=i12_minus_pieces.get("j12", 0.0),
        j15_minus=i12_minus_pieces.get("j15", 0.0),
        quadrature_n=quadrature_n,
    )


def run_convergence_sweep(
    benchmark: str,
    sweep_type: str,
    values: List[int],
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
    convergence_threshold: float = 0.01,
) -> ConvergenceTable:
    """
    Run convergence sweep varying a single parameter.

    Args:
        benchmark: 'kappa' or 'kappa_star'
        sweep_type: Currently only 'quadrature_n' is supported
        values: List of parameter values to test
        laurent_mode: Laurent mode for evaluation
        convergence_threshold: Relative change threshold for convergence

    Returns:
        ConvergenceTable with all records and convergence summary
    """
    if sweep_type != "quadrature_n":
        raise ValueError(
            f"Unsupported sweep type: {sweep_type}. "
            f"Currently only 'quadrature_n' is supported."
        )

    records = []
    for val in values:
        record = compute_delta_record(
            benchmark=benchmark,
            laurent_mode=laurent_mode,
            quadrature_n=val,
        )
        records.append(record)

    # Compute convergence
    if len(records) >= 2:
        last_ba = records[-1].B_over_A
        prev_ba = records[-2].B_over_A
        relative_change = abs(last_ba - prev_ba) / abs(prev_ba) if prev_ba != 0 else 0
        is_converged = relative_change < convergence_threshold
    else:
        relative_change = 1.0
        is_converged = False

    return ConvergenceTable(
        benchmark=benchmark,
        sweep_parameter=sweep_type,
        parameter_values=tuple(values),
        records=tuple(records),
        is_converged=is_converged,
        relative_change_final=relative_change,
    )


def run_delta_track_report(
    verbose: bool = True,
    output_path: Optional[Path] = None,
) -> Dict[str, DeltaRecord]:
    """
    Run comprehensive delta tracking report for both benchmarks.

    This is the "one-shot debugging lens": whenever something changes,
    run this to see which piece moved.

    Args:
        verbose: Print detailed report
        output_path: Optional path to save JSON report

    Returns:
        Dict with 'kappa' and 'kappa_star' DeltaRecords
    """
    results = {}

    for benchmark in ["kappa", "kappa_star"]:
        for mode in [LaurentMode.RAW_LOGDERIV, LaurentMode.ACTUAL_LOGDERIV]:
            key = f"{benchmark}_{mode.value}"
            results[key] = compute_delta_record(benchmark, mode)

    if verbose:
        print_delta_track_report(results)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)

    return results


def print_delta_track_report(results: Dict[str, DeltaRecord]) -> None:
    """Print formatted delta tracking report."""
    print()
    print("=" * 80)
    print("PHASE 18.2: DELTA TRACK REPORT")
    print("=" * 80)

    # Group by benchmark
    for benchmark in ["kappa", "kappa_star"]:
        print(f"\n{benchmark.upper()}:")
        print("-" * 70)
        print(f"{'Mode':<20} {'B/A':<10} {'delta':<10} {'gap%':<10} {'A':<12} {'D':<12}")
        print("-" * 70)

        for mode in [LaurentMode.RAW_LOGDERIV, LaurentMode.ACTUAL_LOGDERIV]:
            key = f"{benchmark}_{mode.value}"
            if key in results:
                record = results[key]
                gap_pct = (record.B_over_A - 5.0) / 5.0 * 100
                print(
                    f"{mode.value:<20} {record.B_over_A:<10.4f} {record.delta:<10.4f} "
                    f"{gap_pct:+9.2f}% {record.A:<12.6f} {record.I12_plus + record.I34_plus:<12.6f}"
                )

        print("-" * 70)

    # Per-piece comparison
    print("\n\nPER-PIECE BREAKDOWN (ACTUAL_LOGDERIV mode):")
    print("-" * 70)
    print(f"{'Piece':<10} {'kappa':<15} {'kappa*':<15} {'ratio':<10}")
    print("-" * 70)

    kappa = results.get("kappa_actual_logderiv")
    kappa_star = results.get("kappa_star_actual_logderiv")

    if kappa and kappa_star:
        pieces = [
            ("j11+", kappa.j11_plus, kappa_star.j11_plus),
            ("j12+", kappa.j12_plus, kappa_star.j12_plus),
            ("j15+", kappa.j15_plus, kappa_star.j15_plus),
            ("j13+", kappa.j13_plus, kappa_star.j13_plus),
            ("j14+", kappa.j14_plus, kappa_star.j14_plus),
            ("I12+", kappa.I12_plus, kappa_star.I12_plus),
            ("I12-", kappa.I12_minus, kappa_star.I12_minus),
            ("I34+", kappa.I34_plus, kappa_star.I34_plus),
        ]

        for name, val_k, val_ks in pieces:
            ratio = val_k / val_ks if abs(val_ks) > 1e-10 else float("inf")
            print(f"{name:<10} {val_k:<15.6f} {val_ks:<15.6f} {ratio:<10.4f}")

    print("-" * 70)
    print()


if __name__ == "__main__":
    # Run one-shot diagnostic
    run_delta_track_report(verbose=True)
