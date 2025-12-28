"""
src/evaluator/gap_attribution.py
Phase 25.1: Gap Attribution Harness

PURPOSE:
========
Systematic comparison infrastructure to identify which component(s) account for
the 5-7% gap between unified S12 and empirical DSL evaluators.

This module does NOT fit or correct anything - it only ATTRIBUTES the gap to
specific components (S12, S34, specific pairs, etc.).

NON-NEGOTIABLES (from Phase 24):
================================
- Keep normalization_mode="scalar" as production default
- Keep diagnostic_corrected quarantined (requires explicit opt-in)
- No fitting or new corrections - this is attribution only
- Two-benchmark gate: all findings must hold for BOTH kappa AND kappa*

Created: 2025-12-25
Phase: 25.1
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
import math
import json

# NOTE: Imports are done inside functions to avoid circular imports
# src.evaluate imports from src.evaluator, which would create a cycle if we import here


# =============================================================================
# CONSTANTS
# =============================================================================

# PRZZ benchmark parameters
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167
THETA = 4.0 / 7.0

# PRZZ target values for c
KAPPA_C_TARGET = 2.13745440613217263636
KAPPA_STAR_C_TARGET = 1.938  # approximate


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class GapReport:
    """
    Complete gap attribution report between unified and empirical evaluators.

    This report identifies precisely WHERE the 5-7% gap originates.
    """
    # Input parameters (required)
    theta: float
    R: float
    n_quad: int
    benchmark: str  # "kappa" or "kappa_star"

    # Unified evaluator results (required)
    unified_S12_total: float
    unified_S12_unnormalized: float
    unified_normalization_factor: float
    unified_normalization_mode: str

    # Empirical evaluator results (required)
    empirical_c_total: float
    empirical_S12_plus_total: float
    empirical_S12_minus_total: float
    empirical_S12_combined: float  # S12_plus + m * S12_minus
    empirical_S34_total: float
    empirical_mirror_multiplier: float

    # Fields with defaults must come after required fields
    # Gap metrics
    delta_S12: float = 0.0  # unified - empirical
    delta_S34: float = 0.0  # should be ~0 if S34 invariant
    delta_c: float = 0.0
    ratio_S12: float = 1.0  # unified / empirical
    ratio_S34: float = 1.0  # should be ~1.0

    # Per-pair breakdown
    unified_per_pair: Dict[str, float] = field(default_factory=dict)
    per_pair_unified: Dict[str, float] = field(default_factory=dict)
    per_pair_ratio: Dict[str, float] = field(default_factory=dict)

    # Diagnosis flags
    s34_invariant: bool = True  # |delta_S34| < tolerance
    gap_in_S12: bool = False    # |delta_S12| > tolerance
    largest_gap_pair: str = ""
    largest_gap_ratio: float = 1.0

    # Target comparison
    c_target: float = 0.0
    unified_c_vs_target_pct: float = 0.0
    empirical_c_vs_target_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "benchmark": self.benchmark,
            "R": self.R,
            "theta": self.theta,
            "n_quad": self.n_quad,
            "unified": {
                "S12_total": self.unified_S12_total,
                "S12_unnormalized": self.unified_S12_unnormalized,
                "normalization_factor": self.unified_normalization_factor,
                "normalization_mode": self.unified_normalization_mode,
                "per_pair": self.unified_per_pair,
            },
            "empirical": {
                "c_total": self.empirical_c_total,
                "S12_plus_total": self.empirical_S12_plus_total,
                "S12_minus_total": self.empirical_S12_minus_total,
                "S12_combined": self.empirical_S12_combined,
                "S34_total": self.empirical_S34_total,
                "mirror_multiplier": self.empirical_mirror_multiplier,
            },
            "gap_metrics": {
                "delta_S12": self.delta_S12,
                "delta_S34": self.delta_S34,
                "delta_c": self.delta_c,
                "ratio_S12": self.ratio_S12,
                "ratio_S34": self.ratio_S34,
            },
            "per_pair_ratio": self.per_pair_ratio,
            "diagnosis": {
                "s34_invariant": self.s34_invariant,
                "gap_in_S12": self.gap_in_S12,
                "largest_gap_pair": self.largest_gap_pair,
                "largest_gap_ratio": self.largest_gap_ratio,
            },
            "vs_target": {
                "c_target": self.c_target,
                "unified_c_vs_target_pct": self.unified_c_vs_target_pct,
                "empirical_c_vs_target_pct": self.empirical_c_vs_target_pct,
            },
        }


# =============================================================================
# MAIN ATTRIBUTION FUNCTIONS
# =============================================================================


def compute_gap_report(
    theta: float,
    R: float,
    n_quad: int,
    polynomials: Dict,
    *,
    normalization_mode: str = "scalar",
    benchmark_name: str = "unknown",
    c_target: float = 0.0,
    K: int = 3,
) -> GapReport:
    """
    Compute comprehensive gap attribution report.

    Runs both evaluators and compares results across all components.

    Args:
        theta: PRZZ theta parameter (typically 4/7)
        R: PRZZ R parameter (1.3036 for kappa, 1.1167 for kappa*)
        n_quad: Quadrature points
        polynomials: Dict with P1, P2, P3, Q
        normalization_mode: "scalar" (default), "none", or "diagnostic_corrected"
        benchmark_name: Label for the benchmark ("kappa" or "kappa_star")
        c_target: Target c value for comparison
        K: Mollifier piece count (default 3)

    Returns:
        GapReport with full attribution
    """
    # Lazy imports to avoid circular dependency
    from src.evaluate import compute_c_paper_with_mirror

    # -------------------------------------------------------------------------
    # Step 1: Compute EMPIRICAL baseline (empirical_scalar mode)
    # -------------------------------------------------------------------------
    empirical_result = compute_c_paper_with_mirror(
        theta=theta,
        R=R,
        n=n_quad,
        polynomials=polynomials,
        pair_mode="hybrid",
        mirror_mode="empirical_scalar",
        K=K,
    )

    # Extract empirical components
    per_term = empirical_result.per_term
    mirror_mult = math.exp(R) + (2 * K - 1)  # exp(R) + 5 for K=3

    empirical_S12_plus = per_term.get("_S12_plus_total", 0.0)
    empirical_S12_minus = per_term.get("_S12_minus_total", 0.0)
    # S34 can be stored under different keys depending on mode
    empirical_S34 = (
        per_term.get("_S34_plus_total") or
        per_term.get("_S34_triangle_total") or
        per_term.get("_S34_ordered_total") or
        per_term.get("_I3_I4_plus_total") or
        0.0
    )
    empirical_S12_combined = empirical_S12_plus + mirror_mult * empirical_S12_minus

    # -------------------------------------------------------------------------
    # Step 2: Compute UNIFIED result (difference_quotient_v3 + scalar)
    # -------------------------------------------------------------------------
    unified_result = compute_c_paper_with_mirror(
        theta=theta,
        R=R,
        n=n_quad,
        polynomials=polynomials,
        pair_mode="hybrid",
        mirror_mode="difference_quotient_v3",
        normalization_mode=normalization_mode,
        allow_diagnostic_correction=False,  # Always disallow for attribution
        K=K,
    )

    # Extract unified components
    unified_per_term = unified_result.per_term
    unified_S12_total = unified_per_term.get("_S12_unified_total", 0.0)
    unified_S12_unnormalized = unified_per_term.get("_S12_unnormalized", 0.0)
    unified_S34 = unified_per_term.get("_S34_total", 0.0)
    unified_norm_factor = unified_per_term.get("_normalization_factor", 1.0)
    unified_norm_mode = unified_per_term.get("_normalization_mode", "none")
    unified_per_pair_raw = unified_per_term.get("_per_pair_contributions", {})

    # -------------------------------------------------------------------------
    # Step 3: Compute gap metrics
    # -------------------------------------------------------------------------

    # S12 gap (the key metric)
    delta_S12 = unified_S12_total - empirical_S12_combined
    ratio_S12 = unified_S12_total / empirical_S12_combined if empirical_S12_combined != 0 else float('inf')

    # S34 gap (should be ~0, same computation path)
    delta_S34 = unified_S34 - empirical_S34
    ratio_S34 = unified_S34 / empirical_S34 if empirical_S34 != 0 else float('inf')

    # Total c gap
    delta_c = unified_result.total - empirical_result.total

    # -------------------------------------------------------------------------
    # Step 4: Per-pair breakdown
    # -------------------------------------------------------------------------
    unified_per_pair = {}
    per_pair_ratio = {}
    largest_gap_pair = ""
    largest_gap_ratio = 1.0

    if unified_per_pair_raw:
        for pair_key, value in unified_per_pair_raw.items():
            unified_per_pair[pair_key] = value
            # Note: Empirical per-pair S12 breakdown not easily available
            # We record unified breakdown for analysis

    # Find largest deviation from mean ratio
    if unified_per_pair:
        pair_values = list(unified_per_pair.values())
        mean_val = sum(pair_values) / len(pair_values) if pair_values else 1.0
        for pair_key, value in unified_per_pair.items():
            deviation = abs(value - mean_val) / abs(mean_val) if mean_val != 0 else 0.0
            if deviation > abs(largest_gap_ratio - 1.0):
                largest_gap_pair = pair_key
                largest_gap_ratio = 1.0 + deviation

    # -------------------------------------------------------------------------
    # Step 5: Diagnosis flags
    # -------------------------------------------------------------------------
    S34_TOLERANCE = 1e-6
    S12_GAP_THRESHOLD = 0.01  # 1% gap threshold

    s34_invariant = abs(delta_S34) < S34_TOLERANCE
    gap_in_S12 = abs(delta_S12 / empirical_S12_combined) > S12_GAP_THRESHOLD if empirical_S12_combined != 0 else False

    # -------------------------------------------------------------------------
    # Step 6: Target comparison
    # -------------------------------------------------------------------------
    if c_target > 0:
        # For unified, we need to add S34 to get c
        unified_c = unified_S12_total + unified_S34
        unified_c_vs_target_pct = (unified_c - c_target) / c_target * 100
        empirical_c_vs_target_pct = (empirical_result.total - c_target) / c_target * 100
    else:
        unified_c_vs_target_pct = 0.0
        empirical_c_vs_target_pct = 0.0

    # -------------------------------------------------------------------------
    # Step 7: Build report
    # -------------------------------------------------------------------------
    return GapReport(
        # Input
        theta=theta,
        R=R,
        n_quad=n_quad,
        benchmark=benchmark_name,

        # Unified
        unified_S12_total=unified_S12_total,
        unified_S12_unnormalized=unified_S12_unnormalized,
        unified_normalization_factor=unified_norm_factor,
        unified_normalization_mode=unified_norm_mode,
        unified_per_pair=unified_per_pair,

        # Empirical
        empirical_c_total=empirical_result.total,
        empirical_S12_plus_total=empirical_S12_plus,
        empirical_S12_minus_total=empirical_S12_minus,
        empirical_S12_combined=empirical_S12_combined,
        empirical_S34_total=empirical_S34,
        empirical_mirror_multiplier=mirror_mult,

        # Gap metrics
        delta_S12=delta_S12,
        delta_S34=delta_S34,
        delta_c=delta_c,
        ratio_S12=ratio_S12,
        ratio_S34=ratio_S34,

        # Per-pair
        per_pair_unified=unified_per_pair,
        per_pair_ratio=per_pair_ratio,

        # Diagnosis
        s34_invariant=s34_invariant,
        gap_in_S12=gap_in_S12,
        largest_gap_pair=largest_gap_pair,
        largest_gap_ratio=largest_gap_ratio,

        # Target
        c_target=c_target,
        unified_c_vs_target_pct=unified_c_vs_target_pct,
        empirical_c_vs_target_pct=empirical_c_vs_target_pct,
    )


def run_dual_benchmark_gap_attribution(
    n_quad: int = 60,
    normalization_mode: str = "scalar",
) -> Tuple[GapReport, GapReport]:
    """
    Run gap attribution on both kappa and kappa* benchmarks.

    This enforces the two-benchmark gate: any finding must be consistent
    across BOTH benchmarks.

    Args:
        n_quad: Quadrature points (default 60 for accuracy)
        normalization_mode: "scalar" (default) or "none"

    Returns:
        (kappa_report, kappa_star_report)
    """
    # Lazy imports to avoid circular dependency
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

    # Load polynomials
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    kappa_polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    kappa_star_polys = {"P1": P1s, "P2": P2s, "P3": P3s, "Q": Qs}

    # Compute kappa report
    kappa_report = compute_gap_report(
        theta=THETA,
        R=KAPPA_R,
        n_quad=n_quad,
        polynomials=kappa_polys,
        normalization_mode=normalization_mode,
        benchmark_name="kappa",
        c_target=KAPPA_C_TARGET,
    )

    # Compute kappa* report
    kappa_star_report = compute_gap_report(
        theta=THETA,
        R=KAPPA_STAR_R,
        n_quad=n_quad,
        polynomials=kappa_star_polys,
        normalization_mode=normalization_mode,
        benchmark_name="kappa_star",
        c_target=KAPPA_STAR_C_TARGET,
    )

    return kappa_report, kappa_star_report


# =============================================================================
# PRINT/OUTPUT FUNCTIONS
# =============================================================================


def print_gap_report(report: GapReport) -> None:
    """Print formatted gap attribution report."""
    print("=" * 70)
    print(f"GAP ATTRIBUTION REPORT: {report.benchmark.upper()}")
    print("=" * 70)
    print(f"R = {report.R:.4f}, theta = {report.theta:.6f}, n = {report.n_quad}")
    print()

    print("UNIFIED EVALUATOR (difference_quotient_v3 + scalar normalization):")
    print(f"  S12_total (normalized):   {report.unified_S12_total:>12.6f}")
    print(f"  S12_unnormalized:         {report.unified_S12_unnormalized:>12.6f}")
    print(f"  normalization_factor:     {report.unified_normalization_factor:>12.6f}")
    print(f"  normalization_mode:       {report.unified_normalization_mode}")
    print()

    print("EMPIRICAL EVALUATOR (empirical_scalar mode):")
    print(f"  S12_plus:                 {report.empirical_S12_plus_total:>12.6f}")
    print(f"  S12_minus:                {report.empirical_S12_minus_total:>12.6f}")
    print(f"  mirror_mult:              {report.empirical_mirror_multiplier:>12.6f}")
    print(f"  S12_combined:             {report.empirical_S12_combined:>12.6f}")
    print(f"  S34_total:                {report.empirical_S34_total:>12.6f}")
    print(f"  c_total:                  {report.empirical_c_total:>12.6f}")
    print()

    print("GAP METRICS:")
    print(f"  delta_S12 (unified - empirical): {report.delta_S12:>+12.6f}")
    print(f"  ratio_S12 (unified / empirical): {report.ratio_S12:>12.4f}")
    pct_gap = (report.ratio_S12 - 1.0) * 100
    print(f"  S12 gap percentage:              {pct_gap:>+12.2f}%")
    print()
    print(f"  delta_S34:                       {report.delta_S34:>+12.6e}")
    print(f"  ratio_S34:                       {report.ratio_S34:>12.6f}")
    print()
    print(f"  delta_c:                         {report.delta_c:>+12.6f}")
    print()

    print("DIAGNOSIS:")
    print(f"  S34 invariant (|delta| < 1e-6):  {report.s34_invariant}")
    print(f"  Gap in S12 (|delta| > 1%):       {report.gap_in_S12}")
    if report.largest_gap_pair:
        print(f"  Largest deviation pair:          {report.largest_gap_pair}")
        print(f"  Largest deviation ratio:         {report.largest_gap_ratio:.4f}")
    print()

    if report.c_target > 0:
        print("VS TARGET:")
        print(f"  c_target:                        {report.c_target:>12.6f}")
        print(f"  unified c vs target:             {report.unified_c_vs_target_pct:>+12.2f}%")
        print(f"  empirical c vs target:           {report.empirical_c_vs_target_pct:>+12.2f}%")

    print()

    if report.unified_per_pair:
        print("UNIFIED PER-PAIR BREAKDOWN:")
        for pair_key, value in sorted(report.unified_per_pair.items()):
            print(f"  {pair_key}: {value:>12.6f}")

    print("=" * 70)


def print_dual_benchmark_summary(
    kappa_report: GapReport,
    kappa_star_report: GapReport,
) -> None:
    """Print summary comparing both benchmarks."""
    print()
    print("=" * 70)
    print("DUAL BENCHMARK SUMMARY (Two-Benchmark Gate)")
    print("=" * 70)
    print()

    print("S12 GAP COMPARISON:")
    print(f"  kappa:       ratio = {kappa_report.ratio_S12:.4f}  ({(kappa_report.ratio_S12 - 1)*100:+.2f}%)")
    print(f"  kappa*:      ratio = {kappa_star_report.ratio_S12:.4f}  ({(kappa_star_report.ratio_S12 - 1)*100:+.2f}%)")
    print()

    print("S34 INVARIANCE CHECK:")
    print(f"  kappa:       delta = {kappa_report.delta_S34:+.2e}  (invariant: {kappa_report.s34_invariant})")
    print(f"  kappa*:      delta = {kappa_star_report.delta_S34:+.2e}  (invariant: {kappa_star_report.s34_invariant})")
    print()

    # Check consistency
    ratio_diff = abs(kappa_report.ratio_S12 - kappa_star_report.ratio_S12)
    print("CONSISTENCY CHECK:")
    print(f"  S12 ratio difference:  {ratio_diff:.4f}")

    if ratio_diff < 0.02:  # 2% difference
        print("  STATUS: CONSISTENT (ratio difference < 2%)")
    else:
        print("  STATUS: INCONSISTENT (ratio differs by more than 2%)")
        print("  WARNING: R-dependent effect detected!")

    print()
    print("=" * 70)


def save_gap_reports_json(
    kappa_report: GapReport,
    kappa_star_report: GapReport,
    filepath: str,
) -> None:
    """Save both reports to a JSON file."""
    data = {
        "kappa": kappa_report.to_dict(),
        "kappa_star": kappa_star_report.to_dict(),
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved gap reports to: {filepath}")
