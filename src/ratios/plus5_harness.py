"""
src/ratios/plus5_harness.py
Phase 19.1: Split +5 Harness with Mode Separation

PURPOSE:
========
Provide explicit separation between:
- MAIN_TERM_ONLY: B/A computation without J₁,₅ (A^{(1,1)} contribution)
- WITH_ERROR_TERMS: Full computation including J₁,₅

This implements GPT's Phase 19.1.1 guidance: "Make it impossible to 'pass +5'
by using error terms."

KEY INSIGHT (from TRUTH_SPEC.md):
================================
J₁,₅ involves A^{(1,1)} which is explicitly an error term (Lines 1621-1628):
"Hence the term associated to A_{α,β}^{(1,1)}(0,0;β,α) is an error term."

If our derivation NEEDS J₁,₅ to achieve B/A = 5, we're computing the wrong
main-term object.

USAGE:
======
>>> from src.ratios.plus5_harness import compute_plus5_signature_split
>>> result = compute_plus5_signature_split("kappa")
>>> print(f"Main-only B/A: {result.B_over_A_main_only}")
>>> print(f"With-error B/A: {result.B_over_A_with_error}")
>>> print(f"J15 contribution: {result.j15_contribution}")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum, auto

from src.evaluation_modes import (
    EvaluationMode,
    get_evaluation_mode,
    evaluation_mode_context,
)
from src.ratios.przz_polynomials import load_przz_k3_polynomials
from src.ratios.j1_euler_maclaurin import (
    compute_m1_with_mirror_assembly,
    LaurentMode,
    DEFAULT_LAURENT_MODE,
)


@dataclass(frozen=True)
class J15Provenance:
    """
    Provenance metadata for J15 contribution tracking.

    This documents exactly where the J15 contribution comes from,
    enabling reconciliation between the code's J15 and TRUTH_SPEC's I5.

    Phase 20.1: Implements GPT guidance on provenance tagging.
    """

    # Source identification
    source_module: str
    """Module that computes J15 (e.g., 'src.ratios.j1_euler_maclaurin')"""

    source_function: str
    """Function that computes J15 (e.g., 'j15_contribution_integral')"""

    # PRZZ reference
    przz_line_numbers: str
    """PRZZ TeX lines defining J15 (e.g., '1621-1628 (A^{(1,1)} error term)')"""

    truth_spec_reference: str
    """TRUTH_SPEC.md section reference"""

    # Guardrail status
    passed_evaluation_mode_guardrails: bool
    """Whether computation went through evaluation_modes.py checks"""

    guardrail_mode: Optional[str]
    """Which EvaluationMode was active (if any)"""

    # Mathematical identity
    formula_description: str
    """Mathematical description of what J15 computes"""

    is_error_term_per_spec: bool
    """True if TRUTH_SPEC classifies this as error-order"""

    # Warnings
    reconciliation_notes: List[str] = field(default_factory=list)
    """Notes on J15 vs I5 reconciliation status"""


# Default J15 provenance for the current implementation
DEFAULT_J15_PROVENANCE = J15Provenance(
    source_module="src.ratios.j1_euler_maclaurin",
    source_function="j15_contribution_integral",
    przz_line_numbers="1621-1628 (A^{(1,1)} terms)",
    truth_spec_reference="TRUTH_SPEC.md Lines 1621-1628: 'Hence the term associated to A_{α,β}^{(1,1)}(0,0;β,α) is an error term.'",
    passed_evaluation_mode_guardrails=False,  # Not yet wired through guardrails
    guardrail_mode=None,
    formula_description=(
        "J15 = contribution from A^{(1,1)} derivative terms in the I12 integral. "
        "This involves ζ'/ζ evaluated at specific points, contributing to both "
        "exp(R) coefficient and constant offset."
    ),
    is_error_term_per_spec=True,
    reconciliation_notes=[
        "OPEN QUESTION: Is code's 'J15' the same as TRUTH_SPEC's 'I5'?",
        "J15 uses j1_euler_maclaurin.py Euler-Maclaurin approximation",
        "I5 per TRUTH_SPEC is the A^{(1,1)} error term contribution",
        "Reconciliation needed: compare code path to PRZZ formula",
    ],
)


@dataclass(frozen=True)
class Plus5SplitResult:
    """
    Result of +5 harness with explicit mode separation.

    The key output is the separation between:
    - B_over_A_main_only: Computed WITHOUT J₁,₅ (the paper's main term)
    - B_over_A_with_error: Computed WITH J₁,₅ (includes error-order term)

    If these differ significantly, the "+5 signature" relies on error terms.
    """

    benchmark: str
    mode: str
    R: float
    theta: float
    K: int

    # Core metrics (full computation with J15)
    A: float
    """exp(R) coefficient = I₁₂(-R) [with J15]"""

    B: float
    """Constant offset [with J15]"""

    B_over_A: float
    """Full B/A ratio [with J15]"""

    # Main-only metrics (WITHOUT J15)
    A_main_only: float
    """exp(R) coefficient WITHOUT J₁,₅"""

    B_main_only: float
    """Constant offset WITHOUT J₁,₅"""

    B_over_A_main_only: float
    """B/A ratio computed WITHOUT J₁,₅ error term"""

    # With-error metrics (same as full, explicit label)
    B_over_A_with_error: float
    """B/A ratio computed WITH J₁,₅ (same as B_over_A)"""

    # J15 contribution tracking
    j15_contribution_A: float
    """J₁,₅ contribution to A coefficient"""

    j15_contribution_B: float
    """J₁,₅ contribution to B coefficient"""

    j15_contribution_ratio: float
    """How much J₁,₅ affects the final B/A (B_over_A - B_over_A_main_only)"""

    # Delta metrics
    delta: float
    delta_main_only: float

    # Gap analysis
    gap_percent: float
    """Gap from target 5 (full)"""

    gap_percent_main_only: float
    """Gap from target 5 (main-only)"""

    # Warning flags
    j15_required_for_target: bool
    """True if main-only misses target but with-error hits it"""

    # Provenance tracking (Phase 20.1)
    j15_provenance: J15Provenance = DEFAULT_J15_PROVENANCE
    """Metadata about where J15 contribution comes from"""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        from dataclasses import asdict
        return asdict(self)


def compute_plus5_signature_split(
    benchmark: str,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
    theta: float = 4.0 / 7.0,
    K: int = 3,
) -> Plus5SplitResult:
    """
    Compute +5 signature with explicit main-only vs with-error separation.

    This is the Phase 19.1.1 implementation: we run the computation twice,
    once with J₁,₅ and once without, to see if the +5 signature depends
    on error terms.

    Args:
        benchmark: 'kappa' or 'kappa_star'
        laurent_mode: Laurent mode for evaluation
        theta: theta parameter
        K: Number of pieces

    Returns:
        Plus5SplitResult with both modes and J₁,₅ contribution analysis
    """
    polys = load_przz_k3_polynomials(benchmark)
    R = polys.R

    # Compute WITH J15 (full, includes error term)
    decomp_full = compute_m1_with_mirror_assembly(
        theta=theta,
        R=R,
        polys=polys,
        K=K,
        laurent_mode=laurent_mode,
        include_j15=True,
    )

    # Compute WITHOUT J15 (main-only, no error term)
    decomp_main = compute_m1_with_mirror_assembly(
        theta=theta,
        R=R,
        polys=polys,
        K=K,
        laurent_mode=laurent_mode,
        include_j15=False,
    )

    # Extract full metrics
    A = decomp_full["exp_coefficient"]
    B = decomp_full["constant_offset"]
    B_over_A = decomp_full["B_over_A"]
    delta = decomp_full["delta"]

    # Extract main-only metrics
    A_main = decomp_main["exp_coefficient"]
    B_main = decomp_main["constant_offset"]
    B_over_A_main = decomp_main["B_over_A"]
    delta_main = decomp_main["delta"]

    # J15 contribution
    j15_A = A - A_main
    j15_B = B - B_main
    j15_ratio = B_over_A - B_over_A_main

    # Gap analysis
    target = 2 * K - 1  # = 5 for K=3
    gap_full = (B_over_A - target) / target * 100
    gap_main = (B_over_A_main - target) / target * 100

    # Check if J15 is required to hit target
    # "Required" = main-only has >5% gap but with-error has <5% gap
    j15_required = (abs(gap_main) > 5.0) and (abs(gap_full) < 5.0)

    return Plus5SplitResult(
        benchmark=benchmark,
        mode=laurent_mode.value,
        R=R,
        theta=theta,
        K=K,
        A=A,
        B=B,
        B_over_A=B_over_A,
        A_main_only=A_main,
        B_main_only=B_main,
        B_over_A_main_only=B_over_A_main,
        B_over_A_with_error=B_over_A,
        j15_contribution_A=j15_A,
        j15_contribution_B=j15_B,
        j15_contribution_ratio=j15_ratio,
        delta=delta,
        delta_main_only=delta_main,
        gap_percent=gap_full,
        gap_percent_main_only=gap_main,
        j15_required_for_target=j15_required,
    )


def run_plus5_split_report(
    benchmarks: list = None,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
    verbose: bool = True,
) -> Dict[str, Plus5SplitResult]:
    """
    Run +5 split analysis for specified benchmarks.

    Args:
        benchmarks: List of benchmark names (default: both)
        laurent_mode: Laurent mode
        verbose: Print detailed report

    Returns:
        Dict with results per benchmark
    """
    if benchmarks is None:
        benchmarks = ["kappa", "kappa_star"]

    results = {}
    for bench in benchmarks:
        results[bench] = compute_plus5_signature_split(
            bench, laurent_mode=laurent_mode
        )

    if verbose:
        print_plus5_split_report(results, laurent_mode)

    return results


def print_plus5_split_report(
    results: Dict[str, Plus5SplitResult],
    laurent_mode: LaurentMode,
) -> None:
    """Print formatted +5 split report."""
    print()
    print("=" * 80)
    print("PHASE 19.1: +5 SIGNATURE SPLIT ANALYSIS")
    print(f"Laurent Mode: {laurent_mode.value}")
    print("=" * 80)
    print()
    print("KEY: Does the +5 signature REQUIRE J₁,₅ (error term)?")
    print("     If B/A_main_only ≈ 5, we're matching the paper's main term.")
    print("     If B/A_with_error ≈ 5 but B/A_main_only ≠ 5, we're using error terms.")
    print()

    # Summary table
    print("-" * 80)
    print(f"{'Benchmark':<12} {'B/A (full)':<12} {'B/A (main)':<12} "
          f"{'J15 effect':<12} {'J15 req?':<10}")
    print("-" * 80)

    for name, result in results.items():
        j15_req = "YES ⚠" if result.j15_required_for_target else "no"
        print(f"{name:<12} {result.B_over_A:<12.4f} {result.B_over_A_main_only:<12.4f} "
              f"{result.j15_contribution_ratio:+12.4f} {j15_req:<10}")

    print("-" * 80)
    print()

    # Detailed per-benchmark
    for name, result in results.items():
        print(f"\n{name.upper()} (R={result.R}):")
        print("-" * 50)

        print(f"  FULL (with J₁,₅):")
        print(f"    A = {result.A:.6f}")
        print(f"    B = {result.B:.6f}")
        print(f"    B/A = {result.B_over_A:.4f} (gap: {result.gap_percent:+.2f}%)")

        print(f"  MAIN-ONLY (without J₁,₅):")
        print(f"    A = {result.A_main_only:.6f}")
        print(f"    B = {result.B_main_only:.6f}")
        print(f"    B/A = {result.B_over_A_main_only:.4f} (gap: {result.gap_percent_main_only:+.2f}%)")

        print(f"  J₁,₅ CONTRIBUTION:")
        print(f"    ΔA = {result.j15_contribution_A:+.6f}")
        print(f"    ΔB = {result.j15_contribution_B:+.6f}")
        print(f"    Δ(B/A) = {result.j15_contribution_ratio:+.4f}")

        if result.j15_required_for_target:
            print(f"  ⚠ WARNING: J₁,₅ is REQUIRED to achieve B/A ≈ 5")
            print(f"    This means the derivation relies on error terms!")
            print(f"    Per TRUTH_SPEC Lines 1621-1628, this is incorrect.")
        else:
            print(f"  ✓ J₁,₅ is NOT required for B/A ≈ 5")

    print()
    print("=" * 80)


def check_main_term_sufficiency(
    benchmark: str,
    laurent_mode: LaurentMode = DEFAULT_LAURENT_MODE,
    tolerance_percent: float = 5.0,
) -> bool:
    """
    Check if main-term computation (without J₁,₅) achieves B/A ≈ 5.

    This is the key Phase 19.1.1 gate: the main term MUST be sufficient.
    If J₁,₅ is required, we're computing the wrong object.

    Args:
        benchmark: Benchmark name
        laurent_mode: Laurent mode
        tolerance_percent: Gap tolerance (default 5%)

    Returns:
        True if main-term B/A is within tolerance of 5
    """
    result = compute_plus5_signature_split(benchmark, laurent_mode)
    return abs(result.gap_percent_main_only) < tolerance_percent


if __name__ == "__main__":
    from src.ratios.j1_euler_maclaurin import LaurentMode

    # Run split analysis
    print("\n" + "#" * 70)
    print("# PHASE 19.1: +5 GATE SPLIT ANALYSIS")
    print("#" * 70)

    run_plus5_split_report(
        benchmarks=["kappa", "kappa_star"],
        laurent_mode=LaurentMode.ACTUAL_LOGDERIV,
        verbose=True,
    )
