"""
src/abd_diagnostics.py
A, B, D Diagnostic Definitions for Phase 21

PURPOSE:
========
This module provides the SINGLE SOURCE OF TRUTH for the A, B, D decomposition
used in Phase 21 gate tests. All code investigating the "+5" signature and
D → 0 goal MUST use these definitions.

DEFINITIONS:
============
The mirror assembly formula is:
    c = I₁₂(+R) + m × I₁₂(-R) + I₃₄(+R)

where m = exp(R) + 5 is the empirical mirror weight.

We decompose this as:
    c = A × exp(R) + B

where:
    A = I₁₂(-R)                           # exp(R) coefficient
    D = I₁₂(+R) + I₃₄(+R)                 # "leftover" terms
    B = D + 5 × A                         # constant term

The KEY DIAGNOSTIC is:
    B/A = D/A + 5

For the DERIVED (first-principles) structure:
    D = 0  →  B/A = 5  (exactly)

For the EMPIRICAL (current) structure:
    D ≠ 0  →  B/A ≠ 5  (requires empirical m = exp(R) + 5 to match c)

SUCCESS CRITERIA:
=================
Phase 21 succeeds when the unified bracket evaluator produces:
    |D| < 1e-6      (D → 0 analytically)
    |B/A - 5| < 1e-6  (B/A = 5 exactly)

for BOTH benchmarks (κ and κ*).

REFERENCES:
===========
- docs/PHASE_20_2_FINDINGS.md: B/A gap analysis
- docs/PHASE_20_3_FINDINGS.md: exp(R) coefficient analysis
- docs/PLAN_PHASE_21_DIFFERENCE_QUOTIENT.md: Implementation plan
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ABDDecomposition:
    """
    The A, B, D decomposition of the mirror assembly.

    This is the CANONICAL representation for Phase 21 diagnostics.

    Attributes:
        A: exp(R) coefficient = I₁₂(-R)
        B: constant term = D + 5*A
        D: leftover = I₁₂(+R) + I₃₄(+R)

        # Derived quantities
        B_over_A: B/A ratio (target = 5)
        D_over_A: D/A ratio (target = 0)

        # Source components
        I12_plus: I₁₂(+R) value
        I12_minus: I₁₂(-R) value
        I34_plus: I₃₄(+R) value

        # Computed c value
        c_computed: A*exp(R) + B

        # Benchmark info
        benchmark: "kappa" or "kappa_star"
        R: PRZZ R parameter
    """
    # Core decomposition
    A: float
    B: float
    D: float

    # Derived ratios
    B_over_A: float
    D_over_A: float

    # Source components
    I12_plus: float
    I12_minus: float
    I34_plus: float

    # Computed value
    c_computed: float

    # Benchmark info
    benchmark: str
    R: float

    def is_derived_structure(self, tol: float = 1e-6) -> bool:
        """Check if this decomposition matches the derived (D=0) structure."""
        return abs(self.D) < tol and abs(self.B_over_A - 5.0) < tol

    def gap_from_derived(self) -> Tuple[float, float]:
        """Return (D_gap, B_over_A_gap) from derived structure."""
        return (abs(self.D), abs(self.B_over_A - 5.0))

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"ABD Decomposition for {self.benchmark.upper()} (R={self.R})",
            "=" * 50,
            "",
            "Core values:",
            f"  A = I₁₂(-R) = {self.A:.10f}",
            f"  B = D + 5A  = {self.B:.10f}",
            f"  D = I₁₂(+R) + I₃₄(+R) = {self.D:.10f}",
            "",
            "Ratios:",
            f"  B/A = {self.B_over_A:.6f}  (target: 5.0)",
            f"  D/A = {self.D_over_A:.6f}  (target: 0.0)",
            "",
            "Components:",
            f"  I₁₂(+R) = {self.I12_plus:.10f}",
            f"  I₁₂(-R) = {self.I12_minus:.10f}",
            f"  I₃₄(+R) = {self.I34_plus:.10f}",
            "",
            f"Computed c = A*exp(R) + B = {self.c_computed:.10f}",
            "",
            f"Derived structure: {'YES' if self.is_derived_structure() else 'NO'}",
        ]
        if not self.is_derived_structure():
            d_gap, ba_gap = self.gap_from_derived()
            lines.append(f"  D gap: {d_gap:.6f}")
            lines.append(f"  B/A gap: {ba_gap:.6f}")
        return "\n".join(lines)


def compute_abd_decomposition(
    I12_plus: float,
    I12_minus: float,
    I34_plus: float,
    R: float,
    benchmark: str = "kappa"
) -> ABDDecomposition:
    """
    Compute the ABD decomposition from I-term values.

    This is the CANONICAL way to create an ABDDecomposition.

    Args:
        I12_plus: I₁₂(+R) value
        I12_minus: I₁₂(-R) value
        I34_plus: I₃₄(+R) value
        R: PRZZ R parameter
        benchmark: "kappa" or "kappa_star"

    Returns:
        ABDDecomposition with all fields computed
    """
    # Core decomposition
    A = I12_minus
    D = I12_plus + I34_plus
    B = D + 5 * A

    # Derived ratios (guard against division by zero)
    if abs(A) > 1e-15:
        B_over_A = B / A
        D_over_A = D / A
    else:
        B_over_A = float('inf')
        D_over_A = float('inf')

    # Computed c
    c_computed = A * np.exp(R) + B

    return ABDDecomposition(
        A=A,
        B=B,
        D=D,
        B_over_A=B_over_A,
        D_over_A=D_over_A,
        I12_plus=I12_plus,
        I12_minus=I12_minus,
        I34_plus=I34_plus,
        c_computed=c_computed,
        benchmark=benchmark,
        R=R,
    )


def compute_abd_from_evaluator(
    result: Dict,
    R: float,
    benchmark: str = "kappa"
) -> ABDDecomposition:
    """
    Extract ABD decomposition from an evaluator result dictionary.

    This handles the common case where results come from evaluate.py
    functions like compute_c_paper_with_mirror().

    Args:
        result: Dictionary with per_term breakdown
        R: PRZZ R parameter
        benchmark: "kappa" or "kappa_star"

    Returns:
        ABDDecomposition

    Expected keys in result.per_term:
        '_S12_plus_total' or 'S12_plus': I₁₂(+R)
        '_S12_minus_total' or 'S12_minus': I₁₂(-R)
        '_S34_total' or 'S34': I₃₄(+R)
    """
    per_term = result.get('per_term', result)

    # Try different key conventions
    I12_plus = per_term.get('_S12_plus_total', per_term.get('S12_plus', 0.0))
    I12_minus = per_term.get('_S12_minus_total', per_term.get('S12_minus', 0.0))
    I34_plus = per_term.get('_S34_total', per_term.get('S34', 0.0))

    return compute_abd_decomposition(
        I12_plus=I12_plus,
        I12_minus=I12_minus,
        I34_plus=I34_plus,
        R=R,
        benchmark=benchmark,
    )


# =============================================================================
# GATE TEST HELPERS
# =============================================================================


def check_derived_structure_gate(
    decomp: ABDDecomposition,
    d_tol: float = 1e-6,
    ba_tol: float = 1e-6
) -> Tuple[bool, str]:
    """
    Check if decomposition passes the derived structure gate.

    This is the CANONICAL gate test for Phase 21.

    Args:
        decomp: ABDDecomposition to check
        d_tol: Tolerance for D ~ 0
        ba_tol: Tolerance for B/A ~ 5

    Returns:
        (passed, message)
    """
    d_ok = abs(decomp.D) < d_tol
    ba_ok = abs(decomp.B_over_A - 5.0) < ba_tol

    passed = d_ok and ba_ok

    if passed:
        message = f"{decomp.benchmark}: PASS (D={decomp.D:.2e}, B/A={decomp.B_over_A:.6f})"
    else:
        parts = []
        if not d_ok:
            parts.append(f"D={decomp.D:.6f} (want <{d_tol})")
        if not ba_ok:
            parts.append(f"B/A={decomp.B_over_A:.6f} (want 5±{ba_tol})")
        message = f"{decomp.benchmark}: FAIL - " + ", ".join(parts)

    return passed, message


def run_dual_benchmark_gate(
    kappa_decomp: ABDDecomposition,
    kappa_star_decomp: ABDDecomposition,
    d_tol: float = 1e-6,
    ba_tol: float = 1e-6
) -> Tuple[bool, str]:
    """
    Run gate test on both benchmarks.

    Phase 21 requires BOTH benchmarks to pass.

    Args:
        kappa_decomp: ABDDecomposition for κ
        kappa_star_decomp: ABDDecomposition for κ*
        d_tol: Tolerance for D ~ 0
        ba_tol: Tolerance for B/A ~ 5

    Returns:
        (both_passed, combined_message)
    """
    k_passed, k_msg = check_derived_structure_gate(kappa_decomp, d_tol, ba_tol)
    ks_passed, ks_msg = check_derived_structure_gate(kappa_star_decomp, d_tol, ba_tol)

    both_passed = k_passed and ks_passed

    status = "GATE PASSED" if both_passed else "GATE FAILED"
    message = f"{status}\n  {k_msg}\n  {ks_msg}"

    return both_passed, message


# =============================================================================
# COMPARISON HELPERS
# =============================================================================


def compare_empirical_vs_derived(
    empirical: ABDDecomposition,
    derived: ABDDecomposition
) -> Dict[str, float]:
    """
    Compare empirical (current) vs derived (unified bracket) decompositions.

    This helps quantify the improvement from Phase 21.

    Args:
        empirical: ABDDecomposition from current pipeline
        derived: ABDDecomposition from unified bracket evaluator

    Returns:
        Dictionary with comparison metrics
    """
    return {
        "D_empirical": empirical.D,
        "D_derived": derived.D,
        "D_improvement": abs(empirical.D) - abs(derived.D),
        "BA_empirical": empirical.B_over_A,
        "BA_derived": derived.B_over_A,
        "BA_gap_empirical": abs(empirical.B_over_A - 5.0),
        "BA_gap_derived": abs(derived.B_over_A - 5.0),
        "c_empirical": empirical.c_computed,
        "c_derived": derived.c_computed,
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("ABD DIAGNOSTICS - DEFINITION VERIFICATION")
    print("=" * 60)
    print()

    # Example with synthetic values matching Phase 20.3 findings
    print("Example: Current production pipeline (empirical)")
    print("-" * 60)

    # κ benchmark values from Phase 20.3
    kappa_decomp = compute_abd_decomposition(
        I12_plus=0.80,    # Approximate from findings
        I12_minus=0.22,   # A ≈ 0.22
        I34_plus=-0.60,   # Approximate
        R=1.3036,
        benchmark="kappa"
    )
    print(kappa_decomp.summary())
    print()

    # Check gate
    passed, msg = check_derived_structure_gate(kappa_decomp)
    print(f"Gate check: {msg}")
    print()

    print("Target for Phase 21:")
    print("  D → 0")
    print("  B/A → 5")
    print("  Both benchmarks must pass")
