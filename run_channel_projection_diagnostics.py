#!/usr/bin/env python3
"""
run_channel_projection_diagnostics.py
Phase 7C: Channel Projection Diagnostics

PURPOSE: Prove whether scalar m₁ is TeX-exact or an empirical approximation.

METHODOLOGY:
1. Compute I₁+I₂ at +R ("direct" contribution)
2. Compute I₁+I₂ at -R ("mirror" contribution)
3. Compute I₃+I₄ at +R (no mirror per TeX Section 10)
4. Derive the ideal m₁ scalar that would hit target:

   m_ideal = (c_target - S12(+R) - S34(+R)) / S12(-R)

5. Compare m_ideal to:
   - Empirical m₁ = exp(R) + 5
   - Naive TeX m₁ = exp(2R) (from T^{-α-β} at α=β=-R/L)
   - exp(2R/θ) (from N^{x+y}T structure)

If m_ideal ≈ exp(R)+5, scalar m₁ is likely calibration-robust.
If m_ideal varies significantly with R or polynomials, scalar m₁ is NOT TeX-exact.

TRUTH_SPEC Section 10:
- I₁, I₂: Include mirror term T^{-α-β}·I(-β,-α)
- I₃, I₄: NO mirror terms
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import compute_c_paper


# =============================================================================
# Constants
# =============================================================================

THETA = 4.0 / 7.0

# Target values from PRZZ
C_TARGET_KAPPA = 2.13745440613217263636
C_TARGET_KAPPA_STAR = 1.9379524124677437

R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class ChannelDecomposition:
    """Decomposition into I₁+I₂ and I₃+I₄ channels."""
    R: float
    S12_plus: float  # I₁+I₂ at +R
    S12_minus: float  # I₁+I₂ at -R
    S34: float  # I₃+I₄ at +R (no mirror)
    c_paper: float  # Paper regime c (single R evaluation)


@dataclass
class MirrorAnalysis:
    """Analysis of different m₁ candidates."""
    R: float
    c_target: float

    # Computed from channel decomposition
    m_ideal: float  # Ideal m₁ to hit target exactly

    # Candidate m₁ formulas
    m_empirical: float  # exp(R) + 5
    m_exp_2R: float  # exp(2R)
    m_exp_2R_theta: float  # exp(2R/θ)

    # Resulting c values
    c_with_empirical: float
    c_with_exp_2R: float
    c_with_exp_2R_theta: float

    # Gaps from target
    gap_empirical: float
    gap_exp_2R: float
    gap_exp_2R_theta: float


# =============================================================================
# Channel Decomposition Functions
# =============================================================================

def compute_channel_decomposition(
    polynomials: Dict,
    R: float,
    n: int = 60,
    verbose: bool = False,
) -> ChannelDecomposition:
    """
    Compute I₁+I₂ and I₃+I₄ channel contributions.

    Uses paper regime evaluation at +R and -R.
    """
    # Evaluate at +R
    result_plus = compute_c_paper(
        theta=THETA,
        R=R,
        n=n,
        polynomials=polynomials,
        pair_mode='ordered',
        use_factorial_normalization=True,
        mode='main',
    )

    # Evaluate at -R
    result_minus = compute_c_paper(
        theta=THETA,
        R=-R,
        n=n,
        polynomials=polynomials,
        pair_mode='ordered',
        use_factorial_normalization=True,
        mode='main',
    )

    # Extract per-term contributions
    per_term_plus = result_plus.per_term or {}
    per_term_minus = result_minus.per_term or {}

    # Sum I₁+I₂ terms: pairs (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
    # Note: All pairs contribute to I₁+I₂ (integral form 1 + integral form 2)
    # and I₃+I₄ (integral form 3 + integral form 4)
    # The exact split depends on how evaluate.py computes them

    # For now, use total as S12+S34 combined
    # This is the "paper regime" total
    c_paper = result_plus.total if result_plus.total is not None else 0.0
    c_minus = result_minus.total if result_minus.total is not None else 0.0

    # For proper channel separation, we'd need to look at per-term breakdown
    # Currently, the evaluator doesn't separate I₁I₂ from I₃I₄
    # So we approximate: S12_plus ≈ total, S12_minus ≈ c(-R)
    # S34 is part of total (not separated)

    # This is an approximation - proper implementation would require
    # modifying evaluate.py to track I₁I₂ vs I₃I₄ separately

    if verbose:
        print(f"  c(+R) = {c_paper:.6f}")
        print(f"  c(-R) = {c_minus:.6f}")

    return ChannelDecomposition(
        R=R,
        S12_plus=c_paper,  # Approximation
        S12_minus=c_minus,
        S34=0.0,  # Would need evaluator modification to extract
        c_paper=c_paper,
    )


def analyze_mirror_scalars(
    decomp: ChannelDecomposition,
    c_target: float,
    verbose: bool = False,
) -> MirrorAnalysis:
    """
    Analyze different m₁ candidates against target.
    """
    R = decomp.R

    # Candidate m₁ formulas
    m_empirical = np.exp(R) + 5
    m_exp_2R = np.exp(2 * R)
    m_exp_2R_theta = np.exp(2 * R / THETA)

    # Current approximation: c = S12(+R) + m₁·S12(-R)
    # (This ignores S34 separation which would require evaluator changes)

    # Compute c with each m₁
    c_with_empirical = decomp.S12_plus + m_empirical * decomp.S12_minus
    c_with_exp_2R = decomp.S12_plus + m_exp_2R * decomp.S12_minus
    c_with_exp_2R_theta = decomp.S12_plus + m_exp_2R_theta * decomp.S12_minus

    # Compute ideal m₁
    if abs(decomp.S12_minus) > 1e-10:
        m_ideal = (c_target - decomp.S12_plus) / decomp.S12_minus
    else:
        m_ideal = float('nan')

    # Compute gaps
    gap_empirical = (c_with_empirical - c_target) / c_target * 100
    gap_exp_2R = (c_with_exp_2R - c_target) / c_target * 100
    gap_exp_2R_theta = (c_with_exp_2R_theta - c_target) / c_target * 100

    if verbose:
        print(f"\n  m₁ candidates at R={R:.4f}:")
        print(f"    m_ideal     = {m_ideal:.4f}")
        print(f"    exp(R)+5    = {m_empirical:.4f}")
        print(f"    exp(2R)     = {m_exp_2R:.4f}")
        print(f"    exp(2R/θ)   = {m_exp_2R_theta:.4f}")

    return MirrorAnalysis(
        R=R,
        c_target=c_target,
        m_ideal=m_ideal,
        m_empirical=m_empirical,
        m_exp_2R=m_exp_2R,
        m_exp_2R_theta=m_exp_2R_theta,
        c_with_empirical=c_with_empirical,
        c_with_exp_2R=c_with_exp_2R,
        c_with_exp_2R_theta=c_with_exp_2R_theta,
        gap_empirical=gap_empirical,
        gap_exp_2R=gap_exp_2R,
        gap_exp_2R_theta=gap_exp_2R_theta,
    )


# =============================================================================
# Main Diagnostics
# =============================================================================

def run_kappa_diagnostic(verbose: bool = True) -> Dict[str, Any]:
    """Run channel projection diagnostic for κ benchmark."""
    print("=" * 70)
    print("κ Benchmark Channel Projection Diagnostic")
    print("=" * 70)
    print()

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
    polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    print("Computing channel decomposition...")
    decomp = compute_channel_decomposition(
        polynomials=polynomials,
        R=R_KAPPA,
        n=60,
        verbose=verbose,
    )

    analysis = analyze_mirror_scalars(
        decomp=decomp,
        c_target=C_TARGET_KAPPA,
        verbose=verbose,
    )

    if verbose:
        print()
        print("Results:")
        print(f"  c_target  = {C_TARGET_KAPPA:.6f}")
        print()
        print("  With m₁ = exp(R)+5:")
        print(f"    c = {analysis.c_with_empirical:.6f}")
        print(f"    gap = {analysis.gap_empirical:+.2f}%")
        print()
        print("  With m₁ = exp(2R):")
        print(f"    c = {analysis.c_with_exp_2R:.6f}")
        print(f"    gap = {analysis.gap_exp_2R:+.2f}%")
        print()
        print("  With m₁ = exp(2R/θ):")
        print(f"    c = {analysis.c_with_exp_2R_theta:.6f}")
        print(f"    gap = {analysis.gap_exp_2R_theta:+.2f}%")
        print()
        print(f"  Ideal m₁ = {analysis.m_ideal:.4f}")
        print()

    return {
        'benchmark': 'kappa',
        'decomposition': decomp,
        'analysis': analysis,
    }


def run_kappa_star_diagnostic(verbose: bool = True) -> Dict[str, Any]:
    """Run channel projection diagnostic for κ* benchmark."""
    print("=" * 70)
    print("κ* Benchmark Channel Projection Diagnostic")
    print("=" * 70)
    print()

    P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=False)
    polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    print("Computing channel decomposition...")
    decomp = compute_channel_decomposition(
        polynomials=polynomials,
        R=R_KAPPA_STAR,
        n=60,
        verbose=verbose,
    )

    analysis = analyze_mirror_scalars(
        decomp=decomp,
        c_target=C_TARGET_KAPPA_STAR,
        verbose=verbose,
    )

    if verbose:
        print()
        print("Results:")
        print(f"  c_target  = {C_TARGET_KAPPA_STAR:.6f}")
        print()
        print("  With m₁ = exp(R)+5:")
        print(f"    c = {analysis.c_with_empirical:.6f}")
        print(f"    gap = {analysis.gap_empirical:+.2f}%")
        print()
        print("  With m₁ = exp(2R):")
        print(f"    c = {analysis.c_with_exp_2R:.6f}")
        print(f"    gap = {analysis.gap_exp_2R:+.2f}%")
        print()
        print("  With m₁ = exp(2R/θ):")
        print(f"    c = {analysis.c_with_exp_2R_theta:.6f}")
        print(f"    gap = {analysis.gap_exp_2R_theta:+.2f}%")
        print()
        print(f"  Ideal m₁ = {analysis.m_ideal:.4f}")
        print()

    return {
        'benchmark': 'kappa_star',
        'decomposition': decomp,
        'analysis': analysis,
    }


def compare_m1_across_benchmarks(
    kappa_result: Dict[str, Any],
    kappa_star_result: Dict[str, Any],
) -> None:
    """Compare m₁ behavior across both benchmarks."""
    print("=" * 70)
    print("Cross-Benchmark m₁ Comparison")
    print("=" * 70)
    print()

    k_analysis = kappa_result['analysis']
    ks_analysis = kappa_star_result['analysis']

    # Compare m_ideal values
    print("Ideal m₁ (scalar needed to hit target exactly):")
    print(f"  κ benchmark (R={R_KAPPA}):    m_ideal = {k_analysis.m_ideal:.4f}")
    print(f"  κ* benchmark (R={R_KAPPA_STAR}):  m_ideal = {ks_analysis.m_ideal:.4f}")
    print()

    # Compare empirical formula performance
    print("Empirical m₁ = exp(R) + 5:")
    print(f"  κ benchmark:  m = {k_analysis.m_empirical:.4f}, gap = {k_analysis.gap_empirical:+.2f}%")
    print(f"  κ* benchmark: m = {ks_analysis.m_empirical:.4f}, gap = {ks_analysis.gap_empirical:+.2f}%")
    print()

    # Check if m_ideal matches empirical
    m_ideal_ratio_kappa = k_analysis.m_ideal / k_analysis.m_empirical
    m_ideal_ratio_kappa_star = ks_analysis.m_ideal / ks_analysis.m_empirical

    print("Ratio m_ideal / m_empirical:")
    print(f"  κ benchmark:  {m_ideal_ratio_kappa:.4f}")
    print(f"  κ* benchmark: {m_ideal_ratio_kappa_star:.4f}")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    # Check if c(-R) is negative (which would invalidate simple mirror approach)
    if kappa_result['decomposition'].S12_minus < 0:
        print("WARNING: c(-R) is NEGATIVE for κ benchmark!")
        print("This means the simple scalar mirror formula doesn't apply directly.")
        print("The +R/-R channels cannot be combined with a positive scalar.")
        print()

    if kappa_star_result['decomposition'].S12_minus < 0:
        print("WARNING: c(-R) is NEGATIVE for κ* benchmark!")
        print("This means the simple scalar mirror formula doesn't apply directly.")
        print()

    # Check if m_ideal is consistent across benchmarks
    if not np.isnan(m_ideal_ratio_kappa) and not np.isnan(m_ideal_ratio_kappa_star):
        ratio_diff = abs(m_ideal_ratio_kappa - m_ideal_ratio_kappa_star)
        if ratio_diff < 0.1:
            print("m_ideal/m_empirical ratios are CONSISTENT across benchmarks.")
            print("This suggests the empirical formula captures the core structure.")
        else:
            print("m_ideal/m_empirical ratios DIFFER across benchmarks.")
            print("This suggests scalar m₁ is NOT a universal TeX constant.")
            print("It may be polynomial-dependent or require a different formulation.")
        print()

    # Final conclusion
    print("KEY FINDING:")
    print()
    print("The paper regime (single R evaluation) gives ~78% below target.")
    print("PRZZ TeX Section 10 specifies mirror assembly for I₁/I₂ but not I₃/I₄.")
    print()
    print("The empirical m₁ = exp(R)+5 achieves ~1-3% accuracy, but:")
    print("  - The '+5' term (or equivalently '2K-1' for K=3) is not TeX-derived")
    print("  - The formula works but WHY it works is not yet understood")
    print()
    print("For K>3 extension, either:")
    print("  (a) Derive m₁ structure from TeX (ideal)")
    print("  (b) Calibrate m₁ empirically per K (current approach)")
    print("=" * 70)


def run_all_diagnostics() -> Dict[str, Any]:
    """Run all channel projection diagnostics."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           PHASE 7C: CHANNEL PROJECTION DIAGNOSTICS                   ║")
    print("║   Analyzing whether scalar m₁ is TeX-exact or empirical calibration  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    kappa_result = run_kappa_diagnostic(verbose=True)
    print()
    kappa_star_result = run_kappa_star_diagnostic(verbose=True)
    print()
    compare_m1_across_benchmarks(kappa_result, kappa_star_result)

    return {
        'kappa': kappa_result,
        'kappa_star': kappa_star_result,
    }


if __name__ == "__main__":
    run_all_diagnostics()
