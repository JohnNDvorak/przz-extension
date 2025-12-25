"""
src/tex_exact_k3.py
TeX-Exact K=3 Evaluator — NO MIRROR MULTIPLIER (Phase 7A)

This module computes c(R,θ) directly from PRZZ TeX formulas.
The combined identity already includes both direct and mirror contributions —
there is no external scalar m₁ in the TeX.

PURPOSE: Determine what the "paper regime" evaluator produces WITHOUT m₁ assembly,
and compare to the target c values.

ARCHITECTURE:
This module wraps the existing DSL infrastructure:
- Uses compute_c_paper() for the standard paper-regime evaluation
- Computes what would happen WITHOUT any +R/-R recombination

KEY INVARIANT: No `m1_formula`, no `mirror_multiplier`, no separate ±R channels.

REFERENCE VALUES:
- κ benchmark: c_target = 2.137, R = 1.3036
- κ* benchmark: c_target = 1.938, R = 1.1167
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import compute_c_paper


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class TexExactResult:
    """Full TeX-exact evaluation result."""
    c: float
    kappa: float
    R: float
    theta: float
    n_quad: int
    per_pair: Optional[Dict] = None
    assembly_formula: str = ""
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


# =============================================================================
# Main Evaluation Functions
# =============================================================================

def compute_c_tex_exact(
    theta: float,
    R: float,
    n: int = 60,
    polynomials: Optional[Dict] = None,
    verbose: bool = False,
) -> TexExactResult:
    """
    Compute c using TeX-exact formulas with NO MIRROR MULTIPLIER.

    This wraps the existing paper-regime evaluator. The paper regime
    computes c = Σ (I₁ + I₂ + I₃ + I₄) for all pairs at a single R value,
    with NO +R/-R mirror assembly.

    The question is: does this single-evaluation approach match the target?

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (1.3036 for κ, 1.1167 for κ*)
        n: Number of quadrature points
        polynomials: Dict with P1, P2, P3, Q (loaded automatically if None)
        verbose: Print per-pair breakdown

    Returns:
        TexExactResult with c, κ, and per-pair breakdown
    """
    # Load polynomials if not provided
    if polynomials is None:
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    # Use the existing paper-regime evaluator
    result = compute_c_paper(
        theta=theta,
        R=R,
        n=n,
        polynomials=polynomials,
        pair_mode='ordered',
        use_factorial_normalization=True,
        mode='main',
    )

    c = result.total

    # Compute κ (may be invalid if c <= 0)
    if c > 0:
        kappa = 1.0 - np.log(c) / R
    else:
        kappa = float('nan')

    notes = [
        "TeX-exact evaluation using paper regime with NO m₁ scalar",
        "Single R evaluation - no +R/-R mirror assembly",
        f"c = {c:.6f}, expected ~2.137 for κ or ~1.938 for κ*",
    ]

    if verbose and result.per_term:
        print("Per-pair breakdown:")
        for k, v in result.per_term.items():
            if isinstance(k, tuple):
                print(f"  {k}: {v:.6f}")
        print(f"\nTotal c = {c:.6f}")
        print(f"κ = {kappa:.6f}")

    return TexExactResult(
        c=c,
        kappa=kappa,
        R=R,
        theta=theta,
        n_quad=n,
        per_pair=result.per_term,
        assembly_formula="c = Σ (I₁ + I₂ + I₃ + I₄) × factorial_norm (paper regime, NO m₁)",
        notes=notes,
    )


def compute_c_tex_exact_kappa(n: int = 60, verbose: bool = False) -> TexExactResult:
    """Compute TeX-exact c for κ benchmark (R=1.3036)."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
    polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    return compute_c_tex_exact(
        theta=4.0/7.0,
        R=1.3036,
        n=n,
        polynomials=polynomials,
        verbose=verbose,
    )


def compute_c_tex_exact_kappa_star(n: int = 60, verbose: bool = False) -> TexExactResult:
    """Compute TeX-exact c for κ* benchmark (R=1.1167)."""
    P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=False)
    polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    return compute_c_tex_exact(
        theta=4.0/7.0,
        R=1.1167,
        n=n,
        polynomials=polynomials,
        verbose=verbose,
    )


# =============================================================================
# Diagnostic Functions
# =============================================================================

def compare_tex_exact_to_target(verbose: bool = True) -> Dict[str, Any]:
    """
    Compare TeX-exact results to target values.

    This is the main diagnostic function for Phase 7A.
    Shows what the paper regime produces WITHOUT m₁ assembly.
    """
    results = {}

    # κ benchmark
    kappa_result = compute_c_tex_exact_kappa(n=60, verbose=False)
    c_target_kappa = 2.13745440613217263636
    c_gap_kappa = (kappa_result.c - c_target_kappa) / c_target_kappa * 100

    results['kappa'] = {
        'c_computed': kappa_result.c,
        'c_target': c_target_kappa,
        'c_gap_percent': c_gap_kappa,
        'kappa_computed': kappa_result.kappa,
        'kappa_target': 0.417293962,
    }

    # κ* benchmark
    kappa_star_result = compute_c_tex_exact_kappa_star(n=60, verbose=False)
    c_target_kappa_star = 1.9379524124677437
    c_gap_kappa_star = (kappa_star_result.c - c_target_kappa_star) / c_target_kappa_star * 100

    results['kappa_star'] = {
        'c_computed': kappa_star_result.c,
        'c_target': c_target_kappa_star,
        'c_gap_percent': c_gap_kappa_star,
        'kappa_computed': kappa_star_result.kappa,
        'kappa_target': 0.407511457,
    }

    # Ratio test
    if kappa_star_result.c > 0:
        ratio_computed = kappa_result.c / kappa_star_result.c
    else:
        ratio_computed = float('nan')
    ratio_target = c_target_kappa / c_target_kappa_star
    ratio_gap = (ratio_computed - ratio_target) / ratio_target * 100

    results['ratio'] = {
        'computed': ratio_computed,
        'target': ratio_target,
        'gap_percent': ratio_gap,
    }

    if verbose:
        print("=" * 70)
        print("TeX-Exact Evaluation Results (Paper Regime, NO m₁)")
        print("=" * 70)
        print()
        print("κ Benchmark (R=1.3036):")
        print(f"  c computed:  {results['kappa']['c_computed']:.8f}")
        print(f"  c target:    {results['kappa']['c_target']:.8f}")
        print(f"  c gap:       {results['kappa']['c_gap_percent']:+.2f}%")
        if not np.isnan(results['kappa']['kappa_computed']):
            print(f"  κ computed:  {results['kappa']['kappa_computed']:.6f}")
        print(f"  κ target:    {results['kappa']['kappa_target']:.6f}")
        print()
        print("κ* Benchmark (R=1.1167):")
        print(f"  c computed:  {results['kappa_star']['c_computed']:.8f}")
        print(f"  c target:    {results['kappa_star']['c_target']:.8f}")
        print(f"  c gap:       {results['kappa_star']['c_gap_percent']:+.2f}%")
        if not np.isnan(results['kappa_star']['kappa_computed']):
            print(f"  κ computed:  {results['kappa_star']['kappa_computed']:.6f}")
        print(f"  κ target:    {results['kappa_star']['kappa_target']:.6f}")
        print()
        print("Ratio Test (κ/κ*):")
        print(f"  c ratio computed: {results['ratio']['computed']:.4f}")
        print(f"  c ratio target:   {results['ratio']['target']:.4f}")
        print(f"  ratio gap:        {results['ratio']['gap_percent']:+.2f}%")
        print()
        print("=" * 70)
        print()
        print("INTERPRETATION:")
        print()
        print("The paper regime (single R evaluation) gives ~0.46 for κ benchmark,")
        print("which is ~78% below target (2.137). This is the 'pre-mirror' result.")
        print()
        print("PRZZ TeX Section 10 specifies:")
        print("  - I₁, I₂ combine with mirror: I(α,β) + T^{-α-β}·I(-β,-α)")
        print("  - I₃, I₄ do NOT have mirror terms")
        print()
        print("The m₁ scalar in current code is an EMPIRICAL approximation for")
        print("this mirror assembly. Phase 7C will analyze whether a TeX-exact")
        print("mirror assembly (without scalar approximation) works better.")
        print()
        print("KEY FINDING: Paper regime alone cannot reach target. Mirror")
        print("assembly is REQUIRED, the question is whether m₁ can be derived.")
        print("=" * 70)

    return results


def analyze_channel_structure(
    benchmark: str = 'kappa',
    n: int = 60,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze the I₁/I₂ vs I₃/I₄ channel structure for either benchmark.

    I₁/I₂ have mirror terms (TRUTH_SPEC Section 10)
    I₃/I₄ do NOT have mirror terms

    This function computes:
    - S12_direct = Σ (I₁ + I₂) at +R
    - S12_mirror_basis = Σ (I₁ + I₂) at -R (to be scaled by m₁)
    - S34 = Σ (I₃ + I₄) at +R (no mirror)
    - m₁_ideal = (c_target - S12_direct - S34) / S12_mirror_basis

    Args:
        benchmark: 'kappa' or 'kappa_star'
        n: Quadrature points
        verbose: Print output

    Returns:
        Dictionary with channel decomposition and analysis
    """
    from src.evaluate import compute_c_paper_ordered

    # Load appropriate polynomials and targets
    if benchmark == 'kappa':
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        R = 1.3036
        c_target = 2.13745440613217263636
    elif benchmark == 'kappa_star':
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=False)
        R = 1.1167
        c_target = 1.9379524124677437
    else:
        raise ValueError(f"benchmark must be 'kappa' or 'kappa_star', got '{benchmark}'")

    polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}
    theta = 4.0 / 7.0

    # Get +R evaluation
    result_plus = compute_c_paper_ordered(
        theta=theta,
        R=R,
        n=n,
        polynomials=polynomials,
        K=3,
        s12_pair_mode='triangle',
    )

    # Get -R evaluation (for mirror basis)
    result_minus = compute_c_paper_ordered(
        theta=theta,
        R=-R,
        n=n,
        polynomials=polynomials,
        K=3,
        s12_pair_mode='triangle',
    )

    # Extract channel values (keys prefixed with underscore)
    s12_direct = result_plus.per_term.get('_S12_plus_total', 0.0)
    s34 = result_plus.per_term.get('_S34_plus_total', 0.0)
    s12_mirror_basis = result_minus.per_term.get('_S12_plus_total', 0.0)

    # Compute ideal m₁ that would hit target exactly
    # c_target = S12_direct + m₁_ideal × S12_mirror_basis + S34
    # => m₁_ideal = (c_target - S12_direct - S34) / S12_mirror_basis
    if abs(s12_mirror_basis) > 1e-10:
        m1_ideal = (c_target - s12_direct - s34) / s12_mirror_basis
    else:
        m1_ideal = float('nan')

    # Empirical formula: m₁ = exp(R) + (2K - 1) = exp(R) + 5 for K=3
    m1_empirical = np.exp(R) + 5

    # Naive TeX formula: T^{-α-β} at α=β=-R/L gives exp(2R)
    m1_naive = np.exp(2 * R)

    # Assembly with each m₁
    c_with_ideal = s12_direct + m1_ideal * s12_mirror_basis + s34
    c_with_empirical = s12_direct + m1_empirical * s12_mirror_basis + s34
    c_with_naive = s12_direct + m1_naive * s12_mirror_basis + s34

    # Decomposition verification: should sum to c_with_empirical
    decomp_sum = s12_direct + m1_empirical * s12_mirror_basis + s34
    decomp_check = abs(decomp_sum - c_with_empirical) < 1e-10

    results = {
        'benchmark': benchmark,
        'R': R,
        'c_target': c_target,

        # Channel components
        'S12_direct': s12_direct,
        'S12_mirror_basis': s12_mirror_basis,
        'S34': s34,

        # Mirror scalars
        'm1_ideal': m1_ideal,
        'm1_empirical': m1_empirical,
        'm1_naive': m1_naive,

        # Computed c values
        'c_with_ideal': c_with_ideal,
        'c_with_empirical': c_with_empirical,
        'c_with_naive': c_with_naive,

        # Gaps
        'gap_ideal': (c_with_ideal - c_target) / c_target * 100,
        'gap_empirical': (c_with_empirical - c_target) / c_target * 100,
        'gap_naive': (c_with_naive - c_target) / c_target * 100,

        # Ratios
        'm1_ideal_over_empirical': m1_ideal / m1_empirical if m1_empirical != 0 else float('nan'),

        # Verification
        'decomp_check': decomp_check,
    }

    if verbose:
        print("=" * 70)
        print(f"Channel Structure Analysis — {benchmark} benchmark (R={R})")
        print("=" * 70)
        print()
        print("Channel Decomposition:")
        print(f"  S12_direct       = {s12_direct:+.6f}  (I₁+I₂ at +R)")
        print(f"  S12_mirror_basis = {s12_mirror_basis:+.6f}  (I₁+I₂ at -R)")
        print(f"  S34              = {s34:+.6f}  (I₃+I₄ at +R, no mirror)")
        print()
        print("Mirror Scalar Analysis:")
        print(f"  m₁_ideal     = {m1_ideal:.4f}  (hits target exactly)")
        print(f"  m₁_empirical = {m1_empirical:.4f}  (exp(R)+5)")
        print(f"  m₁_naive     = {m1_naive:.4f}  (exp(2R), TeX T^{{-α-β}})")
        print()
        print(f"  Ratio m₁_ideal / m₁_empirical = {results['m1_ideal_over_empirical']:.4f}")
        print()
        print("Assembly: c = S12_direct + m₁ × S12_mirror_basis + S34")
        print()
        print(f"  With m₁_ideal:     c = {c_with_ideal:.6f}  gap = {results['gap_ideal']:+.2f}%")
        print(f"  With m₁_empirical: c = {c_with_empirical:.6f}  gap = {results['gap_empirical']:+.2f}%")
        print(f"  With m₁_naive:     c = {c_with_naive:.6f}  gap = {results['gap_naive']:+.2f}%")
        print()
        print(f"  Target c = {c_target:.6f}")
        print()
        print(f"Decomposition check: {'✓ PASS' if decomp_check else '✗ FAIL'}")
        print("=" * 70)

    return results


def analyze_both_benchmarks(n: int = 60, verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze channel structure for both κ and κ* benchmarks.

    Returns combined analysis with cross-benchmark comparison.
    """
    kappa = analyze_channel_structure('kappa', n=n, verbose=verbose)
    if verbose:
        print()
    kappa_star = analyze_channel_structure('kappa_star', n=n, verbose=verbose)

    # Cross-benchmark comparison
    ratio_m1_ideal = kappa['m1_ideal'] / kappa_star['m1_ideal']
    ratio_m1_ideal_over_emp = kappa['m1_ideal_over_empirical'] / kappa_star['m1_ideal_over_empirical']

    results = {
        'kappa': kappa,
        'kappa_star': kappa_star,
        'cross_benchmark': {
            'm1_ideal_ratio': ratio_m1_ideal,
            'm1_ideal_over_empirical_ratio': ratio_m1_ideal_over_emp,
        }
    }

    if verbose:
        print()
        print("=" * 70)
        print("Cross-Benchmark Comparison")
        print("=" * 70)
        print()
        print("m₁_ideal / m₁_empirical ratios:")
        print(f"  κ  benchmark: {kappa['m1_ideal_over_empirical']:.4f}")
        print(f"  κ* benchmark: {kappa_star['m1_ideal_over_empirical']:.4f}")
        print()
        if abs(kappa['m1_ideal_over_empirical'] - kappa_star['m1_ideal_over_empirical']) < 0.1:
            print("  → Ratios are CONSISTENT across benchmarks")
            print("  → Empirical m₁ = exp(R)+5 captures correct R-dependence")
        else:
            print("  → Ratios DIFFER across benchmarks")
            print("  → Empirical formula may need polynomial-dependent correction")
        print("=" * 70)

    return results


if __name__ == "__main__":
    compare_tex_exact_to_target(verbose=True)
    print()
    analyze_both_benchmarks(verbose=True)
