"""
src/przz_exact_assembly.py
PRZZ Exact Assembly Function

This module assembles I₁, I₂, I₃, I₄ to compute the main-term constant c
and the κ bound using PRZZ's exact formulas from TeX lines 1500-1570.

CRITICAL FINDING (2025-12-29):
==============================
The PRZZ exact formulas give values ~2.75x larger than the "Paper regime"
implementation that uses Case C kernel attenuation for P₂/P₃.

The split-channel formula c = S12+ + m×S12- + S34+ with m = exp(R)+5
was empirically validated with Paper regime values, NOT with PRZZ exact values.

Status: c_computed from PRZZ exact is ~50% of c_target for κ benchmark.
        Further investigation needed on the normalization.

Created: 2025-12-29
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional

from src.przz_exact_i1 import compute_I1_all_pairs, I1Result
from src.przz_exact_i2 import compute_I2_all_pairs, I2Result
from src.przz_exact_i34 import compute_I34_all_pairs, I34Result


@dataclass
class PRZZExactResult:
    """Complete result from PRZZ exact computation."""
    # Raw integral totals (sum over pairs with factorial normalization)
    I1_total: float
    I2_total: float
    I3_total: float
    I4_total: float

    # Computed c and κ
    c: float
    kappa: float

    # Parameters
    R: float
    theta: float

    # Targets for comparison
    c_target: Optional[float] = None
    kappa_target: Optional[float] = None

    # Per-pair breakdowns
    I1_by_pair: Optional[Dict[str, I1Result]] = None
    I2_by_pair: Optional[Dict[str, I2Result]] = None
    I3_by_pair: Optional[Dict[str, I34Result]] = None
    I4_by_pair: Optional[Dict[str, I34Result]] = None

    @property
    def S12(self) -> float:
        """I₁ + I₂ combined."""
        return self.I1_total + self.I2_total

    @property
    def S34(self) -> float:
        """I₃ + I₄ combined."""
        return self.I3_total + self.I4_total

    @property
    def c_gap_pct(self) -> Optional[float]:
        """Percentage gap from c_target."""
        if self.c_target is None:
            return None
        return (self.c - self.c_target) / self.c_target * 100

    @property
    def kappa_gap_pct(self) -> Optional[float]:
        """Percentage gap from kappa_target."""
        if self.kappa_target is None:
            return None
        return (self.kappa - self.kappa_target) / self.kappa_target * 100


def compute_przz_exact(
    theta: float,
    R: float,
    polynomials: Dict,
    n_quad: int = 80,
    c_target: Optional[float] = None,
    kappa_target: Optional[float] = None,
    store_per_pair: bool = False,
) -> PRZZExactResult:
    """
    Compute c and κ using PRZZ exact formulas.

    Formula: c = I₁ + I₂ + I₃ + I₄ (summed over all pairs with normalization)

    Note: This gives values ~50% of PRZZ target. See module docstring.

    Args:
        theta: PRZZ θ parameter (= 4/7)
        R: PRZZ R parameter
        polynomials: Dict with P1, P2, P3, Q polynomial objects
        n_quad: Number of quadrature points
        c_target: Optional target c for comparison
        kappa_target: Optional target κ for comparison
        store_per_pair: If True, store per-pair results

    Returns:
        PRZZExactResult with all computed values
    """
    # Compute all pairs
    I1_results = compute_I1_all_pairs(theta, R, polynomials, n_quad)
    I2_results = compute_I2_all_pairs(theta, R, polynomials, n_quad)
    I34_results = compute_I34_all_pairs(theta, R, polynomials, n_quad)

    # Sum with factorial normalization
    I1_total = 0.0
    I2_total = 0.0
    I3_total = 0.0
    I4_total = 0.0

    for key in ["11", "22", "33", "12", "13", "23"]:
        r1 = I1_results[key]
        r2 = I2_results[key]
        r3 = I34_results["I3"][key]
        r4 = I34_results["I4"][key]

        # Symmetry factor for off-diagonal
        sym = 2.0 if r1.ell1 != r1.ell2 else 1.0
        # Factorial normalization
        norm = 1.0 / (math.factorial(r1.ell1) * math.factorial(r1.ell2))

        I1_total += sym * norm * r1.value
        I2_total += sym * norm * r2.value
        I3_total += sym * norm * r3.value
        I4_total += sym * norm * r4.value

    # Assemble c
    c = I1_total + I2_total + I3_total + I4_total

    # Compute κ
    if c > 0:
        kappa = 1 - math.log(c) / R
    else:
        kappa = float('nan')

    return PRZZExactResult(
        I1_total=I1_total,
        I2_total=I2_total,
        I3_total=I3_total,
        I4_total=I4_total,
        c=c,
        kappa=kappa,
        R=R,
        theta=theta,
        c_target=c_target,
        kappa_target=kappa_target,
        I1_by_pair=I1_results if store_per_pair else None,
        I2_by_pair=I2_results if store_per_pair else None,
        I3_by_pair=I34_results["I3"] if store_per_pair else None,
        I4_by_pair=I34_results["I4"] if store_per_pair else None,
    )


def print_przz_exact_report(result: PRZZExactResult) -> None:
    """Print a detailed report of PRZZ exact computation."""
    print("=" * 60)
    print(f"PRZZ EXACT COMPUTATION REPORT")
    print(f"R = {result.R}, θ = {result.theta:.6f}")
    print("=" * 60)

    print(f"\n  Integral totals (with factorial normalization):")
    print(f"    I₁ = {result.I1_total:>12.6f}")
    print(f"    I₂ = {result.I2_total:>12.6f}")
    print(f"    I₃ = {result.I3_total:>12.6f}")
    print(f"    I₄ = {result.I4_total:>12.6f}")
    print(f"    ─────────────────────")
    print(f"    c  = {result.c:>12.6f}")

    print(f"\n  Combined totals:")
    print(f"    S12 = I₁+I₂ = {result.S12:.6f}")
    print(f"    S34 = I₃+I₄ = {result.S34:.6f}")

    print(f"\n  κ = 1 - log(c)/R = {result.kappa:.6f}")

    if result.c_target is not None:
        print(f"\n  Comparison to target:")
        print(f"    c_target   = {result.c_target:.6f}")
        print(f"    c gap      = {result.c_gap_pct:+.2f}%")
    if result.kappa_target is not None:
        print(f"    κ_target   = {result.kappa_target:.6f}")
        print(f"    κ gap      = {result.kappa_gap_pct:+.2f}%")


if __name__ == "__main__":
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

    theta = 4.0 / 7.0

    print("=" * 70)
    print("PRZZ EXACT ASSEMBLY TEST")
    print("=" * 70)

    for name, R, c_target, kappa_target, loader in [
        ("kappa", 1.3036, 2.137, 0.417293962, load_przz_polynomials),
        ("kappa_star", 1.1167, 1.938, 0.407511457, load_przz_polynomials_kappa_star),
    ]:
        print(f"\n\n{'='*60}")
        print(f"BENCHMARK: {name.upper()}")
        print(f"{'='*60}")

        P1, P2, P3, Q = loader()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        result = compute_przz_exact(
            theta, R, polynomials, n_quad=80,
            c_target=c_target, kappa_target=kappa_target
        )

        print_przz_exact_report(result)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  The PRZZ exact formulas give c values ~50% of target.

  This is because the split-channel formula c = S12+ + m×S12- + S34+
  was calibrated with "Paper regime" values that use Case C kernel
  attenuation, NOT with the raw PRZZ formulas.

  The relationship between PRZZ exact and Paper regime needs further
  investigation. The key difference is that Paper regime uses
  K_omega kernel attenuation for P₂ and P₃ (pieces with ℓ ≥ 2).

  Next steps:
  1. Investigate if PRZZ uses Case C kernels internally
  2. Find the correct normalization/assembly formula for PRZZ exact
  3. Or, continue using Paper regime which is empirically validated
""")
