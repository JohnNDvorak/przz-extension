"""
src/unified_s12/correction_semantics.py
Phase 36: S12-level correction measurement.

This module measures the correction factor at the SAME semantic level
where the derived formula is applied: the mirror multiplier m.

Key insight from GPT:
- The 1.047 "aggregate correction" measured at I₁ level is NOT the same
  as the correction that multiplies m at the S12 channel level.
- We need to measure: m_needed := (c_target - S12_plus - S34) / S12_minus
- Then: corr_needed := m_needed / (exp(R) + (2K-1))

This should match the Beta moment prediction when Q=1, P=real.

Created: 2025-12-26 (Phase 36)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math


@dataclass
class CorrectionSemantics:
    """Results of S12-level correction measurement."""

    # Decomposition components
    S12_plus: float
    S12_minus: float
    S34: float

    # Target and computed c
    c_target: float
    c_computed: float

    # The correction at m level
    m_needed: float           # (c_target - S12_plus - S34) / S12_minus
    m_base: float             # exp(R) + (2K-1)
    corr_needed: float        # m_needed / m_base

    # Predictions
    corr_beta: float          # 1 + θ/(2K(2K+1)) - Beta moment prediction
    corr_derived: float       # What the "derived" formula uses

    # Parameters
    R: float
    theta: float
    K: int

    @property
    def gap_from_beta(self) -> float:
        """Gap between needed correction and Beta moment, as percentage."""
        return (self.corr_needed / self.corr_beta - 1) * 100

    @property
    def delta_Q(self) -> float:
        """
        The Q-induced deviation from Beta moment.

        delta_Q = corr_needed - corr_beta

        If delta_Q < 0, the Beta moment overcorrects (makes m too large, c too large).
        If delta_Q > 0, the Beta moment undercorrects (makes m too small, c too small).
        """
        return self.corr_needed - self.corr_beta


def compute_correction_semantics(
    theta: float,
    R: float,
    K: int,
    polynomials: Dict,
    c_target: float,
    *,
    kernel_regime: str = "paper",
    n_quad: int = 60,
) -> CorrectionSemantics:
    """
    Compute the S12-level correction needed to match c_target.

    This measures the correction at the level where m is applied:
        c = S12_plus + m × S12_minus + S34

    Solving for m:
        m_needed = (c_target - S12_plus - S34) / S12_minus

    The correction factor is:
        corr_needed = m_needed / (exp(R) + (2K-1))

    This should equal the Beta moment prediction 1 + θ/(2K(2K+1))
    when the derivation is complete.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        K: Number of mollifier pieces
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        c_target: The target c value from PRZZ
        kernel_regime: "paper" or "raw"
        n_quad: Quadrature points

    Returns:
        CorrectionSemantics with all measurements
    """
    from src.mirror_transform_paper_exact import compute_S12_paper_sum
    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term
    from src.evaluator.decomposition import compute_mirror_multiplier

    # 1. Compute S12 at +R and -R
    S12_plus = compute_S12_paper_sum(R, theta, polynomials, n_quad=n_quad)
    S12_minus = compute_S12_paper_sum(-R, theta, polynomials, n_quad=n_quad)

    # 2. Compute S34
    all_terms = make_all_terms_k3(theta, R, kernel_regime=kernel_regime)

    factorial_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }
    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0,
    }

    S34 = 0.0
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = all_terms[pair_key]
        norm = factorial_norm[pair_key]
        sym = symmetry_factor[pair_key]
        full_norm = sym * norm

        # I₃ and I₄ are indices 2 and 3
        for term in terms[2:4]:
            result = evaluate_term(term, polynomials, n_quad, R=R, theta=theta)
            S34 += full_norm * result.value

    # 3. Compute m_needed to match c_target
    # c_target = S12_plus + m_needed × S12_minus + S34
    # m_needed = (c_target - S12_plus - S34) / S12_minus
    if abs(S12_minus) < 1e-15:
        raise ValueError("S12_minus is too small to compute m_needed")

    m_needed = (c_target - S12_plus - S34) / S12_minus

    # 4. Compute m_base (empirical formula without correction)
    m_base = math.exp(R) + (2 * K - 1)

    # 5. Compute correction needed
    corr_needed = m_needed / m_base

    # 6. Compute predictions
    corr_beta = 1 + theta / (2 * K * (2 * K + 1))
    m_derived, _ = compute_mirror_multiplier(R, K, formula="derived")
    corr_derived = m_derived / m_base

    # 7. Compute c with derived formula for comparison
    c_computed = S12_plus + m_derived * S12_minus + S34

    return CorrectionSemantics(
        S12_plus=S12_plus,
        S12_minus=S12_minus,
        S34=S34,
        c_target=c_target,
        c_computed=c_computed,
        m_needed=m_needed,
        m_base=m_base,
        corr_needed=corr_needed,
        corr_beta=corr_beta,
        corr_derived=corr_derived,
        R=R,
        theta=theta,
        K=K,
    )


def print_correction_semantics_report(
    theta: float = 4/7,
    R: float = 1.3036,
    K: int = 3,
    c_target: float = 2.137454406132173,
    polynomials: Optional[Dict] = None,
    n_quad: int = 60,
) -> CorrectionSemantics:
    """
    Print a report comparing S12-level correction to Beta moment.

    Returns the CorrectionSemantics for programmatic use.
    """
    from src.polynomials import load_przz_polynomials

    if polynomials is None:
        P1, P2, P3, Q = load_przz_polynomials()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    cs = compute_correction_semantics(
        theta=theta,
        R=R,
        K=K,
        polynomials=polynomials,
        c_target=c_target,
        n_quad=n_quad,
    )

    print("=" * 70)
    print("S12-LEVEL CORRECTION SEMANTICS (Phase 36)")
    print("=" * 70)
    print()
    print(f"Parameters: θ={theta:.6f}, R={R}, K={K}")
    print(f"c_target = {c_target:.10f}")
    print()

    print("DECOMPOSITION")
    print("-" * 50)
    print(f"  S12(+R):   {cs.S12_plus:+.10f}")
    print(f"  S12(-R):   {cs.S12_minus:+.10f}")
    print(f"  S34:       {cs.S34:+.10f}")
    print()

    print("M-LEVEL ANALYSIS")
    print("-" * 50)
    print(f"  m_base = exp(R) + (2K-1) = {cs.m_base:.8f}")
    print(f"  m_needed = (c_target - S12_plus - S34) / S12_minus")
    print(f"           = ({c_target:.6f} - {cs.S12_plus:.6f} - ({cs.S34:.6f})) / {cs.S12_minus:.6f}")
    print(f"           = {cs.m_needed:.8f}")
    print()

    print("CORRECTION FACTORS")
    print("-" * 50)
    print(f"  corr_needed = m_needed / m_base = {cs.corr_needed:.10f}")
    print(f"  corr_beta   = 1 + θ/(2K(2K+1)) = {cs.corr_beta:.10f}")
    print(f"  corr_derived (current) = {cs.corr_derived:.10f}")
    print()

    print("GAP ANALYSIS")
    print("-" * 50)
    gap_pct = cs.gap_from_beta
    print(f"  Gap from Beta: {gap_pct:+.4f}%")
    print(f"  delta_Q = corr_needed - corr_beta = {cs.delta_Q:+.10f}")
    print()

    if cs.delta_Q < 0:
        print("  Interpretation: Beta moment OVERCORRECTS")
        print("    → Makes m too large → c too large → κ too low")
        print("    → Need to REDUCE the correction factor")
    else:
        print("  Interpretation: Beta moment UNDERCORRECTS")
        print("    → Makes m too small → c too small → κ too high")
        print("    → Need to INCREASE the correction factor")
    print()

    print("C VALUE CHECK")
    print("-" * 50)
    c_gap = (cs.c_computed / cs.c_target - 1) * 100
    print(f"  c_computed (derived formula): {cs.c_computed:.10f}")
    print(f"  c_target:                     {cs.c_target:.10f}")
    print(f"  c_gap:                        {c_gap:+.4f}%")
    print()

    return cs


if __name__ == "__main__":
    print_correction_semantics_report()
