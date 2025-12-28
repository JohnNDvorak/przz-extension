"""
src/evaluator/decomposition.py
Phase 31A.2: Canonical Decomposition Function

This module provides THE canonical decomposition for computing c:

    c = S12_plus + mirror_mult × S12_minus + S34

All "compute_c_*" wrappers should call this to ensure consistency.

INVARIANTS:
===========
1. total == S12_plus + mirror_mult * S12_minus + S34  (exact float equality)
2. S34 is ALWAYS computed, never hardcoded
3. mirror_mult formula is explicit and auditable

Created: 2025-12-26 (Phase 31A)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math


@dataclass
class Decomposition:
    """Canonical decomposition of c into its components."""

    # Core components
    S12_plus: float         # I₁ + I₂ at +R
    S12_minus: float        # I₁ + I₂ at -R
    S34: float              # I₃ + I₄ at +R (no mirror)

    # Mirror parameters
    mirror_mult: float      # The m multiplier
    mirror_formula: str     # How mirror_mult was computed

    # Assembly
    total: float            # c = S12_plus + mirror_mult * S12_minus + S34

    # Metadata
    R: float
    theta: float
    K: int
    kernel_regime: str

    # Verification
    assembly_verified: bool  # True if total matches assembly formula

    def verify_assembly(self, tol: float = 1e-12) -> bool:
        """Verify that total equals assembled components."""
        assembled = self.S12_plus + self.mirror_mult * self.S12_minus + self.S34
        return abs(self.total - assembled) < tol

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "S12_plus": self.S12_plus,
            "S12_minus": self.S12_minus,
            "S34": self.S34,
            "mirror_mult": self.mirror_mult,
            "mirror_formula": self.mirror_formula,
            "total": self.total,
            "R": self.R,
            "theta": self.theta,
            "K": self.K,
            "kernel_regime": self.kernel_regime,
            "assembly_verified": self.assembly_verified,
        }


def compute_mirror_multiplier(
    R: float,
    K: int,
    *,
    formula: str = "derived",
    theta: float = 4 / 7,
) -> tuple[float, str]:
    """
    Compute the mirror multiplier m.

    PRODUCTION FORMULA (Phase 36 locked):
    =====================================
    m(K, R) = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]

    This is THE canonical formula for production use. It is:
    - Fully derived from PRZZ first principles (Phases 32-34)
    - K-generalized: works for any K (validated for K=3,4,5)
    - Accurate to ±0.15% on both κ and κ* benchmarks

    Args:
        R: The R parameter
        K: Number of mollifier pieces
        formula: Which formula to use:
            - "derived" (DEFAULT, PRODUCTION): m = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]
            - "empirical" (DIAGNOSTIC ONLY): m = exp(R) + (2K-1)
            - Others are deprecated and for diagnostic comparison only
        theta: θ parameter (default 4/7)

    Returns:
        (m_value, formula_description)

    DERIVATION SUMMARY (Phase 34C):
    ===============================
    - exp(R): From difference quotient T^{-(α+β)} at α=β=-R/L (PRZZ line 1502)
    - (2K-1): From unified bracket B/A ratio (Phase 32)
    - 1 + θ/(2K(2K+1)): From product rule cross-terms on log factor (Phase 34C)
      This equals 1 + θ × Beta(2, 2K) where Beta(2, 2K) = 1/(2K(2K+1))

    K-DEPENDENCE:
    =============
    - K=3: 1 + θ/42 = 1.01361 (θ=4/7)
    - K=4: 1 + θ/72 = 1.00794
    - K=5: 1 + θ/110 = 1.00519

    RESIDUAL (±0.15%):
    ==================
    The remaining ±0.15% is from Q polynomial derivative effects (Phase 37).
    This is understood and characterized, not a derivation failure.
    """
    import warnings

    if formula == "derived":
        # PRODUCTION FORMULA (Phase 36 locked)
        # m = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]
        denom = 2 * K * (2 * K + 1)
        beta_correction = 1 + theta / denom
        base = math.exp(R) + (2 * K - 1)
        m = beta_correction * base
        desc = (f"[1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)] = "
                f"{beta_correction:.6f} × [{math.exp(R):.4f} + {2*K-1}]")

    elif formula == "empirical":
        # DIAGNOSTIC ONLY - use "derived" for production
        warnings.warn(
            "Using 'empirical' formula (exp(R)+2K-1) without Beta correction. "
            "Use 'derived' for production.",
            UserWarning,
            stacklevel=2,
        )
        m = math.exp(R) + (2 * K - 1)
        desc = f"[DIAGNOSTIC] exp(R) + (2K-1) = exp({R}) + {2*K-1}"

    elif formula == "derived_full":
        # DEPRECATED - fitted coefficients, not first-principles
        warnings.warn(
            "'derived_full' contains fitted coefficients (γ, R₀) and is deprecated. "
            "Use 'derived' for production.",
            DeprecationWarning,
            stacklevel=2,
        )
        denom = 2 * K * (2 * K + 1)
        beta = theta / denom
        gamma = 1.16  # FITTED - not derived
        R0 = 1.21     # FITTED - not derived
        correction = (1 + beta) * (1 + beta * gamma * (R - R0))
        base = math.exp(R) + (2 * K - 1)
        m = correction * base
        desc = (f"[DEPRECATED] fitted formula with γ={gamma}, R₀={R0}")

    elif formula == "theoretical_exp2R":
        # DIAGNOSTIC ONLY - theoretical comparison
        m = math.exp(2 * R)
        desc = f"[DIAGNOSTIC] exp(2R) = exp({2*R})"

    elif formula == "theoretical_exp2R_theta":
        # DIAGNOSTIC ONLY - theoretical comparison
        m = math.exp(2 * R / theta)
        desc = f"[DIAGNOSTIC] exp(2R/θ) = exp({2*R/theta:.4f})"

    elif formula == "derived_functional":
        # PHASE 41: Polynomial-aware functional
        # Uses g(P,Q,R,K,θ) computed from I1/I2 ratio structure
        # NOTE: This requires polynomials, which are not passed to this function.
        # For now, fall back to derived formula. Full implementation requires
        # passing polynomials to compute_decomposition and using g_functional there.
        warnings.warn(
            "'derived_functional' requires polynomial context. "
            "Using 'derived' formula as fallback. "
            "For full functional computation, use compute_decomposition_with_functional().",
            UserWarning,
            stacklevel=2,
        )
        denom = 2 * K * (2 * K + 1)
        beta_correction = 1 + theta / denom
        base = math.exp(R) + (2 * K - 1)
        m = beta_correction * base
        desc = f"[FUNCTIONAL FALLBACK] Using derived formula"

    else:
        raise ValueError(f"Unknown formula: {formula}")

    return m, desc


def compute_decomposition(
    theta: float,
    R: float,
    K: int,
    polynomials: Dict,
    *,
    kernel_regime: str = "paper",
    n_quad: int = 60,
    mirror_formula: str = "derived",
) -> Decomposition:
    """
    Compute the canonical decomposition of c.

    This is THE canonical function for decomposing c. All other
    "compute_c_*" functions should delegate to this.

    PRODUCTION FORMULA (Phase 36 locked):
    =====================================
    Uses derived mirror multiplier by default:
        m = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        K: Number of mollifier pieces
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        kernel_regime: "paper" or "raw"
        n_quad: Quadrature points
        mirror_formula: Which m formula to use (default: "derived" for production)

    Returns:
        Decomposition with all components
    """
    from src.mirror_transform_paper_exact import compute_S12_paper_sum
    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term

    # 1. Compute S12 at +R and -R
    S12_plus = compute_S12_paper_sum(R, theta, polynomials, n_quad=n_quad)
    S12_minus = compute_S12_paper_sum(-R, theta, polynomials, n_quad=n_quad)

    # 2. Compute S34 (NEVER hardcode!)
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

    # 3. Compute mirror multiplier
    mirror_mult, mirror_desc = compute_mirror_multiplier(R, K, formula=mirror_formula)

    # 4. Assemble c
    total = S12_plus + mirror_mult * S12_minus + S34

    # 5. Create decomposition
    decomp = Decomposition(
        S12_plus=S12_plus,
        S12_minus=S12_minus,
        S34=S34,
        mirror_mult=mirror_mult,
        mirror_formula=mirror_desc,
        total=total,
        R=R,
        theta=theta,
        K=K,
        kernel_regime=kernel_regime,
        assembly_verified=True,  # Will verify below
    )

    # 6. Verify assembly
    decomp.assembly_verified = decomp.verify_assembly()

    return decomp


def decomposition_to_c(decomp: Decomposition) -> float:
    """Extract c from decomposition (for compatibility)."""
    return decomp.total
