"""
src/kappa_engine_k4.py
Experimental K=4 Engine for κ Computation (10 pairs instead of 6)

EXPERIMENTAL - DO NOT USE IN PRODUCTION until validated.

This extends the K=3 framework to K=4 with a 4th mollifier piece P₄.

K=4 STRUCTURE:
==============
- 10 pairs: (1,1), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,3), (3,4), (4,4)
- 4 polynomials: P₁, P₂, P₃, P₄

K=4 FORMULAS (first-principles):
================================
g_I1(K=4) = 1 + θ(1-θ)(2(K-1)+θ) / (8K(2K+1)²)
          = 1 + θ(1-θ)(6+θ) / 2592
          ≈ 1.00070 for θ=4/7

g_I2(K=4) = 1 + θ(2-θ) / (2K(2K+1))
          = 1 + θ(2-θ) / 72
          ≈ 1.01359 for θ=4/7

base(K=4) = exp(R) + 7
          ≈ 10.6757 for R=1.3036

Created: 2025-12-28 (Phase 48)
Status: EXPERIMENTAL
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import math
import numpy as np
import logging

# Import from K=3 engine for the formulas that generalize
from src.kappa_engine import (
    IntegralComponents,
    CorrectionFactors,
    KappaResult,
    compute_g_I1,
    compute_g_I2,
    compute_base,
    compute_mirror_multiplier,
    compute_c_from_integrals,
    compute_kappa_from_c,
)
from src.evaluator.pairs import (
    get_triangle_pairs,
    pair_key,
    factorial_norm,
    symmetry_factor,
)

logger = logging.getLogger(__name__)


class KappaEngineK4:
    """
    Experimental K=4 engine for computing κ with 4 mollifier pieces.

    This uses the same first-principles formulas as K=3 but with:
    - 10 pairs instead of 6
    - Optional P4 polynomial

    EXPERIMENTAL - validate against microcase ladder before trusting.
    """

    def __init__(
        self,
        P1_coeffs: List[float],
        P2_coeffs: List[float],
        P3_coeffs: List[float],
        P4_coeffs: List[float],
        Q_coeffs: List[float],
        theta: float = 4/7,
        R: float = 1.3036,
        n_quad: int = 80,
    ):
        """
        Initialize the K=4 engine.

        Args:
            P1_coeffs: Coefficients for P1 polynomial (in tilde basis)
            P2_coeffs: Coefficients for P2 polynomial (in tilde basis)
            P3_coeffs: Coefficients for P3 polynomial (in tilde basis)
            P4_coeffs: Coefficients for P4 polynomial (in tilde basis)
            Q_coeffs: Coefficients for Q polynomial
            theta: Mollifier exponent (default 4/7)
            R: Shift parameter (default 1.3036)
            n_quad: Number of quadrature points (default 80)
        """
        self.P1_coeffs = P1_coeffs
        self.P2_coeffs = P2_coeffs
        self.P3_coeffs = P3_coeffs
        self.P4_coeffs = P4_coeffs
        self.Q_coeffs = Q_coeffs
        self.theta = theta
        self.K = 4  # Fixed for this engine
        self.R = R
        self.n_quad = n_quad

        # Lazy-loaded polynomial objects
        self._polynomials = None

    @classmethod
    def from_baseline(cls, n_quad: int = 80) -> "KappaEngineK4":
        """
        Create engine with K=4 baseline polynomials.

        Uses K=3 PRZZ polynomials for P1-P3 and initializes P4 with small values.

        Returns:
            KappaEngineK4 configured for baseline
        """
        from src.polynomials import load_przz_polynomials

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)

        # P4 initialized with small values (optimization will find better)
        P4_initial = [0.1, -0.05, 0.0]

        return cls(
            P1_coeffs=P1.tilde_coeffs.tolist(),
            P2_coeffs=P2.tilde_coeffs.tolist(),
            P3_coeffs=P3.tilde_coeffs.tolist(),
            P4_coeffs=P4_initial,
            Q_coeffs=Q.to_monomial().coeffs.tolist(),
            theta=4/7,
            R=1.3036,
            n_quad=n_quad,
        )

    @classmethod
    def from_json(cls, filepath: str, n_quad: int = 80) -> "KappaEngineK4":
        """
        Load engine from JSON file.

        Args:
            filepath: Path to JSON file with polynomial coefficients
            n_quad: Number of quadrature points

        Returns:
            KappaEngineK4 configured from file
        """
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            P1_coeffs=data["P1_tilde"],
            P2_coeffs=data["P2_tilde"],
            P3_coeffs=data["P3_tilde"],
            P4_coeffs=data["P4_tilde"],
            Q_coeffs=data["Q_mono"],
            theta=data.get("theta", 4/7),
            R=data.get("R", 1.3036),
            n_quad=n_quad,
        )

    def _load_polynomials(self):
        """Lazy-load polynomial objects and create polynomials dict."""
        if self._polynomials is not None:
            return

        from src.polynomials import P1Polynomial, PellPolynomial, Polynomial

        # Create polynomial objects with correct classes
        self._P1 = P1Polynomial(tilde_coeffs=np.array(self.P1_coeffs))
        self._P2 = PellPolynomial(tilde_coeffs=np.array(self.P2_coeffs))
        self._P3 = PellPolynomial(tilde_coeffs=np.array(self.P3_coeffs))
        self._P4 = PellPolynomial(tilde_coeffs=np.array(self.P4_coeffs))

        # Q polynomial is stored as monomial coefficients
        self._Q = Polynomial(coeffs=np.array(self.Q_coeffs))

        # Create polynomials dict for compute functions
        self._polynomials = {
            "P1": self._P1,
            "P2": self._P2,
            "P3": self._P3,
            "P4": self._P4,
            "Q": self._Q,
        }

    def _get_pairs(self) -> List[Tuple[int, int]]:
        """Get all K=4 pairs."""
        return get_triangle_pairs(self.K)

    def compute_integrals(self) -> IntegralComponents:
        """
        Compute all I1, I2, I3, I4 integrals at ±R summed over all 10 pairs.

        Returns:
            IntegralComponents with total integral values
        """
        self._load_polynomials()

        from src.unified_i1_paper import compute_I1_unified_paper
        from src.unified_i2_paper import compute_I2_unified_paper
        from src.terms_k3_d1 import make_all_terms_k4
        from src.evaluate import evaluate_term

        pairs = self._get_pairs()

        # Sum I1 and I2 over all pairs at +R and -R
        I1_plus_total = 0.0
        I1_minus_total = 0.0
        I2_plus_total = 0.0
        I2_minus_total = 0.0

        for ell1, ell2 in pairs:
            norm = factorial_norm(ell1, ell2)
            sym = symmetry_factor(ell1, ell2)

            # I1 at +R
            I1_plus_result = compute_I1_unified_paper(
                R=self.R, theta=self.theta, ell1=ell1, ell2=ell2,
                polynomials=self._polynomials,
                n_quad_u=self.n_quad, n_quad_t=self.n_quad, n_quad_a=40,
                include_Q=True, apply_factorial_norm=True,
            )
            I1_plus_total += I1_plus_result.I1_value * norm * sym

            # I1 at -R
            I1_minus_result = compute_I1_unified_paper(
                R=-self.R, theta=self.theta, ell1=ell1, ell2=ell2,
                polynomials=self._polynomials,
                n_quad_u=self.n_quad, n_quad_t=self.n_quad, n_quad_a=40,
                include_Q=True, apply_factorial_norm=True,
            )
            I1_minus_total += I1_minus_result.I1_value * norm * sym

            # I2 at +R
            I2_plus_result = compute_I2_unified_paper(
                R=self.R, theta=self.theta, ell1=ell1, ell2=ell2,
                polynomials=self._polynomials,
                n_quad_u=self.n_quad, n_quad_t=self.n_quad, n_quad_a=40,
                include_Q=True,
            )
            I2_plus_total += I2_plus_result.I2_value * norm * sym

            # I2 at -R
            I2_minus_result = compute_I2_unified_paper(
                R=-self.R, theta=self.theta, ell1=ell1, ell2=ell2,
                polynomials=self._polynomials,
                n_quad_u=self.n_quad, n_quad_t=self.n_quad, n_quad_a=40,
                include_Q=True,
            )
            I2_minus_total += I2_minus_result.I2_value * norm * sym

        # Compute I3 and I4 at +R (no mirror needed)
        # Use K=4 terms for all 10 pairs
        all_terms_plus = make_all_terms_k4(self.theta, self.R, kernel_regime="paper")

        I3_plus_total = 0.0
        I4_plus_total = 0.0

        for ell1, ell2 in pairs:
            pk = pair_key(ell1, ell2)
            terms_plus = all_terms_plus.get(pk)
            if terms_plus is None:
                continue

            norm = factorial_norm(ell1, ell2)
            sym = symmetry_factor(ell1, ell2)
            full_norm = sym * norm

            # I3 (index 2) and I4 (index 3)
            if len(terms_plus) > 2:
                I3_result = evaluate_term(
                    terms_plus[2], self._polynomials, self.n_quad,
                    R=self.R, theta=self.theta, n_quad_a=40
                )
                I3_plus_total += full_norm * I3_result.value

            if len(terms_plus) > 3:
                I4_result = evaluate_term(
                    terms_plus[3], self._polynomials, self.n_quad,
                    R=self.R, theta=self.theta, n_quad_a=40
                )
                I4_plus_total += full_norm * I4_result.value

        return IntegralComponents(
            I1_plus=I1_plus_total,
            I1_minus=I1_minus_total,
            I2_plus=I2_plus_total,
            I2_minus=I2_minus_total,
            I3_plus=I3_plus_total,
            I4_plus=I4_plus_total,
        )

    def compute_kappa(self) -> KappaResult:
        """
        Compute κ using the complete first-principles pipeline.

        Returns:
            KappaResult with κ and all intermediate values
        """
        # Step 1: Compute integrals
        integrals = self.compute_integrals()

        # Step 2: Compute correction factors (K=4 formulas)
        corrections = compute_mirror_multiplier(
            theta=self.theta,
            K=self.K,
            R=self.R,
            f_I1=integrals.f_I1,
        )

        # Step 3: Assemble c
        c = compute_c_from_integrals(integrals, corrections.m)

        # Step 4: Compute κ
        kappa = compute_kappa_from_c(c, self.R)

        return KappaResult(
            kappa=kappa,
            c=c,
            integrals=integrals,
            corrections=corrections,
            theta=self.theta,
            K=self.K,
            R=self.R,
        )


def print_k4_formula_comparison():
    """Print formula comparison between K=3 and K=4."""
    theta = 4/7
    R = 1.3036

    print("K=3 vs K=4 Formula Comparison")
    print("=" * 60)

    for K in [3, 4]:
        g_I1 = compute_g_I1(theta, K)
        g_I2 = compute_g_I2(theta, K)
        base = compute_base(R, K)

        print(f"\nK={K}:")
        print(f"  g_I1 = {g_I1:.10f}")
        print(f"  g_I2 = {g_I2:.10f}")
        print(f"  base = {base:.10f}")
        print(f"  num_pairs = {K*(K+1)//2}")


if __name__ == "__main__":
    print_k4_formula_comparison()

    print("\n" + "=" * 60)
    print("Testing K=4 Engine (K=3 pairs only for now)")
    print("=" * 60)

    # Test with baseline
    engine = KappaEngineK4.from_baseline(n_quad=40)

    print("\nComputing with K=3 pairs only (skipping K=4 pairs)...")
    result = engine.compute_kappa()

    print(f"\nResult (partial, K=3 pairs only):")
    print(f"  c = {result.c:.10f}")
    print(f"  κ = {result.kappa:.10f}")
    print(f"  K = {result.K}")
    print(f"  g_I1 = {result.corrections.g_I1:.10f}")
    print(f"  g_I2 = {result.corrections.g_I2:.10f}")
    print(f"  base = {result.corrections.base:.10f}")
    print(f"  m = {result.corrections.m:.10f}")
