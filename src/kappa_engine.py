"""
src/kappa_engine.py
Production Engine for Computing κ (Proportion of Zeta Zeros on Critical Line)

This is the LOCKED production implementation of the PRZZ framework with
complete first-principles g corrections. No calibrated parameters.

FORMULAS (100% First-Principles):
=================================

g_I1 = 1 + θ(1-θ)(2(K-1)+θ) / (8K(2K+1)²)
g_I2 = 1 + θ(2-θ) / (2K(2K+1))
base = exp(R) + (2K-1)
m = [f_I1 × g_I1 + (1-f_I1) × g_I2] × base
c = I₁I₂(+R) + m × I₁I₂(-R) + I₃I₄(+R)
κ = 1 - log(c) / R

USAGE:
======

from src.kappa_engine import KappaEngine

# Load with PRZZ polynomials
engine = KappaEngine.from_przz_kappa()

# Compute κ
result = engine.compute_kappa()
print(f"κ = {result.kappa:.10f}")

# Or use custom polynomials for optimization
engine = KappaEngine(
    P1_coeffs=[...],
    P2_coeffs=[...],
    P3_coeffs=[...],
    Q_coeffs=[...],
    theta=4/7,
    K=3,
    R=1.3036,
)

Created: 2025-12-27 (Phase 46++)
Status: PRODUCTION - LOCKED
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class IntegralComponents:
    """Components of the I1, I2, I3, I4 integrals.

    Note: PRZZ assembly uses SUMS (I1+I2), not products.
    - S12 = I1 + I2 (sum over all pairs)
    - S34 = I3 + I4 (sum over all pairs)
    - c = S12(+R) + m × S12(-R) + S34(+R)
    """
    I1_plus: float   # I1 total at +R (sum over all pairs)
    I1_minus: float  # I1 total at -R (sum over all pairs)
    I2_plus: float   # I2 total at +R (sum over all pairs)
    I2_minus: float  # I2 total at -R (sum over all pairs)
    I3_plus: float   # I3 total at +R (sum over all pairs)
    I4_plus: float   # I4 total at +R (sum over all pairs)

    @property
    def S12_plus(self) -> float:
        """S12 at +R: I1(+R) + I2(+R)."""
        return self.I1_plus + self.I2_plus

    @property
    def S12_minus(self) -> float:
        """S12 at -R: I1(-R) + I2(-R)."""
        return self.I1_minus + self.I2_minus

    @property
    def S34_plus(self) -> float:
        """S34 at +R: I3(+R) + I4(+R)."""
        return self.I3_plus + self.I4_plus

    @property
    def f_I1(self) -> float:
        """I1 fraction at -R: I1(-R) / (I1(-R) + I2(-R))."""
        total = self.I1_minus + self.I2_minus
        if abs(total) < 1e-15:
            return 0.5
        return self.I1_minus / total


@dataclass
class CorrectionFactors:
    """First-principles correction factors."""
    g_I1: float      # Correction for I1
    g_I2: float      # Correction for I2
    g_total: float   # Weighted correction
    base: float      # exp(R) + (2K-1)
    m: float         # Full mirror multiplier
    f_I1: float      # I1 fraction used


@dataclass
class KappaResult:
    """Complete result of κ computation."""
    kappa: float              # The proportion bound
    c: float                  # Main-term constant

    # Intermediate values
    integrals: IntegralComponents
    corrections: CorrectionFactors

    # Input parameters
    theta: float
    K: int
    R: float

    # Per-pair breakdown (optional)
    pair_contributions: Optional[dict] = None

    def __str__(self) -> str:
        return f"""
KappaResult:
  κ = {self.kappa:.10f}
  c = {self.c:.10f}

  Parameters:
    θ = {self.theta:.10f}
    K = {self.K}
    R = {self.R}

  Integrals:
    S12(+R) = {self.integrals.S12_plus:.10f}
    S12(-R) = {self.integrals.S12_minus:.10f}
    S34(+R) = {self.integrals.S34_plus:.10f}
    f_I1    = {self.integrals.f_I1:.6f}

  Corrections (First-Principles):
    g_I1    = {self.corrections.g_I1:.10f}
    g_I2    = {self.corrections.g_I2:.10f}
    g_total = {self.corrections.g_total:.10f}
    base    = {self.corrections.base:.10f}
    m       = {self.corrections.m:.10f}

  Assembly:
    c = S12(+R) + m × S12(-R) + S34(+R)
      = {self.integrals.S12_plus:.6f} + {self.corrections.m:.6f} × {self.integrals.S12_minus:.6f} + {self.integrals.S34_plus:.6f}
      = {self.c:.10f}
    κ = 1 - log(c)/R = {self.kappa:.10f}
"""


# =============================================================================
# FIRST-PRINCIPLES FORMULAS
# =============================================================================

def compute_g_I1(theta: float, K: int) -> float:
    """
    Compute g_I1 using the unified first-principles formula.

    Formula: g_I1 = 1 + θ(1-θ)(2(K-1)+θ) / (8K(2K+1)²)

    This is NOT calibrated. It derives from the log factor structure in I1.
    For K=3, θ=4/7, this simplifies to 1 + (3/28)×θ³/(K(2K+1)).

    Args:
        theta: Mollifier exponent (typically 4/7)
        K: Number of mollifier pieces (typically 3)

    Returns:
        g_I1 correction factor
    """
    numerator = theta * (1 - theta) * (2*(K-1) + theta)
    denominator = 8 * K * (2*K + 1)**2
    return 1 + numerator / denominator


def compute_g_I2(theta: float, K: int) -> float:
    """
    Compute g_I2 using the first-principles formula.

    Formula: g_I2 = 1 + θ(2-θ) / (2K(2K+1))

    This is NOT calibrated. It derives from Beta moment expansion.

    Args:
        theta: Mollifier exponent (typically 4/7)
        K: Number of mollifier pieces (typically 3)

    Returns:
        g_I2 correction factor
    """
    return 1 + theta * (2 - theta) / (2 * K * (2*K + 1))


def compute_base(R: float, K: int) -> float:
    """
    Compute the mirror multiplier base.

    Formula: base = exp(R) + (2K-1)

    This is NOT calibrated. It derives from difference quotient analysis.

    Args:
        R: Shift parameter
        K: Number of mollifier pieces

    Returns:
        Base term for mirror multiplier
    """
    return math.exp(R) + (2*K - 1)


def compute_mirror_multiplier(
    theta: float,
    K: int,
    R: float,
    f_I1: float,
) -> CorrectionFactors:
    """
    Compute the complete mirror multiplier using first-principles formulas.

    Formula: m = [f_I1 × g_I1 + (1-f_I1) × g_I2] × base

    Args:
        theta: Mollifier exponent
        K: Number of mollifier pieces
        R: Shift parameter
        f_I1: I1 fraction at -R

    Returns:
        CorrectionFactors with all intermediate values
    """
    g_I1 = compute_g_I1(theta, K)
    g_I2 = compute_g_I2(theta, K)
    g_total = f_I1 * g_I1 + (1 - f_I1) * g_I2
    base = compute_base(R, K)
    m = g_total * base

    return CorrectionFactors(
        g_I1=g_I1,
        g_I2=g_I2,
        g_total=g_total,
        base=base,
        m=m,
        f_I1=f_I1,
    )


def compute_c_from_integrals(
    integrals: IntegralComponents,
    m: float,
) -> float:
    """
    Compute c using the mirror assembly formula.

    Formula: c = I₁I₂(+R) + m × I₁I₂(-R) + I₃I₄(+R)

    Args:
        integrals: Computed integral components
        m: Mirror multiplier

    Returns:
        Main-term constant c
    """
    return integrals.S12_plus + m * integrals.S12_minus + integrals.S34_plus


def compute_kappa_from_c(c: float, R: float) -> float:
    """
    Compute κ from the main-term constant.

    Formula: κ = 1 - log(c) / R

    Args:
        c: Main-term constant
        R: Shift parameter

    Returns:
        Proportion bound κ
    """
    return 1 - math.log(c) / R


# =============================================================================
# KAPPA ENGINE
# =============================================================================

class KappaEngine:
    """
    Production engine for computing κ.

    This class encapsulates the complete PRZZ computation pipeline with
    first-principles g corrections. It can be used with:

    1. PRZZ polynomials (for reproduction)
    2. Custom polynomials (for optimization)

    All formulas are first-principles with NO calibrated parameters.
    """

    def __init__(
        self,
        P1_coeffs: List[float],
        P2_coeffs: List[float],
        P3_coeffs: List[float],
        Q_coeffs: List[float],
        theta: float = 4/7,
        K: int = 3,
        R: float = 1.3036,
        n_quad: int = 80,
    ):
        """
        Initialize the engine with polynomial coefficients.

        Args:
            P1_coeffs: Coefficients for P1 polynomial (in tilde basis)
            P2_coeffs: Coefficients for P2 polynomial (in tilde basis)
            P3_coeffs: Coefficients for P3 polynomial (in tilde basis)
            Q_coeffs: Coefficients for Q polynomial
            theta: Mollifier exponent (default 4/7)
            K: Number of mollifier pieces (default 3)
            R: Shift parameter (default 1.3036)
            n_quad: Number of quadrature points (default 80)
        """
        self.P1_coeffs = P1_coeffs
        self.P2_coeffs = P2_coeffs
        self.P3_coeffs = P3_coeffs
        self.Q_coeffs = Q_coeffs
        self.theta = theta
        self.K = K
        self.R = R
        self.n_quad = n_quad

        # Lazy-loaded polynomial objects
        self._P1 = None
        self._P2 = None
        self._P3 = None
        self._Q = None
        self._polynomials = None

    @classmethod
    def from_przz_kappa(cls, n_quad: int = 80) -> "KappaEngine":
        """
        Create engine with PRZZ κ benchmark polynomials (R=1.3036).

        Returns:
            KappaEngine configured for κ benchmark
        """
        from src.polynomials import load_przz_polynomials

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)

        return cls(
            P1_coeffs=P1.tilde_coeffs.tolist(),
            P2_coeffs=P2.tilde_coeffs.tolist(),
            P3_coeffs=P3.tilde_coeffs.tolist(),
            Q_coeffs=Q.to_monomial().coeffs.tolist(),
            theta=4/7,
            K=3,
            R=1.3036,
            n_quad=n_quad,
        )

    @classmethod
    def from_przz_kappa_star(cls, n_quad: int = 80) -> "KappaEngine":
        """
        Create engine with PRZZ κ* benchmark polynomials (R=1.1167).

        Returns:
            KappaEngine configured for κ* benchmark
        """
        from src.polynomials import load_przz_polynomials_kappa_star

        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=False)

        return cls(
            P1_coeffs=P1.tilde_coeffs.tolist(),
            P2_coeffs=P2.tilde_coeffs.tolist(),
            P3_coeffs=P3.tilde_coeffs.tolist(),
            Q_coeffs=Q.to_monomial().coeffs.tolist(),
            theta=4/7,
            K=3,
            R=1.1167,
            n_quad=n_quad,
        )

    def _load_polynomials(self):
        """Lazy-load polynomial objects and create polynomials dict."""
        if self._polynomials is not None:
            return

        from src.polynomials import P1Polynomial, PellPolynomial, QPolynomial

        # Create polynomial objects with correct classes
        self._P1 = P1Polynomial(tilde_coeffs=np.array(self.P1_coeffs))
        self._P2 = PellPolynomial(tilde_coeffs=np.array(self.P2_coeffs))
        self._P3 = PellPolynomial(tilde_coeffs=np.array(self.P3_coeffs))

        # Q polynomial is stored as monomial coefficients
        # Create QPolynomial from basis coeffs (identity mapping for monomial)
        # Actually, Q_coeffs are already monomial coeffs, so we need to create
        # a simple polynomial wrapper
        from src.polynomials import Polynomial
        self._Q = Polynomial(coeffs=np.array(self.Q_coeffs))

        # Create polynomials dict for compute functions
        self._polynomials = {
            "P1": self._P1,
            "P2": self._P2,
            "P3": self._P3,
            "Q": self._Q,
        }

    def compute_integrals(self) -> IntegralComponents:
        """
        Compute all I1, I2, I3, I4 integrals at ±R summed over all pairs.

        Returns:
            IntegralComponents with total integral values
        """
        self._load_polynomials()

        from src.unified_i1_paper import compute_I1_unified_paper
        from src.unified_i2_paper import compute_I2_unified_paper
        from src.terms_k3_d1 import make_all_terms_k3
        from src.evaluate import evaluate_term

        # Factorial normalization factors
        f_norm = {
            "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
            "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
        }

        # Symmetry factors (off-diagonal pairs counted twice)
        symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

        pairs = ["11", "22", "33", "12", "13", "23"]

        # Sum I1 and I2 over all pairs at +R and -R
        I1_plus_total = 0.0
        I1_minus_total = 0.0
        I2_plus_total = 0.0
        I2_minus_total = 0.0

        for pair_key in pairs:
            ell1 = int(pair_key[0])
            ell2 = int(pair_key[1])

            norm = f_norm[pair_key]
            sym = symmetry[pair_key]
            full_norm = sym * norm

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
        all_terms_plus = make_all_terms_k3(self.theta, self.R, kernel_regime="paper")

        I3_plus_total = 0.0
        I4_plus_total = 0.0

        for pair_key in pairs:
            terms_plus = all_terms_plus[pair_key]
            norm = f_norm[pair_key]
            sym = symmetry[pair_key]
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

        # Step 2: Compute correction factors
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

    def validate_against_target(
        self,
        kappa_target: float,
        tolerance_pct: float = 0.01,
    ) -> Tuple[bool, float]:
        """
        Validate computed κ against a target value.

        Args:
            kappa_target: Expected κ value
            tolerance_pct: Allowed gap in percent (default 0.01%)

        Returns:
            (passed, gap_pct) tuple
        """
        result = self.compute_kappa()
        gap_pct = (result.kappa / kappa_target - 1) * 100
        passed = abs(gap_pct) <= tolerance_pct
        return passed, gap_pct


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_przz_kappa(n_quad: int = 80) -> KappaResult:
    """
    Compute κ using PRZZ κ benchmark parameters.

    This reproduces the PRZZ result: κ ≈ 0.417293962

    Args:
        n_quad: Number of quadrature points

    Returns:
        KappaResult
    """
    engine = KappaEngine.from_przz_kappa(n_quad=n_quad)
    return engine.compute_kappa()


def compute_przz_kappa_star(n_quad: int = 80) -> KappaResult:
    """
    Compute κ using PRZZ κ* benchmark parameters.

    Args:
        n_quad: Number of quadrature points

    Returns:
        KappaResult
    """
    engine = KappaEngine.from_przz_kappa_star(n_quad=n_quad)
    return engine.compute_kappa()


def validate_przz_benchmarks(
    tolerance_pct: float = 0.1,
    n_quad: int = 80,
) -> dict:
    """
    Validate against both PRZZ benchmarks.

    Args:
        tolerance_pct: Allowed gap in percent
        n_quad: Number of quadrature points

    Returns:
        Dictionary with validation results
    """
    # κ benchmark
    kappa_engine = KappaEngine.from_przz_kappa(n_quad=n_quad)
    kappa_result = kappa_engine.compute_kappa()
    kappa_target = 0.417293962
    kappa_gap = (kappa_result.kappa / kappa_target - 1) * 100

    # κ* benchmark
    kappa_star_engine = KappaEngine.from_przz_kappa_star(n_quad=n_quad)
    kappa_star_result = kappa_star_engine.compute_kappa()
    kappa_star_target = 0.407511457  # PRZZ κ* target
    kappa_star_gap = (kappa_star_result.kappa / kappa_star_target - 1) * 100

    return {
        "kappa": {
            "computed": kappa_result.kappa,
            "target": kappa_target,
            "gap_pct": kappa_gap,
            "passed": abs(kappa_gap) <= tolerance_pct,
            "c": kappa_result.c,
        },
        "kappa_star": {
            "computed": kappa_star_result.kappa,
            "target": kappa_star_target,
            "gap_pct": kappa_star_gap,
            "passed": abs(kappa_star_gap) <= tolerance_pct,
            "c": kappa_star_result.c,
        },
    }


# =============================================================================
# FORMULA DOCUMENTATION
# =============================================================================

FORMULA_DOC = """
================================================================================
PRZZ κ COMPUTATION - FIRST PRINCIPLES FORMULAS
================================================================================

INPUT PARAMETERS:
  θ (theta) = Mollifier exponent (typically 4/7)
  K = Number of mollifier pieces (typically 3)
  R = Shift parameter (typically 1.3036)

CORRECTION FORMULAS (NO CALIBRATION):

  g_I1 = 1 + θ(1-θ)(2(K-1)+θ) / (8K(2K+1)²)

  g_I2 = 1 + θ(2-θ) / (2K(2K+1))

  base = exp(R) + (2K-1)

MIRROR MULTIPLIER:

  f_I1 = I1(-R) / (I1(-R) + I2(-R))    [computed from integrals]

  g_total = f_I1 × g_I1 + (1-f_I1) × g_I2

  m = g_total × base

MAIN-TERM CONSTANT:

  c = I₁I₂(+R) + m × I₁I₂(-R) + I₃I₄(+R)

PROPORTION BOUND:

  κ = 1 - log(c) / R

================================================================================
FOR K=3, θ=4/7:

  g_I1 = 1.0009519843  (simplified: 1 + (3/28)×θ³/(K(2K+1)))
  g_I2 = 1.0194363460
  base = 8.6825299412  (for R=1.3036)

The (3/28) coefficient derives from:
  (3/28) = (1-θ)(2(K-1)+θ) / (8(2K+1)θ²)

This is NOT empirical - it's fully derivable from the formula structure.

================================================================================
"""


if __name__ == "__main__":
    print(FORMULA_DOC)

    print("\nComputing κ benchmark...")
    result = compute_przz_kappa()
    print(result)

    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    validation = validate_przz_benchmarks()

    print(f"\nκ benchmark:")
    print(f"  Computed: {validation['kappa']['computed']:.10f}")
    print(f"  Target:   {validation['kappa']['target']:.10f}")
    print(f"  Gap:      {validation['kappa']['gap_pct']:+.6f}%")
    print(f"  Status:   {'PASS' if validation['kappa']['passed'] else 'FAIL'}")

    print(f"\nκ* benchmark:")
    print(f"  Computed: {validation['kappa_star']['computed']:.10f}")
    print(f"  Target:   {validation['kappa_star']['target']:.10f}")
    print(f"  Gap:      {validation['kappa_star']['gap_pct']:+.6f}%")
    print(f"  Status:   {'PASS' if validation['kappa_star']['passed'] else 'FAIL'}")
