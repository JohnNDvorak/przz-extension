"""
src/evaluator/s12_spec.py
Phase 28/29: Canonical S12 Specification

Defines THE single canonical representation for I1/I2 values across all backends.
This file answers: "What do we mean by I1(ℓ₁,ℓ₂)?"

The key insight from Phase 27 is that unified_general and evaluate.py give
different I1/I2 values because they use different conventions. This spec
defines the conventions explicitly so we can reconcile them.

FOUR TOGGLES that define an S12 value:

1. kernel_regime: "raw" vs "paper" (PHASE 29 - CRITICAL)
   - raw: P(u) used directly (Case B for all polynomials)
   - paper: P₂/P₃ use Case C kernel attenuation
   - This is NOT a convention - values are MATHEMATICALLY DIFFERENT
   - Cross-regime comparisons are FORBIDDEN

2. factorial_mode: "derivative" vs "coefficient"
   - derivative: returns ∂^{ℓ₁+ℓ₂}/∂x^{ℓ₁}∂y^{ℓ₂} evaluated at 0
   - coefficient: returns [x^{ℓ₁}y^{ℓ₂}] coefficient
   - Relationship: derivative = ℓ₁! × ℓ₂! × coefficient

3. sign_mode: how off-diagonal signs are handled
   - "none": raw value, no sign adjustment
   - "offdiag_alternating": multiply by (-1)^{ℓ₁+ℓ₂} for ℓ₁ ≠ ℓ₂

4. pair_order_mode: interpretation of pair keys
   - "ordered": pair_key "13" means (ℓ₁=1, ℓ₂=3), distinct from "31"
   - "triangle": pair_key "13" means the triangle element for {1,3}

Created: 2025-12-26 (Phase 28)
Updated: 2025-12-26 (Phase 29 - added kernel_regime)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict, Tuple, Optional
from enum import Enum
import math


class FactorialMode(Enum):
    """How factorial normalization is applied."""
    DERIVATIVE = "derivative"    # Value includes ℓ₁!×ℓ₂! factor (d^n/dx^n at 0)
    COEFFICIENT = "coefficient"  # Raw [x^ℓ₁y^ℓ₂] coefficient


class SignMode(Enum):
    """How off-diagonal signs are handled."""
    NONE = "none"                      # Raw value, no sign adjustment
    OFFDIAG_ALTERNATING = "offdiag_alternating"  # (-1)^{ℓ₁+ℓ₂} for ℓ₁≠ℓ₂


class PairOrderMode(Enum):
    """How pair keys are interpreted."""
    ORDERED = "ordered"      # "13" means (1,3), distinct from "31" which means (3,1)
    TRIANGLE = "triangle"    # "13" means the triangle element, equivalent to "31"


class KernelRegime(Enum):
    """
    Which kernel transformation is applied to polynomial factors.

    PHASE 29 CRITICAL: This is NOT a convention toggle - values are mathematically different.
    Cross-regime comparisons are FORBIDDEN and will raise errors.
    """
    RAW = "raw"      # All polynomials use Case B (direct P(u))
    PAPER = "paper"  # P₂/P₃ use Case C kernel attenuation


class RegimeMismatchError(Exception):
    """Raised when attempting to compare S12 values from different kernel regimes."""
    pass


@dataclass(frozen=True)
class S12CanonicalValue:
    """
    Canonical representation of an I1 or I2 value for a specific pair.

    This is the SINGLE definition of what an I-term value means.
    All backends must be able to produce this representation.
    """

    # Pair identification
    ell1: int
    ell2: int

    # The actual value
    value: float

    # PHASE 29: Kernel regime (CRITICAL - values are mathematically different)
    kernel_regime: KernelRegime

    # Convention flags (how this value was computed)
    factorial_mode: FactorialMode
    sign_mode: SignMode

    # Source information
    backend: str
    term_type: Literal["I1", "I2"]

    def assert_same_regime(self, other: S12CanonicalValue) -> None:
        """Raise RegimeMismatchError if regimes differ."""
        if self.kernel_regime != other.kernel_regime:
            raise RegimeMismatchError(
                f"Cannot compare S12 values from different kernel regimes: "
                f"{self.kernel_regime.value} vs {other.kernel_regime.value}. "
                f"This comparison is mathematically meaningless."
            )

    def to_derivative_mode(self) -> S12CanonicalValue:
        """Convert to derivative mode (includes ℓ₁!×ℓ₂! factor)."""
        if self.factorial_mode == FactorialMode.DERIVATIVE:
            return self

        # coefficient → derivative: multiply by ℓ₁!×ℓ₂!
        factor = math.factorial(self.ell1) * math.factorial(self.ell2)
        return S12CanonicalValue(
            ell1=self.ell1,
            ell2=self.ell2,
            value=self.value * factor,
            kernel_regime=self.kernel_regime,
            factorial_mode=FactorialMode.DERIVATIVE,
            sign_mode=self.sign_mode,
            backend=self.backend,
            term_type=self.term_type,
        )

    def to_coefficient_mode(self) -> S12CanonicalValue:
        """Convert to coefficient mode (raw [x^ℓ₁y^ℓ₂] coefficient)."""
        if self.factorial_mode == FactorialMode.COEFFICIENT:
            return self

        # derivative → coefficient: divide by ℓ₁!×ℓ₂!
        factor = math.factorial(self.ell1) * math.factorial(self.ell2)
        return S12CanonicalValue(
            ell1=self.ell1,
            ell2=self.ell2,
            value=self.value / factor,
            kernel_regime=self.kernel_regime,
            factorial_mode=FactorialMode.COEFFICIENT,
            sign_mode=self.sign_mode,
            backend=self.backend,
            term_type=self.term_type,
        )

    def with_sign_mode(self, target_mode: SignMode) -> S12CanonicalValue:
        """Convert to specified sign mode."""
        if self.sign_mode == target_mode:
            return self

        # Compute the sign factor
        is_offdiag = (self.ell1 != self.ell2)
        sign_factor = ((-1) ** (self.ell1 + self.ell2)) if is_offdiag else 1

        if self.sign_mode == SignMode.NONE and target_mode == SignMode.OFFDIAG_ALTERNATING:
            # Apply the sign
            new_value = self.value * sign_factor
        elif self.sign_mode == SignMode.OFFDIAG_ALTERNATING and target_mode == SignMode.NONE:
            # Remove the sign
            new_value = self.value * sign_factor  # Multiply again to undo
        else:
            new_value = self.value

        return S12CanonicalValue(
            ell1=self.ell1,
            ell2=self.ell2,
            value=new_value,
            kernel_regime=self.kernel_regime,
            factorial_mode=self.factorial_mode,
            sign_mode=target_mode,
            backend=self.backend,
            term_type=self.term_type,
        )

    def canonicalize(
        self,
        target_factorial: FactorialMode = FactorialMode.COEFFICIENT,
        target_sign: SignMode = SignMode.NONE,
    ) -> S12CanonicalValue:
        """Convert to canonical form with specified conventions."""
        result = self
        if target_factorial == FactorialMode.COEFFICIENT:
            result = result.to_coefficient_mode()
        else:
            result = result.to_derivative_mode()
        return result.with_sign_mode(target_sign)


@dataclass
class S12MatrixRow:
    """One row of the 3x3 I1/I2 matrix."""
    values: Dict[int, S12CanonicalValue]  # ell2 -> value


@dataclass
class S12FullMatrix:
    """
    Full 3x3 ordered matrix of I1 or I2 values.

    This captures ALL ordered pairs:
    (1,1) (1,2) (1,3)
    (2,1) (2,2) (2,3)
    (3,1) (3,2) (3,3)

    NOT the triangle representation that folds (1,2) with (2,1).
    """

    term_type: Literal["I1", "I2"]
    backend: str
    R: float
    theta: float
    kernel_regime: KernelRegime  # PHASE 29: Required

    # values[(ell1, ell2)] = S12CanonicalValue
    values: Dict[Tuple[int, int], S12CanonicalValue]

    def assert_same_regime(self, other: S12FullMatrix) -> None:
        """Raise RegimeMismatchError if regimes differ."""
        if self.kernel_regime != other.kernel_regime:
            raise RegimeMismatchError(
                f"Cannot compare S12 matrices from different kernel regimes: "
                f"{self.kernel_regime.value} vs {other.kernel_regime.value}. "
                f"This comparison is mathematically meaningless."
            )

    def get(self, ell1: int, ell2: int) -> Optional[S12CanonicalValue]:
        """Get value for ordered pair (ell1, ell2)."""
        return self.values.get((ell1, ell2))

    def print_matrix(self, title: str = "") -> None:
        """Print the matrix in a readable format."""
        if title:
            print(f"\n{title}")
        print(f"Backend: {self.backend}, R={self.R}, θ={self.theta:.6f}")
        print(f"Term: {self.term_type}, Regime: {self.kernel_regime.value}")

        # Header
        print("\n     ", end="")
        for ell2 in [1, 2, 3]:
            print(f"    ℓ₂={ell2}     ", end="")
        print()
        print("-" * 50)

        for ell1 in [1, 2, 3]:
            print(f"ℓ₁={ell1} ", end="")
            for ell2 in [1, 2, 3]:
                val = self.get(ell1, ell2)
                if val is not None:
                    print(f" {val.value:>12.6e}", end="")
                else:
                    print(f" {'N/A':>12}", end="")
            print()

    def compare_to(
        self,
        other: S12FullMatrix,
        normalize_factorial: bool = True,
        normalize_sign: bool = True,
        enforce_regime: bool = True,
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Compare this matrix to another, returning per-cell comparison.

        Args:
            other: Matrix to compare against
            normalize_factorial: Convert both to coefficient mode before comparing
            normalize_sign: Remove off-diagonal sign convention before comparing
            enforce_regime: If True (default), raise RegimeMismatchError if regimes differ

        Returns dict[(ell1,ell2)] -> {
            "self": value,
            "other": value,
            "ratio": self/other,
            "diff": self - other,
            "match": bool (within tolerance),
        }

        Raises:
            RegimeMismatchError: If enforce_regime=True and regimes differ
        """
        # PHASE 29: Enforce regime checking
        if enforce_regime:
            self.assert_same_regime(other)

        results = {}

        for ell1 in [1, 2, 3]:
            for ell2 in [1, 2, 3]:
                v1 = self.get(ell1, ell2)
                v2 = other.get(ell1, ell2)

                if v1 is None or v2 is None:
                    continue

                # Optionally normalize to same convention
                if normalize_factorial:
                    v1 = v1.to_coefficient_mode()
                    v2 = v2.to_coefficient_mode()

                if normalize_sign:
                    v1 = v1.with_sign_mode(SignMode.NONE)
                    v2 = v2.with_sign_mode(SignMode.NONE)

                val1 = v1.value
                val2 = v2.value

                ratio = val1 / val2 if abs(val2) > 1e-15 else float('inf')
                diff = val1 - val2
                rel_err = abs(diff) / max(abs(val1), abs(val2), 1e-15)
                match = rel_err < 1e-6

                results[(ell1, ell2)] = {
                    "self": val1,
                    "other": val2,
                    "ratio": ratio,
                    "diff": diff,
                    "rel_err": rel_err,
                    "match": match,
                }

        return results


# ============================================================================
# BACKEND CONVENTION REGISTRY
# ============================================================================

# Document the conventions used by each backend
BACKEND_CONVENTIONS = {
    "unified_general": {
        "kernel_regime": KernelRegime.RAW,           # Phase 28: raw = unified_general
        "factorial_mode": FactorialMode.DERIVATIVE,  # Includes ℓ₁!×ℓ₂!
        "sign_mode": SignMode.OFFDIAG_ALTERNATING,   # (-1)^{ℓ₁+ℓ₂} for off-diag
        "pair_order": PairOrderMode.TRIANGLE,
        "notes": "Phase 26B validated against OLD DSL (raw regime)",
    },
    "unified_paper": {
        "kernel_regime": KernelRegime.PAPER,         # Phase 29: Case C for P2/P3
        "factorial_mode": FactorialMode.DERIVATIVE,  # Includes ℓ₁!×ℓ₂!
        "sign_mode": SignMode.OFFDIAG_ALTERNATING,   # (-1)^{ℓ₁+ℓ₂} for off-diag
        "pair_order": PairOrderMode.TRIANGLE,
        "notes": "Phase 29: paper regime with Case C kernel attenuation",
    },
    "term_dsl_raw": {
        "kernel_regime": KernelRegime.RAW,           # OLD DSL with raw regime
        "factorial_mode": FactorialMode.COEFFICIENT,  # Raw coefficient
        "sign_mode": SignMode.NONE,                   # No sign adjustment
        "pair_order": PairOrderMode.TRIANGLE,
        "notes": "evaluate.py term DSL with kernel_regime='raw'",
    },
    "term_dsl_paper": {
        "kernel_regime": KernelRegime.PAPER,         # OLD DSL with paper regime
        "factorial_mode": FactorialMode.COEFFICIENT,  # Raw coefficient
        "sign_mode": SignMode.NONE,                   # No sign adjustment
        "pair_order": PairOrderMode.TRIANGLE,
        "notes": "evaluate.py term DSL with kernel_regime='paper'",
    },
}


def get_backend_convention(backend: str) -> Dict:
    """Get the conventions for a backend."""
    if backend in BACKEND_CONVENTIONS:
        return BACKEND_CONVENTIONS[backend]
    return {
        "factorial_mode": FactorialMode.COEFFICIENT,
        "sign_mode": SignMode.NONE,
        "pair_order": PairOrderMode.TRIANGLE,
        "notes": "Unknown backend, assuming coefficient mode",
    }


# ============================================================================
# PAIR KEY UTILITIES
# ============================================================================

def parse_pair_key(key: str) -> Tuple[int, int]:
    """
    Parse a pair key string to (ell1, ell2).

    LOCKED CONVENTION: "13" always means (ell1=1, ell2=3).
    """
    if len(key) != 2:
        raise ValueError(f"Invalid pair key: {key}")
    return (int(key[0]), int(key[1]))


def make_pair_key(ell1: int, ell2: int) -> str:
    """Make a pair key from (ell1, ell2)."""
    return f"{ell1}{ell2}"


def get_triangle_pairs() -> list:
    """Get the 6 triangle pairs in canonical order."""
    return ["11", "22", "33", "12", "13", "23"]


def get_ordered_pairs() -> list:
    """Get all 9 ordered pairs."""
    return [
        "11", "12", "13",
        "21", "22", "23",
        "31", "32", "33",
    ]


def triangle_key_for_ordered(ell1: int, ell2: int) -> str:
    """Get the triangle key for an ordered pair (smaller index first)."""
    if ell1 <= ell2:
        return f"{ell1}{ell2}"
    else:
        return f"{ell2}{ell1}"
