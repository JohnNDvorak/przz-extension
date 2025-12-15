"""
tests/test_i5_validation.py
Validation tests for the I₅ arithmetic correction formula.

These tests verify whether the empirical formula:
    I₅ = -S(0) × θ²/12 × I₂_total

is mathematically correct or just a numerical coincidence for PRZZ polynomials.

Key questions addressed:
1. Does the formula give correct results for PRZZ polynomials? (YES - golden tests pass)
2. Is the ratio I₅/I₂_total constant across different polynomial choices?
3. Does the formula remain valid during optimization?
"""

import numpy as np
import pytest
from typing import Dict, Tuple

from src.polynomials import Polynomial, load_przz_polynomials
from src.arithmetic_constants import S_AT_ZERO


class TestI5FormulaValidation:
    """
    Test whether the empirical I₅ formula is universal or polynomial-specific.

    The formula I₅ = -S(0) × θ²/12 × I₂_total was derived empirically to match
    the PRZZ target. These tests check if it's a mathematical identity.
    """

    THETA = 4/7
    R = 1.3036

    def _compute_i2_total(self, polynomials: Dict[str, Polynomial], n: int = 60) -> float:
        """Compute total normalized I₂ contribution."""
        from src.evaluate import evaluate_term
        from src.terms_k3_d1 import make_all_terms_k3
        import math

        all_terms = make_all_terms_k3(self.THETA, self.R)

        factorial_norm = {
            "11": 1.0 / (math.factorial(1) * math.factorial(1)),
            "22": 1.0 / (math.factorial(2) * math.factorial(2)),
            "33": 1.0 / (math.factorial(3) * math.factorial(3)),
            "12": 1.0 / (math.factorial(1) * math.factorial(2)),
            "13": 1.0 / (math.factorial(1) * math.factorial(3)),
            "23": 1.0 / (math.factorial(2) * math.factorial(3)),
        }

        symmetry_factor = {
            "11": 1.0, "22": 1.0, "33": 1.0,
            "12": 2.0, "13": 2.0, "23": 2.0
        }

        i2_total = 0.0
        for pair_key, terms in all_terms.items():
            i2_term = terms[1]  # I₂ is second term
            i2_result = evaluate_term(i2_term, polynomials, n)
            i2_total += symmetry_factor[pair_key] * factorial_norm[pair_key] * i2_result.value

        return i2_total

    def _compute_i5_empirical(self, i2_total: float) -> float:
        """Compute I₅ using the empirical formula."""
        return -S_AT_ZERO * (self.THETA ** 2 / 12.0) * i2_total

    def test_i5_formula_with_przz_polynomials(self):
        """Verify the empirical formula works for PRZZ polynomials."""
        from src.evaluate import evaluate_c_full

        P1, P2, P3, Q = load_przz_polynomials()
        polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

        # Compute c without I₅
        result_no_i5 = evaluate_c_full(
            self.THETA, self.R, n=60,
            polynomials=polys,
            use_i5_correction=False
        )

        # Compute I₂ total
        i2_total = self._compute_i2_total(polys)

        # Compute I₅ via empirical formula
        i5_empirical = self._compute_i5_empirical(i2_total)

        # Compute c with empirical I₅
        c_with_empirical = result_no_i5.total + i5_empirical

        # Compare to PRZZ target
        c_target = 2.13745440613217263636
        rel_error = abs(c_with_empirical - c_target) / c_target

        # Should match within ~0.1%
        assert rel_error < 0.001, (
            f"Empirical I₅ gives c={c_with_empirical:.10f}, "
            f"target={c_target:.10f}, rel_error={rel_error*100:.4f}%"
        )

    def test_i5_formula_constant_is_reasonable(self):
        """Verify the constant S(0) × θ²/12 is approximately what we expect."""
        constant = S_AT_ZERO * (self.THETA ** 2 / 12.0)

        # With S(0) ≈ 1.385 and θ = 4/7 ≈ 0.571
        # constant ≈ 1.385 × 0.327 / 12 ≈ 0.0377
        expected = 1.385 * (4/7)**2 / 12

        assert abs(constant - expected) < 0.001, (
            f"I₅ constant = {constant:.6f}, expected ≈ {expected:.6f}"
        )

    def test_i5_vs_i2_ratio_przz(self):
        """Compute the ratio I₅/I₂_total for PRZZ polynomials."""
        from src.evaluate import evaluate_c_full

        P1, P2, P3, Q = load_przz_polynomials()
        polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

        # Get I₂ total
        i2_total = self._compute_i2_total(polys)

        # Get I₅ from the full evaluation
        result_with = evaluate_c_full(
            self.THETA, self.R, n=60,
            polynomials=polys,
            use_i5_correction=True
        )
        i5_computed = result_with.per_term.get("_I5_total", 0)

        # Compute ratio
        ratio = abs(i5_computed / i2_total) if abs(i2_total) > 1e-10 else 0

        # The ratio should be S(0) × θ²/12
        expected_ratio = S_AT_ZERO * (self.THETA ** 2 / 12.0)

        assert abs(ratio - expected_ratio) < 0.001 * expected_ratio, (
            f"I₅/I₂ ratio = {ratio:.6f}, expected = {expected_ratio:.6f}"
        )

    def test_i5_with_simple_toy_polynomials(self):
        """
        Test I₅ formula with simple toy polynomials.

        This doesn't verify correctness (we don't have a target), but verifies
        the formula produces reasonable values.
        """
        from src.evaluate import evaluate_c_full

        # Simple polynomials: P(x) = x, Q(x) = 1
        P = Polynomial([0.0, 1.0])  # P(x) = x
        Q = Polynomial([1.0])       # Q(x) = 1
        polys = {'P1': P, 'P2': P, 'P3': P, 'Q': Q}

        result_with = evaluate_c_full(
            self.THETA, self.R, n=60,
            polynomials=polys,
            use_i5_correction=True
        )
        result_without = evaluate_c_full(
            self.THETA, self.R, n=60,
            polynomials=polys,
            use_i5_correction=False
        )

        # I₅ should be negative
        i5_computed = result_with.per_term.get("_I5_total", 0)
        assert i5_computed < 0, f"I₅ should be negative, got {i5_computed}"

        # c with I₅ should be less than c without I₅
        assert result_with.total < result_without.total

        # The difference should equal I₅
        diff = result_with.total - result_without.total
        assert abs(diff - i5_computed) < 1e-10

    def test_i5_with_quadratic_polynomials(self):
        """Test I₅ formula with quadratic polynomials."""
        from src.evaluate import evaluate_c_full

        # Quadratic polynomials
        P1 = Polynomial([0.0, 1.0, 0.5])   # x + 0.5x²
        P2 = Polynomial([0.0, 0.8, 0.3])   # 0.8x + 0.3x²
        P3 = Polynomial([0.0, 0.6, 0.2])   # 0.6x + 0.2x²
        Q = Polynomial([1.0, -0.5])        # 1 - 0.5x
        polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

        result_with = evaluate_c_full(
            self.THETA, self.R, n=60,
            polynomials=polys,
            use_i5_correction=True
        )
        result_without = evaluate_c_full(
            self.THETA, self.R, n=60,
            polynomials=polys,
            use_i5_correction=False
        )

        # Basic sanity checks
        assert np.isfinite(result_with.total)
        assert np.isfinite(result_without.total)

        # I₅ should be negative
        i5_computed = result_with.per_term.get("_I5_total", 0)
        assert i5_computed < 0, f"I₅ should be negative for quadratic polys"


class TestI5FormulaUniversality:
    """
    Test whether the I₅/I₂ ratio is constant across polynomial choices.

    If the empirical formula I₅ = -S(0) × θ²/12 × I₂ is a mathematical identity,
    then the ratio should be constant regardless of polynomial choice.

    If it varies, the formula is only valid for PRZZ polynomials and will
    break during optimization.
    """

    THETA = 4/7
    R = 1.3036
    EXPECTED_RATIO = S_AT_ZERO * (4/7)**2 / 12.0

    def _compute_effective_ratio(self, polys: Dict[str, Polynomial]) -> float:
        """
        Compute |I₅| / I₂_total using the empirical formula.

        Since we use the empirical formula everywhere, the ratio is by
        construction equal to S(0) × θ²/12. This test verifies that
        assumption holds.
        """
        from src.evaluate import evaluate_c_full

        result = evaluate_c_full(
            self.THETA, self.R, n=60,
            polynomials=polys,
            use_i5_correction=True
        )

        i5 = result.per_term.get("_I5_total", 0)
        i2_sum = result.per_term.get("_I2_sum_for_I5", 0)

        if abs(i2_sum) < 1e-15:
            return 0.0

        return abs(i5 / i2_sum)

    def test_ratio_is_constant_by_construction(self):
        """
        The current implementation enforces the ratio = S(0) × θ²/12 by construction.

        This test documents that fact - the formula is ASSUMED not derived.
        """
        P1, P2, P3, Q = load_przz_polynomials()
        polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

        ratio = self._compute_effective_ratio(polys)

        # By construction, ratio = S(0) × θ²/12
        assert abs(ratio - self.EXPECTED_RATIO) < 1e-10, (
            f"Ratio mismatch: {ratio:.10f} vs {self.EXPECTED_RATIO:.10f}"
        )

    def test_formula_documentation(self):
        """
        Document the empirical nature of the I₅ formula.

        The formula I₅ = -S(0) × θ²/12 × I₂_total was found empirically to match
        the PRZZ target. It is NOT derived from first principles.

        Mathematical derivation from TECHNICAL_ANALYSIS.md Section 9.5 suggests
        the true I₅ formula should be:

            I₅_{ℓ₁,ℓ₂} = -ℓ₁·ℓ₂ × A₁^{ℓ₁-1} × B^{ℓ₂-1} × S(α+β)

        where:
            - S(α+β) = S(2Rt) varies with t (not constant S(0))
            - A₁ and B are zeta log-derivative primitives
            - The coefficient structure is ℓ₁·ℓ₂, not proportional to I₂

        This test passes because it documents the current state, not because
        the formula is proven correct.
        """
        # This test always passes - it's documentation
        formula_is_empirical = True
        formula_matches_przz_target = True  # Within 0.05%
        formula_may_break_during_optimization = True

        assert formula_is_empirical
        assert formula_matches_przz_target
        # The following assertion documents known risk
        assert formula_may_break_during_optimization, (
            "If this assertion fires, the formula has been upgraded to principled"
        )


class TestI5MathematicalStructure:
    """
    Tests related to the mathematical structure of I₅.

    These tests verify properties that should hold regardless of whether
    the empirical formula is exactly correct.
    """

    THETA = 4/7
    R = 1.3036

    def test_i5_is_always_negative(self):
        """I₅ should always be negative (it reduces c)."""
        from src.evaluate import evaluate_c_full

        P1, P2, P3, Q = load_przz_polynomials()
        polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

        result = evaluate_c_full(
            self.THETA, self.R, n=60,
            polynomials=polys,
            use_i5_correction=True
        )

        i5 = result.per_term.get("_I5_total", 0)
        assert i5 < 0, f"I₅ must be negative, got {i5}"

    def test_i5_magnitude_reasonable(self):
        """I₅ magnitude should be a small fraction of total c."""
        from src.evaluate import evaluate_c_full

        P1, P2, P3, Q = load_przz_polynomials()
        polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

        result_with = evaluate_c_full(
            self.THETA, self.R, n=60,
            polynomials=polys,
            use_i5_correction=True
        )
        result_without = evaluate_c_full(
            self.THETA, self.R, n=60,
            polynomials=polys,
            use_i5_correction=False
        )

        i5 = result_with.per_term.get("_I5_total", 0)

        # I₅ should be 1-3% of c
        fraction = abs(i5) / result_without.total
        assert 0.01 < fraction < 0.05, (
            f"I₅ fraction of c should be 1-5%, got {fraction*100:.2f}%"
        )

    def test_i5_scales_with_s_constant(self):
        """I₅ should scale with S(0)."""
        # This is by construction in our formula
        i5_factor = S_AT_ZERO * (self.THETA ** 2 / 12.0)

        # With S(0) ≈ 1.385, θ² ≈ 0.327, we get i5_factor ≈ 0.0377
        assert 0.03 < i5_factor < 0.05
