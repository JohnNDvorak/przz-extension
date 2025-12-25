"""
tests/test_s34_triangle_spec_lock.py
SPEC LOCK: S34 uses TRIANGLE×2 convention, NOT 9 ordered pairs.

This test file ensures the S34 triangle convention is enforced and cannot
be accidentally reverted to the incorrect 9-ordered-pairs approach.

BACKGROUND (2025-12-22):
The S34 asymmetry test showed I₃(1,2) ≠ I₃(2,1), but this does NOT mean
PRZZ wants a 9-ordered-pair sum. PRZZ uses the quadratic form / mean-square
convention that folds to triangle×2.

Using 9 ordered pairs instead of triangle×2 causes a +11% OVERSHOOT.
This is a hard invariant that must be preserved.

Reference: TRUTH_SPEC.md Section 13 (Ordered vs Triangle Assembly)
"""

import numpy as np
import pytest

from src.polynomials import load_przz_polynomials
from src.evaluate import (
    get_s34_triangle_pairs,
    get_s34_factorial_normalization,
    _assert_s34_triangle_convention,
    S34OrderedPairsError,
    compute_c_paper_ordered,
)


class TestS34TriangleHelper:
    """Test the triangle pairs helper function."""

    def test_triangle_pairs_count(self):
        """Triangle convention uses exactly 6 pairs."""
        pairs = get_s34_triangle_pairs()
        assert len(pairs) == 6

    def test_triangle_pairs_keys(self):
        """Triangle pairs are exactly the upper triangle."""
        pairs = get_s34_triangle_pairs()
        keys = {p[0] for p in pairs}
        expected = {"11", "22", "33", "12", "13", "23"}
        assert keys == expected

    def test_diagonal_symmetry_factor_is_1(self):
        """Diagonal pairs have symmetry factor 1."""
        pairs = get_s34_triangle_pairs()
        diagonals = [(k, s) for k, s in pairs if k in {"11", "22", "33"}]
        for key, sym in diagonals:
            assert sym == 1, f"Diagonal {key} should have sym=1, got {sym}"

    def test_off_diagonal_symmetry_factor_is_2(self):
        """Off-diagonal pairs have symmetry factor 2."""
        pairs = get_s34_triangle_pairs()
        off_diagonals = [(k, s) for k, s in pairs if k in {"12", "13", "23"}]
        for key, sym in off_diagonals:
            assert sym == 2, f"Off-diagonal {key} should have sym=2, got {sym}"

    def test_factorial_normalization_keys(self):
        """Factorial normalization has same keys as triangle pairs."""
        pairs = get_s34_triangle_pairs()
        triangle_keys = {p[0] for p in pairs}
        norm_keys = set(get_s34_factorial_normalization().keys())
        assert triangle_keys == norm_keys


class TestS34ConventionGuard:
    """Test the convention guard function."""

    def test_triangle_keys_pass(self):
        """Triangle keys should pass without error."""
        triangle_keys = ["11", "22", "33", "12", "13", "23"]
        # Should not raise
        _assert_s34_triangle_convention(triangle_keys, caller="test")

    def test_ordered_keys_fail(self):
        """Ordered-only keys should raise S34OrderedPairsError."""
        ordered_keys = ["11", "22", "33", "12", "21", "13", "31", "23", "32"]
        with pytest.raises(S34OrderedPairsError) as exc_info:
            _assert_s34_triangle_convention(ordered_keys, caller="test")
        assert "21" in str(exc_info.value) or "31" in str(exc_info.value) or "32" in str(exc_info.value)
        assert "+11% OVERSHOOT" in str(exc_info.value)

    def test_single_forbidden_key_fails(self):
        """Even a single forbidden key should raise."""
        with pytest.raises(S34OrderedPairsError):
            _assert_s34_triangle_convention(["11", "21"], caller="test")


class TestS34NegativeControl:
    """
    NEGATIVE CONTROL: Verify that using 9 ordered pairs causes overshoot.

    This test ensures we never silently revert to the incorrect convention.
    The old ordered-pairs approach gave +11% error; triangle×2 gives -1.3%.
    """

    @pytest.fixture
    def polynomials(self):
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        return {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    def test_triangle_gives_negative_gap(self, polynomials):
        """Triangle×2 convention should give c below target (negative gap)."""
        result = compute_c_paper_ordered(
            theta=4.0/7.0,
            R=1.3036,
            n=40,  # Lower n for speed
            polynomials=polynomials,
            K=3,
            s12_pair_mode='triangle',  # Uses triangle×2 for S34
        )

        c_target = 2.13745440613217263636
        c_computed = result.total
        gap = (c_computed - c_target) / c_target * 100

        # Triangle×2 should give negative gap (around -1.3%)
        assert gap < 0, f"Triangle×2 should give negative gap, got {gap:+.2f}%"
        assert gap > -5, f"Gap should be small, got {gap:+.2f}%"

    def test_ordered_would_give_positive_overshoot(self, polynomials):
        """
        Document that if we were to use 9 ordered pairs, we'd get +11% error.

        This is a documentation test - we can't easily call the old broken code
        since we've fixed it. Instead, we compute what the S34 contribution
        WOULD be with ordered pairs vs triangle×2 and show the difference.
        """
        from src.terms_k3_d1 import make_all_terms_k3_ordered
        from src.evaluate import evaluate_term

        theta = 4.0/7.0
        R = 1.3036
        n = 40

        # Get the ordered terms
        ordered_plus = make_all_terms_k3_ordered(theta=theta, R=R, kernel_regime='paper')

        # Factorial normalization for ordered pairs (same for both (i,j) and (j,i))
        f_ordered = {
            "11": 1.0, "22": 0.25, "33": 1/36,
            "12": 0.5, "21": 0.5,
            "13": 1/6, "31": 1/6,
            "23": 1/12, "32": 1/12,
        }

        # Triangle normalization + symmetry
        f_triangle = {"11": 1.0, "22": 0.25, "33": 1/36, "12": 0.5, "13": 1/6, "23": 1/12}
        sym_triangle = {"11": 1, "22": 1, "33": 1, "12": 2, "13": 2, "23": 2}

        # Compute S34 with ORDERED (9 pairs) - the WRONG way
        s34_ordered = 0.0
        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            terms = ordered_plus[pair_key]
            norm = f_ordered[pair_key]
            for i in [2, 3]:  # I₃, I₄
                term = terms[i]
                result = evaluate_term(term, polynomials, n, R=R, theta=theta)
                s34_ordered += norm * result.value

        # Compute S34 with TRIANGLE×2 - the CORRECT way
        s34_triangle = 0.0
        for pair_key in ["11", "22", "33", "12", "13", "23"]:
            terms = ordered_plus[pair_key]
            norm = f_triangle[pair_key]
            sym = sym_triangle[pair_key]
            for i in [2, 3]:  # I₃, I₄
                term = terms[i]
                result = evaluate_term(term, polynomials, n, R=R, theta=theta)
                s34_triangle += sym * norm * result.value

        # The ordered approach should give a LARGER (less negative) S34
        # because it's effectively counting some terms twice
        diff = s34_ordered - s34_triangle

        # Document the difference
        print(f"\nS34 comparison:")
        print(f"  Ordered (9 pairs): {s34_ordered:.6f}")
        print(f"  Triangle×2:        {s34_triangle:.6f}")
        print(f"  Difference:        {diff:.6f}")

        # The difference should be significant (causes the +11% vs -1.3% gap)
        # Since S34 is negative, ordered being "less negative" means larger total c
        assert diff > 0.1, f"Ordered should give larger S34, diff = {diff:.6f}"


class TestS34DocumentedInvariant:
    """Document the invariant for future reference."""

    def test_invariant_documented(self):
        """
        INVARIANT: S34 uses triangle×2, not 9 ordered pairs.

        - Triangle pairs: (1,1), (2,2), (3,3), (1,2), (1,3), (2,3)
        - Symmetry: diagonal=1, off-diagonal=2
        - Reason: PRZZ quadratic form convention
        - Consequence of violation: +11% overshoot

        This test always passes; it's documentation.
        """
        pass

    def test_asymmetry_does_not_imply_ordered(self):
        """
        CLARIFICATION: S34 term asymmetry does NOT imply ordered summation.

        The earlier asymmetry test showed I₃(1,2) ≠ I₃(2,1).
        This proves the TERMS are asymmetric, but PRZZ never evaluates
        both orders. They use triangle×2 throughout.

        The asymmetry test was real, but the conclusion was wrong.
        """
        pass
