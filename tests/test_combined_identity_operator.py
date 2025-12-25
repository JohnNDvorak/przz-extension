"""
tests/test_combined_identity_operator.py
Tests for the combined identity operator module (Leibniz expansion).

Tests verify:
1. Leibniz expansion produces correct number of terms
2. Term coefficients are correct
3. Evaluation at sample points is finite and stable
4. Branch separation (plus vs minus) is correct
5. Grouping by 1/(α+β) derivative order works
"""

import pytest
import numpy as np

from src.combined_identity_operator import (
    BracketTermContribution,
    OperatorExpansionResult,
    expand_QQ_on_bracket,
    evaluate_term_at_point,
    evaluate_QQB_at_point,
    compute_QQB_full,
    compute_QQB_by_branch,
    compute_QQB_by_inv_order,
    compute_QQB_at_przz_point,
    analyze_L_scaling,
)
from src.analytic_derivatives import binomial


class TestBracketTermContribution:
    """Test the BracketTermContribution dataclass."""

    def test_properties(self):
        """Test derived properties."""
        term = BracketTermContribution(
            i=2, j=3, q_i=1.0, q_j=1.0,
            k=1, l=2,
            binom_ik=2, binom_jl=3,
            sign=1.0
        )

        assert term.exp_deriv_alpha == 1
        assert term.exp_deriv_beta == 2
        assert term.inv_deriv_alpha == 1  # i-k = 2-1
        assert term.inv_deriv_beta == 1   # j-l = 3-2

    def test_coefficient(self):
        """Test coefficient computation."""
        term = BracketTermContribution(
            i=2, j=1, q_i=2.0, q_j=3.0,
            k=1, l=0,
            binom_ik=2, binom_jl=1,
            sign=-1.0
        )

        # coeff = q_i × q_j × C(i,k) × C(j,l) × sign
        # = 2 × 3 × 2 × 1 × (-1) = -12
        assert term.coefficient == -12.0

    def test_string_representation(self):
        """Test __str__ produces readable output."""
        term = BracketTermContribution(
            i=1, j=1, q_i=1.0, q_j=1.0,
            k=0, l=1,
            binom_ik=1, binom_jl=1,
            sign=1.0
        )
        s = str(term)
        assert "Term[+]" in s
        assert "q_1" in s


class TestExpandQQOnBracket:
    """Test the Leibniz expansion function."""

    def test_constant_Q(self):
        """Q = 1 (constant): only (i,j) = (0,0) contributes."""
        Q_coeffs = [1.0]  # Q(x) = 1
        expansion = expand_QQ_on_bracket(Q_coeffs)

        # D^0(fg) = fg, so we have 2 terms: plus and minus
        assert expansion.n_terms == 2

        # Both terms have i=j=k=l=0
        for term in expansion.terms:
            assert term.i == 0
            assert term.j == 0
            assert term.k == 0
            assert term.l == 0

    def test_linear_Q(self):
        """Q = 1 + x: (i,j) ∈ {(0,0), (0,1), (1,0), (1,1)}."""
        Q_coeffs = [1.0, 1.0]  # Q(x) = 1 + x
        expansion = expand_QQ_on_bracket(Q_coeffs)

        # For each (i,j), we have:
        # - (i+1) choices for k
        # - (j+1) choices for l
        # - 2 branches (plus and minus)
        # (0,0): 1×1×2 = 2
        # (0,1): 1×2×2 = 4
        # (1,0): 2×1×2 = 4
        # (1,1): 2×2×2 = 8
        # Total: 2 + 4 + 4 + 8 = 18
        assert expansion.n_terms == 18

    def test_quadratic_Q(self):
        """Q = 1 + x + x²: count total terms."""
        Q_coeffs = [1.0, 1.0, 1.0]  # Q(x) = 1 + x + x²
        expansion = expand_QQ_on_bracket(Q_coeffs)

        # Sum over i,j in {0,1,2}:
        # n_terms = 2 × Σ_{i,j} (i+1)(j+1)
        # = 2 × [1×1 + 1×2 + 1×3 + 2×1 + 2×2 + 2×3 + 3×1 + 3×2 + 3×3]
        # = 2 × [1 + 2 + 3 + 2 + 4 + 6 + 3 + 6 + 9]
        # = 2 × 36 = 72
        expected = 2 * sum((i + 1) * (j + 1) for i in range(3) for j in range(3))
        assert expansion.n_terms == expected

    def test_branch_separation(self):
        """Test that plus and minus branches are correctly separated."""
        Q_coeffs = [1.0, 1.0]
        expansion = expand_QQ_on_bracket(Q_coeffs)

        plus_terms, minus_terms = expansion.terms_by_branch()

        assert len(plus_terms) == len(minus_terms)
        assert all(t.sign > 0 for t in plus_terms)
        assert all(t.sign < 0 for t in minus_terms)

    def test_inv_order_grouping(self):
        """Test grouping by 1/(α+β) derivative order."""
        Q_coeffs = [1.0, 1.0, 1.0]
        expansion = expand_QQ_on_bracket(Q_coeffs)

        by_order = expansion.terms_by_inv_order()

        # inv_order = (i-k) + (j-l) ranges from 0 to max(i)+max(j)
        assert 0 in by_order  # Some terms have no derivatives on inv
        assert max(by_order.keys()) <= 4  # max(i) + max(j) = 2 + 2 = 4

    def test_max_order_limit(self):
        """Test that max_order limits the expansion."""
        Q_coeffs = [1.0, 1.0, 1.0, 1.0]  # degree 3
        expansion_full = expand_QQ_on_bracket(Q_coeffs, max_order=3)
        expansion_limited = expand_QQ_on_bracket(Q_coeffs, max_order=1)

        assert expansion_limited.n_terms < expansion_full.n_terms

        # Limited should only have i,j ≤ 1
        for term in expansion_limited.terms:
            assert term.i <= 1
            assert term.j <= 1


class TestEvaluateTerm:
    """Test evaluation of individual terms."""

    @pytest.fixture
    def simple_term_plus(self):
        """A simple plus branch term: D^0(exp × inv)."""
        return BracketTermContribution(
            i=0, j=0, q_i=1.0, q_j=1.0,
            k=0, l=0,
            binom_ik=1, binom_jl=1,
            sign=1.0
        )

    @pytest.fixture
    def simple_term_minus(self):
        """A simple minus branch term."""
        return BracketTermContribution(
            i=0, j=0, q_i=1.0, q_j=1.0,
            k=0, l=0,
            binom_ik=1, binom_jl=1,
            sign=-1.0
        )

    def test_plus_term_evaluation(self, simple_term_plus):
        """Evaluate plus branch term at a point."""
        theta = 4.0 / 7.0
        L = 10.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        value = evaluate_term_at_point(
            simple_term_plus, alpha, beta, x, y, theta, L
        )

        # Plus branch: exp(θL(αx+βy)) × 1/(α+β)
        exp_val = np.exp(theta * L * (alpha * x + beta * y))
        inv_val = 1.0 / (alpha + beta)
        expected = exp_val * inv_val

        assert np.isclose(value, expected)

    def test_minus_term_evaluation(self, simple_term_minus):
        """Evaluate minus branch term at a point."""
        theta = 4.0 / 7.0
        L = 10.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        value = evaluate_term_at_point(
            simple_term_minus, alpha, beta, x, y, theta, L
        )

        # Minus branch: -exp(-L(α+β) - θL(βx+αy)) × 1/(α+β)
        exponent = -L * ((alpha + beta) + theta * (beta * x + alpha * y))
        exp_val = np.exp(exponent)
        inv_val = 1.0 / (alpha + beta)
        expected = -1.0 * exp_val * inv_val

        assert np.isclose(value, expected)

    def test_term_with_derivatives(self):
        """Evaluate term with derivatives on both factors."""
        term = BracketTermContribution(
            i=1, j=1, q_i=1.0, q_j=1.0,
            k=1, l=0,  # 1 deriv on exp in α, 0 in β; 0 on inv in α, 1 in β
            binom_ik=1, binom_jl=1,
            sign=1.0
        )

        theta = 4.0 / 7.0
        L = 10.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        value = evaluate_term_at_point(term, alpha, beta, x, y, theta, L)

        # Should be finite
        assert np.isfinite(value)


class TestComputeQQB:
    """Test the full QQB computation."""

    def test_simple_Q_finite_result(self):
        """Q=1 should give finite result."""
        Q_coeffs = [1.0]
        theta = 4.0 / 7.0
        L = 10.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        value = compute_QQB_full(Q_coeffs, alpha, beta, x, y, theta, L)

        assert np.isfinite(value)

    def test_linear_Q_finite_result(self):
        """Q=1+x should give finite result."""
        Q_coeffs = [1.0, 1.0]
        theta = 4.0 / 7.0
        L = 10.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        value = compute_QQB_full(Q_coeffs, alpha, beta, x, y, theta, L)

        assert np.isfinite(value)

    def test_branch_sum_equals_total(self):
        """Plus + minus branches should equal total."""
        Q_coeffs = [1.0, 0.5, 0.25]
        theta = 4.0 / 7.0
        L = 10.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        plus, minus, total = compute_QQB_by_branch(
            Q_coeffs, alpha, beta, x, y, theta, L
        )

        assert np.isclose(plus + minus, total)

    def test_inv_order_sum_equals_total(self):
        """Sum over 1/(α+β) orders should equal total."""
        Q_coeffs = [1.0, 0.5, 0.25]
        theta = 4.0 / 7.0
        L = 10.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        total = compute_QQB_full(Q_coeffs, alpha, beta, x, y, theta, L)
        by_order = compute_QQB_by_inv_order(Q_coeffs, alpha, beta, x, y, theta, L)

        order_sum = sum(by_order.values())
        assert np.isclose(order_sum, total)


class TestPRZZPoint:
    """Test evaluation at the PRZZ point α=β=-R/L."""

    def test_przz_point_finite(self):
        """Result at PRZZ point should be finite."""
        Q_coeffs = [1.0, -0.5, 0.1]
        R = 1.3036
        L = 100.0
        x, y = 0.3, 0.4
        theta = 4.0 / 7.0

        result = compute_QQB_at_przz_point(Q_coeffs, R, L, x, y, theta)

        assert np.isfinite(result["total"])
        assert np.isfinite(result["plus"])
        assert np.isfinite(result["minus"])

    def test_przz_point_correct_alpha_beta(self):
        """Check α=β=-R/L at PRZZ point."""
        Q_coeffs = [1.0]
        R = 1.3036
        L = 100.0
        x, y = 0.3, 0.4

        result = compute_QQB_at_przz_point(Q_coeffs, R, L, x, y)

        expected_alpha = -R / L
        assert np.isclose(result["alpha"], expected_alpha)
        assert np.isclose(result["beta"], expected_alpha)
        assert np.isclose(result["sum_ab"], -2 * R / L)

    def test_przz_point_xy_zero(self):
        """At x=y=0, the bracket simplifies."""
        Q_coeffs = [1.0]  # Q=1, only zeroth order
        R = 1.3036
        L = 100.0
        x, y = 0.0, 0.0

        result = compute_QQB_at_przz_point(Q_coeffs, R, L, x, y)

        # At x=y=0:
        # Plus: exp(0) × 1/(α+β) = 1/(α+β) = -L/(2R)
        # Minus: -exp(-L(α+β)) × 1/(α+β) = -exp(2R) × (-L/(2R)) = exp(2R) × L/(2R)
        # Total: -L/(2R) + exp(2R) × L/(2R) = L/(2R) × (exp(2R) - 1)
        inv_val = -L / (2 * R)
        exp_2R = np.exp(2 * R)
        expected_total = inv_val + exp_2R * inv_val * (-1)
        # Wait, let me recalculate:
        # plus = exp(0) × 1/(α+β) = 1 × (-L/(2R)) = -L/(2R)
        # minus = -1 × exp(2R) × (-L/(2R)) = exp(2R) × L/(2R)
        # total = -L/(2R) + exp(2R) × L/(2R) = (L/(2R)) × (exp(2R) - 1)
        expected = (L / (2 * R)) * (exp_2R - 1)

        assert np.isclose(result["total"], expected, rtol=1e-6)


class TestLScaling:
    """Test L-scaling behavior at PRZZ point."""

    def test_l_scaling_finite_all(self):
        """Results should be finite for all L values."""
        Q_coeffs = [1.0, -0.5]
        R = 1.3036
        x, y = 0.3, 0.4
        L_values = [10, 50, 100, 500]

        results = analyze_L_scaling(Q_coeffs, R, x, y, L_values)

        for r in results:
            assert np.isfinite(r["total"])
            assert np.isfinite(r["plus"])
            assert np.isfinite(r["minus"])

    def test_l_scaling_with_real_Q(self):
        """Test with realistic Q coefficients."""
        # Simplified Q coefficients (real ones have more terms)
        Q_coeffs = [1.0, -0.3, 0.05]
        R = 1.3036
        x, y = 0.5, 0.5
        L_values = [10, 100, 1000]

        results = analyze_L_scaling(Q_coeffs, R, x, y, L_values)

        # Record the L-dependence
        totals = [r["total"] for r in results]

        # All should be finite
        assert all(np.isfinite(t) for t in totals)


class TestConstantQIdentity:
    """Test that Q=1 gives the expected bracket structure."""

    def test_constant_Q_bracket_value(self):
        """For Q=1, QQB = B = [exp_plus - exp_minus]/(α+β)."""
        Q_coeffs = [1.0]
        theta = 4.0 / 7.0
        L = 10.0
        alpha, beta = 0.1, 0.2
        x, y = 0.3, 0.4

        value = compute_QQB_full(Q_coeffs, alpha, beta, x, y, theta, L)

        # Direct computation of B
        exp_plus = np.exp(theta * L * (alpha * x + beta * y))
        exp_minus = np.exp(-L * (alpha + beta) - theta * L * (beta * x + alpha * y))
        inv = 1.0 / (alpha + beta)
        expected = (exp_plus - exp_minus) * inv

        assert np.isclose(value, expected)
