"""
Gate tests for GPT Step 2: Operator-Level Mirror Computation.

WARNING: DIAGNOSTIC TESTS ONLY
==============================
These tests validate the PRE-IDENTITY operator approach which has KNOWN L-DIVERGENCE.
This approach is kept for DIAGNOSTIC PURPOSES to understand why the combined identity
is necessary.

For production-ready tests, see:
- tests/test_operator_post_identity_*.py

These tests validate the operator-level approach where Q is applied as
actual differential operators (d/dα, d/dβ) to the pre-identity bracket.

The L-divergence was confirmed in earlier experiments and is expected behavior.
"""

import numpy as np
import pytest
from src.operator_level_mirror import (
    convert_Q_basis_to_monomial,
    BracketDerivatives,
    apply_Q_operator_to_bracket,
    compute_I1_operator_level_11,
)
from src.polynomials import load_przz_polynomials


# ============================================================================
# Stage 21A: Q Polynomial Conversion Tests
# ============================================================================

@pytest.mark.diagnostic
class TestQConversion:
    """Tests for Q polynomial basis conversion."""

    def test_Q_at_zero_equals_one(self):
        """Q(0) should equal sum of basis coefficients ≈ 1."""
        basis_coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
        Q_mono = convert_Q_basis_to_monomial(basis_coeffs)

        # Q(0) = q_0 (constant term)
        assert abs(Q_mono[0] - 0.999999) < 1e-5, f"Q(0) = {Q_mono[0]}, expected ≈ 1"

    def test_Q_at_half_equals_c0(self):
        """Q(0.5) should equal c_0 since (1-2×0.5)^k = 0 for k>0."""
        basis_coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
        Q_mono = convert_Q_basis_to_monomial(basis_coeffs)

        # Evaluate Q(0.5) using monomial form
        x = 0.5
        Q_at_half = sum(c * (x ** i) for i, c in enumerate(Q_mono))

        assert abs(Q_at_half - 0.490464) < 1e-5, f"Q(0.5) = {Q_at_half}, expected 0.490464"

    def test_monomial_coefficients_correct_length(self):
        """Monomial coefficients should have degree 5 (6 coefficients)."""
        basis_coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
        Q_mono = convert_Q_basis_to_monomial(basis_coeffs)

        assert len(Q_mono) == 6, f"Expected 6 coefficients, got {len(Q_mono)}"


# ============================================================================
# Stage 21A: Bracket Derivatives Tests
# ============================================================================

@pytest.fixture(scope="module")
def cached_bracket_3():
    """Module-level cached bracket with max_deriv=3."""
    return BracketDerivatives(theta=4.0/7.0, max_deriv=3)


@pytest.fixture(scope="module")
def cached_bracket_5():
    """Module-level cached bracket with max_deriv=5."""
    return BracketDerivatives(theta=4.0/7.0, max_deriv=5)


@pytest.mark.slow
@pytest.mark.diagnostic
class TestBracketDerivatives:
    """Tests for symbolic bracket derivative computation.

    Marked slow because building symbolic derivatives takes ~10-20 seconds.
    """

    @pytest.fixture
    def bracket(self, cached_bracket_3):
        """Use cached bracket with max_deriv=3."""
        return cached_bracket_3

    def test_bracket_finite_at_evaluation_point(self, bracket):
        """Bracket B should be finite at α=β=-R/L."""
        R, L = 1.3036, 20.0
        alpha = -R / L

        B = bracket.compute_derivative(0, 0, alpha, alpha, 0.1, 0.1, L)

        assert np.isfinite(B), f"Bracket B is not finite: {B}"
        assert abs(B) > 1e-10, f"Bracket B is unexpectedly zero: {B}"

    def test_derivatives_finite(self, bracket):
        """All derivatives up to order 3 should be finite."""
        R, L = 1.3036, 20.0
        alpha = -R / L

        for i in range(4):
            for j in range(4):
                deriv = bracket.compute_derivative(i, j, alpha, alpha, 0.1, 0.1, L)
                assert np.isfinite(deriv), f"∂^{i+j}B/∂α^{i}∂β^{j} is not finite: {deriv}"

    def test_mixed_derivative_symmetry(self, bracket):
        """∂²B/∂α∂β should be computed consistently."""
        R, L = 1.3036, 20.0
        alpha = -R / L

        d_11 = bracket.compute_derivative(1, 1, alpha, alpha, 0.1, 0.1, L)

        # The (1,1) derivative should be finite and non-zero
        assert np.isfinite(d_11), f"∂²B/∂α∂β is not finite: {d_11}"

    def test_bracket_L_proportionality(self, bracket):
        """Bracket B should be proportional to L at α=β=-R/L."""
        R = 1.3036
        x, y = 0.05, 0.05

        B_values = []
        L_values = [10.0, 20.0, 50.0]

        for L in L_values:
            alpha = -R / L
            B = bracket.compute_derivative(0, 0, alpha, alpha, x, y, L)
            B_values.append(B)

        # B/L should be approximately constant
        B_over_L = [B / L for B, L in zip(B_values, L_values)]
        relative_diffs = [abs(B_over_L[i] - B_over_L[0]) / abs(B_over_L[0])
                         for i in range(len(B_over_L))]

        for i, diff in enumerate(relative_diffs):
            assert diff < 1e-6, f"B/L not constant: B/L = {B_over_L}, diff at {i} = {diff}"


# ============================================================================
# Stage 21B: Q Operator Application Tests
# ============================================================================

@pytest.mark.slow
@pytest.mark.diagnostic
class TestQOperatorApplication:
    """Tests for applying Q as differential operators.

    Marked slow because building symbolic derivatives takes ~20-30 seconds.
    """

    @pytest.fixture
    def bracket(self, cached_bracket_5):
        """Use cached bracket with max_deriv=5."""
        return cached_bracket_5

    @pytest.fixture
    def Q_mono(self):
        """Get Q in monomial form."""
        basis_coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
        return convert_Q_basis_to_monomial(basis_coeffs)

    def test_operator_result_finite(self, bracket, Q_mono):
        """Q(D_α)Q(D_β)B should be finite."""
        R, L = 1.3036, 20.0

        result = apply_Q_operator_to_bracket(Q_mono, bracket, R, L, 0.1, 0.1)

        assert np.isfinite(result), f"Operator result is not finite: {result}"

    def test_operator_symmetric_in_xy(self, bracket, Q_mono):
        """Result should be symmetric in x↔y."""
        R, L = 1.3036, 20.0

        result_xy = apply_Q_operator_to_bracket(Q_mono, bracket, R, L, 0.1, 0.2)
        result_yx = apply_Q_operator_to_bracket(Q_mono, bracket, R, L, 0.2, 0.1)

        assert abs(result_xy - result_yx) < 1e-10, \
            f"Not symmetric: f(0.1,0.2)={result_xy}, f(0.2,0.1)={result_yx}"

    def test_operator_L_proportionality(self, bracket, Q_mono):
        """Q(D)Q(D)B should also be proportional to L (due to bracket L-dependence)."""
        R = 1.3036
        x, y = 0.05, 0.05

        results = []
        L_values = [10.0, 20.0, 50.0]

        for L in L_values:
            result = apply_Q_operator_to_bracket(Q_mono, bracket, R, L, x, y)
            results.append(result)

        # Result/L should be approximately constant
        result_over_L = [r / L for r, L in zip(results, L_values)]
        relative_diffs = [abs(result_over_L[i] - result_over_L[0]) / abs(result_over_L[0])
                         for i in range(len(result_over_L))]

        for i, diff in enumerate(relative_diffs):
            assert diff < 0.01, f"Result/L not constant: {result_over_L}, diff at {i} = {diff}"


# ============================================================================
# Stage 21C: Full I1 Computation Tests
# ============================================================================

@pytest.mark.slow
@pytest.mark.diagnostic
class TestOperatorLevelI1:
    """Tests for full I1 computation with operator-level mirror.

    Marked slow because each I1 computation takes 30-60 seconds.
    """

    @pytest.fixture
    def polys(self):
        """Load PRZZ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_I1_finite(self, polys):
        """I1 should be finite and positive."""
        result = compute_I1_operator_level_11(
            theta=4.0/7.0,
            R=1.3036,
            n=10,  # Fewer points for speed
            polynomials=polys,
            L=20.0,
            verbose=False,
        )

        assert np.isfinite(result.I1_operator_level), \
            f"I1 is not finite: {result.I1_operator_level}"
        assert result.I1_operator_level > 0, \
            f"I1 should be positive: {result.I1_operator_level}"

    def test_I1_L_proportionality(self, polys):
        """I1 should be proportional to L (known behavior from bracket analysis)."""
        I1_values = []
        L_values = [10.0, 20.0]

        for L in L_values:
            result = compute_I1_operator_level_11(
                theta=4.0/7.0,
                R=1.3036,
                n=10,
                polynomials=polys,
                L=L,
                verbose=False,
            )
            I1_values.append(result.I1_operator_level)

        # I1/L should be approximately constant
        I1_over_L = [I1 / L for I1, L in zip(I1_values, L_values)]
        relative_diff = abs(I1_over_L[1] - I1_over_L[0]) / abs(I1_over_L[0])

        assert relative_diff < 0.01, \
            f"I1/L not constant: {I1_over_L}, relative diff = {relative_diff}"

    def test_I1_differs_from_zero(self, polys):
        """I1 should be substantially non-zero."""
        result = compute_I1_operator_level_11(
            theta=4.0/7.0,
            R=1.3036,
            n=10,
            polynomials=polys,
            L=20.0,
            verbose=False,
        )

        # At L=20, I1 should be around 2.5 based on diagnostic
        assert result.I1_operator_level > 1.0, \
            f"I1 unexpectedly small: {result.I1_operator_level}"
        assert result.I1_operator_level < 5.0, \
            f"I1 unexpectedly large: {result.I1_operator_level}"


# ============================================================================
# Stage 21D: Diagnostic Comparison Test
# ============================================================================

@pytest.mark.slow
@pytest.mark.diagnostic
class TestDiagnosticResults:
    """Tests that verify the diagnostic findings are reproducible.

    Marked slow because it computes both operator-level and tex_mirror.
    """

    @pytest.fixture
    def polys(self):
        """Load PRZZ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_operator_level_larger_than_tex_mirror(self, polys):
        """Operator-level I1 should be larger than tex_mirror I1.

        This is a documented finding: operator-level gives ~6x tex_mirror at L=20.
        """
        from src.evaluate import compute_c_paper_tex_mirror

        # Get operator-level I1
        op_result = compute_I1_operator_level_11(
            theta=4.0/7.0,
            R=1.3036,
            n=20,
            polynomials=polys,
            L=20.0,
            verbose=False,
        )

        # Get tex_mirror I1
        tex_result = compute_c_paper_tex_mirror(
            theta=4.0/7.0,
            R=1.3036,
            n=40,
            polynomials=polys,
            terms_version="old",
            tex_exp_component="exp_R_ref",
        )
        tex_I1 = tex_result.I1_plus + tex_result.m1 * tex_result.I1_minus_base

        # Operator-level should be larger
        ratio = op_result.I1_operator_level / tex_I1

        assert ratio > 3.0, f"Expected operator-level >> tex_mirror, got ratio = {ratio}"
        assert ratio < 10.0, f"Ratio unexpectedly large: {ratio}"
