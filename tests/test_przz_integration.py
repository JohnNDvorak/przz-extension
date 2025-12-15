"""
PRZZ Integration Tests - Phase 0, Step 7

Tests evaluate_c11() with real PRZZ polynomials from przz_parameters.json.

Key targets:
    R = 1.3036
    θ = 4/7 ≈ 0.5714285714285714
    c_target = 2.13745440613217263636
    κ_target = 0.417293962

Note: c₁₁ is only ONE of six pair contributions. The full c requires:
    c = c₁₁ + c₂₂ + c₃₃ + 2*c₁₂ + 2*c₁₃ + 2*c₂₃
So c₁₁ alone will NOT match c_target.
"""

import pytest
import numpy as np
from src.polynomials import load_przz_polynomials
from src.evaluate import evaluate_c11, convergence_sweep


# PRZZ parameters
THETA = 4.0 / 7.0
R = 1.3036


# =============================================================================
# Polynomial Loading Tests
# =============================================================================

class TestPRZZPolynomials:
    """Verify PRZZ polynomials load correctly."""

    def test_load_polynomials_success(self):
        """Polynomials load without error."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
        assert P1 is not None
        assert P2 is not None
        assert P3 is not None
        assert Q is not None

    def test_P1_boundary_conditions(self):
        """P1(0)=0, P1(1)=1."""
        P1, _, _, _ = load_przz_polynomials()
        assert abs(P1.eval(np.array([0.0]))[0]) < 1e-12
        assert abs(P1.eval(np.array([1.0]))[0] - 1.0) < 1e-12

    def test_P2_boundary_condition(self):
        """P2(0)=0."""
        _, P2, _, _ = load_przz_polynomials()
        assert abs(P2.eval(np.array([0.0]))[0]) < 1e-12

    def test_P3_boundary_condition(self):
        """P3(0)=0."""
        _, _, P3, _ = load_przz_polynomials()
        assert abs(P3.eval(np.array([0.0]))[0]) < 1e-12

    def test_Q_at_zero_paper_literal(self):
        """Q(0) with enforce_Q0=False matches paper (≈0.999999)."""
        _, _, _, Q = load_przz_polynomials(enforce_Q0=False)
        Q0 = Q.Q_at_zero()
        # Paper value: 0.999999
        assert abs(Q0 - 0.999999) < 1e-5

    def test_Q_at_zero_enforced(self):
        """Q(0) with enforce_Q0=True is exactly 1."""
        _, _, _, Q = load_przz_polynomials(enforce_Q0=True)
        Q0 = Q.Q_at_zero()
        assert abs(Q0 - 1.0) < 1e-14


# =============================================================================
# c₁₁ Evaluation Tests
# =============================================================================

class TestC11Evaluation:
    """Test c₁₁ computation with PRZZ polynomials."""

    @pytest.fixture
    def polynomials_literal(self):
        """PRZZ polynomials with paper-literal Q(0)."""
        P1, _, _, Q = load_przz_polynomials(enforce_Q0=False)
        return {"P1": P1, "Q": Q}

    @pytest.fixture
    def polynomials_enforced(self):
        """PRZZ polynomials with enforced Q(0)=1."""
        P1, _, _, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "Q": Q}

    def test_c11_evaluates_without_error(self, polynomials_literal):
        """c₁₁ computation runs successfully."""
        result = evaluate_c11(THETA, R, n=40, polynomials=polynomials_literal)
        assert np.isfinite(result.total)

    def test_c11_positive(self, polynomials_literal):
        """c₁₁ should be positive (main term in c)."""
        result = evaluate_c11(THETA, R, n=60, polynomials=polynomials_literal)
        # c₁₁ should be positive and contribute to c
        assert result.total > 0

    def test_c11_breakdown_complete(self, polynomials_literal):
        """Per-term breakdown has all four terms."""
        result = evaluate_c11(THETA, R, n=40, polynomials=polynomials_literal)
        assert "I1_11" in result.per_term
        assert "I2_11" in result.per_term
        assert "I3_11" in result.per_term
        assert "I4_11" in result.per_term

    def test_c11_breakdown_sums_correctly(self, polynomials_literal):
        """Sum of per-term values equals total."""
        result = evaluate_c11(THETA, R, n=60, polynomials=polynomials_literal)
        breakdown_sum = sum(result.per_term.values())
        assert abs(result.total - breakdown_sum) < 1e-12


# =============================================================================
# Quadrature Convergence Tests
# =============================================================================

class TestQuadratureConvergence:
    """Verify c₁₁ converges as n increases."""

    @pytest.fixture
    def polynomials(self):
        """PRZZ polynomials (enforced Q(0)=1 for consistency)."""
        P1, _, _, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "Q": Q}

    def test_convergence_small_errors(self, polynomials):
        """All errors vs reference should be small."""
        sweep = convergence_sweep(
            THETA, R, polynomials,
            ns=[40, 60, 80, 100],
            n_ref=150
        )

        # At n=100, error should be < 1e-8 (exponential GL convergence)
        assert sweep['errors'][100] < 1e-8
        # At n=80, error should still be quite small
        assert sweep['errors'][80] < 1e-6

    def test_convergence_all_errors_tiny(self, polynomials):
        """All errors should be tiny (< 1e-10) due to GL convergence."""
        sweep = convergence_sweep(
            THETA, R, polynomials,
            ns=[40, 60, 80, 100],
            n_ref=150
        )

        # All errors should be extremely small
        for n in [40, 60, 80, 100]:
            assert sweep['errors'][n] < 1e-10

    def test_n60_vs_n100_difference_small(self, polynomials):
        """c₁₁ at n=60 and n=100 should differ by < 1e-6."""
        result_60 = evaluate_c11(THETA, R, n=60, polynomials=polynomials)
        result_100 = evaluate_c11(THETA, R, n=100, polynomials=polynomials)

        diff = abs(result_60.total - result_100.total)
        assert diff < 1e-6


# =============================================================================
# Comparison: enforce_Q0=True vs False
# =============================================================================

class TestQ0EnforcementComparison:
    """Compare results with Q(0) enforcement on/off."""

    def test_c11_differs_with_Q0_mode(self):
        """c₁₁ values should differ slightly between modes."""
        P1_lit, _, _, Q_lit = load_przz_polynomials(enforce_Q0=False)
        P1_enf, _, _, Q_enf = load_przz_polynomials(enforce_Q0=True)

        polys_lit = {"P1": P1_lit, "Q": Q_lit}
        polys_enf = {"P1": P1_enf, "Q": Q_enf}

        c11_lit = evaluate_c11(THETA, R, n=80, polynomials=polys_lit)
        c11_enf = evaluate_c11(THETA, R, n=80, polynomials=polys_enf)

        # They should be slightly different (Q(0) = 0.999999 vs 1.0)
        # But not by much - the difference is in 6th decimal place of Q(0)
        rel_diff = abs(c11_lit.total - c11_enf.total) / abs(c11_enf.total)

        # Should differ by less than 0.1%
        assert rel_diff < 1e-3
        # But should not be exactly equal
        assert c11_lit.total != c11_enf.total


# =============================================================================
# α/β Separation Guard Tests
# =============================================================================

class TestAlphaBetaSeparation:
    """
    Guard tests to ensure Arg_α and Arg_β are never accidentally collapsed.

    CRITICAL: Q(Arg_α)·Q(Arg_β) ≠ Q(Arg)² because the affine expressions
    have SWAPPED coefficients for x and y variables.
    """

    def test_I1_alpha_beta_x1_coeffs_differ(self):
        """Arg_α and Arg_β have different x1 coefficients in I₁."""
        from src.terms_k3_d1 import make_I1_11

        term = make_I1_11(THETA, R)

        # Get the Q factors (should be at indices 2 and 3)
        Q_factors = [f for f in term.poly_factors if f.poly_name == "Q"]
        assert len(Q_factors) == 2, "I₁ should have exactly 2 Q factors"

        Q_alpha = Q_factors[0]
        Q_beta = Q_factors[1]

        # Evaluate the x1 coefficients at a test point
        U = np.array([[0.3]])
        T = np.array([[0.7]])

        alpha_x1 = Q_alpha.argument.var_coeffs["x1"](U, T)
        beta_x1 = Q_beta.argument.var_coeffs["x1"](U, T)

        # These MUST be different (θt vs θ(t-1))
        assert not np.allclose(alpha_x1, beta_x1), \
            "CRITICAL: α and β have same x1 coefficient - they've been collapsed!"

    def test_I1_alpha_beta_y1_coeffs_differ(self):
        """Arg_α and Arg_β have different y1 coefficients in I₁."""
        from src.terms_k3_d1 import make_I1_11

        term = make_I1_11(THETA, R)

        Q_factors = [f for f in term.poly_factors if f.poly_name == "Q"]
        Q_alpha = Q_factors[0]
        Q_beta = Q_factors[1]

        U = np.array([[0.3]])
        T = np.array([[0.7]])

        alpha_y1 = Q_alpha.argument.var_coeffs["y1"](U, T)
        beta_y1 = Q_beta.argument.var_coeffs["y1"](U, T)

        # These MUST be different (θ(t-1) vs θt)
        assert not np.allclose(alpha_y1, beta_y1), \
            "CRITICAL: α and β have same y1 coefficient - they've been collapsed!"

    def test_I3_alpha_beta_x1_coeffs_differ(self):
        """Arg_α and Arg_β have different x1 coefficients in I₃."""
        from src.terms_k3_d1 import make_I3_11

        term = make_I3_11(THETA, R)

        Q_factors = [f for f in term.poly_factors if f.poly_name == "Q"]
        assert len(Q_factors) == 2

        Q_alpha = Q_factors[0]
        Q_beta = Q_factors[1]

        U = np.array([[0.3]])
        T = np.array([[0.7]])

        alpha_x1 = Q_alpha.argument.var_coeffs["x1"](U, T)
        beta_x1 = Q_beta.argument.var_coeffs["x1"](U, T)

        assert not np.allclose(alpha_x1, beta_x1), \
            "CRITICAL: I₃ α and β have same x1 coefficient!"

    def test_I4_alpha_beta_y1_coeffs_differ(self):
        """Arg_α and Arg_β have different y1 coefficients in I₄."""
        from src.terms_k3_d1 import make_I4_11

        term = make_I4_11(THETA, R)

        Q_factors = [f for f in term.poly_factors if f.poly_name == "Q"]
        assert len(Q_factors) == 2

        Q_alpha = Q_factors[0]
        Q_beta = Q_factors[1]

        U = np.array([[0.3]])
        T = np.array([[0.7]])

        alpha_y1 = Q_alpha.argument.var_coeffs["y1"](U, T)
        beta_y1 = Q_beta.argument.var_coeffs["y1"](U, T)

        assert not np.allclose(alpha_y1, beta_y1), \
            "CRITICAL: I₄ α and β have same y1 coefficient!"


# =============================================================================
# Logging / Reporting (run with -v to see output)
# =============================================================================

class TestC11Reporting:
    """Tests that print c₁₁ values for manual inspection."""

    def test_print_c11_values(self, capsys):
        """Print c₁₁ at various n for inspection."""
        P1, _, _, Q = load_przz_polynomials(enforce_Q0=True)
        polynomials = {"P1": P1, "Q": Q}

        print("\n" + "="*60)
        print("PRZZ Integration Test: c₁₁ Values")
        print("="*60)
        print(f"θ = {THETA:.10f}")
        print(f"R = {R}")
        print("-"*60)

        for n in [40, 60, 80, 100, 120]:
            result = evaluate_c11(THETA, R, n=n, polynomials=polynomials)
            print(f"n={n:3d}: c₁₁ = {result.total:.12f}")

        print("-"*60)

        # Print per-term breakdown at n=100
        result_100 = evaluate_c11(THETA, R, n=100, polynomials=polynomials)
        print("\nPer-term breakdown (n=100):")
        for name, value in sorted(result_100.per_term.items()):
            print(f"  {name}: {value:+.12f}")
        print(f"  Total: {result_100.total:.12f}")
        print("="*60)

        # Capture output for assertion
        captured = capsys.readouterr()
        assert "c₁₁" in captured.out

    def test_print_convergence_sweep(self, capsys):
        """Print convergence sweep results."""
        P1, _, _, Q = load_przz_polynomials(enforce_Q0=True)
        polynomials = {"P1": P1, "Q": Q}

        sweep = convergence_sweep(
            THETA, R, polynomials,
            ns=[40, 60, 80, 100],
            n_ref=150
        )

        print("\n" + "="*60)
        print("Quadrature Convergence Sweep")
        print("="*60)
        print(f"Reference (n={sweep['n_ref']}): c₁₁ = {sweep['reference']:.12f}")
        print("-"*60)
        print(f"{'n':>5s}  {'c₁₁':>18s}  {'error':>12s}")
        print("-"*60)
        for n in sorted(sweep['values'].keys()):
            print(f"{n:5d}  {sweep['values'][n]:18.12f}  {sweep['errors'][n]:12.2e}")
        print("="*60)

        captured = capsys.readouterr()
        assert "Convergence" in captured.out
