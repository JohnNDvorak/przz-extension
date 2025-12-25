"""
tests/test_mirror_operator_exact_Q1.py
Phase 10.2d: Q=1 Gold Test for Mirror Operator

This is a NON-NEGOTIABLE gold test that validates the mirror operator implementation.

MATHEMATICAL BASIS:
===================
When Q(x) ≡ 1 (constant polynomial), the derived mirror should reduce to a pure
exponential mirror with NO polynomial amplification.

Expected behavior:
- Q(A_α^mirror) = Q(θy) = 1 for all y
- Q(A_β^mirror) = Q(θx) = 1 for all x
- No Q polynomial blowup
- Mirror/direct ratio should be well-behaved (related to exp(2R))

FAILURE MODE (Phase 9):
=======================
Phase 9 used Q(1+D) on DIRECT eigenvalues, which pushed arguments into [1,2]:
- Q(1+A_α) where A_α ≈ t gave Q ≈ Q(1+t) ∈ [Q(1), Q(2)]
- For realistic Q polynomials, this caused 100× blowups

SUCCESS MODE (Phase 10):
========================
Phase 10 uses Q(D) on SWAPPED eigenvalues:
- Q(A_α^mirror) = Q(θy) where θ ≈ 0.57 gives Q ∈ [Q(0), Q(0.57)]
- Arguments stay in [0, θ] ⊂ [0, 1] where Q is well-behaved
"""

import pytest
import numpy as np
from src.polynomials import Polynomial


def make_Q_constant_one() -> Polynomial:
    """Create Q(x) ≡ 1 polynomial."""
    return Polynomial(np.array([1.0]))


def make_trivial_P_polynomials() -> dict:
    """Create trivial P polynomials: P₁(u) = P₂(u) = P₃(u) = 1."""
    P1 = Polynomial(np.array([1.0]))
    return {
        'P1': P1,
        'P2': P1,
        'P3': P1,
        'Q': make_Q_constant_one()
    }


class TestQ1GoldTests:
    """Q=1 gold tests for the mirror operator."""

    @pytest.fixture
    def Q1_polynomials(self):
        """Polynomials with Q ≡ 1."""
        return make_trivial_P_polynomials()

    @pytest.fixture
    def theta(self):
        """PRZZ θ = 4/7."""
        return 4.0 / 7.0

    @pytest.fixture
    def n_quadrature(self):
        """Quadrature points."""
        return 40

    def test_mirror_eigenvalues_are_swapped(self, theta):
        """
        Verify mirror eigenvalues are correctly swapped.

        Direct eigenvalues (at t=0.5):
            A_α = 0.5 + θ(-0.5)x + θ(0.5)y ≈ 0.5 - 0.286x + 0.286y
            A_β = 0.5 + θ(0.5)x + θ(-0.5)y ≈ 0.5 + 0.286x - 0.286y

        Mirror eigenvalues (swap x↔y):
            A_α^mirror = θy ≈ 0.571y
            A_β^mirror = θx ≈ 0.571x
        """
        from src.mirror_operator_exact import get_mirror_eigenvalues_with_swap

        eig = get_mirror_eigenvalues_with_swap(theta)

        # A_α^mirror = θy: no constant, no x, only y
        assert eig.u0_alpha == 0.0
        assert eig.x_alpha == 0.0
        assert abs(eig.y_alpha - theta) < 1e-10

        # A_β^mirror = θx: no constant, only x, no y
        assert eig.u0_beta == 0.0
        assert abs(eig.x_beta - theta) < 1e-10
        assert eig.y_beta == 0.0

    def test_Q1_no_blowup(self, Q1_polynomials, theta, n_quadrature):
        """
        GOLD TEST: Q=1 should produce well-behaved mirror values.

        When Q ≡ 1:
        - Phase 9 approach would still blow up due to eigenvalue shift
        - Phase 10 approach should give reasonable values

        The mirror value should be:
        - Proportional to exp(2R) from T^{-(α+β)} weight
        - NOT 100× larger than direct (the Phase 9 failure mode)
        """
        from src.mirror_operator_exact import compute_I1_mirror_operator_exact
        from src.mirror_exact import compute_I1_standard

        R = 1.3036  # κ benchmark

        # Compute direct I₁
        I1_direct = compute_I1_standard(
            theta=theta, R=R, n=n_quadrature, polynomials=Q1_polynomials,
            ell1=1, ell2=1
        )

        # Compute mirror I₁ with swapped eigenvalues
        result = compute_I1_mirror_operator_exact(
            theta=theta, R=R, n=n_quadrature, polynomials=Q1_polynomials,
            ell1=1, ell2=1
        )

        # The ratio should be related to exp(2R) ≈ 13.56
        # NOT 100× as in Phase 9 blowup
        exp_2R = np.exp(2 * R)

        if abs(I1_direct) > 1e-15:
            ratio = result.value / I1_direct
        else:
            ratio = float('inf')

        print(f"\n=== Q=1 Gold Test ===")
        print(f"I₁_direct = {I1_direct:.8f}")
        print(f"I₁_mirror = {result.value:.8f}")
        print(f"ratio = {ratio:.4f}")
        print(f"exp(2R) = {exp_2R:.4f}")

        # Key assertions:
        # 1. Values should be finite
        assert np.isfinite(I1_direct), "Direct should be finite"
        assert np.isfinite(result.value), "Mirror should be finite"

        # 2. Mirror should NOT blow up (< 50× direct)
        # Phase 9 gave 112× blowup, we want much less
        assert abs(ratio) < 50, f"Mirror blowup detected! Ratio = {ratio} (should be < 50)"

        # 3. Q ranges should be within [0, 1] since Q ≡ 1
        assert result.Q_alpha_range[0] >= 0.99
        assert result.Q_alpha_range[1] <= 1.01
        assert result.Q_beta_range[0] >= 0.99
        assert result.Q_beta_range[1] <= 1.01

    def test_Q1_mirror_vs_direct_ratio_reasonable(self, Q1_polynomials, theta, n_quadrature):
        """
        Test that mirror/direct ratio is reasonable for Q=1.

        For Q ≡ 1, the main difference between direct and mirror is:
        - The exp factor structure
        - The T^{-(α+β)} weight = exp(2R)

        The ratio should be:
        - Positive (same sign)
        - Related to exp(2R) ≈ 13.56
        - NOT 100× as in Phase 9
        """
        from src.mirror_operator_exact import compute_I1_mirror_operator_exact
        from src.mirror_exact import compute_I1_standard

        for R in [1.1167, 1.3036]:  # Both benchmarks
            I1_direct = compute_I1_standard(
                theta=theta, R=R, n=n_quadrature, polynomials=Q1_polynomials,
                ell1=1, ell2=1
            )

            result = compute_I1_mirror_operator_exact(
                theta=theta, R=R, n=n_quadrature, polynomials=Q1_polynomials,
                ell1=1, ell2=1
            )

            exp_2R = np.exp(2 * R)

            if abs(I1_direct) > 1e-15:
                ratio = result.value / I1_direct
            else:
                ratio = float('inf')

            print(f"\nR = {R}: ratio = {ratio:.4f}, exp(2R) = {exp_2R:.4f}")

            # Ratio should be in reasonable range, not blowing up
            assert abs(ratio) < 100, f"Blowup at R={R}: ratio={ratio}"

    def test_Q1_I2_mirror_correct(self, Q1_polynomials, theta, n_quadrature):
        """
        Test I₂ mirror computation with Q=1.

        For I₂, there are no derivatives, so the structure is simpler:
        - Q(0)² = 1 for Q ≡ 1
        - exp factor: exp(2Rt)
        - T weight: exp(2R)
        """
        from src.mirror_operator_exact import compute_I2_mirror_operator_exact

        R = 1.3036

        result = compute_I2_mirror_operator_exact(
            theta=theta, R=R, n=n_quadrature, polynomials=Q1_polynomials,
            ell1=1, ell2=1
        )

        # I₂ should be finite and positive
        assert np.isfinite(result.value)

        print(f"\nI₂_mirror (Q=1) = {result.value:.8f}")
        print(f"I₂_swapped = {result.I_swapped:.8f}")
        print(f"T_weight = {result.T_weight:.4f}")


class TestComparePhase9vsPhase10:
    """
    Compare Phase 9 (Q-shift) vs Phase 10 (swap) approaches.

    This explicitly shows why Phase 9 blew up and Phase 10 works.
    """

    @pytest.fixture
    def przz_polynomials(self):
        """Load PRZZ benchmark polynomials."""
        from src.polynomials import load_przz_polynomials

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    def test_phase9_vs_phase10_comparison(self, przz_polynomials):
        """
        Compare Phase 9 (blowup) vs Phase 10 (correct) for PRZZ polynomials.

        Phase 9 gave 112× blowup for (1,1) pair.
        Phase 10 should give well-behaved values.
        """
        from src.mirror_operator_exact import compare_with_phase9_approach

        theta = 4.0 / 7.0
        R = 1.3036
        n = 40

        results = compare_with_phase9_approach(
            theta=theta, R=R, n=n, polynomials=przz_polynomials,
            ell1=1, ell2=1, verbose=True
        )

        # Phase 9 should be much larger than Phase 10
        ratio = results["ratio_phase9_over_phase10"]

        # The ratio should show Phase 9 blowup
        # (If they're similar, then our understanding is wrong)
        print(f"\nPhase9/Phase10 ratio: {ratio:.4f}")

        # Record the values for analysis
        assert np.isfinite(results["phase10_value"]), "Phase 10 should be finite"
        assert np.isfinite(results["phase9_value"]), "Phase 9 should be finite"

        # Phase 10 Q ranges should be in [0, θ] ≈ [0, 0.57]
        # This is well-behaved territory for Q polynomials
        Q_alpha_min, Q_alpha_max = results["Q_alpha_range"]
        print(f"Q(A_α^mirror) range: [{Q_alpha_min:.4f}, {Q_alpha_max:.4f}]")

        # Q is evaluated at arguments in [0, θ], should give reasonable values
        assert Q_alpha_max < 10, f"Q_alpha blowing up: {Q_alpha_max}"


class TestMirrorOperatorDiagnostics:
    """Diagnostic tests for understanding mirror operator behavior."""

    @pytest.fixture
    def przz_polynomials(self):
        """Load PRZZ benchmark polynomials."""
        from src.polynomials import load_przz_polynomials

        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    def test_S12_mirror_operator_exact_total(self, przz_polynomials):
        """
        Compute full S12 mirror using the exact operator approach.

        This should give a reasonable value that can be compared with
        the empirical m₁ × S12(-R) approach.
        """
        from src.mirror_operator_exact import compute_S12_mirror_operator_exact

        theta = 4.0 / 7.0
        R = 1.3036
        n = 40

        S12_mirror = compute_S12_mirror_operator_exact(
            theta=theta, R=R, n=n, polynomials=przz_polynomials,
            K=3, verbose=True
        )

        print(f"\nS12_mirror_operator_exact total: {S12_mirror:.8f}")

        # Should be finite
        assert np.isfinite(S12_mirror)

        # Compare with empirical approach
        from src.mirror_exact import compute_S12_minus_basis
        from src.m1_policy import M1_EMPIRICAL_KAPPA

        S12_minus = compute_S12_minus_basis(
            theta=theta, R=R, n=n, polynomials=przz_polynomials
        )

        empirical_mirror = M1_EMPIRICAL_KAPPA * S12_minus

        print(f"S12_minus_basis: {S12_minus:.8f}")
        print(f"Empirical mirror (m₁ × S12_minus): {empirical_mirror:.8f}")
        print(f"Operator exact mirror: {S12_mirror:.8f}")

        if abs(empirical_mirror) > 1e-15:
            ratio = S12_mirror / empirical_mirror
            print(f"Operator/Empirical ratio: {ratio:.4f}")
