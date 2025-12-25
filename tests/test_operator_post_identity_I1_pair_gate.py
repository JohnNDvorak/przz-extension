"""
tests/test_operator_post_identity_I1_pair_gate.py
Case-C pair validation: Post-identity operator I1 vs DSL I1 for pairs beyond (1,1).

This is the SKEPTIC-PROOF validation that the post-identity operator approach
generalizes to Case-C pairs (ℓ ∈ {2, 3}), proving it's not a P₁-only coincidence.

GPT Guidance Phase 2, Step 5:
- Test (1,2), (1,3), (2,2), (2,3), (3,3) pairs
- Initially use 20% tolerance, tighten to strict equality once working
- Validates that the Case-C kernel handling (via case_c_taylor_coeffs) is correct

Key insight:
- (1,1) uses Case B (ω=0) for both P factors
- (1,2), (1,3) use Case B + Case C
- (2,2), (2,3), (3,3) use Case C + Case C
"""

import pytest
import numpy as np
from src.operator_post_identity import (
    compute_I1_operator_post_identity_pair,
)
from src.polynomials import load_przz_polynomials
from src.terms_k3_d1 import (
    make_all_terms_11_v2,
    make_all_terms_12_v2,
    make_all_terms_13_v2,
    make_all_terms_22_v2,
    make_all_terms_23_v2,
    make_all_terms_33_v2,
)
from src.evaluate import evaluate_terms


# Tolerance for strict equality
# Machine-level equality: use rtol=1e-10, atol=1e-12
STRICT_RTOL = 1e-10
STRICT_ATOL = 1e-12


def get_DSL_I1_pair(
    theta: float, R: float, ell1: int, ell2: int, n: int, polynomials: dict
) -> float:
    """
    Compute I1(ℓ₁,ℓ₂) using the DSL V2 term builder.

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        ell1, ell2: Pair indices
        n: Number of quadrature points
        polynomials: Dict with P1, P2, P3, Q

    Returns:
        I1(ℓ₁,ℓ₂) value from DSL evaluation
    """
    regime = "paper"

    # Map pair to term builder
    terms_builders = {
        (1, 1): make_all_terms_11_v2,
        (1, 2): make_all_terms_12_v2,
        (1, 3): make_all_terms_13_v2,
        (2, 2): make_all_terms_22_v2,
        (2, 3): make_all_terms_23_v2,
        (3, 3): make_all_terms_33_v2,
    }

    terms_fn = terms_builders.get((ell1, ell2))
    if terms_fn is None:
        raise ValueError(f"No term builder for pair ({ell1}, {ell2})")

    terms = terms_fn(theta, R, kernel_regime=regime)
    i1_terms = [t for t in terms if "I1" in t.name]

    if not i1_terms:
        raise ValueError(f"No I1 terms found for pair ({ell1}, {ell2})")

    result = evaluate_terms(
        i1_terms,
        polynomials,
        n,
        return_breakdown=True,
        R=R,
        theta=theta,
    )

    return result.total


# =============================================================================
# Test Class: Case-C Pair DSL Comparison - STRICT EQUALITY
# =============================================================================

class TestI1CaseCPairDSLComparison:
    """
    Compare post-identity operator I1 to DSL I1 for Case-C pairs.

    The post-identity approach should match the DSL EXACTLY (within machine precision)
    because both use the same (θt-θ) affine structure and the same series algebra.

    This proves the operator approach generalizes beyond P₁-only (Case B).
    """

    @pytest.fixture
    def polys(self):
        """Load PRZZ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    @pytest.mark.slow
    def test_I1_12_matches_DSL_strict(self, polys, theta):
        """
        Post-identity I1(1,2) should EXACTLY match DSL I1(1,2).

        This is a Case B + Case C pair: P₁ uses Case B, P₂ uses Case C (ω=1).
        """
        R = 1.3036
        n = 40

        # Post-identity operator
        result_post = compute_I1_operator_post_identity_pair(
            theta, R, ell1=1, ell2=2, n=n, polynomials=polys
        )
        I1_post = result_post.I1_value

        # DSL I1(1,2)
        I1_dsl = get_DSL_I1_pair(theta, R, 1, 2, n, polys)

        # STRICT EQUALITY assertion
        assert np.isclose(I1_post, I1_dsl, rtol=STRICT_RTOL, atol=STRICT_ATOL), \
            f"STRICT MISMATCH (1,2): post={I1_post:.12f}, dsl={I1_dsl:.12f}, " \
            f"diff={abs(I1_post - I1_dsl):.2e}"

    @pytest.mark.slow
    def test_I1_13_matches_DSL_strict(self, polys, theta):
        """
        Post-identity I1(1,3) should EXACTLY match DSL I1(1,3).

        This is a Case B + Case C pair: P₁ uses Case B, P₃ uses Case C (ω=2).
        """
        R = 1.3036
        n = 40

        # Post-identity operator
        result_post = compute_I1_operator_post_identity_pair(
            theta, R, ell1=1, ell2=3, n=n, polynomials=polys
        )
        I1_post = result_post.I1_value

        # DSL I1(1,3)
        I1_dsl = get_DSL_I1_pair(theta, R, 1, 3, n, polys)

        # STRICT EQUALITY assertion
        assert np.isclose(I1_post, I1_dsl, rtol=STRICT_RTOL, atol=STRICT_ATOL), \
            f"STRICT MISMATCH (1,3): post={I1_post:.12f}, dsl={I1_dsl:.12f}, " \
            f"diff={abs(I1_post - I1_dsl):.2e}"

    @pytest.mark.slow
    def test_I1_22_matches_DSL_strict(self, polys, theta):
        """
        Post-identity I1(2,2) should EXACTLY match DSL I1(2,2).

        This is a Case C + Case C pair: both P₂ factors use Case C (ω=1).
        """
        R = 1.3036
        n = 40

        # Post-identity operator
        result_post = compute_I1_operator_post_identity_pair(
            theta, R, ell1=2, ell2=2, n=n, polynomials=polys
        )
        I1_post = result_post.I1_value

        # DSL I1(2,2)
        I1_dsl = get_DSL_I1_pair(theta, R, 2, 2, n, polys)

        # STRICT EQUALITY assertion
        assert np.isclose(I1_post, I1_dsl, rtol=STRICT_RTOL, atol=STRICT_ATOL), \
            f"STRICT MISMATCH (2,2): post={I1_post:.12f}, dsl={I1_dsl:.12f}, " \
            f"diff={abs(I1_post - I1_dsl):.2e}"

    @pytest.mark.slow
    def test_I1_23_matches_DSL_strict(self, polys, theta):
        """
        Post-identity I1(2,3) should EXACTLY match DSL I1(2,3).

        This is a Case C + Case C pair: P₂ uses ω=1, P₃ uses ω=2.
        """
        R = 1.3036
        n = 40

        # Post-identity operator
        result_post = compute_I1_operator_post_identity_pair(
            theta, R, ell1=2, ell2=3, n=n, polynomials=polys
        )
        I1_post = result_post.I1_value

        # DSL I1(2,3)
        I1_dsl = get_DSL_I1_pair(theta, R, 2, 3, n, polys)

        # STRICT EQUALITY assertion
        assert np.isclose(I1_post, I1_dsl, rtol=STRICT_RTOL, atol=STRICT_ATOL), \
            f"STRICT MISMATCH (2,3): post={I1_post:.12f}, dsl={I1_dsl:.12f}, " \
            f"diff={abs(I1_post - I1_dsl):.2e}"

    @pytest.mark.slow
    def test_I1_33_matches_DSL_strict(self, polys, theta):
        """
        Post-identity I1(3,3) should EXACTLY match DSL I1(3,3).

        This is a Case C + Case C pair: both P₃ factors use Case C (ω=2).
        """
        R = 1.3036
        n = 40

        # Post-identity operator
        result_post = compute_I1_operator_post_identity_pair(
            theta, R, ell1=3, ell2=3, n=n, polynomials=polys
        )
        I1_post = result_post.I1_value

        # DSL I1(3,3)
        I1_dsl = get_DSL_I1_pair(theta, R, 3, 3, n, polys)

        # STRICT EQUALITY assertion
        assert np.isclose(I1_post, I1_dsl, rtol=STRICT_RTOL, atol=STRICT_ATOL), \
            f"STRICT MISMATCH (3,3): post={I1_post:.12f}, dsl={I1_dsl:.12f}, " \
            f"diff={abs(I1_post - I1_dsl):.2e}"

    @pytest.mark.slow
    def test_I1_11_matches_DSL_strict(self, polys, theta):
        """
        Post-identity I1(1,1) should EXACTLY match DSL I1(1,1) via pair function.

        This is a Case B + Case B pair: both P₁ factors use Case B (ω=0).
        Validates that the pair function correctly handles the (1,1) special case.
        """
        R = 1.3036
        n = 40

        # Post-identity operator via pair function
        result_post = compute_I1_operator_post_identity_pair(
            theta, R, ell1=1, ell2=1, n=n, polynomials=polys
        )
        I1_post = result_post.I1_value

        # DSL I1(1,1)
        I1_dsl = get_DSL_I1_pair(theta, R, 1, 1, n, polys)

        # STRICT EQUALITY assertion
        assert np.isclose(I1_post, I1_dsl, rtol=STRICT_RTOL, atol=STRICT_ATOL), \
            f"STRICT MISMATCH (1,1): post={I1_post:.12f}, dsl={I1_dsl:.12f}, " \
            f"diff={abs(I1_post - I1_dsl):.2e}"


# =============================================================================
# Test Class: Kappa-Star Benchmark (R=1.1167)
# =============================================================================

class TestI1CaseCPairKappaStar:
    """
    Test Case-C pairs at the κ* benchmark (R=1.1167).

    This validates that the post-identity approach works across different R values.
    """

    @pytest.fixture
    def polys(self):
        """Load PRZZ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    @pytest.mark.slow
    @pytest.mark.parametrize("ell1,ell2", [(1, 2), (2, 2), (3, 3)])
    def test_I1_matches_DSL_kappa_star(self, ell1, ell2, polys, theta):
        """
        Post-identity I1 should match DSL I1 at κ* benchmark.
        """
        R = 1.1167
        n = 40

        # Post-identity operator
        result_post = compute_I1_operator_post_identity_pair(
            theta, R, ell1=ell1, ell2=ell2, n=n, polynomials=polys
        )
        I1_post = result_post.I1_value

        # DSL I1
        I1_dsl = get_DSL_I1_pair(theta, R, ell1, ell2, n, polys)

        # STRICT EQUALITY assertion
        assert np.isclose(I1_post, I1_dsl, rtol=STRICT_RTOL, atol=STRICT_ATOL), \
            f"STRICT MISMATCH ({ell1},{ell2}) at R={R}: " \
            f"post={I1_post:.12f}, dsl={I1_dsl:.12f}, diff={abs(I1_post - I1_dsl):.2e}"


# =============================================================================
# Test Class: Structural Properties
# =============================================================================

class TestI1PairStructure:
    """
    Verify that the pair function has correct structural properties.
    """

    @pytest.fixture
    def polys(self):
        """Load PRZZ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    def test_power_formula_11(self, polys, theta):
        """(1,1) should use power=2 (special case)."""
        R = 1.3036
        n = 20

        result = compute_I1_operator_post_identity_pair(
            theta, R, ell1=1, ell2=1, n=n, polynomials=polys
        )

        assert result.details["one_minus_u_power"] == 2

    def test_power_formula_12(self, polys, theta):
        """(1,2) should use power=max(0, (1-1)+(2-1))=1."""
        R = 1.3036
        n = 20

        result = compute_I1_operator_post_identity_pair(
            theta, R, ell1=1, ell2=2, n=n, polynomials=polys
        )

        assert result.details["one_minus_u_power"] == 1

    def test_power_formula_22(self, polys, theta):
        """(2,2) should use power=max(0, (2-1)+(2-1))=2."""
        R = 1.3036
        n = 20

        result = compute_I1_operator_post_identity_pair(
            theta, R, ell1=2, ell2=2, n=n, polynomials=polys
        )

        assert result.details["one_minus_u_power"] == 2

    def test_power_formula_33(self, polys, theta):
        """(3,3) should use power=max(0, (3-1)+(3-1))=4."""
        R = 1.3036
        n = 20

        result = compute_I1_operator_post_identity_pair(
            theta, R, ell1=3, ell2=3, n=n, polynomials=polys
        )

        assert result.details["one_minus_u_power"] == 4

    def test_omega_values(self, polys, theta):
        """Omega values should be ℓ-1."""
        R = 1.3036
        n = 20

        for ell1, ell2 in [(1, 2), (2, 3), (3, 3)]:
            result = compute_I1_operator_post_identity_pair(
                theta, R, ell1=ell1, ell2=ell2, n=n, polynomials=polys
            )

            assert result.details["omega1"] == ell1 - 1
            assert result.details["omega2"] == ell2 - 1

    @pytest.mark.parametrize("ell1,ell2,expected_sign", [
        (1, 1, +1),  # (-1)^2 = +1
        (1, 2, -1),  # (-1)^3 = -1
        (1, 3, +1),  # (-1)^4 = +1
        (2, 2, +1),  # (-1)^4 = +1
        (2, 3, -1),  # (-1)^5 = -1
        (3, 3, +1),  # (-1)^6 = +1
    ])
    def test_sign_factor(self, ell1, ell2, expected_sign, polys, theta):
        """Sign factor should be (-1)^{ℓ₁+ℓ₂}."""
        R = 1.3036
        n = 20

        result = compute_I1_operator_post_identity_pair(
            theta, R, ell1=ell1, ell2=ell2, n=n, polynomials=polys
        )

        assert result.details["sign_factor"] == expected_sign

    def test_I1_finite_for_all_pairs(self, polys, theta):
        """I1 should be finite for all K=3 pairs."""
        R = 1.3036
        n = 20

        for ell1 in [1, 2, 3]:
            for ell2 in range(ell1, 4):  # ell2 >= ell1
                result = compute_I1_operator_post_identity_pair(
                    theta, R, ell1=ell1, ell2=ell2, n=n, polynomials=polys
                )

                assert np.isfinite(result.I1_value), \
                    f"I1({ell1},{ell2}) not finite: {result.I1_value}"
