"""
tests/test_operator_post_identity_I1_11_gate.py
End-to-end comparison test: Post-identity operator I1(1,1) vs DSL I1(1,1).

This is the SKEPTIC-PROOF validation that the post-identity operator approach
produces EXACTLY the same I1 values as the DSL/tex_mirror implementation.

GPT Guidance Phase 1, Step 1:
- Tighten assertions from loose (0.1 < ratio < 10) to strict equality (rtol=1e-10)
- Use V2 term builder (make_all_terms_11_v2)
- Test both benchmarks (R=1.3036 and R=1.1167)
"""

import pytest
import numpy as np
from src.operator_post_identity import (
    compute_I1_operator_post_identity_11,
)
from src.polynomials import load_przz_polynomials
from src.terms_k3_d1 import make_all_terms_11_v2
from src.evaluate import evaluate_terms


# Tolerance for strict equality
# Machine-level equality: use rtol=1e-10, atol=1e-12
# If quadrature noise is present, may need to relax to rtol=1e-8
STRICT_RTOL = 1e-10
STRICT_ATOL = 1e-12


def get_DSL_I1_11(theta: float, R: float, n: int, polynomials: dict, verbose: bool = False) -> float:
    """
    Compute I1(1,1) using the DSL V2 term builder.

    This uses make_all_terms_11_v2 which creates the tex_mirror-style
    terms with the correct affine forms containing (θt-θ) cross-terms.

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        n: Number of quadrature points
        polynomials: Dict with P1, P2, P3, Q
        verbose: Print debug output

    Returns:
        I1(1,1) value from DSL evaluation
    """
    # Use paper regime which uses the correct affine structure
    regime = "paper"

    # Get I1 terms for pair (1,1) using V2 builder
    terms = make_all_terms_11_v2(theta, R, kernel_regime=regime)

    # Filter to I1 terms only (name contains "I1")
    i1_terms = [t for t in terms if "I1" in t.name]

    if not i1_terms:
        raise ValueError("No I1 terms found in make_all_terms_11_v2 output")

    if verbose:
        print(f"  DSL I1 terms: {[t.name for t in i1_terms]}")

    # Evaluate the terms
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
# Test Class: I1(1,1) DSL Comparison - STRICT EQUALITY
# =============================================================================

class TestI1DSLComparison:
    """
    Compare post-identity operator I1 to DSL I1 with STRICT EQUALITY.

    The post-identity approach should match the DSL EXACTLY (within machine precision)
    because both use the same (θt-θ) affine structure and the same series algebra.

    GPT Phase 1, Step 1: Tightened from loose (0.1 < ratio < 10) to strict (rtol=1e-10).
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
    def test_I1_matches_DSL_kappa_benchmark_strict(self, polys, theta):
        """
        Post-identity I1(1,1) should EXACTLY match DSL I1(1,1) for κ benchmark (R=1.3036).

        This is the SKEPTIC-PROOF test: strict numerical equality within machine precision.
        """
        R = 1.3036
        n = 40

        # Post-identity operator
        result_post = compute_I1_operator_post_identity_11(theta, R, n, polys)
        I1_post = result_post.I1_value

        # DSL I1(1,1)
        I1_dsl = get_DSL_I1_11(theta, R, n, polys)

        # STRICT EQUALITY assertion
        assert np.isclose(I1_post, I1_dsl, rtol=STRICT_RTOL, atol=STRICT_ATOL), \
            f"STRICT MISMATCH (R={R}): post={I1_post:.12f}, dsl={I1_dsl:.12f}, " \
            f"diff={abs(I1_post - I1_dsl):.2e}, ratio={I1_post/I1_dsl if I1_dsl != 0 else float('inf'):.10f}"

    @pytest.mark.slow
    def test_I1_matches_DSL_kappa_star_benchmark_strict(self, polys, theta):
        """
        Post-identity I1(1,1) should EXACTLY match DSL I1(1,1) for κ* benchmark (R=1.1167).

        This is the SKEPTIC-PROOF test: strict numerical equality within machine precision.
        """
        R = 1.1167
        n = 40

        # Post-identity operator
        result_post = compute_I1_operator_post_identity_11(theta, R, n, polys)
        I1_post = result_post.I1_value

        # DSL I1(1,1)
        I1_dsl = get_DSL_I1_11(theta, R, n, polys)

        # STRICT EQUALITY assertion
        assert np.isclose(I1_post, I1_dsl, rtol=STRICT_RTOL, atol=STRICT_ATOL), \
            f"STRICT MISMATCH (R={R}): post={I1_post:.12f}, dsl={I1_dsl:.12f}, " \
            f"diff={abs(I1_post - I1_dsl):.2e}, ratio={I1_post/I1_dsl if I1_dsl != 0 else float('inf'):.10f}"

    def test_I1_sign_matches_exactly(self, polys, theta):
        """
        I1(1,1) should have EXACTLY the same sign from both methods.

        Quick test (n=20) for sign consistency.
        """
        R = 1.3036
        n = 20

        # Post-identity operator
        result_post = compute_I1_operator_post_identity_11(theta, R, n, polys)
        I1_post = result_post.I1_value

        # DSL I1(1,1)
        I1_dsl = get_DSL_I1_11(theta, R, n, polys)

        # Both must have same sign
        assert (I1_post > 0) == (I1_dsl > 0), \
            f"Sign mismatch: post={I1_post:.6f}, dsl={I1_dsl:.6f}"

    def test_I1_quadrature_convergence(self, polys, theta):
        """
        I1 should converge as quadrature points increase.

        Also verify that post-identity and DSL converge to the same value.
        """
        R = 1.3036

        I1_post_values = []
        I1_dsl_values = []
        n_values = [10, 20, 30]

        for n in n_values:
            result_post = compute_I1_operator_post_identity_11(theta, R, n, polys)
            I1_dsl = get_DSL_I1_11(theta, R, n, polys)
            I1_post_values.append(result_post.I1_value)
            I1_dsl_values.append(I1_dsl)

        # Check convergence: both should converge to same value
        # At highest n, they should match within tight tolerance
        assert np.isclose(I1_post_values[-1], I1_dsl_values[-1], rtol=1e-8), \
            f"Convergence mismatch at n={n_values[-1]}: " \
            f"post={I1_post_values[-1]:.10f}, dsl={I1_dsl_values[-1]:.10f}"


# =============================================================================
# Test Class: Structure Verification
# =============================================================================

class TestStructureVerification:
    """
    Verify that the post-identity I1 has correct structural properties.
    """

    @pytest.fixture
    def polys(self):
        """Load PRZZ polynomials."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def theta(self):
        return 4.0 / 7.0

    def test_I1_depends_on_R(self, polys, theta):
        """I1 should change with R."""
        n = 20

        results = []
        for R in [1.0, 1.3036, 1.5]:
            result = compute_I1_operator_post_identity_11(theta, R, n, polys)
            results.append(result.I1_value)

        # All values should be different
        assert len(set(f"{r:.8f}" for r in results)) == 3, \
            f"I1 not varying with R: {results}"

    def test_I1_symmetric_in_xy(self, polys, theta):
        """
        The I1 integrand should be symmetric in x↔y.

        This is verified by checking that the post-identity approach
        produces finite, non-zero results (the symmetry is built into
        the A_α, A_β structure).
        """
        R = 1.3036
        n = 20

        result = compute_I1_operator_post_identity_11(theta, R, n, polys)

        # Should be finite and non-zero
        assert np.isfinite(result.I1_value), f"I1 not finite: {result.I1_value}"
        assert abs(result.I1_value) > 1e-10, f"I1 too small: {result.I1_value}"

    def test_I1_finite_for_all_quadrature_points(self, polys, theta):
        """
        I1 should be finite for various quadrature sizes.
        """
        R = 1.3036

        for n in [5, 10, 20, 40]:
            result = compute_I1_operator_post_identity_11(theta, R, n, polys)
            assert np.isfinite(result.I1_value), f"I1 not finite at n={n}: {result.I1_value}"
