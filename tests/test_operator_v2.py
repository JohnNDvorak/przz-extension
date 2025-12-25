"""
tests/test_operator_v2.py
Tests for operator mode V2 (Codex Tests from GPT guidance 2025-12-20)

Tests:
1. Shift=0 identity: operator mode with sigma=0 must match base
2. Direct branch bitwise identity: +R computations unchanged
3. Quarantined normalization: q1_ratio requires allow_unstable and warns
4. I₁ factor localization: left/right-only scopes are wired
"""

import pytest
import numpy as np

from src.evaluate import (
    compute_c_paper_operator_v2,
    solve_two_weight_operator,
)
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167
KAPPA_C_TARGET = 2.137
KAPPA_STAR_C_TARGET = 1.938


@pytest.fixture
def polys_kappa():
    """Load κ benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


@pytest.fixture
def polys_kappa_star():
    """Load κ* benchmark polynomials."""
    P1, P2, P3, Q = load_przz_polynomials_kappa_star()
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


class TestSigmaZeroIdentity:
    """Test that sigma=0 reproduces base exactly."""
    
    def test_sigma_zero_i1_only_channels_match(self, polys_kappa):
        """With sigma=0, operator channels should equal base channels."""
        result = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i1_only", sigma=0.0,
        )
        
        # I1 channels should match
        assert abs(result.per_term["_I1_minus_op"] - result.per_term["_I1_minus_base"]) < 1e-10
        # I2 should be unchanged regardless
        assert abs(result.per_term["_I2_minus_op"] - result.per_term["_I2_minus_base"]) < 1e-10
    
    def test_sigma_zero_both_channels_match(self, polys_kappa):
        """With sigma=0 and scope=both, all operator channels should equal base."""
        result = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="both", sigma=0.0,
        )
        
        assert abs(result.per_term["_I1_minus_op"] - result.per_term["_I1_minus_base"]) < 1e-10
        assert abs(result.per_term["_I2_minus_op"] - result.per_term["_I2_minus_base"]) < 1e-10
    
    def test_sigma_zero_2x2_solve_identity(self, polys_kappa, polys_kappa_star):
        """With sigma=0, operator-solved weights should match base-solved weights."""
        result_k = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i1_only", sigma=0.0,
        )
        result_ks = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_STAR_R, n=40, polynomials=polys_kappa_star,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i1_only", sigma=0.0,
        )
        
        op_solve = solve_two_weight_operator(
            result_k, result_ks,
            c_target_k=KAPPA_C_TARGET, c_target_k_star=KAPPA_STAR_C_TARGET,
            use_operator_channels=True,
        )
        base_solve = solve_two_weight_operator(
            result_k, result_ks,
            c_target_k=KAPPA_C_TARGET, c_target_k_star=KAPPA_STAR_C_TARGET,
            use_operator_channels=False,
        )
        
        # Weights should match to machine precision
        assert abs(op_solve["m1"] - base_solve["m1"]) < 1e-8
        assert abs(op_solve["m2"] - base_solve["m2"]) < 1e-8
    
    def test_sigma_zero_flag_set(self, polys_kappa):
        """Verify _is_identity_mode flag is set for sigma=0."""
        result = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i1_only", sigma=0.0,
        )
        
        assert result.per_term["_is_identity_mode"] == True
        assert result.per_term["_sigma"] == 0.0


class TestDirectBranchUnchanged:
    """Test that +R (direct branch) computations are unchanged by operator mode."""
    
    def test_plus_channels_unchanged(self, polys_kappa):
        """I_plus channels should be identical regardless of operator settings."""
        # Run with sigma=0 (identity)
        result_identity = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i1_only", sigma=0.0,
        )
        
        # Run with sigma=1 (operator mode)
        result_operator = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i1_only", sigma=1.0,
        )
        
        # Plus channels should be bitwise identical (only minus branch is modified)
        assert result_identity.per_term["_I1_plus"] == result_operator.per_term["_I1_plus"]
        assert result_identity.per_term["_I2_plus"] == result_operator.per_term["_I2_plus"]
        assert result_identity.per_term["_S34_plus"] == result_operator.per_term["_S34_plus"]
    
    def test_s34_always_base(self, polys_kappa):
        """S34 (I3+I4) should always use base polynomials (no operator effect)."""
        result = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="grid", lift_scope="both", sigma=1.0,
        )
        
        # S34 should be in the output (not NaN or zero due to operator bug)
        assert not np.isnan(result.per_term["_S34_plus"])
        assert abs(result.per_term["_S34_plus"]) > 0.1  # Should be substantial


class TestNearZeroNormalization:
    """Test that near-zero normalizations are properly rejected."""

    def test_q1_ratio_requires_allow_unstable(self, polys_kappa):
        """q1_ratio is quarantined and requires allow_unstable=True."""
        with pytest.raises(ValueError, match="allow_unstable"):
            compute_c_paper_operator_v2(
                theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
                n_quad_a=30, verbose=False,
                normalization="q1_ratio", lift_scope="i1_only", sigma=1.0,
                allow_unstable=False,
            )

    def test_q1_ratio_warns_and_scales(self, polys_kappa):
        """q1_ratio should warn and produce large |norm_factor| for PRZZ κ Q(1)≈0."""
        import warnings
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            result = compute_c_paper_operator_v2(
                theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
                n_quad_a=30, verbose=False,
                normalization="q1_ratio", lift_scope="i1_only", sigma=1.0,
                allow_unstable=True,
            )

        assert any("q1_ratio normalization" in str(w.message) for w in rec)

        # Q(0)/Q(1) = 1.0 / (-0.019) ≈ -52, so abs > 10
        assert abs(result.per_term["_norm_factor"]) > 10
    
    def test_q1_ratio_allowed_with_flag(self, polys_kappa):
        """q1_ratio should work with allow_unstable=True."""
        # Should not raise, but will produce unstable results
        result = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="q1_ratio", lift_scope="i1_only", sigma=1.0,
            allow_unstable=True,
        )
        
        # Should complete but norm_factor will be large
        assert result.per_term["_normalization"] == "q1_ratio"
        assert abs(result.per_term["_norm_factor"]) > 10  # Very large due to Q(1)≈0
    
    def test_safe_normalizations_work(self, polys_kappa):
        """l2 and grid normalizations should work without errors."""
        for norm in ["none", "l2", "grid"]:
            result = compute_c_paper_operator_v2(
                theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
                n_quad_a=30, verbose=False,
                normalization=norm, lift_scope="i1_only", sigma=1.0,
            )
            
            assert result.per_term["_normalization"] == norm
            assert not np.isnan(result.per_term["_norm_factor"])
            assert result.per_term["_norm_factor"] > 0


class TestI1FactorLocalization:
    """Wiring tests for i1_left_only / i1_right_only scopes."""

    def test_left_only_differs_from_both_factors(self, polys_kappa):
        """Shifting only one I₁ Q factor should differ from shifting both."""
        base_kwargs = dict(
            theta=THETA,
            R=KAPPA_R,
            n=30,
            polynomials=polys_kappa,
            n_quad_a=20,
            verbose=False,
            normalization="grid",
            sigma=0.5,
        )

        full = compute_c_paper_operator_v2(**base_kwargs, lift_scope="i1_only")
        left = compute_c_paper_operator_v2(**base_kwargs, lift_scope="i1_left_only")
        right = compute_c_paper_operator_v2(**base_kwargs, lift_scope="i1_right_only")

        assert abs(full.per_term["_I1_minus_op"] - left.per_term["_I1_minus_op"]) > 1e-8
        assert abs(full.per_term["_I1_minus_op"] - right.per_term["_I1_minus_op"]) > 1e-8


class TestValidInputs:
    """Test input validation."""
    
    def test_invalid_scope_rejected(self, polys_kappa):
        """Invalid lift_scope should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid lift_scope"):
            compute_c_paper_operator_v2(
                theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
                n_quad_a=30, verbose=False,
                normalization="none", lift_scope="invalid_scope", sigma=1.0,
            )
    
    def test_invalid_normalization_rejected(self, polys_kappa):
        """Invalid normalization should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization"):
            compute_c_paper_operator_v2(
                theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
                n_quad_a=30, verbose=False,
                normalization="invalid_norm", lift_scope="i1_only", sigma=1.0,
            )
    
    def test_valid_scopes_accepted(self, polys_kappa):
        """All valid scopes should be accepted."""
        valid_scopes = ["both", "i1_only", "i2_only", "i1_left_only", "i1_right_only"]
        
        for scope in valid_scopes:
            result = compute_c_paper_operator_v2(
                theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
                n_quad_a=30, verbose=False,
                normalization="none", lift_scope=scope, sigma=1.0,
            )
            assert result.per_term["_lift_scope"] == scope


class TestOperatorEffect:
    """Test that operator mode actually changes something for sigma != 0."""
    
    def test_sigma_nonzero_changes_channels(self, polys_kappa):
        """Non-zero sigma should change the minus channels."""
        result = compute_c_paper_operator_v2(
            theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
            n_quad_a=30, verbose=False,
            normalization="none", lift_scope="i1_only", sigma=1.0,
        )
        
        # I1 minus should be different
        assert abs(result.per_term["_I1_minus_op"] - result.per_term["_I1_minus_base"]) > 1e-6
        # I2 minus should be unchanged (scope is i1_only)
        assert abs(result.per_term["_I2_minus_op"] - result.per_term["_I2_minus_base"]) < 1e-10
    
    def test_different_sigmas_give_different_results(self, polys_kappa):
        """Different sigma values should produce different results."""
        results = []
        for sigma in [0.0, 0.5, 1.0]:
            r = compute_c_paper_operator_v2(
                theta=THETA, R=KAPPA_R, n=40, polynomials=polys_kappa,
                n_quad_a=30, verbose=False,
                normalization="none", lift_scope="i1_only", sigma=sigma,
            )
            results.append(r.per_term["_I1_minus_op"])
        
        # All three should be different
        assert results[0] != results[1]
        assert results[1] != results[2]
        assert results[0] != results[2]
