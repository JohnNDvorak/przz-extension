"""
tests/test_mirror_exp_kernel_consistency.py
Phase 13.0: Exp-kernel consistency test for mirror operator.

GPT's insight: If eigenvalues satisfy A_mirror(t) = A_direct(1−t),
then exp factor must also be consistent with t→1−t.

This test validates that the mirror exp coefficients match the direct
exp coefficients evaluated at t' = 1-t.
"""

import pytest
import numpy as np
from src.mirror_operator_exact import (
    get_mirror_exp_affine_coeffs,
    get_mirror_exp_affine_coeffs_t_flip,  # New function to be implemented
)
from src.operator_post_identity import get_exp_affine_coeffs


THETA = 4.0 / 7.0
R = 1.3036


class TestExpKernelConsistency:
    """Verify exp kernel transforms consistently with eigenvalues."""

    def test_exp_u0_sums_to_2R(self):
        """
        The u0 terms should satisfy: mir_u0(t) + dir_u0(1-t) = 2R.

        Mirror: u0 = 2Rt
        Direct at 1-t: u0 = 2R(1-t)
        Sum = 2R
        """
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mir_u0, _, _ = get_mirror_exp_affine_coeffs(t, THETA, R)
            dir_u0, _, _ = get_exp_affine_coeffs(1-t, THETA, R)

            # Sum should equal 2R
            assert abs(mir_u0 + dir_u0 - 2*R) < 1e-10, (
                f"u0 sum mismatch at t={t}: mir={mir_u0}, dir={dir_u0}, sum={mir_u0+dir_u0}"
            )

    def test_exp_lin_mismatch_is_the_bug(self):
        """
        Demonstrate that the current mirror exp has wrong lin coefficients.

        Current (buggy): lin = -θR (static)
        Correct: lin = θR(2t-1) (t-dependent, matching direct at 1-t)
        """
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mir_u0, mir_x, mir_y = get_mirror_exp_affine_coeffs(t, THETA, R)
            dir_u0, dir_x, dir_y = get_exp_affine_coeffs(1-t, THETA, R)

            # Current mirror: always -θR
            expected_buggy = -THETA * R
            assert abs(mir_x - expected_buggy) < 1e-10, "Mirror x coeff should be -θR (buggy)"
            assert abs(mir_y - expected_buggy) < 1e-10, "Mirror y coeff should be -θR (buggy)"

            # Direct at 1-t: θR(2(1-t)-1) = θR(1-2t)
            expected_correct = R * THETA * (1 - 2*t)
            assert abs(dir_x - expected_correct) < 1e-10, f"Direct x coeff mismatch at t={t}"
            assert abs(dir_y - expected_correct) < 1e-10, f"Direct y coeff mismatch at t={t}"

            # At t=0.5, these differ maximally
            if abs(t - 0.5) < 0.01:
                assert abs(mir_x - dir_x) > 0.7, "At t=0.5, mirror and direct lin should differ!"

    def test_fixed_exp_matches_direct_t_flip(self):
        """
        After fix, mirror exp should match direct exp at t' = 1-t.

        This test uses the new get_mirror_exp_affine_coeffs_t_flip() function.
        With include_T_weight=True, the u0 should match directly.
        """
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            # New fixed function with T-weight included
            mir_u0, mir_x, mir_y = get_mirror_exp_affine_coeffs_t_flip(
                t, THETA, R, include_T_weight=True
            )

            # Direct at 1-t
            dir_u0, dir_x, dir_y = get_exp_affine_coeffs(1-t, THETA, R)

            # After fix, they should match exactly
            assert abs(mir_u0 - dir_u0) < 1e-10, f"Fixed mirror u0 mismatch at t={t}"
            assert abs(mir_x - dir_x) < 1e-10, f"Fixed mirror x mismatch at t={t}"
            assert abs(mir_y - dir_y) < 1e-10, f"Fixed mirror y mismatch at t={t}"

    def test_fixed_exp_without_T_weight(self):
        """
        With include_T_weight=False (default), exp(2R) × exp(u0) should equal direct(1-t).
        """
        import numpy as np
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mir_u0, mir_x, mir_y = get_mirror_exp_affine_coeffs_t_flip(
                t, THETA, R, include_T_weight=False
            )
            dir_u0, dir_x, dir_y = get_exp_affine_coeffs(1-t, THETA, R)

            # mir_u0 = -2Rt, so exp(2R) × exp(-2Rt) = exp(2R(1-t)) = exp(dir_u0)
            total_u0 = 2 * R + mir_u0  # T_weight contribution + mirror u0
            assert abs(total_u0 - dir_u0) < 1e-10, f"Total u0 mismatch at t={t}"

            # Lin coefficients should still match
            assert abs(mir_x - dir_x) < 1e-10, f"Fixed mirror x mismatch at t={t}"
            assert abs(mir_y - dir_y) < 1e-10, f"Fixed mirror y mismatch at t={t}"


class TestExpKernelMathStructure:
    """Verify the mathematical structure of exp coefficients."""

    def test_direct_exp_structure(self):
        """
        Direct exp: exp(2Rt + θR(2t-1)(x+y))

        u0 = 2Rt
        lin = θR(2t-1)
        """
        t = 0.3
        u0, lin_x, lin_y = get_exp_affine_coeffs(t, THETA, R)

        expected_u0 = 2 * R * t
        expected_lin = R * THETA * (2*t - 1)

        assert abs(u0 - expected_u0) < 1e-10
        assert abs(lin_x - expected_lin) < 1e-10
        assert abs(lin_y - expected_lin) < 1e-10

    def test_mirror_exp_structure_static(self):
        """
        Current (buggy) mirror exp: exp(2Rt - θR(x+y))

        u0 = 2Rt
        lin = -θR (STATIC - this is the bug!)
        """
        for t in [0.0, 0.5, 1.0]:
            u0, lin_x, lin_y = get_mirror_exp_affine_coeffs(t, THETA, R)

            expected_u0 = 2 * R * t
            expected_lin = -THETA * R  # STATIC!

            assert abs(u0 - expected_u0) < 1e-10
            assert abs(lin_x - expected_lin) < 1e-10, f"lin_x at t={t}"
            assert abs(lin_y - expected_lin) < 1e-10, f"lin_y at t={t}"

    def test_what_correct_mirror_exp_should_be(self):
        """
        For consistency with t→1-t, mirror exp should be:
            exp(2Rt + θR(1-2t)(x+y))

        NOT: exp(2Rt - θR(x+y))

        Note: The u0 relationship is more subtle due to T-weight.
        """
        # At t=0, direct(1-t=1) has u0=2R, lin=θR
        # At t=1, direct(1-t=0) has u0=0, lin=-θR
        # At t=0.5, direct(1-t=0.5) has u0=R, lin=0

        for t, expected_lin in [(0.0, THETA*R), (0.5, 0.0), (1.0, -THETA*R)]:
            _, dir_x, _ = get_exp_affine_coeffs(1-t, THETA, R)
            assert abs(dir_x - expected_lin) < 1e-10, (
                f"Direct at 1-t={1-t}: expected lin={expected_lin}, got {dir_x}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
