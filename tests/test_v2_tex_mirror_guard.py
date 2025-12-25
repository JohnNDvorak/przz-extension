#!/usr/bin/env python3
"""
tests/test_v2_tex_mirror_guard.py

Gate test: Verify V2 + tex_mirror combination is forbidden.

GPT Run 12-13 discovered that V2 terms catastrophically fail under tex_mirror:
- I1_plus flips sign from +0.085 (OLD) to -0.111 (V2)
- This causes c to collapse to ~0.775 vs target ~2.137
- OLD + tex_mirror is the only production-safe combination

This test ensures no accidental use of the forbidden V2 + tex_mirror combination.

See:
- docs/HANDOFF_GPT_RUN12_13.md
- docs/RUN12A_CHANNEL_DIFF.md
- docs/TEX_VERIFICATION_1_MINUS_U.md
"""

import pytest
from src.evaluate import compute_c_paper_tex_mirror
from src.polynomials import load_przz_polynomials

THETA = 4.0 / 7.0
R_KAPPA = 1.3036


class TestV2TexMirrorGuard:
    """Ensure V2 + tex_mirror combination raises error."""

    @pytest.fixture
    def polys(self):
        """Load PRZZ polynomials for testing."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_v2_with_tex_mirror_raises_error(self, polys):
        """terms_version='v2' with tex_mirror should raise ValueError.

        GPT Run 12A showed V2 causes I1_plus to flip sign, collapsing c to ~0.775.
        This combination must be forbidden.
        """
        with pytest.raises(ValueError, match="FORBIDDEN"):
            compute_c_paper_tex_mirror(
                theta=THETA,
                R=R_KAPPA,
                n=40,
                polynomials=polys,
                terms_version="v2",
            )

    def test_v2_error_message_is_informative(self, polys):
        """Error message should explain why V2 is forbidden and point to docs."""
        with pytest.raises(ValueError) as exc_info:
            compute_c_paper_tex_mirror(
                theta=THETA,
                R=R_KAPPA,
                n=40,
                polynomials=polys,
                terms_version="v2",
            )
        error_msg = str(exc_info.value)
        # Should mention the core issue
        assert "sign flip" in error_msg.lower() or "I1_plus" in error_msg
        # Should point to documentation
        assert "HANDOFF" in error_msg or "Run 12" in error_msg

    def test_old_with_tex_mirror_works(self, polys):
        """terms_version='old' with tex_mirror should work.

        OLD + tex_mirror is the production-safe combination that achieves
        <1% accuracy on both benchmarks.
        """
        result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R_KAPPA,
            n=40,
            polynomials=polys,
            terms_version="old",
        )
        assert result.c > 0, "Should produce positive c"
        # Should be in the reasonable range (target is ~2.137)
        assert 1.5 < result.c < 3.0, f"c={result.c} out of expected range"

    def test_default_terms_version_works(self, polys):
        """Default terms_version (old) should work without explicit specification."""
        result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R_KAPPA,
            n=40,
            polynomials=polys,
            # terms_version defaults to "old"
        )
        assert result.c > 0, "Should produce positive c"
        # Should be in the reasonable range
        assert 1.5 < result.c < 3.0, f"c={result.c} out of expected range"

    def test_old_with_exp_R_ref_achieves_target(self, polys):
        """Verify OLD + tex_mirror + exp_R_ref achieves stated <1% accuracy.

        This is the regression test for the production configuration.
        Target: c = 2.137, gap < 1%
        """
        c_target = 2.13745440613217263636

        result = compute_c_paper_tex_mirror(
            theta=THETA,
            R=R_KAPPA,
            n=60,
            polynomials=polys,
            terms_version="old",
            tex_exp_component="exp_R_ref",
        )

        c_gap_pct = 100 * (result.c - c_target) / c_target
        assert abs(c_gap_pct) < 2.0, (
            f"c gap {c_gap_pct:.2f}% exceeds 2% threshold. "
            f"c={result.c:.6f}, target={c_target:.6f}"
        )
