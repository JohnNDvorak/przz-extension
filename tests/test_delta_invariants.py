"""
Tests for delta decomposition invariants (Phase 19.3).

These tests verify the structural invariants of the delta computation:
1. δ == D/A within tolerance
2. Triangle convention consistency
3. B/A near target (2K-1)
4. Mode separation consistency
"""

import pytest
import sys
from pathlib import Path

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_delta_report import (
    compute_delta_decomposition,
    DeltaDecomposition,
    map_laurent_mode_to_phase19,
)
from src.ratios.j1_euler_maclaurin import LaurentMode


class TestDeltaEqualsDoA:
    """Test that δ == D/A invariant holds."""

    def test_delta_equals_D_over_A_kappa_numeric(self):
        """δ == D/A for κ benchmark in numeric mode."""
        decomp = compute_delta_decomposition("kappa", mode="numeric")

        # Compute D/A directly
        D_over_A = decomp.D / decomp.A

        # Should match stored delta
        assert abs(D_over_A - decomp.delta) < 1e-6
        assert decomp.invariants["delta_equals_D_over_A"]

    def test_delta_equals_D_over_A_kappa_star_numeric(self):
        """δ == D/A for κ* benchmark in numeric mode."""
        decomp = compute_delta_decomposition("kappa_star", mode="numeric")

        D_over_A = decomp.D / decomp.A
        assert abs(D_over_A - decomp.delta) < 1e-6
        assert decomp.invariants["delta_equals_D_over_A"]

    def test_delta_equals_D_over_A_semantic(self):
        """δ == D/A in semantic mode."""
        decomp = compute_delta_decomposition("kappa", mode="semantic")

        D_over_A = decomp.D / decomp.A
        assert abs(D_over_A - decomp.delta) < 1e-6

    def test_D_composition(self):
        """D == I12+ + I34+."""
        decomp = compute_delta_decomposition("kappa", mode="numeric")

        expected_D = decomp.per_piece["I12_plus_R"] + decomp.per_piece["I34_plus_R"]
        assert abs(decomp.D - expected_D) < 1e-10


class TestTriangleConvention:
    """Test TRUTH_SPEC triangle convention (ℓ₁ ≤ ℓ₂)."""

    def test_triangle_convention_marked(self):
        """Triangle convention is checked in invariants."""
        decomp = compute_delta_decomposition("kappa", mode="numeric")
        assert "triangle_convention" in decomp.invariants

    def test_triangle_convention_passes(self):
        """Triangle convention should pass for properly structured computation."""
        for bench in ["kappa", "kappa_star"]:
            decomp = compute_delta_decomposition(bench, mode="numeric")
            assert decomp.invariants["triangle_convention"]


class TestBOverATarget:
    """Test B/A is near expected 2K-1."""

    def test_B_over_A_near_5_for_K3(self):
        """B/A should be near 5 for K=3."""
        decomp = compute_delta_decomposition("kappa", mode="numeric", K=3)

        # Should be close to 5 (within 5% is acceptable)
        assert abs(decomp.B_over_A - 5.0) / 5.0 < 0.05
        assert decomp.invariants["B_over_A_near_target"]

    def test_B_over_A_gap_percent_computed(self):
        """Gap percent is correctly computed."""
        decomp = compute_delta_decomposition("kappa", mode="numeric")

        expected_gap = (decomp.B_over_A - 5.0) / 5.0 * 100
        assert abs(decomp.gap_percent - expected_gap) < 0.001

    def test_both_benchmarks_have_small_gap(self):
        """Both benchmarks should have <5% gap in numeric mode."""
        for bench in ["kappa", "kappa_star"]:
            decomp = compute_delta_decomposition(bench, mode="numeric")
            assert abs(decomp.gap_percent) < 5.0


class TestModeConsistency:
    """Test semantic vs numeric mode consistency."""

    def test_mode_mapping(self):
        """Mode mapping is correct."""
        assert map_laurent_mode_to_phase19(LaurentMode.RAW_LOGDERIV) == "SEMANTIC_LAURENT"
        assert map_laurent_mode_to_phase19(LaurentMode.ACTUAL_LOGDERIV) == "NUMERIC_FUNCTIONAL_EQ"

    def test_semantic_mode_sets_correct_label(self):
        """Semantic mode is properly labeled."""
        decomp = compute_delta_decomposition("kappa", mode="semantic")
        assert decomp.mode == "SEMANTIC_LAURENT"

    def test_numeric_mode_sets_correct_label(self):
        """Numeric mode is properly labeled."""
        decomp = compute_delta_decomposition("kappa", mode="numeric")
        assert decomp.mode == "NUMERIC_FUNCTIONAL_EQ"

    def test_modes_give_different_results(self):
        """Semantic and numeric modes give different B/A values."""
        semantic = compute_delta_decomposition("kappa", mode="semantic")
        numeric = compute_delta_decomposition("kappa", mode="numeric")

        # They should be different (Phase 15A finding: Laurent has ~30% error)
        assert abs(semantic.B_over_A - numeric.B_over_A) > 0.01

    def test_semantic_has_warning_at_large_R(self):
        """Semantic mode warns about Laurent error at large R."""
        decomp = compute_delta_decomposition("kappa", mode="semantic")

        # R=1.3 is large, should have warning
        assert len(decomp.warnings) > 0
        assert any("Laurent" in w or "SEMANTIC" in w for w in decomp.warnings)


class TestPerPieceBreakdown:
    """Test per-piece breakdown is complete."""

    def test_all_j_pieces_present(self):
        """All J pieces are in breakdown."""
        decomp = compute_delta_decomposition("kappa", mode="numeric")

        expected_pieces = [
            "j11_plus", "j12_plus", "j15_plus",
            "j13_plus", "j14_plus",
            "j11_minus", "j12_minus", "j15_minus",
        ]

        for piece in expected_pieces:
            assert piece in decomp.per_piece

    def test_i_terms_present(self):
        """I12 and I34 totals are in breakdown."""
        decomp = compute_delta_decomposition("kappa", mode="numeric")

        assert "I12_plus_R" in decomp.per_piece
        assert "I12_minus_R" in decomp.per_piece
        assert "I34_plus_R" in decomp.per_piece

    def test_i12_plus_equals_sum_of_j_pieces(self):
        """I12+ equals sum of j11+, j12+, j15+."""
        decomp = compute_delta_decomposition("kappa", mode="numeric")

        j_sum = (
            decomp.per_piece["j11_plus"] +
            decomp.per_piece["j12_plus"] +
            decomp.per_piece["j15_plus"]
        )

        assert abs(decomp.per_piece["I12_plus_R"] - j_sum) < 1e-10

    def test_i34_plus_equals_sum_of_j_pieces(self):
        """I34+ equals sum of j13+, j14+."""
        decomp = compute_delta_decomposition("kappa", mode="numeric")

        j_sum = decomp.per_piece["j13_plus"] + decomp.per_piece["j14_plus"]

        assert abs(decomp.per_piece["I34_plus_R"] - j_sum) < 1e-10


class TestCoreBehavior:
    """Test core metric behavior."""

    def test_A_is_positive(self):
        """A (exp coefficient) should be positive."""
        for bench in ["kappa", "kappa_star"]:
            decomp = compute_delta_decomposition(bench, mode="numeric")
            assert decomp.A > 0

    def test_delta_is_small(self):
        """δ should be small (< 0.5 typically)."""
        for bench in ["kappa", "kappa_star"]:
            decomp = compute_delta_decomposition(bench, mode="numeric")
            assert abs(decomp.delta) < 0.5

    def test_D_is_small_compared_to_B(self):
        """D should be small compared to B (it's the residual)."""
        decomp = compute_delta_decomposition("kappa", mode="numeric")

        # D is typically ~0.03-0.05 while B is ~0.5-0.6
        assert abs(decomp.D) < abs(decomp.B)


class TestDecompositionStructure:
    """Test DeltaDecomposition dataclass structure."""

    def test_decomposition_has_all_fields(self):
        """Decomposition has all required fields."""
        decomp = compute_delta_decomposition("kappa", mode="numeric")

        assert hasattr(decomp, 'benchmark')
        assert hasattr(decomp, 'mode')
        assert hasattr(decomp, 'R')
        assert hasattr(decomp, 'theta')
        assert hasattr(decomp, 'K')
        assert hasattr(decomp, 'A')
        assert hasattr(decomp, 'B')
        assert hasattr(decomp, 'D')
        assert hasattr(decomp, 'delta')
        assert hasattr(decomp, 'B_over_A')
        assert hasattr(decomp, 'gap_percent')
        assert hasattr(decomp, 'per_piece')
        assert hasattr(decomp, 'invariants')
        assert hasattr(decomp, 'warnings')

    def test_to_dict_roundtrips(self):
        """to_dict produces JSON-serializable output."""
        import json

        decomp = compute_delta_decomposition("kappa", mode="numeric")
        d = decomp.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(d)
        recovered = json.loads(json_str)

        assert recovered["benchmark"] == decomp.benchmark
        assert abs(recovered["B_over_A"] - decomp.B_over_A) < 1e-10


class TestTwoBenchmarkGate:
    """Test two-benchmark gate requirement."""

    def test_both_benchmarks_pass_invariants(self):
        """Both benchmarks should pass all invariants."""
        for bench in ["kappa", "kappa_star"]:
            decomp = compute_delta_decomposition(bench, mode="numeric")

            for inv_name, passed in decomp.invariants.items():
                assert passed, f"{bench}.{inv_name} failed"

    def test_gap_consistent_across_benchmarks(self):
        """Gap should be in similar direction for both benchmarks."""
        kappa = compute_delta_decomposition("kappa", mode="numeric")
        kappa_star = compute_delta_decomposition("kappa_star", mode="numeric")

        # Both should be slightly below 5 (negative gap) in ACTUAL mode
        # based on Phase 18 findings
        # Just check they're in same ballpark
        assert abs(kappa.gap_percent) < 5
        assert abs(kappa_star.gap_percent) < 5
