"""
tests/test_delta_track.py
Tests for the delta tracking module (Phase 18.2).
"""

import pytest

from src.ratios.delta_track import (
    DeltaRecord,
    ConvergenceTable,
    compute_delta_record,
    run_convergence_sweep,
    run_delta_track_report,
)
from src.ratios.j1_euler_maclaurin import LaurentMode


class TestDeltaRecord:
    """Tests for DeltaRecord structure."""

    def test_kappa_record_fields(self):
        """Test that kappa DeltaRecord has all expected fields."""
        record = compute_delta_record("kappa")

        assert record.benchmark == "kappa"
        assert record.R == 1.3036
        assert record.K == 3
        assert record.theta == 4.0 / 7.0

        # Core metrics should be numeric
        assert isinstance(record.A, float)
        assert isinstance(record.B, float)
        assert isinstance(record.delta, float)
        assert isinstance(record.B_over_A, float)

        # Channel breakdown
        assert isinstance(record.I12_plus, float)
        assert isinstance(record.I12_minus, float)
        assert isinstance(record.I34_plus, float)

        # Per-piece breakdown
        assert isinstance(record.j11_plus, float)
        assert isinstance(record.j12_plus, float)
        assert isinstance(record.j15_plus, float)
        assert isinstance(record.j13_plus, float)
        assert isinstance(record.j14_plus, float)

    def test_kappa_star_record_fields(self):
        """Test that kappa* DeltaRecord has all expected fields."""
        record = compute_delta_record("kappa_star")

        assert record.benchmark == "kappa_star"
        assert record.R == 1.1167
        assert record.K == 3

    def test_record_to_dict(self):
        """Test that to_dict works correctly."""
        record = compute_delta_record("kappa")
        d = record.to_dict()

        assert isinstance(d, dict)
        assert "benchmark" in d
        assert "B_over_A" in d
        assert d["benchmark"] == "kappa"

    def test_channel_consistency(self):
        """Test that D = I12_plus + I34_plus."""
        for benchmark in ["kappa", "kappa_star"]:
            record = compute_delta_record(benchmark)
            D_computed = record.I12_plus + record.I34_plus
            assert abs(record.delta * record.A - D_computed) < 1e-10

    def test_b_over_a_formula(self):
        """Test that B_over_A = (2K-1) + delta."""
        for benchmark in ["kappa", "kappa_star"]:
            record = compute_delta_record(benchmark)
            expected = (2 * record.K - 1) + record.delta
            assert abs(record.B_over_A - expected) < 1e-10


class TestLaurentModeEffect:
    """Tests for Laurent mode effects on delta."""

    def test_actual_vs_raw_differ(self):
        """Test that ACTUAL and RAW modes give different results."""
        actual = compute_delta_record("kappa", LaurentMode.ACTUAL_LOGDERIV)
        raw = compute_delta_record("kappa", LaurentMode.RAW_LOGDERIV)

        assert actual.B_over_A != raw.B_over_A
        assert actual.delta != raw.delta

    def test_actual_mode_closer_to_target(self):
        """Test that ACTUAL mode gives B/A closer to 5 for at least one benchmark."""
        # ACTUAL mode should generally be better, though may overshoot for kappa*

        actual_k = compute_delta_record("kappa", LaurentMode.ACTUAL_LOGDERIV)
        raw_k = compute_delta_record("kappa", LaurentMode.RAW_LOGDERIV)

        # For kappa, ACTUAL should be closer to 5
        actual_gap_k = abs(actual_k.B_over_A - 5.0)
        raw_gap_k = abs(raw_k.B_over_A - 5.0)
        assert actual_gap_k < raw_gap_k, "ACTUAL should be better for kappa"


class TestConvergenceSweep:
    """Tests for convergence sweep functionality."""

    def test_sweep_returns_table(self):
        """Test that convergence sweep returns a ConvergenceTable."""
        table = run_convergence_sweep(
            "kappa",
            "quadrature_n",
            [40, 60],  # Small sweep for speed
            LaurentMode.ACTUAL_LOGDERIV,
        )

        assert isinstance(table, ConvergenceTable)
        assert table.benchmark == "kappa"
        assert table.sweep_parameter == "quadrature_n"
        assert len(table.records) == 2

    def test_sweep_parameter_values_match(self):
        """Test that parameter values are recorded correctly."""
        values = [40, 60, 80]
        table = run_convergence_sweep("kappa", "quadrature_n", values)

        assert table.parameter_values == tuple(values)
        assert len(table.records) == len(values)

    def test_unsupported_sweep_raises(self):
        """Test that unsupported sweep types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported sweep type"):
            run_convergence_sweep("kappa", "unsupported_param", [1, 2, 3])

    def test_summary_table_format(self):
        """Test that summary_table returns a string."""
        table = run_convergence_sweep("kappa", "quadrature_n", [40, 60])
        summary = table.summary_table()

        assert isinstance(summary, str)
        assert "quadrature_n" in summary
        assert "kappa" in summary


class TestDeltaTrackReport:
    """Tests for the comprehensive report."""

    def test_report_returns_both_benchmarks(self):
        """Test that report includes both benchmarks with both modes."""
        results = run_delta_track_report(verbose=False)

        assert "kappa_actual_logderiv" in results
        assert "kappa_raw_logderiv" in results
        assert "kappa_star_actual_logderiv" in results
        assert "kappa_star_raw_logderiv" in results

    def test_all_records_are_valid(self):
        """Test that all returned records are DeltaRecord instances."""
        results = run_delta_track_report(verbose=False)

        for key, record in results.items():
            assert isinstance(record, DeltaRecord)
            assert record.B_over_A > 0  # Should be positive


class TestPerPieceBreakdown:
    """Tests for per-piece breakdown consistency."""

    def test_i12_plus_equals_sum_of_pieces(self):
        """Test that I12_plus = j11_plus + j12_plus + j15_plus."""
        for benchmark in ["kappa", "kappa_star"]:
            record = compute_delta_record(benchmark)
            sum_pieces = record.j11_plus + record.j12_plus + record.j15_plus
            assert abs(record.I12_plus - sum_pieces) < 1e-10

    def test_i34_plus_equals_sum_of_pieces(self):
        """Test that I34_plus = j13_plus + j14_plus."""
        for benchmark in ["kappa", "kappa_star"]:
            record = compute_delta_record(benchmark)
            sum_pieces = record.j13_plus + record.j14_plus
            assert abs(record.I34_plus - sum_pieces) < 1e-10

    def test_i12_minus_equals_sum_of_pieces(self):
        """Test that I12_minus = j11_minus + j12_minus + j15_minus."""
        for benchmark in ["kappa", "kappa_star"]:
            record = compute_delta_record(benchmark)
            sum_pieces = record.j11_minus + record.j12_minus + record.j15_minus
            assert abs(record.I12_minus - sum_pieces) < 1e-10
