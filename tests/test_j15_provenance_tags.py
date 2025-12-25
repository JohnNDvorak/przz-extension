"""
tests/test_j15_provenance_tags.py
Phase 20.1: Tests for J15 Provenance Metadata

PURPOSE:
========
Verify that J15 provenance tags are correctly attached to Plus5SplitResult
and provide stable, meaningful metadata for reconciliation.

Tests:
1. Result includes 'j15_provenance' field
2. Provenance includes module name
3. Provenance includes function name
4. Provenance stable across runs
5. Provenance includes PRZZ references
6. Provenance includes TRUTH_SPEC reference

USAGE:
======
    pytest tests/test_j15_provenance_tags.py -v
"""

import pytest

from src.ratios.plus5_harness import (
    compute_plus5_signature_split,
    Plus5SplitResult,
    J15Provenance,
    DEFAULT_J15_PROVENANCE,
)


class TestProvenanceFieldExists:
    """Test that provenance field is present in results."""

    def test_result_has_provenance_field(self):
        """Plus5SplitResult should have j15_provenance field."""
        result = compute_plus5_signature_split("kappa")
        assert hasattr(result, "j15_provenance")

    def test_provenance_is_j15provenance_type(self):
        """Provenance should be J15Provenance dataclass."""
        result = compute_plus5_signature_split("kappa")
        assert isinstance(result.j15_provenance, J15Provenance)

    def test_default_provenance_exists(self):
        """DEFAULT_J15_PROVENANCE should be defined."""
        assert DEFAULT_J15_PROVENANCE is not None
        assert isinstance(DEFAULT_J15_PROVENANCE, J15Provenance)


class TestProvenanceModuleInfo:
    """Test that provenance includes source module information."""

    def test_provenance_has_module_name(self):
        """Provenance should include source module name."""
        result = compute_plus5_signature_split("kappa")
        assert hasattr(result.j15_provenance, "source_module")
        assert result.j15_provenance.source_module != ""

    def test_module_name_is_j1_euler_maclaurin(self):
        """Current implementation should use j1_euler_maclaurin module."""
        result = compute_plus5_signature_split("kappa")
        assert "j1_euler_maclaurin" in result.j15_provenance.source_module

    def test_provenance_has_function_name(self):
        """Provenance should include source function name."""
        result = compute_plus5_signature_split("kappa")
        assert hasattr(result.j15_provenance, "source_function")
        assert result.j15_provenance.source_function != ""


class TestProvenancePRZZReferences:
    """Test that provenance includes PRZZ paper references."""

    def test_provenance_has_przz_lines(self):
        """Provenance should include PRZZ TeX line numbers."""
        result = compute_plus5_signature_split("kappa")
        assert hasattr(result.j15_provenance, "przz_line_numbers")
        assert result.j15_provenance.przz_line_numbers != ""

    def test_przz_lines_reference_a11(self):
        """PRZZ lines should reference A^{(1,1)} terms."""
        result = compute_plus5_signature_split("kappa")
        # Should mention line 1621-1628 or A^{(1,1)}
        lines = result.j15_provenance.przz_line_numbers.lower()
        assert "1621" in lines or "a^{(1,1)}" in lines.lower() or "a11" in lines


class TestProvenanceTruthSpecReference:
    """Test that provenance includes TRUTH_SPEC reference."""

    def test_provenance_has_truth_spec_ref(self):
        """Provenance should include TRUTH_SPEC.md reference."""
        result = compute_plus5_signature_split("kappa")
        assert hasattr(result.j15_provenance, "truth_spec_reference")
        assert result.j15_provenance.truth_spec_reference != ""

    def test_truth_spec_mentions_error_term(self):
        """TRUTH_SPEC reference should mention error term classification."""
        result = compute_plus5_signature_split("kappa")
        ref = result.j15_provenance.truth_spec_reference.lower()
        assert "error" in ref


class TestProvenanceStability:
    """Test that provenance is stable across runs."""

    def test_provenance_deterministic(self):
        """Same call should return same provenance."""
        result1 = compute_plus5_signature_split("kappa")
        result2 = compute_plus5_signature_split("kappa")

        assert result1.j15_provenance.source_module == result2.j15_provenance.source_module
        assert result1.j15_provenance.source_function == result2.j15_provenance.source_function
        assert result1.j15_provenance.przz_line_numbers == result2.j15_provenance.przz_line_numbers

    def test_provenance_same_across_benchmarks(self):
        """Provenance should be the same for both benchmarks (same code path)."""
        kappa = compute_plus5_signature_split("kappa")
        kappa_star = compute_plus5_signature_split("kappa_star")

        # Same code path, so same provenance
        assert kappa.j15_provenance.source_module == kappa_star.j15_provenance.source_module
        assert kappa.j15_provenance.source_function == kappa_star.j15_provenance.source_function


class TestProvenanceGuardrailStatus:
    """Test that provenance tracks guardrail status."""

    def test_provenance_has_guardrail_field(self):
        """Provenance should track guardrail status."""
        result = compute_plus5_signature_split("kappa")
        assert hasattr(result.j15_provenance, "passed_evaluation_mode_guardrails")
        assert isinstance(result.j15_provenance.passed_evaluation_mode_guardrails, bool)

    def test_provenance_has_guardrail_mode(self):
        """Provenance should track which guardrail mode was active."""
        result = compute_plus5_signature_split("kappa")
        assert hasattr(result.j15_provenance, "guardrail_mode")


class TestProvenanceFormulaDescription:
    """Test that provenance includes formula description."""

    def test_provenance_has_formula_description(self):
        """Provenance should include formula description."""
        result = compute_plus5_signature_split("kappa")
        assert hasattr(result.j15_provenance, "formula_description")
        assert result.j15_provenance.formula_description != ""

    def test_formula_description_mentions_j15(self):
        """Formula description should mention J15."""
        result = compute_plus5_signature_split("kappa")
        desc = result.j15_provenance.formula_description.lower()
        assert "j15" in desc or "j₁,₅" in desc.lower() or "j1,5" in desc


class TestProvenanceErrorClassification:
    """Test that provenance correctly classifies J15 as error-order."""

    def test_provenance_has_error_classification(self):
        """Provenance should indicate if term is error-order per spec."""
        result = compute_plus5_signature_split("kappa")
        assert hasattr(result.j15_provenance, "is_error_term_per_spec")
        assert isinstance(result.j15_provenance.is_error_term_per_spec, bool)

    def test_j15_classified_as_error_term(self):
        """J15 should be classified as error-order per TRUTH_SPEC."""
        result = compute_plus5_signature_split("kappa")
        assert result.j15_provenance.is_error_term_per_spec is True


class TestProvenanceReconciliationNotes:
    """Test that provenance includes reconciliation notes."""

    def test_provenance_has_reconciliation_notes(self):
        """Provenance should include reconciliation notes."""
        result = compute_plus5_signature_split("kappa")
        assert hasattr(result.j15_provenance, "reconciliation_notes")
        assert isinstance(result.j15_provenance.reconciliation_notes, list)

    def test_reconciliation_notes_not_empty(self):
        """Reconciliation notes should contain guidance."""
        result = compute_plus5_signature_split("kappa")
        assert len(result.j15_provenance.reconciliation_notes) > 0

    def test_reconciliation_notes_mention_j15_vs_i5(self):
        """Notes should mention J15 vs I5 reconciliation question."""
        result = compute_plus5_signature_split("kappa")
        notes_text = " ".join(result.j15_provenance.reconciliation_notes).lower()
        assert "j15" in notes_text or "i5" in notes_text


class TestProvenanceSerialization:
    """Test that provenance can be serialized."""

    def test_provenance_in_to_dict(self):
        """to_dict should include provenance."""
        result = compute_plus5_signature_split("kappa")
        d = result.to_dict()

        assert "j15_provenance" in d
        assert isinstance(d["j15_provenance"], dict)

    def test_provenance_dict_has_fields(self):
        """Serialized provenance should have all fields."""
        result = compute_plus5_signature_split("kappa")
        d = result.to_dict()
        prov = d["j15_provenance"]

        assert "source_module" in prov
        assert "source_function" in prov
        assert "przz_line_numbers" in prov
        assert "truth_spec_reference" in prov
        assert "is_error_term_per_spec" in prov


class TestDocumentation:
    """Document provenance findings for Phase 20.1."""

    def test_phase_20_1_provenance_summary(self):
        """
        PHASE 20.1 PROVENANCE SUMMARY:

        J15 Provenance documents:
        - Source: src.ratios.j1_euler_maclaurin:j15_contribution_integral
        - PRZZ Reference: Lines 1621-1628 (A^{(1,1)} terms)
        - TRUTH_SPEC: Explicitly classified as error-order
        - Guardrails: Not yet wired through evaluation_modes.py

        OPEN QUESTION: Is code's "J15" the same as TRUTH_SPEC's "I5"?
        This needs explicit reconciliation in J15_VS_I5_RECONCILIATION.md.
        """
        result = compute_plus5_signature_split("kappa")
        prov = result.j15_provenance

        # Verify key facts are documented
        assert "j1_euler_maclaurin" in prov.source_module
        assert prov.is_error_term_per_spec is True
        assert len(prov.reconciliation_notes) > 0
