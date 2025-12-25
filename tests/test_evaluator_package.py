"""
tests/test_evaluator_package.py
Tests for the src/evaluator package structure.

Phase 20.4: Safe evaluate.py Refactoring
Created: 2025-12-24

PURPOSE:
========
Verify that:
1. The evaluator package can be imported
2. All exported types are available
3. Backwards compatibility is maintained (imports from evaluate.py still work)
4. Spec lock functions raise correctly

USAGE:
======
    pytest tests/test_evaluator_package.py -v
"""

import pytest


class TestEvaluatorPackageImports:
    """Test that evaluator package imports work."""

    def test_import_package(self):
        """Should be able to import the evaluator package."""
        from src import evaluator
        assert evaluator is not None

    def test_import_term_result(self):
        """Should be able to import TermResult from package."""
        from src.evaluator import TermResult
        assert TermResult is not None

    def test_import_evaluation_result(self):
        """Should be able to import EvaluationResult from package."""
        from src.evaluator import EvaluationResult
        assert EvaluationResult is not None

    def test_import_error_classes(self):
        """Should be able to import error classes from package."""
        from src.evaluator import S34OrderedPairsError, I34MirrorForbiddenError
        assert issubclass(S34OrderedPairsError, ValueError)
        assert issubclass(I34MirrorForbiddenError, ValueError)

    def test_import_helper_functions(self):
        """Should be able to import helper functions from package."""
        from src.evaluator import (
            get_s34_triangle_pairs,
            get_s34_factorial_normalization,
            assert_s34_triangle_convention,
            assert_i34_no_mirror,
        )
        assert callable(get_s34_triangle_pairs)
        assert callable(get_s34_factorial_normalization)
        assert callable(assert_s34_triangle_convention)
        assert callable(assert_i34_no_mirror)


class TestBackwardsCompatibility:
    """Test that imports from evaluate.py still work."""

    def test_import_term_result_from_evaluate(self):
        """TermResult should still be importable from evaluate.py."""
        from src.evaluate import TermResult
        assert TermResult is not None

    def test_import_evaluation_result_from_evaluate(self):
        """EvaluationResult should still be importable from evaluate.py."""
        from src.evaluate import EvaluationResult
        assert EvaluationResult is not None

    def test_import_error_classes_from_evaluate(self):
        """Error classes should still be importable from evaluate.py."""
        from src.evaluate import S34OrderedPairsError, I34MirrorForbiddenError
        assert issubclass(S34OrderedPairsError, ValueError)
        assert issubclass(I34MirrorForbiddenError, ValueError)

    def test_import_helper_functions_from_evaluate(self):
        """Helper functions should still be importable from evaluate.py."""
        from src.evaluate import (
            get_s34_triangle_pairs,
            get_s34_factorial_normalization,
        )
        assert callable(get_s34_triangle_pairs)
        assert callable(get_s34_factorial_normalization)

    def test_types_are_identical(self):
        """Types from both sources should be identical."""
        from src.evaluate import TermResult as TermResultEval
        from src.evaluator import TermResult as TermResultPkg
        assert TermResultEval is TermResultPkg

        from src.evaluate import EvaluationResult as EvalResultEval
        from src.evaluator import EvaluationResult as EvalResultPkg
        assert EvalResultEval is EvalResultPkg


class TestResultTypeConstruction:
    """Test that result types can be instantiated."""

    def test_term_result_construction(self):
        """Should be able to construct TermResult."""
        from src.evaluator import TermResult
        result = TermResult(name="test", value=1.0)
        assert result.name == "test"
        assert result.value == 1.0
        assert result.extracted_coeff_sample is None
        assert result.series_term_count is None

    def test_term_result_with_optional_fields(self):
        """Should be able to construct TermResult with optional fields."""
        from src.evaluator import TermResult
        result = TermResult(
            name="test",
            value=1.0,
            extracted_coeff_sample=0.5,
            series_term_count=10,
        )
        assert result.extracted_coeff_sample == 0.5
        assert result.series_term_count == 10

    def test_evaluation_result_construction(self):
        """Should be able to construct EvaluationResult."""
        from src.evaluator import EvaluationResult
        result = EvaluationResult(
            total=2.0,
            per_term={"a": 1.0, "b": 1.0},
            n=60,
        )
        assert result.total == 2.0
        assert result.per_term == {"a": 1.0, "b": 1.0}
        assert result.n == 60
        assert result.term_results is None


class TestS34TrianglePairs:
    """Test S34 triangle pair functions."""

    def test_get_triangle_pairs_returns_six(self):
        """Should return exactly 6 triangle pairs."""
        from src.evaluator import get_s34_triangle_pairs
        pairs = get_s34_triangle_pairs()
        assert len(pairs) == 6

    def test_get_triangle_pairs_contains_expected(self):
        """Should contain expected pair keys."""
        from src.evaluator import get_s34_triangle_pairs
        pairs = get_s34_triangle_pairs()
        keys = [p[0] for p in pairs]
        assert "11" in keys
        assert "22" in keys
        assert "33" in keys
        assert "12" in keys
        assert "13" in keys
        assert "23" in keys

    def test_diagonal_pairs_have_factor_1(self):
        """Diagonal pairs should have symmetry factor 1."""
        from src.evaluator import get_s34_triangle_pairs
        pairs = dict(get_s34_triangle_pairs())
        assert pairs["11"] == 1
        assert pairs["22"] == 1
        assert pairs["33"] == 1

    def test_off_diagonal_pairs_have_factor_2(self):
        """Off-diagonal pairs should have symmetry factor 2."""
        from src.evaluator import get_s34_triangle_pairs
        pairs = dict(get_s34_triangle_pairs())
        assert pairs["12"] == 2
        assert pairs["13"] == 2
        assert pairs["23"] == 2

    def test_factorial_normalization_values(self):
        """Factorial normalization should have correct values."""
        from src.evaluator import get_s34_factorial_normalization
        norms = get_s34_factorial_normalization()
        assert norms["11"] == 1.0
        assert norms["22"] == 0.25
        assert abs(norms["33"] - 1/36) < 1e-10


class TestSpecLockGuards:
    """Test spec lock guard functions raise correctly."""

    def test_assert_s34_triangle_convention_accepts_triangle(self):
        """Should accept valid triangle pair keys."""
        from src.evaluator import assert_s34_triangle_convention
        # Should not raise
        assert_s34_triangle_convention(["11", "22", "33", "12", "13", "23"])

    def test_assert_s34_triangle_convention_rejects_ordered(self):
        """Should reject ordered pair keys."""
        from src.evaluator import (
            assert_s34_triangle_convention,
            S34OrderedPairsError,
        )
        with pytest.raises(S34OrderedPairsError):
            assert_s34_triangle_convention(["11", "21", "12"])

    def test_assert_i34_no_mirror_accepts_false(self):
        """Should accept apply_mirror=False."""
        from src.evaluator import assert_i34_no_mirror
        # Should not raise
        assert_i34_no_mirror(False)

    def test_assert_i34_no_mirror_rejects_true(self):
        """Should reject apply_mirror=True."""
        from src.evaluator import assert_i34_no_mirror, I34MirrorForbiddenError
        with pytest.raises(I34MirrorForbiddenError):
            assert_i34_no_mirror(True)
