"""
src/evaluator/__init__.py
Modular evaluation package for PRZZ computation.

This package was extracted from the monolithic evaluate.py (6,700+ lines)
to improve maintainability while preserving backwards compatibility.

Phase 20.4: Safe evaluate.py Refactoring
Created: 2025-12-24

REFACTORING RULES:
==================
1. Every move must pass tests/test_evaluate_snapshots.py
2. Keep original function signatures in evaluate.py
3. Add deprecation warnings for direct evaluate.py imports
4. Document each extracted module

PACKAGE STRUCTURE:
==================
- result_types.py: Core dataclasses (TermResult, EvaluationResult, etc.)
- solver_utils.py: Two-weight operator solving utilities
- diagnostics.py: Reporting and diagnostic utilities
- facade.py: Re-exports for backwards compatibility
"""

from src.evaluator.result_types import (
    TermResult,
    EvaluationResult,
    DiagnosticResult,
    S34OrderedPairsError,
    I34MirrorForbiddenError,
    get_s34_triangle_pairs,
    get_s34_factorial_normalization,
    assert_s34_triangle_convention,
    assert_i34_no_mirror,
)

__all__ = [
    # Result types
    "TermResult",
    "EvaluationResult",
    "DiagnosticResult",
    # Error classes
    "S34OrderedPairsError",
    "I34MirrorForbiddenError",
    # Helper functions
    "get_s34_triangle_pairs",
    "get_s34_factorial_normalization",
    "assert_s34_triangle_convention",
    "assert_i34_no_mirror",
]
