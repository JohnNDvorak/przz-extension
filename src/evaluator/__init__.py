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

from src.evaluator.s12_backend import (
    I1Result,
    I2Result,
    S12PairResult,
    S12AllPairsResult,
    BackendComparisonResult,
    compute_I1_backend,
    compute_I2_backend,
    compute_S12_pair,
    compute_S12_all_pairs,
    compare_backends_pair,
    compare_backends_all_pairs,
    assert_backends_equivalent,
    get_s12_factorial_normalization,
    get_s12_symmetry_factors,
)

# NOTE: gap_attribution is NOT imported at module level to avoid circular imports
# Import it directly when needed: from src.evaluator.gap_attribution import ...

__all__ = [
    # Result types
    "TermResult",
    "EvaluationResult",
    "DiagnosticResult",
    # S12 backend result types
    "I1Result",
    "I2Result",
    "S12PairResult",
    "S12AllPairsResult",
    "BackendComparisonResult",
    # Error classes
    "S34OrderedPairsError",
    "I34MirrorForbiddenError",
    # S34 helper functions
    "get_s34_triangle_pairs",
    "get_s34_factorial_normalization",
    "assert_s34_triangle_convention",
    "assert_i34_no_mirror",
    # S12 backend functions
    "compute_I1_backend",
    "compute_I2_backend",
    "compute_S12_pair",
    "compute_S12_all_pairs",
    "compare_backends_pair",
    "compare_backends_all_pairs",
    "assert_backends_equivalent",
    "get_s12_factorial_normalization",
    "get_s12_symmetry_factors",
]
