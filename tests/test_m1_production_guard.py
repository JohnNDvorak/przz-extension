"""
tests/test_m1_production_guard.py
Phase 10.0: Production Guard for DIAGNOSTIC_FITTED m₁

This test ensures that NO production code path uses the DIAGNOSTIC_FITTED mode,
which is calibration creep and masks the underlying derivation problem.

The DIAGNOSTIC_FITTED formula (m₁ = 1.037*exp(R) + 5) achieves 0% gap but is
NOT derived from first principles. Using it in production prevents discovery
of the correct derived mirror operator.
"""

import pytest
import ast
import os
from pathlib import Path


# Production source files that must NEVER use DIAGNOSTIC_FITTED
PRODUCTION_SOURCE_FILES = [
    "src/evaluate.py",
    "src/terms_k3_d1.py",
    "src/term_dsl.py",
    "src/polynomials.py",
    "src/quadrature.py",
    "src/series.py",
    "src/composition.py",
    "src/kernel_registry.py",
]


def get_project_root() -> Path:
    """Get the przz-extension project root."""
    return Path(__file__).parent.parent


class DiagnosticModeVisitor(ast.NodeVisitor):
    """AST visitor that detects usage of DIAGNOSTIC_FITTED mode."""

    def __init__(self):
        self.violations = []

    def visit_Attribute(self, node):
        # Check for M1Mode.DIAGNOSTIC_FITTED
        if (
            isinstance(node.attr, str)
            and node.attr == "DIAGNOSTIC_FITTED"
        ):
            self.violations.append(
                f"Line {node.lineno}: Reference to DIAGNOSTIC_FITTED"
            )
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check for m1_diagnostic_fitted() function calls
        if isinstance(node.func, ast.Name):
            if node.func.id == "m1_diagnostic_fitted":
                self.violations.append(
                    f"Line {node.lineno}: Call to m1_diagnostic_fitted()"
                )
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr == "m1_diagnostic_fitted":
                self.violations.append(
                    f"Line {node.lineno}: Call to m1_diagnostic_fitted()"
                )
        self.generic_visit(node)


@pytest.mark.parametrize("source_file", PRODUCTION_SOURCE_FILES)
def test_no_diagnostic_fitted_in_production(source_file: str):
    """
    Verify that production source files do NOT use DIAGNOSTIC_FITTED mode.

    This is a HARD GUARD against calibration creep. The fitted m₁ formula
    achieves 0% gap but is NOT derived from first principles.
    """
    project_root = get_project_root()
    file_path = project_root / source_file

    if not file_path.exists():
        pytest.skip(f"{source_file} does not exist")

    source_code = file_path.read_text()
    tree = ast.parse(source_code)

    visitor = DiagnosticModeVisitor()
    visitor.visit(tree)

    if visitor.violations:
        violation_details = "\n".join(visitor.violations)
        pytest.fail(
            f"CALIBRATION CREEP DETECTED in {source_file}!\n"
            f"Production code must NOT use DIAGNOSTIC_FITTED mode.\n"
            f"Violations:\n{violation_details}\n\n"
            f"Use M1Mode.K3_EMPIRICAL or implement derived mirror operator instead."
        )


def test_diagnostic_fitted_requires_explicit_opt_in():
    """
    Verify that DIAGNOSTIC_FITTED mode raises without allow_diagnostic=True.

    This ensures the quarantine is enforced at runtime.
    """
    from src.m1_policy import M1Policy, M1Mode, m1_formula, M1DiagnosticError

    policy_without_opt_in = M1Policy(mode=M1Mode.DIAGNOSTIC_FITTED)

    with pytest.raises(M1DiagnosticError) as exc_info:
        m1_formula(K=3, R=1.3036, policy=policy_without_opt_in)

    assert "allow_diagnostic=True" in str(exc_info.value)
    assert "CALIBRATION CREEP" in str(exc_info.value)


def test_diagnostic_fitted_works_with_opt_in():
    """
    Verify that DIAGNOSTIC_FITTED mode works when explicitly enabled.

    This is for diagnostic scripts only, NOT production.
    """
    import warnings
    from src.m1_policy import M1Policy, M1Mode, m1_formula

    policy_with_opt_in = M1Policy(
        mode=M1Mode.DIAGNOSTIC_FITTED,
        allow_diagnostic=True
    )

    # Should emit warning but not raise
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m1 = m1_formula(K=3, R=1.3036, policy=policy_with_opt_in)
        assert len(w) == 1
        assert "DIAGNOSTIC_FITTED" in str(w[0].message)
        assert "NOT derived" in str(w[0].message)

    # Should return the fitted value
    import numpy as np
    expected = 1.037353 * np.exp(1.3036) + 4.993849
    assert abs(m1 - expected) < 1e-6
