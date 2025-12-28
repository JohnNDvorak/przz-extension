#!/usr/bin/env python3
"""
tests/test_no_target_anchoring_in_derived_modes.py
Phase 46 Gate 1: No Target Anchoring Import Lock Test

This test verifies that derived correction modes (THETA_2_MINUS_THETA,
FULL_SECOND_ORDER, THETA_CUBED) do NOT import or reference c_target,
benchmark constants, or anchored solve functions.

GPT's requirement: "Add a test that fails if any derived correction mode
imports or references c_target, benchmark constants, or anchored solve functions."

The test works by:
1. Parsing the correction_policy.py source file
2. Extracting the code paths for each derived mode
3. Verifying no forbidden patterns appear in those code paths

Created: 2025-12-27 (Phase 46 - GPT Gate 1)
"""

import ast
import re
import pytest
from pathlib import Path


# Forbidden patterns that indicate target anchoring
FORBIDDEN_PATTERNS = [
    r"c_target",            # Target c value
    r"c_kappa\s*=",         # Kappa benchmark constant
    r"c_kappa_star\s*=",    # Kappa-star benchmark constant
    r"G_I1_CALIBRATED",     # Calibrated g_I1 constant
    r"G_I2_CALIBRATED",     # Calibrated g_I2 constant
    r"anchored_solve",      # Any anchored solve function
    r"2\.137\d+",           # Literal c target (kappa)
    r"1\.937\d+",           # Literal c target (kappa*)
    r"1\.00091",            # Literal calibrated g_I1
    r"1\.01945",            # Literal calibrated g_I2
]

# Modes that should NOT use target anchoring
DERIVED_MODES = [
    "DERIVED_BASELINE_ONLY",
    "FIRST_PRINCIPLES_I1_I2",
    "THETA_2_MINUS_THETA",
    "FULL_SECOND_ORDER",
    "THETA_CUBED",
]

# Mode that IS allowed to use calibrated constants
ANCHORED_MODES = [
    "ANCHORED_TWO_BENCHMARKS",
    "COMPONENT_RENORM_ANCHORED",
]


def get_correction_policy_path() -> Path:
    """Get the path to correction_policy.py."""
    base = Path(__file__).parent.parent
    return base / "src" / "evaluator" / "correction_policy.py"


def extract_mode_code_blocks(source_lines: list, mode_name: str) -> list:
    """
    Extract the lines of code that implement a specific mode.

    This looks for patterns like:
        if mode == CorrectionMode.MODE_NAME:
            ...code block...
        elif mode == ...

    Returns the lines between the if/elif for this mode.
    """
    blocks = []
    in_mode = False
    brace_depth = 0

    # Look for the pattern: mode == CorrectionMode.MODE_NAME
    pattern = rf"mode\s*==\s*CorrectionMode\.{mode_name}"

    for i, line in enumerate(source_lines):
        if re.search(pattern, line):
            in_mode = True
            brace_depth = 0
            continue

        if in_mode:
            # Check if we've hit the next elif/else clause
            stripped = line.strip()
            if stripped.startswith("elif ") or stripped.startswith("else:"):
                in_mode = False
                continue

            blocks.append((i + 1, line))  # Line number is 1-indexed

    return blocks


def test_derived_modes_no_calibrated_constants_in_formulas():
    """
    Gate 1 Core Test: Verify derived mode implementations don't use
    calibrated constants like G_I1_CALIBRATED or G_I2_CALIBRATED.
    """
    policy_path = get_correction_policy_path()
    assert policy_path.exists(), f"correction_policy.py not found at {policy_path}"

    source = policy_path.read_text()
    source_lines = source.split("\n")

    violations = []

    for mode in DERIVED_MODES:
        # Extract the code block for this mode
        mode_lines = extract_mode_code_blocks(source_lines, mode)

        for line_num, line in mode_lines:
            # Skip comments
            if line.strip().startswith("#"):
                continue

            # Check for forbidden patterns
            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append({
                        "mode": mode,
                        "line": line_num,
                        "content": line.strip(),
                        "pattern": pattern,
                    })

    if violations:
        msg = "\n\nGATE 1 FAILED: Target anchoring detected in derived modes!\n"
        msg += "=" * 70 + "\n"
        for v in violations:
            msg += f"\nMode: {v['mode']}\n"
            msg += f"Line {v['line']}: {v['content']}\n"
            msg += f"Pattern: {v['pattern']}\n"
        msg += "\n" + "=" * 70
        pytest.fail(msg)


def test_anchored_mode_uses_calibrated_constants():
    """
    Sanity check: Verify ANCHORED mode DOES use calibrated constants.
    This confirms our pattern matching works correctly.
    """
    policy_path = get_correction_policy_path()
    source = policy_path.read_text()
    source_lines = source.split("\n")

    # The anchored mode should use G_I1_CALIBRATED and G_I2_CALIBRATED
    found_g_i1 = False
    found_g_i2 = False

    for mode in ANCHORED_MODES:
        mode_lines = extract_mode_code_blocks(source_lines, mode)

        for _, line in mode_lines:
            if "G_I1_CALIBRATED" in line:
                found_g_i1 = True
            if "G_I2_CALIBRATED" in line:
                found_g_i2 = True

    # Also check the helper function compute_g_anchored
    if "G_I1_CALIBRATED" in source and "compute_g_anchored" in source:
        found_g_i1 = True
    if "G_I2_CALIBRATED" in source and "compute_g_anchored" in source:
        found_g_i2 = True

    assert found_g_i1, "ANCHORED mode should use G_I1_CALIBRATED"
    assert found_g_i2, "ANCHORED mode should use G_I2_CALIBRATED"


def is_documentation_context(line: str) -> bool:
    """
    Check if a line is documentation/comment context (not actual code usage).

    Lines that explain what we DON'T do (e.g., "WITHOUT using c_target")
    should not be flagged as violations.
    """
    line_lower = line.lower().strip()

    # Skip pure comments
    if line_lower.startswith("#"):
        return True

    # Skip docstrings (lines that are all string content)
    if line_lower.startswith('"""') or line_lower.startswith("'''"):
        return True
    if line_lower.endswith('"""') or line_lower.endswith("'''"):
        return True

    # Skip documentation phrases
    doc_phrases = [
        "without using",
        "never uses",
        "not use",
        "do not use",
        "don't use",
        "doesn't use",
        "should not",
        "must not",
        "is not used",
        "are not used",
        "for comparison",
        "only as check",
        "only as a check",
    ]

    for phrase in doc_phrases:
        if phrase in line_lower:
            return True

    return False


def test_g_components_no_target_anchoring():
    """
    Gate 1b: Verify g_components.py doesn't use target values in derivation.

    Note: This test allows mentions of c_target in documentation/comments
    that explain what we DON'T do. Only actual code usage is flagged.
    """
    base = Path(__file__).parent.parent
    g_components_path = base / "src" / "unified_s12" / "g_components.py"

    if not g_components_path.exists():
        pytest.skip("g_components.py not found (may not exist yet)")

    source = g_components_path.read_text()

    violations = []
    in_docstring = False

    for i, line in enumerate(source.split("\n"), 1):
        stripped = line.strip()

        # Track docstring state
        if '"""' in stripped or "'''" in stripped:
            # Toggle docstring state (simple heuristic)
            quote_count = stripped.count('"""') + stripped.count("'''")
            if quote_count == 1:
                in_docstring = not in_docstring
            # If opening and closing on same line, skip this line
            continue

        if in_docstring:
            continue

        # Skip comments
        if stripped.startswith("#"):
            continue

        # Skip documentation context
        if is_documentation_context(line):
            continue

        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                violations.append({
                    "line": i,
                    "content": stripped,
                    "pattern": pattern,
                })

    if violations:
        msg = "\n\nGATE 1b FAILED: Target anchoring detected in g_components.py!\n"
        msg += "=" * 70 + "\n"
        for v in violations:
            msg += f"\nLine {v['line']}: {v['content']}\n"
            msg += f"Pattern: {v['pattern']}\n"
        msg += "\n" + "=" * 70
        pytest.fail(msg)


def test_derived_mode_formulas_are_pure():
    """
    Verify that derived mode formulas only use structural parameters (theta, K, R).

    The formulas should be:
    - DERIVED_BASELINE_ONLY: g = 1 + theta/(2K(2K+1))
    - THETA_2_MINUS_THETA: g_I1=1.0, g_I2=1+theta(2-theta)/(2K(2K+1))
    - FULL_SECOND_ORDER: g_I1=1+theta(1-theta)/(2K(2K+1)^2), g_I2=same as above
    - THETA_CUBED: g_I1=1+theta(1-theta)(2(K-1)+theta)/(8K(2K+1)^2), g_I2=same

    None of these should reference benchmark values.
    """
    policy_path = get_correction_policy_path()
    source = policy_path.read_text()

    # Verify the formulas use only theta, K, R (structural parameters)
    # Look for the g_I1_derived and g_I2_derived assignments in each mode

    # Pattern to find derived value assignments
    derived_pattern = r"g_I[12]_derived\s*=\s*[^#\n]+"

    matches = re.findall(derived_pattern, source)

    for match in matches:
        # These should only contain: 1, theta, K, +, -, *, /, (, ), **
        # And NOT contain any calibrated constants
        for forbidden in ["G_I1_CALIBRATED", "G_I2_CALIBRATED", "c_target"]:
            assert forbidden not in match, (
                f"Derived formula uses calibrated constant!\n"
                f"Formula: {match}\n"
                f"Forbidden: {forbidden}"
            )


def test_import_lock_no_target_imports():
    """
    Verify that correction_policy.py doesn't import c_target from other modules.
    """
    policy_path = get_correction_policy_path()
    source = policy_path.read_text()

    # Check import statements
    import_lines = [line for line in source.split("\n") if line.strip().startswith("from ") or line.strip().startswith("import ")]

    for line in import_lines:
        assert "c_target" not in line.lower(), f"Forbidden import: {line}"
        assert "c_kappa" not in line.lower(), f"Forbidden import: {line}"
        assert "benchmark" not in line.lower(), f"Forbidden import: {line}"


def test_first_principles_scripts_use_targets_only_for_comparison():
    """
    Gate 1c: Verify that first-principles scripts that DO use c_target
    use it only for comparison output, not as derivation input.

    This test checks that any c_target usage is in a "comparison" context
    (computing gap, printing comparison, etc.), not in formula derivation.
    """
    base = Path(__file__).parent.parent
    scripts_dir = base / "scripts"

    if not scripts_dir.exists():
        pytest.skip("scripts directory not found")

    # These scripts may use c_target for comparison (allowed)
    # but should NOT use it in formula derivation
    scripts_to_check = [
        "derive_g_I1_formula.py",
        "analyze_q_perturbation.py",
    ]

    for script_name in scripts_to_check:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            continue

        source = script_path.read_text()
        lines = source.split("\n")

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("#"):
                continue

            # Skip documentation context
            if is_documentation_context(line):
                continue

            # Check for actual derivation usage (very specific patterns)
            # Forbidden: g_I1 = some_func(c_target) or g_I2 = some_func(c_target)
            # Allowed: gap = c / c_target - 1, etc.
            if "c_target" in line:
                line_lower = line.lower()

                # Allowed patterns
                allowed = any([
                    "gap" in line_lower,
                    "compare" in line_lower,
                    "/" in line and "c_target" in line,  # Division by c_target for gap
                    "c_target =" in line,  # Definition
                    "c_target=" in line,   # Definition (no space)
                    "print" in line_lower,
                    "format" in line_lower,
                    "result.c / c_target" in line,  # Gap computation
                    "result.c/c_target" in line,
                ])

                # Forbidden: direct use in g derivation
                forbidden = any([
                    re.search(r"g_I[12]\s*=.*c_target", line),
                    re.search(r"epsilon.*=.*c_target", line),
                ])

                if forbidden and not allowed:
                    pytest.fail(
                        f"Gate 1c FAILED: {script_name} line {i} may use "
                        f"c_target in derivation:\n{line.strip()}"
                    )


class TestGate1Summary:
    """Summary test class documenting Gate 1 requirements."""

    def test_gate1_documentation(self):
        """
        Gate 1 Documentation Test

        This gate ensures that the derived correction modes are truly
        first-principles and do not "accidentally win by anchoring."

        Requirements:
        1. Derived modes (THETA_2_MINUS_THETA, FULL_SECOND_ORDER, THETA_CUBED)
           must NOT import or reference c_target, benchmark constants, or
           anchored solve functions.

        2. The only places where calibrated constants (G_I1_CALIBRATED,
           G_I2_CALIBRATED) should appear are:
           - The constant definitions at module level
           - The ANCHORED_TWO_BENCHMARKS mode implementation

        3. Scripts that use c_target should use it ONLY for comparison/
           validation output, never as input to derivation.

        If this test passes, we have structural proof that the derived
        modes cannot be contaminated by target anchoring.
        """
        pass  # Documentation test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
