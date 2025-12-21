"""
tests/test_i2_source_gate.py
Gate tests: i2_source="direct_case_c" must match DSL output.

Run 8A: This test verifies that the proven Case C kernel evaluation
produces the SAME I2 values as the DSL-based evaluation within 2% tolerance.

This is a REGRESSION GATE - if this test fails, something is broken.

Usage:
    pytest tests/test_i2_source_gate.py -v
    pytest tests/test_i2_source_gate.py -v -m calibration
"""

import pytest
import math
from typing import Dict

from src.polynomials import (
    load_przz_polynomials,
    load_przz_polynomials_kappa_star,
)
from src.evaluate import compute_operator_implied_weights


THETA = 4.0 / 7.0
TOLERANCE = 0.02  # 2% tolerance


@pytest.mark.calibration
class TestI2SourceGate:
    """Gate tests: i2_source='direct_case_c' must match DSL output."""

    @pytest.fixture
    def polys_kappa(self):
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    @pytest.fixture
    def polys_kappa_star(self):
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()
        return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    def test_i2_plus_equivalence_kappa(self, polys_kappa):
        """Direct Case C I2_plus must match DSL at κ benchmark."""
        R = 1.3036

        result_dsl = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="dsl"
        )
        result_direct = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="direct_case_c"
        )

        ratio = result_direct.I2_plus / result_dsl.I2_plus
        assert abs(ratio - 1.0) < TOLERANCE, \
            f"κ I2_plus: ratio={ratio:.6f}, expected ≈1.0 (DSL={result_dsl.I2_plus:.6f}, direct={result_direct.I2_plus:.6f})"

    def test_i2_minus_base_equivalence_kappa(self, polys_kappa):
        """Direct Case C I2_minus_base must match DSL at κ benchmark."""
        R = 1.3036

        result_dsl = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="dsl"
        )
        result_direct = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="direct_case_c"
        )

        ratio = result_direct.I2_minus_base / result_dsl.I2_minus_base
        assert abs(ratio - 1.0) < TOLERANCE, \
            f"κ I2_minus_base: ratio={ratio:.6f}, expected ≈1.0"

    def test_i2_plus_equivalence_kappa_star(self, polys_kappa_star):
        """Direct Case C I2_plus must match DSL at κ* benchmark."""
        R = 1.1167

        result_dsl = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa_star,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="dsl"
        )
        result_direct = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa_star,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="direct_case_c"
        )

        ratio = result_direct.I2_plus / result_dsl.I2_plus
        assert abs(ratio - 1.0) < TOLERANCE, \
            f"κ* I2_plus: ratio={ratio:.6f}, expected ≈1.0"

    def test_i2_minus_base_equivalence_kappa_star(self, polys_kappa_star):
        """Direct Case C I2_minus_base must match DSL at κ* benchmark."""
        R = 1.1167

        result_dsl = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa_star,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="dsl"
        )
        result_direct = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa_star,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="direct_case_c"
        )

        ratio = result_direct.I2_minus_base / result_dsl.I2_minus_base
        assert abs(ratio - 1.0) < TOLERANCE, \
            f"κ* I2_minus_base: ratio={ratio:.6f}, expected ≈1.0"

    def test_per_pair_equivalence_kappa(self, polys_kappa):
        """Each pair's direct I2 should match DSL within tolerance at κ benchmark."""
        R = 1.3036

        result_dsl = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="dsl"
        )
        result_direct = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="direct_case_c"
        )

        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            dsl_plus = result_dsl.pair_breakdown[pair_key]["I2_plus"]
            direct_plus = result_direct.pair_breakdown[pair_key]["I2_plus"]

            if abs(dsl_plus) > 1e-10:
                ratio = direct_plus / dsl_plus
                assert abs(ratio - 1.0) < TOLERANCE, \
                    f"κ pair {pair_key} I2_plus: ratio={ratio:.6f}, expected ≈1.0"

            dsl_minus = result_dsl.pair_breakdown[pair_key]["I2_minus_base"]
            direct_minus = result_direct.pair_breakdown[pair_key]["I2_minus_base"]

            if abs(dsl_minus) > 1e-10:
                ratio = direct_minus / dsl_minus
                assert abs(ratio - 1.0) < TOLERANCE, \
                    f"κ pair {pair_key} I2_minus_base: ratio={ratio:.6f}, expected ≈1.0"

    def test_c_operator_equivalence_kappa(self, polys_kappa):
        """c_operator should be similar between DSL and direct at κ benchmark."""
        R = 1.3036

        result_dsl = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="dsl"
        )
        result_direct = compute_operator_implied_weights(
            theta=THETA, R=R, polynomials=polys_kappa,
            sigma=5/32, normalization="grid", lift_scope="i1_only",
            n=60, n_quad_a=40,
            i2_source="direct_case_c"
        )

        # c_operator uses I2_minus_op which equals I2_minus_base for direct_case_c
        # and for lift_scope="i1_only" with DSL as well.
        # So they should match.
        ratio = result_direct.c_operator / result_dsl.c_operator
        assert abs(ratio - 1.0) < TOLERANCE, \
            f"κ c_operator: ratio={ratio:.6f}, expected ≈1.0"
