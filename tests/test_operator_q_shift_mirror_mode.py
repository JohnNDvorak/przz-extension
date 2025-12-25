"""
tests/test_operator_q_shift_mirror_mode.py

Structural guardrails for the feature-flagged "operator_q_shift" mirror mode.

These tests intentionally avoid asserting any specific benchmark-derived
coefficients (m, m1, m2). They only enforce wiring invariants:
1) The +R (direct) branch is unchanged by operator_q_shift
2) The mirror (-R) branch is actually affected (sanity check)
"""

from __future__ import annotations

import pytest

from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import Polynomial


def test_operator_q_shift_does_not_change_direct_branch():
    theta = 4.0 / 7.0
    R = 0.9

    # Keep polynomials simple to make this test cheap and deterministic.
    polynomials = {
        "P1": Polynomial([1.0]),
        "P2": Polynomial([1.0]),
        "P3": Polynomial([1.0]),
        "Q": Polynomial([1.0, 0.25, -0.1]),
    }

    base = compute_c_paper_with_mirror(
        theta=theta,
        R=R,
        n=12,
        polynomials=polynomials,
        pair_mode="ordered",
        n_quad_a=6,
        mirror_mode="empirical_scalar",
    ).per_term

    op = compute_c_paper_with_mirror(
        theta=theta,
        R=R,
        n=12,
        polynomials=polynomials,
        pair_mode="ordered",
        n_quad_a=6,
        mirror_mode="operator_q_shift",
    ).per_term

    assert base is not None and op is not None
    assert base["_mirror_q_poly_shift"] == 0.0
    assert op["_mirror_q_poly_shift"] == 1.0

    # Direct (+R) channel totals must be identical.
    for key in [
        "_I1_plus_total",
        "_I2_plus_total",
        "_S12_plus_total",
        "_S34_plus_total",
        "_direct_c",
    ]:
        assert float(op[key]) == pytest.approx(float(base[key]), rel=0.0, abs=1e-12), key

    # Mirror (-R) channels should change (sanity check that the flag is wired).
    assert abs(float(op["_S12_minus_total"]) - float(base["_S12_minus_total"])) > 1e-9
