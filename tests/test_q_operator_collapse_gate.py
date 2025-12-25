"""
tests/test_q_operator_collapse_gate.py

Stage C2b gate: "operator q-shift" wiring sanity check.

This is intentionally integration-adjacent (runs one cheap DSL term), but it is
NOT a PRZZ benchmark or target-matching test.

We verify that two equivalent implementations of the Q(1+D) shift agree:
1) Shift only Q(...) affine argument constants by +1 (a0 += 1)
2) Replace Q(x) with the lifted polynomial Q(x+1) and keep arguments unchanged
"""

from __future__ import annotations

import pytest

from src.evaluate import evaluate_term
from src.mirror_transform import transform_term_q_factors
from src.polynomials import Polynomial
from src.q_operator import lift_poly_by_shift
from src.terms_k3_d1 import make_I2_11


def test_q_argument_shift_matches_lifted_Q_polynomial():
    theta = 4.0 / 7.0
    R = 0.73
    n = 16

    # Keep P constant so the integrand is "as diagonal as possible" with respect
    # to Q's argument handling, and keep the test cheap and deterministic.
    P1 = Polynomial([1.0])
    Q = Polynomial([1.0, -0.3, 0.2, -0.05])

    term = make_I2_11(theta, R, kernel_regime="paper")

    # Path A: shift the Q(...) arguments by +1 (mirror operator identity)
    term_shifted = transform_term_q_factors(term, q_a0_shift=1.0)
    val_shifted = evaluate_term(
        term_shifted,
        {"P1": P1, "Q": Q},
        n,
        R=R,
        theta=theta,
        n_quad_a=4,
    ).value

    # Path B: lift Q to Q(x+1) and keep arguments unchanged
    Q_lift = lift_poly_by_shift(Q, shift=1.0)
    val_lifted = evaluate_term(
        term,
        {"P1": P1, "Q": Q_lift},
        n,
        R=R,
        theta=theta,
        n_quad_a=4,
    ).value

    assert val_shifted == pytest.approx(val_lifted, rel=1e-12, abs=1e-12)

