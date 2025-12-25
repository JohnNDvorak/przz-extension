"""
tests/test_tex_parity_affine_args.py

TeX-parity guardrail: affine argument wiring for I₁/I₃/I₄.

This test asserts that the Term generators encode the affine arguments
appearing in PRZZ TeX for the core Q/exp factors. It intentionally does NOT
assert any numerical benchmark values.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest

from src.term_dsl import AffineExpr, ExpFactor, PolyFactor, Term
from src.terms_k3_d1 import make_I1_11, make_I3_11, make_I4_11


def _grid() -> Tuple[np.ndarray, np.ndarray]:
    # u is irrelevant for these affine args, but AffineExpr expects both.
    U = np.array([[0.2, 0.7], [0.4, 0.9]], dtype=float)
    T = np.array([[0.1, 0.25], [0.6, 0.85]], dtype=float)
    return U, T


def _q_poly_args(term: Term) -> List[AffineExpr]:
    return [pf.argument for pf in term.poly_factors if pf.poly_name == "Q"]


def _exp_args(term: Term) -> List[AffineExpr]:
    return [ef.argument for ef in term.exp_factors]


def _exp_scales(term: Term) -> List[float]:
    return [float(ef.scale) for ef in term.exp_factors]


def _affine_close(
    expr: AffineExpr,
    *,
    theta: float,
    U: np.ndarray,
    T: np.ndarray,
    expected_a0: Callable[[np.ndarray, np.ndarray], np.ndarray],
    expected_coeffs: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    atol: float = 1e-12,
) -> bool:
    try:
        np.testing.assert_allclose(expr.evaluate_a0(U, T), expected_a0(U, T), atol=atol, rtol=0.0)
        if set(expr.var_coeffs.keys()) != set(expected_coeffs.keys()):
            return False
        for var, expected in expected_coeffs.items():
            np.testing.assert_allclose(expr.evaluate_coeff(var, U, T), expected(U, T), atol=atol, rtol=0.0)
    except AssertionError:
        return False
    return True


def _assert_contains_exactly_one(
    exprs: List[AffineExpr],
    *,
    matcher: Callable[[AffineExpr], bool],
    label: str,
) -> None:
    hits = [matcher(e) for e in exprs]
    assert sum(bool(h) for h in hits) == 1, f"Expected exactly one match for {label}, got {hits}"


def test_i1_11_q_and_exp_affine_args_match_tex() -> None:
    theta = 4.0 / 7.0
    R = 1.3036
    term = make_I1_11(theta, R, kernel_regime="paper")

    U, T = _grid()

    q_args = _q_poly_args(term)
    exp_args = _exp_args(term)

    assert len(q_args) == 2
    assert len(exp_args) == 2
    assert _exp_scales(term) == pytest.approx([R, R], abs=0.0)

    # TeX: a1 = t + θt·x + θ(t-1)·y, a2 = t + θ(t-1)·x + θt·y
    def a0(u: np.ndarray, t: np.ndarray) -> np.ndarray:
        return t

    alpha = {
        "x1": lambda u, t, th=theta: th * t,
        "y1": lambda u, t, th=theta: th * t - th,
    }
    beta = {
        "x1": lambda u, t, th=theta: th * t - th,
        "y1": lambda u, t, th=theta: th * t,
    }

    is_alpha = lambda e: _affine_close(e, theta=theta, U=U, T=T, expected_a0=a0, expected_coeffs=alpha)
    is_beta = lambda e: _affine_close(e, theta=theta, U=U, T=T, expected_a0=a0, expected_coeffs=beta)

    _assert_contains_exactly_one(q_args, matcher=is_alpha, label="I1_11 Q alpha")
    _assert_contains_exactly_one(q_args, matcher=is_beta, label="I1_11 Q beta")
    _assert_contains_exactly_one(exp_args, matcher=is_alpha, label="I1_11 exp alpha")
    _assert_contains_exactly_one(exp_args, matcher=is_beta, label="I1_11 exp beta")


def test_i3_11_q_and_exp_affine_args_match_tex() -> None:
    theta = 4.0 / 7.0
    R = 1.3036
    term = make_I3_11(theta, R, kernel_regime="paper")

    U, T = _grid()

    q_args = _q_poly_args(term)
    exp_args = _exp_args(term)

    assert len(q_args) == 2
    assert len(exp_args) == 2
    assert _exp_scales(term) == pytest.approx([R, R], abs=0.0)

    # TeX: t + θt·x  and  t + θ(t-1)·x = t + (θt-θ)·x
    def a0(u: np.ndarray, t: np.ndarray) -> np.ndarray:
        return t

    alpha = {"x1": lambda u, t, th=theta: th * t}
    beta = {"x1": lambda u, t, th=theta: th * t - th}

    is_alpha = lambda e: _affine_close(e, theta=theta, U=U, T=T, expected_a0=a0, expected_coeffs=alpha)
    is_beta = lambda e: _affine_close(e, theta=theta, U=U, T=T, expected_a0=a0, expected_coeffs=beta)

    _assert_contains_exactly_one(q_args, matcher=is_alpha, label="I3_11 Q alpha")
    _assert_contains_exactly_one(q_args, matcher=is_beta, label="I3_11 Q beta")
    _assert_contains_exactly_one(exp_args, matcher=is_alpha, label="I3_11 exp alpha")
    _assert_contains_exactly_one(exp_args, matcher=is_beta, label="I3_11 exp beta")


def test_i4_11_q_and_exp_affine_args_match_tex() -> None:
    theta = 4.0 / 7.0
    R = 1.3036
    term = make_I4_11(theta, R, kernel_regime="paper")

    U, T = _grid()

    q_args = _q_poly_args(term)
    exp_args = _exp_args(term)

    assert len(q_args) == 2
    assert len(exp_args) == 2
    assert _exp_scales(term) == pytest.approx([R, R], abs=0.0)

    # TeX y-analogue: t + θ(t-1)·y  and  t + θt·y
    def a0(u: np.ndarray, t: np.ndarray) -> np.ndarray:
        return t

    alpha = {"y1": lambda u, t, th=theta: th * t - th}
    beta = {"y1": lambda u, t, th=theta: th * t}

    is_alpha = lambda e: _affine_close(e, theta=theta, U=U, T=T, expected_a0=a0, expected_coeffs=alpha)
    is_beta = lambda e: _affine_close(e, theta=theta, U=U, T=T, expected_a0=a0, expected_coeffs=beta)

    _assert_contains_exactly_one(q_args, matcher=is_alpha, label="I4_11 Q alpha")
    _assert_contains_exactly_one(q_args, matcher=is_beta, label="I4_11 Q beta")
    _assert_contains_exactly_one(exp_args, matcher=is_alpha, label="I4_11 exp alpha")
    _assert_contains_exactly_one(exp_args, matcher=is_beta, label="I4_11 exp beta")

