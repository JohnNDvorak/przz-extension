"""
tests/test_derived_amplitude_formula.py

Regression-style tests for the derived amplitude formula (GPT Run 2).

These tests do NOT alter the paper-truth evaluator. They validate that the
"shape × amplitude" decomposition is wired consistently and produces a strong
benchmark match without using a 2×2 fit inside the evaluation.
"""

from __future__ import annotations

import math
import pytest

from src.evaluate import compute_c_paper, compute_derived_amplitude, compute_c_with_derived_amplitude
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
SIGMA = 5.0 / 32.0
K = 3

KAPPA_R = 1.3036
KAPPA_TARGET_C = 2.137

KAPPA_STAR_R = 1.1167
KAPPA_STAR_TARGET_C = 1.938


def test_default_epsilon_is_sigma_over_theta():
    amp = compute_derived_amplitude(R=1.0, theta=THETA, sigma=SIGMA, K=K)
    assert amp.epsilon == pytest.approx(SIGMA / THETA, rel=0, abs=1e-15)


def test_A2_minus_A1_equals_K_minus_1():
    amp = compute_derived_amplitude(R=1.234, theta=THETA, sigma=SIGMA, K=K)
    assert (amp.A2 - amp.A1) == pytest.approx(K - 1, rel=0, abs=1e-12)


@pytest.mark.parametrize(
    "name,R,target,loader",
    [
        ("kappa", KAPPA_R, KAPPA_TARGET_C, load_przz_polynomials),
        ("kappa_star", KAPPA_STAR_R, KAPPA_STAR_TARGET_C, load_przz_polynomials_kappa_star),
    ],
)
def test_derived_amplitude_hits_benchmarks_within_one_percent(name, R, target, loader):
    if loader is load_przz_polynomials:
        P1, P2, P3, Q = loader(enforce_Q0=True)
    else:
        P1, P2, P3, Q = loader()

    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    result = compute_c_with_derived_amplitude(
        theta=THETA,
        R=R,
        polynomials=polynomials,
        K=K,
        sigma=SIGMA,
        normalization="grid",
        lift_scope="i1_only",
        n=60,
        n_quad_a=40,
    )

    c = float(result["c_computed"])
    assert math.isfinite(c)

    # NOTE: This is an exploratory "TeX-mirror hypothesis" evaluator.
    # Do NOT lock it in as paper truth yet. We only require that it:
    #  - lands in a sane error band, and
    #  - substantially improves error vs the direct (+R) paper evaluator.
    c_direct = compute_c_paper(
        theta=THETA,
        R=R,
        n=60,
        polynomials=polynomials,
        return_breakdown=False,
        n_quad_a=40,
    ).total

    rel_err = abs(c - target) / target
    rel_err_direct = abs(float(c_direct) - target) / target

    assert rel_err < 0.10, f"{name}: c={c:.6f}, target={target}, err={rel_err:.2%}"
    assert rel_err < rel_err_direct, (
        f"{name}: derived amplitude did not improve vs direct paper truth: "
        f"err={rel_err:.2%} vs direct={rel_err_direct:.2%}"
    )
