"""
tests/test_mirror_transform.py
Sanity checks for “mirror-style” exp-factor transforms.

These tests do NOT assert any PRZZ benchmark truth. They only verify that our
diagnostic exp-factor transform is implemented consistently.
"""

import math

from src.evaluate import evaluate_c_full, evaluate_c_full_with_exp_transform
from src.polynomials import load_przz_polynomials


THETA = 4.0 / 7.0


def test_exp_scale_multiplier_1_matches_baseline_raw():
    """exp_scale_multiplier=1.0 must reproduce evaluate_c_full exactly."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    R = 1.3036
    n = 16

    base = evaluate_c_full(
        THETA, R, n, polys, return_breakdown=False, kernel_regime="raw"
    ).total
    transformed = evaluate_c_full_with_exp_transform(
        THETA,
        R,
        n,
        polys,
        kernel_regime="raw",
        exp_scale_multiplier=1.0,
        exp_t_flip=False,
        return_breakdown=False,
    ).total

    assert math.isclose(base, transformed, rel_tol=0.0, abs_tol=0.0)


def test_exp_scale_multiplier_minus_one_matches_R_flip_in_raw_regime():
    """In raw regime (ω=0 everywhere), flipping exp scales equals flipping R."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}
    R = 1.3036
    n = 16

    # Global R flip changes only ExpFactor scales in raw regime (P is ω=0).
    r_flip = evaluate_c_full(
        THETA, -R, n, polys, return_breakdown=False, kernel_regime="raw"
    ).total

    exp_flip = evaluate_c_full_with_exp_transform(
        THETA,
        R,
        n,
        polys,
        kernel_regime="raw",
        exp_scale_multiplier=-1.0,
        exp_t_flip=False,
        return_breakdown=False,
    ).total

    assert math.isclose(r_flip, exp_flip, rel_tol=1e-14, abs_tol=1e-14)

