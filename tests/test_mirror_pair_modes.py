"""
tests/test_mirror_pair_modes.py

Regression: mirror assembly supports both "hybrid" and fully "ordered" pair modes,
and exposes I₁/I₂(+R/-R) split totals for two-weight mirror diagnostics.
"""

from __future__ import annotations

import math

import pytest

from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167

# Keep this light; we only validate plumbing + invariants.
N_QUAD = 20
N_QUAD_A = 10


@pytest.fixture(scope="module")
def kappa_polys():
    return load_przz_polynomials(enforce_Q0=True)


@pytest.fixture(scope="module")
def kappa_star_polys():
    return load_przz_polynomials_kappa_star()


@pytest.mark.parametrize(
    "benchmark, polys_fixture, R",
    [
        ("kappa", "kappa_polys", KAPPA_R),
        ("kappa_star", "kappa_star_polys", KAPPA_STAR_R),
    ],
)
@pytest.mark.parametrize("pair_mode, expected_pair_mode, expected_s12_mode", [
    ("hybrid", "hybrid", "triangle"),
    ("ordered", "ordered", "ordered"),
])
def test_mirror_pair_mode_exposes_split_totals(
    request,
    benchmark: str,
    polys_fixture: str,
    R: float,
    pair_mode: str,
    expected_pair_mode: str,
    expected_s12_mode: str,
):
    P1, P2, P3, Q = request.getfixturevalue(polys_fixture)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    res = compute_c_paper_with_mirror(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        pair_mode=pair_mode,
        n_quad_a=N_QUAD_A,
    )

    per = res.per_term
    assert per.get("_pair_mode") == expected_pair_mode
    assert per.get("_s12_pair_mode") == expected_s12_mode

    required = [
        "_I1_plus_total",
        "_I1_minus_total",
        "_I2_plus_total",
        "_I2_minus_total",
        "_S12_plus_total",
        "_S12_minus_total",
        "_S34_plus_total",
        "_mirror_multiplier",
        "_direct_c",
        "_mirror_I12",
    ]
    for key in required:
        assert key in per, f"{benchmark}:{pair_mode}: missing per_term[{key}]"
        assert math.isfinite(float(per[key])), f"{benchmark}:{pair_mode}: per_term[{key}] not finite: {per[key]}"

    # Internal identities (should hold up to roundoff).
    s12_plus = float(per["_S12_plus_total"])
    s12_minus = float(per["_S12_minus_total"])
    i1_plus = float(per["_I1_plus_total"])
    i1_minus = float(per["_I1_minus_total"])
    i2_plus = float(per["_I2_plus_total"])
    i2_minus = float(per["_I2_minus_total"])

    assert abs(s12_plus - (i1_plus + i2_plus)) < 1e-10, (
        f"{benchmark}:{pair_mode}: S12(+R) split mismatch"
    )
    assert abs(s12_minus - (i1_minus + i2_minus)) < 1e-10, (
        f"{benchmark}:{pair_mode}: S12(-R) split mismatch"
    )
    assert abs(float(per["_mirror_I12"]) - s12_minus) < 1e-12, (
        f"{benchmark}:{pair_mode}: _mirror_I12 must equal S12(-R)"
    )

