"""
tests/test_triangle_ordered_equivalence.py

Priority A validation:
Compare triangle×2 (upper-triangle with symmetry factor 2) against ordered-sum
over all 9 ordered pairs (no symmetry factor).

Test staging:
- Must-pass: ordered evaluator and comparison report produce finite results.
"""

from __future__ import annotations

import math

import pytest

from src.evaluate import evaluate_c_ordered, compare_triangle_vs_ordered
from src.evaluate import compute_c_paper
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167

# Keep these moderate so the suite stays responsive; the comparison is structural,
# not a convergence test.
N_QUAD = 40
N_QUAD_A = 30


@pytest.fixture(scope="module")
def kappa_polys():
    return load_przz_polynomials(enforce_Q0=True)


@pytest.fixture(scope="module")
def kappa_star_polys():
    return load_przz_polynomials_kappa_star()


BENCHMARKS = [
    ("kappa", "kappa_polys", KAPPA_R),
    ("kappa_star", "kappa_star_polys", KAPPA_STAR_R),
]


def _assert_finite(x: float, label: str) -> None:
    assert math.isfinite(x), f"{label} is not finite: {x}"


@pytest.mark.parametrize("benchmark, polys_fixture, R", BENCHMARKS)
@pytest.mark.parametrize("kernel_regime", ["raw", "paper"])
def test_ordered_evaluator_finite(request, benchmark: str, polys_fixture: str, R: float, kernel_regime: str):
    P1, P2, P3, Q = request.getfixturevalue(polys_fixture)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    result = evaluate_c_ordered(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        kernel_regime=kernel_regime,
        n_quad_a=N_QUAD_A,
    )

    _assert_finite(float(result.total), f"{benchmark}:{kernel_regime}:total")

    pair_raw = result.per_term.get("_ordered_pair_raw")
    assert isinstance(pair_raw, dict), "Expected _ordered_pair_raw dict in per_term"
    for pair_key, value in pair_raw.items():
        _assert_finite(float(value), f"{benchmark}:{kernel_regime}:pair_raw[{pair_key}]")


@pytest.mark.parametrize("benchmark, polys_fixture, R", BENCHMARKS)
@pytest.mark.parametrize("kernel_regime", ["raw", "paper"])
def test_compare_triangle_vs_ordered_report_finite(
    request, benchmark: str, polys_fixture: str, R: float, kernel_regime: str
):
    P1, P2, P3, Q = request.getfixturevalue(polys_fixture)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    report = compare_triangle_vs_ordered(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        kernel_regime=kernel_regime,
        n_quad_a=N_QUAD_A,
        verbose=False,
    )

    for key in ["C_triangle", "C_ordered", "delta", "delta_rel", "identity_check"]:
        _assert_finite(float(report[key]), f"{benchmark}:{kernel_regime}:{key}")

    off_diagonal = report["off_diagonal"]
    assert isinstance(off_diagonal, dict)
    for key, d in off_diagonal.items():
        for field in ["S_pq", "S_qp", "S12_pq", "S12_qp", "S34_pq", "S34_qp", "delta_S", "delta_S12", "delta_S34"]:
            _assert_finite(float(d[field]), f"{benchmark}:{kernel_regime}:{key}:{field}")


# =============================================================================
# REGRESSION TESTS: Lock in empirical facts (2025-12-19)
# =============================================================================
# These tests assert what we PROVED by measurement:
# - S12 (I1+I2) IS symmetric under pair swap
# - S34 (I3+I4) is NOT symmetric under pair swap
# =============================================================================

@pytest.mark.parametrize("benchmark, polys_fixture, R", BENCHMARKS)
@pytest.mark.parametrize("kernel_regime", ["raw", "paper"])
def test_s12_is_symmetric(
    request, benchmark: str, polys_fixture: str, R: float, kernel_regime: str
):
    """S12 (I1+I2) is symmetric under pair swap - EMPIRICALLY VERIFIED."""
    P1, P2, P3, Q = request.getfixturevalue(polys_fixture)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    report = compare_triangle_vs_ordered(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        kernel_regime=kernel_regime,
        n_quad_a=N_QUAD_A,
        verbose=False,
    )

    # S12 should be symmetric (within machine epsilon)
    tol = 1e-10
    for key, d in report["off_diagonal"].items():
        assert abs(float(d["delta_S12"])) < tol, (
            f"{benchmark}:{kernel_regime}:{key}: "
            f"S12 NOT symmetric: delta_S12 = {d['delta_S12']:.2e}"
        )


@pytest.mark.parametrize("benchmark, polys_fixture, R", BENCHMARKS)
@pytest.mark.parametrize("kernel_regime", ["raw", "paper"])
def test_s34_is_asymmetric(
    request, benchmark: str, polys_fixture: str, R: float, kernel_regime: str
):
    """S34 (I3+I4) is NOT symmetric - EMPIRICALLY VERIFIED.

    This test asserts that delta_S34 for pair 12 is non-negligible,
    which is the structural reason triangle×2 fails.
    """
    P1, P2, P3, Q = request.getfixturevalue(polys_fixture)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    report = compare_triangle_vs_ordered(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        kernel_regime=kernel_regime,
        n_quad_a=N_QUAD_A,
        verbose=False,
    )

    # S34 for pair 12 should NOT be symmetric (this is the key asymmetry)
    d = report["off_diagonal"]["12_vs_21"]
    delta_s34 = abs(float(d["delta_S34"]))

    # The asymmetry is structural, not numerical noise: expect |delta| > 0.1
    assert delta_s34 > 0.1, (
        f"{benchmark}:{kernel_regime}: "
        f"S34 asymmetry vanished? delta_S34_12 = {d['delta_S34']:.6e}"
    )


@pytest.mark.parametrize("benchmark, polys_fixture, R", BENCHMARKS)
@pytest.mark.parametrize("kernel_regime", ["raw", "paper"])
def test_identity_check_equals_delta(
    request, benchmark: str, polys_fixture: str, R: float, kernel_regime: str
):
    """Verify that delta == identity_check (algebraic identity)."""
    P1, P2, P3, Q = request.getfixturevalue(polys_fixture)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    report = compare_triangle_vs_ordered(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        kernel_regime=kernel_regime,
        n_quad_a=N_QUAD_A,
        verbose=False,
    )

    delta = float(report["delta"])
    identity_check = float(report["identity_check"])

    # These should be exactly equal (algebraic identity)
    assert abs(delta - identity_check) < 1e-14, (
        f"{benchmark}:{kernel_regime}: "
        f"identity_check != delta: {identity_check:.12e} vs {delta:.12e}"
    )


@pytest.mark.parametrize("benchmark, polys_fixture, R", BENCHMARKS)
def test_compute_c_paper_defaults_to_ordered(request, benchmark: str, polys_fixture: str, R: float):
    """Paper-truth entrypoint must not silently revert to triangle×2."""
    P1, P2, P3, Q = request.getfixturevalue(polys_fixture)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    c_default = compute_c_paper(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        return_breakdown=False,
        n_quad_a=N_QUAD_A,
    ).total
    c_ordered = compute_c_paper(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        pair_mode="ordered",
        return_breakdown=False,
        n_quad_a=N_QUAD_A,
    ).total
    c_hybrid = compute_c_paper(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        pair_mode="hybrid",
        return_breakdown=False,
        n_quad_a=N_QUAD_A,
    ).total

    assert abs(c_default - c_ordered) < 1e-14, (
        f"{benchmark}: compute_c_paper default must be ordered: {c_default} vs {c_ordered}"
    )
    assert abs(c_hybrid - c_ordered) < 1e-12, (
        f"{benchmark}: hybrid must match ordered: {c_hybrid} vs {c_ordered}"
    )


# =============================================================================
# GUARDRAIL TEST: Prevent regression to triangle×2 in paper truth
# =============================================================================

@pytest.mark.parametrize("benchmark, polys_fixture, R", BENCHMARKS)
def test_paper_truth_equals_evaluate_c_ordered(request, benchmark: str, polys_fixture: str, R: float):
    """
    GUARDRAIL: compute_c_paper() must equal evaluate_c_ordered() within tolerance.

    This test prevents future refactors from accidentally reverting paper truth
    to triangle folding. Added per GPT guidance 2025-12-20.

    If this test fails, check:
    1. compute_c_paper() default pair_mode is "ordered"
    2. No triangle×2 symmetry factor is being applied internally
    """
    P1, P2, P3, Q = request.getfixturevalue(polys_fixture)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Get c from compute_c_paper (the "truth" entrypoint)
    c_paper = compute_c_paper(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        return_breakdown=False,
        n_quad_a=N_QUAD_A,
    ).total

    # Get c from evaluate_c_ordered (the structural reference)
    c_ordered = evaluate_c_ordered(
        theta=THETA,
        R=R,
        n=N_QUAD,
        polynomials=polynomials,
        kernel_regime="paper",
        n_quad_a=N_QUAD_A,
    ).total

    # These must be equal - tight tolerance
    assert abs(c_paper - c_ordered) < 1e-12, (
        f"GUARDRAIL FAILURE: {benchmark}\n"
        f"compute_c_paper()   = {c_paper:.12f}\n"
        f"evaluate_c_ordered() = {c_ordered:.12f}\n"
        f"Paper truth has diverged from ordered evaluation!"
    )
