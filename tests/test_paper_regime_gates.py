"""
tests/test_paper_regime_gates.py
Pre-gates for the DSL paper-truth pipeline.

These are intentionally *not* the two-benchmark gate (which is expected to
fail until the full PRZZ assembly matches the paper).

They instead validate:
- P1: switching raw -> paper moves κ/κ* ratio in the right direction
- P2: term tables correctly tag P₂/P₃ with Case C omegas under paper regime
- P3: Case C a-quadrature is numerically stable at modest settings
"""

import pytest

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import evaluate_c_full
from src.terms_k3_d1 import make_all_terms_k3, make_all_terms_22


THETA = 4.0 / 7.0
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167


def _poly_dict(P1, P2, P3, Q):
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


def _compute_ratio(*, kernel_regime: str, n: int, n_quad_a: int) -> float:
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    result_k = evaluate_c_full(
        THETA,
        R_KAPPA,
        n,
        _poly_dict(P1, P2, P3, Q),
        return_breakdown=False,
        kernel_regime=kernel_regime,
        n_quad_a=n_quad_a,
    )

    P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star()
    result_s = evaluate_c_full(
        THETA,
        R_KAPPA_STAR,
        n,
        _poly_dict(P1s, P2s, P3s, Qs),
        return_breakdown=False,
        kernel_regime=kernel_regime,
        n_quad_a=n_quad_a,
    )

    assert result_s.total != 0.0
    return float(result_k.total / result_s.total)


def test_paper_regime_improves_ratio_directionally():
    """Gate P1: ratio(raw) > ratio(paper) (directional leverage smoke test)."""
    # Keep this cheap: low quadrature, but direction should be robust.
    n = 14
    n_quad_a = 14

    ratio_raw = _compute_ratio(kernel_regime="raw", n=n, n_quad_a=n_quad_a)
    ratio_paper = _compute_ratio(kernel_regime="paper", n=n, n_quad_a=n_quad_a)

    assert ratio_raw > ratio_paper, (
        f"Expected paper regime to reduce κ/κ* ratio, got raw={ratio_raw:.4f}, paper={ratio_paper:.4f}"
    )


def test_all_p2_p3_terms_use_case_c_in_paper_regime():
    """Gate P2: in paper regime, P2/P3 factors must be tagged with ω=1/2."""
    terms_by_pair = make_all_terms_k3(THETA, R_KAPPA, kernel_regime="paper")

    found_p2 = 0
    found_p3 = 0

    for _, terms in terms_by_pair.items():
        for term in terms:
            for factor in term.poly_factors:
                if factor.poly_name == "P1":
                    assert factor.omega == 0
                elif factor.poly_name == "P2":
                    found_p2 += 1
                    assert factor.omega == 1
                elif factor.poly_name == "P3":
                    found_p3 += 1
                    assert factor.omega == 2
                elif factor.poly_name == "Q":
                    assert factor.omega is None
                else:
                    raise AssertionError(f"Unexpected poly_name {factor.poly_name!r}")

    assert found_p2 > 0
    assert found_p3 > 0


def test_case_c_a_quadrature_converges_for_pair_22():
    """Gate P3: Case C introduces an a-integral; verify stability for (2,2)."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polynomials = _poly_dict(P1, P2, P3, Q)

    terms_22 = make_all_terms_22(THETA, R_KAPPA, kernel_regime="paper")

    # Keep u/t quadrature modest; we're only sanity-checking a-integral stability.
    n = 18
    from src.evaluate import evaluate_terms

    c_20 = evaluate_terms(
        terms_22, polynomials, n, return_breakdown=False, R=R_KAPPA, theta=THETA, n_quad_a=20
    ).total
    c_40 = evaluate_terms(
        terms_22, polynomials, n, return_breakdown=False, R=R_KAPPA, theta=THETA, n_quad_a=40
    ).total

    assert c_40 != 0.0
    rel = abs(c_20 - c_40) / abs(c_40)
    assert rel < 0.01, f"(2,2) a-quadrature not stable: rel_diff={rel:.3%}"
