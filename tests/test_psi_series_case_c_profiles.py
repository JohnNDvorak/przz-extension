"""
tests/test_psi_series_case_c_profiles.py

Sanity checks that PsiSeriesEvaluator can swap its P-profile factors to
Case C kernels (omega>0) via `src.mollifier_profiles.case_c_taylor_coeffs`.

We keep this test intentionally simple by choosing:
- Q(t) = 1 (constant), so Q(α)Q(β)=1 with no higher Taylor terms
- R = 0, so exp factors are identically 1 and do not couple x/y

Then the full kernel reduces to just the (left/right) profile shifts:
    F(x,y) = Left(u+x) * Right(u+y)

With Right constant, f[i,0] should match Left^{(i)}(u)/i! and all f[i,j>0]=0.
"""

import numpy as np

from src.polynomials import Polynomial
from src.psi_series_evaluator import PsiSeriesEvaluator


def test_case_c_profile_omega1_R0_matches_closed_form_coeffs():
    """For P(x)=1, omega=1, R=0: K(u)=u, K'(u)=1, higher derivatives 0."""
    P = Polynomial([1.0])  # P(x)=1
    Q = Polynomial([1.0])  # Q(t)=1

    theta = 0.5
    evaluator = PsiSeriesEvaluator(
        P,
        P,
        Q,
        R=0.0,
        theta=theta,
        max_order=3,
        n_quad=8,
        omega_left=1,
        omega_right=0,  # raw (but P is constant so it stays 1)
        n_quad_a=40,
    )

    u = 0.7
    t = 0.3
    f = evaluator._build_F_coefficients(u, t)

    # Left profile is K_1(u)=u for this choice; right is constant 1.
    np.testing.assert_allclose(f[0, 0], u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(f[1, 0], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(f[2, 0], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(f[3, 0], 0.0, rtol=0, atol=1e-12)

    # No y-dependence because Right is constant and all other factors are 1.
    np.testing.assert_allclose(f[:, 1:], 0.0, rtol=0, atol=1e-12)


def test_case_c_profile_omega2_R0_matches_closed_form_coeffs():
    """For P(x)=1, omega=2, R=0: K(u)=u^2/2, K'(u)=u, K''(u)=1."""
    P = Polynomial([1.0])  # P(x)=1
    Q = Polynomial([1.0])  # Q(t)=1

    theta = 0.5
    evaluator = PsiSeriesEvaluator(
        P,
        P,
        Q,
        R=0.0,
        theta=theta,
        max_order=3,
        n_quad=8,
        omega_left=2,
        omega_right=0,
        n_quad_a=40,
    )

    u = 0.6
    t = 0.4
    f = evaluator._build_F_coefficients(u, t)

    np.testing.assert_allclose(f[0, 0], 0.5 * u * u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(f[1, 0], u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(f[2, 0], 0.5, rtol=0, atol=1e-12)  # K''/2!
    np.testing.assert_allclose(f[3, 0], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(f[:, 1:], 0.0, rtol=0, atol=1e-12)


def test_case_c_profile_precomputed_matches_fallback_path():
    """(u_idx,t_idx) fast path matches the fallback path for the same node."""
    P = Polynomial([1.0])
    Q = Polynomial([1.0])

    evaluator = PsiSeriesEvaluator(
        P,
        P,
        Q,
        R=0.0,
        theta=0.5,
        max_order=2,
        n_quad=10,
        omega_left=2,
        omega_right=2,
        n_quad_a=40,
    )

    iu = 3
    it = 4
    u = float(evaluator.u_nodes[iu])
    t = float(evaluator.t_nodes[it])

    f_fallback = evaluator._build_F_coefficients(u, t)
    f_fast = evaluator._build_F_coefficients(u, t, u_idx=iu, t_idx=it)

    np.testing.assert_allclose(f_fast, f_fallback, rtol=0, atol=1e-12)

