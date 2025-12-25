"""
tests/test_amplitude_candidate_fits.py

Pure-math unit tests for the TeX-motivated amplitude candidate library and fit helpers.

These tests do not depend on PRZZ integrals or benchmarks.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import candidate_amplitude_functions, fit_amplitude_candidates, rank_amplitude_fits


def test_candidate_amplitude_functions_return_finite_values():
    theta = 4.0 / 7.0
    R = 1.23

    funcs = candidate_amplitude_functions()
    assert "exp(R)" in funcs

    for name, f in funcs.items():
        val = float(f(R, theta))
        assert np.isfinite(val), f"{name} returned non-finite value {val}"


def test_fit_amplitude_candidates_recovers_scale_for_synthetic_expR():
    theta = 4.0 / 7.0
    R_values = [0.9, 1.0, 1.1, 1.2, 1.3]
    k = 2.5

    A_resid_values = [k * float(np.exp(R)) for R in R_values]
    fits = fit_amplitude_candidates(R_values, A_resid_values, theta)

    exp_fit = fits["exp(R)"]
    assert exp_fit["scale"] == pytest.approx(k, rel=1e-12, abs=1e-12)
    assert exp_fit["rmse"] < 1e-12


def test_rank_amplitude_fits_prefers_true_candidate_for_synthetic_data():
    theta = 4.0 / 7.0
    R_values = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    k1 = 2.5
    k2 = 3.0

    A1_resid = [k1 * float(np.exp(R)) for R in R_values]
    A2_resid = [k2 * float(np.exp(R)) for R in R_values]

    ranked = rank_amplitude_fits(R_values, A1_resid, A2_resid, theta)
    best_name, best = ranked[0]

    assert best_name == "exp(R)"
    assert best["A1_scale"] == pytest.approx(k1, rel=1e-12, abs=1e-12)
    assert best["A2_scale"] == pytest.approx(k2, rel=1e-12, abs=1e-12)
    assert best["scale_ratio"] == pytest.approx(k1 / k2, rel=1e-12, abs=1e-12)
    assert best["combined_rmse"] < 1e-12

