"""
tests/test_paper_integrator.py
Tests for the paper equation integrator.

This validates that the paper_integrator module correctly implements
the PRZZ equations, and compares against DSL output to find discrepancies.
"""

import pytest
import numpy as np
from math import factorial

from src.polynomials import load_przz_polynomials
from src.paper_integrator import (
    compute_pair_11_paper,
    compute_c_paper_11_only,
    compute_pair_paper,
    compute_c_paper_k3,
)
from src.evaluate import evaluate_c_full, compute_kappa


# Load PRZZ polynomials once
@pytest.fixture(scope="module")
def przz_polys():
    return load_przz_polynomials()


@pytest.fixture(scope="module")
def przz_params():
    return {
        "theta": 4.0 / 7.0,
        "R": 1.3036,
        "c_target": 2.13745440613217263636,
        "kappa_target": 0.417293962,
    }


class TestPaperIntegratorBasic:
    """Basic tests that paper integrator runs without errors."""

    def test_pair_11_runs(self, przz_polys, przz_params):
        """Test that (1,1) paper computation runs."""
        P1, P2, P3, Q = przz_polys
        result = compute_pair_11_paper(
            P1, P1, Q,
            przz_params["theta"],
            przz_params["R"],
            n_quadrature=20  # Low for speed
        )
        assert "I1_11" in result
        assert "I2_11" in result
        assert "I3_11" in result
        assert "I4_11" in result
        assert "total_11" in result

    def test_pair_11_total_positive(self, przz_polys, przz_params):
        """Test that (1,1) total is positive (sanity check)."""
        P1, P2, P3, Q = przz_polys
        result = compute_pair_11_paper(
            P1, P1, Q,
            przz_params["theta"],
            przz_params["R"],
            n_quadrature=30
        )
        # c should be positive
        assert result["total_11"] > 0, f"Expected positive total, got {result['total_11']}"

    def test_full_k3_paper_runs(self, przz_polys, przz_params):
        """Test that full K=3 paper computation runs."""
        c_paper, breakdown = compute_c_paper_k3(
            przz_polys,
            przz_params["theta"],
            przz_params["R"],
            n_quadrature=20
        )
        assert c_paper > 0
        assert "c_11" in breakdown
        assert "c_22" in breakdown
        assert "c_33" in breakdown


class TestPaperVsDSL:
    """Compare paper integrator against DSL output."""

    def test_pair_11_paper_vs_dsl(self, przz_polys, przz_params):
        """Compare (1,1) pair: paper vs DSL."""
        from src.terms_k3_d1 import (
            make_I1_11, make_I2_11, make_I3_11, make_I4_11
        )
        from src.evaluate import evaluate_term

        theta = przz_params["theta"]
        R = przz_params["R"]
        n_quad = 40

        # Paper computation
        P1, P2, P3, Q = przz_polys
        paper_result = compute_pair_11_paper(P1, P1, Q, theta, R, n_quad)

        # DSL computation
        polys_dict = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        I1_term = make_I1_11(theta, R)
        I2_term = make_I2_11(theta, R)
        I3_term = make_I3_11(theta, R)
        I4_term = make_I4_11(theta, R)

        dsl_I1_result = evaluate_term(I1_term, polys_dict, n_quad)
        dsl_I2_result = evaluate_term(I2_term, polys_dict, n_quad)
        dsl_I3_result = evaluate_term(I3_term, polys_dict, n_quad)
        dsl_I4_result = evaluate_term(I4_term, polys_dict, n_quad)

        # Extract numeric values from TermResult objects
        dsl_I1 = dsl_I1_result.value
        dsl_I2 = dsl_I2_result.value
        dsl_I3 = dsl_I3_result.value
        dsl_I4 = dsl_I4_result.value

        # Compare term by term
        print(f"\n(1,1) Comparison (n={n_quad}):")
        print(f"  I1: paper={paper_result['I1_11']:.10e}, dsl={dsl_I1:.10e}, diff={abs(paper_result['I1_11']-dsl_I1)/max(abs(dsl_I1), 1e-15)*100:.4f}%")
        print(f"  I2: paper={paper_result['I2_11']:.10e}, dsl={dsl_I2:.10e}, diff={abs(paper_result['I2_11']-dsl_I2)/max(abs(dsl_I2), 1e-15)*100:.4f}%")
        print(f"  I3: paper={paper_result['I3_11']:.10e}, dsl={dsl_I3:.10e}, diff={abs(paper_result['I3_11']-dsl_I3)/max(abs(dsl_I3), 1e-15)*100:.4f}%")
        print(f"  I4: paper={paper_result['I4_11']:.10e}, dsl={dsl_I4:.10e}, diff={abs(paper_result['I4_11']-dsl_I4)/max(abs(dsl_I4), 1e-15)*100:.4f}%")

        dsl_total = dsl_I1 + dsl_I2 + dsl_I3 + dsl_I4
        paper_total = paper_result['total_11']
        print(f"  Total: paper={paper_total:.10e}, dsl={dsl_total:.10e}, diff={abs(paper_total-dsl_total)/max(abs(dsl_total), 1e-15)*100:.4f}%")

        # Allow some tolerance due to different implementation paths
        rel_tol = 0.05  # 5% tolerance for now - will tighten after debugging
        for name, paper_val, dsl_val in [
            ("I1", paper_result['I1_11'], dsl_I1),
            ("I2", paper_result['I2_11'], dsl_I2),
            ("I3", paper_result['I3_11'], dsl_I3),
            ("I4", paper_result['I4_11'], dsl_I4),
        ]:
            if abs(dsl_val) > 1e-15:
                rel_diff = abs(paper_val - dsl_val) / abs(dsl_val)
                # This test will show us WHERE discrepancies are
                # Initially we just document, later we'll tighten
                print(f"  {name} relative diff: {rel_diff:.6f}")


class TestPaperVsPRZZTarget:
    """Test paper integrator against PRZZ published targets."""

    def test_c_paper_vs_target(self, przz_polys, przz_params):
        """Compare full c (paper, no I5) against PRZZ target."""
        c_paper, breakdown = compute_c_paper_k3(
            przz_polys,
            przz_params["theta"],
            przz_params["R"],
            n_quadrature=60,
            verbose=True
        )

        c_target = przz_params["c_target"]
        kappa_paper = compute_kappa(c_paper, przz_params["R"])
        kappa_target = przz_params["kappa_target"]

        print(f"\nPaper integrator results (no I5):")
        print(f"  c_paper = {c_paper:.12f}")
        print(f"  c_target = {c_target:.12f}")
        print(f"  Δc = {(c_paper - c_target)/c_target * 100:.4f}%")
        print(f"  κ_paper = {kappa_paper:.9f}")
        print(f"  κ_target = {kappa_target:.9f}")
        print(f"  Δκ = {(kappa_paper - kappa_target)/kappa_target * 100:.4f}%")

        # Key diagnostic: what is the gap WITHOUT I5?
        # If this matches target, DSL has a bug
        # If this doesn't match, we've been computing the right thing all along
        # and the ~2% is real (meaning PRZZ target includes something we're missing)


class TestPaperQuadratureConvergence:
    """Test that paper integrator converges with quadrature."""

    def test_pair_11_convergence(self, przz_polys, przz_params):
        """Test (1,1) converges as n increases."""
        P1, P2, P3, Q = przz_polys
        theta = przz_params["theta"]
        R = przz_params["R"]

        results = []
        for n in [20, 40, 60]:
            result = compute_pair_11_paper(P1, P1, Q, theta, R, n)
            results.append((n, result['total_11']))

        print(f"\n(1,1) quadrature convergence:")
        for n, val in results:
            print(f"  n={n}: {val:.12f}")

        # Check convergence
        diff_40_20 = abs(results[1][1] - results[0][1])
        diff_60_40 = abs(results[2][1] - results[1][1])
        print(f"  |diff(40,20)| = {diff_40_20:.6e}")
        print(f"  |diff(60,40)| = {diff_60_40:.6e}")

        # Should be converging (diff decreasing), or both at machine epsilon
        # When fully converged, diffs are at ~1e-16 level (machine epsilon)
        machine_eps = 1e-14  # Allow for some floating point accumulation
        if diff_40_20 < machine_eps and diff_60_40 < machine_eps:
            # Both diffs at machine epsilon - fully converged
            pass
        else:
            assert diff_60_40 < diff_40_20 * 2, "Quadrature not converging properly"


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_paper_integrator.py -v -s
    pytest.main([__file__, "-v", "-s"])
