"""
tests/test_przz_polynomials_loader.py
Phase 14D Task D1: Tests for polynomial loader

Tests that the przz_polynomials.py module correctly loads and validates
PRZZ polynomials for the ratios pipeline.
"""

import pytest
import numpy as np
from src.ratios.przz_polynomials import (
    load_przz_k3_polynomials,
    make_poly_callable,
    get_polynomial_functions,
    validate_constraints,
    PrzzK3Polynomials,
    KAPPA_R,
    KAPPA_STAR_R,
)


class TestLoaderStructure:
    """Tests that the loader returns expected data structure."""

    def test_loader_returns_dataclass(self):
        """Loader should return a PrzzK3Polynomials dataclass."""
        polys = load_przz_k3_polynomials("kappa")
        assert isinstance(polys, PrzzK3Polynomials)

    def test_loader_has_all_polynomials(self):
        """Returned dataclass should have P1, P2, P3, Q."""
        polys = load_przz_k3_polynomials("kappa")
        assert hasattr(polys, 'P1')
        assert hasattr(polys, 'P2')
        assert hasattr(polys, 'P3')
        assert hasattr(polys, 'Q')

    def test_loader_has_metadata(self):
        """Returned dataclass should have benchmark, R, theta."""
        polys = load_przz_k3_polynomials("kappa")
        assert hasattr(polys, 'benchmark')
        assert hasattr(polys, 'R')
        assert hasattr(polys, 'theta')

    def test_kappa_benchmark_metadata(self):
        """Kappa benchmark should have correct R value."""
        polys = load_przz_k3_polynomials("kappa")
        assert polys.benchmark == "kappa"
        assert polys.R == KAPPA_R
        assert abs(polys.theta - 4.0 / 7.0) < 1e-10

    def test_kappa_star_benchmark_metadata(self):
        """Kappa* benchmark should have correct R value."""
        polys = load_przz_k3_polynomials("kappa_star")
        assert polys.benchmark == "kappa_star"
        assert polys.R == KAPPA_STAR_R
        assert abs(polys.theta - 4.0 / 7.0) < 1e-10

    def test_invalid_benchmark_raises(self):
        """Invalid benchmark name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            load_przz_k3_polynomials("invalid")


class TestKappaConstraints:
    """Tests that kappa benchmark polynomials satisfy constraints."""

    def test_P1_at_zero(self):
        """P1(0) = 0 (boundary constraint)."""
        polys = load_przz_k3_polynomials("kappa")
        val = float(polys.P1.eval(np.array([0.0]))[0])
        assert abs(val) < 1e-10

    def test_P1_at_one(self):
        """P1(1) = 1 (boundary constraint)."""
        polys = load_przz_k3_polynomials("kappa")
        val = float(polys.P1.eval(np.array([1.0]))[0])
        assert abs(val - 1.0) < 1e-10

    def test_P2_at_zero(self):
        """P2(0) = 0 (boundary constraint)."""
        polys = load_przz_k3_polynomials("kappa")
        val = float(polys.P2.eval(np.array([0.0]))[0])
        assert abs(val) < 1e-10

    def test_P3_at_zero(self):
        """P3(0) = 0 (boundary constraint)."""
        polys = load_przz_k3_polynomials("kappa")
        val = float(polys.P3.eval(np.array([0.0]))[0])
        assert abs(val) < 1e-10

    def test_Q_at_zero_approximately_one(self):
        """Q(0) ≈ 1 (paper has Q(0) ≈ 0.999999)."""
        polys = load_przz_k3_polynomials("kappa")
        val = float(polys.Q.eval(np.array([0.0]))[0])
        # Paper uses Q(0) ≈ 0.999999, not exactly 1.0
        assert abs(val - 1.0) < 1e-5


class TestKappaStarConstraints:
    """Tests that kappa* benchmark polynomials satisfy constraints."""

    def test_P1_at_zero(self):
        """P1(0) = 0 (boundary constraint)."""
        polys = load_przz_k3_polynomials("kappa_star")
        val = float(polys.P1.eval(np.array([0.0]))[0])
        assert abs(val) < 1e-10

    def test_P1_at_one(self):
        """P1(1) = 1 (boundary constraint)."""
        polys = load_przz_k3_polynomials("kappa_star")
        val = float(polys.P1.eval(np.array([1.0]))[0])
        assert abs(val - 1.0) < 1e-10

    def test_P2_at_zero(self):
        """P2(0) = 0 (boundary constraint)."""
        polys = load_przz_k3_polynomials("kappa_star")
        val = float(polys.P2.eval(np.array([0.0]))[0])
        assert abs(val) < 1e-10

    def test_P3_at_zero(self):
        """P3(0) = 0 (boundary constraint)."""
        polys = load_przz_k3_polynomials("kappa_star")
        val = float(polys.P3.eval(np.array([0.0]))[0])
        assert abs(val) < 1e-10

    def test_Q_at_zero_exactly_one(self):
        """Q(0) = 1 exactly for kappa* (linear Q)."""
        polys = load_przz_k3_polynomials("kappa_star")
        val = float(polys.Q.eval(np.array([0.0]))[0])
        # Kappa* has linear Q, Q(0) should be exactly 1
        assert abs(val - 1.0) < 1e-10


class TestPolynomialsAreCallable:
    """Tests that polynomials can be evaluated."""

    def test_P1_is_evaluable(self):
        """P1 should be evaluable at any point in [0,1]."""
        polys = load_przz_k3_polynomials("kappa")
        result = polys.P1.eval(np.array([0.5]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 1

    def test_P2_is_evaluable(self):
        """P2 should be evaluable at any point in [0,1]."""
        polys = load_przz_k3_polynomials("kappa")
        result = polys.P2.eval(np.array([0.5]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 1

    def test_P3_is_evaluable(self):
        """P3 should be evaluable at any point in [0,1]."""
        polys = load_przz_k3_polynomials("kappa")
        result = polys.P3.eval(np.array([0.5]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 1

    def test_Q_is_evaluable(self):
        """Q should be evaluable at any point in [0,1]."""
        polys = load_przz_k3_polynomials("kappa")
        result = polys.Q.eval(np.array([0.5]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 1

    def test_make_poly_callable(self):
        """make_poly_callable should return a simple f(u) -> float."""
        polys = load_przz_k3_polynomials("kappa")
        f = make_poly_callable(polys.P1)
        val = f(0.5)
        assert isinstance(val, float)

    def test_get_polynomial_functions(self):
        """get_polynomial_functions should return dict with all four."""
        polys = load_przz_k3_polynomials("kappa")
        funcs = get_polynomial_functions(polys)
        assert 'P1_func' in funcs
        assert 'P2_func' in funcs
        assert 'P3_func' in funcs
        assert 'Q_func' in funcs
        # Each should be callable
        assert callable(funcs['P1_func'])
        assert callable(funcs['P2_func'])
        assert callable(funcs['P3_func'])
        assert callable(funcs['Q_func'])


class TestValidateConstraints:
    """Tests for the validate_constraints helper."""

    def test_kappa_passes_all(self):
        """Kappa benchmark should pass all constraint checks."""
        polys = load_przz_k3_polynomials("kappa")
        results = validate_constraints(polys)
        assert results['P1(0)=0']
        assert results['P1(1)=1']
        assert results['P2(0)=0']
        assert results['P3(0)=0']
        assert results['Q(0)≈1']
        assert results['all_pass']

    def test_kappa_star_passes_all(self):
        """Kappa* benchmark should pass all constraint checks."""
        polys = load_przz_k3_polynomials("kappa_star")
        results = validate_constraints(polys)
        assert results['P1(0)=0']
        assert results['P1(1)=1']
        assert results['P2(0)=0']
        assert results['P3(0)=0']
        assert results['Q(0)≈1']
        assert results['all_pass']


class TestKappaVsKappaStar:
    """Tests that kappa and kappa* give different polynomial values."""

    def test_P1_differs(self):
        """P1 coefficients differ between benchmarks."""
        polys_k = load_przz_k3_polynomials("kappa")
        polys_ks = load_przz_k3_polynomials("kappa_star")
        # Evaluate at interior point
        val_k = float(polys_k.P1.eval(np.array([0.5]))[0])
        val_ks = float(polys_ks.P1.eval(np.array([0.5]))[0])
        # Should be different
        assert abs(val_k - val_ks) > 1e-6

    def test_P2_differs(self):
        """P2 coefficients differ between benchmarks."""
        polys_k = load_przz_k3_polynomials("kappa")
        polys_ks = load_przz_k3_polynomials("kappa_star")
        val_k = float(polys_k.P2.eval(np.array([0.5]))[0])
        val_ks = float(polys_ks.P2.eval(np.array([0.5]))[0])
        assert abs(val_k - val_ks) > 1e-6

    def test_P3_differs(self):
        """P3 coefficients differ between benchmarks."""
        polys_k = load_przz_k3_polynomials("kappa")
        polys_ks = load_przz_k3_polynomials("kappa_star")
        val_k = float(polys_k.P3.eval(np.array([0.5]))[0])
        val_ks = float(polys_ks.P3.eval(np.array([0.5]))[0])
        assert abs(val_k - val_ks) > 1e-6

    def test_R_differs(self):
        """R values differ between benchmarks."""
        polys_k = load_przz_k3_polynomials("kappa")
        polys_ks = load_przz_k3_polynomials("kappa_star")
        assert polys_k.R != polys_ks.R


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
