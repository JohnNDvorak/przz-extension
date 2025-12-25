"""
tests/test_kernel_registry.py

Lightweight unit tests for the kernel-regime / omega selection contract.

These tests are intentionally simple and purely structural: they do not
validate any numerical integrals, only the mapping rules.
"""

import pytest

from src.kernel_registry import case_from_omega, kernel_spec_for_piece, omega_for_poly_name


def test_case_from_omega_mapping():
    assert case_from_omega(-1) == "A"
    assert case_from_omega(0) == "B"
    assert case_from_omega(1) == "C"
    assert case_from_omega(2) == "C"


def test_kernel_spec_raw_is_always_case_b():
    for ell in (1, 2, 3):
        spec = kernel_spec_for_piece(ell, regime="raw", d=1)
        assert spec.case == "B"
        assert spec.omega == 0


def test_kernel_spec_paper_d1_is_ell_minus_1():
    spec1 = kernel_spec_for_piece(1, regime="paper", d=1)
    assert spec1.case == "B"
    assert spec1.omega == 0

    spec2 = kernel_spec_for_piece(2, regime="paper", d=1)
    assert spec2.case == "C"
    assert spec2.omega == 1

    spec3 = kernel_spec_for_piece(3, regime="paper", d=1)
    assert spec3.case == "C"
    assert spec3.omega == 2


def test_omega_for_poly_name():
    assert omega_for_poly_name("Q", regime="raw") is None
    assert omega_for_poly_name("P1", regime="raw") == 0
    assert omega_for_poly_name("P2", regime="raw") == 0
    assert omega_for_poly_name("P3", regime="raw") == 0

    assert omega_for_poly_name("P1", regime="paper") == 0
    assert omega_for_poly_name("P2", regime="paper") == 1
    assert omega_for_poly_name("P3", regime="paper") == 2

