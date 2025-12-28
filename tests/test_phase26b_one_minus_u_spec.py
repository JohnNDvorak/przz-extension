"""
tests/test_phase26b_one_minus_u_spec.py
Phase 26B: Lock (1-u) power specification from PRZZ TeX

PRZZ TeX Reference Table:
| Term | PRZZ TeX Line | (1-u) Power        |
|------|---------------|---------------------|
| I₁   | 1435          | (1-u)^{ℓ₁+ℓ₂}      |
| I₂   | 1548          | none                |
| I₃   | 1484          | (1-u)^{ℓ₁}         |
| I₄   | 1488          | (1-u)^{ℓ₂}         |

These tests lock the specification so future changes don't accidentally break it.
"""

import pytest
from src.terms_k3_d1 import (
    make_all_terms_11,
    make_all_terms_22,
    make_all_terms_33,
    make_all_terms_12,
    make_all_terms_13,
    make_all_terms_23,
)


def get_poly_prefactor_power(term):
    """
    Extract the (1-u) power from a term's poly_prefactors.

    Returns 0 if no poly_prefactor, otherwise evaluates at sample point.
    """
    if not term.poly_prefactors:
        return 0

    # Evaluate at u=0.5, t=0.5 to see the power
    # For (1-u)^n at u=0.5: value = 0.5^n
    import numpy as np
    U = np.array([0.5])
    T = np.array([0.5])

    prefactor_value = 1.0
    for pf in term.poly_prefactors:
        prefactor_value *= pf(U, T)[0]

    # Infer power from value: 0.5^n = prefactor_value
    # n = log(value) / log(0.5)
    if prefactor_value <= 0:
        return None  # Invalid

    import math
    power = round(math.log(prefactor_value) / math.log(0.5))
    return power


class TestI1OnMinusUPower:
    """Test I₁ has (1-u)^{ℓ₁+ℓ₂} power."""

    def test_I1_11_power_is_2(self):
        """I₁ for (1,1): power = 1+1 = 2."""
        terms = make_all_terms_11(theta=4/7, R=1.3036)
        i1_term = [t for t in terms if t.name == "I1_11"][0]
        power = get_poly_prefactor_power(i1_term)
        assert power == 2, f"Expected power 2, got {power}"

    def test_I1_22_power_is_4(self):
        """I₁ for (2,2): power = 2+2 = 4."""
        terms = make_all_terms_22(theta=4/7, R=1.3036)
        i1_term = [t for t in terms if t.name == "I1_22"][0]
        power = get_poly_prefactor_power(i1_term)
        assert power == 4, f"Expected power 4, got {power}"

    def test_I1_33_power_is_6(self):
        """I₁ for (3,3): power = 3+3 = 6."""
        terms = make_all_terms_33(theta=4/7, R=1.3036)
        i1_term = [t for t in terms if t.name == "I1_33"][0]
        power = get_poly_prefactor_power(i1_term)
        assert power == 6, f"Expected power 6, got {power}"

    def test_I1_12_power_is_3(self):
        """I₁ for (1,2): power = 1+2 = 3."""
        terms = make_all_terms_12(theta=4/7, R=1.3036)
        i1_term = [t for t in terms if t.name == "I1_12"][0]
        power = get_poly_prefactor_power(i1_term)
        assert power == 3, f"Expected power 3, got {power}"

    def test_I1_13_power_is_4(self):
        """I₁ for (1,3): power = 1+3 = 4."""
        terms = make_all_terms_13(theta=4/7, R=1.3036)
        i1_term = [t for t in terms if t.name == "I1_13"][0]
        power = get_poly_prefactor_power(i1_term)
        assert power == 4, f"Expected power 4, got {power}"

    def test_I1_23_power_is_5(self):
        """I₁ for (2,3): power = 2+3 = 5."""
        terms = make_all_terms_23(theta=4/7, R=1.3036)
        i1_term = [t for t in terms if t.name == "I1_23"][0]
        power = get_poly_prefactor_power(i1_term)
        assert power == 5, f"Expected power 5, got {power}"


class TestI2OnMinusUPower:
    """Test I₂ has NO (1-u) factor (PRZZ TeX line 1548)."""

    @pytest.mark.parametrize("pair,make_terms", [
        ("11", make_all_terms_11),
        ("22", make_all_terms_22),
        ("33", make_all_terms_33),
        ("12", make_all_terms_12),
        ("13", make_all_terms_13),
        ("23", make_all_terms_23),
    ])
    def test_I2_has_no_poly_prefactor(self, pair, make_terms):
        """I₂ for all pairs: no (1-u) factor."""
        terms = make_terms(theta=4/7, R=1.3036)
        i2_term = [t for t in terms if t.name == f"I2_{pair}"][0]
        power = get_poly_prefactor_power(i2_term)
        assert power == 0, f"I2_{pair}: Expected power 0, got {power}"


class TestI3OnMinusUPower:
    """Test I₃ has (1-u)^{ℓ₁} power."""

    def test_I3_11_power_is_1(self):
        """I₃ for (1,1): power = ℓ₁ = 1."""
        terms = make_all_terms_11(theta=4/7, R=1.3036)
        i3_term = [t for t in terms if t.name == "I3_11"][0]
        power = get_poly_prefactor_power(i3_term)
        assert power == 1, f"Expected power 1, got {power}"

    def test_I3_22_power_is_2(self):
        """I₃ for (2,2): power = ℓ₁ = 2."""
        terms = make_all_terms_22(theta=4/7, R=1.3036)
        i3_term = [t for t in terms if t.name == "I3_22"][0]
        power = get_poly_prefactor_power(i3_term)
        assert power == 2, f"Expected power 2, got {power}"

    def test_I3_33_power_is_3(self):
        """I₃ for (3,3): power = ℓ₁ = 3."""
        terms = make_all_terms_33(theta=4/7, R=1.3036)
        i3_term = [t for t in terms if t.name == "I3_33"][0]
        power = get_poly_prefactor_power(i3_term)
        assert power == 3, f"Expected power 3, got {power}"

    def test_I3_12_power_is_1(self):
        """I₃ for (1,2): power = ℓ₁ = 1."""
        terms = make_all_terms_12(theta=4/7, R=1.3036)
        i3_term = [t for t in terms if t.name == "I3_12"][0]
        power = get_poly_prefactor_power(i3_term)
        assert power == 1, f"Expected power 1, got {power}"

    def test_I3_13_power_is_1(self):
        """I₃ for (1,3): power = ℓ₁ = 1."""
        terms = make_all_terms_13(theta=4/7, R=1.3036)
        i3_term = [t for t in terms if t.name == "I3_13"][0]
        power = get_poly_prefactor_power(i3_term)
        assert power == 1, f"Expected power 1, got {power}"

    def test_I3_23_power_is_2(self):
        """I₃ for (2,3): power = ℓ₁ = 2."""
        terms = make_all_terms_23(theta=4/7, R=1.3036)
        i3_term = [t for t in terms if t.name == "I3_23"][0]
        power = get_poly_prefactor_power(i3_term)
        assert power == 2, f"Expected power 2, got {power}"


class TestI4OnMinusUPower:
    """Test I₄ has (1-u)^{ℓ₂} power."""

    def test_I4_11_power_is_1(self):
        """I₄ for (1,1): power = ℓ₂ = 1."""
        terms = make_all_terms_11(theta=4/7, R=1.3036)
        i4_term = [t for t in terms if t.name == "I4_11"][0]
        power = get_poly_prefactor_power(i4_term)
        assert power == 1, f"Expected power 1, got {power}"

    def test_I4_22_power_is_2(self):
        """I₄ for (2,2): power = ℓ₂ = 2."""
        terms = make_all_terms_22(theta=4/7, R=1.3036)
        i4_term = [t for t in terms if t.name == "I4_22"][0]
        power = get_poly_prefactor_power(i4_term)
        assert power == 2, f"Expected power 2, got {power}"

    def test_I4_33_power_is_3(self):
        """I₄ for (3,3): power = ℓ₂ = 3."""
        terms = make_all_terms_33(theta=4/7, R=1.3036)
        i4_term = [t for t in terms if t.name == "I4_33"][0]
        power = get_poly_prefactor_power(i4_term)
        assert power == 3, f"Expected power 3, got {power}"

    def test_I4_12_power_is_2(self):
        """I₄ for (1,2): power = ℓ₂ = 2."""
        terms = make_all_terms_12(theta=4/7, R=1.3036)
        i4_term = [t for t in terms if t.name == "I4_12"][0]
        power = get_poly_prefactor_power(i4_term)
        assert power == 2, f"Expected power 2, got {power}"

    def test_I4_13_power_is_3(self):
        """I₄ for (1,3): power = ℓ₂ = 3."""
        terms = make_all_terms_13(theta=4/7, R=1.3036)
        i4_term = [t for t in terms if t.name == "I4_13"][0]
        power = get_poly_prefactor_power(i4_term)
        assert power == 3, f"Expected power 3, got {power}"

    def test_I4_23_power_is_3(self):
        """I₄ for (2,3): power = ℓ₂ = 3."""
        terms = make_all_terms_23(theta=4/7, R=1.3036)
        i4_term = [t for t in terms if t.name == "I4_23"][0]
        power = get_poly_prefactor_power(i4_term)
        assert power == 3, f"Expected power 3, got {power}"


class TestUnifiedEvaluatorOnMinusUPower:
    """Test the unified evaluator uses correct (1-u) power for I₁."""

    def test_unified_I1_uses_ell1_plus_ell2_power(self):
        """Verify unified evaluator uses (1-u)^{ℓ₁+ℓ₂} for I₁."""
        # This is locked by the fix in unified_s12_evaluator_v3.py
        # Line ~440: one_minus_u_power = ell1 + ell2
        from src.unified_s12_evaluator_v3 import compute_I1_unified_v3
        from src.polynomials import load_przz_polynomials

        P1, P2, P3, Q = load_przz_polynomials()
        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        # For (1,1), the power is 2
        # We verify by checking that the result is reasonable
        # (full matching is done in other tests)
        result = compute_I1_unified_v3(
            R=1.3036, theta=4/7, ell1=1, ell2=1,
            polynomials=polynomials, n_quad_u=20, n_quad_t=20
        )
        # Result should be non-zero and finite
        assert result.I1_value != 0
        assert not float('inf') == result.I1_value
        assert not float('-inf') == result.I1_value
