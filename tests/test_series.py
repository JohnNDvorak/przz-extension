"""
tests/test_series.py
Unit tests for the TruncatedSeries multi-variable Taylor series engine.

Tests follow the bitset representation where:
- Variables map to bit positions: x -> bit 0 (mask 1), y -> bit 1 (mask 2)
- Nilpotency: x^2 = y^2 = 0 (any mask & mask != 0 vanishes)
- Multiplication: xy = x | y = mask 3
"""

import numpy as np
import pytest
from src.series import TruncatedSeries


# =============================================================================
# Test 1: Variable Construction and Basic Properties
# =============================================================================

class TestVariableConstruction:
    """Test creating series for individual variables."""

    def test_from_scalar_creates_constant_term(self):
        """from_scalar should create series with only constant term (mask 0)."""
        var_names = ("x", "y")
        s = TruncatedSeries.from_scalar(5.0, var_names)

        assert s.var_names == var_names
        assert 0 in s.coeffs
        np.testing.assert_equal(s.coeffs[0], 5.0)

    def test_variable_creates_linear_term(self):
        """variable('x') should create series with coefficient 1 at mask 1."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        # x maps to bit 0 -> mask 1
        assert 1 in x.coeffs
        np.testing.assert_equal(x.coeffs[1], 1.0)
        assert 0 not in x.coeffs  # No constant term

    def test_variable_y_creates_mask_2(self):
        """variable('y') should create series with coefficient 1 at mask 2."""
        var_names = ("x", "y")
        y = TruncatedSeries.variable("y", var_names)

        # y maps to bit 1 -> mask 2
        assert 2 in y.coeffs
        np.testing.assert_equal(y.coeffs[2], 1.0)

    def test_variable_invalid_name_raises(self):
        """Requesting unknown variable should raise ValueError."""
        var_names = ("x", "y")
        with pytest.raises(ValueError, match="not in"):
            TruncatedSeries.variable("z", var_names)

    def test_three_variables(self):
        """Test with three variables: x, y, z at bits 0, 1, 2."""
        var_names = ("x", "y", "z")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)
        z = TruncatedSeries.variable("z", var_names)

        assert 1 in x.coeffs  # mask 001
        assert 2 in y.coeffs  # mask 010
        assert 4 in z.coeffs  # mask 100


# =============================================================================
# Test 2: Multiplication and Nilpotency
# =============================================================================

class TestMultiplication:
    """Test multiplication with nilpotent truncation."""

    def test_x_times_y_gives_xy(self):
        """x * y = xy (mask 1 | 2 = 3)."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        xy = x * y

        # xy term at mask 3
        assert 3 in xy.coeffs
        np.testing.assert_equal(xy.coeffs[3], 1.0)
        # No other terms
        assert len(xy.coeffs) == 1

    def test_x_times_x_gives_zero(self):
        """x * x = 0 (nilpotent: mask 1 & 1 != 0)."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        xx = x * x

        # Should be empty or all zeros
        for mask, coeff in xx.coeffs.items():
            np.testing.assert_equal(coeff, 0.0)

    def test_xy_times_x_gives_zero(self):
        """(xy) * x = 0 because x appears twice."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        xy = x * y
        xyx = xy * x

        # All terms should be zero
        for mask, coeff in xyx.coeffs.items():
            np.testing.assert_equal(coeff, 0.0)

    def test_scalar_multiplication(self):
        """Multiplying by scalar scales all coefficients."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        scaled = x * 3.0

        assert 1 in scaled.coeffs
        np.testing.assert_equal(scaled.coeffs[1], 3.0)

    def test_rmul_scalar(self):
        """3.0 * x should also work (right multiplication)."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        scaled = 3.0 * x

        assert 1 in scaled.coeffs
        np.testing.assert_equal(scaled.coeffs[1], 3.0)

    def test_multiplication_with_constant_series(self):
        """(1 + x) * 2 = 2 + 2x."""
        var_names = ("x", "y")
        one = TruncatedSeries.from_scalar(1.0, var_names)
        x = TruncatedSeries.variable("x", var_names)

        s = one + x  # 1 + x
        result = s * 2.0

        np.testing.assert_allclose(result.coeffs[0], 2.0)
        np.testing.assert_allclose(result.coeffs[1], 2.0)

    def test_series_times_series(self):
        """(1 + x) * (1 + y) = 1 + x + y + xy."""
        var_names = ("x", "y")
        one = TruncatedSeries.from_scalar(1.0, var_names)
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        s1 = one + x  # 1 + x
        s2 = one + y  # 1 + y
        result = s1 * s2

        np.testing.assert_allclose(result.coeffs[0], 1.0)  # constant
        np.testing.assert_allclose(result.coeffs[1], 1.0)  # x
        np.testing.assert_allclose(result.coeffs[2], 1.0)  # y
        np.testing.assert_allclose(result.coeffs[3], 1.0)  # xy

    def test_square_of_sum_nilpotent(self):
        """(x + y)^2 = 2xy (since x^2 = y^2 = 0)."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        s = x + y
        result = s * s

        # Only xy term survives: x*y + y*x = 2xy
        assert 3 in result.coeffs
        np.testing.assert_allclose(result.coeffs[3], 2.0)
        # x^2 and y^2 vanish
        assert result.coeffs.get(1, 0.0) == 0.0 or 1 not in result.coeffs
        assert result.coeffs.get(2, 0.0) == 0.0 or 2 not in result.coeffs


# =============================================================================
# Test 3: Addition
# =============================================================================

class TestAddition:
    """Test series addition."""

    def test_add_two_variables(self):
        """x + y should have terms at mask 1 and mask 2."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        s = x + y

        np.testing.assert_equal(s.coeffs[1], 1.0)
        np.testing.assert_equal(s.coeffs[2], 1.0)

    def test_add_constant_and_variable(self):
        """1 + x should have terms at mask 0 and mask 1."""
        var_names = ("x", "y")
        one = TruncatedSeries.from_scalar(1.0, var_names)
        x = TruncatedSeries.variable("x", var_names)

        s = one + x

        np.testing.assert_equal(s.coeffs[0], 1.0)
        np.testing.assert_equal(s.coeffs[1], 1.0)

    def test_add_overlapping_terms(self):
        """(1 + x) + (2 + y) = 3 + x + y."""
        var_names = ("x", "y")
        one = TruncatedSeries.from_scalar(1.0, var_names)
        two = TruncatedSeries.from_scalar(2.0, var_names)
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        s1 = one + x  # 1 + x
        s2 = two + y  # 2 + y
        result = s1 + s2

        np.testing.assert_allclose(result.coeffs[0], 3.0)  # 1 + 2
        np.testing.assert_allclose(result.coeffs[1], 1.0)  # x
        np.testing.assert_allclose(result.coeffs[2], 1.0)  # y

    def test_add_scalar_to_series(self):
        """x + 5.0 should add to constant term."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = x + 5.0

        np.testing.assert_allclose(result.coeffs[0], 5.0)
        np.testing.assert_allclose(result.coeffs[1], 1.0)

    def test_radd_scalar(self):
        """5.0 + x should also work."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = 5.0 + x

        np.testing.assert_allclose(result.coeffs[0], 5.0)
        np.testing.assert_allclose(result.coeffs[1], 1.0)


# =============================================================================
# Test 4: Subtraction
# =============================================================================

class TestSubtraction:
    """Test series subtraction."""

    def test_subtract_variables(self):
        """x - y should have coeff +1 at mask 1, -1 at mask 2."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        s = x - y

        np.testing.assert_allclose(s.coeffs[1], 1.0)
        np.testing.assert_allclose(s.coeffs[2], -1.0)

    def test_subtract_scalar(self):
        """x - 2.0 should have -2.0 constant term."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = x - 2.0

        np.testing.assert_allclose(result.coeffs[0], -2.0)
        np.testing.assert_allclose(result.coeffs[1], 1.0)

    def test_rsub_scalar(self):
        """2.0 - x should have +2.0 constant and -1.0 for x."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = 2.0 - x

        np.testing.assert_allclose(result.coeffs[0], 2.0)
        np.testing.assert_allclose(result.coeffs[1], -1.0)


# =============================================================================
# Test 5: Exponentiation
# =============================================================================

class TestExp:
    """Test exp() for nilpotent series."""

    def test_exp_of_zero(self):
        """exp(0) = 1."""
        var_names = ("x", "y")
        zero = TruncatedSeries.from_scalar(0.0, var_names)

        result = zero.exp()

        np.testing.assert_allclose(result.coeffs[0], 1.0)

    def test_exp_of_constant(self):
        """exp(c) = e^c for constant series."""
        var_names = ("x", "y")
        c = 2.5
        s = TruncatedSeries.from_scalar(c, var_names)

        result = s.exp()

        np.testing.assert_allclose(result.coeffs[0], np.exp(c))

    def test_exp_of_single_nilpotent(self):
        """exp(x) = 1 + x (since x^2 = 0)."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = x.exp()

        np.testing.assert_allclose(result.coeffs[0], 1.0)
        np.testing.assert_allclose(result.coeffs[1], 1.0)

    def test_exp_of_x_plus_y(self):
        """exp(x + y) = 1 + x + y + xy (x^2 = y^2 = 0, xy^2 etc vanish)."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        s = x + y
        result = s.exp()

        np.testing.assert_allclose(result.coeffs[0], 1.0)   # 1
        np.testing.assert_allclose(result.coeffs[1], 1.0)   # x
        np.testing.assert_allclose(result.coeffs[2], 1.0)   # y
        np.testing.assert_allclose(result.coeffs[3], 1.0)   # xy (from 1/2! * (x+y)^2 = 1/2 * 2xy = xy)

    def test_exp_with_constant_plus_nilpotent(self):
        """exp(c + x) = e^c * (1 + x) = e^c + e^c * x."""
        var_names = ("x", "y")
        c = 1.5
        const = TruncatedSeries.from_scalar(c, var_names)
        x = TruncatedSeries.variable("x", var_names)

        s = const + x
        result = s.exp()

        e_c = np.exp(c)
        np.testing.assert_allclose(result.coeffs[0], e_c)
        np.testing.assert_allclose(result.coeffs[1], e_c)

    def test_exp_scaled_variable(self):
        """exp(2x) = 1 + 2x."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        s = x * 2.0
        result = s.exp()

        np.testing.assert_allclose(result.coeffs[0], 1.0)
        np.testing.assert_allclose(result.coeffs[1], 2.0)

    def test_exp_three_variables(self):
        """exp(x + y + z) with 3 nilpotent vars."""
        var_names = ("x", "y", "z")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)
        z = TruncatedSeries.variable("z", var_names)

        s = x + y + z
        result = s.exp()

        # exp(x+y+z) = 1 + (x+y+z) + (xy+xz+yz)/1! + xyz/1!
        # Actually: exp(N) where N^2/2! contributes, N^3/3! contributes...
        # (x+y+z)^2 = 2xy + 2xz + 2yz (since x^2=y^2=z^2=0)
        # (x+y+z)^2/2! = xy + xz + yz
        # (x+y+z)^3/3! = 6xyz/6 = xyz
        np.testing.assert_allclose(result.coeffs[0], 1.0)   # 1
        np.testing.assert_allclose(result.coeffs[1], 1.0)   # x
        np.testing.assert_allclose(result.coeffs[2], 1.0)   # y
        np.testing.assert_allclose(result.coeffs[4], 1.0)   # z
        np.testing.assert_allclose(result.coeffs[3], 1.0)   # xy (mask 1|2)
        np.testing.assert_allclose(result.coeffs[5], 1.0)   # xz (mask 1|4)
        np.testing.assert_allclose(result.coeffs[6], 1.0)   # yz (mask 2|4)
        np.testing.assert_allclose(result.coeffs[7], 1.0)   # xyz (mask 1|2|4)


# =============================================================================
# Test 6: Derivative Extraction
# =============================================================================

class TestExtract:
    """Test extracting derivative coefficients."""

    def test_extract_constant(self):
        """Extract () from constant series gives the constant."""
        var_names = ("x", "y")
        s = TruncatedSeries.from_scalar(5.0, var_names)

        result = s.extract(())

        np.testing.assert_allclose(result, 5.0)

    def test_extract_x_coefficient(self):
        """Extract ('x',) from 2 + 3x gives 3."""
        var_names = ("x", "y")
        const = TruncatedSeries.from_scalar(2.0, var_names)
        x = TruncatedSeries.variable("x", var_names)

        s = const + x * 3.0  # 2 + 3x
        result = s.extract(("x",))

        np.testing.assert_allclose(result, 3.0)

    def test_extract_xy_coefficient(self):
        """Extract ('x', 'y') from 5 + 2x + 3xy gives 3."""
        var_names = ("x", "y")
        const = TruncatedSeries.from_scalar(5.0, var_names)
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        s = const + x * 2.0 + (x * y) * 3.0  # 5 + 2x + 3xy

        result = s.extract(("x", "y"))
        np.testing.assert_allclose(result, 3.0)

    def test_extract_nonexistent_term(self):
        """Extract ('y',) from series without y term gives 0."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = x.extract(("y",))

        np.testing.assert_allclose(result, 0.0)

    def test_extract_order_independent(self):
        """extract(('x', 'y')) == extract(('y', 'x'))."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        s = x * y * 7.0

        result_xy = s.extract(("x", "y"))
        result_yx = s.extract(("y", "x"))

        np.testing.assert_allclose(result_xy, result_yx)

    def test_extract_three_variable_term(self):
        """Extract xyz coefficient."""
        var_names = ("x", "y", "z")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)
        z = TruncatedSeries.variable("z", var_names)

        s = x * y * z * 5.0

        result = s.extract(("x", "y", "z"))
        np.testing.assert_allclose(result, 5.0)


# =============================================================================
# Test 7: Numpy Array Coefficients (Grid Evaluation)
# =============================================================================

class TestArrayCoefficients:
    """Test that coefficients can be numpy arrays for vectorized evaluation."""

    def test_from_scalar_array(self):
        """from_scalar with array creates array constant term."""
        var_names = ("x", "y")
        values = np.array([1.0, 2.0, 3.0])
        s = TruncatedSeries.from_scalar(values, var_names)

        np.testing.assert_array_equal(s.coeffs[0], values)

    def test_multiply_scalar_array(self):
        """x * array scales the x coefficient."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        scaled = x * np.array([1.0, 2.0, 3.0])

        np.testing.assert_array_equal(scaled.coeffs[1], np.array([1.0, 2.0, 3.0]))

    def test_add_array_coefficients(self):
        """Adding series with array coefficients broadcasts correctly."""
        var_names = ("x", "y")
        a = TruncatedSeries.from_scalar(np.array([1.0, 2.0]), var_names)
        b = TruncatedSeries.from_scalar(np.array([3.0, 4.0]), var_names)

        result = a + b

        np.testing.assert_array_equal(result.coeffs[0], np.array([4.0, 6.0]))

    def test_multiply_array_coefficients(self):
        """(a + bx) * (c + dy) with arrays broadcasts correctly."""
        var_names = ("x", "y")

        # a + bx where a=[1,2], b=[3,4]
        a = TruncatedSeries.from_scalar(np.array([1.0, 2.0]), var_names)
        x = TruncatedSeries.variable("x", var_names)
        s1 = a + x * np.array([3.0, 4.0])

        # c + dy where c=[5,6], d=[7,8]
        c = TruncatedSeries.from_scalar(np.array([5.0, 6.0]), var_names)
        y = TruncatedSeries.variable("y", var_names)
        s2 = c + y * np.array([7.0, 8.0])

        result = s1 * s2

        # constant: a*c = [1*5, 2*6] = [5, 12]
        np.testing.assert_array_equal(result.coeffs[0], np.array([5.0, 12.0]))
        # x: b*c = [3*5, 4*6] = [15, 24]
        np.testing.assert_array_equal(result.coeffs[1], np.array([15.0, 24.0]))
        # y: a*d = [1*7, 2*8] = [7, 16]
        np.testing.assert_array_equal(result.coeffs[2], np.array([7.0, 16.0]))
        # xy: b*d = [3*7, 4*8] = [21, 32]
        np.testing.assert_array_equal(result.coeffs[3], np.array([21.0, 32.0]))

    def test_exp_with_array_coefficients(self):
        """exp(c + ax) with array c and a."""
        var_names = ("x", "y")
        c_vals = np.array([0.0, 1.0, 2.0])
        a_vals = np.array([1.0, 2.0, 3.0])

        const = TruncatedSeries.from_scalar(c_vals, var_names)
        x = TruncatedSeries.variable("x", var_names)
        s = const + x * a_vals

        result = s.exp()

        # exp(c + ax) = e^c * (1 + ax) = e^c + a*e^c*x
        e_c = np.exp(c_vals)
        np.testing.assert_allclose(result.coeffs[0], e_c)
        np.testing.assert_allclose(result.coeffs[1], a_vals * e_c)

    def test_extract_returns_array(self):
        """extract() with array coefficients returns array."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        s = x * np.array([2.0, 4.0, 6.0])
        result = s.extract(("x",))

        np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))


# =============================================================================
# Test 8: Negation
# =============================================================================

class TestNegation:
    """Test unary negation."""

    def test_neg_variable(self):
        """-x should have coefficient -1 at mask 1."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        neg_x = -x

        np.testing.assert_allclose(neg_x.coeffs[1], -1.0)

    def test_neg_series(self):
        """-(1 + 2x) = -1 - 2x."""
        var_names = ("x", "y")
        one = TruncatedSeries.from_scalar(1.0, var_names)
        x = TruncatedSeries.variable("x", var_names)

        s = one + x * 2.0
        neg_s = -s

        np.testing.assert_allclose(neg_s.coeffs[0], -1.0)
        np.testing.assert_allclose(neg_s.coeffs[1], -2.0)


# =============================================================================
# Test 9: Edge Cases and Integration
# =============================================================================

class TestEdgeCases:
    """Test edge cases and combined operations."""

    def test_six_variables_max_term(self):
        """Six variables should support xyz... term at mask 63."""
        var_names = ("x1", "x2", "x3", "y1", "y2", "y3")
        vars = [TruncatedSeries.variable(name, var_names) for name in var_names]

        # Multiply all together
        product = vars[0]
        for v in vars[1:]:
            product = product * v

        # Should have term at mask 63 (111111 binary)
        assert 63 in product.coeffs
        np.testing.assert_allclose(product.coeffs[63], 1.0)

    def test_empty_coeffs_initialization(self):
        """Series with None coeffs should initialize to {0: 0.0}."""
        var_names = ("x", "y")
        s = TruncatedSeries(var_names, None)

        assert 0 in s.coeffs

    def test_chained_operations(self):
        """Complex expression: exp(R*(a + bx + cy)) derivative extraction."""
        var_names = ("x", "y")
        R = 1.5
        a_val = 0.5
        b_val = 2.0
        c_val = 3.0

        const = TruncatedSeries.from_scalar(a_val, var_names)
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        # S = R*(a + bx + cy)
        s = (const + x * b_val + y * c_val) * R

        # exp(S) = exp(R*a) * exp(R*bx + R*cy)
        result = s.exp()

        # d/dx at x=y=0: R*b * exp(R*a)
        dx = result.extract(("x",))
        expected_dx = R * b_val * np.exp(R * a_val)
        np.testing.assert_allclose(dx, expected_dx)

        # d^2/dxdy at x=y=0: R^2*b*c * exp(R*a)
        dxdy = result.extract(("x", "y"))
        expected_dxdy = R**2 * b_val * c_val * np.exp(R * a_val)
        np.testing.assert_allclose(dxdy, expected_dxdy)


# =============================================================================
# Test 10: Defensive Invariants (Pitfall Guards)
# =============================================================================

class TestDefensiveInvariants:
    """
    Test guards against silent errors from misuse.

    These tests validate that the implementation catches common mistakes
    that would otherwise produce mathematically incorrect results silently.
    """

    # --- Pitfall A: var_names mismatch between series ---

    def test_add_mismatched_var_names_raises(self):
        """Adding series with different var_names must raise ValueError."""
        s1 = TruncatedSeries.from_scalar(1.0, ("x", "y"))
        s2 = TruncatedSeries.from_scalar(2.0, ("a", "b"))

        with pytest.raises(ValueError, match="different var_names"):
            s1 + s2

    def test_mul_mismatched_var_names_raises(self):
        """Multiplying series with different var_names must raise ValueError."""
        s1 = TruncatedSeries.variable("x", ("x", "y"))
        s2 = TruncatedSeries.variable("a", ("a", "b"))

        with pytest.raises(ValueError, match="different var_names"):
            s1 * s2

    def test_sub_mismatched_var_names_raises(self):
        """Subtracting series with different var_names must raise ValueError."""
        s1 = TruncatedSeries.from_scalar(1.0, ("x", "y"))
        s2 = TruncatedSeries.from_scalar(2.0, ("y", "x"))  # Different order!

        with pytest.raises(ValueError, match="different var_names"):
            s1 - s2

    def test_var_names_order_matters(self):
        """var_names ordering is significant - ('x','y') != ('y','x')."""
        # This is critical: x maps to bit 0 in one, bit 1 in other
        s1 = TruncatedSeries.variable("x", ("x", "y"))  # x at mask 1
        s2 = TruncatedSeries.variable("x", ("y", "x"))  # x at mask 2

        with pytest.raises(ValueError, match="different var_names"):
            s1 + s2

    # --- Pitfall B: duplicate variables in extract ---

    def test_extract_duplicate_var_returns_zero(self):
        """extract(('x', 'x')) returns 0 since x^2 = 0."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        # x has coefficient 1 for x term, but x^2 = 0
        result = x.extract(("x", "x"))

        np.testing.assert_allclose(result, 0.0)

    def test_extract_duplicate_in_mixed_returns_zero(self):
        """extract(('x', 'y', 'x')) returns 0 (x appears twice)."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)

        s = x * y * 5.0  # Has xy term with coefficient 5

        # But x*x*y = 0, so should return 0
        result = s.extract(("x", "y", "x"))

        np.testing.assert_allclose(result, 0.0)

    def test_extract_triple_duplicate_returns_zero(self):
        """extract(('x', 'x', 'x')) returns 0."""
        var_names = ("x", "y", "z")
        x = TruncatedSeries.variable("x", var_names)

        result = x.extract(("x", "x", "x"))

        np.testing.assert_allclose(result, 0.0)

    # --- Pitfall C: duplicate variable names in construction ---

    def test_duplicate_var_names_in_init_raises(self):
        """Constructing series with duplicate var_names must raise."""
        with pytest.raises(ValueError, match="unique"):
            TruncatedSeries(("x", "x"), {0: np.array(1.0)})

    def test_duplicate_var_names_in_from_scalar_raises(self):
        """from_scalar with duplicate var_names must raise."""
        with pytest.raises(ValueError, match="unique"):
            TruncatedSeries.from_scalar(1.0, ("a", "b", "a"))

    def test_duplicate_var_names_in_variable_raises(self):
        """variable() with duplicate var_names must raise."""
        with pytest.raises(ValueError, match="unique"):
            TruncatedSeries.variable("x", ("x", "y", "x"))

    # --- Additional safety tests ---

    def test_scalar_ops_dont_require_var_match(self):
        """Scalar operations should work without var_names issues."""
        s = TruncatedSeries.variable("x", ("x", "y"))

        # These should all work fine
        result = s + 5.0
        result = 5.0 + s
        result = s * 3.0
        result = 3.0 * s
        result = s - 2.0
        result = 2.0 - s

        np.testing.assert_allclose(result.coeffs[0], 2.0)  # From 2.0 - s
        np.testing.assert_allclose(result.coeffs[1], -1.0)

    def test_compatible_series_work_fine(self):
        """Series with same var_names should combine correctly."""
        var_names = ("x", "y", "z")
        x = TruncatedSeries.variable("x", var_names)
        y = TruncatedSeries.variable("y", var_names)
        z = TruncatedSeries.variable("z", var_names)

        # Should work without raising
        s = (x + y + z) * (x + y + z)

        # xyz term exists from (x+y+z)^2 expansion... actually no, that's wrong
        # (x+y+z)^2 = 2xy + 2xz + 2yz (no xyz term, that comes from cube)
        np.testing.assert_allclose(s.coeffs.get(3, 0.0), 2.0)  # xy
        np.testing.assert_allclose(s.coeffs.get(5, 0.0), 2.0)  # xz
        np.testing.assert_allclose(s.coeffs.get(6, 0.0), 2.0)  # yz


# =============================================================================
# Test 11: NumPy Scalar Handling
# =============================================================================

class TestNumpyScalars:
    """
    Test that NumPy scalar types (np.float64, np.int64, etc.) work correctly.

    These types are NOT Python float/int, but should behave as scalars.
    This is important because many NumPy operations return these types
    (e.g., np.sum, array indexing, etc.).
    """

    def test_mul_np_float64(self):
        """x * np.float64(2.0) should work."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = x * np.float64(2.0)

        np.testing.assert_allclose(result.coeffs[1], 2.0)

    def test_rmul_np_float64(self):
        """np.float64(3.0) * x should work."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = np.float64(3.0) * x

        np.testing.assert_allclose(result.coeffs[1], 3.0)

    def test_mul_np_int64(self):
        """x * np.int64(5) should work."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = x * np.int64(5)

        np.testing.assert_allclose(result.coeffs[1], 5.0)

    def test_mul_np_sum_result(self):
        """Multiplying by result of np.sum() should work."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        weights = np.array([1.0, 2.0, 3.0])
        total = np.sum(weights)  # Returns np.float64

        result = x * total

        np.testing.assert_allclose(result.coeffs[1], 6.0)

    def test_add_np_float64(self):
        """x + np.float64(5.0) should work."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = x + np.float64(5.0)

        np.testing.assert_allclose(result.coeffs[0], 5.0)
        np.testing.assert_allclose(result.coeffs[1], 1.0)

    def test_sub_np_float64(self):
        """x - np.float64(2.0) should work."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = x - np.float64(2.0)

        np.testing.assert_allclose(result.coeffs[0], -2.0)
        np.testing.assert_allclose(result.coeffs[1], 1.0)

    def test_rsub_np_float64(self):
        """np.float64(3.0) - x should work."""
        var_names = ("x", "y")
        x = TruncatedSeries.variable("x", var_names)

        result = np.float64(3.0) - x

        np.testing.assert_allclose(result.coeffs[0], 3.0)
        np.testing.assert_allclose(result.coeffs[1], -1.0)
