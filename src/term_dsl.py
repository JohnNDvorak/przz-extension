"""
src/term_dsl.py
Term DSL for PRZZ pair computations.

This module defines the data structures for representing integral terms
in the PRZZ framework. Each term is a derivative-at-0 of a multi-dimensional
integral with factors: P, Q, exp, and algebraic prefactors.

Key components:
- SeriesContext: Owns canonical var_names for all series operations
- AffineExpr: Represents a0(u,t) + Σᵢ aᵢ(u,t) · varᵢ
- PolyFactor: Polynomial factor P(argument)^power
- ExpFactor: Exponential factor exp(scale * argument)
- Term: Complete specification of one integral term

Design principles (from GPT review):
- Dtype-safe lifting: preserves complex types
- Shape invariants: all evaluations return grid-shaped arrays
- Zero-pruning: skip identically-zero coefficients to keep len(lin) minimal
- composition.py is the ONLY Taylor expansion engine
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Callable, Union, Tuple, List, Optional
import numpy as np

from src.series import TruncatedSeries
from src.composition import compose_polynomial_on_affine, compose_exp_on_affine, PolyLike


# Type alias: scalar, precomputed array, or callable
GridFunc = Union[float, complex, np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]


def _eval_gridfunc(
    f: GridFunc,
    U: np.ndarray,
    T: np.ndarray,
    context: str = "GridFunc"
) -> np.ndarray:
    """
    Evaluate a GridFunc on the grid, enforcing shape invariants.

    This centralizes grid evaluation and catches shape mismatches early.

    Args:
        f: GridFunc to evaluate (scalar, array, or callable)
        U: Grid array for u variable (defines expected shape)
        T: Grid array for t variable
        context: Description for error messages (e.g., "a0" or "coeff['x1']")

    Returns:
        Array with shape U.shape and appropriate dtype

    Raises:
        ValueError: If callable returns wrong shape
    """
    # Handle callable
    if callable(f):
        val = f(U, T)
    else:
        val = f

    arr = np.asarray(val)

    # Scalar -> broadcast to grid shape
    if arr.shape == ():
        return np.full(U.shape, arr, dtype=np.result_type(U, arr))

    # Already an array - verify shape matches
    if arr.shape != U.shape:
        raise ValueError(
            f"{context} returned shape {arr.shape}, expected {U.shape}. "
            f"GridFunc must return scalar or array matching grid shape."
        )

    # Ensure consistent dtype (preserve complex)
    return arr.astype(np.result_type(U, arr), copy=False)


@dataclass
class SeriesContext:
    """
    Context for creating series with canonical variable names.

    All series created through this context share the same var_names,
    preventing mismatches in binary operations.

    Attributes:
        var_names: Tuple of variable names in canonical order

    Example:
        >>> ctx = SeriesContext(var_names=("x1", "y1"))
        >>> x = ctx.variable_series("x1")
        >>> y = ctx.variable_series("y1")
        >>> product = x * y  # Safe - same var_names
    """
    var_names: Tuple[str, ...]

    def zero_series(self) -> TruncatedSeries:
        """Create a zero series with this context's var_names."""
        return TruncatedSeries.from_scalar(0.0, self.var_names)

    def scalar_series(self, value: Union[float, np.ndarray]) -> TruncatedSeries:
        """Create a scalar (constant) series with this context's var_names."""
        return TruncatedSeries.from_scalar(value, self.var_names)

    def variable_series(self, name: str) -> TruncatedSeries:
        """Create a series representing a single variable."""
        if name not in self.var_names:
            raise ValueError(f"Variable '{name}' not in var_names {self.var_names}")
        return TruncatedSeries.variable(name, self.var_names)


@dataclass(frozen=True)
class AffineExpr:
    """
    Represents an affine expression: a₀(u,t) + Σᵢ aᵢ(u,t) · varᵢ

    The coefficients can be either:
    - Constants (float/complex): lifted to grid shape via np.full
    - Callables: called with (U, T) arrays to evaluate on grid

    Attributes:
        a0: Base term - constant or callable(U, T) -> array
        var_coeffs: Dict mapping variable name to its coefficient
                    Each coefficient is constant or callable(U, T) -> array

    Shape invariants:
    - evaluate_a0(U, T) returns array with shape U.shape
    - evaluate_coeff(name, U, T) returns array with shape U.shape

    Example:
        >>> # Expression: t + θ·t·x + (θ·t - θ)·y
        >>> expr = AffineExpr(
        ...     a0=lambda U, T: T,
        ...     var_coeffs={
        ...         "x": lambda U, T: theta * T,
        ...         "y": lambda U, T: theta * T - theta
        ...     }
        ... )
    """
    a0: GridFunc
    var_coeffs: Dict[str, GridFunc]

    def evaluate_a0(self, U: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Evaluate the base term on grid. Preserves dtype for complex.

        Args:
            U: Grid array for u variable
            T: Grid array for t variable

        Returns:
            Array with shape U.shape

        Raises:
            ValueError: If callable returns wrong shape
        """
        return _eval_gridfunc(self.a0, U, T, context="a0")

    def evaluate_coeff(self, var: str, U: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Evaluate coefficient of var on grid. Preserves dtype for complex.

        Args:
            var: Variable name
            U: Grid array for u variable
            T: Grid array for t variable

        Returns:
            Array with shape U.shape (zeros if var not in var_coeffs)

        Raises:
            ValueError: If callable returns wrong shape
        """
        c = self.var_coeffs.get(var, 0.0)
        return _eval_gridfunc(c, U, T, context=f"coeff['{var}']")

    def to_u0_lin(
        self,
        U: np.ndarray,
        T: np.ndarray,
        ctx: SeriesContext
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract (u0, lin) for use with compose_polynomial_on_affine.

        Zero-pruning: coefficients that are identically zero are excluded
        from lin, keeping len(lin) minimal for efficient truncation.

        Args:
            U: Grid array for u variable
            T: Grid array for t variable
            ctx: SeriesContext for variable names

        Returns:
            Tuple of:
            - u0: evaluated base term
            - lin: dict mapping var_name to evaluated coefficient (non-zero only)

        Raises:
            ValueError: If any var_coeffs key is not in ctx.var_names
        """
        u0 = self.evaluate_a0(U, T)
        lin = {}
        for var in self.var_coeffs.keys():
            # Validate var is in context
            if var not in ctx.var_names:
                raise ValueError(
                    f"AffineExpr var_coeffs key '{var}' not in "
                    f"ctx.var_names {ctx.var_names}"
                )
            coeff = self.evaluate_coeff(var, U, T)
            # Zero-pruning: skip identically zero coefficients
            if not np.all(coeff == 0):
                lin[var] = coeff
        return u0, lin

    def to_series(
        self,
        U: np.ndarray,
        T: np.ndarray,
        ctx: SeriesContext
    ) -> TruncatedSeries:
        """
        Convert to a TruncatedSeries representing the affine expression.

        Zero-pruning: coefficients that are identically zero are not added
        to the series, keeping the coefficient dict minimal.

        Args:
            U: Grid array for u variable
            T: Grid array for t variable
            ctx: SeriesContext for variable names

        Returns:
            TruncatedSeries: a0 + Σ aᵢ·varᵢ (non-zero terms only)

        Raises:
            ValueError: If any var_coeffs key is not in ctx.var_names
        """
        result = ctx.scalar_series(self.evaluate_a0(U, T))
        for var in self.var_coeffs.keys():
            # Validate var is in context
            if var not in ctx.var_names:
                raise ValueError(
                    f"AffineExpr var_coeffs key '{var}' not in "
                    f"ctx.var_names {ctx.var_names}"
                )
            coeff = self.evaluate_coeff(var, U, T)
            # Zero-pruning: skip identically zero coefficients
            if not np.all(coeff == 0):
                result = result + ctx.variable_series(var) * coeff
        return result


@dataclass(frozen=True)
class PolyFactor:
    """
    A polynomial factor: P_name(argument)^power

    Attributes:
        poly_name: Name identifier ("P1", "P2", "P3", "Q")
        argument: AffineExpr for the polynomial argument
        power: Exponent (default 1). For Q² use power=2.

    Example:
        >>> # P₁(x + u)
        >>> factor = PolyFactor(
        ...     poly_name="P1",
        ...     argument=AffineExpr(
        ...         a0=lambda U, T: U,
        ...         var_coeffs={"x1": 1.0}
        ...     )
        ... )
    """
    poly_name: str
    argument: AffineExpr
    power: int = 1

    def evaluate(
        self,
        poly: PolyLike,
        U: np.ndarray,
        T: np.ndarray,
        ctx: SeriesContext
    ) -> TruncatedSeries:
        """
        Evaluate the polynomial factor as a TruncatedSeries.

        Uses compose_polynomial_on_affine from composition.py.

        Args:
            poly: Polynomial object with eval_deriv(x, k)
            U: Grid array for u variable
            T: Grid array for t variable
            ctx: SeriesContext for variable names

        Returns:
            TruncatedSeries representing P(argument)^power
        """
        u0, lin = self.argument.to_u0_lin(U, T, ctx)

        # Compose polynomial on affine expression
        series = compose_polynomial_on_affine(poly, u0, lin, ctx.var_names)

        # Apply power if > 1
        result = series
        for _ in range(self.power - 1):
            result = result * series

        return result


@dataclass(frozen=True)
class ExpFactor:
    """
    An exponential factor: exp(scale * argument)

    Attributes:
        scale: Scaling constant (typically R or 2R)
        argument: AffineExpr for the exponential argument

    Example:
        >>> # exp(R * (t + θ·t·x + (θ·t-θ)·y))
        >>> factor = ExpFactor(
        ...     scale=R,
        ...     argument=Q_arg_alpha
        ... )
    """
    scale: float
    argument: AffineExpr

    def evaluate(
        self,
        U: np.ndarray,
        T: np.ndarray,
        ctx: SeriesContext
    ) -> TruncatedSeries:
        """
        Evaluate the exponential factor as a TruncatedSeries.

        Uses compose_exp_on_affine from composition.py for consistency
        with the "single Taylor engine" principle.

        Args:
            U: Grid array for u variable
            T: Grid array for t variable
            ctx: SeriesContext for variable names

        Returns:
            TruncatedSeries representing exp(scale * argument)
        """
        # Extract (u0, lin) with zero-pruning and validation
        u0, lin = self.argument.to_u0_lin(U, T, ctx)

        # Delegate to composition module
        return compose_exp_on_affine(self.scale, u0, lin, ctx.var_names)


@dataclass
class Term:
    """
    Complete specification of one integral term.

    Each Term represents a derivative-at-0 of a multi-dimensional integral
    with polynomial, exponential, and algebraic factors.

    Attributes:
        name: Identifier (e.g., "I1_11", "I2_12")
        pair: Tuple (ℓ₁, ℓ₂) identifying the pair type
        przz_reference: Reference to PRZZ paper section
        vars: Formal variables in canonical order
        deriv_orders: Dict mapping var -> derivative order (all 1 for d=1)
        domain: Integration domain description
        numeric_prefactor: Constant multiplier
        algebraic_prefactor: Optional AffineExpr prefactor (e.g., (θS+1)/θ)
        poly_prefactors: List of grid functions (e.g., (1-u)^2)
        poly_factors: List of PolyFactor objects
        exp_factors: List of ExpFactor objects

    Invariants (enforced in __post_init__):
        - Phase 0: all derivative orders must be 0 or 1
        - vars and deriv_orders.keys() must match exactly
        - Higher-order derivatives require d>1 infrastructure not yet implemented

    Example:
        >>> term = Term(
        ...     name="I1_11",
        ...     pair=(1, 1),
        ...     przz_reference="Section 6.2.1, I₁",
        ...     vars=("x1", "y1"),
        ...     deriv_orders={"x1": 1, "y1": 1},
        ...     domain="[0,1]^2",
        ...     numeric_prefactor=1.0,
        ...     algebraic_prefactor=None,
        ...     poly_prefactors=[lambda U, T: (1 - U)**2],
        ...     poly_factors=[PolyFactor("P1", P1_arg), ...],
        ...     exp_factors=[ExpFactor(R, Q_arg_alpha), ...]
        ... )
    """
    # Identification
    name: str
    pair: Tuple[int, int]
    przz_reference: Optional[str]

    # Formal variables
    vars: Tuple[str, ...]
    deriv_orders: Dict[str, int]

    # Integration
    domain: str

    # Prefactors
    numeric_prefactor: float
    algebraic_prefactor: Optional[AffineExpr]
    poly_prefactors: List[GridFunc]

    # Factors to expand and multiply
    poly_factors: List[PolyFactor]
    exp_factors: List[ExpFactor]

    def __post_init__(self):
        """Validate Term invariants after construction."""
        # Validate vars/deriv_orders consistency
        vars_set = set(self.vars)
        deriv_set = set(self.deriv_orders.keys())

        extra = deriv_set - vars_set
        if extra:
            raise ValueError(
                f"deriv_orders contains variables not in vars: {extra}. "
                f"vars={self.vars}, deriv_orders keys={list(self.deriv_orders.keys())}"
            )

        missing = vars_set - deriv_set
        if missing:
            raise ValueError(
                f"vars contains variables not in deriv_orders: {missing}. "
                f"For d=1, every var in vars must have an entry in deriv_orders."
            )

        # Phase 0: Enforce d=1 (all derivative orders must be 0 or 1)
        # Higher-order derivatives (d>1) require infrastructure not yet implemented
        for var, order in self.deriv_orders.items():
            if order > 1:
                raise ValueError(
                    f"Derivative order > 1 not supported in Phase 0 (d=1 only). "
                    f"Got deriv_orders['{var}'] = {order}. "
                    f"Higher-order derivatives require d>1 infrastructure."
                )
            if order < 0:
                raise ValueError(
                    f"Derivative order must be non-negative, "
                    f"got deriv_orders['{var}'] = {order}"
                )

    def total_vars(self) -> int:
        """Number of formal variables."""
        return len(self.vars)

    def target_mask(self) -> int:
        """
        Bitmask for the target derivative (all 1's for d=1).

        For n variables, returns 2^n - 1 (all bits set).
        This is the mask to extract the coefficient of x₁x₂...xₙ.
        """
        return (1 << len(self.vars)) - 1

    def deriv_tuple(self) -> Tuple[str, ...]:
        """
        Return tuple of variables to differentiate (for series.extract()).

        For d=1, this returns all vars with deriv_orders[var] == 1.
        The order matches self.vars (canonical order).

        Returns:
            Tuple of variable names to pass to series.extract()
        """
        return tuple(v for v in self.vars if self.deriv_orders.get(v, 0) == 1)

    def create_context(self) -> SeriesContext:
        """Create a SeriesContext from this term's variables."""
        return SeriesContext(var_names=self.vars)
