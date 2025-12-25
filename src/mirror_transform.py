"""src/mirror_transform.py

Utilities for constructing "mirror-transformed" Term objects.

Context
-------
The DSL evaluation pipeline (`src/evaluate.py`) treats `kernel_regime="paper"`
as TeX-driven (ω → Case B/C selection). That regime fixes κ/κ* ratio
directionally, but absolute c values are still far from PRZZ targets.

One high-probability missing ingredient is the PRZZ "mirror" recombination,
which involves evaluating a transformed integrand (e.g., exp factors with
opposite sign) and combining with a large multiplier.

This module provides small, explicit transforms for Terms so we can run
controlled diagnostics without conflating:
- flipping exp factor signs, vs
- globally flipping R (which also flips Case C kernel internal exponents).

Q-Operator Shift Identity (GPT Analysis 2025-12-19)
---------------------------------------------------
In PRZZ, the mirror term appears INSIDE the Q-differential operator:

    I_d = Q(-∂_α/logT) Q(-∂_β/logT) [ I_{1,d}(α,β) + T^{-α-β} I_{1,d}(-β,-α) ]

Since D_α(T^{-α-β}) = T^{-α-β}, we get the shift identity:

    Q(D_α) [T^{-α-β} F] = T^{-α-β} Q(1 + D_α) F

The mirror contribution uses Q(1+D), not Q(D)!

This means for the mirror term, Q-argument affine expressions should have
their constant term shifted by +1. This is implemented in `shift_q_argument_a0()`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, List, Optional

import numpy as np

from src.term_dsl import AffineExpr, ExpFactor, GridFunc, PolyFactor, Term


def _wrap_gridfunc_add(f: GridFunc, delta: float) -> GridFunc:
    """Return a GridFunc that adds a scalar `delta` to the original GridFunc."""

    if delta == 0.0:
        return f
    if callable(f):
        return lambda U, T, _f=f, _d=delta: _f(U, T) + _d
    return f + delta


def _wrap_gridfunc_t(f: GridFunc, t_map: Callable[[np.ndarray], np.ndarray]) -> GridFunc:
    """Return a GridFunc that applies T -> t_map(T) for callable GridFuncs."""

    if callable(f):
        return lambda U, T, _f=f, _t_map=t_map: _f(U, _t_map(T))
    return f


def transform_affine_expr_t(expr: AffineExpr, *, t_map: Callable[[np.ndarray], np.ndarray]) -> AffineExpr:
    """Transform an AffineExpr by mapping its (U,T) evaluation through T -> t_map(T)."""

    a0 = _wrap_gridfunc_t(expr.a0, t_map)
    var_coeffs = {k: _wrap_gridfunc_t(v, t_map) for k, v in expr.var_coeffs.items()}
    return replace(expr, a0=a0, var_coeffs=var_coeffs)


def transform_affine_expr_a0_shift(expr: AffineExpr, *, delta: float) -> AffineExpr:
    """Shift the AffineExpr constant term: a0(u,t) -> a0(u,t) + delta."""

    if delta == 0.0:
        return expr
    return replace(expr, a0=_wrap_gridfunc_add(expr.a0, delta))


def transform_exp_factors(
    exp_factors: Iterable[ExpFactor],
    *,
    scale_multiplier: float = 1.0,
    t_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> List[ExpFactor]:
    """Transform a list of ExpFactors.

    Args:
        exp_factors: original exp factors
        scale_multiplier: multiplies each ExpFactor.scale
        t_map: if provided, apply T -> t_map(T) to each ExpFactor.argument

    Returns:
        New list of ExpFactor objects.
    """

    out: List[ExpFactor] = []
    for ef in exp_factors:
        arg = ef.argument
        if t_map is not None:
            arg = transform_affine_expr_t(arg, t_map=t_map)
        out.append(replace(ef, scale=scale_multiplier * ef.scale, argument=arg))
    return out


def transform_q_poly_factors(
    poly_factors: Iterable[PolyFactor],
    *,
    a0_shift: float = 0.0,
) -> List[PolyFactor]:
    """Transform PolyFactors, applying a0 shifts only to Q(...) arguments."""

    if a0_shift == 0.0:
        return list(poly_factors)

    out: List[PolyFactor] = []
    for pf in poly_factors:
        if pf.poly_name == "Q":
            out.append(
                replace(
                    pf,
                    argument=transform_affine_expr_a0_shift(pf.argument, delta=a0_shift),
                )
            )
        else:
            out.append(pf)
    return out


def transform_term_exp_factors(
    term: Term,
    *,
    scale_multiplier: float = 1.0,
    t_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Term:
    """Return a Term with transformed exp_factors (all other fields unchanged)."""

    return Term(
        name=term.name,
        pair=term.pair,
        przz_reference=term.przz_reference,
        vars=term.vars,
        deriv_orders=term.deriv_orders,
        domain=term.domain,
        numeric_prefactor=term.numeric_prefactor,
        algebraic_prefactor=term.algebraic_prefactor,
        poly_prefactors=term.poly_prefactors,
        poly_factors=term.poly_factors,
        exp_factors=transform_exp_factors(
            term.exp_factors,
            scale_multiplier=scale_multiplier,
            t_map=t_map,
        ),
    )


def transform_term_q_factors(
    term: Term,
    *,
    q_a0_shift: float = 0.0,
) -> Term:
    """Return a Term with transformed Q(...) PolyFactors (all other fields unchanged)."""

    if q_a0_shift == 0.0:
        return term

    return Term(
        name=term.name,
        pair=term.pair,
        przz_reference=term.przz_reference,
        vars=term.vars,
        deriv_orders=term.deriv_orders,
        domain=term.domain,
        numeric_prefactor=term.numeric_prefactor,
        algebraic_prefactor=term.algebraic_prefactor,
        poly_prefactors=term.poly_prefactors,
        poly_factors=transform_q_poly_factors(term.poly_factors, a0_shift=q_a0_shift),
        exp_factors=term.exp_factors,
    )


def transform_terms_exp_factors(
    terms: Iterable[Term],
    *,
    scale_multiplier: float = 1.0,
    t_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> List[Term]:
    """Transform a list of Terms by modifying only exp_factors."""

    if scale_multiplier == 1.0 and t_map is None:
        return list(terms)

    return [
        transform_term_exp_factors(term, scale_multiplier=scale_multiplier, t_map=t_map)
        for term in terms
    ]


def transform_terms_q_factors(
    terms: Iterable[Term],
    *,
    q_a0_shift: float = 0.0,
) -> List[Term]:
    """Transform a list of Terms by modifying only Q(...) PolyFactors."""

    if q_a0_shift == 0.0:
        return list(terms)

    return [transform_term_q_factors(term, q_a0_shift=q_a0_shift) for term in terms]


# =============================================================================
# Q-ARGUMENT SHIFT TRANSFORMS (GPT 2025-12-19)
# =============================================================================


def shift_affine_expr_a0(expr: AffineExpr, *, shift: float = 1.0) -> AffineExpr:
    """Shift the constant term a0 of an AffineExpr by `shift`.

    For the Q(1+D) identity: if original a0 = f(U,T), new a0 = f(U,T) + shift.

    This implements the "+1" shift from the operator identity:
        Q(D) [T^{-α-β} F] = T^{-α-β} Q(1 + D) F

    Args:
        expr: Original AffineExpr
        shift: Amount to add to constant term (default +1.0)

    Returns:
        New AffineExpr with shifted a0
    """
    original_a0 = expr.a0

    if callable(original_a0):
        # a0 is a GridFunc: create new function that adds shift
        def shifted_a0(U: np.ndarray, T: np.ndarray, _orig=original_a0, _s=shift) -> np.ndarray:
            return _orig(U, T) + _s
        new_a0: GridFunc = shifted_a0
    else:
        # a0 is a constant
        new_a0 = original_a0 + shift

    return replace(expr, a0=new_a0)


def transform_term_q_shift(
    term: Term,
    *,
    shift: float = 1.0,
) -> Term:
    """Transform a Term by shifting Q-argument constant terms.

    This implements the Q(1+D) operator shift for mirror terms.

    The transform:
    1. For each poly_factor that is a Q polynomial, shift its argument.a0 by +1
    2. For each exp_factor, shift its argument.a0 by +1
       (exp factors also use Q-arguments in our DSL)

    Args:
        term: Original Term
        shift: Amount to shift constant terms (default +1.0)

    Returns:
        New Term with shifted Q-arguments
    """
    from src.term_dsl import PolyFactor

    # Transform poly_factors: shift Q-argument a0
    new_poly_factors: List[PolyFactor] = []
    for pf in term.poly_factors:
        if pf.poly_name == "Q":
            # This is a Q polynomial - shift its argument
            new_arg = shift_affine_expr_a0(pf.argument, shift=shift)
            new_poly_factors.append(replace(pf, argument=new_arg))
        else:
            # P polynomial - leave unchanged
            new_poly_factors.append(pf)

    # Transform exp_factors: shift argument a0
    # (exp factors use the same Q-argument structure)
    new_exp_factors: List[ExpFactor] = []
    for ef in term.exp_factors:
        new_arg = shift_affine_expr_a0(ef.argument, shift=shift)
        new_exp_factors.append(replace(ef, argument=new_arg))

    return Term(
        name=term.name + "_qshift",
        pair=term.pair,
        przz_reference=term.przz_reference,
        vars=term.vars,
        deriv_orders=term.deriv_orders,
        domain=term.domain,
        numeric_prefactor=term.numeric_prefactor,
        algebraic_prefactor=term.algebraic_prefactor,
        poly_prefactors=term.poly_prefactors,
        poly_factors=new_poly_factors,
        exp_factors=new_exp_factors,
    )


def transform_terms_q_shift(
    terms: Iterable[Term],
    *,
    shift: float = 1.0,
) -> List[Term]:
    """Transform a list of Terms by shifting Q-arguments.

    This is the main entry point for creating mirror terms using
    the Q(1+D) operator identity.
    """
    return [transform_term_q_shift(term, shift=shift) for term in terms]
