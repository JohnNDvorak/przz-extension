"""
src/evaluate.py
Evaluation pipeline for computing c from PRZZ terms.

This module implements the staged evaluation pipeline:
1. Build quadrature grid
2. Create series context from term
3. Multiply all formal-variable-dependent factors (poly_factors, exp_factors)
4. If algebraic_prefactor exists, multiply it in
5. Extract derivative coefficient via term.deriv_tuple()
6. Multiply by grid-only prefactors (numeric_prefactor, poly_prefactors)
7. Integrate with quadrature weights
8. Sum contributions from all terms

Key design principles:
- Always return per-term breakdown for debugging
- Shape coercion: extracted coefficients coerced to grid shape before integration
- Staged pipeline prevents silent errors
- Reference-based validation tests

PRZZ Reference: Section 6.2.1
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from src.term_dsl import Term, SeriesContext, PolyFactor, ExpFactor
from src.series import TruncatedSeries
from src.quadrature import tensor_grid_2d
from src.composition import PolyLike


@dataclass
class TermResult:
    """Result of evaluating a single term."""
    name: str
    value: float
    # For debugging: intermediate values
    extracted_coeff_sample: Optional[float] = None  # coeff at grid center
    series_term_count: Optional[int] = None  # number of series terms


@dataclass
class EvaluationResult:
    """Result of evaluating multiple terms."""
    total: float
    per_term: Dict[str, float]
    n: int
    # Additional metadata
    term_results: Optional[List[TermResult]] = None


def _coerce_to_grid_shape(
    coeff: Any,
    W: np.ndarray,
    term_name: str
) -> np.ndarray:
    """
    Coerce extracted coefficient to grid shape.

    This prevents silent broadcasting errors and makes shape mismatches
    fail immediately rather than producing wrong results.

    Args:
        coeff: Extracted coefficient (may be scalar or array)
        W: Weight grid (defines expected shape)
        term_name: For error messages

    Returns:
        Array with shape W.shape

    Raises:
        ValueError: If coeff has wrong non-scalar shape
    """
    arr = np.asarray(coeff)

    # Scalar -> broadcast to grid shape
    if arr.shape == ():
        return np.full_like(W, arr)

    # Already grid-shaped
    if arr.shape == W.shape:
        return arr

    # Wrong shape - fail immediately
    raise ValueError(
        f"Term '{term_name}': extracted coefficient has shape {arr.shape}, "
        f"expected scalar or {W.shape}"
    )


def evaluate_term(
    term: Term,
    polynomials: Dict[str, PolyLike],
    n: int,
    return_debug: bool = False
) -> TermResult:
    """
    Evaluate a single term's contribution to c.

    Implements the staged evaluation pipeline:
    1. Build quadrature grid
    2. Create series context
    3. Multiply all series factors (poly + exp)
    4. Multiply by algebraic prefactor if present
    5. Extract coefficient via deriv_tuple()
    6. Multiply by numeric_prefactor and poly_prefactors
    7. Integrate

    Args:
        term: Term to evaluate
        polynomials: Dict mapping poly_name to polynomial object
            e.g., {"P1": p1_poly, "Q": q_poly}
        n: Number of quadrature points per dimension
        return_debug: If True, include debug info in result

    Returns:
        TermResult with the term's contribution
    """
    # Stage 1: Build quadrature grid
    U, T, W = tensor_grid_2d(n)

    # Stage 2: Create series context
    ctx = term.create_context()

    # Stage 3: Build the series integrand from formal-var-dependent factors
    # Start with 1 (multiplicative identity)
    if len(term.vars) > 0:
        integrand = ctx.scalar_series(np.ones_like(U))
    else:
        # For terms with no formal variables, series is just scalar
        integrand = ctx.scalar_series(np.ones_like(U))

    # Multiply in all poly factors
    for factor in term.poly_factors:
        poly = polynomials.get(factor.poly_name)
        if poly is None:
            raise ValueError(
                f"Term '{term.name}': polynomial '{factor.poly_name}' not found"
            )
        factor_series = factor.evaluate(poly, U, T, ctx)
        integrand = integrand * factor_series

    # Multiply in all exp factors
    for factor in term.exp_factors:
        factor_series = factor.evaluate(U, T, ctx)
        integrand = integrand * factor_series

    # Stage 4: Multiply by algebraic prefactor if present
    if term.algebraic_prefactor is not None:
        prefactor_series = term.algebraic_prefactor.to_series(U, T, ctx)
        integrand = integrand * prefactor_series

    # Stage 5: Extract the derivative coefficient
    deriv_vars = term.deriv_tuple()
    coeff = integrand.extract(deriv_vars)

    # Coerce to grid shape (catches scalar 0 for missing terms)
    coeff = _coerce_to_grid_shape(coeff, W, term.name)

    # Stage 6: Multiply by grid-only prefactors
    # First: numeric prefactor (scalar)
    coeff = coeff * term.numeric_prefactor

    # Then: poly_prefactors (grid functions)
    for prefactor_func in term.poly_prefactors:
        prefactor_vals = prefactor_func(U, T)
        coeff = coeff * prefactor_vals

    # Stage 7: Integrate
    value = float(np.sum(W * coeff))

    # Build result
    result = TermResult(name=term.name, value=value)

    if return_debug:
        # Sample coefficient at grid center for debugging
        mid = n // 2
        result.extracted_coeff_sample = float(coeff[mid, mid])
        result.series_term_count = len(integrand.coeffs)

    return result


def evaluate_terms(
    terms: List[Term],
    polynomials: Dict[str, PolyLike],
    n: int,
    return_breakdown: bool = True
) -> EvaluationResult:
    """
    Evaluate multiple terms and sum their contributions.

    Args:
        terms: List of Term objects
        polynomials: Dict mapping poly_name to polynomial object
        n: Number of quadrature points per dimension
        return_breakdown: If True, include per-term breakdown

    Returns:
        EvaluationResult with total and optional breakdown
    """
    term_results = []
    per_term = {}
    total = 0.0

    for term in terms:
        result = evaluate_term(term, polynomials, n, return_debug=return_breakdown)
        term_results.append(result)
        per_term[term.name] = result.value
        total += result.value

    return EvaluationResult(
        total=total,
        per_term=per_term,
        n=n,
        term_results=term_results if return_breakdown else None
    )


def evaluate_c11(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    return_breakdown: bool = True
) -> EvaluationResult:
    """
    Evaluate c₁₁ from all (1,1) terms.

    Convenience function that builds all (1,1) terms and evaluates them.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "Q" mapping to polynomial objects
        return_breakdown: If True, include per-term breakdown

    Returns:
        EvaluationResult with c₁₁ and optional breakdown
    """
    from src.terms_k3_d1 import make_all_terms_11

    terms = make_all_terms_11(theta, R)
    return evaluate_terms(terms, polynomials, n, return_breakdown)


def convergence_sweep(
    theta: float,
    R: float,
    polynomials: Dict[str, PolyLike],
    ns: List[int] = None,
    n_ref: int = 200
) -> Dict[str, Any]:
    """
    Run convergence sweep at multiple quadrature resolutions.

    Args:
        theta: θ parameter
        R: R parameter
        polynomials: Dict with polynomial objects
        ns: List of n values to test (default [40, 60, 80, 100])
        n_ref: Reference n for computing errors

    Returns:
        Dict with:
        - 'reference': c₁₁ at n_ref
        - 'values': {n: c₁₁} for each n
        - 'errors': {n: |c₁₁(n) - reference|} for each n
        - 'per_term_ref': per-term breakdown at n_ref
    """
    if ns is None:
        ns = [40, 60, 80, 100]

    # Compute reference
    ref_result = evaluate_c11(theta, R, n_ref, polynomials, return_breakdown=True)
    reference = ref_result.total

    # Compute at each n
    values = {}
    errors = {}
    for n in ns:
        result = evaluate_c11(theta, R, n, polynomials, return_breakdown=False)
        values[n] = result.total
        errors[n] = abs(result.total - reference)

    return {
        'reference': reference,
        'n_ref': n_ref,
        'values': values,
        'errors': errors,
        'per_term_ref': ref_result.per_term
    }


# =============================================================================
# Specialized evaluation for analytic tests
# =============================================================================

def evaluate_I1_with_P1_Q1(
    theta: float,
    R: float,
    n: int
) -> float:
    """
    Evaluate I₁ with P≡1, Q≡1 for analytic comparison.

    When P=Q=1, all polynomial factors become 1, and the only
    formal-variable dependence comes from:
    - algebraic prefactor: (1/θ + x + y)
    - exp factors: exp(R·Arg_α) * exp(R·Arg_β)

    The xy coefficient can be computed analytically (see test).

    Args:
        theta: θ parameter
        R: R parameter
        n: Number of quadrature points

    Returns:
        I₁ contribution with P=Q=1
    """
    from src.terms_k3_d1 import make_I1_11
    from src.polynomials import Polynomial

    # Create constant polynomial P(x) = 1
    one_poly = Polynomial([1.0])

    term = make_I1_11(theta, R)
    polynomials = {"P1": one_poly, "Q": one_poly}

    result = evaluate_term(term, polynomials, n)
    return result.value


def evaluate_c_full(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    return_breakdown: bool = True,
    use_factorial_normalization: bool = True,
    mode: str = "main"
) -> EvaluationResult:
    """
    Evaluate full c from all K=3 pair contributions.

    c = Σ_{ℓ₁ ≤ ℓ₂} multiplier(ℓ₁, ℓ₂) × c_{ℓ₁,ℓ₂}

    where:
    - Off-diagonal pairs (ℓ₁ < ℓ₂) have symmetry factor 2
    - If use_factorial_normalization=True, each pair has factor 1/(ℓ₁! × ℓ₂!)
      from the bracket combinatorics (residue extraction normalization)

    MODES:
    - "main": PRZZ asymptotic main term constant (I₅ FORBIDDEN)
              This is what PRZZ uses to compute published κ.
              PRZZ TeX lines 1626-1628: I₅ ≪ T/L (error term)
    - "with_error_terms": Includes I₅ as diagnostic. PRINTS WARNING.
              Do NOT use this mode for golden target matching.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        return_breakdown: If True, include per-term breakdown
        use_factorial_normalization: If True, apply 1/(ℓ₁!×ℓ₂!) normalization
        mode: "main" (default) or "with_error_terms"

    Returns:
        EvaluationResult with full c and optional breakdown

    Raises:
        ValueError: If mode is not "main" or "with_error_terms"
    """
    # Validate mode
    if mode not in ("main", "with_error_terms"):
        raise ValueError(f"mode must be 'main' or 'with_error_terms', got '{mode}'")
    from src.terms_k3_d1 import make_all_terms_k3
    import math

    all_terms = make_all_terms_k3(theta, R)

    # Normalization factors: 1/(ℓ₁! × ℓ₂!)
    # This comes from the bracket combinatorics for residue extraction
    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1/1 = 1
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 1/4
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 1/2
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    }

    # Evaluate each pair
    pair_results = {}
    total = 0.0
    all_per_term = {}

    for pair_key, terms in all_terms.items():
        pair_result = evaluate_terms(terms, polynomials, n, return_breakdown)
        pair_results[pair_key] = pair_result.total

        # Off-diagonal pairs get factor of 2 for symmetry
        if pair_key in ("12", "13", "23"):
            symmetry_factor = 2.0
        else:
            symmetry_factor = 1.0

        # Apply factorial normalization if requested
        if use_factorial_normalization:
            norm = factorial_norm[pair_key]
        else:
            norm = 1.0

        total += symmetry_factor * norm * pair_result.total

        # Add per-term breakdown with pair prefix
        if return_breakdown:
            for term_name, term_val in pair_result.per_term.items():
                all_per_term[term_name] = term_val

    # I₅ arithmetic correction (ERROR TERM - diagnostic only)
    # PRZZ TeX lines 1626-1628: I₅ ≪ T/L (lower order than main term)
    #
    # WARNING: I₅ is NOT part of the asymptotic main constant that feeds κ.
    # Using it to "hit κ" means we're not computing PRZZ's object.
    #
    # Formula derived empirically: I₅ ≈ -S(0) × θ²/12 × I₂_total
    # This is included ONLY for diagnostic purposes in mode="with_error_terms".
    if mode == "with_error_terms":
        import warnings
        warnings.warn(
            "\n" + "=" * 60 + "\n"
            "WARNING: mode='with_error_terms' includes I₅.\n"
            "I₅ is O(T/L), NOT part of PRZZ's main constant.\n"
            "PRZZ TeX lines 1626-1628: 'I₅ ≪ T/L'\n"
            "Do NOT use this mode for golden target matching!\n"
            + "=" * 60,
            UserWarning,
            stacklevel=2
        )

        from src.arithmetic_constants import S_AT_ZERO

        # Compute I₂ sum (already computed as part of main evaluation)
        i2_total = 0.0
        for pair_key, terms in all_terms.items():
            # I₂ is the second term (index 1) in each pair's term list
            i2_term = terms[1]
            i2_result = evaluate_term(i2_term, polynomials, n)

            # Apply normalization
            if pair_key in ("12", "13", "23"):
                sym_factor = 2.0
            else:
                sym_factor = 1.0

            if use_factorial_normalization:
                norm = factorial_norm[pair_key]
            else:
                norm = 1.0

            i2_total += sym_factor * norm * i2_result.value

        # I₅ = -S(0) × θ²/12 × I₂_total
        i5_correction = -S_AT_ZERO * (theta ** 2 / 12.0) * i2_total
        total += i5_correction

        if return_breakdown:
            all_per_term["_I5_total"] = i5_correction
            all_per_term["_I2_sum_for_I5"] = i2_total
            all_per_term["_mode"] = "with_error_terms (I₅ INCLUDED - DIAGNOSTIC ONLY)"
    elif return_breakdown:
        all_per_term["_mode"] = "main (I₅ excluded per PRZZ)"

    # Add pair-level contributions to breakdown
    if return_breakdown:
        all_per_term["_c11_raw"] = pair_results["11"]
        all_per_term["_c22_raw"] = pair_results["22"]
        all_per_term["_c33_raw"] = pair_results["33"]
        all_per_term["_c12_raw"] = pair_results["12"]
        all_per_term["_c13_raw"] = pair_results["13"]
        all_per_term["_c23_raw"] = pair_results["23"]

        if use_factorial_normalization:
            all_per_term["_c11_norm"] = factorial_norm["11"] * pair_results["11"]
            all_per_term["_c22_norm"] = factorial_norm["22"] * pair_results["22"]
            all_per_term["_c33_norm"] = factorial_norm["33"] * pair_results["33"]
            all_per_term["_c12_norm"] = 2 * factorial_norm["12"] * pair_results["12"]
            all_per_term["_c13_norm"] = 2 * factorial_norm["13"] * pair_results["13"]
            all_per_term["_c23_norm"] = 2 * factorial_norm["23"] * pair_results["23"]

    return EvaluationResult(
        total=total,
        per_term=all_per_term,
        n=n,
        term_results=None  # Too many terms for detailed results
    )


def compute_kappa(c: float, R: float) -> float:
    """
    Compute κ from c using the Levinson bound.

    κ = 1 - log(c)/R

    Args:
        c: Main term constant
        R: R parameter

    Returns:
        κ value
    """
    import math
    return 1.0 - math.log(c) / R


def evaluate_I2_separable(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike]
) -> Tuple[float, float, float]:
    """
    Evaluate I₂ and verify separability.

    I₂ is separable in u and t:
    - u-integrand: P₁(u)²
    - t-integrand: Q(t)² · exp(2R·t) · (1/θ)

    Returns both the 2D evaluation and the product of 1D integrals
    for comparison.

    Args:
        theta: θ parameter
        R: R parameter
        n: Number of quadrature points
        polynomials: Dict with "P1", "Q"

    Returns:
        Tuple of (2D_result, 1D_product, relative_error)
    """
    from src.terms_k3_d1 import make_I2_11
    from src.quadrature import gauss_legendre_01

    # 2D evaluation
    term = make_I2_11(theta, R)
    result_2d = evaluate_term(term, polynomials, n)
    val_2d = result_2d.value

    # 1D evaluation for comparison
    nodes, weights = gauss_legendre_01(n)

    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    # u-integral: ∫ P₁(u)² du
    P1_vals = P1.eval(nodes)
    u_integral = np.sum(weights * P1_vals**2)

    # t-integral: ∫ Q(t)² · exp(2R·t) dt · (1/θ)
    Q_vals = Q.eval(nodes)
    exp_vals = np.exp(2 * R * nodes)
    t_integral = np.sum(weights * Q_vals**2 * exp_vals) / theta

    # Product should equal 2D result
    val_1d_product = u_integral * t_integral

    # Compute relative error
    if abs(val_1d_product) > 1e-15:
        rel_error = abs(val_2d - val_1d_product) / abs(val_1d_product)
    else:
        rel_error = abs(val_2d - val_1d_product)

    return val_2d, val_1d_product, rel_error


# =============================================================================
# Diagnostic report for discrepancy analysis
# =============================================================================

# PRZZ target values
PRZZ_C_TARGET = 2.13745440613217263636
PRZZ_KAPPA_TARGET = 0.417293962


@dataclass
class DiagnosticResult:
    """Complete diagnostic breakdown for debugging discrepancies."""
    # Summary
    c_computed: float
    c_target: float
    delta_c: float
    kappa_computed: float
    kappa_target: float
    delta_kappa: float

    # Per-pair raw values (before normalization)
    pair_raw: Dict[str, float]

    # Per-pair normalized values (after factorial and symmetry factors)
    pair_normalized: Dict[str, float]

    # Per-term breakdown (I₁, I₂, I₃, I₄ for each pair)
    per_term: Dict[str, float]

    # Normalization factors applied
    norm_factors: Dict[str, float]

    # Effective flags
    flags: Dict[str, Any]

    # Quadrature info
    n: int


def print_diagnostic_report(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    use_factorial_normalization: bool = True,
    mode: str = "main",
    enforce_Q0: bool = True
) -> DiagnosticResult:
    """
    Compute c and print detailed diagnostic report for discrepancy analysis.

    This prints:
    - Raw and normalized per-pair contributions
    - Per-pair per-term breakdown (I₁, I₂, I₃, I₄)
    - Normalization factor applied for each pair
    - Exact assembly formula used
    - Effective model flags
    - Totals and deltas vs target

    MODES:
    - "main": PRZZ asymptotic main term (I₅ FORBIDDEN) - DEFAULT
    - "with_error_terms": Includes I₅, prints warning

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        use_factorial_normalization: If True, apply 1/(ℓ₁!×ℓ₂!) normalization
        mode: "main" (default) or "with_error_terms"
        enforce_Q0: The Q(0) normalization mode used for polynomials

    Returns:
        DiagnosticResult with full breakdown
    """
    import math
    from src.terms_k3_d1 import make_all_terms_k3

    # Compute full c with breakdown
    all_terms = make_all_terms_k3(theta, R)

    # Factorial normalization factors
    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 1/4
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 1/2
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    }

    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0
    }

    # Evaluate each pair
    pair_raw = {}
    pair_normalized = {}
    per_term = {}
    norm_factors = {}
    total = 0.0

    for pair_key, terms in all_terms.items():
        pair_result = evaluate_terms(terms, polynomials, n, return_breakdown=True)
        pair_raw[pair_key] = pair_result.total

        # Store per-term breakdown
        for term_name, term_val in pair_result.per_term.items():
            per_term[term_name] = term_val

        # Compute normalization factor
        if use_factorial_normalization:
            norm = factorial_norm[pair_key]
        else:
            norm = 1.0

        sym = symmetry_factor[pair_key]
        norm_factors[pair_key] = sym * norm

        # Normalized contribution
        pair_normalized[pair_key] = sym * norm * pair_result.total
        total += pair_normalized[pair_key]

    # I₅ arithmetic correction (ERROR TERM - diagnostic only)
    # PRZZ TeX lines 1626-1628: I₅ ≪ T/L
    # Formula: I₅ = -S(0) × θ²/12 × I₂_total
    i5_contrib_total = 0.0
    i2_sum_for_i5 = 0.0
    if mode == "with_error_terms":
        import warnings
        warnings.warn(
            "mode='with_error_terms' includes I₅ - DIAGNOSTIC ONLY",
            UserWarning,
            stacklevel=2
        )

        from src.arithmetic_constants import S_AT_ZERO

        # Compute I₂ sum
        for pair_key, terms in all_terms.items():
            i2_term = terms[1]
            i2_result = evaluate_term(i2_term, polynomials, n)

            sym = symmetry_factor[pair_key]
            if use_factorial_normalization:
                norm = factorial_norm[pair_key]
            else:
                norm = 1.0

            i2_sum_for_i5 += sym * norm * i2_result.value

        # I₅ = -S(0) × θ²/12 × I₂_total
        i5_contrib_total = -S_AT_ZERO * (theta ** 2 / 12.0) * i2_sum_for_i5
        total += i5_contrib_total
        per_term["_I5_total"] = i5_contrib_total

    # Compute κ
    kappa = compute_kappa(total, R)

    # Build result
    result = DiagnosticResult(
        c_computed=total,
        c_target=PRZZ_C_TARGET,
        delta_c=total - PRZZ_C_TARGET,
        kappa_computed=kappa,
        kappa_target=PRZZ_KAPPA_TARGET,
        delta_kappa=kappa - PRZZ_KAPPA_TARGET,
        pair_raw=pair_raw,
        pair_normalized=pair_normalized,
        per_term=per_term,
        norm_factors=norm_factors,
        flags={
            "use_factorial_normalization": use_factorial_normalization,
            "mode": mode,
            "enforce_Q0": enforce_Q0,
            "theta": theta,
            "R": R,
        },
        n=n
    )

    # Print report
    print("\n" + "=" * 70)
    print("PRZZ DIAGNOSTIC REPORT")
    print("=" * 70)

    print("\n--- Effective Model Flags ---")
    print(f"  use_factorial_normalization: {use_factorial_normalization}")
    print(f"  mode:                        {mode}")
    print(f"  enforce_Q0:                  {enforce_Q0}")
    print(f"  theta:                       {theta:.10f}")
    print(f"  R:                           {R}")
    print(f"  n (quadrature):              {n}")

    print("\n--- Raw Per-Pair Values (before normalization) ---")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        print(f"  c_{pair}_raw:  {pair_raw[pair]:+18.12f}")

    print("\n--- Normalization Factors Applied ---")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        fact_str = f"1/{math.factorial(int(pair[0])) * math.factorial(int(pair[1]))}"
        sym_str = "×2" if pair in ("12", "13", "23") else "×1"
        print(f"  {pair}: {fact_str:>6s} {sym_str} = {norm_factors[pair]:.6f}")

    print("\n--- Normalized Per-Pair Values ---")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        print(f"  c_{pair}_norm: {pair_normalized[pair]:+18.12f}")

    print("\n--- Assembly Formula ---")
    formula_parts = []
    for pair in ["11", "22", "33", "12", "13", "23"]:
        if pair in ("12", "13", "23"):
            formula_parts.append(f"2×c_{pair}/({math.factorial(int(pair[0]))}!×{math.factorial(int(pair[1]))}!)")
        else:
            formula_parts.append(f"c_{pair}/({math.factorial(int(pair[0]))}!×{math.factorial(int(pair[1]))}!)")
    print(f"  c = {' + '.join(formula_parts)}")

    print("\n--- Per-Term Breakdown (by integral type) ---")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        print(f"  Pair ({pair[0]},{pair[1]}):")
        for i in range(1, 5):  # I₁-I₄
            term_name = f"I{i}_{pair}"
            if term_name in per_term:
                print(f"    I{i}: {per_term[term_name]:+18.12f}")
            else:
                print(f"    I{i}: (not computed)")
        # I₅ (arithmetic correction - only in with_error_terms mode)
        term_name = f"I5_{pair}"
        if term_name in per_term:
            print(f"    I₅: {per_term[term_name]:+18.12f} (S correction)")
        elif mode == "with_error_terms":
            print(f"    I₅: (not computed)")

    if mode == "with_error_terms":
        from src.arithmetic_constants import S_AT_ZERO as S0
        print(f"\n--- I₅ Arithmetic Correction (DIAGNOSTIC ONLY) ---")
        print(f"  WARNING: I₅ ≪ T/L per PRZZ lines 1626-1628")
        print(f"  Formula: I₅ = -S(0) × θ²/12 × I₂_total")
        print(f"  S(0) = {S0:.10f}")
        print(f"  θ² = {theta**2:.10f}")
        print(f"  θ²/12 = {theta**2/12:.10f}")
        print(f"  I₂_total (normalized) = {i2_sum_for_i5:+18.12f}")
        print(f"  I₅ correction = {i5_contrib_total:+18.12f}")
    else:
        print(f"\n--- I₅ Status ---")
        print(f"  mode='main': I₅ EXCLUDED (correct per PRZZ)")

    print("\n--- Summary ---")
    print(f"  c_computed:    {total:20.15f}")
    print(f"  c_target:      {PRZZ_C_TARGET:20.15f}")
    print(f"  Δc:            {total - PRZZ_C_TARGET:+20.15f}")
    print(f"  Δc relative:   {(total - PRZZ_C_TARGET) / PRZZ_C_TARGET * 100:+.6f}%")
    print()
    print(f"  κ_computed:    {kappa:20.15f}")
    print(f"  κ_target:      {PRZZ_KAPPA_TARGET:20.15f}")
    print(f"  Δκ:            {kappa - PRZZ_KAPPA_TARGET:+20.15f}")

    print("\n" + "=" * 70)

    return result
