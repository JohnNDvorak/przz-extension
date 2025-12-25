"""
src/combined_identity_operator.py
Combined Identity Operator Module — Leibniz Expansion of Q(D_α)Q(D_β)B.

This module applies Q(D_α)Q(D_β) to the PRZZ pre-identity bracket:
    B(α,β,x,y) = [N^{αx+βy} - T^{-(α+β)}N^{-βx-αy}] / (α+β)

The key insight: the bracket is a PRODUCT of two factors:
1. exp_bracket = N^{αx+βy} - T^{-(α+β)}N^{-βx-αy}  (the exponential difference)
2. inv_factor = 1/(α+β)  (the inverse sum)

When the operator acts on this product, we use the Leibniz rule:
    D^n(f × g) = Σ_{k=0}^{n} C(n,k) × D^k(f) × D^{n-k}(g)

For Q(D_α)Q(D_β) = [Σ_i q_i D_α^i][Σ_j q_j D_β^j], we expand:
    Q(D_α)Q(D_β)(exp_bracket × inv_factor)
    = Σ_{i,j} q_i × q_j × D_α^i D_β^j (exp_bracket × inv_factor)

And each D_α^i D_β^j expands via double Leibniz:
    D_α^i D_β^j (f × g)
    = Σ_{k=0}^{i} Σ_{l=0}^{j} C(i,k) C(j,l) × D_α^k D_β^l(f) × D_α^{i-k} D_β^{j-l}(g)

This produces terms with varying numbers of derivatives on each factor.

CRITICAL: This is where derivatives hitting 1/(α+β) produce additional terms
that may cancel the L-divergence seen in the naive approach.

See: docs/DECISIONS.md for why this approach is necessary.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from math import factorial

from src.analytic_derivatives import (
    deriv_inverse_sum_at_point,
    deriv_exp_linear_mixed_PRZZ,
    deriv_mirror_exp_factor_PRZZ,
    binomial,
    leibniz_coefficients,
)


# =============================================================================
# Dataclasses for term contributions
# =============================================================================

@dataclass
class BracketTermContribution:
    """
    One term in the Leibniz expansion of Q(D_α)Q(D_β)B.

    This represents a single contribution from:
    - Q coefficient q_i times Q coefficient q_j
    - Leibniz split: k derivatives on exp in α, i-k on inv in α
    - Leibniz split: l derivatives on exp in β, j-l on inv in β
    - Sign: +1 for plus branch, -1 for minus branch
    """
    # Q polynomial indices
    i: int  # D_α order from Q(D_α)
    j: int  # D_β order from Q(D_β)
    q_i: float  # Q coefficient for D_α^i
    q_j: float  # Q coefficient for D_β^j

    # Leibniz distribution
    k: int  # derivatives on exp in α
    l: int  # derivatives on exp in β
    # Remaining: (i-k) on inv in α, (j-l) on inv in β

    # Binomial coefficients from Leibniz
    binom_ik: int  # C(i, k)
    binom_jl: int  # C(j, l)

    # Plus branch (+1) or minus branch (-1)
    sign: float

    @property
    def exp_deriv_alpha(self) -> int:
        """Derivatives hitting exponential in α."""
        return self.k

    @property
    def exp_deriv_beta(self) -> int:
        """Derivatives hitting exponential in β."""
        return self.l

    @property
    def inv_deriv_alpha(self) -> int:
        """Derivatives hitting 1/(α+β) in α."""
        return self.i - self.k

    @property
    def inv_deriv_beta(self) -> int:
        """Derivatives hitting 1/(α+β) in β."""
        return self.j - self.l

    @property
    def coefficient(self) -> float:
        """Total coefficient: q_i × q_j × C(i,k) × C(j,l) × sign."""
        return self.q_i * self.q_j * self.binom_ik * self.binom_jl * self.sign

    def __str__(self) -> str:
        branch = "+" if self.sign > 0 else "-"
        return (
            f"Term[{branch}]: q_{self.i}×q_{self.j} × C({self.i},{self.k})×C({self.j},{self.l}) "
            f"× D_α^{self.k}D_β^{self.l}[exp] × D_α^{self.i-self.k}D_β^{self.j-self.l}[1/(α+β)]"
        )


@dataclass
class OperatorExpansionResult:
    """Result of expanding Q(D_α)Q(D_β)B via Leibniz rule."""

    terms: List[BracketTermContribution]
    """All individual terms from the expansion."""

    Q_coeffs: List[float]
    """Q polynomial coefficients used."""

    max_order: int
    """Maximum derivative order considered."""

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    def terms_by_branch(self) -> Tuple[List[BracketTermContribution], List[BracketTermContribution]]:
        """Separate into plus and minus branch terms."""
        plus = [t for t in self.terms if t.sign > 0]
        minus = [t for t in self.terms if t.sign < 0]
        return plus, minus

    def terms_by_inv_order(self) -> Dict[int, List[BracketTermContribution]]:
        """Group terms by total derivatives on 1/(α+β)."""
        result = {}
        for t in self.terms:
            order = t.inv_deriv_alpha + t.inv_deriv_beta
            if order not in result:
                result[order] = []
            result[order].append(t)
        return result


# =============================================================================
# Leibniz expansion functions
# =============================================================================

def expand_QQ_on_bracket(
    Q_coeffs: List[float],
    max_order: Optional[int] = None
) -> OperatorExpansionResult:
    """
    Expand Q(D_α)Q(D_β)B using Leibniz rule for both branches.

    The bracket is:
        B = [exp_plus - exp_minus] / (α+β)
          = exp_plus × inv - exp_minus × inv

    For each branch, we apply double Leibniz:
        D_α^i D_β^j (exp × inv)
        = Σ_k Σ_l C(i,k) C(j,l) × D_α^k D_β^l[exp] × D_α^{i-k} D_β^{j-l}[inv]

    Args:
        Q_coeffs: List of Q polynomial coefficients [q_0, q_1, q_2, ...]
        max_order: Maximum derivative order to consider (default: len(Q_coeffs)-1)

    Returns:
        OperatorExpansionResult containing all Leibniz terms
    """
    if max_order is None:
        max_order = len(Q_coeffs) - 1

    terms = []

    # Q(D_α) = Σ_i q_i D_α^i
    # Q(D_β) = Σ_j q_j D_β^j
    for i, q_i in enumerate(Q_coeffs):
        if i > max_order or q_i == 0:
            continue

        for j, q_j in enumerate(Q_coeffs):
            if j > max_order or q_j == 0:
                continue

            # Apply D_α^i D_β^j to (exp × inv) using double Leibniz
            # Σ_k Σ_l C(i,k) C(j,l) × D_α^k D_β^l[exp] × D_α^{i-k} D_β^{j-l}[inv]
            for k in range(i + 1):
                binom_ik = binomial(i, k)

                for l in range(j + 1):
                    binom_jl = binomial(j, l)

                    # Plus branch: exp_plus = N^{αx+βy} = exp(θL(αx+βy))
                    terms.append(BracketTermContribution(
                        i=i, j=j, q_i=q_i, q_j=q_j,
                        k=k, l=l,
                        binom_ik=binom_ik, binom_jl=binom_jl,
                        sign=+1.0
                    ))

                    # Minus branch: exp_minus = T^{-(α+β)} × N^{-βx-αy}
                    terms.append(BracketTermContribution(
                        i=i, j=j, q_i=q_i, q_j=q_j,
                        k=k, l=l,
                        binom_ik=binom_ik, binom_jl=binom_jl,
                        sign=-1.0
                    ))

    return OperatorExpansionResult(
        terms=terms,
        Q_coeffs=Q_coeffs,
        max_order=max_order
    )


def evaluate_term_at_point(
    term: BracketTermContribution,
    alpha: float,
    beta: float,
    x: float,
    y: float,
    theta: float,
    L: float
) -> float:
    """
    Evaluate a single Leibniz term at a specific point.

    This computes:
        coeff × D_α^k D_β^l[exp] × D_α^{i-k} D_β^{j-l}[1/(α+β)]

    where:
    - Plus branch: exp = exp(θL(αx+βy))
    - Minus branch: exp = exp(-L(α+β) - θL(βx+αy))

    Args:
        term: The BracketTermContribution to evaluate
        alpha, beta: Point of evaluation
        x, y: Series variables
        theta: θ parameter
        L: Large parameter

    Returns:
        Numerical value of this term at the point
    """
    # Derivative on exponential
    k = term.exp_deriv_alpha
    l = term.exp_deriv_beta

    if term.sign > 0:
        # Plus branch: exp(θL(αx+βy))
        exp_deriv_value = deriv_exp_linear_mixed_PRZZ(k, l, alpha, beta, x, y, theta, L)
    else:
        # Minus branch: T^{-(α+β)} × N^{-βx-αy}
        exp_deriv_value = deriv_mirror_exp_factor_PRZZ(k, l, alpha, beta, x, y, theta, L)

    # Derivative on 1/(α+β)
    inv_deriv_alpha = term.inv_deriv_alpha
    inv_deriv_beta = term.inv_deriv_beta
    inv_deriv_value = deriv_inverse_sum_at_point(inv_deriv_alpha, inv_deriv_beta, alpha, beta)

    # Total value: coefficient × exp_deriv × inv_deriv
    return term.coefficient * exp_deriv_value * inv_deriv_value


def evaluate_QQB_at_point(
    expansion: OperatorExpansionResult,
    alpha: float,
    beta: float,
    x: float,
    y: float,
    theta: float,
    L: float
) -> float:
    """
    Evaluate Q(D_α)Q(D_β)B at a specific point by summing all Leibniz terms.

    This is the CORRECT operator application that captures how derivatives
    falling on 1/(α+β) modify the result.

    Args:
        expansion: Result from expand_QQ_on_bracket
        alpha, beta: Point of evaluation
        x, y: Series variables
        theta: θ parameter
        L: Large parameter

    Returns:
        Total value of Q(D_α)Q(D_β)B at the point
    """
    total = 0.0
    for term in expansion.terms:
        total += evaluate_term_at_point(term, alpha, beta, x, y, theta, L)
    return total


# =============================================================================
# Convenience functions
# =============================================================================

def compute_QQB_full(
    Q_coeffs: List[float],
    alpha: float,
    beta: float,
    x: float,
    y: float,
    theta: float,
    L: float,
    max_order: Optional[int] = None
) -> float:
    """
    One-shot computation of Q(D_α)Q(D_β)B at a specific point.

    This expands and evaluates in one call.

    Args:
        Q_coeffs: Q polynomial coefficients [q_0, q_1, ...]
        alpha, beta: Point of evaluation
        x, y: Series variables
        theta: θ parameter
        L: Large parameter
        max_order: Maximum derivative order (default: len(Q_coeffs)-1)

    Returns:
        Value of Q(D_α)Q(D_β)B
    """
    expansion = expand_QQ_on_bracket(Q_coeffs, max_order)
    return evaluate_QQB_at_point(expansion, alpha, beta, x, y, theta, L)


def compute_QQB_by_branch(
    Q_coeffs: List[float],
    alpha: float,
    beta: float,
    x: float,
    y: float,
    theta: float,
    L: float,
    max_order: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute Q(D_α)Q(D_β)B, returning plus and minus branch separately.

    Returns:
        (plus_branch_value, minus_branch_value, total)
    """
    expansion = expand_QQ_on_bracket(Q_coeffs, max_order)
    plus_terms, minus_terms = expansion.terms_by_branch()

    plus_value = sum(
        evaluate_term_at_point(t, alpha, beta, x, y, theta, L)
        for t in plus_terms
    )
    minus_value = sum(
        evaluate_term_at_point(t, alpha, beta, x, y, theta, L)
        for t in minus_terms
    )

    return plus_value, minus_value, plus_value + minus_value


def compute_QQB_by_inv_order(
    Q_coeffs: List[float],
    alpha: float,
    beta: float,
    x: float,
    y: float,
    theta: float,
    L: float,
    max_order: Optional[int] = None
) -> Dict[int, float]:
    """
    Compute Q(D_α)Q(D_β)B, grouped by derivatives on 1/(α+β).

    This helps analyze how terms with different powers of 1/(α+β)
    contribute to the total. The L-dependence differs by power.

    Returns:
        Dict mapping inv_order -> contribution from terms with that many
        derivatives on 1/(α+β).
    """
    expansion = expand_QQ_on_bracket(Q_coeffs, max_order)
    terms_by_order = expansion.terms_by_inv_order()

    result = {}
    for order, terms in terms_by_order.items():
        result[order] = sum(
            evaluate_term_at_point(t, alpha, beta, x, y, theta, L)
            for t in terms
        )

    return result


# =============================================================================
# PRZZ-specific evaluation
# =============================================================================

def compute_QQB_at_przz_point(
    Q_coeffs: List[float],
    R: float,
    L: float,
    x: float,
    y: float,
    theta: float = 4.0 / 7.0
) -> Dict[str, float]:
    """
    Compute Q(D_α)Q(D_β)B at the PRZZ evaluation point α=β=-R/L.

    This is the key function for deriving m1 from first principles.

    Args:
        Q_coeffs: Q polynomial coefficients
        R: R parameter (typically 1.3036 for κ, 1.1167 for κ*)
        L: Large parameter
        x, y: Series variables (will be integrated over [0,1]²)
        theta: θ parameter (default 4/7)

    Returns:
        Dict with:
        - "total": Total value
        - "plus": Plus branch contribution
        - "minus": Minus branch contribution
        - "by_inv_order": Dict of contributions by 1/(α+β) derivative order
        - "alpha", "beta", "sum_ab": Evaluation point info
    """
    alpha = beta = -R / L

    plus, minus, total = compute_QQB_by_branch(
        Q_coeffs, alpha, beta, x, y, theta, L
    )

    by_inv_order = compute_QQB_by_inv_order(
        Q_coeffs, alpha, beta, x, y, theta, L
    )

    return {
        "total": total,
        "plus": plus,
        "minus": minus,
        "by_inv_order": by_inv_order,
        "alpha": alpha,
        "beta": beta,
        "sum_ab": alpha + beta,
        "R": R,
        "L": L,
    }


# =============================================================================
# Analysis utilities
# =============================================================================

def analyze_L_scaling(
    Q_coeffs: List[float],
    R: float,
    x: float,
    y: float,
    L_values: List[float],
    theta: float = 4.0 / 7.0
) -> List[Dict]:
    """
    Analyze how Q(D_α)Q(D_β)B scales with L at the PRZZ point.

    This is crucial for understanding if the operator application
    causes convergence (L-independence) or divergence (L-dependence).

    Args:
        Q_coeffs: Q polynomial coefficients
        R: R parameter
        x, y: Fixed series variable values
        L_values: List of L values to test
        theta: θ parameter

    Returns:
        List of result dicts, one per L value
    """
    results = []
    for L in L_values:
        result = compute_QQB_at_przz_point(Q_coeffs, R, L, x, y, theta)
        results.append(result)
    return results


def print_expansion_summary(expansion: OperatorExpansionResult) -> None:
    """Print a summary of the Leibniz expansion."""
    print(f"Q(D_α)Q(D_β)B Expansion Summary")
    print(f"=" * 60)
    print(f"Q coefficients: {expansion.Q_coeffs}")
    print(f"Max order: {expansion.max_order}")
    print(f"Total terms: {expansion.n_terms}")

    plus_terms, minus_terms = expansion.terms_by_branch()
    print(f"Plus branch terms: {len(plus_terms)}")
    print(f"Minus branch terms: {len(minus_terms)}")

    by_inv_order = expansion.terms_by_inv_order()
    print(f"\nTerms by 1/(α+β) derivative order:")
    for order in sorted(by_inv_order.keys()):
        print(f"  Order {order}: {len(by_inv_order[order])} terms")
