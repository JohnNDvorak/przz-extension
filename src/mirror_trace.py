"""
src/mirror_trace.py
Trace PRZZ Mirror Combination to Find (1 + θ/6) Factor

PRZZ TeX Lines 1502-1511 (Mirror Combination Identity):
    (N^{αx+βy} - T^{-α-β}N^{-βx-αy})/(α+β)
    = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt

At α=β=-R/L, with N=T^θ:
    log(N^{x+y}T) = log(T^{θ(x+y)+1}) = [θ(x+y)+1] × log(T)

Key Question:
When derivatives are extracted, does [θ(x+y)+1] contribute to the constant?

Structure:
    F(x,y) = [θ(x+y)+1] × log(T) × G(x,y)

Derivatives:
    ∂F/∂x = θ × log(T) × G + [θ(x+y)+1] × log(T) × ∂G/∂x

At x=y=0:
    ∂F/∂x|₀ = θ × log(T) × G(0,0) + log(T) × ∂G/∂x|₀

The log(T) factors out, but the θ×G(0,0) term is an EXTRA contribution
from the [θ(x+y)+1] factor that we may be missing.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Callable
import math

from src.polynomials import load_przz_polynomials
from src.quadrature import tensor_grid_2d


def analyze_derivative_structure(verbose: bool = True) -> Dict:
    """
    Analyze how [1 + θ(x+y)] affects derivative extraction.

    For pair (ℓ₁, ℓ₂), we take ℓ₁ + ℓ₂ derivatives.
    Each derivative can either:
    1. Act on [1 + θ(x+y)] → contributes θ
    2. Act on G(x,y) → contributes derivative of G

    The total contribution is a sum over all ways to distribute derivatives.
    """
    theta = 4/7

    # For (1,1): 2 derivatives (∂/∂x ∂/∂y)
    # F = [1 + θ(x+y)] × G
    # ∂²F/∂x∂y = θ × ∂G/∂y + θ × ∂G/∂x + [1 + θ(x+y)] × ∂²G/∂x∂y
    # At x=y=0: = θ(∂G/∂x + ∂G/∂y)|₀ + ∂²G/∂x∂y|₀

    # For (2,2): 4 derivatives (∂⁴/∂x₁∂x₂∂y₁∂y₂)
    # This gets more complex...

    # General pattern: For n derivatives, we get contributions where
    # 0, 1, or at most 1 derivative acts on [1 + θ(x+y)] (since it's linear)

    # If 0 derivatives act on prefactor: coefficient is 1
    # If 1 derivative acts on prefactor: coefficient is θ, and we have n-1 derivatives of G

    # So the structure is:
    # ∂ⁿF/∂x...∂y...|₀ = ∂ⁿG/∂x...∂y...|₀ + θ × Σ (lower derivatives of G)

    results = {}

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    if verbose:
        print("\n" + "=" * 70)
        print("DERIVATIVE STRUCTURE ANALYSIS")
        print("=" * 70)
        print("\nFor F = [1 + θ(x+y)] × G, the derivative ∂ⁿF|₀ contains:")
        print("  1. ∂ⁿG|₀ (main term)")
        print("  2. θ × (sum of lower derivatives of G)")
        print("\nThe 'extra' θ-terms are what we might be missing.")
        print()

    for ell1, ell2 in pairs:
        n_derivs = ell1 + ell2

        # Number of ways to choose which 1 derivative hits prefactor
        # (from the n total derivatives)
        n_extra_terms = n_derivs  # Each of n derivatives can hit prefactor

        results[(ell1, ell2)] = {
            "n_derivs": n_derivs,
            "n_extra_terms": n_extra_terms,
        }

        if verbose:
            print(f"  ({ell1},{ell2}): {n_derivs} derivs → {n_derivs} extra θ-terms")

    if verbose:
        print()
        print("The contribution of extra θ-terms depends on the VALUES of lower")
        print("derivatives of G. To get factor (1+θ/6), the weighted average")
        print("of these contributions must equal θ/6 × (main term).")
        print("=" * 70)

    return results


def compute_with_and_without_log_prefactor(
    theta: float = 4/7,
    R: float = 1.3036,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict:
    """
    Compare I₁ computation with and without the [1 + θ(x+y)] factor.

    This tests whether including this factor in the integrand (and then
    differentiating it) gives the missing (1 + θ/6) correction.
    """
    from src.terms_k3_d1 import make_I1_11
    from src.evaluate import evaluate_term

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    # Current I1_11 (without log prefactor)
    term = make_I1_11(theta, R)
    result_current = evaluate_term(term, polys, n_quad)
    i1_current = result_current.value

    # The question: what would I1 be if we included [1 + θ(x+y)] as an
    # additional factor in the integrand?

    # Our current algebraic prefactor is (θS+1)/θ where S = x+y
    # This equals [1 + θ(x+y)]/θ = 1/θ + (x+y)
    # So we ALREADY have [1 + θ(x+y)] in the prefactor!

    # Wait - this is important. Let me check the algebraic prefactor structure.

    if verbose:
        print("\n" + "=" * 70)
        print("ALGEBRAIC PREFACTOR CHECK")
        print("=" * 70)
        print(f"\nFor (1,1), the algebraic prefactor is: (θS+1)/θ where S = x+y")
        print(f"  = [θ(x+y) + 1]/θ")
        print(f"  = (x+y) + 1/θ")
        print(f"\nAt θ = {theta:.10f}:")
        print(f"  1/θ = {1/theta:.10f}")
        print(f"\nThis ALREADY contains [1 + θ(x+y)] (divided by θ)!")
        print(f"\nSo our current formulas DO include the log factor structure.")
        print(f"The issue must be elsewhere.")
        print("=" * 70)

    return {
        "i1_current": i1_current,
        "prefactor_structure": "(θS+1)/θ = [1 + θ(x+y)]/θ",
        "already_included": True,
    }


def check_log_T_cancellation(verbose: bool = True) -> Dict:
    """
    Trace how log(T) should cancel in the asymptotic expansion.

    PRZZ's result is: ∫₁ᵀ |Vψ|² dt = T × c + O(T/L)

    The "c" is T-independent. But intermediate formulas involve log(T).
    How does this cancel?
    """
    if verbose:
        print("\n" + "=" * 70)
        print("LOG(T) CANCELLATION ANALYSIS")
        print("=" * 70)
        print("""
PRZZ's asymptotic expansion structure:

∫₁ᵀ |Vψ(σ₀+it)|² dt = T × c + O(T/L)

where:
- c is the "main term constant" (T-independent)
- O(T/L) includes I₅ and other lower-order terms
- L = log(T)

The constant c comes from the leading-order term in the expansion.

Key insight from PRZZ TeX 1502-1511:
The mirror combination introduces log(N^{x+y}T) = [θ(x+y)+1] × log(T).

But in the FINAL asymptotic extraction:
- The log(T) multiplies the entire term
- This log(T) is then "divided out" when extracting c
- Because c is defined as: c = lim_{T→∞} (1/T) × ∫₁ᵀ |Vψ|² dt

So the question is: does the [θ(x+y)+1] factor inside the log
contribute to c, or does it only affect lower-order terms?

Hypothesis:
If PRZZ's formula is:
    I₁ = log(T) × [1 + θ(x+y)] × (integral over u,t)

And we extract c by:
    c = coefficient of T in the asymptotic expansion

Then the [1 + θ(x+y)] contributes to c through its derivatives.

But wait - our formulas already include (θS+1)/θ as the algebraic prefactor!
So we SHOULD be computing the same thing.

The mystery deepens: why is there still a ~(1+θ/6) gap?
""")
        print("=" * 70)

    return {
        "log_T_structure": "log(T) × [1 + θ(x+y)] × G",
        "c_extraction": "c = coeff of T in asymptotic",
        "prefactor_included": "Yes, as (θS+1)/θ",
    }


def investigate_remaining_possibilities(verbose: bool = True) -> Dict:
    """
    List remaining hypotheses for the (1+θ/6) gap.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("REMAINING HYPOTHESES FOR (1 + θ/6) GAP")
        print("=" * 70)
        print("""
Our algebraic prefactor (θS+1)/θ = [1+θ(x+y)]/θ already includes the
structure from PRZZ's log(N^{x+y}T) factor. So the gap must come from
somewhere else.

Hypothesis 1: DIFFERENT NORMALIZATION OF c
    PRZZ might define c differently than we think.
    Our c = sum of normalized pair contributions
    PRZZ's c might have an additional overall factor

Hypothesis 2: MOLLIFIER NORMALIZATION
    PRZZ TeX line 542-548 defines the mollifier with specific conventions.
    We might be missing a normalization factor in the mollifier definition.

Hypothesis 3: Φ̂(0) NORMALIZATION
    We assume Φ̂(0) = 1 (test function normalization).
    If PRZZ uses a different value, this would scale everything.

Hypothesis 4: INTEGRATION DOMAIN
    We integrate over [0,1]² for (u,t).
    PRZZ might use a different parametrization.

Hypothesis 5: THE TARGET IS WRONG
    PRZZ's published κ = 0.417293962 might include something we're
    calling "error term", or be computed with a different prefactor.

Hypothesis 6: VARIABLE SCALING (x → x·log N)
    PRZZ TeX line 2309 mentions this scaling.
    If our variables are at a different stage, we'd have systematic factors.

Most Likely: Hypothesis 1 or 5
    Given that (1+θ/6) ≈ 1.095 is such a clean factor, it's likely
    either a normalization we're missing, or PRZZ's published value
    uses a different definition than their "paper-correct" formulas.
""")
        print("=" * 70)

    return {
        "hypotheses": [
            "Different normalization of c",
            "Mollifier normalization",
            "Φ̂(0) normalization",
            "Integration domain",
            "Target value definition",
            "Variable scaling",
        ],
        "most_likely": ["normalization", "target_definition"],
    }


def numerical_test_prefactor_with_log(
    theta: float = 4/7,
    R: float = 1.3036,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict:
    """
    Numerically test if adding an extra [1+θ(x+y)] factor gives (1+θ/6).

    Our current prefactor: (θS+1)/θ = [1+θS]/θ
    Test new prefactor: [1+θS]/θ × [1+θS]^{1/6} or similar
    """
    from src.evaluate import evaluate_c_full

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    # Current c
    result = evaluate_c_full(theta, R, n=n_quad, polynomials=polys, mode="main")
    c_current = result.total

    # Target
    c_target = 2.13745440613217263636

    # What factor is needed?
    factor_needed = c_target / c_current

    # (1 + θ/6)
    factor_theta_6 = 1 + theta/6

    # Check if applying factor works
    c_corrected = c_current * factor_theta_6

    results = {
        "c_current": c_current,
        "c_target": c_target,
        "factor_needed": factor_needed,
        "factor_theta_6": factor_theta_6,
        "c_corrected": c_corrected,
        "correction_error": abs(c_corrected - c_target) / c_target,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("NUMERICAL CHECK: APPLYING (1 + θ/6) FACTOR")
        print("=" * 70)
        print(f"\n  c_current:     {c_current:.10f}")
        print(f"  c_target:      {c_target:.10f}")
        print(f"  factor_needed: {factor_needed:.10f}")
        print(f"  (1 + θ/6):     {factor_theta_6:.10f}")
        print(f"\n  c × (1+θ/6):   {c_corrected:.10f}")
        print(f"  Error:         {results['correction_error']*100:.4f}%")

        if results['correction_error'] < 0.001:
            print(f"\n  ✓ (1+θ/6) factor gives excellent match!")
            print(f"\n  IMPLICATION: We need to find where this factor belongs")
            print(f"  in the mathematical derivation.")

        print("=" * 70)

    return results


if __name__ == "__main__":
    analyze_derivative_structure(verbose=True)
    compute_with_and_without_log_prefactor(verbose=True)
    check_log_T_cancellation(verbose=True)
    investigate_remaining_possibilities(verbose=True)
    numerical_test_prefactor_with_log(verbose=True)
