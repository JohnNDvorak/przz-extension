# GPT Feedback Integration - December 18, 2025

## Three Critical Corrections from GPT

### 1. t-integral domain is [0,1], NOT [0,∞)

**Status: ALREADY CORRECT ✓**

Our `gauss_legendre_01()` function maps to [0,1]. The PRZZ difference quotient identity (TeX ~1510) gives:
```
[1 - (N^{x+y}T)^{-(α+β)}]/(α+β) = log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

### 2. (1-u) power is a+b (monomial-level), NOT ℓ+ℓ̄-2p (block-level)

**Status: ALREADY CORRECT in psi_separated_c.py ✓**

Our Ψ expansion correctly computes (1-u)^{a+b} per monomial:
- (1,1) AB: weight = (1-u)²
- (1,1) D: weight = (1-u)⁰ = 1
- (1,1) AC_α: weight = (1-u)¹
- (1,1) BC_β: weight = (1-u)¹

### 3. Mirror factor: don't collapse T^{-α-β} until after Q acts

**Status: GenEval uses "post-Q identity" which is correct ✓**

The GenEval computes I₁-I₄ using the POST-Q formulas where:
- The t-integral with exp(2Rt) appears naturally
- Q derivatives act on the arguments, not a symbolic T^{-α-β}

## Ψ Expansion Validation

Our `psi_separated_c.py` matches GPT's exact expansions:

| Pair | Expected | Our Count | Match |
|------|----------|-----------|-------|
| (1,1) | 4 terms | 4 | ✓ |
| (2,2) | 12 terms | 12 | ✓ |
| (3,3) | 27 terms | 27 | ✓ |

Coefficients verified exactly against GPT's formulas:
- Ψ₁,₁ = AB - AC_α - BC_β + D ✓
- Ψ₂,₂ = A²B² - 2A²BC_α + ... (12 terms) ✓

## Key Finding: GenEval Sign Convention

The GenEval I-term formulas **already include Ψ signs**:
```python
I₃ = -(I3_base + (1/θ) × I3_deriv)  # Built-in negative
I₄ = -(I4_base + (1/θ) × I4_deriv)  # Built-in negative
```

For (1,1), the total is simply:
```
Total = I₁ + I₂ + I₃ + I₄ = 0.426 + 0.385 - 0.226 - 0.226 = 0.359 ✓
```

**DO NOT multiply by Ψ coefficients again** - they're already encoded.

## Why MonoMialFdEvaluator Fails

The `MonoMialFdEvaluator` in `fd_evaluation.py` has two problems:

1. **Uses factored formula** `u_integral × t_integral / θ` which is wrong for derivative terms
2. **Multiplies by Ψ coefficients** when signs are already in the integral formulas

This gives 0.631 instead of 0.359 for (1,1).

## Why GenEval is Approximate for (2,2)+

For (1,1), there's a 1:1 mapping between monomials and I-terms:
- AB → I₁
- D → I₂
- AC_α → I₃
- BC_β → I₄

For (2,2), the 12 monomials include:
- A²B² (I₁-like)
- ABD (mixed AB and D)
- D² (squared D)
- A²C_α² (multiple C factors)
- etc.

These don't map cleanly to the I₁-I₄ structure. The GenEval applies a single weight per I-term type, which is an approximation.

## Recommended Path Forward

### Option 1: Accept GenEval Approximation (Pragmatic)

- GenEval is **exact** for (1,1) (perfect oracle match)
- GenEval is **approximate** for (2,2)+ (contributes to 7-12% c error)
- For optimization purposes, this may be acceptable

### Option 2: Derive Per-Monomial Integrands (Comprehensive)

For each of the 12 monomial types in (2,2), derive:
1. The correct integrand structure
2. How the derivatives couple u and t
3. The appropriate integral formula

This is substantial work but would give exact results.

### Option 3: Use GenEval with Adjusted Weights (Compromise)

Keep the I₁-I₄ structure but adjust weights to better match the monomial distribution in (2,2)+.

## What GPT Confirmed

1. **Ψ expansion is correct** - 4/12/27 term counts match
2. **t ∈ [0,1]** - our quadrature is correct
3. **(1-u)^{a+b}** - our per-monomial weights are correct
4. **Post-Q identity** - GenEval uses the right approach

## What Remains Unsolved

1. **7-12% gap in total c** - due to (2,2)+ approximation error
2. **Two-benchmark ratio mismatch** - polynomial degree differences
3. **Per-monomial integral formulas for (2,2)+** - not yet derived

## Test Commands

```bash
# Verify (1,1) is exact
PYTHONPATH=. python3 -c "
from src.polynomials import load_przz_polynomials
from src.przz_generalized_iterm_evaluator import GeneralizedItermEvaluator
P1, _, _, Q = load_przz_polynomials(enforce_Q0=True)
eval = GeneralizedItermEvaluator(P1, P1, Q, 4/7, 1.3036, 1, 1, 60)
print(f'(1,1) Total = {eval.eval_all().total:.6f} (target: 0.359159)')
"

# Check Ψ expansion counts
PYTHONPATH=. python3 -c "
from src.psi_separated_c import expand_pair_to_monomials_separated
for ell in [1, 2, 3]:
    n = len(expand_pair_to_monomials_separated(ell, ell))
    expected = {1: 4, 2: 12, 3: 27}[ell]
    print(f'({ell},{ell}): {n} monomials (expected: {expected}) {\"✓\" if n == expected else \"✗\"}')"
```
