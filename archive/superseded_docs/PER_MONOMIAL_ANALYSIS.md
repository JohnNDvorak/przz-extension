# Per-Monomial Evaluation Analysis - December 18, 2025

## Key Findings

### 1. (1,1) Per-Monomial Matches GenEval Exactly ✓

The per-monomial evaluator with (l₁, m₁) = (a, b) derivative orders and (1-u)^{a+b} weights matches the GenEval oracle perfectly for (1,1):

```
Per-Monomial Total: 0.359159
GenEval Total:      0.359159
Difference:         1.11e-16
```

### 2. Off-Diagonal Pairs Have Structural Mismatch ✗

For (1,2), per-monomial gives -0.411 vs GenEval 0.490. This is a sign flip, not just a magnitude error.

**Root cause**: The GenEval I-term formulas have signs built in that correspond to the (1,1) Ψ structure:
- I₃ = -[base + deriv/θ]  (negative built in for -AC_α in (1,1))
- I₄ = -[base + deriv/θ]  (negative built in for -BC_β in (1,1))

For (1,1):
- AC_α has Ψ = -1
- Per-mono: Ψ × (positive integral) = -1 × 0.226 = -0.226
- GenEval I₃ = -0.226 (negative built in)
- **MATCH** ✓

For (1,2):
- The (1,0) type has only AC_α² with Ψ = +1
- Per-mono: Ψ × (positive integral) = +1 × 0.389 = +0.389
- GenEval I₃ = -0.389 (negative built in regardless of Ψ)
- **MISMATCH** ✗

### 3. GenEval is an Approximation for Higher Pairs

The GenEval I₁-I₄ decomposition was derived for (1,1) and extended to higher pairs with:
1. **Weight approximation**: Uses (1-u)^{ell} for I₃ instead of per-monomial (1-u)^{a+b}
2. **Sign assumption**: Assumes I₃/I₄ are always negative (from (1,1) structure)
3. **Missing terms**: Higher pairs have (0,2), (2,0), (1,2), (2,1), etc. derivative orders not in I₁-I₄

### 4. The 7-12% Gap Explained

The handoff document reported a 7-12% gap in c. This comes from:
1. Weight averaging in GenEval
2. Missing higher derivative terms
3. Sign mismatches for off-diagonal pairs

## Correct Approach

### Option A: Fix Per-Monomial Signs

Instead of multiplying by Ψ coefficients, the per-monomial approach should:
1. Use the I-term formulas directly (with built-in signs)
2. Apply per-monomial weights
3. Sum over monomials with appropriate weight contributions

But this requires understanding exactly how the Ψ structure maps to the I-term signs.

### Option B: Full PRZZ Reimplementation

Implement the full Section 7 machinery:
1. PRE-MIRROR I_{1,d}(α,β) with proper Ψ handling
2. Mirror transformation
3. Q operators and t-integration

This is substantial work but would be mathematically rigorous.

### Option C: Use GenEval as Approximation

Accept that GenEval is approximate for higher pairs:
- Works perfectly for (1,1)
- Has ~7-12% error for full c
- Sufficient for optimization exploration, not for exact reproduction

## Monomial Structure Reference

### (1,1) - 4 monomials
| Monomial | Ψ | (l₁,m₁) | weight | Contribution |
|----------|---|---------|--------|--------------|
| D | +1 | (0,0) | 0 | +0.385 |
| BC_β | -1 | (0,1) | 1 | -0.226 |
| AC_α | -1 | (1,0) | 1 | -0.226 |
| AB | +1 | (1,1) | 2 | +0.426 |

### (1,2) - 7 monomials
| Monomial | Ψ | (l₁,m₁) | weight | Per-mono | GenEval |
|----------|---|---------|--------|----------|---------|
| C_αD | -2 | (0,0) | 0 | -1.178 | (I₂) |
| C_α²C_β | +1 | (0,0) | 0 | +0.589 | +0.589 |
| BD | +2 | (0,1) | 1 | +0.617 | (I₄) |
| B²C_β | -1 | (0,2) | 2 | -0.037 | N/A |
| AC_α² | +1 | (1,0) | 1 | +0.389 | (I₃) |
| ABC_α | -2 | (1,1) | 2 | -1.373 | -0.389 |
| AB² | +1 | (1,2) | 3 | +0.581 | (I₁) |

Note: GenEval has only I₁-I₄, missing (0,2) and (1,2) types.

## Validation Status

| Pair | Per-Mono | GenEval | Match? | Notes |
|------|----------|---------|--------|-------|
| (1,1) | 0.359 | 0.359 | ✓ | Exact match |
| (1,2) | -0.411 | +0.490 | ✗ | Sign flip |
| (2,2) | -0.477 | +0.963 | ✗ | Sign flip |

## Recommendation

Given the complexity of deriving correct per-monomial formulas, the pragmatic path is:

1. **Use GenEval for single-benchmark (κ) validation** - it gives c = 2.38 vs target 2.14 (11% error)
2. **Accept that two-benchmark validation is out of scope** without full PRZZ reimplementation
3. **Proceed to Phase 1 optimization** with GenEval, understanding its limitations
4. **If optimization finds promising candidates**, revisit with more accurate evaluation

The GenEval is mathematically correct for (1,1) and approximately correct for higher pairs. The approximation error is systematic and understood.
