# Session Summary: December 17, 2025

## Session Focus
Investigation of the Ψ combinatorial structure from PRZZ and discovery of critical V1 vs V2 DSL structural differences.

---

## Key Accomplishments

### 1. I-Term Monomial Evaluator for (1,1) - VALIDATED ✓

Created `src/przz_iterm_monomial_evaluator.py` that successfully maps Ψ monomials to I-term integral structures for the (1,1) pair:

| Monomial | Coefficient | Maps To | Value |
|----------|-------------|---------|-------|
| AB | +1 | I₁ (d²/dxdy) | +0.426 |
| D | +1 | I₂ (base) | +0.385 |
| AC | -1 | \|I₃\| (d/dx) | -0.226 |
| BC | -1 | \|I₄\| (d/dy) | -0.226 |
| **Total** | | | **0.359** |

This matches the oracle exactly (difference < 1e-15).

### 2. Critical V1 vs V2 DSL Discovery

**The V2 "fix" from GPT guidance is WRONG for cross-pairs!**

#### V1 (Multi-variable) Structure - CLOSER TO TARGET
- Uses (ℓ₁ + ℓ₂) total variables
- P arguments are SUMMED: P_ℓ(u + x₁ + ... + x_ℓ)
- Derivative order = ℓ₁ + ℓ₂
- For (1,2): d/dx₁ d/dy₁ d/dy₂ extracts P₁'(u) × P₂''(u)
- **Total c = 1.950** (91.3% of target 2.137)

#### V2 (Single-variable) Structure - FAILS FOR CROSS-PAIRS
- Uses only 2 variables (x, y) for ALL pairs
- P arguments are single: P_ℓ(u + x)
- Derivative order = 2 always
- For (1,2): d/dx d/dy extracts P₁'(u) × P₂'(u) (WRONG!)
- **Total c = -1.646** (negative, completely wrong!)

#### Why V2 Appeared to Work
V2 was validated only for (2,2), where it matches the oracle within 3%.
The oracle itself is designed specifically for the PRZZ ℓ₁=ℓ₂=1 case (our (2,2)).
For cross-pairs, V2 extracts the wrong polynomial derivatives.

### 3. Weight Structure Analysis

Discovered that PRZZ uses pair-dependent weights:
- PRZZ ℓ = our index - 1 (PRZZ starts at 0)
- I₁ weight = (1-u)^{(ℓ₁-1)+(ℓ₂-1)}
- I₃ weight = (1-u)^{ℓ₁-1}
- I₄ weight = (1-u)^{ℓ₂-1}

The oracle uses fixed weights (1-u)² for I₁, (1-u)¹ for I₃/I₄, which is only correct for (2,2).

Created `src/przz_generalized_iterm_evaluator.py` to implement correct pair-dependent weights.

---

## Current State of the Codebase

### Working Components
1. **V1 DSL** (`make_all_terms_k3` in `terms_k3_d1.py`): Multi-variable structure, gives c=1.950
2. **Oracle** (`przz_22_exact_oracle.py`): Correct for (2,2) only
3. **I-term monomial evaluator**: Validates (1,1) against oracle

### Broken/Incomplete Components
1. **V2 DSL** (`make_all_terms_k3_v2`): Wrong for cross-pairs, should NOT be used
2. **Generalized I-term evaluator**: Validates weight structure but not a complete solution

### Key Metrics
| Benchmark | Target c | V1 DSL c | Factor Needed |
|-----------|----------|----------|---------------|
| κ (R=1.3036) | 2.137 | 1.950 | 1.096 |
| κ* (R=1.1167) | 1.939 | 0.937* | 2.07 |

*κ* uses different polynomials with simpler structure

The factor ratio 1.096/2.07 = 0.53 indicates a **structural issue**, not global normalization.

---

## Files Modified/Created This Session

| File | Status | Purpose |
|------|--------|---------|
| `src/przz_iterm_monomial_evaluator.py` | Created | Maps Ψ monomials to I-term structures |
| `src/przz_generalized_iterm_evaluator.py` | Created | Pair-dependent weight evaluation |
| `docs/PSI_INVESTIGATION_FINDINGS.md` | Updated | Added V1 vs V2 analysis section |
| `docs/SESSION_SUMMARY_2025_12_17.md` | Created | This document |

---

## Next Steps (Priority Order)

### 1. Investigate the 9% Gap in V1 DSL
The V1 DSL gives c=1.950, missing 9% to reach target 2.137. Possible causes:
- Missing I₅ arithmetic correction term
- Polynomial coefficient transcription errors
- Missing normalization factor in PRZZ formula

**Action**: Compare V1 DSL term-by-term against PRZZ TeX formulas for each pair.

### 2. Two-Benchmark Structural Analysis
The gap differs drastically between κ (9%) and κ* (107%). This suggests:
- Polynomial-degree-dependent normalization
- Different formula structure for different polynomial degrees
- Possible transcription error in κ* coefficients

**Action**:
- Re-extract κ* coefficients from PRZZ TeX lines 2587-2598
- Compare polynomial structures (κ has degree 3 P₂/P₃, κ* has degree 2)
- Test if using κ polynomial degrees with κ* coefficients changes the ratio

### 3. Do NOT Use V2 DSL
The V2 single-variable structure gives wrong results for cross-pairs.
The GPT guidance about d²/dxdy for all pairs was incorrect.
V1's multi-variable structure (ℓ₁ + ℓ₂ total variables) is closer to PRZZ.

### 4. Consider Alternative Approaches
- Search for Feng's original code (mentioned in PRZZ TeX line 2566)
- High-precision arithmetic verification with mpmath
- Polynomial optimization to find coefficients that maximize c

---

## Technical Details for Next Session

### To Reproduce V1 DSL Result
```python
from src.evaluate import evaluate_c_full
from src.polynomials import load_przz_polynomials

P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}
result = evaluate_c_full(theta=4/7, R=1.3036, n=80, polynomials=polynomials, mode='main')
print(f"Total c: {result.total}")  # Should print ~1.950
```

### Key Constants
- θ = 4/7 ≈ 0.5714285714
- R (κ) = 1.3036
- R (κ*) = 1.1167
- Target c (κ) = 2.137
- Target κ = 0.417293962

### Important Files to Read
- `src/terms_k3_d1.py`: V1 term builders (use `make_all_terms_k3`, NOT `_v2`)
- `src/przz_22_exact_oracle.py`: Oracle for (2,2) validation
- `docs/PSI_INVESTIGATION_FINDINGS.md`: Full investigation history
- `CLAUDE.md`: Project guidelines and conventions

---

## Questions Still Unanswered

1. **Why does V2 match oracle for (2,2) but fail for cross-pairs?**
   - The oracle may be implementing a simplified formula that only works for diagonal pairs

2. **What is the correct normalization for cross-pairs?**
   - V1 uses factorial normalization 1/(ℓ₁! × ℓ₂!)
   - Are there additional factors?

3. **Why does the two-benchmark ratio differ (1.09 vs 2.07)?**
   - Is there polynomial-degree-dependent normalization in PRZZ?
   - Are the κ* polynomial coefficients correctly transcribed?

4. **Can Ψ monomials be evaluated for (2,2) and higher pairs?**
   - The (1,1) monomial approach works
   - Extending to 12 monomials for (2,2) requires understanding derivative mapping

---

## Summary Statement

**The session revealed that the V2 single-variable DSL "fix" is incorrect for cross-pairs. The original V1 multi-variable structure (using ℓ₁+ℓ₂ variables) gives c=1.950, which is 91.3% of the target. The remaining 9% gap, combined with the different factors needed for κ vs κ* benchmarks (1.09 vs 2.07), indicates an unresolved structural issue in the formula interpretation.**
