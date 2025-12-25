# Handoff Document: Session 6 - December 18, 2025

## Executive Summary

**Major progress on (1-u) weight formula and I-term computation**. The GeneralizedItermEvaluator now correctly computes all I-terms for (1,1) with perfect match to oracle (difference < 1e-16).

**However**, the two-benchmark validation gate still fails due to polynomial structure differences between κ and κ* benchmarks causing ratio mismatch (2.01 vs target 1.10).

## Key Accomplishments

### 1. Correct (1-u) Weight Formula Discovered

**Bug found**: The GeneralizedItermEvaluator used wrong weight formula:
```python
# WRONG (old):
self.I1_weight_exp = przz_ell1 + przz_ell2  # = (ell-1) + (ellbar-1) = ell + ellbar - 2
self.I3_weight_exp = przz_ell1              # = ell - 1
self.I4_weight_exp = przz_ell2              # = ellbar - 1
```

**Correct formula (validated against oracle)**:
```python
# CORRECT (new):
self.I1_weight_exp = ell + ellbar  # (1-u)^{ell+ellbar}
self.I3_weight_exp = ell           # (1-u)^{ell}
self.I4_weight_exp = ellbar        # (1-u)^{ellbar}
```

**Key insight**: The (1-u) weight comes from the number of singleton blocks in the Ψ expansion:
- I₁ (AB monomial): 2 singletons → (1-u)²
- I₂ (D monomial): 0 singletons (paired block) → no weight
- I₃ (-AC monomial): 1 singleton → (1-u)¹
- I₄ (-BC monomial): 1 singleton → (1-u)¹

### 2. (1,1) Oracle Validation: PERFECT MATCH

```
I₁ = 0.426028  (target: 0.426028) ✓
I₂ = 0.384629  (target: 0.384629) ✓
I₃ = -0.225749 (target: -0.225749) ✓
I₄ = -0.225749 (target: -0.225749) ✓
Total = 0.359159 (target: 0.359159) ✓
```

### 3. TWO-C Ψ Expansion Improvements

- Added `source_p` tracking to `MonomialSeparatedC` for weight computation
- Updated `expand_pair_to_monomials_separated()` to combine across p-configs by default
- Verified monomial counts: (1,1)=4, (2,2)=12, (3,3)=27 ✓

### 4. Clean-Path Evaluator Updates

- Fixed structural bug in `compute_c_contribution()` (was multiplying by t-integral twice)
- Updated `eval_monomial()` to use correct (1-u)^{a+b} weight formula
- Added `compute_c_direct_postmirror()` for direct POST-MIRROR computation

## Outstanding Issue: Two-Benchmark Ratio Mismatch

### Current Results

| Pair | κ Total | κ* Total | Ratio |
|------|---------|----------|-------|
| (1,1) | 0.359 | 0.300 | 1.20 |
| (2,2) | 0.963 | 0.403 | 2.39 |
| (3,3) | 0.031 | 0.003 | 11.74 |
| **Total** | **2.385** | **1.187** | **2.01** |

**Target**: κ=2.137, κ*=1.938, Ratio=1.10

### Root Cause Analysis

The κ and κ* polynomials have fundamentally different structures:
- κ: P₂ is degree 3, P₃ is degree 5, Q is quadratic
- κ*: P₂ is degree 2, P₃ is degree 2, Q is linear

This leads to very different integral magnitudes, which is **mathematically correct**.

### (2,2) I-Term Breakdown

| Term | κ | κ* | Ratio |
|------|---|-----|-------|
| I₁ | 0.573 | 0.271 | 2.11 |
| I₂ | 0.909 | 0.341 | 2.67 |
| I₃ | -0.259 | -0.105 | 2.48 |
| I₄ | -0.259 | -0.105 | 2.48 |
| **Total** | **0.963** | **0.403** | **2.39** |

All I-terms have consistent ~2.4 ratio, suggesting the issue is in the integral structure, not a single term.

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/psi_separated_c.py` | Added `source_p` tracking, updated combine logic |
| `src/section7_clean_evaluator.py` | Fixed (1-u) weight formula, structural bug |
| `src/przz_generalized_iterm_evaluator.py` | **CRITICAL FIX**: Corrected weight formula |
| `docs/AGENT_FINDINGS_CONSOLIDATED.md` | Created with agent investigation results |

## Recommended Next Steps

### Priority 1: Verify κ* Polynomial Transcription
- Re-extract coefficients from PRZZ TeX lines 2587-2598
- Compare against `data/przz_parameters_kappa_star.json`
- Look for any transcription errors that could affect integral magnitudes

### Priority 2: Check for Polynomial Normalization
- Search PRZZ for any degree-dependent normalization factors
- Check if PRZZ's c definition includes polynomial normalization
- Look for ∫P²du = 1 or similar constraints

### Priority 3: Test with Artificial Polynomials
- What if we use κ polynomial degrees with κ* coefficient magnitudes?
- This would separate degree effects from coefficient effects

## Test Commands

```bash
# Verify (1,1) oracle match
cd przz-extension && PYTHONPATH=. python3 -c "
from src.polynomials import load_przz_polynomials
from src.przz_generalized_iterm_evaluator import GeneralizedItermEvaluator
P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
eval = GeneralizedItermEvaluator(P1, P1, Q, 4/7, 1.3036, 1, 1, 60)
print(f'I1={eval.eval_I1():.6f}, I2={eval.eval_I2():.6f}')
print(f'I3={eval.eval_I3():.6f}, I4={eval.eval_I4():.6f}')
print(f'Total={eval.eval_I1()+eval.eval_I2()+eval.eval_I3()+eval.eval_I4():.6f}')
"

# Run two-benchmark gate
cd przz-extension && PYTHONPATH=. python3 src/test_two_benchmark_gate.py
```

## SESSION 6 CONTINUATION - Investigation Results

### Priority 1 COMPLETED: κ* Polynomial Transcription VERIFIED

**All κ* polynomials match PRZZ TeX lines 2587-2598 exactly**:

| Polynomial | PRZZ TeX | Our JSON | Match |
|------------|----------|----------|-------|
| P₁ tilde_coeffs | [0.052703, -0.657999, -0.003193, -0.101832] | ✓ | ✓ |
| P₂ coeffs | [1.049837, -0.097446] | ✓ | ✓ |
| P₃ coeffs | [0.035113, -0.156465] | ✓ | ✓ |
| Q coeffs | [0.483777, 0.516223] | ✓ | ✓ |
| R | 1.1167 | ✓ | ✓ |

**Conclusion**: Transcription is NOT the problem.

### Priority 2 COMPLETED: Polynomial Normalization Search

**Searched PRZZ TeX for**:
- `∫P²du = 1` constraints → NOT FOUND
- Degree-dependent factors → NOT FOUND
- Polynomial norm normalization → NOT FOUND

**Found instead**:
- P(0)=0 constraint for all P polynomials ✓
- P₁(1)=1 constraint only for P₁ ✓
- No constraints on ∫P²du

### Key Finding: Polynomial L² Norm Ratios

| Integral | κ | κ* | Ratio |
|----------|---|-----|-------|
| ∫P₁²du | 0.307 | 0.300 | 1.02 |
| ∫P₂²du | 0.725 | 0.318 | **2.28** |
| ∫P₃²du | 0.007 | 0.003 | **2.83** |
| ∫Q²e^{2Rt}dt | 0.716 | 0.612 | 1.17 |

The (2,2) polynomial integral ratio (2.28) directly explains the (2,2) contribution ratio (2.39).

### Weight Formula Testing

Tested alternative (1-u) weight formulas:

| Formula | κ c | κ* c | Ratio |
|---------|-----|------|-------|
| ell+ellbar (current) | 2.385 | 1.187 | 2.01 |
| (ell-1)+(ellbar-1) | 2.934 | 1.687 | 1.74 |
| **Target** | **2.137** | **1.938** | **1.10** |

Alternative weight formulas make things worse, not better.

### Per-Pair Analysis

| Pair | κ c | κ* c | Ratio | Comment |
|------|-----|------|-------|---------|
| (1,1) | 0.359 | 0.300 | 1.20 | Closest to target 1.10 |
| (1,2) | 0.981 | 0.569 | 1.72 | Off |
| (1,3) | -0.020 | -0.040 | 0.51 | Inverted! |
| (2,2) | 0.963 | 0.403 | 2.39 | **Main culprit** |
| (2,3) | 0.072 | -0.048 | -1.51 | Sign flip! |
| (3,3) | 0.031 | 0.003 | 11.74 | Extreme |

**Key observation**: (1,1) ratio (1.20) is close to target (1.10). Higher pairs have increasingly bad ratios.

### L² Norm Scaling Test

Tested various polynomial norm scalings:

| Scaling | κ c | κ* c | Ratio |
|---------|-----|------|-------|
| None (raw) | 2.385 | 1.187 | 2.01 |
| × √(norms) | 1.275 | 0.391 | 3.26 |
| × norms | 0.758 | 0.122 | 6.22 |
| / √(norms) | 9.382 | 2.021 | 4.64 |
| **Target** | **2.137** | **1.938** | **1.10** |

No simple polynomial norm scaling fixes the ratio.

## Conclusion

The (1-u) weight formula is correct and validated. The (1,1) oracle match is perfect.

The two-benchmark ratio mismatch (2.01 vs 1.10) is caused by:
1. **Polynomial structure differences**: κ P₂ degree 3, κ* P₂ degree 2
2. **∫P²du ratios**: κ/κ* = 2.28 for P₂
3. **This cascades to I-term ratios**: (2,2) pair ratio = 2.39

### Remaining Hypotheses

1. **PRZZ optimization includes implicit polynomial normalization** that we're not enforcing
2. **PRZZ uses a different c formula** with additional factors we haven't identified
3. **PRZZ's published κ* values may be incorrect** (unverified independently)
4. **Missing I₅ or other correction terms** that affect the balance

### Files Created This Session

| File | Contents |
|------|----------|
| `docs/INVESTIGATION_POLYNOMIAL_NORMALIZATION.md` | Detailed normalization search results |

### Next Steps

1. **Contact PRZZ authors** to clarify the c computation formula
2. **Search for PRZZ source code** if available online
3. **Try I₅ correction** - may rebalance the ratio
4. **Check if κ* values verified** by other researchers
