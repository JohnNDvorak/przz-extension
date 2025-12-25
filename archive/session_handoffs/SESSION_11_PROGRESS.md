# Session 11 Progress: Series Engine Validation

## Summary

**Coefficient Gate: PASS**
- Series engine f[1,1] matches finite difference within 1e-7 relative error
- Convention alignment verified (both use +P' without sign alternation)

**Two-Benchmark Gate: FAIL**
- Hybrid evaluator improves c slightly but ratio error remains 80%+
- The I1 bug fix accounts for ~2-3% improvement, not the 80% needed

## Key Accomplishments

### 1. Coefficient Gate Test Created and Passing

File: `tests/test_coefficient_gate.py`

Verifies that `PsiSeriesEvaluator._build_F_coefficients()` produces correct Taylor coefficients:
- f[0,0]: matches direct F(0,0) evaluation
- f[1,0]: matches finite-difference ∂F/∂x|_{0,0}
- f[0,1]: matches finite-difference ∂F/∂y|_{0,0}
- f[1,1]: matches finite-difference ∂²F/∂x∂y|_{0,0} within **1.86e-07** relative error

All 9 tests pass.

### 2. Sign Convention Aligned

The test uses the same convention as PsiSeriesEvaluator:
- P(u-x) Taylor expansion: f[i,0] = +P^(i)(u)/i! (no alternation)
- This matches the PRZZ/Section7 convention

### 3. Hybrid Evaluator Created

File: `src/hybrid_evaluator.py`

Combines:
- **Series-derived I1**: Uses `PsiSeriesEvaluator.compute_integral_grid()[(1,1,weight)]`
- **GenEval I2/I3/I4**: Unchanged from `GeneralizedItermEvaluator`

### 4. Hybrid vs GenEval Comparison

For all pairs, I1 differs by ~2-3%, I2/I3/I4 are identical:

| Pair | Hybrid I1 | GenEval I1 | Diff |
|------|-----------|------------|------|
| (1,1) | 0.413 | 0.426 | -0.013 |
| (2,2) | 0.560 | 0.573 | -0.013 |

The new (1,1) oracle value: **0.346606** (was 0.359159 with GenEval)

### 5. Two-Benchmark Gate Results

| Benchmark | Target c | Hybrid c | Error |
|-----------|----------|----------|-------|
| κ (R=1.3036) | 2.137 | 2.325 | +8.8% |
| κ* (R=1.1167) | 1.938 | 1.166 | -39.8% |
| **Ratio** | **1.10** | **1.99** | **+80.8%** |

**Gate: FAIL**

## Key Finding

The I1 cross-term bug identified by GPT is **real but insufficient**:
- Fixing I1 via series engine improves c by ~0.06 (2.5%)
- But the ratio error remains 80%+
- The fundamental issue is NOT in derivative computation

## Per-Pair Analysis

| Pair | κ contrib | κ* contrib | Ratio | Issue |
|------|-----------|------------|-------|-------|
| (1,1) | 0.347 | 0.293 | 1.18 | OK |
| (2,2) | 0.950 | 0.399 | **2.38** | P₂ degree differs |
| (3,3) | 0.030 | 0.003 | **11.6** | P₃ degree differs |
| (1,2) | 0.955 | 0.558 | 1.71 | |
| (1,3) | -0.024 | -0.040 | 0.60 | |
| (2,3) | 0.067 | -0.048 | **-1.41** | Sign flip! |

The (2,2) and (3,3) pairs have extreme ratios because:
- κ* P₂ is degree 2, κ P₂ is degree 3
- κ* P₃ is degree 2, κ P₃ is degree 3
- κ* Q is linear, κ Q is degree 4

## Root Cause Hypothesis (Refined from Session 10)

The κ* polynomials have simpler structure, leading to fundamentally different integral magnitudes. This is **mathematically correct** - ∫P²du depends on polynomial degree.

**Key insight from HANDOFF_SUMMARY:**
- PRZZ decomposition: `c = const × ∫Q²e^{2Rt}dt`
- Our naive const ratio: **1.71** (κ > κ*)
- PRZZ needs const ratio: **0.94** (κ < κ*)
- **Ratios are in OPPOSITE directions!**

This means PRZZ's formula has **NEGATIVE correlation** between ||P|| and contribution:
- Larger polynomials → smaller contributions (after all corrections)
- This is opposite to naive ∫P² expectation

The I1 cross-term bug we fixed accounts for only ~3% improvement. The fundamental issue is the entire const formula structure, not just one derivative term.

**Candidate causes** (from HANDOFF_SUMMARY):
- Derivative term subtraction
- (1-u) weights
- Case C kernels
- Ψ sign patterns

## Files Modified/Created

| File | Status |
|------|--------|
| `tests/test_coefficient_gate.py` | Created, 9 tests passing |
| `src/hybrid_evaluator.py` | Created |
| `run_two_benchmark_gate.py` | Created |
| `SESSION_11_PROGRESS.md` | This file |

## Recommended Next Steps

The coefficient gate confirms our series engine is mathematically correct. The issue is elsewhere:

1. **Search PRZZ paper for degree-dependent normalization**
   - Any factors like 1/(deg P)! or similar?

2. **Check if "c" definition includes polynomial normalization**
   - Is c normalized by ||P||² or similar?

3. **Test with same-degree polynomials**
   - Use κ polynomial structure with κ* coefficients
   - Isolate degree effect from coefficient effect

4. **Re-read PRZZ TeX around lines 2587-2598**
   - Where κ* polynomials are defined
   - Look for normalizing factors

## Disproven Hypotheses

1. **I1 cross-term bug causes ratio error** - Bug is real but only ~3% effect
2. **Series convention wrong** - Matches finite differences exactly
3. **Off-diagonal polynomial assignment bug** - Verified correct

## What Works

- Series coefficient extraction (verified by coefficient gate)
- I2, I3, I4 computation (identical in Hybrid vs GenEval)
- Sign conventions (aligned between series and tests)

## What Doesn't Work

- Two-benchmark gate (80% ratio error persists)
- Understanding the R-dependent/degree-dependent scaling

---

Date: Session 11
