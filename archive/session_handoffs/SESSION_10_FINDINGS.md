# Session 10 Findings: Series vs Section7 Discrepancy

## Executive Summary

**Critical Discovery**: The `Section7Evaluator` formula for computing `d²F/dxdy` has an error that causes a 0.55% discrepancy from the mathematically correct product rule. However, this "error" matches the PRZZ oracle, suggesting the oracle was computed with the same formula.

## The Bug in Section7

### Location
In `Section7Evaluator._integral_11()`, when computing cross terms involving `dQ/dx` and `dexp/dy`:

### What Section7 Does (WRONG)
```python
# term_B4 computes:
P_L * P_R * Qp * da_dx * Qt * R * db_dy * E2
```

### What It Should Be (CORRECT)
```python
# The full C_dx × E_dy term should use the SUM:
P_L * P_R * Qp * da_dx * Qt * R * (da_dy + db_dy) * E2
```

### Numerical Impact
At test point (u,t) = (0.512980, 0.512980):
- Section7's term_B4: -0.0501511765
- Correct C_dx × E_dy: -0.0025379419
- Missing term: 0.0476132346

This missing term (and similar ones in other terms) causes:
- Section7 d²F/dxdy: 1.5952617477
- Correct product rule: 1.5865388079
- Discrepancy: 8.72e-03 (0.55%)

## Validation Results

### Series Approach (Mathematically Correct)
- Computes f[1,1] via Taylor coefficient multiplication
- Result: 1.5865388079
- **Matches full product rule exactly** (diff: 3.64e-11)
- **Matches finite difference** ground truth

### Section7 Approach (Matches Oracle)
- Computes d²F/dxdy using explicit product rule
- Result: 1.5952617477
- Has the exp derivative splitting bug
- **Matches the PRZZ oracle** (0.359159)

## Implications

1. **The PRZZ oracle 0.359159 was computed with Section7's formula**, not the pure product rule.

2. **PsiSeriesEvaluator is mathematically correct** but won't match the oracle because it uses the correct product rule.

3. **The 3.5% error** in PsiSeriesEvaluator vs oracle is not a bug in Series - it's because Section7 (and PRZZ) use a different formula.

## Decision Point

For PRZZ reproduction: Use Section7's formula (matches oracle)
For mathematical correctness: Use Series approach (correct product rule)

The discrepancy may have implications for optimization - using the wrong formula could affect κ improvement strategies.

## Recommendation

Keep both evaluators:
- Section7 for oracle matching and PRZZ reproduction
- Series for mathematically rigorous computations

Document that the PRZZ oracle uses a specific formula structure that differs from the pure product rule by approximately 0.55% in the mixed derivative term.

---

## Additional Finding: Section7 vs GenEval Sign Convention

### The Issue

Section7Evaluator and GeneralizedItermEvaluator give different results for pairs other than (1,1):

| Pair | Section7 | GenEval | Match? |
|------|----------|---------|--------|
| (1,1) | +0.359159 | +0.359159 | ✓ |
| (1,2) ×2 | -1.247955 | +0.980812 | ✗ |
| (2,2) | +1.804065 | +0.962902 | ✗ |

### Root Cause

GenEval's `eval_I3()` and `eval_I4()` methods return **negative** values:
```python
return -(I3_base + (1.0 / self.theta) * I3_deriv)  # Built-in negative sign
```

Section7's `_integral_10` and `_integral_01` return **positive** values:
```python
return base + (1.0 / self.theta) * deriv  # No negative sign
```

### Why (1,1) Works But Higher Pairs Don't

For (1,1), the Ψ expansion has:
- I3-type monomial: `-1×A×C_α` with coefficient **-1**
- Section7 computes: (-1) × (positive integral) = negative ✓

For (1,2), the Ψ expansion has:
- I3-type monomial: `A×C_α²` with coefficient **+1**
- Section7 computes: (+1) × (positive integral) = positive ✗
- GenEval computes: -(positive value) = negative

### Implications

1. **Section7Evaluator only works correctly for (1,1)**
2. **Higher pairs require GenEval's I-term decomposition** with built-in sign conventions
3. **Two-benchmark gate fails with both evaluators** (different root causes)

### Two-Benchmark Gate Results

| Benchmark | Target c | GenEval c | Section7 c |
|-----------|----------|-----------|------------|
| κ (R=1.3036) | 2.137 | 2.385 (+11.6%) | 2.431 (+13.7%) |
| κ* (R=1.1167) | 1.938 | 1.187 (-38.8%) | 0.224 (-88.4%) |

**GenEval is more reliable** (closer to targets for both benchmarks), but both evaluators fail the two-benchmark gate.

The ratio analysis:
- Target ratio: 2.137 / 1.938 = 1.103
- GenEval ratio: 2.385 / 1.187 = 2.009 (82% off)
- Section7 ratio: 2.431 / 0.224 = 10.85 (884% off)

### Recommendation

Use **GeneralizedItermEvaluator (GenEval)** for production. Section7Evaluator has the (1,1) oracle match but fails for higher pairs due to sign convention mismatch.

The remaining 11-39% error in GenEval suggests a missing normalization or formula factor that affects R-dependent scaling.

---
Date: Session 10
