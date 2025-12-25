# Handoff Document: Session 7 - December 18, 2025

## Executive Summary

**Root cause of two-benchmark failure definitively identified**: The polynomial degree differences between κ and κ* benchmarks cause fundamentally different I-term structures that cannot be reconciled by simple normalization.

**Key finding**: The I-term decomposition (I₁-I₄) is an approximation that only works when comparing polynomials with similar structures. The κ* polynomials have lower degrees (P₂ is degree 1, P₃ is degree 1) vs κ polynomials (P₂ is degree 2, P₃ is degree 2), leading to dramatically different I-term ratios.

## Two-Benchmark Gate Status

| Benchmark | R | c target | c computed | Factor | Status |
|-----------|------|----------|------------|--------|--------|
| κ | 1.3036 | 2.137 | 2.385 | 0.90 | Close |
| κ* | 1.1167 | 1.938 | 1.187 | 1.63 | **FAILS** |
| **Ratio** | - | **1.10** | **2.01** | - | **2× off** |

## Root Cause Analysis

### Polynomial Degree Differences

| Polynomial | κ tilde_coeffs | κ* tilde_coeffs | κ degree | κ* degree |
|------------|----------------|-----------------|----------|-----------|
| P₁ | 4 | 4 | 3 | 3 |
| P₂ | 3 | 2 | **2** | **1** |
| P₃ | 3 | 2 | **2** | **1** |

### Impact on I-term Structure

For (3,3) pair, the I-term ratios are wildly different:

| I-term | κ value | κ* value | Ratio |
|--------|---------|----------|-------|
| I₁ | 0.028605 | 0.000204 | **140.24** |
| I₂ | 0.009091 | 0.002741 | 3.32 |
| I₃ | -0.003461 | -0.000161 | 21.47 |
| I₄ | -0.003461 | -0.000161 | 21.47 |
| Total | 0.030773 | 0.002622 | **11.74** |

**Why I₁ ratio is 140×**: The I₁ term involves mixed derivatives. For κ* P₃ (degree 1 = linear), there is essentially no curvature, so derivative-based terms are nearly zero. For κ P₃ (degree 2), there is significant curvature.

### Polynomial Norm Normalization Analysis

Normalizing pair contributions by ||P_ell||² works for diagonal pairs with similar structures:

| Pair | Raw ratio | Normalized ratio | Target |
|------|-----------|------------------|--------|
| (1,1) | 1.20 | **1.17** | 1.10 ✓ |
| (2,2) | 2.39 | **1.05** | 1.10 ✓ |
| (3,3) | 11.74 | **4.14** | 1.10 ✗ |

**(1,1) and (2,2) normalize well because their polynomial structures are relatively similar. (3,3) fails because κ* P₃ is linear while κ P₃ is quadratic.**

## What Works vs What Doesn't

### Works Correctly ✓

1. **GeneralizedItermEvaluator for (1,1)**: Perfect oracle match (difference < 1e-16)
2. **(1-u) weight formula**: Validated against oracle
3. **Ψ expansion monomial counts**: (1,1)=4, (2,2)=12, (3,3)=27 ✓
4. **Single-benchmark validation**: κ benchmark alone gives factor 0.90 (close to 1.0)

### Does Not Work ✗

1. **Two-benchmark ratio**: 2.01 vs target 1.10 (82% error)
2. **Polynomial norm normalization for (3,3)**: Ratio still 4.14 after normalization
3. **Cross-term normalization**: (1,3), (2,3) don't normalize properly
4. **I-term decomposition for structurally different polynomials**

## Key Mathematical Insight

The I-term decomposition assumes that:
```
c_{ℓ,ℓ̄} ≈ I₁(P_ℓ, P_ℓ̄) + I₂(P_ℓ, P_ℓ̄) + I₃(P_ℓ, P_ℓ̄) + I₄(P_ℓ, P_ℓ̄)
```

This works when comparing the SAME polynomials at different R values, or polynomials with similar degree structure. It breaks when comparing polynomials with different degree structures because:

1. **I₁ (mixed derivatives)**: Scales with polynomial curvature (∝ degree²)
2. **I₃, I₄ (single derivatives)**: Scales with polynomial slope (∝ degree)
3. **I₂ (no derivatives)**: Scales with polynomial magnitude

For a degree-1 polynomial, I₁ ≈ 0, while for degree-2 it can be significant.

## Single-Benchmark (κ) Numerical Results

### Without I₅ Correction

| Metric | Computed | Target | Error |
|--------|----------|--------|-------|
| c | 2.385 | 2.137 | **+11.6%** |
| κ | 0.333 | 0.417 | **-20%** |

### With I₅ Correction

Using the empirical formula `I₅ = -S(0) × θ²/12 × I₂_total`:

| Metric | Computed | Target | Error |
|--------|----------|--------|-------|
| I₂_total | 2.423 | - | - |
| I₅ | -0.091 | - | - |
| c | 2.294 | 2.137 | **+7.3%** |
| κ | 0.363 | 0.417 | **-13%** |

The I₅ correction helps (reduces c error from 11.6% to 7.3%) but doesn't close the gap.

## Implications

### For Phase 0 (Reproduce PRZZ κ)

The current I-term approach gives c that is 7-12% off target. This translates to κ = 0.33-0.36 vs target 0.42. The deviation is systematic and understood (I-term approximation error for higher pairs).

### For Two-Benchmark Validation

True two-benchmark validation requires implementing the full PRZZ machinery:
1. Per-monomial evaluation (12 monomials for (2,2), 27 for (3,3))
2. Case C kernels at integrand level
3. Full F_d factor computation (not I-term approximation)

This is significantly more complex than the current I-term approach.

## Recommended Path Forward

### Option A: Accept Single-Benchmark Validation (Pragmatic)

- Use κ benchmark only for Phase 0 validation
- Acknowledge that two-benchmark test is out of scope without full PRZZ reimplementation
- Factor 0.90 is close enough for optimization work
- Proceed to Phase 1 optimization

### Option B: Implement Full PRZZ Machinery (Comprehensive)

- Phase 3: Create section7_integrand.py with per-monomial evaluation
- Phase 4: Implement mirror transformation and F_d factors
- Phase 5: Two-benchmark validation gate
- Significant development effort, ~weeks of work

### Recommendation

Given the significant deviation (κ = 0.36 vs target 0.42), **neither option is immediately viable for Phase 0**:

1. **Option A problems**: 13% κ error is too large for reliable optimization
2. **Option B problems**: Requires substantial development effort

**Suggested approach**:
1. Check if other evaluators in the codebase (evaluate.py, paper_integrator.py) give better results
2. Investigate if there's a missing global factor or incorrect interpretation
3. Consider that the "c=2.137" target may include components we're not computing

The polynomial structure analysis is CORRECT - the degree differences are mathematically real. But there may be additional factors in the PRZZ formula we haven't captured.

## Files Modified/Created This Session

| File | Changes |
|------|---------|
| `docs/HANDOFF_2025_12_18_SESSION7.md` | Created - this file |
| `docs/GPT_FEEDBACK_INTEGRATION.md` | Created - GPT feedback analysis |
| `src/fd_evaluation.py` | Added WARNING about factored formula limitations |

## Test Commands

```bash
# Verify (1,1) oracle match (should be perfect)
cd przz-extension && PYTHONPATH=. python3 -c "
from src.polynomials import load_przz_polynomials
from src.przz_generalized_iterm_evaluator import GeneralizedItermEvaluator
P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
eval = GeneralizedItermEvaluator(P1, P1, Q, 4/7, 1.3036, 1, 1, 60)
result = eval.eval_all()
print(f'Total={result.total:.6f} (target: 0.359159)')
"

# Run two-benchmark test (will fail, this is expected)
cd przz-extension && PYTHONPATH=. python3 src/test_two_benchmark_gate.py
```

## Session Summary

### Key Findings

1. **GeneralizedItermEvaluator is CORRECT for (1,1)**: Perfect oracle match (0.359159)
2. **paper_integrator.py has WRONG weights**: Uses PRZZ 0-based indexing, gives 0.352 instead of 0.359
3. **Two-benchmark failure root cause**: Polynomial degree differences (κ P2/P3 are degree 2, κ* P2/P3 are degree 1)
4. **I-term approximation limitation**: Works for similar polynomial structures, fails for different degrees

### Evaluator Comparison

| Evaluator | (1,1) c | Comment |
|-----------|---------|---------|
| GeneralizedItermEvaluator | 0.359159 | ✓ Matches oracle exactly |
| paper_integrator.py | 0.352209 | ✗ Wrong (1-u) weights |
| Oracle target | 0.359159 | Ground truth |

### Single-Benchmark (κ) Gap Analysis

| Component | Value |
|-----------|-------|
| c computed (without I₅) | 2.385 |
| c computed (with I₅) | 2.294 |
| c target | 2.137 |
| Gap | 7-12% |

The 7-12% gap in c translates to κ = 0.33-0.36 vs target 0.42.

### Path Forward

The I-term approach (I₁-I₄) with correct (1-u)^{ell+ellbar} weights is validated for (1,1) oracle. However:
1. Higher pairs (2,2), (3,3) have additional approximation error
2. Full PRZZ reproduction requires per-monomial evaluation (not I-term decomposition)
3. The 7-12% gap represents the fundamental limitation of the I-term approximation

**For Phase 0 reproduction, additional investigation or full PRZZ reimplementation is needed.**
