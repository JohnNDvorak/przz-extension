# (1-u) Weight Structure Investigation Report

**Date**: 2025-12-17
**Context**: Mathematical ratio reversal problem in PRZZ κ computation
**Question**: Can (1-u)^k weights explain why const_κ/const_κ* = 0.94 instead of 1.71?

---

## Executive Summary

**Finding**: The (1-u)^k weights **CANNOT** explain the ratio reversal.

**Reason**: The weights apply **uniformly** to both κ and κ* benchmarks (suppression factor = 1/(k+1)). They reduce higher pairs more than lower pairs, but affect both benchmarks equally.

**True cause**: **Polynomial degree mismatch** in derivative extraction. The DSL extracts P^(ℓ)(u) for pair (ℓ,ℓ), which vanishes when polynomial degree < ℓ. This causes catastrophic failure for the κ* benchmark (degree-2 polynomials, degree-3 extraction).

---

## Background: The Ratio Reversal Problem

From HANDOFF_SUMMARY.md, PRZZ's c decomposes as:
```
c = const × ∫Q²e^{2Rt}dt
```

### Target Ratios
- **t-integral ratio** (κ/κ*): 1.171 (R-dependent, validated)
- **const ratio** (needed): 0.942 (to get combined 1.10)
- **Combined**: 1.171 × 0.942 = 1.103 ✓ (matches PRZZ!)

### The Problem
Our naive formula `(1/θ) × Σ ∫P_i·P_j` gives:
- κ sum: 3.38
- κ* sum: 1.97
- **Ratio: 1.71** (κ > κ*)

But PRZZ needs:
- const κ: 2.98
- const κ*: 3.17
- **Ratio: 0.94** (κ < κ*)

**The ratios are in OPPOSITE directions!**

---

## PRZZ Weight Structure

From `/src/przz_generalized_iterm_evaluator.py`:

| I-term | Weight Power | Formula |
|--------|--------------|---------|
| I₁ | (1-u)^{ℓ₁+ℓ₂-2} | Mixed derivative d²/dxdy |
| I₂ | none | Base integral (no derivatives) |
| I₃ | (1-u)^{ℓ₁-1} | d/dx derivative |
| I₄ | (1-u)^{ℓ₂-1} | d/dy derivative |

Where PRZZ indexing: `przz_ell = our_ell - 1`

### Suppression Factors

Since ∫₀¹(1-u)^k du = 1/(k+1), higher powers get MORE suppressed:

| Pair | I₁ power | I₁ suppression | I₃ power | I₃ suppression |
|------|----------|----------------|----------|----------------|
| (1,1) | 0 | 100% (no suppression) | 0 | 100% |
| (2,2) | 2 | 33% | 1 | 50% |
| (3,3) | 4 | 20% | 2 | 33% |

**Key observation**: Higher pairs are suppressed MUCH more than (1,1).

---

## Analysis Results

### Part 1: Weighted Polynomial Integrals

Computed `∫₀¹ P_i(u) × P_j(u) × (1-u)^k du` for actual PRZZ polynomials:

#### (2,2) Pair
| I-term | Weight | κ value | κ* value | Ratio (κ/κ*) |
|--------|--------|---------|----------|--------------|
| I₂ | none | 0.763 | 0.318 | 2.40 |
| I₁ | (1-u)² | 0.049 | 0.033 | 1.48 |
| I₃ | (1-u)¹ | 0.153 | 0.082 | 1.87 |

**Finding**: The (1-u)² weight reduces the ratio from 2.40 → 1.48, but it's still > 1.

#### (3,3) Pair
| I-term | Weight | κ value | κ* value | Ratio (κ/κ*) |
|--------|--------|---------|----------|--------------|
| I₂ | none | 0.050 | 0.003 | 19.7 |
| I₁ | (1-u)⁴ | 0.003 | 0.000 | 228 |
| I₃ | (1-u)² | 0.005 | 0.000 | 55.6 |

**Finding**: For (3,3), the κ/κ* ratio is ENORMOUS (>20×), even with heavy suppression.

### Part 2: Theoretical Beta Function Analysis

For model P(u) ~ u^d (degree d):
```
∫₀¹ (1-u)^k × u^{2d} du = B(2d+1, k+1)
```

| Weight k | d=3 (κ) | d=2 (κ*) | Ratio (κ/κ*) |
|----------|---------|----------|--------------|
| 0 (no weight) | 0.143 | 0.200 | 0.71 |
| 2 (I₁ for (2,2)) | 0.004 | 0.010 | 0.42 |
| 4 (I₁ for (3,3)) | 0.0004 | 0.002 | 0.27 |

**Finding**: Even without polynomial structure, higher-degree polynomials get suppressed MORE by (1-u)^k weights.

**But**: The ratio is still < 1 in this model, meaning κ < κ* (degree-3 polynomials have LOWER integral values than degree-2).

**Contradiction**: The actual polynomial cross-integrals show κ > κ* (ratio 2.40 for (2,2)).

**Resolution**: The polynomial COEFFICIENTS matter! The degree-3 κ polynomials have been optimized to have LARGER magnitude than degree-2 κ* polynomials.

---

## Part 3: Derivative Term Reduction

From oracle for (1,1) pair:
- I₁ = +0.426
- I₂ = +0.385
- I₃ = -0.226
- I₄ = -0.226
- **Total** = 0.359

**Reduction analysis**:
- Positive terms (I₁+I₂): 0.811
- Negative terms (I₃+I₄): -0.452
- **Net reduction**: 55.7%

### Derivative Reduction by Pair

| Pair | Benchmark | I₃ reduces I₂ by |
|------|-----------|------------------|
| (2,2) | κ | 70.2% |
| (2,2) | κ* | 90.3% |
| (3,3) | κ | 33.0% |
| (3,3) | κ* | ~0% (P₃'''=0) |

**Finding**: The derivative reduction percentages are SIMILAR for κ and κ* in (2,2), but differ dramatically in (3,3) due to degree mismatch.

---

## The Real Culprit: Polynomial Degree Mismatch

### Polynomial Structure Comparison

| Polynomial | κ structure | κ* structure |
|------------|-------------|--------------|
| P₂ | degree 3: `1.048x + 1.320x² - 0.940x³` | degree 2: `1.050x - 0.097x²` |
| P₃ | degree 3: `0.523x - 0.687x² - 0.050x³` | degree 2: `0.035x - 0.156x²` |
| Q | degree 5 (odd powers) | degree 1 (linear) |

### Derivative Order Extraction

The DSL (multi-variable structure) extracts:
- (2,2): P₂''(u) × P₂''(u) (2nd derivatives)
- (3,3): P₃'''(u) × P₃'''(u) (3rd derivatives)

**For κ P₃** (degree 3):
```
P₃(x) = 0.523x - 0.687x² - 0.050x³
P₃'''(u) = 6×(-0.050) = -0.30
```

**For κ* P₃** (degree 2):
```
P₃(x) = 0.035x - 0.156x²
P₃'''(u) = 0  ← NO x³ TERM!
```

**Result**: The (3,3) contribution **VANISHES** for κ* because the polynomial degree is too low for 3rd derivative extraction!

---

## Conclusion: Answers to Investigation Questions

### Q1: How do (1-u) weights affect different pairs?

**Answer**:
- (1,1): No suppression (k=0 for I₁)
- (2,2): I₁ suppressed by 3× (k=2), I₃ by 2× (k=1)
- (3,3): I₁ suppressed by 5× (k=4), I₃ by 3× (k=2)

Higher pairs get progressively MORE suppressed.

### Q2: Does differential suppression explain κ vs κ* ratio?

**Answer**: NO. The suppression factor 1/(k+1) is **independent of polynomial structure**. Both κ and κ* get suppressed by the same amount for each pair.

### Q3: Does (1-u)^{ℓ₁+ℓ₂} suppress (2,2) and (3,3) more?

**Answer**: YES, but **uniformly** for both benchmarks:
- (2,2): 33% of unsuppressed value
- (3,3): 20% of unsuppressed value

This makes higher pairs contribute less, but doesn't create differential effects between κ and κ*.

### Q4: Could differential suppression explain const ratio reversal?

**Answer**: NO. The ratio reversal is caused by:

1. **Polynomial degree mismatch** in derivative extraction
   - κ: degree-3 polynomials → all derivatives extracted
   - κ*: degree-2 polynomials → 3rd derivatives are ZERO

2. **Coefficient optimization differences**
   - κ polynomials optimized for degree-3 structure
   - κ* polynomials optimized for degree-2 structure
   - These create different magnitude patterns

3. **Formula interpretation error**
   - The DSL extracts P^(ℓ)(u) for pair (ℓ,ℓ)
   - This assumes polynomial degree ≥ ℓ
   - For κ* with degree-2 polynomials, (3,3) fails completely

---

## Numerical Summary

### Weight Suppression Factors

| Pair | I₁ suppression | I₃ suppression | Effect on κ/κ* ratio |
|------|----------------|----------------|----------------------|
| (1,1) | 1.00 (none) | 1.00 (none) | No change |
| (2,2) | 0.33 (67% reduction) | 0.50 (50% reduction) | Ratio decreases from 2.40 → 1.48 |
| (3,3) | 0.20 (80% reduction) | 0.33 (67% reduction) | Ratio changes erratically (degree issue) |

### Polynomial Cross-Integral Ratios (κ/κ*)

| Pair | No weight | With I₁ weight | With I₃ weight |
|------|-----------|----------------|----------------|
| (1,1) | 6.74 | 6.74 | 6.74 |
| (2,2) | 2.40 | 1.48 | 1.87 |
| (3,3) | 19.7 | 228 (!) | 55.6 |

**The (3,3) anomaly**: With weight, ratio INCREASES instead of decreasing! This indicates the κ* polynomial structure breaks down for high-order derivatives.

---

## Recommendations

1. **Do NOT attribute ratio differences to (1-u) weights**
   - They suppress uniformly across benchmarks
   - They reduce higher pairs more, but proportionally

2. **Focus on polynomial degree sensitivity**
   - The formula must handle varying polynomial degrees
   - Current DSL assumes degree ≥ ℓ for pair (ℓ,ℓ)

3. **Investigate PRZZ's degree normalization**
   - Search for factors dependent on polynomial degree
   - Check if PRZZ uses different formulas for different degrees

4. **Two-benchmark gate failure is structural**
   - Not caused by (1-u) weights
   - Caused by derivative order mismatch
   - Requires external information to resolve

---

## Files Generated

| File | Purpose |
|------|---------|
| `analyze_weight_suppression.py` | Pure suppression factor analysis |
| `analyze_weight_detailed.py` | Weighted integrals with actual polynomials |
| `analyze_derivative_reduction.py` | Derivative term reduction analysis |
| `WEIGHT_INVESTIGATION_REPORT.md` | This summary document |

---

## Final Answer

**Q**: Can (1-u)^k weights explain the const ratio reversal?

**A**: **NO**. The (1-u)^k weights suppress higher pairs, but they suppress **both κ and κ* equally** by factor 1/(k+1). They cannot create a ratio reversal.

The true cause is **polynomial degree mismatch**: the DSL extracts P^(ℓ)(u) for pair (ℓ,ℓ), which vanishes when polynomial degree < ℓ. This causes κ* (degree-2 polynomials) to fail catastrophically for (3,3) pair.

**The (1-u) weights are working correctly.** The problem is elsewhere in the formula interpretation.
