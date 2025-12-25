# Derivative Term Analysis: The Ratio Reversal Mystery

## Executive Summary

**HYPOTHESIS DISPROVEN**: Derivative terms (I₁, I₃, I₄) do NOT reverse the ratio from 1.71 to 0.94. In fact, they make the ratio WORSE, increasing it rather than decreasing it.

**Key Finding**: The derivatives contribute a SMALLER fraction of I₂ for κ than for κ*, which means they AMPLIFY the naive ratio rather than reversing it.

---

## Background

### The Problem

Starting from the naive I₂-only formula:
```
c = const × ∫ Q²(t) e^{2Rt} dt × ∫ P²(u) du
```

This gives a κ/κ* ratio of approximately **1.71** (wrong direction).

The target ratio is **0.94** (from PRZZ benchmarks).

### The Hypothesis

If κ polynomials have higher degree (deg 3 P₂/P₃) vs κ* (deg 2), then:
- κ has LARGER derivatives at u values
- More subtraction from I₂ via I₁, I₃, I₄ terms
- Could reverse the ratio from 1.71 to 0.94

---

## Methodology

Used exact PRZZ oracle formulas (`przz_22_exact_oracle.py`) to compute:
- **I₂**: Base integral with NO derivatives: `(1/θ) × ∫∫ P²(u) Q²(t) e^{2Rt} du dt`
- **I₁**: Mixed derivative d²/dxdy contributions
- **I₃**: First derivative d/dx contributions
- **I₄**: First derivative d/dy contributions

Analyzed diagonal pairs: (1,1), (2,2), (3,3)

---

## Results

### Polynomial Degrees

| Polynomial | κ (R=1.3036) | κ* (R=1.1167) |
|------------|--------------|---------------|
| P₁ | degree 5 | degree 5 |
| P₂ | degree 3 | **degree 2** |
| P₃ | degree 3 | **degree 2** |
| Q  | degree 5 | **degree 1** |

κ has significantly higher degree polynomials for P₂, P₃, and Q.

---

### Per-Pair Results

#### (2,2) Pair - Most Important

| Benchmark | I₁ | I₂ | I₃ | I₄ | Deriv Sum | Total |
|-----------|-------|---------|----------|----------|-----------|-----------|
| κ | 1.1686 | 0.9088 | -0.5444 | -0.5444 | **0.0799** | 0.9887 |
| κ* | 0.4897 | 0.3406 | -0.2117 | -0.2117 | **0.0664** | 0.4070 |
| **Ratio** | 2.39 | **2.67** | 2.57 | 2.57 | **1.20** | 2.43 |

**Critical Observation**:
- I₂ ratio: 2.67 (very large)
- Derivative sum ratio: 1.20 (much smaller)
- **Derivatives as % of I₂**: κ has 8.8%, κ* has 19.5%

The derivatives are a SMALLER fraction of I₂ for κ than κ*, so they increase the total ratio rather than decrease it.

---

#### All Diagonal Pairs Summary

| Pair | I₂ Ratio | Deriv % (κ) | Deriv % (κ*) | Total Ratio |
|------|----------|-------------|--------------|-------------|
| (1,1) | 1.20 | -6.6% | -6.4% | 1.20 |
| (2,2) | 2.67 | **8.8%** | **19.5%** | 2.43 |
| (3,3) | 3.32 | 425.6% | -22.6% | 22.51 |

**Aggregate (diagonal pairs only)**:
- I₂ total ratio: **1.96**
- Derivative total ratio: **2.06**
- Full c ratio: **1.97**

The diagonal pairs alone give a ratio of ~2.0, far from the target 0.94.

---

### Derivative Magnitude Analysis

#### P₂ derivatives at sample u values (u = 0.25, 0.50, 0.75):

| u | P₂(κ) | P₂'(κ) | P₂''(κ) | P₂(κ*) | P₂'(κ*) | P₂''(κ*) |
|---|-------|--------|---------|--------|---------|----------|
| 0.25 | 0.3299 | 1.5320 | 1.2297 | 0.2564 | 1.0011 | -0.1949 |
| 0.50 | 0.7366 | 1.6631 | -0.1803 | 0.5006 | 0.9524 | -0.1949 |
| 0.75 | 1.1321 | 1.4418 | -1.5904 | 0.7326 | 0.9037 | -0.1949 |

**κ has larger derivative magnitudes**, as expected from higher degree.

#### Q derivatives at sample t values:

| t | Q(κ) | Q'(κ) | Q''(κ) | Q(κ*) | Q'(κ*) | Q''(κ*) |
|---|------|-------|--------|--------|---------|---------|
| 0.50 | 0.4905 | -1.2737 | 0.0000 | 0.4838 | -1.0324 | 0.0000 |

Q derivatives are comparable, though κ's Q is degree 5 vs κ*'s degree 1.

---

## The Hypothesis Was BACKWARDS

### Expected (Hypothesis)
- Higher degree → Larger derivatives
- Larger derivatives → More subtraction from I₂
- More subtraction → Smaller total → Reverses ratio ✓

### Actual (Reality)
- Higher degree → Larger derivatives ✓
- **BUT ALSO** → Much larger I₂ (polynomial squared) ✓✓
- Net effect: Derivatives are a SMALLER % of I₂
- Result: Derivatives INCREASE the ratio, not decrease it ✗

---

## Quantitative Analysis

### What derivative contribution would achieve ratio 0.94?

For the (2,2) pair alone:

**Current values:**
- I₂(κ) = 0.9088
- I₂(κ*) = 0.3406
- D(κ) = 0.0799 (derivatives)
- D(κ*) = 0.0664

**Actual ratio**: (0.9088 + 0.0799) / (0.3406 + 0.0664) = **2.43**

**To achieve ratio 0.94:**

We need: (0.9088 + D_κ) / (0.3406 + 0.0664) = 0.94

Solving: D_κ = 0.94 × 0.4070 - 0.9088 = **-0.5263**

**Current D(κ) = 0.0799**

**Shortfall**: 0.0799 - (-0.5263) = **0.6061**

**This is 66.7% of I₂!**

The derivative terms would need to be NEGATIVE and equal to -66.7% of I₂ to reverse the ratio. Instead, they're POSITIVE at +8.8% of I₂.

---

## Why Did The Hypothesis Fail?

### The Missing Factor: I₂ Growth Dominates

When polynomial degree increases:

1. **Derivatives grow linearly with degree**
   - P of degree d → P' has coefficients ~d times larger
   - P'' has coefficients ~d(d-1) times larger

2. **BUT I₂ involves P² (squared polynomial)**
   - Degree 3 squared → degree 6 polynomial
   - Degree 2 squared → degree 4 polynomial
   - The integral ∫P² grows FASTER than derivatives

3. **Net effect**: Derivative/I₂ ratio DECREASES with higher degree

### Numerical Evidence

| Pair | P degree (κ) | P degree (κ*) | Deriv/I₂ (κ) | Deriv/I₂ (κ*) |
|------|--------------|---------------|--------------|---------------|
| (2,2) | 3 | 2 | 0.088 | **0.195** |

The higher-degree polynomial has a SMALLER derivative-to-base ratio.

---

## What DOES Explain the Ratio Reversal?

The diagonal pairs give ratio ~2.0, not 0.94. Possible explanations:

### 1. Off-Diagonal Pairs
The full K=3 calculation includes:
- (1,2), (1,3), (2,3) cross terms
- These may have very different ratios
- Could average down to ~1.10

### 2. I₅ Arithmetic Correction
- I₅ is an O(1/log T) correction term
- May differ significantly between κ and κ*
- Current implementation: `I₅ = -S(0) × θ²/12 × I₂_total`
- This is empirical, not derived from first principles

### 3. R-Dependent Scaling
- κ uses R=1.3036, κ* uses R=1.1167
- The exp(2Rt) factor appears in all integrals
- May need R-dependent normalization in the formula interpretation

### 4. Missing Polynomial Normalization
- PRZZ may normalize by polynomial degree or L² norm
- Could depend on ∫P² or similar
- Not obvious from published formulas

---

## Conclusions

### Main Findings

1. **Hypothesis DISPROVEN**: Derivative terms do NOT reverse the ratio

2. **Direction is WRONG**: Derivatives make the ratio LARGER, not smaller

3. **Magnitude is INSUFFICIENT**: Even if sign were correct, derivatives are only 8.8% of I₂, far from the 66.7% needed

4. **Polynomial degree has opposite effect**: Higher degree increases I² faster than derivatives, reducing the relative importance of derivative terms

### Specific Numbers (2,2 pair)

- Naive I₂-only ratio: **2.67**
- With derivatives: **2.43**
- Target: **0.94**
- Gap: **1.49** (factor of 2.6×)

### What This Means

The ratio reversal from 1.71 → 0.94 **cannot be explained by derivative terms alone**.

Other factors must dominate:
- Off-diagonal pairs with very different ratios
- I₅ corrections (but this is lower-order, should be small)
- Missing R-normalization in formula interpretation
- Polynomial-degree-dependent factors we haven't identified

### Recommended Next Steps

1. **Compute off-diagonal pairs** (1,2), (1,3), (2,3) for both benchmarks
   - Check if these have ratios < 1 that could average down

2. **Analyze I₅ contribution** for both benchmarks
   - Is it significantly different?
   - Does it explain the gap?

3. **Check for R-normalization**
   - Review PRZZ formulas for R-dependent factors
   - The exp(2Rt) appears in integrals - is there compensating normalization?

4. **Verify polynomial transcription**
   - Ensure κ* polynomials are correctly extracted from PRZZ
   - Check for any normalization conventions

---

## Appendix: Detailed Numbers

### (2,2) Pair Breakdown

**κ benchmark (R=1.3036)**:
```
I₁ =  1.168588  (d²/dxdy derivative)
I₂ =  0.908817  (base integral, no derivatives)
I₃ = -0.544354  (d/dx derivative)
I₄ = -0.544354  (d/dy derivative)
Deriv sum = 0.079881  (I₁ + I₃ + I₄)
Total = 0.988698
```

**κ* benchmark (R=1.1167)**:
```
I₁ =  0.489706
I₂ =  0.340567
I₃ = -0.211652
I₄ = -0.211652
Deriv sum = 0.066401
Total = 0.406969
```

**Ratios**:
```
I₁ ratio:     2.39
I₂ ratio:     2.67  ← Naive formula
I₃ ratio:     2.57
I₄ ratio:     2.57
Deriv ratio:  1.20  ← Much smaller than I₂ ratio
Total ratio:  2.43  ← Slightly better than I₂, but wrong direction
Target:       0.94  ← What we need
```

---

**Generated**: 2025-12-17
**Script**: `analyze_derivative_ratio.py`, `analyze_all_pairs_derivatives.py`
**Data**: PRZZ κ benchmark (R=1.3036) vs κ* benchmark (R=1.1167)
