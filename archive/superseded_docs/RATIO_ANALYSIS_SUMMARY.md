# κ vs κ* Ratio Analysis Summary

## Executive Summary

The analysis reveals that **individual pair ratios vary dramatically (0.43 to 2.83)**, but this variation is NOT the problem. The real issue is that **both κ and κ* computations give negative c values**, indicating a systematic error in the derivative term evaluation (I₁, I₃, I₄).

## Key Findings

### 1. U-Integral Ratios (I₂ component only)

| Pair | κ value | κ* value | Ratio κ/κ* | Target Ratio |
|------|---------|----------|------------|--------------|
| (1,1) | 0.307 | 0.300 | 1.024 | 1.103 |
| (1,2) | 0.470 | 0.307 | **1.530** | 1.103 |
| (1,3) | -0.011 | -0.027 | **0.430** | 1.103 |
| (2,2) | 0.725 | 0.318 | **2.279** | 1.103 |
| (2,3) | -0.011 | -0.027 | **0.430** | 1.103 |
| (3,3) | 0.007 | 0.003 | **2.833** | 1.103 |

**Range:** 0.430 to 2.833 (6.6x variation)
**Coefficient of Variation:** 63.4%

### 2. T-Integral Ratio

| Component | κ value | κ* value | Ratio |
|-----------|---------|----------|-------|
| ∫Q²e^(2Rt) dt | 0.716 | 0.612 | 1.171 |

### 3. Full c Values (Including I₁+I₂+I₃+I₄)

| Pair | κ value | κ* value | Ratio |
|------|---------|----------|-------|
| (1,1) | +0.347 | +0.293 | 1.18 |
| (1,2) | **-2.202** | **-1.291** | 1.71 |
| (1,3) | -0.003 | +0.009 | -0.31 |
| (2,2) | +0.955 | +0.398 | 2.40 |
| (2,3) | -0.175 | -0.000 | 379 (!) |
| (3,3) | +0.035 | +0.002 | 14.7 |
| **TOTAL** | **-1.042** | **-0.587** | **1.77** |

**Target total c:** κ = 2.137, κ* = 1.938, ratio = 1.103

### 4. Polynomial Coefficient Differences

The polynomials between κ and κ* differ significantly:

**P2 coefficients:**
- κ:  `1.048x + 1.320x² - 0.940x³` (degree 3)
- κ*: `1.050x - 0.097x²` (degree 2)
- Ratio at x²: **-13.5** (sign flip!)

**P3 coefficients:**
- κ:  `0.523x - 0.687x² - 0.050x³` (degree 3)
- κ*: `0.035x - 0.156x²` (degree 2)
- Ratio at x¹: **14.9** (order of magnitude difference!)

**Q coefficients:**
- κ:  degree 5 polynomial (has x², x³, x⁴, x⁵ terms)
- κ*: **linear** (only constant + x¹)
- Higher degree terms are MISSING in κ*

## Root Cause Analysis

### Why Individual Pairs Have Different Ratios

This is **EXPECTED** behavior. The pairs combine with different weights in the final sum, so:

```
c_total(κ) / c_total(κ*) = 1.103  ← This should hold
c_pair(κ) / c_pair(κ*)  ≠ 1.103  ← These can vary
```

The individual pair ratios compensate each other to achieve the target total ratio.

### Why We Get Negative c Values

The derivative terms (I₁, I₃, I₄) are **dominating** the sum and making c negative:

- I₂ (u-integral × t-integral) is positive for most pairs
- I₁+I₃+I₄ (derivative terms) are large and negative
- This overwhelms I₂, resulting in negative total

**This indicates:**
1. Either the derivative formulas have sign errors
2. Or there's a missing normalization factor
3. Or the κ* polynomials are fundamentally incompatible with the current implementation

### The κ* Polynomial Simplicity

The κ* benchmark uses:
- **Linear Q** (only 2 terms vs 4 terms for κ)
- **Lower degree P₂, P₃** (degree 2 vs degree 3)
- **Smaller R** (1.1167 vs 1.3036)

This suggests κ* was optimized under different constraints (simple zeros vs all zeros).

## Key Questions

### Q1: Is the ratio reversal coming from specific pairs or uniform?

**ANSWER:** The ratio variation is **pair-specific**, ranging from 0.43 to 2.83. However, this variation is EXPECTED - the pairs should compensate when summed.

### Q2: Which pairs show the LARGEST discrepancy from target ratio 1.103?

**ANSWER:** Ranked by absolute deviation:
1. (3,3): ratio = 2.833 (+157% error)
2. (2,2): ratio = 2.279 (+107% error)
3. (1,2): ratio = 1.530 (+39% error)
4. (2,3): ratio = 0.430 (-61% error)
5. (1,3): ratio = 0.430 (-61% error)
6. (1,1): ratio = 1.024 (-7% error)

### Q3: Are the polynomial degree differences causing the issue?

**PARTIAL YES:** The κ* polynomials have lower degrees, which changes the integral magnitudes. But this alone doesn't explain the negative c values.

## Recommendations

### Immediate Actions

1. **Verify (1,1) pair independently** - It's the simplest and should work
2. **Check sign convention for I₁ in cross-pair terms** - The (2,3) ratio is 379x!
3. **Validate the κ* polynomial transcription** - Degree mismatches are suspicious

### Medium-term Investigation

1. **Compare I₂-only vs full computation** - Isolate whether issue is in derivatives
2. **Test with modified polynomial degrees** - Use κ* degree with κ coefficients
3. **Check for R-dependent normalization** - κ* has smaller R

### Long-term Considerations

The κ* benchmark may require:
- Different formula interpretation for linear Q
- Modified derivative handling for lower-degree polynomials
- Separate validation pathway from the main κ computation

## Files Created

1. `analyze_kappa_ratio.py` - Computes u-integral and t-integral ratios
2. `detailed_ratio_analysis.py` - Polynomial coefficient and shape comparison
3. `simple_ratio_breakdown.py` - Full c computation per pair
4. `RATIO_ANALYSIS_SUMMARY.md` - This summary document

## Mathematical Insight

The target ratio is:

```
c(κ) / c(κ*) = 2.137 / 1.938 = 1.103
```

But this comes from:

```
c = exp(R·(1-κ))
```

So:
```
c(κ) / c(κ*) = exp(R_κ·(1-κ)) / exp(R_*·(1-κ*))
             = exp(1.3036·0.5827) / exp(1.1167·0.5925)
             = exp(0.7598) / exp(0.6616)
             = 2.138 / 1.938
             = 1.103
```

The ratio depends on **both R and κ**, not just the polynomials!

Therefore, individual component ratios (which don't include R scaling) are NOT expected to equal 1.103.

**The mystery is not the ratio variation - it's the negative c values.**
