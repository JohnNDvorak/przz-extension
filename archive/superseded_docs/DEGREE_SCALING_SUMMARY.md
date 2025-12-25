# Polynomial Degree Scaling Analysis - Executive Summary

**Date:** 2025-12-17
**Analyst:** Claude Opus 4.5
**Status:** CRITICAL FINDINGS

## TL;DR

**The c computation is fundamentally broken.** Full DSL evaluation gives c_κ = -1.04 (should be +2.14).

This is NOT a degree-scaling issue. This is a sign error or missing term in the base formula.

## Key Findings

### 1. Full DSL Gives Negative c (CRITICAL BUG)

```
Expected:  c_κ  = +2.137
Computed:  c_κ  = -1.042  ← NEGATIVE!

Expected:  c_κ* = +1.939
Computed:  c_κ* = -0.587  ← NEGATIVE!
```

The ratio (1.77) is meaningless when both values have the wrong sign.

### 2. Per-Pair Breakdown Shows Wild Variation

| Pair | c_κ | c_κ* | Ratio | Expected ~1.10 |
|------|-----|------|-------|----------------|
| (1,1) | +0.347 | +0.293 | 1.18 | ✓ Close! |
| (1,2) | **-2.202** | -1.291 | 1.71 | ✗ Wrong sign |
| (2,2) | +0.955 | +0.398 | 2.40 | ✗ Too high |
| (2,3) | -0.175 | -0.000 | **379** | ✗ Exploded |
| (3,3) | +0.035 | +0.002 | 14.7 | ✗ Way too high |

### 3. I₂-Only vs Full DSL Comparison

| Metric | I₂-only test | Full DSL | Expected |
|--------|--------------|----------|----------|
| c_κ | +0.614 | **-1.042** | +2.137 |
| c_κ* | +0.275 | **-0.587** | +1.939 |
| Sign | Correct | **WRONG** | Positive |

**The derivatives made it WORSE, not better.**

### 4. Polynomial Degree Effects Confirmed

From I₂-only tests:

| Pair | κ P-degrees | κ* P-degrees | I₂ ratio |
|------|-------------|--------------|----------|
| (1,1) | (5,5) | (5,5) | 1.19 ✓ |
| (2,2) | (3,3) | (2,2) | 2.59 |
| (3,3) | (3,3) | (2,2) | **28.18** |

Higher polynomial degrees amplify the integral, especially for (3,3).

**L2 norms:** κ P₃ has 5.4× larger coefficients than κ* P₃.

## Root Cause Analysis

### Eliminated Hypotheses

1. **Degree-dependent normalization** ✗
   - Tested ℓ₁!·ℓ₂!, powers, degree-based factors
   - None bring ratio close to 1.10
   - Best attempt (1/(ℓ₁³·ℓ₂³)) still 18% off

2. **R-dependent scaling issue** ✗
   - R effect is consistent: factor of 1.29× across all pairs
   - Matches e^{2ΔR} prediction
   - This part works correctly

3. **Missing I₁,I₃,I₄ derivative terms** ✗
   - Including derivatives makes c NEGATIVE
   - Ratio still wrong (1.77 vs 1.10)
   - Not the solution

### Active Hypotheses

#### 1. Sign Convention Error (MOST LIKELY)

The full DSL gives c_κ = -1.042 vs expected +2.137.

Possible causes:
- Sign flip in I₁ algebraic prefactor
- Wrong (-1)^(ℓ₁+ℓ₂) convention for cross terms
- Derivative extraction has wrong sign

**Evidence:**
- (1,2) and (2,3) pairs are negative (expected for cross terms?)
- I₁ contributions are large and sometimes negative

**Next step:** Review sign conventions in `terms_k3_d1.py`, especially I₁ terms.

#### 2. Missing Global Factor

Raw c values are off by factors:
- c_κ: need ×(-2.05) to match
- c_κ*: need ×(-3.30) to match

Different factors suggest it's NOT just a global constant.

**Next step:** Check if PRZZ has a global factor involving R, θ, or polynomial norms.

#### 3. Formula Interpretation Error

Maybe the integral we're computing is not what PRZZ actually uses.

Possibilities:
- Different domain (not [0,1]×[0,1])
- Missing weight function
- Wrong exponential (e^{Rt} vs e^{2Rt})

**Next step:** Re-read PRZZ Section 6-7 integral definitions line-by-line.

#### 4. κ* Uses Different Formula

The κ* (simple zeros) formula might be different from κ (all zeros).

**Evidence:**
- Different needed correction factors (×-2.05 vs ×-3.30)
- Different polynomial structures

**Next step:** Check PRZZ Section 8 for distinctions between κ and κ*.

## Polynomial Degree Effects (Confirmed)

From the I₂-only analysis, we confirmed:

1. **Higher degrees amplify integrals non-linearly**
   - Synthetic test: degree 5 gives 2.4× larger integral than degree 1
   - Coefficient structure matters more than degree alone

2. **Per-pair variation is enormous**
   - (1,1): ratio 1.19 (close to expected)
   - (2,2): ratio 2.59 (2.4× too high)
   - (3,3): ratio 28.18 (25× too high!)

3. **Coefficient magnitude matters**
   - κ P₃ L2 norm: 5.4× larger than κ* P₃
   - This explains (3,3) explosion

4. **No simple normalization fixes it**
   - Tested factorial, power, degree-based schemes
   - Best: 1/(ℓ₁³·ℓ₂³) gives ratio 1.30 (still 18% off)

**Conclusion:** Polynomial degree effects are real and significant, but they're a secondary issue. The primary issue is that c has the **wrong sign**.

## Recommendations

### Immediate Priority

1. **Fix the sign error**
   - Review all sign conventions in DSL
   - Check I₁ algebraic prefactor
   - Verify (-1)^(ℓ₁+ℓ₂) terms
   - Test with (1,1) pair only (simplest case)

2. **Verify against known result**
   - PRZZ must have computed (1,1) pair value
   - Match our (1,1) term-by-term against their formula
   - This is the "golden checkpoint"

3. **Check for global factor**
   - Search PRZZ for normalization constants
   - Look for R, θ, or polynomial-dependent factors
   - Might be in Section 6 or 7

### Secondary Actions

4. **Polynomial degree normalization**
   - Once sign is fixed, revisit degree-dependent factors
   - Focus on why (3,3) explodes (25× too high)
   - May need ω-dependent normalization

5. **Verify κ* formula**
   - Check if simple zeros use different integral
   - Compare PRZZ Section 8 to Section 6-7

6. **Polynomial transcription check**
   - Re-extract κ* coefficients from PRZZ TeX
   - Verify no transcription errors

## Generated Artifacts

All analysis code in `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/`:

1. **analyze_polynomial_degree_scaling.py**
   - Synthetic degree tests
   - I₂-only integral ratios
   - Polynomial coefficient analysis
   - R-dependent scaling tests

2. **test_normalization_correction.py**
   - Tests factorial, power, degree normalizations
   - Per-pair normalization search
   - Combined schemes

3. **test_advanced_normalization.py**
   - Empirical correction factors
   - Weighted contribution schemes
   - Q-degree-dependent factors

4. **test_full_dsl_degree_scaling.py**
   - Complete DSL evaluation
   - Per-pair breakdown with I₁,I₃,I₄
   - Derivative contribution analysis

5. **POLYNOMIAL_DEGREE_ANALYSIS_REPORT.md**
   - Detailed technical report

## For Next Session

**DO NOT:**
- Continue testing normalization schemes
- Try to fix polynomial degrees
- Optimize polynomials

**DO:**
1. Fix the sign error (c should be positive!)
2. Match (1,1) pair term-by-term against PRZZ
3. Find any missing global factors
4. Once sign is fixed, THEN revisit degree effects

**Key Question:**
> "Why is c negative when it should be positive +2.137?"

This must be answered before any other work.
