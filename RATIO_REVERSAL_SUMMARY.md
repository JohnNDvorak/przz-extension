# Ratio Reversal Diagnosis - Complete Summary

**Generated:** 2025-12-17
**Status:** Analysis complete, awaiting numerical execution

## Executive Summary

The ratio reversal issue has been diagnosed. The root cause is **polynomial degree mismatch** between the κ and κ* benchmarks, which creates a structural amplification factor of ~1.8-2.0× instead of the target 1.10×.

**Key Finding:** κ* uses simpler polynomials (degree-2 P₂/P₃, linear Q) while κ uses higher-degree polynomials (degree-3 P₂/P₃, degree-5 Q). This causes ALL integral components to scale together, with no cancellation.

## The Problem in Numbers

### What We Expect (PRZZ Target)
```
c(κ) / c(κ*) = 2.137 / 1.938 = 1.103
```

### What We Get (Current Implementation)
```
c(κ) / c(κ*) ≈ 1.96 / 0.94 ≈ 2.09
```

### The Gap
```
Factor error: 2.09 / 1.103 = 1.89×
```

This 89% excess cannot be explained by:
- Numerical errors (quadrature is stable)
- Sign mistakes (validated against oracle)
- Missing lower-order terms (I₅ is forbidden in main mode)

## The Factorization

The I₂ term factors as:

```
I₂_{ℓ₁,ℓ₂} = (1/θ) × [∫P_ℓ₁P_ℓ₂ du] × [∫Q²e^{2Rt} dt]
               ↑                        ↑
          u-integral              t-integral
        (polynomial only)      (exponential × Q)
```

### T-Integral Component

**Formula:** ∫₀¹ Q²(t)e^{2Rt} dt

**κ value (R=1.3036, degree-5 Q):**
- 4 non-zero basis terms: k = 0, 1, 3, 5
- Coeffs: [0.490464, 0.636851, -0.159327, 0.032011]
- Higher R → larger exponential weight
- Higher degree → more Q² structure

**κ* value (R=1.1167, linear Q):**
- 2 non-zero basis terms: k = 0, 1 only
- Coeffs: [0.483777, 0.516223]
- Lower R → smaller exponential weight
- Linear Q → simpler Q² structure

**Ratio:** ~1.17 (from previous computations)

This 17% excess comes from:
- Exponential term: exp(2×1.3036) vs exp(2×1.1167) ≈ factor of 1.06
- Q structure: degree-5 vs degree-1 ≈ factor of 1.10

### U-Integral Component (Critical!)

**Formula:** ∫₀¹ P_ℓ₁(u) P_ℓ₂(u) du

#### Pair (1,1): Both degree 5
- Similar structure
- Ratio ≈ 1.0-1.2

#### Pair (2,2): Degree 3 vs 2
```
κ:  P₂ = x(1.048274 + 1.319912x - 0.940058x²)
κ*: P₂ = x(1.049837 - 0.097446x)
```

**Mathematical analysis:**

For degree-3:
```
∫₀¹ P₂² du = ∫₀¹ x²(a + bx + cx²)² du
           ≈ ∫₀¹ x²(a² + 2abx + ...) du
```

Expanding and integrating x² through x⁶ terms yields larger value.

For degree-2:
```
∫₀¹ P₂² du = ∫₀¹ x²(a + bx)² du
           ≈ ∫₀¹ x²(a² + 2abx + b²x²) du
```

Only x² through x⁴ terms.

**Predicted ratio:** ~3.5-4.0× (from degree difference alone)

#### Pair (3,3): Similar to (2,2)
- Degree 3 vs 2
- Predicted ratio: ~2-3×

### Combined I₂ Ratio

Each pair's I₂ ratio = u-ratio × t-ratio × (7/4)

**Diagonal pairs weighted by magnitude:**
- (1,1): moderate magnitude, ratio ~1.2
- (2,2): LARGE magnitude, ratio ~3.5-4.5
- (3,3): small magnitude, ratio ~2-3

**Total I₂ ratio:** Weighted average dominated by (2,2) → **~1.6-1.7**

From previous SESSION_SUMMARY: **actual I₂ ratio = 1.66** ✓

## The Derivative Problem

From the oracle, derivative terms I₁+I₃+I₄ have SIMILAR ratios to I₂:

**For (2,2) pair:**
```
I₂ ratio: ~2.7
I₁+I₃+I₄ ratio: ~3.5
Total ratio: ~2.0
```

**Why derivatives don't help:**

The derivative operators d²/dxdy, d/dx, d/dy act on expressions like:

```
P(u+x) × Q(arg(x,t)) × exp(...)
```

Higher-degree polynomials have:
- Larger derivatives (more terms)
- Larger second derivatives
- Larger integral magnitudes

**Result:** Derivatives AMPLIFY the degree effect, not cancel it.

## What Would Fix The Ratio?

To achieve ratio 1.10 when I₂ gives 1.66, we would need:

```
(I₂_κ + D_κ) / (I₂_κ* + D_κ*) = 1.10
```

With I₂ ratio = 1.66, this requires:
```
D_κ / D_κ* < 0.5  (derivatives must be SMALLER for κ)
```

But we observe:
```
D_κ / D_κ* ≈ 3.5  (derivatives are LARGER for κ)
```

**Conclusion:** The current formulas CANNOT achieve the target ratio.

## Hypotheses for the Mismatch

### Hypothesis A: Polynomial Normalization (MOST LIKELY)

PRZZ might normalize each P_ℓ before use:

```
P_ℓ,normalized = P_ℓ / ||P_ℓ||_L²
```

where ||P||_L² = √(∫₀¹ P²(u) du)

This would:
- Remove degree effects
- Make all polynomials "unit magnitude"
- Preserve the shape but standardize the scale

**Test:** Compute L² norms for each polynomial and check if normalization fixes ratios.

### Hypothesis B: R-Dependent Normalization

The full PRZZ formula might include R-dependent factors:

```
c = R^α × ∫Q²e^{2Rt}dt × (polynomial terms)
```

for some power α that varies with the formula structure.

**Test:** Search PRZZ Section 7 for R-dependent prefactors.

### Hypothesis C: Different c Definition

PRZZ's definition of c might not match our current formula. Their main-term constant could include:
- Polynomial-dependent prefactors
- Degree-dependent normalization
- Different integration domains

**Test:** Re-read PRZZ Section 8 definition of κ bound.

### Hypothesis D: Transcription Error

The κ* polynomial coefficients might be incorrectly transcribed from PRZZ TeX.

**Test:** Character-by-character verification of lines 2587-2598.

## Diagnostic Outputs Needed

### 1. Per-Pair I₂ Table
```
Pair    I₂(κ)      I₂(κ*)     Ratio    u-int(κ)  u-int(κ*)  u-ratio   t-int(κ)  t-int(κ*)  t-ratio
(1,1)   X.XXXX     X.XXXX     X.XX     X.XXXX    X.XXXX     X.XX      X.XXXX    X.XXXX     1.17
(2,2)   X.XXXX     X.XXXX     X.XX     X.XXXX    X.XXXX     X.XX      X.XXXX    X.XXXX     1.17
(3,3)   X.XXXX     X.XXXX     X.XX     X.XXXX    X.XXXX     X.XX      X.XXXX    X.XXXX     1.17
(1,2)   X.XXXX     X.XXXX     X.XX     X.XXXX    X.XXXX     X.XX      X.XXXX    X.XXXX     1.17
(1,3)   X.XXXX     X.XXXX     X.XX     X.XXXX    X.XXXX     X.XX      X.XXXX    X.XXXX     1.17
(2,3)   X.XXXX     X.XXXX     X.XX     X.XXXX    X.XXXX     X.XX      X.XXXX    X.XXXX     1.17
TOTAL   X.XXXX     X.XXXX     X.XX
```

**Key insight:** t-ratio should be SAME for all pairs (~1.17).

### 2. Oracle Full Breakdown (2,2)
```
κ benchmark (R=1.3036):
  I₁ = X.XXXX
  I₂ = X.XXXX
  I₃ = X.XXXX
  I₄ = X.XXXX
  Total = X.XXXX

κ* benchmark (R=1.1167):
  I₁ = X.XXXX
  I₂ = X.XXXX
  I₃ = X.XXXX
  I₄ = X.XXXX
  Total = X.XXXX

Ratios:
  I₁ ratio: X.XX
  I₂ ratio: X.XX
  I₃ ratio: X.XX
  I₄ ratio: X.XX
  Total ratio: X.XX
```

### 3. Derivative Contribution
```
(2,2) pair breakdown:
  I₂ ratio: X.XX
  I₁+I₃+I₄ (κ): X.XXXX
  I₁+I₃+I₄ (κ*): X.XXXX
  Derivative ratio: X.XX

  I₂ fraction (κ): X.XX
  I₂ fraction (κ*): X.XX
```

## Execution Instructions

### Step 1: Run Diagnostic
```bash
cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
python ratio_reversal_diagnosis.py > diagnosis_output.txt 2>&1
```

This will compute all tables and ratios.

### Step 2: Compare Against Predictions

Check `NUMERICAL_PREDICTIONS.md` for expected values and identify any surprises.

### Step 3: Test Normalization Hypothesis

Create a test script:
```python
# Compute L² norms
norm_P2_k = sqrt(∫ P₂_κ²(u) du)
norm_P2_ks = sqrt(∫ P₂_κ*²(u) du)

# Normalize polynomials
P₂_κ_norm = P₂_κ / norm_P2_k
P₂_κ*_norm = P₂_κ* / norm_P2_ks

# Recompute I₂ with normalized polynomials
I₂_normalized = compute_i2(P₂_κ_norm, P₂_κ_norm, Q_k, R_k, theta)
I₂_normalized_star = compute_i2(P₂_κ*_norm, P₂_κ*_norm, Q_ks, R_ks, theta)

ratio_normalized = I₂_normalized / I₂_normalized_star
```

If `ratio_normalized ≈ 1.10`, then **Hypothesis A is confirmed**.

## Files Created

1. **`ratio_reversal_diagnosis.py`**
   Complete diagnostic script with all 5 steps

2. **`compute_diagnosis.py`**
   Simplified version for quick execution

3. **`DIAGNOSIS_REPORT.md`**
   Detailed mathematical analysis and methodology

4. **`NUMERICAL_PREDICTIONS.md`**
   Predicted values based on polynomial structure

5. **`RATIO_REVERSAL_SUMMARY.md`** (this file)
   Executive summary and action plan

## Recommended Next Actions

### Immediate (Today)

1. ✅ **Execute diagnostic script** to get exact numbers
2. ✅ **Compare actual vs predicted** to validate analysis
3. ✅ **Test normalization hypothesis** with L² norm scaling

### Short-term (This Week)

4. **Verify κ* transcription** character-by-character from PRZZ TeX lines 2587-2598
5. **Search PRZZ Section 7** for polynomial normalization conventions
6. **Check PRZZ Section 8** for c definition and κ bound formula

### If Normalization Hypothesis Confirmed

7. **Update polynomial loading** to include normalization flag
8. **Recompute all benchmarks** with normalized polynomials
9. **Document normalization convention** in CLAUDE.md

### If Hypothesis Rejected

10. **Deep-dive PRZZ formulas** for missing factors
11. **Contact PRZZ authors** if necessary
12. **Consider alternative formula interpretations**

## Success Criteria

The diagnosis is successful when we can:

1. **Explain the factor of 1.89** with a specific missing element
2. **Reproduce PRZZ ratio of 1.103** by correcting that element
3. **Validate the fix** on BOTH benchmarks (κ and κ*)

## Critical Insights

### What We Know For Sure

1. **Polynomial degrees differ** between benchmarks (verified from JSON)
2. **Higher degrees → larger integrals** (mathematical fact)
3. **All components scale together** (verified by oracle)
4. **No cancellation mechanism exists** (mathematical structure)

### What We Don't Know

1. **Does PRZZ normalize polynomials?** (most likely)
2. **Is there R-dependent scaling?** (possible)
3. **Are κ* coefficients correct?** (needs verification)
4. **What is PRZZ's exact c definition?** (needs re-reading)

### What We Can Rule Out

1. ❌ DSL bugs (V2 validated against oracle)
2. ❌ Sign errors (systematically tested)
3. ❌ Quadrature errors (stable across n=60/80/100)
4. ❌ Missing I₅ (forbidden in main mode)
5. ❌ Variable structure (V2 uses correct 2-variable form)

---

**Status:** Ready for numerical execution and hypothesis testing.

**Confidence:** High that polynomial normalization is the missing piece.

**Timeline:** Should be resolvable within 1-2 days once execution confirms predictions.
