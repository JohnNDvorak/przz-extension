# Complete Ratio Breakdown Analysis: I₁, I₂, I₃, I₄ for (2,2) Pair

**Date:** 2025-12-17
**Authors:** Analysis based on exact oracle implementation
**Key Question:** Where does the 2.43x ratio (vs target 1.10) come from?

---

## Executive Summary

The (2,2) pair shows a **total ratio of 2.43** instead of the expected 1.10. We have decomposed this into:

1. **Polynomial P₂ contribution: 2.28x**
2. **Q·exp(2Rt) contribution: 1.17x**
3. **Combined: 2.28 × 1.17 = 2.67x for I₂**

**KEY FINDING:** The 2.43x ratio comes almost entirely from the **polynomial P₂ having much larger L² norm in the κ benchmark** (∫P₂² = 0.725) compared to κ* (∫P₂² = 0.318).

This is a **2.28x factor from P₂ alone**, with an additional **1.17x from Q·exp(2Rt)**.

**Derivatives actually HELP** by reducing the ratio from 2.67 to 2.43 (an 9% improvement).

---

## Full Numerical Results

### Individual Term Ratios: κ / κ*
| Term | κ Value | κ* Value | Ratio | Distance from Target 1.10 |
|------|---------|----------|-------|--------------------------|
| I₁ (mixed deriv) | 1.1686 | 0.4897 | 2.39 | +1.29 |
| I₂ (base) | 0.9088 | 0.3406 | **2.67** | **+1.57 (WORST)** |
| I₃ (d/dx) | -0.5444 | -0.2117 | 2.57 | +1.47 |
| I₄ (d/dy) | -0.5444 | -0.2117 | 2.57 | +1.47 |
| **Total** | **0.9887** | **0.4070** | **2.43** | **+1.33** |

### Polynomial Integrals: ∫₀¹ ... du
| Integral | κ Value | κ* Value | Ratio |
|----------|---------|----------|-------|
| ∫ P₂² du | 0.7250 | 0.3181 | **2.28** |
| ∫ P₂ du | 0.7291 | 0.4924 | 1.48 |
| ∫ P₂·P₂' du | 1.0198 | 0.4535 | 2.25 |
| ∫ (P₂')² du | 2.0864 | 0.9102 | 2.29 |
| ∫ (1-u)²·P₂² du | 0.0693 | 0.0334 | 2.07 |

### Q Integrals: ∫₀¹ ... dt
| Integral | κ Value (R=1.3036) | κ* Value (R=1.1167) | Ratio |
|----------|-------------------|-------------------|-------|
| ∫ Q² dt | 0.3436 | 0.3229 | 1.06 |
| ∫ Q²·e^{2Rt} dt | 0.7163 | 0.6117 | **1.17** |

---

## Decomposition of I₂ Ratio

I₂ has the cleanest structure:
```
I₂ = (1/θ) × [∫ P₂²(u) du] × [∫ Q²(t) e^{2Rt} dt]
```

### κ Benchmark (R=1.3036)
- ∫ P₂² du = 0.7250
- ∫ Q²·e^{2Rt} dt = 0.7163
- I₂ = (1/θ) × 0.7250 × 0.7163 = 0.9088

### κ* Benchmark (R=1.1167)
- ∫ P₂² du = 0.3181
- ∫ Q²·e^{2Rt} dt = 0.6117
- I₂ = (1/θ) × 0.3181 × 0.6117 = 0.3406

### Ratio Decomposition
```
I₂(κ) / I₂(κ*) = [0.7250/0.3181] × [0.7163/0.6117]
                = 2.2790 × 1.1710
                = 2.6685
```

**Polynomial contributes:** 2.28x (85% of the ratio)
**Q·exp contributes:** 1.17x (15% of the ratio)

---

## Why Is P₂ So Different?

### Structural Differences
| Benchmark | P₂ Structure | Tilde Degree | ∫P₂² du |
|-----------|-------------|--------------|---------|
| κ | degree 3 (x·P̃, P̃ deg 2) | 2 | 0.7250 |
| κ* | degree 2 (x·P̃, P̃ deg 1) | 1 | 0.3181 |

The κ P₂ is **higher degree** and has **larger coefficients**, leading to a 2.28x larger L² norm.

### Coefficient Magnitudes
From the data files:
- κ P₂: tilde_coeffs has 3 terms (degree 2 in tilde)
- κ* P₂: tilde_coeffs has 2 terms (degree 1 in tilde)

This is a **fundamental structural difference** in the polynomials, not a transcription error.

---

## Why Does exp(2Rt) Contribute 1.17x?

The exponential integral ratio is:
```
∫ Q²·e^{2Rt} dt |_κ  / ∫ Q²·e^{2Rt} dt |_κ* = 1.17
```

This comes from:
1. **Different R values:** R_κ = 1.3036 vs R_κ* = 1.1167 (ratio 1.17)
2. **Different Q polynomials:** ∫Q² dt ratio = 1.06 (Q polynomials are similar)

The exponential weighting e^{2Rt} amplifies the effect of R:
- At t=1: e^{2·1.3036}/e^{2·1.1167} = e^{2.607}/e^{2.233} = 13.6/9.3 = 1.46
- At t=0.5: e^{1.3036}/e^{1.1167} = 3.68/3.05 = 1.21
- Integrated over [0,1] with Q²(t): ratio ≈ 1.17

---

## Effect of Derivatives on Ratio

From the term structure analysis:

| Component | Ratio |
|-----------|-------|
| I₂ alone | 2.67 |
| I₂ + I₁ | 2.50 |
| I₂ + I₁ + I₃ + I₄ | 2.43 |

**Derivatives reduce the ratio by 0.24** (from 2.67 to 2.43), a **9% improvement**.

### Why Do Derivatives Help?

1. **I₁ has a better ratio (2.39)** than I₂ (2.67)
2. **I₃+I₄ are negative** and have ratio 2.57
3. The **sum of all derivatives** (I₁+I₃+I₄) has ratio **1.20** - very close to target!

```
Derivatives sum:
  κ:  I₁+I₃+I₄ = 1.169 - 0.544 - 0.544 = 0.080
  κ*: I₁+I₃+I₄ = 0.490 - 0.212 - 0.212 = 0.066
  Ratio: 0.080/0.066 = 1.20
```

The derivative terms have **strong internal cancellation** that is nearly proportional between the two benchmarks.

---

## Implications for Missing Normalization

### What We Can Rule Out

1. **Derivative extraction is broken:** All terms scale consistently
2. **Sign conventions:** Would affect absolute values, not ratios
3. **I₅ calibration:** I₅ is lower-order and forbidden in main mode
4. **DSL variable structure:** Oracle also shows 2.43x ratio

### What Remains Plausible

Given that the ratio decomposes cleanly as:
```
2.43 = 2.28(polynomial) × 1.17(Q·exp) × 0.92(derivative cancellation)
```

The missing normalization could be:

1. **Polynomial-dependent normalization**
   - PRZZ may normalize by ∫P₂² or some other polynomial norm
   - Factor: would need ~2.28x for P₂

2. **R-dependent normalization**
   - Factor involving R or exp(-R) in the prefactor
   - Would need to give ~1.17x scaling

3. **Combined polynomial × R normalization**
   - Most likely: something like (1/R) × (1/∫P₂²)
   - Would explain both components of the ratio

### Most Likely: Missing 1/(∫P²) Normalization

The fact that ∫P₂² differs by **exactly the main factor** (2.28 out of 2.43) suggests PRZZ might normalize polynomials by their L² norm.

**Hypothesis:** The actual formula should include:
```
I₂ = (1/θ) × [∫P₂²/∫P₂²] × [∫Q²·e^{2Rt} dt] = (1/θ) × [∫Q²·e^{2Rt} dt]
```

This would make I₂ **independent** of the polynomial magnitude, with all polynomial dependence coming through the derivative terms.

But wait - that can't be right, because then I₂ would depend only on Q and R, not on P₂ at all...

### Alternative: Different Integral Bounds

What if PRZZ uses different bounds? Let's check:
- If bounds were [-1,1] instead of [0,1], integral would be 2x larger (uniformly)
- This doesn't explain the 2.28x factor varying between benchmarks

---

## Key Numerical Facts

### P₂ L² Norms
- κ: ∫₀¹ P₂² du = 0.7250
- κ*: ∫₀¹ P₂² du = 0.3181
- **Ratio: 2.28**

### Q·exp Integrals
- κ: ∫₀¹ Q²·e^{2Rt} dt = 0.7163 (R=1.3036)
- κ*: ∫₀¹ Q²·e^{2Rt} dt = 0.6117 (R=1.1167)
- **Ratio: 1.17**

### Combined I₂
- κ: I₂ = 0.9088
- κ*: I₂ = 0.3406
- **Ratio: 2.67 = 2.28 × 1.17**

### Derivatives Reduce Ratio
- I₂ only: 2.67
- With derivatives: 2.43
- **Improvement: 9%**

---

## Recommendations

### Priority 1: Check PRZZ for Polynomial Normalization
Search PRZZ TeX for:
- Any division by ∫P² or ∫P_ℓ²
- Factors like "normalized polynomial"
- Any mention of L² norms in the polynomial setup

### Priority 2: Verify Q Polynomial Transcription
The Q difference is smaller (1.06 for ∫Q², 1.17 with exp), but should still verify:
- κ Q coefficients from PRZZ lines 1456-1460
- κ* Q coefficients from PRZZ lines 2595-2598

### Priority 3: Check PRZZ Section 7 Formulas
Re-read PRZZ Section 7 line-by-line for:
- Any prefactors on the I₂ formula (line 1548)
- Any R-dependent normalization
- Any polynomial-dependent normalization

### Priority 4: Test Polynomial Normalization Hypothesis
Try computing c with:
```python
I2_normalized = I2 / (integral_P2_squared)
```
and see if that brings the ratio closer to 1.10.

---

## Conclusion

The 2.43x ratio comes from:
- **85% from polynomial P₂ having 2.28x larger L² norm in κ vs κ***
- **15% from Q·exp integral being 1.17x larger** (due to R difference and Q difference)
- **Derivatives actually help by 9%** (reduce ratio from 2.67 to 2.43)

The problem is **NOT** in the derivative extraction - it's in understanding how PRZZ normalizes the polynomial contribution.

**Most likely missing piece:** A normalization factor involving ∫P₂² or some other polynomial-dependent quantity that would make the different benchmarks comparable.
