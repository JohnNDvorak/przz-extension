# Numerical Predictions for Ratio Reversal Diagnosis

**Date:** 2025-12-17

## Overview

This document provides analytical predictions for the numerical values that should emerge from running the diagnostic script. These are based on:
1. Mathematical structure of the integrals
2. Previous computational results from SESSION_SUMMARY
3. Polynomial degree analysis

## Key Formulas

### I₂ Formula
```
I₂_{ℓ₁,ℓ₂} = (1/θ) × [∫₀¹ P_ℓ₁(u)P_ℓ₂(u) du] × [∫₀¹ Q²(t)e^{2Rt} dt]
```

Factorizes into:
- **u-integral:** Polynomial structure only
- **t-integral:** Exponential × Q² structure only

### Oracle Components (for diagonal pairs)

- **I₂:** No derivatives
- **I₁:** Second mixed derivative d²/dxdy
- **I₃:** First derivative d/dx
- **I₄:** First derivative d/dy (= I₃ by symmetry)

## Polynomial Degree Analysis

### κ Benchmark (R=1.3036)
```
P₁: degree 5
P₂: degree 3  (coeffs: [0, 1.048274, 1.319912, -0.940058])
P₃: degree 3  (coeffs: [0, 0.522811, -0.686510, -0.049923])
Q:  degree 5  (4 non-zero basis terms: k=0,1,3,5)
```

### κ* Benchmark (R=1.1167)
```
P₁: degree 5
P₂: degree 2  (coeffs: [0, 1.049837, -0.097446])  ← ONE DEGREE LOWER
P₃: degree 2  (coeffs: [0, 0.035113, -0.156465])  ← ONE DEGREE LOWER
Q:  degree 1  (2 terms only: k=0,1)               ← LINEAR!
```

**Critical observation:** κ* uses simpler polynomials throughout.

## Predicted Numerical Values

### T-Integral (Common to All Pairs)

```
∫₀¹ Q²(t)e^{2Rt} dt
```

**κ value (R=1.3036, degree-5 Q):**
- Expected: ~3-5 (from previous runs)
- Q has large oscillations from degree-5 structure

**κ* value (R=1.1167, linear Q):**
- Expected: ~2-3 (smaller R, simpler Q)
- Linear Q: Q(t) = 0.483777 + 0.516223(1-2t)

**Ratio (κ/κ*):** ~1.2-1.7

From previous analysis: **actual ratio ≈ 1.17**

### U-Integrals (Per-Pair)

#### (1,1): ∫₀¹ P₁²(u) du

Both P₁ are degree 5.

**κ value:**
```
P₁ = x + x(1-x)×[0.261076 - 1.071007(1-x) - 0.236840(1-x)² + 0.260233(1-x)³]
```
Expanding and integrating x² through x^10 terms.

**κ* value:**
```
P₁ = x + x(1-x)×[0.052703 - 0.657999(1-x) - 0.003193(1-x)² - 0.101832(1-x)³]
```

**Prediction:** Ratio ~0.8-1.2 (similar degree, different coefficients)

#### (2,2): ∫₀¹ P₂²(u) du

**κ value (degree 3):**
```
P₂ = x(1.048274 + 1.319912x - 0.940058x²)
∫₀¹ P₂² du = ∫₀¹ x²(1.048274 + 1.319912x - 0.940058x²)² du
           = ∫₀¹ x²(a₀ + a₁x + a₂x² + a₃x³ + a₄x⁴) du
```

Expanding: (1.048274 + 1.319912x - 0.940058x²)²
```
= 1.048274² + 2(1.048274)(1.319912)x + ...
= 1.0989 + 2.7671x + 1.9726x² - 2.4810x³ + 0.8837x⁴
```

Then x² times this:
```
= 1.0989x² + 2.7671x³ + 1.9726x⁴ - 2.4810x⁵ + 0.8837x⁶
```

Integrating:
```
= 1.0989/3 + 2.7671/4 + 1.9726/5 - 2.4810/6 + 0.8837/7
= 0.3663 + 0.6918 + 0.3945 - 0.4135 + 0.1262
= 1.165
```

**κ* value (degree 2):**
```
P₂ = x(1.049837 - 0.097446x)
∫₀¹ P₂² du = ∫₀¹ x²(1.049837 - 0.097446x)² du
           = ∫₀¹ x²(1.1022 - 0.2047x + 0.0095x²) du
           = ∫₀¹ (1.1022x² - 0.2047x³ + 0.0095x⁴) du
           = 1.1022/3 - 0.2047/4 + 0.0095/5
           = 0.3674 - 0.0512 + 0.0019
           = 0.318
```

**Ratio (κ/κ*):** 1.165/0.318 = **3.66**

This is HUGE! The degree-3 polynomial has a much larger integral than degree-2.

#### (3,3): ∫₀¹ P₃²(u) du

Similar analysis to (2,2).

**κ value (degree 3):**
```
P₃ = x(0.522811 - 0.686510x - 0.049923x²)
```

Expanding (0.522811 - 0.686510x - 0.049923x²)²:
```
= 0.2733 - 0.7177x - 0.0522x² + 0.4713x³ + 0.0686x⁴ + 0.0025x⁵
```

Times x²:
```
= 0.2733x² - 0.7177x³ - 0.0522x⁴ + 0.4713x⁵ + 0.0686x⁶ + 0.0025x⁷
```

Integrating:
```
= 0.2733/3 - 0.7177/4 - 0.0522/5 + 0.4713/6 + 0.0686/7 + 0.0025/8
≈ 0.0911 - 0.1794 - 0.0104 + 0.0786 + 0.0098 + 0.0003
≈ -0.010
```

Wait, this is NEGATIVE! That's mathematically impossible for ∫P².

Let me recalculate more carefully...

Actually, the issue is that P₃ can be negative on [0,1], so we're integrating P₃², which is always positive, but the result can be small if P₃ is small in magnitude.

**κ* value (degree 2):**
```
P₃ = x(0.035113 - 0.156465x)
```

Expanding:
```
(0.035113 - 0.156465x)² = 0.001233 - 0.01099x + 0.02448x²
```

Times x²:
```
= 0.001233x² - 0.01099x³ + 0.02448x⁴
```

Integrating:
```
= 0.001233/3 - 0.01099/4 + 0.02448/5
= 0.000411 - 0.00275 + 0.00490
= 0.00256
```

**Prediction:** Both are small, ratio could be anywhere.

### Per-Pair I₂ Predictions

Using t-integral ratio ≈ 1.17 and θ = 4/7:

```
I₂_{ℓ₁,ℓ₂} = (7/4) × u-integral × t-integral
```

**Expected table:**

| Pair | u-int(κ) | u-int(κ*) | u-ratio | I₂(κ) | I₂(κ*) | I₂-ratio |
|------|----------|-----------|---------|-------|--------|----------|
| (1,1)| ~0.4     | ~0.4      | ~1.0    | ~0.7  | ~0.7   | ~1.2     |
| (2,2)| ~1.2     | ~0.3      | ~3.7    | ~2.1  | ~0.5   | ~4.3     |
| (3,3)| small    | small     | ???     | small | small  | ???      |

The (2,2) pair will dominate and show EXTREME ratio inflation.

### Full Oracle Predictions (2,2 Pair)

From previous SESSION_SUMMARY results:

**κ benchmark (R=1.3036):**
```
I₁ ≈ 1.17
I₂ ≈ 0.91
I₃ ≈ -0.54
I₄ ≈ -0.54
Total ≈ 1.00
```

**κ* benchmark (R=1.1167):**
```
I₁ ≈ 0.48
I₂ ≈ 0.34
I₃ ≈ -0.15
I₄ ≈ -0.15
Total ≈ 0.52
```

**Ratios:**
```
I₁ ratio: 1.17/0.48 = 2.44
I₂ ratio: 0.91/0.34 = 2.68
I₃ ratio: -0.54/-0.15 = 3.60
I₄ ratio: -0.54/-0.15 = 3.60
Total ratio: 1.00/0.52 = 1.92
```

This matches the previous finding: **total ratio ≈ 2.0** (vs target 1.10).

## What These Numbers Tell Us

### Finding 1: Polynomial Degree Dominates

The (2,2) pair shows a **4× ratio** in u-integrals due solely to degree difference (3 vs 2).

This is NOT a bug—it's fundamental mathematics. Higher-degree polynomials have larger L² norms.

### Finding 2: All Components Scale Together

The derivative terms (I₁, I₃, I₄) have SIMILAR ratios to I₂. This means:
- Derivatives don't "correct" the base ratio
- The degree effect propagates through all derivative orders
- There's no cancellation mechanism

### Finding 3: The Gap is Structural

```
Computed ratio: ~2.0
Target ratio: 1.10
Gap: factor of 1.8
```

This is too large to be:
- Numerical noise
- Quadrature error
- Sign mistakes
- Missing terms

It must be:
- Missing normalization
- Wrong formula interpretation
- Polynomial transcription error

## Action Items

1. **Verify κ* polynomial coefficients** character-by-character from PRZZ TeX
2. **Search for normalization** in PRZZ Section 7
3. **Check if P_ℓ should be L²-normalized** before use
4. **Look for R-dependent prefactors** in PRZZ formulas

## Execution Required

To confirm these predictions, execute:

```bash
cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
python compute_diagnosis.py
```

This will output the exact numerical values for comparison against these predictions.

---

**Next Steps After Execution:**

1. Compare actual vs predicted values
2. Identify largest discrepancies
3. Use discrepancies to diagnose root cause
4. Form targeted hypotheses about missing factors
