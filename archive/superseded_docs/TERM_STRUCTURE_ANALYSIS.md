# Term Structure Analysis: I₁, I₂, I₃, I₄ Comparison for (2,2) Pair

**Date:** 2025-12-17
**Context:** Investigating why κ/κ* ratio is 2.43 instead of expected 1.10

## Executive Summary

The (2,2) pair shows **ALL individual terms have ratios ~2.4-2.7**, but derivatives actually **help** bring the ratio closer to the target:

- **I₂ (base integral) ratio: 2.67** (WORST)
- **I₁ (mixed derivative) ratio: 2.39** (BEST)
- **I₃+I₄ (single derivatives) ratio: 2.57**
- **Total ratio: 2.43**

**KEY FINDING:** Adding derivatives **improves** the ratio from 2.67 → 2.43. The problem is not that derivatives make it worse; the problem is that **ALL terms are roughly 2.4x too large**, and I₂ is actually the worst offender.

---

## Detailed Results

### κ Benchmark (R=1.3036)
| Term | Value | Contribution to Total |
|------|-------|----------------------|
| I₁ | 1.1686 | 118.2% |
| I₂ | 0.9088 | 91.9% |
| I₃ | -0.5444 | -55.1% |
| I₄ | -0.5444 | -55.1% |
| **Total** | **0.9887** | **100%** |

### κ* Benchmark (R=1.1167)
| Term | Value | Contribution to Total |
|------|-------|----------------------|
| I₁ | 0.4897 | 120.3% |
| I₂ | 0.3406 | 83.7% |
| I₃ | -0.2117 | -52.0% |
| I₄ | -0.2117 | -52.0% |
| **Total** | **0.4070** | **100%** |

### Ratio Analysis: κ / κ*
| Component | Ratio | Distance from Target 1.10 |
|-----------|-------|--------------------------|
| I₂ (base) | 2.67 | +1.57 (WORST) |
| I₁ (mixed deriv) | 2.39 | +1.29 (BEST) |
| I₃+I₄ (single deriv) | 2.57 | +1.47 |
| **All derivatives** | **1.20** | **+0.10** |
| **Total** | **2.43** | **+1.33** |

---

## Key Observations

### 1. All Terms Are Scaled Too High
Every individual term has a ratio between 2.39 and 2.67. This suggests a **global scaling issue**, not a problem specific to derivatives.

### 2. Derivatives Actually Help
If we only had I₂: ratio = 2.67
Adding I₁: ratio = 2.50
Adding I₃+I₄: ratio = 2.43

**Derivatives bring the ratio DOWN** from 2.67 to 2.43.

### 3. The Derivative Sum Has Better Ratio
The sum of all derivatives (I₁+I₃+I₄) has ratio = **1.20**, much closer to target than any individual term!

This is because:
- I₁ is positive and large
- I₃+I₄ are negative and partially cancel I₁
- The **cancellation** is nearly the same in both benchmarks

### 4. Contradicts Handoff Statement
Handoff said: "Derivative terms make ratio WORSE (1.92 vs 1.71)"

For the (2,2) pair specifically:
- I₂ alone: 2.67
- With derivatives: 2.43

**Derivatives make the ratio BETTER** (by 0.24 or ~9%).

This contradiction suggests either:
1. The handoff was referring to a different pair or aggregate
2. There's a sign convention issue in how derivatives are summed
3. The "1.92 vs 1.71" refers to a different diagnostic

---

## Mechanistic Interpretation

### Why Are All Terms ~2.4x?

The ratio 2.4 is suspiciously close to:
- **R ratio:** R_κ/R_κ* = 1.3036/1.1167 = 1.17 (not quite)
- **exp(ΔR):** exp(1.3036-1.1167) = exp(0.187) = 1.21 (closer!)
- **R² ratio:** (1.3036/1.1167)² = 1.36 (not quite)

But wait, let's check **exp(2ΔR)**:
exp(2×0.187) = exp(0.374) = **1.45** (still not 2.4)

Let's check the **integral structure** more carefully:
- I₂ contains e^{2Rt} terms
- At t=1: e^{2R_κ} / e^{2R_κ*} = exp(2×1.3036) / exp(2×1.1167) = exp(2.607)/exp(2.233) = 13.6 / 9.3 = **1.46**
- At t=0.5: similar ratio

So the exponential ratio alone doesn't explain 2.4x either.

### Polynomial Contribution

From Track 3 results, we know:
- κ* P₂ is degree 2
- κ P₂ is degree 3
- ∫P₂² du depends on polynomial structure

The 2.4x factor likely comes from:
**Polynomial magnitude difference × Exponential difference × Q difference**

---

## Implications for Root Cause

### What We Can Rule Out
1. **Derivative extraction is wrong:** All terms scale similarly
2. **Only I₁ is wrong:** I₂ is actually worse
3. **Sign convention:** Wouldn't affect ratios, only absolute values
4. **I₅ missing:** I₅ is lower-order and forbidden in main mode

### What Remains Plausible
1. **Missing R-dependent normalization:** All terms scale by ~R^α
2. **Polynomial normalization:** PRZZ may normalize by ∫P² or similar
3. **θ-dependent factors:** Could be buried in PRZZ formulas
4. **Different integral bounds:** We use [0,1], maybe PRZZ uses something else
5. **κ* polynomial transcription error:** But all terms scale the same way, so seems unlikely

### Most Likely Hypothesis
**Missing global factor that depends on R and/or polynomial structure.**

Evidence:
- ALL terms scale by ~2.4
- Derivatives don't make it worse (they help!)
- The ratio is too uniform across terms to be random errors
- Must be systematic in the integral setup or normalization

---

## Recommended Next Steps

### Priority 1: Search for Normalization
Search PRZZ TeX for:
- Factors involving R in denominators
- Polynomial normalization (e.g., 1/∫P₂²)
- θ-dependent prefactors we might have missed
- Any mention of "normalization" or "standardization"

### Priority 2: Check Integral Bounds
Verify our [0,1]×[0,1] bounds match PRZZ exactly:
- Check if PRZZ uses [0,1] or [-1,1] or something else
- Verify the change of variables from PRZZ's (u,t) to our coordinates

### Priority 3: Polynomial Magnitude Test
Compute ∫P₂² du for both benchmarks:
- κ: ∫P₂² du = ?
- κ*: ∫P₂² du = ?
- Check if ratio explains the 2.4x factor

### Priority 4: Compare Against PRZZ Section 7
If PRZZ gives numerical values for any (2,2) sub-integrals, check against our oracle.

---

## Data for Reference

### Quadrature Settings
- n_quad = 80 (high quality)
- Gauss-Legendre on [0,1]

### Raw Values
```
κ benchmark:
  I₁ = 1.168588
  I₂ = 0.908817
  I₃ = -0.544354
  I₄ = -0.544354
  Total = 0.988698

κ* benchmark:
  I₁ = 0.489706
  I₂ = 0.340567
  I₃ = -0.211652
  I₄ = -0.211652
  Total = 0.406969
```

### Ratios
```
I₁: 2.3863
I₂: 2.6685
I₃: 2.5719
I₄: 2.5719
Total: 2.4294
```

---

## Conclusion

The problem is **NOT** that derivatives make the ratio worse. The problem is that **all integrals are ~2.4x too large for the κ benchmark relative to κ***, with I₂ being the worst offender.

This points to a systematic normalization or scaling issue that affects all terms uniformly, with derivatives actually helping to bring the ratio closer to the target through their different R-dependence and sign structure.

**Action Required:** Find the missing R-dependent or polynomial-dependent normalization factor in PRZZ's formulas.
