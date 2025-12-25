# Polynomial Degree Scaling Analysis Report

## Executive Summary

**Date:** 2025-12-17
**Analysis:** How polynomial degree affects c computation in PRZZ
**Key Finding:** The c ratio mystery is NOT explained by simple degree-dependent normalization

## The Mystery

- **κ benchmark:** P₂,P₃ degree 3, Q degree 5, R=1.3036, expected c≈2.137
- **κ* benchmark:** P₂,P₃ degree 2, Q degree 1, R=1.1167, expected c≈1.939
- **Expected ratio:** c_κ/c_κ* ≈ 1.10
- **Observed ratio (I₂-like integrals):** ≈2.09

Despite κ having **higher-degree polynomials**, it should give **similar c** to κ*.

## Test Results

### Test 1: Synthetic Degree Scaling

Created synthetic polynomials of degrees 1-5 with fixed Q:

| Degree | ∫P²du | ∫∫P²Q²e^{2Rt} | Ratio to deg=1 |
|--------|-------|---------------|----------------|
| 1 | 0.00034 | 0.14388 | 1.000 |
| 2 | 0.00014 | 0.08658 | 0.602 |
| 3 | 0.00037 | 0.19555 | 1.359 |
| 4 | 0.00033 | 0.24605 | 1.710 |
| 5 | 0.00046 | 0.34165 | 2.375 |

**Conclusion:** Integral magnitude does NOT scale linearly with degree. Coefficient structure matters enormously.

### Test 2: Polynomial Degrees

**κ polynomials:**
- P₁: degree 5
- P₂: degree 3
- P₃: degree 3
- Q: degree 5

**κ* polynomials:**
- P₁: degree 5 (same)
- P₂: degree 2 (lower)
- P₃: degree 2 (lower)
- Q: degree 1 (much lower)

### Test 3: Actual Polynomial Integral Ratios

Computing ∫∫P_{ℓ₁}(u)P_{ℓ₂}(u)Q²(u)e^{2Rt} du dt for each pair:

| Pair | κ integral | κ* integral | Ratio κ/κ* | Expected ~1.10 |
|------|-----------|-------------|------------|----------------|
| (1,1) | 0.091742 | 0.077368 | 1.186 | ✓ |
| (1,2) | 0.162497 | 0.092292 | 1.761 | ✗ |
| (1,3) | 0.019169 | -0.004172 | -4.595 | ✗ |
| (2,2) | 0.294063 | 0.113660 | **2.587** | ✗ |
| (2,3) | 0.038852 | -0.004553 | -8.533 | ✗ |
| (3,3) | 0.007893 | 0.000280 | **28.184** | ✗ |

**Critical Finding:** The ratio **varies wildly** by pair:
- (1,1): 1.186 - close to expected
- (2,2): 2.587 - too high
- (3,3): 28.184 - **WAY too high**
- (1,3), (2,3): **negative** (κ* has negative cross terms!)

This is NOT a simple normalization issue.

### Test 4: Coefficient Magnitude Analysis

L2 norm ratios (κ/κ*):
- P₁: 0.82 (κ has smaller norm)
- P₂: **1.83** (κ has larger norm)
- P₃: **5.39** (κ has MUCH larger norm)
- Q: **2.31** (κ has larger norm)

**Key insight:** κ P₃ has 5.4× larger L2 norm than κ* P₃. This explains the (3,3) pair explosion.

### Test 5: R-Dependent Scaling

Using **same κ polynomials** at both R values:

| Pair | I(R=1.3036) | I(R=1.1167) | Ratio | e^{2ΔR} prediction |
|------|-------------|-------------|-------|---------------------|
| (1,1) | 0.091742 | 0.071036 | 1.291 | 1.213 |
| (2,2) | 0.294063 | 0.227692 | 1.291 | 1.213 |
| (3,3) | 0.007893 | 0.006112 | 1.291 | 1.213 |

**Conclusion:** R-dependence is **consistent across pairs** and close to predicted e^{2ΔR} scaling. The R-effect is well-understood.

## Normalization Hypothesis Testing

### Standard Normalizations Tested

| Scheme | c_κ | c_κ* | Ratio | Expected | Match Quality |
|--------|-----|------|-------|----------|---------------|
| None (raw) | 0.614 | 0.275 | 2.235 | 1.102 | 1.132 |
| 1/(ℓ₁!·ℓ₂!) | 0.253 | 0.151 | 1.678 | 1.102 | 0.576 |
| 1/(ℓ₁·ℓ₂) | 0.260 | 0.150 | 1.737 | 1.102 | 0.635 |
| 1/(ℓ₁²·ℓ₂²) | 0.154 | 0.107 | 1.440 | 1.102 | **0.338** |
| 1/(ℓ₁³·ℓ₂³) | 0.118 | 0.091 | 1.299 | 1.102 | **0.197** |

**Best match:** 1/(ℓ₁³·ℓ₂³) gives ratio 1.299 (still 18% off target).

**Conclusion:** No standard normalization brings us close to 1.10.

### Per-Pair Normalization Analysis

To achieve ratio 1.10 for each pair independently, we'd need:

| Pair | Raw Ratio | Needed N_κ/N_κ* | ℓ₁·ℓ₂ | deg_ratio |
|------|-----------|-----------------|-------|-----------|
| (1,1) | 1.186 | 1.076 | 1 | 1.000 |
| (1,2) | 1.761 | 1.598 | 2 | 1.143 |
| (1,3) | -4.595 | -4.169 | 3 | 1.143 |
| (2,2) | 2.587 | **2.347** | 4 | 1.500 |
| (2,3) | -8.533 | -7.742 | 6 | 1.500 |
| (3,3) | 28.184 | **25.572** | 9 | 1.500 |

**Pattern:** Needed normalization correlates with both ℓ₁·ℓ₂ AND degree ratio, but the relationship is complex.

### Empirical Correction Factor

Current raw totals:
- c_κ (raw): 0.614 → Expected: 2.137 → **Factor needed: 3.48**
- c_κ* (raw): 0.275 → Expected: 1.939 → **Factor needed: 7.05**

**CRITICAL FINDING:** The needed correction factors are **DIFFERENT** (3.48 vs 7.05).

This means:
1. **No simple global rescaling exists**
2. The formula interpretation is fundamentally different for different polynomial degrees
3. We're missing major terms or using the wrong integral structure

## Root Cause Analysis

### What We Know

1. **Polynomial structure matters enormously**
   - κ P₃ has 5.4× larger coefficients than κ* P₃
   - This causes the (3,3) integral to explode (ratio 28×)

2. **R-dependence is well-behaved**
   - Consistent 1.29× ratio across pairs
   - Matches e^{2ΔR} prediction

3. **No simple normalization works**
   - Tested ℓ₁!·ℓ₂!, powers of ℓ, degrees, combinations
   - Best attempt (1/(ℓ₁³·ℓ₂³)) still 18% off

4. **Negative cross terms in κ***
   - (1,3) and (2,3) pairs are negative
   - This is mathematically valid (polynomial cross-product can be negative)
   - But suggests very different polynomial structure

### Hypotheses for the Discrepancy

#### Hypothesis 1: Missing I₁, I₃, I₄ Derivative Terms
**Status:** LIKELY

The I₂-only integrals we tested (∫∫P×P×Q²×exp) are just **one piece** of the full c formula.

The full formula includes:
- I₂: ∫∫P×P×Q²×exp (what we tested)
- I₁: derivative terms d²/dxdy
- I₃, I₄: first derivative terms
- I₅: arithmetic corrections (known to be lower-order)

**Evidence:**
- Our raw total c_κ = 0.614 vs expected 2.137
- Track 3 I₂ baseline showed I₂-only gives c=1.96 (closer!)
- Derivatives contribute significantly and might have different degree-dependence

**Next step:** Re-run this analysis using the **full DSL evaluator** with all I₁,I₃,I₄ terms.

#### Hypothesis 2: Degree-Dependent Integral Formula
**Status:** POSSIBLE

Maybe PRZZ Section 7 formula has different structure for different polynomial degrees:

```
c = Σ_{ℓ₁,ℓ₂} A₁^{ℓ₁-1} × A₂^{ℓ₂-1} × I_{ℓ₁,ℓ₂}(deg(P_{ℓ₁}), deg(P_{ℓ₂}), deg(Q))
```

Where the integral definition itself changes based on degrees.

**Evidence:** Per-pair needed normalizations correlate with both ℓ and degrees.

**Next step:** Re-read PRZZ Section 7 for degree-dependent formulas.

#### Hypothesis 3: Optimization Quality Difference
**Status:** UNLIKELY

Maybe the κ* polynomials were optimized with a different objective function?

**Evidence Against:**
- Both κ and κ* are published results from the same paper
- Authors claim same methodology
- Polynomial constraints (P(0)=0, Q(0)=1) are satisfied for both

#### Hypothesis 4: We're Computing the Wrong Integral
**Status:** VERY LIKELY

The integral we tested was:
```
∫₀¹∫₀¹ P_{ℓ₁}(u) P_{ℓ₂}(u) Q²(u) e^{2Rt} du dt
```

But the actual PRZZ formula might be:
```
∫₀¹∫₀¹ P_{ℓ₁}(u+X) P_{ℓ₂}(u+Y) Q²(u) e^{2Rt} g(X,Y,...) dX dY
```

Where X,Y are formal variables that get differentiated away.

**Evidence:**
- Track 3 showed I₂-only baseline c_κ = 1.96 vs our 0.61
- Factor of 3.2× difference suggests structural formula error
- The DSL uses P(u+X) not P(u)

**Next step:** Verify the integrand formula in our test matches PRZZ exactly.

## Conclusions

### Primary Findings

1. **Polynomial degree scaling is highly non-linear**
   - (3,3) pair ratio is 28× vs expected 1.10
   - Due to coefficient magnitude, not just degree

2. **No simple normalization factor exists**
   - Tested factorial, power, degree-based schemes
   - Best attempt (1/(ℓ₁³·ℓ₂³)) still 18% off
   - Needed correction differs by pair (1.08× to 25×)

3. **The test integrals are incomplete**
   - We only tested I₂-like terms
   - Missing I₁, I₃, I₄ derivative contributions
   - These likely have different degree-dependence

4. **Different global factors needed (3.48 vs 7.05)**
   - Suggests fundamental formula interpretation error
   - Not just missing a constant

### Recommendations

#### Immediate Actions

1. **Re-run with full DSL evaluator**
   - Use `evaluate.py` with complete terms including I₁,I₃,I₄
   - Compare per-pair breakdown for κ vs κ*
   - Check if derivative terms have different degree-scaling

2. **Verify integrand formula**
   - Confirm we're using P(u+X) vs P(u) correctly
   - Check if Q² should be Q(u+X)Q(u+Y) vs Q²(u)
   - Review PRZZ Section 6-7 integral definitions

3. **Test with swapped polynomials**
   - Use κ polynomial **degrees** with κ* **coefficients**
   - Isolate degree effect from coefficient magnitude effect

#### Research Questions

1. **Does PRZZ normalize by polynomial degree?**
   - Search paper for deg(P), deg(Q) factors
   - Check if ω (mollifier degree) appears in normalization

2. **Are the κ and κ* formulas actually the same?**
   - Maybe simple zeros (κ*) use different formula?
   - Check PRZZ Section 8 for distinctions

3. **Is the I₅ arithmetic correction degree-dependent?**
   - Current I₅ formula: `-S(0) × θ²/12 × I₂_total`
   - Maybe should be `-S(0) × θ²/12 × I₂_total / f(deg)`?

## Appendix: Code Artifacts

Generated analysis scripts:
- `analyze_polynomial_degree_scaling.py` - Main degree scaling tests
- `test_normalization_correction.py` - Normalization scheme testing
- `test_advanced_normalization.py` - Per-pair analysis and empirical factors

All scripts are in `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/`

## Next Session Guidance

**DO NOT:**
- Try more normalization schemes (we've exhausted the simple ones)
- Assume the I₂-only integrals are representative
- Apply empirical correction factors without understanding

**DO:**
1. Use full DSL evaluator to get complete c breakdown
2. Verify the integrand formula matches PRZZ
3. Search PRZZ paper for "degree", "ω", "normalization"
4. Compare I₁,I₃,I₄ degree-dependence to I₂

**Key question to answer:**
> "When we use the FULL evaluator (not just I₂), do we still see 2.09× ratio, or does including derivatives fix it?"
