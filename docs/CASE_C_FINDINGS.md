# Case C Investigation Findings

**Date:** 2025-12-16
**Status:** Key insight reached - multiplicative correction is wrong approach

## Summary

Investigated the missing Case C auxiliary integral (PRZZ TeX 2360-2362, 2371-2374, 2382-2384).
Found that a **simple multiplicative correction does not work**.

## Key Findings

### 1. Case C Structure Confirmed

All K=3 pairs correctly classified:

| Pair | Cases | a-integrals | omega values |
|------|-------|-------------|--------------|
| (1,1) | B×B | 0 | 0, 0 |
| (1,2) | B×C | 1 | 0, 1 |
| (1,3) | B×C | 1 | 0, 2 |
| (2,2) | C×C | 2 | 1, 1 |
| (2,3) | C×C | 2 | 1, 2 |
| (3,3) | C×C | 2 | 2, 2 |

### 2. Exact Case C Kernel Implemented

Implemented the PRZZ kernel (TeX 2371-2374):
```
K_omega(u; R) = integral_0^1 P((1-a)*u) * a^{omega-1} * exp(R*theta*u*a) da
```

Test results:
- Vectorized and loop implementations match
- For constant P=1, R=0: K_omega1 = 1.0, K_omega2 = 0.5 (correct)

### 3. Correction Ratios (polynomial product only)

| Pair | Cases | R=1.3036 | R=1.1167 | R-sensitivity |
|------|-------|----------|----------|---------------|
| (1,1) | B×B | 1.000 | 1.000 | 0% |
| (1,2) | B×C | 0.461 | 0.447 | +3.14% |
| (1,3) | B×C | -0.958 | -0.904 | +6.00% |
| (2,2) | C×C | 0.221 | 0.207 | +6.64% |
| (2,3) | C×C | -0.716 | -0.653 | +9.74% |
| (3,3) | C×C | 0.058 | 0.051 | +12.93% |

### 4. Why Multiplicative Correction Fails

Applying correction ratios to raw pair values:
- c_raw = 1.950 (8.8% below target)
- c_corrected = 0.422 (80% below target!)

**Problem:** Correction ratios are < 1 because P((1-a)*u) evaluates P at smaller arguments,
reducing the integral. But our raw c is already BELOW target, so we need something that
INCREASES c, not decreases it.

### 5. Root Cause

The multiplicative approach is fundamentally wrong because:

1. Case C is **part of the integrand definition**, not a post-hoc correction
2. The kernel K_omega(u; R) replaces P(u) in the full integrand, changing how it interacts with:
   - Q factors
   - Exponential factors
   - Other polynomial factors
3. Simple multiplication doesn't capture the full structural change

### 6. Implications

**Our raw computation (c = 1.95) is not "PRZZ without Case C"** - it's something different.
The PRZZ structure defines the mean square integral with Case C built in from the start.

Two possibilities:
1. Our raw computation has implicit structure that partially accounts for some effects
2. There are multiple missing pieces, some increasing c and some decreasing it

## What This Tells Us

1. **Can't fix with multiplicative factors**: Simple scaling doesn't capture the physics
2. **Need structural change**: Properly implementing Case C requires modifying how
   polynomials enter the integrand, not post-hoc corrections
3. **R-sensitivity improved**: Despite wrong absolute value, R-sensitivity of the
   corrected values is closer to expected (17% vs 10% target, better than 19% raw)

## Recommended Next Steps

Per GPT guidance:

1. **Build Q-operator oracle** (TeX 1514-1517) - This validates the substitution step
   that converts Q(d/dalpha) into Q(affine(t,u,x,y)) in the integrand

2. **Re-examine integrand assembly** - Understand what our current "raw" computation
   actually represents vs PRZZ's definition

3. **Consider mirror combination** (TeX 1502-1511) - May need to verify analytic
   combination is done correctly

## Files Created

- `src/case_c_integral.py` - Simple exponential model (wrong but diagnostic)
- `src/case_c_exact.py` - Exact PRZZ kernel implementation
- `src/test_case_c_correction.py` - Benchmark comparison
- `src/compute_corrected_c.py` - Multiplicative correction (fails)
- `src/compute_c_exact_case_c.py` - Full computation with exact ratios (fails)
- `docs/CASE_C_FINDINGS.md` - This document

## Update 2025-12-16: R-Dependent Gap Analysis

### 7. θ/6 Hypothesis Test Results

Tested whether (1 + θ/6) factor works at both PRZZ benchmarks:

| Benchmark | R | (1+θ/6) Gap | Factor Needed |
|-----------|------|-------------|---------------|
| 1 | 1.3036 | **-0.08%** ✓ | 1.0961 |
| 2 | 1.1167 | -7.18% ✗ | 1.1799 |

**Conclusion:** (1+θ/6) only works at R=1.3036. The gap is R-dependent.

### 8. R-Sensitivity Analysis

Our computation is **1.82x MORE R-sensitive** than PRZZ target:

- Target c change (R₁→R₂): +10.29%
- Our c change (R₁→R₂): +18.73%
- Excess R-sensitivity: **+8.44%**

### 9. Per-Pair R-Sensitivity Attribution

| Pair | Case | Contribution to Excess |
|------|------|------------------------|
| (2,2) | C×C | **+45.4%** |
| (1,2) | B×C | **+33.2%** |
| (2,3) | C×C | +13.6% |
| (1,1) | B×B | +8.8% |
| (3,3) | C×C | +0.5% |
| (1,3) | B×C | -1.5% (helps) |

**Key finding:** 78.6% of excess R-sensitivity comes from pairs involving P₂ (first Case C polynomial).

### 10. Interpretation

1. **Wrong R-dependence**: Our integrals respond too strongly to R changes
2. **P₂ is the main issue**: Both (2,2) and (1,2) contribute most of the excess
3. **Case C paradox**: Multiplicative Case C makes c smaller (wrong direction), but would REDUCE R-sensitivity (right direction)
4. **Two missing pieces**: Likely need both Case C structure AND something that increases c with weak R-dependence

### 11. Refined Hypothesis

The raw computation has two problems:
1. Missing Case C a-integral structure → wrong R-dependence in P₂ terms
2. Missing positive term with weak R-dependence → explains gap direction

These two effects partially cancel in the multiplicative factor, making the gap appear R-dependent even though the root causes are structural.

## Update 2025-12-16: Case C Kernel Implementation Results

### 12. Case C Kernel vs Raw Polynomial

Implemented the exact PRZZ Case C kernel:
```
K_ω(u; R) = u^ω / (ω-1)! × ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθau) da
```

Results for polynomial RMS comparison:
| Polynomial | ω | K/P ratio (R1) | K/P ratio (R2) | K ratio R1/R2 |
|------------|---|----------------|----------------|---------------|
| P₁ | 0 | 1.000 | 1.000 | 1.000 |
| P₂ | 1 | 0.470 | 0.455 | 1.033 |
| P₃ | 2 | 0.240 | 0.226 | 1.063 |

**Key observation**: Case C kernels are SMALLER than raw polynomials and have intrinsic R-dependence.

### 13. Case C I₂ Term Analysis

Applied Case C kernel to I₂ terms:

| Pair | ω | Raw R-sens | Case C R-sens |
|------|---|------------|---------------|
| (1,1) | 0,0 | 12.90% | 12.90% |
| (1,2) | 0,1 | 12.90% | 16.46% |
| (1,3) | 0,2 | 12.90% | 19.68% |
| (2,2) | 1,1 | 12.90% | 20.40% |
| (2,3) | 1,2 | 12.90% | 23.90% |
| (3,3) | 2,2 | 12.90% | 27.50% |
| **Total** | | **12.90%** | **14.81%** |

**Critical finding**: Case C makes I₂ R-sensitivity WORSE (12.9% → 14.8%).

### 14. Paradox: Case C Alone Cannot Explain the Gap

| Observation | Implication |
|-------------|-------------|
| Raw c = 1.95 < target c = 2.14 | We need c to INCREASE |
| Case C kernel K < P | Case C makes c SMALLER |
| Case C I₂ totals: 0.71 vs 1.19 | I₂ reduced by 40% |
| R-sensitivity increases | Wrong direction for I₂ |

**Conclusion**: Naive Case C implementation makes things WORSE, not better.

### 15. Hope: Derivative Terms May Behave Differently

Earlier finding: scaling I₃/I₄ by Case C ratio (~0.65) INCREASED c.

Why? Because I₃_12 is NEGATIVE (-0.31). Reducing its magnitude increases total c.

Case C derivative structure:
```
d/dy K(y+u; R)|_{y=0} = (u^ω/(ω-1)!) × ∫₀¹ [(1-a)P'((1-a)u) + Rθa·P((1-a)u)] × a^{ω-1} × exp(Rθua) da
```

The extra Rθa·P(...) term adds positive contribution that may change sign patterns.

### 16. Implementation Specification

To properly implement Case C, need to:

1. **Replace P(argument) with K(argument; R)** for ω > 0 polynomials
2. **Use Case C derivative** when taking derivatives:
   - Not just P'(u), but the full d/darg K(arg; R)
3. **Modify series expansion** to handle K's argument-dependent R-structure
4. **Preserve Q and exp factors** unchanged

This requires DSL modification, not just post-hoc correction.

## Key PRZZ TeX References

| Lines | Content |
|-------|---------|
| 2301-2310 | omega definition, x -> x log N rescaling |
| 2350-2355 | Υ_C derivation with auxiliary a-integral |
| 2358-2361 | P((1-a)u) appearance |
| 2364-2368 | Main object built from sum over n |
| 2370-2375 | F_d definition for ω = -1, 0 |
| 2379-2383 | F_d definition for ω > 0 (CRITICAL) |
| 1514-1517 | Q-operator substitution (validated correct) |
| 2596-2598 | kappa* benchmark at R=1.1167 |
