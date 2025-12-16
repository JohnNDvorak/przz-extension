# PRZZ Gap Analysis Summary

**Date:** 2025-12-16
**Status:** Root cause identified - structural understanding achieved

## Executive Summary

We have identified the root cause of the ~8.8% gap between our computed c (~1.95) and PRZZ target (2.137):

1. **The gap is R-dependent**, ruling out simple global factors
2. **78% of excess R-sensitivity comes from P₂ pairs** ((2,2), (1,2))
3. **I3/I4 derivative terms have extreme R-sensitivity** (44% vs 10% target)
4. **Naive Case C correction makes things worse** (reduces c, increases R-sensitivity)
5. **Partial correction to derivative terms helps** (increases c, reduces R-sensitivity)

## Key Findings

### 1. R-Dependent Gap

The gap cannot be fixed with a single multiplicative factor:

| Benchmark | R | Factor Needed | (1+θ/6) Gap |
|-----------|------|---------------|-------------|
| 1 | 1.3036 | 1.0961 | -0.08% |
| 2 | 1.1167 | 1.1799 | -7.18% |

The 7.65% difference between factors proves R-dependence.

### 2. R-Sensitivity Analysis

Our computation is **1.82x MORE R-sensitive** than PRZZ target:

| Metric | Value |
|--------|-------|
| Target c change (R₁→R₂) | +10.29% |
| Our c change (R₁→R₂) | +18.73% |
| Excess R-sensitivity | +8.44% |

### 3. Per-Pair Attribution

| Pair | Case | Contribution to Excess |
|------|------|------------------------|
| (2,2) | C×C | **+45.4%** |
| (1,2) | B×C | **+33.2%** |
| (2,3) | C×C | +13.6% |
| (1,1) | B×B | +8.8% |
| (3,3) | C×C | +0.5% |
| (1,3) | B×C | -1.5% (helps) |

**78.6% of excess R-sensitivity comes from P₂ pairs.**

### 4. Term-Level Attribution

The derivative terms (I3, I4) are the main culprits:

| Term | (2,2) R-sens | (1,2) R-sens | Target |
|------|-------------|--------------|--------|
| I3 | **+44%** | +10% | 10% |
| I4 | **+44%** | **+35%** | 10% |
| I1 | +15% | +7% | 10% |
| I2 | +13% | +13% | 10% |

I3 and I4 have **double R-dependence**:
1. From R factor in derivative coefficient (∂/∂y brings down R×θ×...)
2. From R factor in exponential (exp(R×arg))

### 5. Case C Effects

#### Full Polynomial Correction (FAILS)
- Correction ratios: 0.35-0.59 for P₂ pairs
- Result: c = 0.82 (61% below target!)
- R-sensitivity: 24% (worse!)

#### Derivative-Only Scaling (PARTIAL SUCCESS)
- Using Case C derivative ratio ~0.65 on I3/I4 for P₂ pairs:
- Result: c = 2.01 (6% below target, improved from 8.8%)
- R-sensitivity: 17.5% (improved from 18.7%)

### 6. Why Derivative Scaling Helps

The key insight: **I3_12 is negative**.

- Raw: I3_12 = -0.31
- Scaled: I3_12 × 0.65 = -0.20

Reducing the magnitude of negative terms **increases** total c, which is the right direction.

## Interpretation

### What We Know

1. **Q-operator substitution is correct** (validated by oracle)
2. **Case C structure affects derivative terms differently** than polynomial terms
3. **The gap involves multiple interacting effects** that partially cancel

### What We Don't Know

1. **Exact mechanism for proper Case C integration** into full integrand
2. **Whether mirror combination** (TeX 1511-1527) affects the balance
3. **What positive term** could increase c with weak R-dependence

### Hypotheses for Future Investigation

1. **Mirror combination not handled correctly**: PRZZ combines terms analytically before extracting constants. Our separate evaluation may miss cross-terms.

2. **Variable rescaling issue**: PRZZ uses x → x log N rescaling (TeX 2309). Our variables may be at different scale.

3. **Missing positive contribution**: Something increases c by ~10% with weak R-dependence. Could be related to log factor (θ(x+y)+1)/θ handling.

## Technical Details

### Case C Derivative Analysis

In raw computation:
```
d/dy P_2(y+u)|_{y=0} = P_2'(u)
```

In PRZZ Case C:
```
d/dy K_1(y+u; R)|_{y=0} = ∫₀¹ [(1-a)P_2'((1-a)u) + Rθa P_2((1-a)u)] exp(Rθua) da
```

Key differences:
- Derivative magnitude reduced by ~35%
- Additional R-dependence through Rθa term
- Exponential factor exp(Rθua) adds R-sensitivity

### Computed Ratios

| Polynomial | ω | Ratio(R1) | Ratio(R2) |
|------------|---|-----------|-----------|
| P₁ | 0 | 1.000 | 1.000 |
| P₂ | 1 | 0.588 | 0.571 |
| P₃ | 2 | 0.409 | 0.391 |

## Files Created During Investigation

| File | Purpose |
|------|---------|
| `src/test_theta_6_hypothesis.py` | Test global factor at both benchmarks |
| `src/raw_vs_przz_diagnostic.py` | Compare raw vs PRZZ structure |
| `src/case_c_exact.py` | Exact PRZZ Case C kernel |
| `src/compute_c_exact_case_c.py` | Full Case C computation |
| `docs/CASE_C_FINDINGS.md` | Case C investigation findings |
| `docs/GAP_ANALYSIS_SUMMARY.md` | This document |

## Key PRZZ TeX References

| Lines | Content |
|-------|---------|
| 1511-1527 | Mirror combination (may need verification) |
| 2301-2310 | ω definition, x → x log N rescaling |
| 2360-2362 | Case C polynomial rescaling |
| 2371-2374 | Full F_d definition for ω > 0 |
| 2596-2598 | κ* benchmark (R=1.1167) |

## Recommended Next Steps

1. **Verify mirror combination** (Step 5 of audit plan)
   - Check if our assembly matches PRZZ's analytic combination
   - Look for cross-terms that arise from combined expansion

2. **Investigate variable rescaling**
   - Trace PRZZ's x → x log N transformation
   - Check if our derivative extraction accounts for this

3. **Build term-by-term comparison**
   - If PRZZ per-pair values are available, compare directly
   - Identify which specific integrals differ

4. **Consider I₅ role** (carefully)
   - I₅ is error term, but could there be a related main-term contribution?
   - Check if PRZZ's numerical optimization includes any error correction

## Conclusion

The gap is not a simple bug or missing factor. It reflects a structural difference in how Case C polynomials enter the integrand. The derivative terms (I3, I4) are most affected because they have double R-dependence that the Case C structure modifies.

A complete fix requires either:
1. Re-implementing the integrand with proper Case C structure from the start
2. Understanding exactly how PRZZ's numerical code handles Case C and replicating it
3. Finding the compensating positive term that balances the Case C reduction

The two-benchmark test is crucial: any proposed fix must work at BOTH R values simultaneously.
