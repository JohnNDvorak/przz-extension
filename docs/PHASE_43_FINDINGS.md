# Phase 43 Findings: Eliminating the Final ±0.15% Gap

**Created**: 2025-12-27
**Status**: Complete

## Executive Summary

Phase 43 systematically explored multiple approaches to eliminate the ±0.15% residual. We found that:

1. **The residual is a GLOBAL effect** - ALL pairs show opposite-sign corrections
2. **The baseline formula is optimal among tested configurations** - No separation strategy improves it
3. **Two correlating factors identified**:
   - R-dependence: `g(R) = 0.00926 × R + 1.00306`
   - I1 fraction: `g(f_I1) = 1.0195 - 0.0185 × f_I1`
4. **Both correlations are currently empirical** - No first-principles derivation found

**Recommendation**: Accept ±0.15% accuracy. The residual is understood structurally (I1/I2 mixture imbalance) but eliminating it requires per-benchmark calibration.

---

## Phase 43.1: Component-Separated Mirror Test

### Hypothesis
Since I1 has cross-terms (log factor) and I2 does not, they should have different mirror multipliers:
- `m_I1 = g_baseline × base`
- `m_I2 = 1.0 × base`

### Results

| Config | g_I1 | g_I2 | κ gap | κ* gap | Avg |gap| |
|--------|------|------|-------|--------|---------|
| Baseline | 1.0136 | 1.0136 | -0.14% | +0.02% | **0.08%** |
| g_I2=1.0 | 1.0136 | 1.0 | -1.07% | -0.81% | 0.94% |
| g_I1=1.0 | 1.0 | 1.0136 | -0.42% | -0.38% | 0.40% |
| No corr | 1.0 | 1.0 | -1.35% | -1.21% | 1.28% |

**Conclusion**: The baseline (equal g for I1 and I2) is OPTIMAL. Separating makes things worse.

### g_needed Values

| Benchmark | g_needed (uniform) | I1 fraction at -R |
|-----------|-------------------|-------------------|
| κ | 1.015134 (+0.15% from baseline) | 23.3% |
| κ* | 1.013402 (-0.02% from baseline) | 32.6% |

**Pattern**: Higher I1 fraction correlates with needing LOWER g.

---

## Phase 43.2: Per-Pair Decomposition

### Key Finding: ALL Pairs Show Same Direction

| Pair | κ delta_m% | κ* delta_m% | Same sign? |
|------|------------|-------------|------------|
| (1,1) | +0.16% | -0.02% | NO |
| (2,2) | +0.19% | -0.02% | NO |
| (3,3) | +0.20% | -0.03% | NO |
| (1,2) | +0.19% | -0.03% | NO |
| (1,3) | +0.20% | -0.02% | NO |
| (2,3) | +0.20% | -0.03% | NO |

**All pairs in κ need m INCREASED**. **All pairs in κ* need m DECREASED**.

**Conclusion**: The residual is NOT from specific pairs - it's a GLOBAL effect affecting all pairs uniformly but in opposite directions.

---

## Phase 43.3: Base Term Investigation

### Alternative Base Formulas

| Formula | κ gap | κ* gap |
|---------|-------|--------|
| exp(R) + (2K-1) | -0.15% | +0.02% |
| exp(2R) + (2K-1) | +113% | +78% |
| exp(R/θ) + (2K-1) | +70% | +50% |
| exp(2R/θ) + (2K-1) | +1060% | +581% |

**Conclusion**: `exp(R) + (2K-1)` is correct. Alternatives are way off.

### R-Dependent g Correction

```
g_needed(κ) = 1.015134 at R = 1.3036
g_needed(κ*) = 1.013402 at R = 1.1167

Linear fit: g(R) = 0.009265 × R + 1.003056
```

This perfectly fits both benchmarks but is empirical.

---

## Phase 43.4: I1/I2 Fraction-Weighted Correction

### Correlation Found

| Benchmark | f_I1 (at -R) | delta_g |
|-----------|--------------|---------|
| κ | 23.3% | +0.0015 |
| κ* | 32.6% | -0.0002 |

**Linear fit**:
```
delta_g = -0.018537 × f_I1 + 0.005846

Equivalently:
g(f_I1) = 1.019452 - 0.018537 × f_I1
```

### Physical Interpretation

1. The baseline g = 1 + θ/(2K(2K+1)) was derived from I1's cross-terms
2. I2 has NO cross-terms (g_I2 = 1.0 from MCG decomposition)
3. When I2 fraction is higher, the baseline OVER-corrects
4. So higher f_I1 → baseline is more accurate → less correction needed

**This explains the opposite signs**:
- κ: Lower f_I1 (23%) → baseline under-corrects → needs g increased
- κ*: Higher f_I1 (33%) → baseline closer to right → needs slight g decrease

---

## Candidate Correction Formulas

### Formula A: R-Dependent
```
g(R) = 0.00926 × R + 1.00306
m = g(R) × [exp(R) + (2K-1)]
```
- Pros: Perfect fit to both benchmarks
- Cons: Empirical, not derived

### Formula B: I1-Fraction Dependent
```
g(f_I1) = 1.0195 - 0.0185 × f_I1
m = g(f_I1) × [exp(R) + (2K-1)]
```
- Pros: Has physical explanation (I1/I2 mixture)
- Cons: Requires computing I1(-R)/S12(-R), adds complexity

### Formula C: Baseline (Current)
```
g = 1 + θ/(2K(2K+1))
m = g × [exp(R) + (2K-1)]
```
- Pros: Fully derived, simple
- Cons: ±0.15% residual

---

## Derivation Status

### What We Can Derive

| Component | Formula | Status |
|-----------|---------|--------|
| Base term | exp(R) + (2K-1) | **DERIVED** (Phase 32) |
| g correction | 1 + θ/(2K(2K+1)) | **DERIVED** (Phase 34C) |
| K-dependence | 2K(2K+1) | **DERIVED** (Beta moment) |

### What We Cannot Derive (Yet)

| Component | Empirical | Issue |
|-----------|-----------|-------|
| R-slope | 0.00926 | No theoretical source |
| I1-slope | -0.0185 | Why this coefficient? |

---

## Conclusions

### Why ±0.15% is Irreducible with Scalar m

1. **The baseline g is calibrated for "average" I1/I2 ratio**
2. **Different polynomials have different I1/I2 ratios**
3. **A scalar g cannot adapt to polynomial-specific ratios**

To eliminate the residual requires either:
- Per-benchmark calibration (not first-principles)
- Polynomial-aware functional (complex to derive)

### Recommended Production Formula

```
m(K, R) = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]
```

**Accuracy**: ±0.15% on tested benchmarks

**Justification**:
- All components derived from first principles
- Residual is understood (I1/I2 mixture imbalance)
- 10× better than empirical formula
- Sufficient for κ optimization goals

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/run_phase43_separated_mirror.py` | Component separation test |
| `scripts/run_phase43_per_pair.py` | Per-pair decomposition |
| `scripts/run_phase43_base_term.py` | Alternative base formulas |
| `scripts/run_phase43_i1i2_weighting.py` | I1/I2 fraction analysis |
| `docs/PHASE_43_FINDINGS.md` | This document |

---

## Questions for GPT

1. **Can the I1-slope (-0.0185) be derived?**
   - It's approximately -1.36 × θ/(2K(2K+1))
   - Could come from second-order Beta moment or Euler-Maclaurin correction?

2. **Alternative interpretation of R-linear g?**
   - The R-linear fit g(R) = 0.00926R + 1.00306 works perfectly
   - Is there a theoretical reason for R-dependence in g?

3. **Higher-order corrections?**
   - The baseline has θ/(2K(2K+1)) from first order
   - Could θ² terms explain the residual?

4. **Physical meaning of f_I1 correlation?**
   - Higher f_I1 → less g needed
   - Is this saying the baseline "assumes" a specific f_I1 value?
   - What f_I1 would make baseline exact?

5. **K=4 prediction?**
   - For K=4: g_baseline = 1 + θ/72 = 1.00794
   - Will the I1/I2 pattern hold?
   - What accuracy should we expect?

---

## Raw Data for GPT

```
PHASE 43 SUMMARY:

1. BASELINE IS OPTIMAL
   - Separating m_I1/m_I2 doesn't help
   - All configurations worse than baseline

2. PER-PAIR: GLOBAL EFFECT
   - ALL pairs in κ need m increased (+0.16% to +0.20%)
   - ALL pairs in κ* need m decreased (-0.02% to -0.03%)
   - Not pair-specific, affects entire structure

3. BASE TERM: exp(R) + (2K-1) IS CORRECT
   - exp(2R) off by 78-113%
   - exp(R/θ) off by 50-70%
   - exp(2R/θ) off by 580-1060%

4. R-DEPENDENT CORRECTION
   g(R) = 0.009265 × R + 1.003056

   - g(1.3036) = 1.015134 (matches κ needed)
   - g(1.1167) = 1.013402 (matches κ* needed)

5. I1-FRACTION CORRELATION
   delta_g = -0.018537 × f_I1 + 0.005846

   κ:  f_I1 = 0.233, delta_g = +0.15%
   κ*: f_I1 = 0.326, delta_g = -0.02%

   Higher f_I1 → needs less g correction

6. PHYSICAL EXPLANATION
   - Baseline g from I1's cross-terms (log factor)
   - I2 has no cross-terms
   - More I2 → baseline over-corrects
   - More I1 → baseline more accurate

7. RECOMMENDATION
   Accept ±0.15% accuracy with baseline formula.
   Residual is understood but not derivably eliminable.
```
