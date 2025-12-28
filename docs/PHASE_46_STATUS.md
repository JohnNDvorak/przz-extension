# Phase 46 Status: Derive g from Integrals (No Target Anchoring)

**Date:** 2025-12-27
**Status:** PARTIAL SUCCESS - Structural derivation validated, integral ratio approach inconclusive

---

## Goal

Derive g_I1 and g_I2 from the integral structure WITHOUT using c_target values as inputs. Replace calibrated constants with a derived functional.

---

## What Was Attempted

### Approach 1: Coefficient-Level Log Factor Split

**Method:** Extract F_xy, F_x, F_y from the series algebra and compute:
```
internal_correction = (main + cross) / main = 1 + θ × (F_x + F_y) / F_xy
```

**Result:** FAILED

| Pair | Main | Cross | Correction |
|------|------|-------|------------|
| 11 | +0.62 | +0.94 | **2.51** |
| 22 | +5.41 | +7.54 | 2.39 |
| 33 | +12.75 | +16.91 | 2.33 |
| 12 | -1.67 | -2.88 | 2.72 |
| 13 | +2.45 | +4.34 | 2.77 |
| 23 | -8.22 | -11.24 | 2.37 |

Per-pair corrections are ~2.3-2.8, NOT ~1.0136 (g_baseline).

**Why It Failed:** The Beta moment (1/(2K(2K+1))) is an "emergent property" of the full integration with (1-u)^{2K-1} weight, not a pointwise coefficient ratio. This was explicitly warned in the Phase 45 status file.

### Approach 2: Integral Ratio I1/M1

**Method:** Compute the ratio of the full I1 integral (with log factor) to M1 (main term only):
```
internal_correction = I1(computed) / M1(extracted from split)
```

**Result:** PARTIAL

| Benchmark | R | I1/M1 | g_baseline | Gap |
|-----------|------|-------|------------|-----|
| κ (Q=real) | 1.3036 | 1.047 | 1.0136 | +3.3% |
| κ* (Q=real) | 1.1167 | 1.054 | 1.0136 | +4.0% |
| κ (Q=1) | 1.3036 | 1.038 | 1.0136 | +2.4% |
| κ* (Q=1) | 1.1167 | 1.052 | 1.0136 | +3.8% |

The ratio I1/M1 ≈ 1.04-1.05 is in the right ballpark but doesn't match g_baseline exactly.

**Why It's Inconclusive:** The aggregate I1/M1 ratio differs from per-pair ratios due to sign cancellations (off-diagonal pairs have negative main terms). The semantic layer where the split is computed may not be correct.

---

## The Best Available First-Principles Formula

Based on the structural analysis (earlier in Phase 45):

```
g_I1 = 1.0                        (log factor cross-terms self-correct)
g_I2 = 1 + θ/(2K(2K+1))           (full Beta moment correction)
```

**Validation:**

| Benchmark | c_derived | c_target | Gap |
|-----------|-----------|----------|-----|
| κ | 2.1285 | 2.1375 | **-0.42%** |
| κ* | 1.9306 | 1.9380 | **-0.38%** |

This achieves **< 0.5% accuracy** without any calibrated parameters.

---

## Comparison of Approaches

| Approach | g_I1 | g_I2 | κ Gap | κ* Gap | Status |
|----------|------|------|-------|--------|--------|
| Calibrated (2-benchmark solve) | 1.0009 | 1.0195 | ~0% | ~0% | Curve-fit |
| Structural derivation | 1.0 | 1.0136 | -0.42% | -0.38% | **Best derived** |
| Integral ratio (I1/M1) | ? | ? | ? | ? | Inconclusive |
| Coefficient split | N/A | N/A | N/A | N/A | Failed |

---

## Honest Assessment

### What Is Actually Derived

1. **g_I1 = 1.0**: Justified by the log factor (1/θ + x + y) in the I1 integrand creating internal cross-terms that provide the correction.

2. **g_I2 = g_baseline = 1 + θ/(2K(2K+1))**: Justified by I2 lacking the log factor, so it needs full external correction.

3. **Accuracy: ~0.4%**: This is better than Phase 36's ±0.15% uniform formula but not as good as the calibrated ~0% formula.

### What Remains Empirical

1. **The ~0.4% residual**: The derived formula doesn't close the gap to 0%. The residual likely comes from:
   - Q polynomial differential attenuation (Q attenuates I2 ~15% more than I1)
   - Second-order corrections not captured by the structural model

2. **No closed-form M/C split**: We cannot derive g_I1 and g_I2 directly from an integral ratio that matches the calibrated values.

---

## GPT's Guidance vs Reality

| GPT Recommendation | Outcome |
|-------------------|---------|
| Task 46.0: Lock anchored mode | ✓ Already done in correction_policy.py |
| Task 46.1: Define M/C split mathematically | ✓ Defined, but doesn't give g_baseline |
| Task 46.2: Implement compute_g_components_from_integrals() | ✓ Implemented |
| Task 46.3: Q=1 gate (internal_correction = g_baseline) | ✗ FAILED (gap = 2-4%) |
| Task 46.4: Replace anchored with derived | Partial (structural derivation works) |
| Task 46.5: Validation gates | ✓ Targets used only as checks |

---

## Conclusion

**Phase 46 is PARTIALLY successful:**

1. ✓ The structural derivation (g_I1=1.0, g_I2=g_baseline) is a valid first-principles formula with ~0.4% accuracy.

2. ✗ The M/C integral split approach does NOT give exact agreement with calibrated values.

3. ✓ The anchored mode is properly locked and labeled as calibration, not derivation.

**The gap between "derived" (~0.4% accuracy) and "calibrated" (~0% accuracy) represents the limits of our current understanding.**

To truly close this gap without anchoring, we would need:
- A different semantic layer for the log factor split
- Understanding of why Q creates differential attenuation
- Possibly a non-scalar mirror operator treatment

---

## Files Created

| File | Description |
|------|-------------|
| `src/unified_s12/g_components.py` | Phase 46 g derivation module |
| `scripts/test_integral_ratio_approach.py` | Integral ratio testing |
| `docs/PHASE_46_STATUS.md` | This document |

---

## Recommended Next Steps

1. **Accept the structural derivation** as the best available first-principles formula:
   - g_I1 = 1.0
   - g_I2 = 1 + θ/(2K(2K+1))
   - Accuracy: ~0.4%

2. **For production use**, either:
   - Use the structural derivation with acknowledged ~0.4% residual
   - Use the calibrated formula with explicit "ANCHORED" labeling

3. **For paper-complete derivation**, investigate:
   - The (1-u)^{2K-1} weight function and its role in the emergent Beta moment
   - Whether the mirror operator has eigenvalues that depend on I1 vs I2
