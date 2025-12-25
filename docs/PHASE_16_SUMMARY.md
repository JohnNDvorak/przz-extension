# Phase 16 Summary: J13/J14 Laurent Factor Fix

**Date:** 2025-12-24
**Status:** Complete

---

## Overview

Phase 16 extended the Phase 15 Laurent factor fix to J13 and J14 integrals. Previously, Phase 15 fixed J12's Laurent factor by using actual `(zeta'/zeta)(1-R)^2` values instead of the Laurent approximation `(1/R + gamma)^2`. This Phase extends the same treatment to J13 and J14, which use the SINGLE factor `(zeta'/zeta)(1-R)` (not squared).

---

## Changes Made

### 1. New Function: `compute_zeta_logderiv_actual()`
**File:** `src/ratios/g_product_full.py`

```python
def compute_zeta_logderiv_actual(R: float, precision: int = 50) -> float:
    """Compute ACTUAL (zeta'/zeta)(1-R) value (single factor, not squared)."""
```

Returns the single zeta log-derivative factor used by J13/J14.

### 2. Updated J13/J14 with `laurent_mode` parameter
**File:** `src/ratios/j1_euler_maclaurin.py`

Both `j13_as_integral()` and `j14_as_integral()` now accept `laurent_mode` parameter:
- `RAW_LOGDERIV`: Uses Laurent approximation `(1/R + gamma)`
- `ACTUAL_LOGDERIV`: Uses actual numerical value via mpmath

### 3. Updated `compute_I34_components()` to thread mode
The mode is now passed through to J13/J14.

### 4. Updated `compute_m1_with_mirror_assembly()` to thread mode
The `laurent_mode` parameter is now threaded to I34 components (J13/J14).

---

## Results

### Zeta Log-Derivative Values

| Benchmark | R | Actual | Laurent | Error |
|-----------|------|--------|---------|-------|
| kappa | 1.3036 | 1.732 | 1.344 | 29% |
| kappa* | 1.1167 | 1.778 | 1.473 | 21% |

### B/A Improvement

| Benchmark | Laurent B/A | Actual B/A | Change |
|-----------|-------------|------------|--------|
| kappa | 5.253 (5.05% gap) | 4.953 (0.93% gap) | **+4.12 pp improvement** |
| kappa* | 5.079 (1.58% gap) | 4.867 (2.66% gap) | -1.08 pp worse |

### Average Gap Analysis

- Laurent average B/A: 5.166 (3.32% gap)
- Actual average B/A: 4.910 (1.80% gap)
- **Overall improvement: 1.52 pp**

---

## Key Findings

### 1. Asymmetric Behavior
The fix improved kappa dramatically (+4.12 pp) but made kappa* slightly worse (-1.08 pp). This asymmetry suggests there may be additional factors that differ between benchmarks.

### 2. Average Gap Decreased
Despite asymmetry, the average gap decreased from 3.32% to 1.80%, indicating overall improvement.

### 3. J13/J14 Ratio Confirmed
The J13/J14 ratio (actual/Laurent) = 1.29 for kappa matches the underlying zeta factor ratio, confirming the fix is mathematically correct.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/ratios/g_product_full.py` | Added `compute_zeta_logderiv_actual()` |
| `src/ratios/j1_euler_maclaurin.py` | Added `laurent_mode` to j13, j14, compute_I34_components; threaded mode in compute_m1_with_mirror_assembly |
| `tests/test_phase16_j13_j14_fix.py` | New test file (10 tests) |
| `docs/PHASE_16_SUMMARY.md` | This summary |

---

## Test Results

All 10 Phase 16 tests pass:
- `TestZetaLogderivActual`: 2 tests
- `TestJ13ModeComparison`: 2 tests
- `TestFullAssemblyImprovement`: 2 tests
- `TestComparisonLaurentVsActual`: 1 test
- `TestNoRegression`: 3 tests

Key regression tests (38 tests) also pass.

---

## Recommendations

### What Works
The ACTUAL_LOGDERIV mode is now applied consistently to J12, J13, and J14. The kappa benchmark shows significant improvement.

### Open Questions
1. **Why asymmetric?** Kappa improved dramatically but kappa* got worse. Investigate whether there are additional polynomial-dependent or R-dependent factors.

2. **Consider Phase 16B**: If we need to eliminate "mystery library behavior", implement the functional equation evaluator as described in the plan.

3. **Per-piece attribution**: The delta harness could be extended to track J13/J14 contributions separately for better debugging.

---

## Conclusion

Phase 16 successfully extended the Laurent factor fix to J13/J14. The kappa benchmark B/A gap decreased from 5.05% to 0.93%, bringing it very close to the target of 5.0. The kappa* benchmark showed slight worsening, suggesting additional investigation may be needed for full resolution across both benchmarks.
