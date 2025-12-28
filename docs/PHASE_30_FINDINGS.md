# Phase 30 Findings: κ* Parity Restored

**Date:** 2025-12-26
**Status:** COMPLETE - Both benchmarks now have ~1% gap

---

## Executive Summary

Phase 30 identified and fixed the root cause of the κ* 9.29% gap reported in Phase 29:

**ROOT CAUSE:** `compute_c_paper_derived()` used hardcoded `S34 = -0.6` instead of computing actual S34.

- **κ benchmark:** Real S34 ≈ -0.600, hardcoded ≈ -0.600 → error ≈ 0 → gap = 1.35%
- **κ* benchmark:** Real S34 ≈ -0.443, hardcoded = -0.600 → error = -0.157 → gap = 9.29%

**FIX:** Updated `compute_c_paper_derived()` to compute actual S34 via term DSL.

---

## Before/After Comparison

### Before Fix (Phase 29 reported)

| Benchmark | c computed | c target | c gap |
|-----------|------------|----------|-------|
| κ | 2.109 | 2.137 | -1.35% |
| κ* | 1.758 | 1.938 | **-9.29%** |

### After Fix (Phase 30)

| Benchmark | c computed | c target | c gap |
|-----------|------------|----------|-------|
| κ | 2.109 | 2.137 | -1.35% |
| κ* | 1.915 | 1.938 | **-1.21%** |

Both benchmarks now have consistent ~1% accuracy.

---

## Key Insight: S34 Varies with Polynomials

S34 (I₃ + I₄) depends heavily on the polynomial coefficients:

| Benchmark | S34(+R) | Why Different |
|-----------|---------|---------------|
| κ | -0.600 | P2/P3 degree 3 |
| κ* | -0.443 | P2/P3 degree 2 (simpler) |

The hardcoded `-0.6` was calibrated for κ but wrong for κ*.

---

## Dual Benchmark Decomposition

Created `scripts/run_phase30_dual_benchmark_decomposition.py` which prints exact components:

```
=== BENCHMARK: KAPPA ===
Polynomial Provenance:
  loader: load_przz_polynomials()
  R: 1.3036, θ: 0.571429
  Q degree: 5, P2 degree: 3, P3 degree: 3
  Fingerprint: 175bffb87a77b2d8

S12 Components (paper regime):
  S12(+R): 0.797477
  S12(-R): 0.220121
  Ratio: 3.62

S34 Component:
  S34(+R): -0.600152

Assembly:
  m = exp(R) + 5 = 8.68
  c = 0.797 + 8.68 × 0.220 + (-0.600)
  c = 2.109 (target 2.137, gap -1.35%)

=== BENCHMARK: KAPPA_STAR ===
Polynomial Provenance:
  loader: load_przz_polynomials_kappa_star()
  R: 1.1167, θ: 0.571429
  Q degree: 1, P2 degree: 2, P3 degree: 2
  Fingerprint: ce0e7f4edf7d4971

S12 Components (paper regime):
  S12(+R): 0.614582
  S12(-R): 0.216444
  Ratio: 2.84

S34 Component:
  S34(+R): -0.443398  ← Different from κ!

Assembly:
  m = exp(R) + 5 = 8.05
  c = 0.615 + 8.05 × 0.216 + (-0.443)
  c = 1.915 (target 1.938, gap -1.21%)
```

---

## Polynomial Fingerprinting

Created `tests/test_phase30_polynomial_fingerprints.py` with 12 tests:

1. **Fingerprint distinctness:** κ ≠ κ* fingerprints
2. **Degree validation:** P2/P3 degrees match expectations
3. **Stability anchors:** Fingerprints are regression-locked

Fingerprints (stable):
- κ: `175bffb87a77b2d8`
- κ*: `ce0e7f4edf7d4971`

---

## Files Modified/Created

| File | Change |
|------|--------|
| `scripts/run_phase30_dual_benchmark_decomposition.py` | NEW - Decomposition diagnostic |
| `tests/test_phase30_polynomial_fingerprints.py` | NEW - 12 fingerprint tests |
| `src/mirror_transform_paper_exact.py` | MODIFIED - Fixed hardcoded S34 |
| `docs/PHASE_30_FINDINGS.md` | NEW - This document |

---

## Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| test_phase29_regime_lock.py | 19 | ✓ Pass |
| test_phase29_unified_paper_matches_dsl_paper.py | 22 | ✓ Pass |
| test_phase29_mirror_paper_sanity.py | 8 | ✓ Pass |
| test_phase30_polynomial_fingerprints.py | 12 | ✓ Pass |
| **Total** | **61** | **All Pass** |

---

## Why the Hardcoded Value Existed

The original `compute_c_paper_derived()` was a diagnostic function created during Phase 29 development. It used a hardcoded S34 = -0.6 as a "quick approximation" from the production evaluator's κ benchmark output.

The production evaluator `compute_c_paper_with_mirror()` always computed actual S34 and was working correctly all along.

The Phase 29 tests used `compute_c_paper_derived()` which had the bug.

---

## Lessons Learned

1. **Never hardcode computed values** - Always derive from inputs
2. **Two-benchmark gate catches bugs** - κ* failure revealed the issue
3. **Fingerprinting prevents configuration drift** - Polynomial identity verified
4. **Decomposition scripts are valuable** - Component breakdown identifies issues

---

## Conclusion

Phase 30 successfully:
1. Identified root cause (hardcoded S34 = -0.6)
2. Fixed `compute_c_paper_derived()` to compute actual S34
3. Added polynomial fingerprinting for regression protection
4. Restored κ* benchmark to ~1% accuracy

Both benchmarks now have consistent accuracy:
- **κ:** c gap = -1.35%
- **κ*:** c gap = -1.21%

The codebase is now ready for Phase 31 (derive m from first principles).
