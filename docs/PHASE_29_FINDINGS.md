# Phase 29 Findings: Unified Paper Regime Implementation

**Date:** 2025-12-26
**Status:** COMPLETE - 49 tests passing

---

## Executive Summary

Phase 29 implemented the paper regime in the unified bivariate engine, providing a clean, validated backend for computing I1/I2 with Case C kernel attenuation.

**Key Achievements:**
1. Hard-locked kernel regime in S12 identity (prevents cross-regime comparisons)
2. Created `unified_i1_paper.py` and `unified_i2_paper.py` matching OLD DSL to machine precision
3. Validated critical diagnostic pairs: (2,2) ~4x attenuation, (1,3)/(2,3) sign flips
4. Confirmed m_eff ≠ empirical m (3.6 vs 8.68) - m is calibrated, not derived

---

## Key Numerical Results

### kappa Benchmark (R=1.3036)

| Quantity | Value |
|----------|-------|
| S12_direct(+R) | 0.797 |
| S12_proxy(-R) | 0.220 |
| Ratio S12(+R)/S12(-R) | **3.62** |
| m_empirical = exp(R)+5 | **8.68** |
| c computed | 2.109 |
| c target | 2.137 |
| **c gap** | **1.35%** |

### kappa* Benchmark (R=1.1167)

| Quantity | Value |
|----------|-------|
| c computed | 1.758 |
| c target | 1.938 |
| **c gap** | **9.29%** |

**UPDATE (Phase 30):** This 9.29% gap was caused by a bug in `compute_c_paper_derived()` which used hardcoded S34 = -0.6 instead of computing actual S34. After the Phase 30 fix, κ* gap is now **-1.21%**, consistent with κ. See `PHASE_30_FINDINGS.md`.

---

## Paper Regime Validation

### (2,2) Magnitude Attenuation

| Regime | I1(2,2) | Ratio |
|--------|---------|-------|
| Raw | 3.884 | 1.00 |
| Paper | 0.917 | **0.24** |

The ~4x attenuation is due to Case C kernel integration for P2.

### Sign Flips (1,3) and (2,3)

| Pair | Raw | Paper | Sign Match |
|------|-----|-------|------------|
| (1,3) | -0.582 | +0.072 | **NO** (flip) |
| (2,3) | +3.571 | -0.173 | **NO** (flip) |

These sign flips are mathematically correct - Case C kernel changes the derivative structure.

---

## Files Created

| File | Purpose |
|------|---------|
| `src/unified_i1_paper.py` | Paper regime I1 with bivariate engine |
| `src/unified_i2_paper.py` | Paper regime I2 with Case C kernels |
| `src/mirror_transform_paper_exact.py` | Mirror analysis in paper regime |
| `src/evaluator/s12_spec.py` | Updated with KernelRegime enum |
| `tests/test_phase29_regime_lock.py` | 19 tests for regime locking |
| `tests/test_phase29_unified_paper_matches_dsl_paper.py` | 22 tests for backend validation |
| `tests/test_phase29_mirror_paper_sanity.py` | 8 tests for mirror analysis |

---

## Critical Findings

### 1. m_eff vs m_empirical

The ratio S12(+R)/S12(-R) = 3.62 is fundamentally different from m_empirical = 8.68.

This proves that **m = exp(R) + 5 is an empirical calibration**, not a derived quantity from the direct/proxy ratio.

### 2. Regime Lock Infrastructure

Created `KernelRegime` enum with `RAW` and `PAPER` values:
- S12CanonicalValue now requires `kernel_regime` field
- `RegimeMismatchError` raised on cross-regime comparisons
- All backend conventions updated with explicit regime

### 3. Case C Kernel Integration

The paper regime uses `case_c_taylor_coeffs()` for P2/P3:
```
K_ω(u; R, θ) = u^ω/(ω-1)! × ∫₀¹ a^{ω-1} P((1-a)u) exp(Rθua) da
```

where ω = ℓ - 1:
- P1 (ℓ=1): ω=0 → Case B (raw polynomial)
- P2 (ℓ=2): ω=1 → Case C
- P3 (ℓ=3): ω=2 → Case C

---

## Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| test_phase29_regime_lock.py | 19 | ✓ Pass |
| test_phase29_unified_paper_matches_dsl_paper.py | 22 | ✓ Pass |
| test_phase29_mirror_paper_sanity.py | 8 | ✓ Pass |
| **Total** | **49** | **All Pass** |

---

## Open Questions

1. **Why does m = exp(R) + 5 work?**
   The empirical formula achieves ~1.35% accuracy but the theoretical basis is unclear.

2. **kappa* gap larger (9%) than kappa (1.35%)**
   This suggests the empirical m formula may need R-dependence correction.

3. **Can we derive m from first principles?**
   Phase 28/29 show this is not straightforward - m encodes more than the direct/proxy ratio.

---

## Conclusion

Phase 29 successfully implemented paper regime in the unified bivariate engine. The key insight is that m = exp(R) + 5 is **empirically calibrated** rather than **theoretically derived**.

The paper regime achieves ~1.35% accuracy on the kappa benchmark using the empirical mirror formula:
```
c = S12(+R) + m × S12(-R) + S34(+R)
```

where m = exp(R) + 5 for K=3.
