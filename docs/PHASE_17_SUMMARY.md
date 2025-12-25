# Phase 17 Summary: GPT Guidance Integration

**Date:** 2025-12-24
**Status:** COMPLETE
**Predecessor:** Phase 16 (J13/J14 Laurent Factor Fix)

---

## Executive Summary

Phase 17 addressed GPT's guidance on five priority items. Key findings:

1. **Phase 17B Complete:** Per-piece delta decomposition identifies J13/J14 as the source of κ vs κ* asymmetry (ratio 1.89)

2. **Phase 17A Clarified:** Case C a-integrals ARE already implemented in the production pipeline (Term DSL). The J1x diagnostic uses a simplified approach but that's acceptable.

3. **Phase 17C Complete:** Documented semantic vs numeric mode policy (Decision 8)

4. **TRUTH_SPEC Updated:** Changed "MISSING" status to reflect current Case C implementation

---

## Phase 17B: Per-Piece Delta Decomposition

### Implementation

Extended `DeltaMetricsExtended` dataclass with per-piece breakdown:
- `j11_plus`, `j12_plus`, `j15_plus`, `j13_plus`, `j14_plus` (at +R)
- `j11_minus`, `j12_minus`, `j15_minus` (at -R)

Added `compare_laurent_modes_per_piece()` function for diagnostic comparison.

### Key Finding

**Most asymmetric pieces: j13_plus and j14_plus** (ratio κ/κ* = 1.89)

| Piece | κ Change | κ* Change | Ratio |
|-------|----------|-----------|-------|
| j11_plus | 0.000 | 0.000 | N/A |
| j12_plus | -0.215 | -0.136 | 1.57 |
| j13_plus | -0.073 | -0.039 | **1.89** |
| j14_plus | -0.073 | -0.039 | **1.89** |

The asymmetry arises because Laurent approximation error is R-dependent:
- κ (R=1.3036): 29% error
- κ* (R=1.1167): 21% error

---

## Phase 17A: Case C Status Clarification

### Key Discovery

Case C a-integrals ARE already implemented in the production pipeline:
- `PolyFactor.evaluate()` dispatches to `case_c_taylor_coeffs()` when omega > 0
- `kernel_regime="paper"` enables proper Case C handling
- Production `compute_c_paper_with_mirror()` uses this infrastructure

### Two Pipelines

| Pipeline | Uses Case C? | Purpose |
|----------|--------------|---------|
| Term DSL (`evaluate.py`) | ✓ Yes | Production κ computation |
| J1x diagnostic (`j1_euler_maclaurin.py`) | No | B/A ratio validation |

### Current Accuracy (with Case C)

- κ (R=1.3036): c gap = **-1.35%**
- κ* (R=1.1167): c gap = **-1.21%**

This ~1.5% accuracy is NOT due to missing Case C — it's from:
- Residual m₁ calibration error (m₁ is empirical)
- Possible polynomial coefficient transcription errors
- Missing higher-order terms

---

## Phase 17C: Semantic vs Numeric Mode Policy

### Decision 8 Established

Two distinct computation modes:

1. **Semantic Mode** (`RAW_LOGDERIV`)
   - Uses Laurent expansion `(1/R + γ)²`
   - Matches what the TeX formula says
   - Use for: theoretical validation

2. **Numeric Mode** (`ACTUAL_LOGDERIV`)
   - Uses actual numerical `(ζ'/ζ)(1-R)²`
   - Best accuracy at finite R
   - Use for: production κ computation

---

## TRUTH_SPEC Updated

Changed Case C status from "MISSING" to reflect current implementation:

| Pair | Cases | Term DSL | J1x Diagnostic |
|------|-------|----------|----------------|
| (1,1) | B×B | ✓ Correct | ✓ Correct |
| (1,2) | B×C | ✓ Implemented | Simplified |
| (1,3) | B×C | ✓ Implemented | Simplified |
| (2,2) | C×C | ✓ Implemented | Simplified |
| (2,3) | C×C | ✓ Implemented | Simplified |
| (3,3) | C×C | ✓ Implemented | Simplified |

---

## GPT Guidance Response

### Priority 1: Fix Case C a-integral counting
**Status:** Already done in Term DSL. No action needed.

### Priority 2: Decompose Phase 16's κ* drift
**Status:** Done. J13/J14 identified as most asymmetric (ratio 1.89).

### Priority 3: Split semantic vs numeric mode
**Status:** Done. Decision 8 documents the policy.

### Priority 4: Determine if δ(R) should be zero or carried
**Status:** Deferred. Current accuracy is sufficient for K=4 readiness.

### Priority 5: K=4 implementation
**Status:** Ready. Phase 15E showed amplification factor 0.71 (< 1), meaning gap shrinks at K=4. Case C is implemented in production pipeline.

---

## K=4 Readiness Assessment

### Go Criteria Met

1. ✓ Case C implemented in production pipeline
2. ✓ Phase 15E showed amplification 0.71 < 1 (gap shrinks at K=4)
3. ✓ Per-piece asymmetry understood (J13/J14 are R-sensitive)
4. ✓ Mode policy documented (Decision 8)

### Residual Risk

- m₁ = exp(R) + (2K-1) is extrapolated for K=4 (empirical, not derived)
- If K=4 fails, check m₁ = exp(R) + 7 calibration first

---

## Files Modified

| File | Changes |
|------|---------|
| `src/ratios/delta_track_harness.py` | Per-piece breakdown, `compare_laurent_modes_per_piece()` |
| `docs/TRUTH_SPEC.md` | Updated Case C status table |
| `docs/DECISIONS.md` | Added Decision 8 (semantic vs numeric) |
| `docs/PHASE_17B_SUMMARY.md` | Per-piece analysis details |
| `docs/PHASE_17_SUMMARY.md` | This document |

---

## Test Status

All 63 relevant tests pass:
- `test_delta_track_harness_invariants.py`: 36 passed
- `test_m1_policy_gate.py`: 27 passed

---

## Next Steps

1. **K=4 Implementation:** Case C infrastructure is ready. Need to:
   - Add P₄ polynomial with degree/constraints
   - Extend term tables for K=4 pairs (10 pair types)
   - Use `allow_extrapolation=True` for m₁ at K=4
   - Validate against any available K=4 reference

2. **Residual Gap Investigation:** The ~1.5% gap could be from:
   - m₁ calibration (try slightly different formula)
   - Polynomial transcription (verify coefficients from TeX)
   - Missing I₅ lower-order terms

