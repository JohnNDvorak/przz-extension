# Phase 17B Summary: Per-Piece Delta Decomposition

**Date:** 2025-12-24
**Status:** Complete
**Predecessor:** Phase 16 (J13/J14 Laurent Factor Fix)

---

## Executive Summary

Phase 17B extended the delta tracking harness to decompose contributions by individual J-piece (j11, j12, j13, j14, j15). This identifies which piece causes the κ vs κ* asymmetry observed in Phase 16.

### Key Finding

**The most asymmetric pieces are j13_plus and j14_plus** with a κ/κ* ratio of 1.89.

The asymmetry arises because:
1. Laurent approximation error is R-dependent (29% for κ, 21% for κ*)
2. ACTUAL_LOGDERIV correctly applies larger corrections to κ
3. But both benchmarks move in the same direction, overshooting for κ*

---

## Implementation

### Changes Made

**File:** `src/ratios/delta_track_harness.py`

1. **Extended `DeltaMetricsExtended` dataclass** with per-piece fields:
   - `j11_plus`, `j12_plus`, `j15_plus`, `j13_plus`, `j14_plus` (at +R)
   - `j11_minus`, `j12_minus`, `j15_minus` (at -R)

2. **Updated `compute_delta_metrics_extended()`** to populate new fields from existing `i12_plus_pieces`, `i12_minus_pieces`, `i34_plus_pieces` dicts.

3. **Added `compare_laurent_modes_per_piece()` function** for diagnostic comparison between RAW and ACTUAL modes per piece.

---

## Diagnostic Results

### Per-Piece Changes (RAW → ACTUAL)

| Piece | κ Change | κ* Change | Ratio κ/κ* |
|-------|----------|-----------|------------|
| j11_plus | 0.000 | 0.000 | N/A (no change) |
| j12_plus | -0.215 | -0.136 | 1.57 |
| j15_plus | 0.000 | 0.000 | N/A (no change) |
| j13_plus | -0.073 | -0.039 | 1.89 |
| j14_plus | -0.073 | -0.039 | 1.89 |

### Interpretation

- **J11 and J15**: No change (they don't use Laurent factors)
- **J12**: Largest magnitude change, but ratio 1.57 is moderate
- **J13/J14**: Highest asymmetry (ratio 1.89) — the root cause

### Why J13/J14 Cause Asymmetry

Phase 16 applied ACTUAL_LOGDERIV to J13/J14 which uses the SINGLE factor (ζ'/ζ)(1-R):

| Benchmark | R | Laurent | Actual | Error |
|-----------|------|---------|--------|-------|
| κ | 1.3036 | 1.35 | 1.73 | 29% |
| κ* | 1.1167 | 1.47 | 1.78 | 21% |

The 29% vs 21% error difference means κ gets a larger correction than κ*.

### B/A Movement

| Benchmark | RAW B/A | ACTUAL B/A | Δ | Gap |
|-----------|---------|------------|---|-----|
| κ | 5.253 | 4.953 | -0.299 | -0.93% |
| κ* | 5.079 | 4.867 | -0.212 | -2.66% |

Both benchmarks move DOWN (toward 5), but:
- κ starts above 5 → moves TOWARD 5 → improves
- κ* starts closer to 5 → moves BELOW 5 → overshoots

---

## Implications for Phase 17A (Case C)

The asymmetry is now understood:
1. It's R-dependent, not polynomial-dependent
2. It's concentrated in J13/J14 (the pieces using single ζ'/ζ factor)
3. Case C cross-terms (17A) will add structure to B×C and C×C pairs

Case C is expected to:
- Modify J-pieces for pairs involving P₂/P₃
- May change the balance between I12 and I34 contributions
- Could either help or hurt the asymmetry depending on which pairs dominate

---

## Recommendations

### Option 1: Accept Phase 16 Results (Recommended for Now)
- Average gap is 1.80% (improved from 3.32%)
- κ is within 1% of target
- Proceed to Case C (17A) to see if structural improvement helps κ*

### Option 2: Revert J13/J14 to RAW_LOGDERIV
- Would make κ* better but κ worse
- Not recommended as it's semantically inconsistent with J12

### Option 3: Polynomial-Specific Calibration
- Could tune J13/J14 factor differently for each benchmark
- NOT recommended — feels like overfitting

---

## Tests

All 36 existing delta harness tests pass with the new per-piece fields.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/ratios/delta_track_harness.py` | Added per-piece fields and `compare_laurent_modes_per_piece()` |
| `docs/PHASE_17B_SUMMARY.md` | This document |

---

## Next Steps

1. **Phase 17A:** Implement Case C a-integral structure in J1x pipeline
2. **Phase 17C:** Document semantic vs numeric mode policy
3. Observe if Case C changes the κ/κ* balance

