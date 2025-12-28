# Phase 21C Summary: True Unified Bracket Implementation

**Date:** 2025-12-25
**Status:** COMPLETE (structural) / OPEN (c accuracy)

---

## Executive Summary

Phase 21C successfully implemented the TRUE unified bracket approach that does NOT compute at +R and -R separately. The key achievements:

1. **D=0 emerges naturally** from the bracket structure (not forced by setting S12_plus=0)
2. **B/A=5 holds exactly** (to machine precision ~1e-15)
3. **All 29 tests pass** (15 ladder tests + 14 gate tests)
4. **Anti-cheat verification** confirms no S12_plus=0 hack

However, the **absolute c magnitude is ~4x too large**, requiring further investigation.

---

## What Was Built

### New Files

1. **`src/unified_s12_evaluator_v3.py`**
   - Builds unified bracket at each (u,t) point
   - Includes P and Q polynomial factors
   - Does NOT compute at +R and -R separately
   - Does NOT set S12_plus=0 artificially
   - Computes all 6 triangle pairs

2. **`tests/test_phase22_symmetry_ladder.py`** (15 tests)
   - Scalar limit preservation tests
   - XY coefficient finiteness tests
   - All 6 pairs finite tests
   - Q factor sanity tests
   - Quadrature stability tests
   - Anti-cheat tests

3. **`tests/test_phase21c_gate.py`** (14 tests)
   - ABD decomposition structure tests
   - Unified value magnitude tests
   - Per-pair contribution tests
   - Quadrature stability tests
   - Anti-cheat tests (AST-based)
   - Dual benchmark tests

### Modified Files

1. **`src/evaluate.py`**
   - Added `mirror_mode="difference_quotient_v3"` option
   - Wired unified_s12_evaluator_v3 into the pipeline

---

## Test Results

### ABD Decomposition (Non-Tautological)

| Benchmark | V (unified) | A | B | D | B/A |
|-----------|-------------|---|---|---|-----|
| κ | 8.886e+00 | 1.023e+00 | 5.117e+00 | ~1e-15 | 5.000000 |
| κ* | 5.883e+00 | 7.304e-01 | 3.652e+00 | ~1e-15 | 5.000000 |

**Key insight:** D=0 and B/A=5 emerge from the bracket structure, NOT from artificially setting S12_plus=0.

### c Accuracy Issue

| Benchmark | c target | c computed (v3) | Gap |
|-----------|----------|-----------------|-----|
| κ | 2.137 | 8.286 | +287% |
| κ* | 1.938 | 5.440 | +181% |

The unified bracket value V is inflated by approximately `(exp(2R)-1)/(2R) ≈ 4.8x`.

---

## Root Cause Analysis

The c magnitude discrepancy arises from the difference quotient identity's inherent structure:

```
[N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β) = bracket
```

At x=y=0 and α=β=-R/L, the scalar limit of the bracket is:
```
(exp(2R) - 1) / (2R) ≈ 4.82 (for R=1.3036)
```

This factor is intrinsic to the difference quotient but is NOT present in the empirical I1 evaluation. The empirical mode evaluates the integrand directly without the t-integral that generates this factor.

### Comparison

| Mode | S12_minus (A equivalent) | Ratio |
|------|--------------------------|-------|
| Empirical | 0.220 | 1.0x |
| Unified V3 | 1.023 | 4.65x |
| Difference | - | ≈ (exp(2R)-1)/(2R) |

---

## What This Means

### Achievement: Structural Correctness

The unified bracket approach correctly implements the PRZZ difference quotient identity:
- The bracket at each (u,t) combines direct and mirror via the t-integral structure
- D=0 emerges from the identity, not from subtraction
- B/A=5 holds exactly (for K=3)

### Gap: Normalization Mismatch

The unified bracket computes a **different mathematical quantity** than the empirical I1:
- **Unified:** Regularized difference quotient with t-integral
- **Empirical:** Direct evaluation at ±R

These differ by a factor related to the t-integral's scalar contribution.

---

## Next Steps (Phase 22)

To achieve c accuracy with the unified approach:

1. **Derive the correct normalization** from PRZZ TeX Sections 15-16
2. **Divide by the t-integral scalar factor** or reformulate the c assembly
3. **Verify both benchmarks** pass the 0.5% accuracy gate

Alternatively:
- Continue using the **empirical mode** (which gives c ≈ 2.11 for κ, 1.5% accuracy)
- Use the unified approach only for **structural verification** (D=0, B/A=5)

---

## Files Summary

```
przz-extension/
├── src/
│   ├── unified_s12_evaluator_v3.py   # NEW: True unified bracket
│   └── evaluate.py                    # MODIFIED: Added v3 mode
├── tests/
│   ├── test_phase22_symmetry_ladder.py  # NEW: 15 ladder tests
│   └── test_phase21c_gate.py            # NEW: 14 gate tests
└── docs/
    └── PHASE_21C_SUMMARY.md           # This file
```

---

## Commands

```bash
# Run Phase 21C tests
PYTHONPATH=. python3 -m pytest tests/test_phase22_symmetry_ladder.py tests/test_phase21c_gate.py -v

# Run unified evaluator directly
PYTHONPATH=. python3 src/unified_s12_evaluator_v3.py

# Use in evaluate.py
result = compute_c_paper_with_mirror(..., mirror_mode="difference_quotient_v3")
```

---

*Generated 2025-12-25 as part of Phase 21C completion.*
