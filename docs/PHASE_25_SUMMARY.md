# Phase 25 Summary: Gap Attribution (No Fitting, No Corrections)

**Created:** 2025-12-25
**Status:** COMPLETE
**Outcome:** Gap isolated to S12 with constant factor-of-3 structural issue in P=Q=1 microcase

---

## Executive Summary

Phase 25 accomplished systematic gap attribution between the unified S12 evaluator and the empirical DSL evaluator, identifying precisely where the 5-7% gap originates.

**Key Findings:**

1. **S34 is INVARIANT** between evaluators (delta_S34 = 0)
2. **Gap is entirely in S12**: -3.12% for kappa, -4.55% for kappa*
3. **P=Q=1 microcase reveals factor-of-3**: unified = 3x empirical for constant polynomials
4. **Eigenvalue mapping is consistent** (24 tests pass)

---

## Task 25.1: Gap Attribution Harness

### What Was Done

Created comprehensive gap attribution infrastructure:
- `src/evaluator/gap_attribution.py`: Main module with `GapReport` dataclass
- `scripts/run_phase25_gap_attribution.py`: Runner script for both benchmarks
- `tests/test_phase25_gap_attribution.py`: 22 tests for gap attribution

### Key Results

| Benchmark | unified S12 | empirical S12 | ratio | gap |
|-----------|-------------|---------------|-------|-----|
| kappa | 2.624 | 2.709 | 0.969 | -3.1% |
| kappa* | 2.251 | 2.358 | 0.955 | -4.5% |

**Interpretation:** The unified evaluator computes S12 ~3-4.5% LOWER than the empirical evaluator.

---

## Task 25.2: S34 Invariance Gate

### What Was Done

Verified S34 computation is identical between evaluators:

```
kappa:   delta_S34 = 0.0, ratio_S34 = 1.0
kappa*:  delta_S34 = 0.0, ratio_S34 = 1.0
```

### Conclusion

**S34 is NOT the source of the gap.** The I3/I4 terms use the same code path in both modes, confirming the gap is isolated to S12 (I1/I2).

---

## Task 25.3: P=Q=1 Microcase Ladder

### What Was Done

Created microcase module to isolate bracket structure:
- `src/unified_s12_microcases.py`: P=Q=1 microcase functions
- `tests/test_phase25_microcases.py`: 14 tests

### Critical Finding: Factor of 3

**The unified evaluator returns exactly 3x the empirical value for P=Q=1:**

| Benchmark | unified I1 | empirical I1 | ratio |
|-----------|------------|--------------|-------|
| kappa | 4.678e+00 | 1.559e+00 | **3.0** |
| kappa* | 2.658e+00 | 8.861e-01 | **3.0** |

### Interpretation

**The factor of 3 is constant across benchmarks**, indicating a structural issue in the bracket, not polynomial interactions.

Possible sources of the factor-of-3:
1. PRZZ difference quotient denominator handling
2. Factorial normalization placement
3. Log factor contribution `(1/θ + x + y)`
4. Missing division in the unified bracket

---

## Task 25.5: Eigenvalue Mapping Tests

### What Was Done

Created 24 tests verifying eigenvalue structure:
- `tests/test_phase25_eigenvalue_mapping.py`

### Properties Verified

1. **At origin:** A_alpha = A_beta = t (for x=y=0)
2. **Symmetry:** A_alpha(x,y) = A_beta(y,x)
3. **Boundary conditions:** Correct behavior at t=0 and t=1
4. **Derivatives:** dA/dx, dA/dy match analytic formulas
5. **Q consistency:** Q(A_alpha) * Q(A_beta) = Q(t)² at origin

---

## Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| test_phase25_gap_attribution.py | 22 | PASS |
| test_phase25_microcases.py | 14 | PASS |
| test_phase25_eigenvalue_mapping.py | 24 | PASS |
| **Total Phase 25** | **60** | **ALL PASS** |

---

## Phase 26 Recommendations

Based on Phase 25 findings, the next phase should focus on:

### Priority 1: Find the Factor-of-3 Source

The P=Q=1 microcase shows a constant factor-of-3 discrepancy. This is the most concrete diagnostic finding. Investigate:

1. **Difference quotient denominator:** The PRZZ identity has (α+β) = -2Rθ in the denominator. Is this being applied twice somewhere?

2. **Log factor structure:** The log factor `log(N^{x+y}T)` becomes `(1 + θ(x+y))/θ`. Check if there's a θ factor missing.

3. **Factorial normalization:** The unified evaluator applies 1/(ℓ₁! × ℓ₂!). Check if this overlaps with something in the empirical path.

### Priority 2: Per-Pair Analysis

The gap attribution shows some pairs have negative contributions (13, 23). Investigate whether the sign pattern is correct.

### Priority 3: Factor Toggles (Task 25.4)

Add diagnostic toggles to isolate individual factor contributions:
- `include_log_factor`: Toggle the (1/θ + x + y) factor
- `include_Q`: Toggle Q polynomial (already exists)
- `include_poly_P`: Toggle P polynomials

---

## Files Created/Modified

### New Files
- `src/evaluator/gap_attribution.py` (gap attribution harness)
- `src/unified_s12_microcases.py` (P=Q=1 microcase functions)
- `scripts/run_phase25_gap_attribution.py` (runner script)
- `tests/test_phase25_gap_attribution.py` (22 tests)
- `tests/test_phase25_microcases.py` (14 tests)
- `tests/test_phase25_eigenvalue_mapping.py` (24 tests)
- `docs/PHASE_25_SUMMARY.md` (this document)

### Modified Files
- `src/evaluator/__init__.py` (added note about gap_attribution)

---

## Key Takeaways

1. **"Derived > tuned" discipline maintained:** No fitting or empirical corrections added
2. **Gap is in S12, not S34:** S34 computation is identical in both modes
3. **Factor-of-3 is the key finding:** P=Q=1 microcase shows unified = 3x empirical
4. **Eigenvalues are consistent:** No structural issues in A_alpha, A_beta definitions
5. **60 new tests:** Comprehensive coverage of gap attribution

---

*Phase 25 completed 2025-12-25*
