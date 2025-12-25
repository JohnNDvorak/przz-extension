# Phase 17 Implementation Plan: GPT Guidance Integration

**Date:** 2025-12-24
**Status:** PLAN MODE — Awaiting Approval
**Predecessor:** Phase 16 (J13/J14 Laurent Factor Fix)

---

## Executive Summary

GPT's guidance identifies five priority items. After codebase analysis, this plan addresses them in order of impact on K=4 readiness:

1. **Priority 1 (CRITICAL):** Fix Case C cross-term a-integral bookkeeping
2. **Priority 2 (HIGH):** Decompose Phase 16's κ* drift to specific J-pieces
3. **Priority 3 (MEDIUM):** Reconcile semantic vs numeric mode policy
4. **Priority 4 (LOW):** Determine if δ(R) should be zero or carried
5. **Priority 5 (BLOCKED):** K=4 implementation (awaits 1-4)

---

## Current State Analysis

### What Works
- Paper regime + mirror assembly achieves ~1.5% accuracy on both benchmarks
- Phase 15's ACTUAL_LOGDERIV for J12 reduced gap from 5% to ~1%
- Phase 16 extended to J13/J14 but introduced κ/κ* asymmetry

### What's Missing
| Pair | Cases | a-integrals Required | Status |
|------|-------|---------------------|--------|
| (1,1) | B×B | 0 | ✓ Correct |
| (1,2) | B×C | 1 | ⚠️ MISSING in J1x pipeline |
| (1,3) | B×C | 1 | ⚠️ MISSING in J1x pipeline |
| (2,2) | C×C | 2 | ⚠️ MISSING in J1x pipeline |
| (2,3) | C×C | 2 | ⚠️ MISSING in J1x pipeline |
| (3,3) | C×C | 2 | ⚠️ MISSING in J1x pipeline |

**Key Finding:** Case C kernels ARE implemented (`case_c_kernel.py`, `case_c_exact.py`) but only for the I2 channel. The J1x pipeline (`j1_euler_maclaurin.py`) that drives the B/A diagnostic does NOT incorporate Case C structure.

### Phase 16 Asymmetry
| Benchmark | RAW B/A | ACTUAL B/A | Change |
|-----------|---------|------------|--------|
| κ (R=1.3036) | 5.253 | 4.953 | **+4.12 pp** (closer to 5) |
| κ* (R=1.1167) | 5.079 | 4.867 | **-1.08 pp** (further from 5) |

This asymmetry suggests:
- The fix direction is polynomial/R-dependent
- J13/J14 may need different treatment than J12

---

## Implementation Plan

### Phase 17A: Case C a-integral Integration into J1x Pipeline
**Priority:** CRITICAL
**Effort:** Medium-High

#### Rationale
TRUTH_SPEC Section 8 explicitly marks B×C and C×C pairs as MISSING the auxiliary a-integral. GPT identifies this as the "#1 unresolved item that blocks confident K=4."

#### Implementation Steps

**Step 17A.1:** Create unified Case C kernel interface for J1x
- New module: `src/ratios/j1_case_c_adapter.py`
- Expose Case C kernel computation compatible with J1x polynomial interface
- Handle both single (B×C) and product (C×C) kernel structure

**Step 17A.2:** Extend `j12_as_integral()` to support Case C
- Add `case_c_left` and `case_c_right` parameters
- For B×C: Replace P_right(u) with u^ω × K_ω(u; R)
- For C×C: Replace both factors with kernels

**Step 17A.3:** Extend `j13_as_integral()` and `j14_as_integral()` for Case C
- Same pattern as j12
- Note: These use (1-u) weight factor which interacts with Case C kernel

**Step 17A.4:** Thread Case C through `compute_m1_with_mirror_assembly()`
- Determine pair type from polynomial indices
- Apply appropriate Case C structure

**Step 17A.5:** Validation tests
- Compare with existing `compute_i2_all_pairs_case_c()` output
- Verify B×B pairs unchanged
- Check ratio of Case C to raw matches `diagnose_case_c_ratios()` output

#### Acceptance Criteria
- All 6 pair types correctly dispatch to B×B, B×C, or C×C logic
- (1,1) pair values unchanged
- (1,2), (1,3) pairs use ONE a-integral
- (2,2), (2,3), (3,3) pairs use TWO a-integrals
- B/A diagnostic updated with Case C structure

---

### Phase 17B: Per-Piece Delta Decomposition
**Priority:** HIGH
**Effort:** Low-Medium

#### Rationale
Phase 16 shows J13/J14 "actual log-deriv" extension pushes κ* down more than κ. GPT says: "Extend the delta harness to report δ decomposed as contribution from J12(+R), J13(+R), J14(+R)... so you can point to the exact sub-term."

#### Implementation Steps

**Step 17B.1:** Extend `DeltaMetricsExtended` dataclass
```python
@dataclass(frozen=True)
class DeltaMetricsExtended:
    # ... existing fields ...
    # NEW: Per-piece breakdown
    j11_plus: float
    j12_plus: float
    j15_plus: float
    j13_plus: float
    j14_plus: float
    j11_minus: float
    j12_minus: float
    j15_minus: float
```

**Step 17B.2:** Update `compute_delta_metrics_extended()`
- Capture individual j1x values from `compute_m1_with_mirror_assembly()` result
- The function already returns `i12_plus_pieces` and `i34_plus_pieces` dicts

**Step 17B.3:** Add comparative analysis function
```python
def compare_laurent_modes_per_piece(benchmark: str) -> Dict:
    """Compare RAW vs ACTUAL mode contribution by piece."""
```

**Step 17B.4:** Create diagnostic report
- Show which J-piece changes most between RAW and ACTUAL
- Identify which piece drives κ* asymmetry

#### Acceptance Criteria
- Can identify J13 vs J14 contribution to κ* drift
- Clear attribution of which piece is responsible for asymmetry

---

### Phase 17C: Semantic vs Numeric Mode Reconciliation
**Priority:** MEDIUM
**Effort:** Low

#### Rationale
GPT says: "Don't flip the default globally; split semantic mode vs numerical diagnostic mode."

Current confusion:
- Decision 7 says RAW_LOGDERIV is semantically correct
- Phase 15B sets DEFAULT_LAURENT_MODE = ACTUAL_LOGDERIV
- These conflict

#### Implementation Steps

**Step 17C.1:** Update `LaurentMode` enum documentation
```python
class LaurentMode(str, Enum):
    """Laurent factor handling modes.

    SEMANTIC modes (what the TeX formula says):
    - RAW_LOGDERIV: (1/R + γ)² — The literal Laurent expansion

    NUMERIC modes (practical computation):
    - ACTUAL_LOGDERIV: True numerical (ζ'/ζ)(1-R)² — Best accuracy

    DEPRECATED/DIAGNOSTIC:
    - POLE_CANCELLED, FULL_G_PRODUCT
    """
```

**Step 17C.2:** Add semantic/numeric mode selectors
```python
SEMANTIC_DEFAULT = LaurentMode.RAW_LOGDERIV
NUMERIC_DEFAULT = LaurentMode.ACTUAL_LOGDERIV
```

**Step 17C.3:** Update `compute_m1_with_mirror_assembly()` signature
```python
def compute_m1_with_mirror_assembly(
    # ... existing params ...
    laurent_mode: LaurentMode = None,  # None = use context-appropriate default
    compute_mode: str = "numeric",  # "semantic" | "numeric"
):
```

**Step 17C.4:** Document in DECISIONS.md
- Add Decision 8: Semantic vs Numeric computation modes
- Clarify when to use each

#### Acceptance Criteria
- Clear separation of semantic (what formula says) vs numeric (best accuracy)
- Neither mode is implicitly favored without explicit choice
- Documentation updated

---

### Phase 17D: δ(R) Analysis
**Priority:** LOW (Informational)
**Effort:** Low

#### Rationale
GPT asks: Is δ(R) supposed to be zero (missing piece) or supposed to be carried (real correction)?

#### Implementation Steps

**Step 17D.1:** Theoretical analysis
- Review PRZZ TeX for δ structure
- Determine if asymptotic claims δ→0 or allows δ(R)

**Step 17D.2:** Numerical sweep
```python
def delta_vs_R_sweep(R_values: list) -> Dict:
    """Compute δ(R) across R range to see trend."""
```

**Step 17D.3:** Document finding
- If δ→0 as R→∞, then we have missing piece
- If δ(R) is O(1/R) or similar, it's a legitimate correction

#### Acceptance Criteria
- Clear answer: δ is missing piece OR structural correction
- Documented in summary

---

### Phase 17E: K=4 Readiness Assessment
**Priority:** BLOCKED (Awaits 17A-17D)
**Effort:** N/A

#### Prerequisites
1. ✓ Phase 15E showed amplification factor 0.71 < 1 (gap shrinks at K=4)
2. ⬜ Case C cross-terms implemented (17A)
3. ⬜ Phase 16 asymmetry understood (17B)
4. ⬜ Mode policy reconciled (17C)

#### Go/No-Go Criteria
- Case C for K=3 reduces gap (or at least doesn't worsen)
- κ and κ* both improve or stay stable
- Amplification factor remains < 1

---

## Files to Create/Modify

### New Files
| File | Purpose |
|------|---------|
| `src/ratios/j1_case_c_adapter.py` | Case C integration for J1x pipeline |
| `tests/test_phase17_case_c_j1x.py` | Tests for Case C in J1x |
| `docs/PHASE_17_SUMMARY.md` | Phase 17 completion summary |

### Modified Files
| File | Changes |
|------|---------|
| `src/ratios/j1_euler_maclaurin.py` | Add Case C support to j12, j13, j14 |
| `src/ratios/delta_track_harness.py` | Per-piece decomposition |
| `docs/DECISIONS.md` | Add Decision 8 (semantic vs numeric) |
| `docs/TRUTH_SPEC.md` | Update Case C status from MISSING to IMPLEMENTED |

---

## Test Plan

### Unit Tests
- Case C kernel output matches between J1x and I2 channels
- B×B pairs unchanged when Case C added
- Per-piece decomposition sums to total

### Integration Tests
- B/A ratio with Case C vs without
- Comparison at both κ and κ* benchmarks
- No regression in existing Phase 16 tests

### Acceptance Tests
- κ gap ≤ 1% (current: 0.93%)
- κ* gap ≤ 2% (current: 2.66%, hoping to improve)
- κ/κ* ratio error ≤ 0.5%

---

## Risk Assessment

### Risk 1: Case C Makes Things Worse
**Probability:** Medium
**Mitigation:** The I2 channel already uses Case C and works. If J1x Case C worsens results, we have a formula mismatch to investigate.

### Risk 2: κ* Asymmetry Persists
**Probability:** High
**Mitigation:** 17B will identify exactly which piece causes asymmetry. May indicate polynomial-dependent correction factor.

### Risk 3: δ(R) ≠ 0 is Real
**Probability:** Medium
**Mitigation:** If δ(R) is a legitimate correction, promote it into the m₁ formula (still derived, not fitted).

---

## Summary

This plan addresses GPT's guidance in priority order:

1. **Case C a-integrals (17A):** The #1 K-sensitive structural gap
2. **Per-piece decomposition (17B):** Understand Phase 16 asymmetry
3. **Semantic/numeric split (17C):** Clean policy for mode selection
4. **δ analysis (17D):** Informational for final gap closure
5. **K=4 (17E):** Blocked until 17A-17D complete

Expected outcomes:
- Case C integration may reduce or at least stabilize the gap
- Per-piece analysis will identify root cause of κ* drift
- Clear mode policy prevents future confusion
- K=4 will have solid foundation

---

## Questions for User

Before proceeding:

1. **Case C priority:** Should we implement 17A (Case C) first, or 17B (per-piece decomposition) to understand Phase 16 asymmetry?

2. **Mode policy:** For production κ computation, do you prefer:
   - Semantic (RAW_LOGDERIV) for theoretical correctness
   - Numeric (ACTUAL_LOGDERIV) for best accuracy
   - Both, switchable via parameter

3. **Scope:** Should Phase 17 include all five sub-phases, or focus on 17A+17B only?

