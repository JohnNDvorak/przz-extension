# K-Safe PRZZ Baseline Lock-Down

**Date:** 2025-12-22
**Status:** Phases 1-3B Complete → Moving to SPEC-LOCK and HARD-LOCK
**Goal:** Lock down the exact PRZZ baseline so the evaluator doesn't drift (or blow up) when K increases

---

## Executive Summary

This document describes the implementation of the "K-Safe PRZZ Baseline Lock-Down" — a systematic effort to validate and gate the PRZZ evaluator before extending to K>3.

### What Was Accomplished

| Phase | Description | Status | Tests |
|-------|-------------|--------|-------|
| Phase 1 | i1_source switch for independent I1 validation | ✅ Complete | 21 pass |
| Phase 2 | S34 mirror gate (Step-3 closure) | ✅ Complete | 16 pass |
| Phase 3 | m1 L-sweep diagnostic | ⚠️ DIVERGES | 21 pass, 2 skip |
| Phase 3B | Unified-t combined identity | ✅ Complete | 31 pass |

**Total: 89 tests passing, 2 skipped**

**Phase 3 Update:** The finite-L approach DIVERGES linearly with L. Phase 3B (unified-t) shows that the correct TeX structure (without extra variable) is IDENTICAL to post-identity — m1 remains empirical calibration.

### Key Findings

1. **Step-2 Mystery RESOLVED**: The pre-identity bracket was missing the t-integration variable. Post-identity operator now matches DSL exactly for all K=3 pairs.

2. **I3/I4 Mirror: SPEC-LOCKED** *(Updated 2025-12-22)*
   - **TRUTH_SPEC.md Section 10 (lines 370-388) definitively states:**
     - I₁/I₂: HAVE mirror structure (combined with T^{-α-β} × mirror term)
     - I₃/I₄: NO mirror structure (single evaluation only)
   - The 350%/182% overshoot is a **negative control** — it proves external mirror is WRONG
   - See: "Phase 2 Resolution" section below

3. **m1 = exp(R)+5 is CALIBRATION**: The empirical mirror weight formula works but is not derived from first principles. Both finite-L (Option A) and unified-t (Phase 3B) approaches failed to derive m1 — it remains empirical.

---

## Phase 1: i1_source Switch

### Motivation

The post-identity operator was validated to match DSL exactly for all K=3 pairs at both benchmarks (Run 21). Making it an alternate `i1_source` (like the existing `i2_source`) provides:

- **Regression safety net** for K>3 extension
- **Independent verification path** for I1 computation
- **Early warning** of structural drift

### Implementation

**Files Modified:**
- `src/evaluate.py`

**Functions Updated:**
1. `compute_c_paper_operator_v2()` — core evaluator
2. `compute_operator_implied_weights()` — intermediate wrapper
3. `compute_c_paper_tex_mirror()` — top-level API

**New Parameter:**
```python
i1_source: str = "dsl"  # "dsl" (default) or "post_identity_operator"
```

### How It Works

When `i1_source="post_identity_operator"`:

```python
from src.operator_post_identity import compute_I1_operator_post_identity_pair

for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
    ell1, ell2 = int(pair_key[0]), int(pair_key[1])

    # Post-identity gives I1 at +R and -R
    result_plus = compute_I1_operator_post_identity_pair(theta, R, ell1, ell2, n, polys)
    result_minus = compute_I1_operator_post_identity_pair(theta, -R, ell1, ell2, n, polys)

    # Skip Q-lift for post_identity_operator (like direct_case_c does for I2)
    i1_minus_op_raw = i1_minus_base_raw
```

### Validation Results

**Per-Pair I1 Agreement (DSL vs Post-Identity Operator):**

| Pair | I1 (κ, R=1.3036) | I1 (κ*, R=1.1167) | Match |
|------|------------------|-------------------|-------|
| (1,1) | +0.4134741024 | +0.3719867125 | ✅ <1e-15 |
| (1,2) | -0.5681319521 | -0.4974137715 | ✅ <1e-15 |
| (1,3) | +0.0117784160 | +0.0101420508 | ✅ <1e-17 |
| (2,2) | +0.1608645355 | +0.1366798303 | ✅ <1e-16 |
| (2,3) | -0.0035626350 | -0.0029851019 | ✅ <1e-18 |
| (3,3) | +0.0000879161 | +0.0000728019 | ✅ <1e-19 |

All pairs match to machine precision (~10⁻¹⁵ to 10⁻¹⁹).

### Test File

**`tests/test_i1_source_gate.py`** — 21 tests

```
TestI1SourceAgreement::test_i1_sources_match_kappa[1-1] PASSED
TestI1SourceAgreement::test_i1_sources_match_kappa[1-2] PASSED
TestI1SourceAgreement::test_i1_sources_match_kappa[1-3] PASSED
TestI1SourceAgreement::test_i1_sources_match_kappa[2-2] PASSED
TestI1SourceAgreement::test_i1_sources_match_kappa[2-3] PASSED
TestI1SourceAgreement::test_i1_sources_match_kappa[3-3] PASSED
TestI1SourceAgreement::test_i1_sources_match_kappa_star[1-1] PASSED
... (12 total per-pair tests)
TestI1SourceConsistency::test_ordered_pairs_match_canonical_* (6 tests)
TestI1SourceRegressionSafety::test_default_dsl_unchanged_kappa PASSED
TestI1SourceRegressionSafety::test_post_identity_operator_runs_kappa PASSED
```

---

## Phase 2: S34 Mirror Gate (Step-3 Closure)

### ⚠️ RESOLUTION: Superseded by TRUTH_SPEC.md Section 10

**Date:** 2025-12-22

**TRUTH_SPEC.md Section 10 (lines 370-388) definitively resolves this question:**

```
The exact PRZZ formula producing published c:

From Section 6.2.1, after combining:
- I₁(α,β) + T^{-α-β}I₁(-β,-α) (mirror terms)  ← HAS MIRROR
- I₂(α,β) + T^{-α-β}I₂(-β,-α)                 ← HAS MIRROR
- I₃(α,β) and I₄(α,β)                          ← NO MIRROR
```

**I3/I4 are NON-MIRRORED by design.** The tests below now serve as **negative controls** — they prove that incorrectly adding mirror causes catastrophic overshoot.

---

### Original Motivation (Historical)

PRZZ TeX lines 1553-1570 suggest I3/I4 have mirror structure:

```
I₃ involves: (N^{αx} - T^{-α-β}N^{-βx}) / (α+β)
```

The current tex_mirror uses plus-only for S34. We needed to verify whether the mirror contribution is negligible or significant.

### Key Finding: External S34 Mirror Recombination is INVALID (Now Understood Why)

**An external S34 mirror recombination using the naive exp(2R) rule produces a catastrophic overshoot.**

| Benchmark | R | S34 Mirror Delta | % of c_target |
|-----------|------|-----------------|---------------|
| κ | 1.3036 | 7.493 | **350%** |
| κ* | 1.1167 | 3.525 | **182%** |

**Why this happens:** I3/I4 are fundamentally NON-MIRRORED in the TeX (per TRUTH_SPEC.md Section 10). Adding external mirror is **mathematically wrong**, not just numerically unstable.

**These tests are now NEGATIVE CONTROLS:** They prove that if someone accidentally adds mirror to I3/I4, the result blows up. This is the EXPECTED behavior when applying an incorrect transformation.

### Per-Pair S34 Mirror Breakdown (κ benchmark)

| Pair | S34_plus | S34_mirror | Delta | Delta % |
|------|----------|------------|-------|---------|
| (1,1) | -0.338 | 2.049 | 2.387 | 112% |
| (1,2) | -0.224 | 1.359 | 1.583 | 74% |
| (1,3) | -0.012 | 0.075 | 0.087 | 4% |
| (2,2) | -0.207 | 1.252 | 1.459 | 68% |
| (2,3) | -0.020 | 0.120 | 0.140 | 7% |
| (3,3) | -0.004 | 0.022 | 0.026 | 1% |

### Implications (Updated Post-TRUTH_SPEC Resolution)

1. **Current tex_mirror (plus-only) is CORRECT** — matches PRZZ TeX specification exactly
2. **The overshoot tests are NEGATIVE CONTROLS** — they verify that wrong mirror handling fails spectacularly
3. **I3/I4 non-mirrored is SPEC-LOCKED** — TRUTH_SPEC.md Section 10 is definitive
4. **No further investigation needed for K>3** — this question is resolved

### Why Plus-Only Works (Now Definitively Answered)

**TRUTH_SPEC.md Section 10 explicitly states I3/I4 have NO mirror structure.**

The plus-only approach is not an "empirical workaround" — it is the mathematically correct implementation of the PRZZ formula. The overshoot tests document what happens when you incorrectly try to add mirror.

### Test File (Reframed as Negative Controls)

**`tests/test_s34_mirror_gate.py`** — 16 tests

**Purpose:** These tests now serve as **negative controls**. They verify that:
- Incorrectly adding mirror to I3/I4 causes 350%/182% overshoot
- This overshoot is catastrophic and unmistakable
- The tests will FAIL if someone "fixes" the code to add I3/I4 mirroring

```
TestS34MirrorFinding::test_s34_mirror_significant_kappa PASSED     # Negative control
TestS34MirrorFinding::test_s34_mirror_significant_kappa_star PASSED # Negative control
TestS34MirrorPerPair::test_s34_finite_kappa[1-1..3-3] (6 tests) PASSED
TestS34MirrorPerPair::test_s34_finite_kappa_star[1-1..3-3] (6 tests) PASSED
TestS34MirrorStructure::test_mirror_sign_convention PASSED
TestS34MirrorStructure::test_delta_is_exp2R_times_minus PASSED
```

---

## Phase 3: m1 L-Sweep Diagnostic

### Motivation

The empirical formula `m1 = exp(R) + 5` (or `exp(R) + (2K-1)` for general K) was found through calibration, not derivation. This formula will explode at higher K if it's wrong.

The goal was to derive m1 from the combined identity via L-sweep.

### Method (Conceptual)

For each L in [10, 20, 50, 100]:
1. Compute `I1_combined(L)` from combined identity at finite L
2. Compute `I1+` (plus branch at +R) — L-independent
3. Compute `I1−base` (minus branch at -R) — L-independent
4. Solve: `m1_eff(L) = (I1_combined - I1+) / I1−base`

If `m1_eff(L)` converges as L → ∞, we have derived m1.

### CRITICAL FINDING: Finite-L Approach DIVERGES

**Date:** 2025-12-22

The finite-L combined identity approach was implemented (`src/combined_identity_finite_L.py`) and tested. **The approach DIVERGES linearly with L instead of converging.**

**L-Sweep Results (κ benchmark):**

| L | I1_combined | m1_eff |
|---|-------------|--------|
| 10 | -15.86 | -93.6 |
| 20 | -31.72 | -184.8 |
| 50 | -79.30 | -458.5 |
| 100 | -158.61 | -914.6 |

**Key Observation:** `m1_eff ≈ -9.15 × L` — grows linearly with L!

### Root Cause Analysis

At `α = β = -R/L`, the combined identity becomes:
```
L/(2R) × [exp(-Rθ(x+y)) - exp(2R)·exp(Rθ(x+y))]
```

**The problem:**
- The `1/(α+β) = -L/(2R)` prefactor grows **linearly** with L
- The bracket `[exp(-...) - exp(2R)×exp(+...)]` is **L-independent**
- Nothing cancels the L-dependence!

**Why post-identity works:** The post-identity approach computes the L→∞ limit **analytically** using asymptotic expansion, NOT by evaluating at finite L.

### Current Status

**Option A (finite-L combined identity) does NOT work.** The 2 gate tests for m1 convergence remain skipped.

**Files Created:**
- `src/combined_identity_finite_L.py` — Implementation of finite-L approach
- `tests/test_combined_identity_finite_L.py` — 14 tests validating the L-divergence

**Conclusion:** The empirical formula `m1 = exp(R) + 5` remains the working approach. A proper derivation would require asymptotic analysis of the combined identity, not finite-L evaluation.

---

### Phase 3B: Unified-t Combined Identity (GPT Guidance)

**Date:** 2025-12-22

Following GPT deep review of RMS_PRZZ.tex, implemented the "unified-t" combined identity approach.

**Key Insight:** The TeX combined identity's `t∈[0,1]` parameter is the SAME `t` appearing in Q affine arguments (`A_α = t + θ(t-1)x + θty`). Run 20 (TexCombinedMirrorCore) incorrectly introduced a SEPARATE `s`-integral.

**Implementation:**
- `src/combined_identity_unified_t.py` — Unified-t kernel implementation
- `tests/test_combined_identity_unified_t.py` — 31 tests passing

**Critical Finding: Unified-t WITHOUT log factor is IDENTICAL to post-identity.**

```
Without log factor:
  All pairs: ratio = 1.000000 (unified-t == post-identity)
  m1_eff = 0.0 (because unified == plus branch)
```

**With log factor (1+θ(x+y)) from log(N^{x+y}T):**

| Benchmark | I1_unified | I1_plus | m1_eff | m1_empirical | Ratio |
|-----------|------------|---------|--------|--------------|-------|
| κ | 0.5220 | 0.4135 | 0.62 | 8.68 | 7.2% |
| κ* | 0.4542 | 0.3601 | 0.55 | 8.05 | 6.8% |

**Per-pair scaling (with log factor):**
- Mean ratio: 1.24 (about 24% larger than post-identity)
- CV: 3.4% (consistent across pairs — GLOBAL_FACTOR diagnosis)

**Conclusion:**
1. The unified-t kernel (without log factor) is structurally identical to post-identity
2. Adding log factor doesn't explain m1 — only adds ~24% to I1, not 8-9× needed
3. **m1 = exp(R) + 5 remains empirical calibration**, not derived from TeX structure
4. The log factor may already be absorbed in the algebraic prefactor (1/θ + x + y)

**Test Results:** 31 passed (sanity checks 1-3, unified-t kernel, m1 derivation)

---

### What We Have Now

**Post-Identity Reference Values (L → ∞):**

| Benchmark | I1+ | I1−base | Empirical m1 |
|-----------|-----|---------|--------------|
| κ (R=1.3036) | 0.41347410 | 0.17387560 | 8.683 |
| κ* (R=1.1167) | 0.36006181 | 0.17127461 | 8.055 |

**Mirror Weight Comparison:**

| Formula | κ Value | κ* Value | Notes |
|---------|---------|----------|-------|
| Empirical: exp(R)+5 | 8.683 | 8.055 | Works empirically |
| Naive: exp(2R) | 13.561 | 9.332 | From combined identity |
| Ratio (naive/empirical) | 1.56 | 1.16 | Naive is larger |

The naive `exp(2R)` from the combined identity is significantly larger than the empirical `exp(R)+5`.

### Diagnostic Script

**`run_m1_from_combined_identity_L_sweep.py`**

```bash
python3 run_m1_from_combined_identity_L_sweep.py
```

Output shows structure and documents what needs to be implemented:
```
Goal: Derive m1 from combined identity, not calibration.
Method: Compute m1_eff(L) = (I1_combined - I1+) / I1−base
        and observe convergence as L → ∞

NOTE: compute_I1_combined_at_L is NOT YET IMPLEMENTED.
```

### Test File

**`tests/test_m1_eff_converges.py`** — 9 tests (7 pass, 2 skip)

```
TestM1EmpiricalFormula::test_m1_empirical_formula_kappa PASSED
TestM1EmpiricalFormula::test_m1_empirical_formula_kappa_star PASSED
TestPostIdentityReference::test_i1_plus_minus_finite_kappa PASSED
TestPostIdentityReference::test_i1_plus_minus_finite_kappa_star PASSED
TestNaiveMirrorWeight::test_naive_vs_empirical_kappa PASSED
TestNaiveMirrorWeight::test_naive_vs_empirical_kappa_star PASSED
TestM1EffConvergence::test_m1_eff_converges_kappa SKIPPED (not implemented)
TestM1EffConvergence::test_m1_eff_matches_empirical SKIPPED (not implemented)
TestM1StatusDocumentation::test_m1_is_calibrated_not_derived PASSED
```

---

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| `tests/test_i1_source_gate.py` | Gate tests for i1_source agreement (21 tests) |
| `tests/test_s34_mirror_gate.py` | Gate tests for S34 mirror finding (16 tests) |
| `tests/test_m1_eff_converges.py` | Gate tests for m1 derivation (9 tests, 2 skipped) |
| `tests/test_combined_identity_finite_L.py` | Tests for finite-L divergence (14 tests) |
| `tests/test_combined_identity_unified_t.py` | Tests for unified-t implementation (31 tests) |
| `src/combined_identity_finite_L.py` | Finite-L combined identity implementation |
| `src/combined_identity_unified_t.py` | Unified-t combined identity implementation |
| `run_m1_from_combined_identity_L_sweep.py` | L-sweep diagnostic script |
| `docs/K_SAFE_BASELINE_LOCKDOWN.md` | This document |

### Modified Files

| File | Changes |
|------|---------|
| `src/evaluate.py` | Added `i1_source` parameter to 3 functions |

---

## Summary: What This Gets You

After Phases 1-3B:

1. **Independent I1 validation** via post-identity operator (regression safety net)
2. **Step-3 documented** — S34 external mirror is INVALID (350%/182% overshoot proves wrong recombination)
3. **m1 status clarified** — both finite-L and unified-t approaches failed; m1 = exp(R)+5 remains empirical calibration
4. **Unified-t insight** — the correct TeX structure is IDENTICAL to post-identity (no new I1 values)
5. **Solid test base** — 89 new tests gating the baseline

---

## What Remains for K>3 Extension

### Prerequisites (must be addressed)

| Item | Status | Notes |
|------|--------|-------|
| I1 validated by two routes | ✅ Done | DSL and post-identity operator match |
| I2 proven + gated | ✅ Done | Run 7/8 (direct_case_c) |
| I3/I4 mirror resolved | ⚠️ Documented | External mirror invalid (350% overshoot); plus-only works |
| m1 derived or gated | ⚠️ Documented | L-sweep structure ready, needs finite-L impl |

### K=4 Checklist (only after above)

1. **Extend evaluator plumbing:**
   - Add K parameter (already exists in tex_mirror signature)
   - Add P4 loading + storage

2. **Define Case-C kernel for ω=3:**
   - Add to `src/mollifier_profiles.py`
   - Follow same pattern as ω=1, ω=2

3. **Add gates per new pair:**
   - Direct-vs-DSL for I1 and I2
   - Use gate strategy from Run 7-9

4. **DO NOT use terms_version="v2" with tex_mirror:**
   - evaluate.py has hard guard: V2 + tex_mirror is forbidden
   - Due to catastrophic sign flip under assembly

---

## Running the Tests

```bash
# Run all new gate tests
python3 -m pytest tests/test_i1_source_gate.py tests/test_s34_mirror_gate.py tests/test_m1_eff_converges.py -v

# Run golden diagnostic (validates post-identity operator)
python3 run_operator_post_identity_golden.py

# Run m1 L-sweep diagnostic
python3 run_m1_from_combined_identity_L_sweep.py
```

---

## Appendix: Mathematical Background

### Post-Identity Operator Approach

The post-identity exponential core:
```
E(α,β;x,y,t) = exp(θL(αx+βy)) · exp(-t(α+β)L(1+θ(x+y)))
```

Under `D_α = -1/L × d/dα`, we get eigenvalue form:
```
Q(D_α)Q(D_β)E = Q(A_α)Q(A_β)E
```

where the affine forms are:
```
A_α = t + θ(t-1)·x + θt·y
A_β = t + θt·x + θ(t-1)·y
```

The `(θt-θ)` cross-terms create the correct asymmetry.

### Combined Identity Structure

PRZZ combined identity (TeX lines 1502-1511):
```
(N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)
```

At `α = β = -R/L`:
```
= L/(2R) × (exp(-Rθ(x+y)) - exp(2R)·exp(Rθ(x+y)))
```

The naive mirror weight `exp(2R)` comes from `T^{-α-β}` evaluated at `α=β=-R/L`.

### S34 Mirror Formula

For I3/I4:
```
I3_combined = I3_plus + exp(2R) × I3_minus
I4_combined = I4_plus + exp(2R) × I4_minus
```

Delta from plus-only:
```
delta = exp(2R) × (I3_minus + I4_minus)
```

---

*Document created as part of GPT Phase 1-3 implementation for K-Safe PRZZ Baseline Lock-Down.*
