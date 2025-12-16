# Session Transition Document
**Date:** 2025-12-15
**Purpose:** Context handoff for continuing PRZZ Assembly Audit

---

## Executive Summary

We are investigating a ~10% gap between computed c (~1.95) and PRZZ target c (~2.137) after fixing the I3/I4 prefactor from -1 to the paper-correct -1/θ. A comprehensive audit plan has been created and approved, but implementation has not yet begun.

---

## Current State

### What Was Discovered
1. **I3/I4 Prefactor Fix**: Changed from -1 to -1/θ (paper-correct per PRZZ lines 1562-1563)
   - With -1 prefactor + I5: c = 2.138, κ = 0.417 (matched PRZZ target)
   - With -1/θ prefactor: c = 1.950, κ = 0.488 (impossible - above known barriers)

2. **Factor 1+θ/6 Pattern**: Found that c_computed × (1 + θ/6) ≈ c_target with ~99.9% accuracy
   - This is a CLUE, not a FIX - indicates systematic structural issue

3. **Lower Derivative Terms Non-Zero**: g(0,0), g_x(0,0), g_y(0,0) are non-zero
   - These should cancel somewhere in PRZZ's analytic combination
   - Missing mirror term structure: I(α,β) + T^{-α-β}·I(-β,-α)

4. **Paper Integrator Confirms DSL**: src/paper_integrator.py matches DSL exactly
   - The -1/θ prefactor IS mathematically correct
   - The gap is in global assembly, not local term formulas

### Key Files Modified (Not Committed)
- `src/terms_k3_d1.py` - Changed 12 I3/I4 prefactors from -1 to -1/θ
- `tests/test_terms_k3_d1.py` - Updated prefactor tests
- `tests/test_evaluate.py` - Added 2 xfail golden tests
- `tests/test_i5_validation.py` - Added 1 xfail test
- `tests/test_bracket_weights.py` - Updated prefactor expectations
- `CLAUDE.md` - Updated status to "INVESTIGATING"

### New Files (Untracked)
- `src/paper_integrator.py` - Direct PRZZ equation implementation for validation
- `tests/test_paper_integrator.py` - Tests for paper integrator

### Git Status
```
On branch main
Your branch is ahead of 'origin/main' by 2 commits.
Modified (unstaged): 6 files
Untracked: 2 files
```

---

## Approved Plan Location

**Plan file:** `/Users/john.n.dvorak/.claude/plans/iterative-squishing-acorn.md`

### Plan Summary (7 Steps)

1. **Step 0: Stop the Bleeding**
   - Freeze branch/tag: `paper-prefactor-audit-start`
   - Make `mode="main"` the default in evaluate.py
   - xfail tests already appropriately marked

2. **Step 1: TRUTH_SPEC.md**
   - Single source of truth for PRZZ equations
   - Pinpoint exactly which formula produces published c
   - Document normalizations, polynomial mappings

3. **Step 2: Finite-Difference Oracle**
   - Independent validation that DSL matches PRZZ at correct variable stage
   - Must respect variable scaling (is x raw or scaled by log(N)?)
   - Test convergence in both h (step size) and n (quadrature)

4. **Step 3: Pair-Local Gap Attribution** (HIGHEST VALUE)
   - Print per-pair I1+I2+I3+I4 contributions
   - Determine if gap is in P3-involving pairs (Case C issue) or evenly spread (global assembly)

5. **Step 4: ω-Case Mapping Proof**
   - PROVE piece ↔ ℓ₁ mapping from PRZZ mollifier definition
   - P1: ℓ₁=? → ω=? → Case?
   - P2: ℓ₁=? → ω=? → Case?
   - P3: ℓ₁=? → ω=? → Case?

6. **Step 5: Mirror/Analytic Combination Checker**
   - Verify PRZZ's I(α,β) + T^{-α-β}·I(-β,-α) structure
   - Check where lower-derivative terms cancel

7. **Step 6-7: Implement Missing Structure & Re-enable Tests**
   - Only after Steps 3-5 identify the issue
   - No calibration - every factor must trace to PRZZ equation

---

## Critical GPT Corrections Incorporated

### Fix A: Beta-Function Numerology Was WRONG
- Previous plan claimed B(1,3) = 1/6. Actually B(1,3) = ∫₀¹(1-a)²da = 1/3
- Remove specific Beta values until exact exponents confirmed from PRZZ

### Fix B: ω/Case Mapping Must Be PROVEN
- Our piece index is NOT automatically PRZZ's ℓ₁
- Need explicit PRZZ reference for mapping

### Fix C: FD Oracle Must Respect Variable Scaling
- PRZZ may use scaled variables (x → x·log(N))
- Oracle test must use same stage as PRZZ formula

### Fix D: Don't Enforce mode=main Until TRUTH_SPEC Pins Target
- "Main" means PRZZ's main, not "I1-I4 in our code"
- Must identify which PRZZ formula produces published c

---

## What NOT To Do (From Plan)

1. DO NOT multiply c by (1 + θ/6) to match target
2. DO NOT use I₅ correction to hit κ
3. DO NOT revert to -1 prefactor without FD validation
4. DO NOT conclude "PRZZ had same bug" without full audit
5. DO NOT declare golden tests passing until assembly verified
6. DO NOT use Beta function numerology without PRZZ verification

---

## Key Technical Values

```
PRZZ Config 1:
  θ = 4/7 ≈ 0.5714285714285714
  R = 1.3036
  c_target = 2.13745440613217263636
  κ_target = 0.417293962

Current Computed (with -1/θ prefactor, no I5):
  c ≈ 1.950
  κ ≈ 0.488 (impossible - diagnostic siren)

Factor relationship:
  c_target / c_computed ≈ 1 + θ/6 ≈ 1.0952
```

---

## Likely Root Causes (from GPT Analysis)

### Gap Type 1: Analytic Combination / Mirror Structure
PRZZ forms I(α,β) + T^{-α-β}·I(-β,-α) BEFORE extracting constants.
If we expand separately and add → lose cancellations, different constant.
This explains why lower derivative terms look like divergence.

### Gap Type 2: ω-Case / Auxiliary a-Integral
PRZZ's ω-case split introduces auxiliary integral for ω>0 (Case C).
If our I1-I4 formulas implement intermediate stage without a-integral → systematic missing contribution.

---

## Files Summary

### Core Implementation
| File | Status | Purpose |
|------|--------|---------|
| `src/terms_k3_d1.py` | Modified | K=3 term definitions with -1/θ prefactor |
| `src/evaluate.py` | Current | Evaluation pipeline (needs mode="main") |
| `src/paper_integrator.py` | New | Direct PRZZ equation implementation |

### Tests
| File | Status | Notes |
|------|--------|-------|
| `tests/test_evaluate.py` | 2 xfail | Golden target tests |
| `tests/test_i5_validation.py` | 1 xfail | I5 formula test |
| Others | Passing | 436 tests pass |

### Documentation
| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project guidelines, status: INVESTIGATING |
| `TECHNICAL_ANALYSIS.md` | Mathematical derivations |
| Plan file | Approved audit plan |

---

## Next Session Instructions

1. **Start by running tests** to verify current state:
   ```bash
   python -m pytest tests/ -q
   ```
   Expected: ~439 passing, 3 xfail

2. **Execute Step 0**: Tag current state, make mode="main" default
   ```bash
   git add .
   git commit -m "I3/I4 prefactor fix: -1 → -1/θ (paper-correct, investigating gap)"
   git tag paper-prefactor-audit-start
   ```

3. **Execute Step 1**: Create docs/TRUTH_SPEC.md with PRZZ equation references

4. **Execute Step 3**: Run pair-local gap attribution (highest value diagnostic)
   - This tells us whether to focus on Case C or mirror assembly

5. Continue through plan Steps 2-7

---

## Contact Points in PRZZ Paper

- Lines 1562-1563: I₃ prefactor formula (1+θx)/θ → 1/θ at x=0
- Lines 1626-1628: I₅ is O(T/L), lower order
- Section on ω-cases: Cases A (ω=-1), B (ω=0), C (ω>0)
- Mirror term assembly: I(α,β) + T^{-α-β}·I(-β,-α)

---

## Summary for New Session

**What:** PRZZ κ reproduction has ~10% gap after prefactor fix
**Why:** Missing global assembly structure (mirror terms? Case C a-integral?)
**How:** Follow 7-step audit plan - find root cause, don't calibrate
**Key insight:** 1+θ/6 factor is a CLUE pointing to systematic structural issue
**Plan status:** Approved, Step 0 in_progress, not yet executed
