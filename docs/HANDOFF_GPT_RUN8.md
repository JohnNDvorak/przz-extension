# GPT Run 8 Handoff (2025-12-20)

## Overview

Run 8 implements two key deliverables:
- **Run 8A**: Locks in the Run 7 I2 proof by adding `i2_source` plumbing
- **Run 8B1**: Validates I1 for pair (1,1) directly, proving derivative extraction is correct

---

## Run 8A: I2 Source Plumbing

### The Fix: Add `i2_source` Parameter

Added `i2_source` parameter to:
- `compute_c_paper_operator_v2()` - core evaluator
- `compute_operator_implied_weights()` - wrapper
- `compute_c_operator_sigma_shift()` - wrapper

### Options

- `i2_source="dsl"` (default): Use DSL-based term evaluation
- `i2_source="direct_case_c"`: Use proven Case C kernel evaluation from Run 7

### Implementation

When `i2_source="direct_case_c"`:
```python
from src.case_c_kernel import compute_i2_all_pairs_case_c
direct_i2 = compute_i2_all_pairs_case_c(
    theta=theta, R=R, polynomials=polys_base, n=n, n_a=n_quad_a
)
```

The direct Case C evaluation matches DSL exactly (ratio = 1.0000) at both benchmarks.

### Gate Test Results

File: `tests/test_i2_source_gate.py` (6 tests)

| Test | Status |
|------|--------|
| test_i2_plus_equivalence_kappa | PASSED |
| test_i2_minus_base_equivalence_kappa | PASSED |
| test_i2_plus_equivalence_kappa_star | PASSED |
| test_i2_minus_base_equivalence_kappa_star | PASSED |
| test_per_pair_equivalence_kappa | PASSED |
| test_c_operator_equivalence_kappa | PASSED |

---

## Run 8B1: Direct I1(1,1) Validation

### The Goal

Validate that the DSL derivative extraction machinery produces correct results
by computing I1(1,1) directly using the series engine.

### Mathematical Structure

For pair (1,1), we compute:
```
I₁ = d²/dxdy |_{x=y=0} [∫∫ F(x, y, u, t) du dt]
```

The integrand F has structure:
```
F = (1/θ + x + y)(1-u)² × P₁(x+u)P₁(y+u)
    × Q(Arg_α)Q(Arg_β) × exp(R×Arg_α + R×Arg_β)
```

where:
- `Arg_α = t + θt·x + θ(t-1)·y`
- `Arg_β = t + θ(t-1)·x + θt·y`

### Why Case B is Simple

For pair (1,1), P₁ is Case B (ω=0), so no Case C kernel derivatives are needed:
- P₁(x+u) = P₁(u) + P₁'(u)·x + O(x²)

### Results

| Benchmark | DSL I1(1,1) | Direct I1(1,1) | Ratio |
|-----------|-------------|----------------|-------|
| κ (R=1.3036) | +0.41347410 | +0.41347410 | **1.000000** |
| κ* (R=1.1167) | +0.36006181 | +0.36006181 | **1.000000** |

**PERFECT MATCH** - The direct evaluation matches DSL exactly.

### Gate Test Results

File: `tests/test_direct_i1_11_gate.py` (4 tests)

| Test | Status |
|------|--------|
| test_i1_11_alignment_kappa | PASSED |
| test_i1_11_alignment_kappa_star | PASSED |
| test_i1_11_exact_match_kappa | PASSED |
| test_i1_11_exact_match_kappa_star | PASSED |

---

## Files Created/Modified in Run 8

| File | Action | Purpose |
|------|--------|---------|
| `src/case_c_kernel.py` | Modified | Added `compute_i2_all_pairs_case_c` vectorized function |
| `src/evaluate.py` | Modified | Added `i2_source` parameter to 3 functions |
| `tests/test_i2_source_gate.py` | Created | 6 gate tests for I2 source equivalence |
| `run_gpt_run8b1_direct_i1_11.py` | Created | Direct I1(1,1) validation script |
| `tests/test_direct_i1_11_gate.py` | Created | 4 gate tests for I1(1,1) equivalence |
| `docs/HANDOFF_GPT_RUN8.md` | Created | This document |

---

## Classification Update

### PROVEN (Run 7 + Run 8)

| Component | Evidence | Status |
|-----------|----------|--------|
| **I2 for ALL K=3 pairs** | Direct matches DSL (ratio=1.0) | **Proven (Run 7)** |
| **I1 for pair (1,1)** | Direct matches DSL (ratio=1.0) | **NEW: Proven (Run 8B1)** |
| `i2_source` plumbing | Gate tests pass | **Locked (Run 8A)** |

### Still Using DSL

| Component | Status |
|-----------|--------|
| I1 for pairs involving P2/P3 | Needs Case C kernel derivatives |
| I3, I4 terms | Needs investigation |

---

## What This Means

### I2 is Now a Regression Lock

With `i2_source="direct_case_c"`:
- Any future bug in the DSL would be caught by the gate tests
- The proven Case C kernel evaluation is the source of truth

### I1(1,1) is Validated

The derivative extraction machinery is proven correct for:
- Pair (1,1) with Case B polynomials
- This is the foundation for extending to Case C pairs

---

## Recommended Next Steps

### Option A: Extend I1 to Case C Pairs

For pairs involving P2/P3, need Case C kernel derivatives:
```
K'_ω(u; R) = d/darg K_ω(arg; R)|_{arg=u}
```

The formula is in `src/case_c_kernel.py:compute_case_c_kernel_derivative()`.

### Option B: Add i1_source Parameter

Similar to Run 8A, add `i1_source` parameter to allow switching between
DSL and direct evaluation for I1.

### Option C: Focus on Remaining Gaps

With I2 proven and I1(1,1) proven, focus on:
1. Understanding why I3/I4 might need similar treatment
2. Investigating the remaining c gap (2-3%)

---

## Test Suite Summary

All 21 Run 7/8 gate tests pass:
- 11 from Run 7 (I2 Case C)
- 6 from Run 8A (I2 source equivalence)
- 4 from Run 8B1 (I1(1,1) equivalence)

```
tests/test_direct_i2_caseC_gate.py: 11 passed
tests/test_i2_source_gate.py: 6 passed
tests/test_direct_i1_11_gate.py: 4 passed
============================== 21 passed ==============================
```

---

## GPT Run 8 Success Criteria

| Criterion | Status |
|-----------|--------|
| `i2_source="direct_case_c"` matches DSL | ✓ 6/6 tests PASSED |
| I1(1,1) direct matches DSL | ✓ 4/4 tests PASSED |
| All gate tests pass | ✓ 21/21 PASSED |
| Documentation updated | ✓ This file |

---

## Summary

**GPT Run 8 achieved:**
1. **I2 regression lock** - The proven Case C evaluation is now available via `i2_source="direct_case_c"`
2. **I1(1,1) validation** - The derivative extraction machinery is proven correct for Case B

This builds on Run 7's achievement (full I2 proof) and establishes a beachhead for
proving I1 from first principles by starting with the simplest pair (1,1).
