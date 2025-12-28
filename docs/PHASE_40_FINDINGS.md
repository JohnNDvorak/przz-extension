# Phase 40 Findings: Q Correction Analysis and K-Generic Infrastructure

**Date:** 2025-12-27
**Status:** COMPLETE
**Outcome:** Single analytical δ_Q formula NOT viable; K-generic infrastructure COMPLETE

---

## Executive Summary

Phase 40 investigated whether the ±0.15% residual in the derived formula could be reduced by adding an analytical δ_Q correction. The investigation found that:

1. **A single δ_Q formula cannot work** - the required corrections for κ and κ* have opposite signs
2. **The current ±0.15% accuracy is already excellent** - further refinement would require per-polynomial calibration
3. **K-generic infrastructure completed** - ready for K=4 without code path divergence

---

## Part A: Q Correction Investigation

### The Question

Can we derive δ_Q such that:
```
m = [1 + θ/(2K(2K+1)) + δ_Q] × [exp(R) + (2K-1)]
```
reduces the residual from ±0.15% to <±0.05%?

### Approach

1. **Computed Q moments** for both benchmarks:
   - ⟨Q²⟩, ⟨Q'²⟩, ⟨Q'² × t(t-1)⟩
   - exp(2Rt)-weighted versions

2. **Measured empirical δ_Q** from frozen-Q experiment:
   - Q derivative effect on correction ratio

3. **Computed required δ_Q** to close the c gap exactly

### Key Findings

#### Q Derivative Effect (from Frozen-Q Experiment)

| Benchmark | Q Effect on Ratio | Empirical δ_Q |
|-----------|-------------------|---------------|
| κ | -0.47% | -0.00477 |
| κ* | -1.51% | -0.01532 |

Both are negative, consistent with Q' reducing the correction.

#### Required δ_Q to Close C Gap

| Benchmark | c_gap | δ_Q Needed |
|-----------|-------|------------|
| κ | -0.14% | **+0.00153** |
| κ* | +0.02% | **-0.00018** |

**Critical Finding:** The required δ_Q values have **opposite signs**.

### Why a Single Formula Fails

1. **Q derivative effect ≠ Required correction**
   - Frozen-Q measures how Q derivatives affect the ratio
   - This is NOT the same as what's needed to match the target c

2. **Different polynomial structures**
   - κ: Q has degree 5, complex structure
   - κ*: Q has degree 1, simple linear structure
   - The interaction with P polynomials differs fundamentally

3. **No universal λ**
   - Fitting λ from κ gives 0.012
   - Fitting λ from κ* gives 0.047
   - 4× difference, cannot reconcile

### Conclusion

The ±0.15% residual is **irreducible without per-polynomial calibration**. The derived formula:
```
m = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]
```
is the correct first-principles formula. The residual comes from:
1. Linearization approximations in the derivation
2. Q polynomial structure variations
3. P polynomial interactions

**Recommendation:** Accept current accuracy. It's already 10× better than the empirical formula.

---

## Part B: K-Generic Infrastructure

### Completed Components

#### 1. K-Generic Pairs Module
**File:** `src/evaluator/pairs.py`

Functions:
- `get_triangle_pairs(K)` - Returns all (ℓ₁, ℓ₂) pairs
- `pair_count(K)` - Returns K(K+1)/2
- `factorial_norm(l1, l2)` - Returns 1/(ℓ₁! × ℓ₂!)
- `symmetry_factor(l1, l2)` - Returns 2 for off-diagonal, 1 for diagonal
- `full_norm(l1, l2)` - Returns combined normalization
- `validate_k_pairs(K)` - Validates pair generation

#### 2. K-Generic Pairs Tests
**File:** `tests/test_pairs_k_generic.py`

26 tests covering:
- Pair count formula (triangular numbers)
- Pair generation for K=3,4,5
- Factorial normalization
- Symmetry factors
- K=4 specific norms (matching K4_IMPLEMENTATION_PLAN.md)

All tests pass.

### GPT Guidance Integration

From GPT's review, the following improvements were incorporated:

1. **Step 0 Added** - P=Q=1 kernel-only sanity check before Steps 1-3
2. **Dual Gate Thresholds**:
   - Safety: |Q_effect| < 5%
   - Precision: |Q_effect| < 0.5%
3. **K-Generic Design** - `pairs.py` instead of `terms_k4_d1.py` fork
4. **Pair Count Gate Test** - Validates K(K+1)/2 pairs with correct norms

---

## Q Correction Module

**File:** `src/diagnostics/q_correction_formula.py`

While a universal δ_Q formula wasn't found, the module provides:

1. **Q Moment Computation**
   - ⟨Q²⟩, ⟨Q'²⟩, ⟨Q'² × t(t-1)⟩
   - R-weighted versions

2. **Empirical δ_Q Computation**
   - From frozen-Q experiment
   - Measures actual Q derivative effect

3. **Diagnostic Reports**
   - Detailed breakdown of Q effects
   - Comparison of empirical vs analytical

---

## Files Created

| File | Purpose |
|------|---------|
| `src/diagnostics/q_correction_formula.py` | Q moment and δ_Q computation |
| `src/evaluator/pairs.py` | K-generic pairs and normalization |
| `tests/test_pairs_k_generic.py` | K-generic pairs tests (26 tests) |
| `docs/PHASE_40_FINDINGS.md` | This document |

---

## Validation Status

### Q Correction
- ❌ Single analytical formula NOT viable
- ✅ Mechanism understood (Q derivatives + polynomial interactions)
- ✅ Current ±0.15% accuracy accepted as production-ready

### K-Generic Infrastructure
- ✅ `pairs.py` created and tested
- ✅ 26 tests passing
- ✅ K=4 pair count and norms validated
- ✅ Ready for K=4 implementation when polynomials available

---

## Next Steps

1. **K=4 Polynomials** - When available, run microcase ladder (Steps 0-3)
2. **K-Generic Terms** - Extend `build_terms.py` to use `pairs.py`
3. **Production Use** - Current formula is ready for κ optimization

---

## Appendix: Numerical Details

### Q Moments (κ benchmark, R=1.3036)

| Moment | Value |
|--------|-------|
| ⟨Q²⟩ | 0.3436 |
| ⟨Q'²⟩ | 1.0804 |
| ⟨Q'² × t(t-1)⟩ | -0.2096 |
| ⟨Q² × exp(2Rt)⟩ | 0.7163 |
| ⟨Q'² × t(t-1) × exp(2Rt)⟩ | -0.8714 |
| Weighted moment ratio | -1.2165 |

### Q Moments (κ* benchmark, R=1.1167)

| Moment | Value |
|--------|-------|
| ⟨Q²⟩ | 0.3229 |
| ⟨Q'²⟩ | 1.0659 |
| ⟨Q'² × t(t-1)⟩ | -0.1777 |
| ⟨Q² × exp(2Rt)⟩ | 0.6117 |
| ⟨Q'² × t(t-1) × exp(2Rt)⟩ | -0.6135 |
| Weighted moment ratio | -1.0028 |

### Decomposition (κ benchmark)

| Component | Value |
|-----------|-------|
| S12+ | 0.7975 |
| S12- | 0.2201 |
| S34 | -0.6002 |
| m_current | 8.8007 |
| c_current | 2.1345 |
| c_target | 2.1375 |
| c_gap | -0.14% |
