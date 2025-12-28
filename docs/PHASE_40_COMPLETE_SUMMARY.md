# Phase 40 Complete Summary: Q Correction Analysis & K-Generic Infrastructure

**Date:** 2025-12-27
**Author:** Claude Code (Phase 40 Implementation)
**Status:** COMPLETE

---

## Executive Summary

Phase 40 had two objectives:
1. **Investigate the ±0.15% residual** - Can we derive an analytical δ_Q correction to reduce it?
2. **Build K-generic infrastructure** - Prepare for K=4 without code path divergence

### Key Results

| Objective | Outcome |
|-----------|---------|
| Single δ_Q formula | ❌ **NOT VIABLE** - required corrections have opposite signs |
| Current accuracy | ✅ **±0.15% is production-ready** - already 10× better than empirical |
| K-generic pairs | ✅ **COMPLETE** - `pairs.py` with 26 tests |
| K=4 preparation | ✅ **READY** - plan updated with GPT guidance |

---

## Part A: Q Correction Investigation

### The Question

The derived mirror multiplier formula:
```
m = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]
```

has a ±0.15% residual on the κ and κ* benchmarks. Can we add a δ_Q correction to reduce this?

### Investigation Approach

1. **Created Q moment computation module** (`src/diagnostics/q_correction_formula.py`)
   - Computes ⟨Q²⟩, ⟨Q'²⟩, ⟨Q'² × t(t-1)⟩
   - Computes exp(2Rt)-weighted versions
   - Measures effective δ_Q from frozen-Q experiment

2. **Computed empirical δ_Q** from frozen-Q ratio comparison
   - κ benchmark: Q effect on ratio = -0.47%
   - κ* benchmark: Q effect on ratio = -1.51%

3. **Computed required δ_Q** to close the c gap exactly
   - κ: needs δ_Q = **+0.15%** (increase correction)
   - κ*: needs δ_Q = **-0.02%** (decrease correction)

### Critical Finding: Opposite Signs

The required δ_Q values have **opposite signs**:

| Benchmark | c_gap | Required δ_Q | Direction |
|-----------|-------|--------------|-----------|
| κ (R=1.3036) | -0.14% | +0.00153 | INCREASE m |
| κ* (R=1.1167) | +0.02% | -0.00018 | DECREASE m |

This means **no single analytical formula can work for both benchmarks**.

### Why the Analytical Approach Failed

1. **Q derivative effect ≠ Required correction**
   - The frozen-Q experiment measures how Q derivatives affect the ratio
   - This is NOT the same as what's needed to match the target c

2. **Different polynomial structures**
   - κ: Q has degree 5 with complex structure
   - κ*: Q has degree 1 (simple linear)
   - The interaction with P polynomials differs fundamentally

3. **Fitting λ gives inconsistent values**
   - From κ: λ = 0.012
   - From κ*: λ = 0.047
   - 4× difference, cannot reconcile

### Q Moments Computed

#### κ Benchmark (R=1.3036)

| Moment | Value |
|--------|-------|
| ⟨Q²⟩ | 0.3436 |
| ⟨Q'²⟩ | 1.0804 |
| ⟨Q'² × t(t-1)⟩ | -0.2096 |
| ⟨Q² × exp(2Rt)⟩ | 0.7163 |
| ⟨Q'² × t(t-1) × exp(2Rt)⟩ | -0.8714 |
| Weighted moment ratio | -1.2165 |

#### κ* Benchmark (R=1.1167)

| Moment | Value |
|--------|-------|
| ⟨Q²⟩ | 0.3229 |
| ⟨Q'²⟩ | 1.0659 |
| ⟨Q'² × t(t-1)⟩ | -0.1777 |
| ⟨Q² × exp(2Rt)⟩ | 0.6117 |
| ⟨Q'² × t(t-1) × exp(2Rt)⟩ | -0.6135 |
| Weighted moment ratio | -1.0028 |

### Conclusion: Accept Current Accuracy

The ±0.15% residual is **irreducible without per-polynomial calibration**. The current formula is:
- Fully derived from first principles
- Already 10× better than the empirical formula
- Sufficient for κ optimization goals

---

## Part B: K-Generic Infrastructure

### The Problem

GPT's guidance warned against creating `terms_k4_d1.py` as a fork of `terms_k3_d1.py`. This leads to:
- Code path divergence
- Silent semantic drift
- Technical debt (6,700-line evaluate.py style)

### The Solution

Created `src/evaluator/pairs.py` with K-generic functions that work for any K:

```python
from src.evaluator.pairs import (
    get_triangle_pairs,    # Returns all (ℓ₁, ℓ₂) pairs
    pair_count,            # Returns K(K+1)/2
    factorial_norm,        # Returns 1/(ℓ₁! × ℓ₂!)
    symmetry_factor,       # Returns 2 for off-diagonal, 1 for diagonal
    full_norm,             # Returns combined normalization
    validate_k_pairs,      # Validates pair generation
)
```

### Pair Counts by K

| K | Pairs | Count |
|---|-------|-------|
| 2 | (1,1), (1,2), (2,2) | 3 |
| 3 | (1,1), (1,2), (1,3), (2,2), (2,3), (3,3) | 6 |
| 4 | (1,1), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,3), (3,4), (4,4) | 10 |
| 5 | ... | 15 |

### K=4 Factorial Normalizations

| Pair | 1/(ℓ₁! × ℓ₂!) |
|------|---------------|
| (1,1) | 1.0 |
| (1,2) | 0.5 |
| (1,3) | 1/6 ≈ 0.1667 |
| (1,4) | 1/24 ≈ 0.0417 |
| (2,2) | 0.25 |
| (2,3) | 1/12 ≈ 0.0833 |
| (2,4) | 1/48 ≈ 0.0208 |
| (3,3) | 1/36 ≈ 0.0278 |
| (3,4) | 1/144 ≈ 0.0069 |
| (4,4) | 1/576 ≈ 0.0017 |

### Tests Created

`tests/test_pairs_k_generic.py` with 26 tests:

- **TestPairCount** (4 tests): Validates K(K+1)/2 formula
- **TestPairGeneration** (3 tests): Validates correct pairs for K=3,4
- **TestFactorialNorm** (7 tests): Validates 1/(ℓ₁! × ℓ₂!)
- **TestSymmetryFactor** (2 tests): Validates 1 for diagonal, 2 for off-diagonal
- **TestPairKey** (2 tests): Validates string key generation
- **TestFullNorm** (1 test): Validates combined normalization
- **TestGetAllNorms** (2 tests): Validates bulk normalization
- **TestValidation** (4 tests): Validates pair validation function
- **TestK4SpecificNorms** (1 test): Validates K=4 norms match plan

---

## GPT Guidance Integration

### K=4 Microcase Ladder (Updated)

GPT recommended adding **Step 0** before Steps 1-3:

| Step | Configuration | Purpose | Expected |
|------|---------------|---------|----------|
| **0** | P=Q=1 | Kernel-only sanity | Beta = 1.00794 cleanly |
| 1 | P=real, Q=1 | Validate Beta | Ratio ≈ 1.00794 |
| 2 | P=1, Q=real | Isolate Q effect | Negative, bounded |
| 3 | P=real, Q=real | Production | All gates pass |

### Dual Gate Thresholds

GPT recommended replacing the single 2% threshold with dual gates:

| Gate | Threshold | Purpose |
|------|-----------|---------|
| **Safety** | `abs(Q_effect) < 5%` | Prevents blow-ups / sign catastrophe |
| **Precision** | `abs(Q_effect) < 0.5%` | Keeps near K=3 regime |

### K-Generic Design

Instead of forking `terms_k3_d1.py` → `terms_k4_d1.py`:
- Use `pairs.py` for K-generic pair generation
- Drive both K=3 and K=4 with same machinery
- Only polynomial objects change

### Important Caveat

Phase 39 validated K=4 structure using **K=3 polynomials as proxy**. This is useful structurally but NOT a substitute for real K=4 polynomials. "Production-ready" is conditional on real K=4 polynomials passing Steps 0-3.

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/diagnostics/q_correction_formula.py` | ~420 | Q moments, δ_Q computation |
| `src/evaluator/pairs.py` | ~180 | K-generic pairs and normalization |
| `tests/test_pairs_k_generic.py` | ~200 | 26 tests for K-generic pairs |
| `docs/PHASE_40_FINDINGS.md` | ~150 | Detailed findings document |
| `docs/PHASE_40_COMPLETE_SUMMARY.md` | This file | Complete summary |

## Files Modified

| File | Changes |
|------|---------|
| `docs/K4_IMPLEMENTATION_PLAN.md` | Added Step 0, dual gates, K-generic design, updated references |

---

## Test Results

### All 46 Tests Pass

```
tests/test_pairs_k_generic.py ................ 26 passed
tests/test_q_residual_gates.py ............... 9 passed
tests/test_mirror_formula_locked.py .......... 11 passed
======================== 46 passed in 171.79s ========================
```

### Test Breakdown

| Test File | Tests | Purpose |
|-----------|-------|---------|
| `test_pairs_k_generic.py` | 26 | K-generic pairs validation |
| `test_q_residual_gates.py` | 9 | Q residual gates for κ and κ* |
| `test_mirror_formula_locked.py` | 11 | Derived formula K-dependence |

---

## Current Accuracy

### κ Benchmark (R=1.3036)

| Metric | Value |
|--------|-------|
| c_target | 2.137454 |
| c_computed | 2.134533 |
| c_gap | **-0.14%** |
| κ_target | 0.417294 |
| κ_computed | ~0.4175 |

### κ* Benchmark (R=1.1167)

| Metric | Value |
|--------|-------|
| c_target | 1.938000 |
| c_computed | 1.938306 |
| c_gap | **+0.02%** |

### Formula

```
m(K, R) = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]
```

For K=3, θ=4/7:
- Beta correction: 1 + θ/42 = 1.01361
- Base: exp(R) + 5

For K=4, θ=4/7:
- Beta correction: 1 + θ/72 = 1.00794
- Base: exp(R) + 7

---

## Decomposition Values (κ Benchmark)

| Component | Value |
|-----------|-------|
| S12+ | 0.7975 |
| S12- | 0.2201 |
| S34 | -0.6002 |
| m | 8.8007 |
| c | 2.1345 |

Assembly: `c = S12+ + m × S12- + S34`

---

## What's Ready for K=4

When K=4 polynomials become available:

1. **Run microcase ladder** (Steps 0-3)
2. **Use `pairs.py`** for K-generic pair generation
3. **Check dual gates** (safety <5%, precision <0.5%)
4. **Validate pair count** = 10 with correct norms
5. **If all pass** → K=4 is production-ready

---

## Lessons Learned

### 1. Analytical δ_Q is Benchmark-Dependent

The residual cannot be eliminated with a single analytical formula because:
- Different Q polynomial structures affect the integrand differently
- The required corrections have opposite signs for κ vs κ*
- Per-polynomial calibration would be needed, defeating first-principles goal

### 2. ±0.15% is Excellent Accuracy

The derived formula is already:
- 10× better than empirical (from ~1.3% to ~0.15%)
- Sufficient for κ optimization goals
- Based on first principles, not curve fitting

### 3. K-Generic Design Prevents Debt

Creating K-specific code forks leads to:
- Semantic drift between K=3 and K=4 paths
- Maintenance burden
- Hidden bugs when logic diverges

Using K-generic functions (`pairs.py`) ensures:
- Single source of truth
- Consistent behavior for all K
- Easier testing and validation

### 4. Microcase Ladder is Essential

The Step 0-1-2-3 progression:
- Isolates plumbing bugs (Step 0)
- Validates Beta correction (Step 1)
- Isolates Q effects (Step 2)
- Only then tests full production (Step 3)

This prevents misdiagnosis and wasted debugging time.

---

## References

- **Phase 36**: Derived formula locked, Q residual diagnostic
- **Phase 37**: Frozen-Q experiment, I1/I2 split discovery
- **Phase 38**: Q moment analysis
- **Phase 39**: K=4 safety check, K-sweep validation
- **Phase 40**: Q correction analysis, K-generic infrastructure
- **GPT Guidance**: Step 0, dual gates, K-generic design

---

## Appendix: Module APIs

### q_correction_formula.py

```python
# Compute Q moments
moments = compute_q_moments(Q_coeffs, R, n_quad=100)
# Returns: QMoments dataclass with all moment values

# Compute effective δ_Q from frozen-Q experiment
delta_Q = compute_effective_delta_Q(R, theta, K, polynomials)
# Returns: float (typically ~-0.005 to -0.015)

# Compute δ_Q with full diagnostics
result = compute_delta_Q_with_diagnostics(R, theta, K, polynomials, method="empirical")
# Returns: DeltaQResult with δ_Q and diagnostic info
```

### pairs.py

```python
# Get all pairs for K
pairs = get_triangle_pairs(K)  # [(1,1), (1,2), ...]

# Get pair count
count = pair_count(K)  # K(K+1)/2

# Get normalization
norm = factorial_norm(l1, l2)  # 1/(l1! * l2!)
sym = symmetry_factor(l1, l2)  # 2 if l1 < l2 else 1
full = full_norm(l1, l2)  # norm * sym

# Get all norms as dict
norms = get_all_norms(K)  # {"11": 1.0, "12": 1.0, ...}

# Validate
valid = validate_k_pairs(K)  # True if all checks pass
```

---

*End of Phase 40 Summary*
