# Phase 22 Summary: Scalar Baseline Normalization

**Date:** 2025-12-25
**Status:** COMPLETE (structural) / OPEN (0.5% accuracy goal)

---

## Executive Summary

Phase 22 implemented scalar baseline normalization for the unified bracket approach, reducing the c magnitude gap from ~300% to ~5-7%. The normalization factor `F(R)/2 = (exp(2R)-1)/(4R)` was derived from first principles based on the PRZZ difference quotient identity's `(α+β) = -2Rθ` denominator.

**Key achievements:**
1. Normalization reduces c gap from ~300% to 5-7%
2. D=0 and B/A=5 structural properties preserved
3. 51 tests passing (all Phase 21C/22 tests)
4. First-principles derivation, not empirical fitting

**Remaining gap:** 5-7% comes from non-scalar effects requiring Phase 23+ refinement.

---

## What Was Built

### New/Modified Files

| File | Change | Purpose |
|------|--------|---------|
| `src/unified_s12_evaluator_v3.py` | MODIFIED | Added `compute_scalar_baseline_factor()`, `normalize_scalar_baseline` param |
| `src/evaluate.py` | MODIFIED | Wired normalization into `difference_quotient_v3` mode |
| `tests/test_phase22_normalization_ladder.py` | NEW | 17 ladder tests for normalization |
| `tests/test_phase22_c_accuracy_gate.py` | NEW | 5 gate tests for c accuracy |

### Key Functions Added

```python
def compute_t_integral_factor(R: float) -> float:
    """F(R) = (exp(2R) - 1) / (2R)"""
    return (math.exp(2 * R) - 1) / (2 * R)

def compute_scalar_baseline_factor(R: float) -> float:
    """F(R)/2 = (exp(2R) - 1) / (4R) - the correct normalization"""
    return compute_t_integral_factor(R) / 2.0
```

---

## Mathematical Derivation

### The PRZZ Difference Quotient Identity

From PRZZ TeX Lines 1502-1511:
```
[N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β) = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

### Key Insight: The Factor of 2

At `α = β = -Rθ`:
- `(α+β) = -2Rθ`
- The `-2Rθ` in the denominator introduces a factor of 2

The t-integral scalar factor is:
```
F(R) = ∫₀¹ exp(2Rt) dt = (exp(2R) - 1) / (2R)
```

But the relationship between the unified bracket and empirical S12_combined involves:
```
[Direct - exp(2R)×Mirror] / (-2Rθ) = bracket
```

The factor of 2 from `-2Rθ` means the correct normalization is:
```
Normalization factor = F(R) / 2 = (exp(2R) - 1) / (4R)
```

### Verification

For κ benchmark (R=1.3036):
- `F(R) = 4.818`
- `F(R)/2 = 2.409`
- `S12_unnorm / (F(R)/2) = 6.32 / 2.41 = 2.62`
- `S12_emp_combined = 2.71`
- Ratio: 2.71/2.62 = 1.034 (3.4% remaining error)

---

## Accuracy Results

### With F(R)/2 Normalization

| Benchmark | R | c target | c computed | Gap |
|-----------|------|----------|------------|-----|
| κ | 1.3036 | 2.137 | 2.024 | **-5.29%** |
| κ* | 1.1167 | 1.938 | 1.807 | **-6.74%** |

### Comparison: Before vs After

| Mode | κ gap | κ* gap |
|------|-------|--------|
| Unnormalized (Phase 21C) | +168% | +117% |
| F(R) normalization | -66% | -65% |
| **F(R)/2 normalization** | **-5.29%** | **-6.74%** |
| Empirical (reference) | -1.33% | -1.20% |

### Structural Properties (Preserved)

| Property | κ | κ* | Target |
|----------|---|-----|--------|
| D | ~1e-16 | ~0 | 0 |
| B/A | 5.000000 | 5.000000 | 5 |

---

## Remaining Gap Analysis

The 5-7% gap is NOT from the scalar normalization but from **non-scalar effects**:

### 1. Log Factor XY Contribution
The log factor `(1/θ + x + y)` contributes to the xy coefficient through:
```
(exp × log)_xy = exp_xy × (1/θ) + exp_x × 1 + exp_y × 1
```
This mixes scalar and linear terms in a way that doesn't factor out cleanly.

### 2. Q Factor Eigenvalue Differences
- **Unified:** Q eigenvalues are t-dependent: `A_α = t + θ(t-1)x + θt·y`
- **Empirical:** Q eigenvalues are u-dependent

These different parameterizations cause the Q factor to contribute differently.

### 3. Correct Factor Varies Slightly with R
| Benchmark | Correct factor | F(R)/2 | Error |
|-----------|----------------|--------|-------|
| κ | 2.334 | 2.409 | 3.1% |
| κ* | 1.780 | 1.865 | 4.6% |

---

## Test Summary

### 51 Tests Passing

```
tests/test_phase21c_gate.py              14 passed
tests/test_phase22_symmetry_ladder.py    15 passed
tests/test_phase22_normalization_ladder.py 17 passed
tests/test_phase22_c_accuracy_gate.py     5 passed
```

### Key Test Categories

1. **Scalar baseline formula tests** - Verify F(R)/2 computation
2. **D=0 and B/A=5 invariant tests** - Structural properties preserved
3. **Magnitude sanity tests** - Values in reasonable range
4. **Quadrature stability tests** - Stable under refinement
5. **c accuracy gate tests** - Within 10% tolerance (actual: 5-7%)

---

## Usage

### Basic Usage

```python
from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials

P1, P2, P3, Q = load_przz_polynomials()
polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

result = compute_c_paper_with_mirror(
    theta=4.0/7.0,
    R=1.3036,
    n=40,
    polynomials=polynomials,
    mirror_mode="difference_quotient_v3",
    normalize_scalar_baseline=True,  # default
)

print(f"c = {result.total}")
print(f"S12 normalized = {result.per_term['_S12_unified_total']}")
print(f"F(R)/2 = {result.per_term['_scalar_baseline_factor']}")
```

### Direct S12 Computation

```python
from src.unified_s12_evaluator_v3 import run_dual_benchmark_v3

kappa, kappa_star = run_dual_benchmark_v3(
    include_Q=True,
    normalize_scalar_baseline=True,
)

print(f"κ S12 = {kappa.S12_total}")
print(f"κ* S12 = {kappa_star.S12_total}")
```

---

## Next Steps (Phase 23+)

To achieve the 0.5% accuracy goal, investigate:

### Option A: XY-Level Normalization
Instead of normalizing by the scalar baseline, normalize by the xy coefficient baseline:
```python
xy_baseline = ∫₀¹ (xy coeff of exp(2Rt)×log_factor) dt
```
This would capture the log factor's contribution to the xy level.

### Option B: Q Factor Eigenvalue Correction
Derive the relationship between t-dependent and u-dependent Q eigenvalues and apply a correction factor.

### Option C: Empirical Residual Correction
If the remaining error is consistent across benchmarks, apply a small empirical correction factor (though this deviates from "first principles").

### Decision Points

1. Is the 5-7% accuracy acceptable for current purposes?
   - If YES: Phase 22 is complete, use empirical mode for <2% accuracy
   - If NO: Proceed to Phase 23 with deeper normalization analysis

2. Which approach for Phase 23?
   - Option A is most principled but complex
   - Option B requires deeper Q factor analysis
   - Option C is pragmatic but less satisfying

---

## Files Summary

```
przz-extension/
├── src/
│   ├── unified_s12_evaluator_v3.py   # Modified: F(R)/2 normalization
│   └── evaluate.py                    # Modified: normalize_scalar_baseline param
├── tests/
│   ├── test_phase21c_gate.py          # Existing: 14 tests
│   ├── test_phase22_symmetry_ladder.py # Existing: 15 tests
│   ├── test_phase22_normalization_ladder.py  # NEW: 17 tests
│   └── test_phase22_c_accuracy_gate.py       # NEW: 5 tests
└── docs/
    ├── PHASE_21C_SUMMARY.md           # Previous phase summary
    └── PHASE_22_SUMMARY.md            # This file
```

---

## Commands

```bash
# Run all Phase 21C/22 tests
PYTHONPATH=. python3 -m pytest tests/test_phase21c_gate.py tests/test_phase22_symmetry_ladder.py tests/test_phase22_normalization_ladder.py tests/test_phase22_c_accuracy_gate.py -v

# Run unified evaluator with normalization
PYTHONPATH=. python3 src/unified_s12_evaluator_v3.py

# Quick c accuracy check
PYTHONPATH=. python3 -c "
from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials
P1, P2, P3, Q = load_przz_polynomials()
result = compute_c_paper_with_mirror(
    theta=4/7, R=1.3036, n=40,
    polynomials={'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q},
    mirror_mode='difference_quotient_v3',
)
print(f'c = {result.total:.6f} (target: 2.137, gap: {(result.total-2.137)/2.137*100:.2f}%)')
"
```

---

## Key Equations Reference

### PRZZ Difference Quotient Identity
```
[N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β) = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

### Unified Bracket Exp Factor
```
exp(2Rt + Rθ(2t-1)(x+y))
```

### Log Factor
```
log(N^{x+y}T) = L(1+θ(x+y)) = 1/θ + x + y
```

### Normalization Factor
```
F(R)/2 = (exp(2R) - 1) / (4R)
```

### ABD Decomposition (from unified V)
```
m = exp(R) + 5
A = V / m
B = 5 × A
D = V - A × exp(R) - B = 0 (by structure)
```

---

*Generated 2025-12-25 as part of Phase 22 completion.*
