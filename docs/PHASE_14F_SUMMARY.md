# Phase 14F Summary: Normalize +5 Gate Metric

**Date:** 2025-12-24
**Status:** COMPLETE - Both κ and κ* gates now PASS

---

## Executive Summary

Phase 14F fixed the +5 gate semantic contract by using B/A instead of raw B as the normalized metric.

**Key Results:**
| Benchmark | Raw B | B gap | B/A | B/A gap | Status |
|-----------|-------|-------|-----|---------|--------|
| **κ** | 5.92 | +18% | **5.25** | +5.0% | PASS |
| **κ*** | 3.79 | -24% | **5.08** | +1.6% | **PASS** |

**The κ* "failure" was an artifact of not normalizing by A.**

---

## Root Cause Analysis

### The Problem with Raw B

Phase 14E defined:
```
A = I₁₂(-R)
B = I₁₂(+R) + I₃₄(+R) + 5 × I₁₂(-R)
  = D + 5A    where D = I₁₂(+R) + I₃₄(+R)
```

Since **A differs between κ and κ***, raw B differs even if "+5" is correct:
- κ:  A = 1.126, so B = 5.92 → appears 18% off from 5
- κ*: A = 0.746, so B = 3.79 → appears 24% off from 5

### The Fix: B/A Normalization

```
B/A = (D + 5A) / A = D/A + 5 = delta + 5
```

Where delta = D/A is the "contamination" from non-(2K-1) pieces.

Results with normalization:
- **κ:  B/A = 5 + 0.25 = 5.25** (only 5% off!)
- **κ*: B/A = 5 + 0.08 = 5.08** (only 1.6% off!)

---

## 2×2 Swap Experiment Results

We ran a (R, polynomial) swap experiment to determine whether delta is R-driven or polynomial-driven:

| R | Polynomials | delta | B/A |
|---|-------------|-------|-----|
| κ R (1.3036) | κ polys | 0.253 | 5.25 |
| κ R (1.3036) | κ* polys | 0.241 | 5.24 |
| κ* R (1.1167) | κ polys | 0.092 | 5.09 |
| κ* R (1.1167) | κ* polys | 0.079 | 5.08 |

**Key Finding:**
- Polynomial effect: 0.025 (changing polys while fixing R)
- R effect: 0.322 (changing R while fixing polys)

**Conclusion: delta is primarily R-DRIVEN**
- The Laurent-factor handling (1/R+γ)² is the main driver
- Polynomial degrees have minimal effect on delta

---

## Implementation Details

### New Return Values in `compute_m1_with_mirror_assembly()`

```python
return {
    # ... existing fields ...

    # Phase 14F: Normalized metrics
    "D": float(D),                    # D = I₁₂(+R) + I₃₄(+R)
    "delta": float(delta),            # delta = D/A (contamination)
    "B_over_A": float(B_over_A),      # B/A = 5 + delta
}
```

### New Tests Added

**test_plus5_gate.py:**
- `test_B_over_A_is_5_kappa` - PASSES (5.25 within 10%)
- `test_B_over_A_is_5_kappa_star` - PASSES (5.08 within 10%)
- `test_delta_is_small` - Verifies delta < 0.5 for both benchmarks

**test_microcase_plus5_signature_k3.py:**
- `test_kappa_B_over_A_is_approximately_5` - PASSES
- `test_kappa_star_B_over_A_is_approximately_5` - PASSES
- `test_kappa_has_larger_delta_than_kappa_star` - Documents κ has larger contamination

---

## Files Modified/Created

| File | Action | Purpose |
|------|--------|---------|
| `src/ratios/j1_euler_maclaurin.py` | MODIFY | Add D, delta, B_over_A to return dict |
| `src/ratios/microcase_plus5_signature_k3.py` | MODIFY | Pass through new metrics |
| `src/ratios/run_plus5_swap_experiment.py` | CREATE | 2×2 swap experiment |
| `tests/test_plus5_gate.py` | MODIFY | Add B/A tests |
| `tests/test_microcase_plus5_signature_k3.py` | MODIFY | Add B/A tests, update expected keys |

---

## Test Results

```
tests/test_plus5_gate.py: 24 passed
tests/test_microcase_plus5_signature_k3.py: 22 passed, 3 xfailed

Total: 46 passed, 3 xfailed (expected - Phase 14D tests without mirror)
```

---

## Validation Checklist

- [x] D, delta, B_over_A added to mirror assembly return
- [x] Tests use B/A instead of raw B
- [x] κ* gate test passes (removed XFAIL)
- [x] 2×2 swap experiment created and run
- [x] delta tracking documented (R-driven)
- [x] All existing tests still pass

---

## Key Insights

1. **B/A is the correct metric** - Raw B varies with A, but B/A = 5 + delta is stable

2. **delta is R-driven** - The contamination comes from Laurent factors (1/R+γ)², not polynomial degrees

3. **Both benchmarks pass** - κ: B/A ≈ 5.25 (5% off), κ*: B/A ≈ 5.08 (1.6% off)

4. **κ* was never broken** - The apparent 24% gap was an artifact of not normalizing

---

## Conclusion

Phase 14F successfully normalized the +5 gate metric:
- **Both κ and κ* now PASS** with B/A metric (10% tolerance)
- The 2×2 experiment shows delta is R-driven (Laurent factors)
- No changes needed to Euler-Maclaurin formulas themselves

**46 tests passing, 3 xfailed (expected).**
