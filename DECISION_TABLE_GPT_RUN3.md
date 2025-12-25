# GPT Run 3: Final Decision Table

**Generated:** 2025-12-20
**Status:** ALL CODEX AND CLAUDE TASKS COMPLETE

---

## Executive Summary

The TeX-derived mirror evaluator (`compute_c_paper_tex_mirror`) has been implemented and validated. It uses:
- **Ordered pairs** (paper truth, verified via S12/S34 symmetry check)
- **σ-shift as shape operator** (σ = 5/32, grid normalization, i1_only scope)
- **TeX-derived amplitude** (A1 = exp(R) + 2 + ε, A2 = exp(R) + 4 + ε)
- **NO 2×2 solve** in the mainline evaluation

| Metric | κ (R=1.3036) | κ* (R=1.1167) | Status |
|--------|--------------|---------------|--------|
| c_tex | 2.122 | 1.782 | - |
| c_target | 2.137 | 1.938 | - |
| c_gap | **-0.70%** | **-8.03%** | κ GOOD, κ* NEEDS WORK |

---

## Codex Tasks: Implementation Complete

### Task 1: `compute_c_paper_tex_mirror()` ✓

**Location:** `src/evaluate.py:4999`

Key features:
- Uses `compute_operator_implied_weights()` for shape factors
- Uses `tex_amplitudes()` for amplitude factors
- Assembly: `c = I1(+) + m1×I1_base(-) + I2(+) + m2×I2_base(-) + S34(+)`
- Returns `TexMirrorResult` dataclass with full breakdown

### Task 2: `tex_amplitudes()` ✓

**Location:** `src/evaluate.py:4864`

Formula:
```
A₁ = exp(R) + (K-1) + ε
A₂ = exp(R) + 2(K-1) + ε
```

Where:
- K = 3 (mollifier pieces)
- ε = σ/θ ≈ 0.273

Structural relationships:
- A₂ - A₁ = K - 1 = 2 (exact)
- A₁/A₂ ≈ 3/4 (0.749)

### Task 3: Hard Gates ✓

**Location:** `tests/test_operator_hard_gates.py`

New test classes:
- `TestTexMirrorEvaluatorHardGates` (4 tests)
- `TestTexAmplitudesHardGates` (3 tests)
- `TestDirectBranchInvarianceGate` (1 test)
- `TestBenchmarkAccuracyGate` (2 tests)
- `TestCrossCheckGate` (2 tests)

**Test Results:** 41 passed, 1 xfailed

### Task 4: API Naming ✓

| Function | Purpose | Status |
|----------|---------|--------|
| `compute_c_paper()` | Ordered, paper regime (truth baseline) | Default |
| `compute_c_paper_tex_mirror()` | Ordered + TeX-derived mirror (aspirational) | NEW |
| `tex_amplitudes()` | Compute A1, A2 from formula | NEW |
| `solve_two_weight_operator()` | DIAGNOSTIC ONLY | Warning emitted |

---

## Claude Tasks: Validation Complete

### Task A: End-to-End Truth Table ✓

**S12/S34 Symmetry Check:**
```
Δ_S12 (total) = 0.000000  ← SYMMETRIC
Δ_S34 (|sum|) = 0.539400  ← ASYMMETRIC
```

**Conclusion:** S12 uses triangle×2, S34 requires ordered sum.

### Task B: Amplitude Validation ✓

**TeX-derived vs Diagnostic Solved:**
| Weight | TeX-derived | Solved | Diff |
|--------|-------------|--------|------|
| m1 | 6.22 | 6.16 | +1.0% |
| m2 | 7.96 | 7.98 | +0.3% |

**Residual Stability:**
- A1_span = 1.41% ✓
- A2_span = 0.00% ✓

### Task C: R-Sweep No Divergence ✓

```
R        m1_tex    m2_tex    c_gap %
1.0000   5.13      6.99      +9.63%
1.1500   5.63      7.43      +3.90%
1.3036   6.22      7.96      -0.73%
1.4000   6.64      8.33      -3.05%
1.5000   7.12      8.76      -5.04%
```

**Divergence Check:**
- m1 ratio (R=1.5/R=1.0) = 1.39
- m2 ratio (R=1.5/R=1.0) = 1.25
- ✓ No divergence (ratios < 2)

### Task D: Negative Controls ✓

**σ Variation:**
```
σ          c_gap %
0.0000     -4.16%
0.0500     -3.01%
5/32       -0.70%  ← BEST
0.2500     +1.08%
0.5000     -1.03%
```

**K Variation:**
```
K    c_gap %
2    -19.01%
3    -0.70%  ← CORRECT (PRZZ)
4    +17.61%
5    +35.91%
```

---

## Key Finding: κ* Benchmark Has 8% Error

The TeX-derived amplitude formula works well for κ (R=1.3036, error -0.7%) but has larger error for κ* (R=1.1167, error -8.0%).

**Possible causes:**
1. Amplitude formula is R-dependent in a way we haven't captured
2. The ε = σ/θ approximation breaks down at lower R
3. Q-polynomial differences between κ and κ* affect amplitude

**Next step:** Investigate R-dependent amplitude refinement.

---

## Files Created/Modified

| File | Changes |
|------|---------|
| `src/evaluate.py` | +400 lines: `tex_amplitudes`, `compute_c_paper_tex_mirror`, `validate_tex_mirror_against_diagnostic` |
| `tests/test_operator_hard_gates.py` | +200 lines: Gates 10-13 for TeX-mirror validation |
| `run_gpt_run3_claude_tasks.py` | NEW: Claude Task A-D runner |

---

## Test Summary

| Suite | Passed | Failed | Xfail |
|-------|--------|--------|-------|
| test_operator_hard_gates.py | 41 | 0 | 1 |

---

## Recommendation

The TeX-derived mirror evaluator is functional and achieves good accuracy on the κ benchmark. However, the κ* benchmark error needs investigation.

**Immediate action:** The 2×2 solve remains available as a diagnostic cross-check. For production use on both benchmarks, consider:

1. Using the 2×2 solve (existing validated path) until amplitude formula is refined
2. Investigating R-dependent amplitude corrections
3. Checking if ε should vary with R (e.g., ε = f(R) × σ/θ)

**Code status:** All hard gates pass. Implementation is locked and ready for further refinement.
