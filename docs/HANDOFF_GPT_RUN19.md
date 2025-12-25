# GPT Run 19 Handoff (2025-12-21)

## Executive Summary

Run 19 implemented the "TeX-exact mirror core" with Q-shift inside the combined structure. **The result is that the naive combined interpretation does NOT produce tex_mirror-like values.**

| Benchmark | tex_exact I1 | tex_mirror I1 | Ratio | Assembly Gap |
|-----------|--------------|---------------|-------|--------------|
| κ | 4.30 | 0.40 | **10.7x** | +98% |
| κ* | 0.80 | 0.55 | **1.4x** | -62% |

**Conclusion**: The CombinedI1Integrand structure (plus_branch + exp(2R)×minus_branch with Q-shift inside) is fundamentally different from what tex_mirror computes. The interpretation needs revision.

---

## What Run 19 Implemented

### Stage 19A: CombinedI1Integrand Class

**File**: `src/term_dsl.py` (lines 540-688)

**Structure**:
```python
plus_branch = Q(arg_α) × Q(arg_β) × exp(R·arg_α) × exp(R·arg_β)
minus_branch = Q(arg_α+1) × Q(arg_β+1) × exp(-R·arg_α) × exp(-R·arg_β) × exp(2R)
combined = plus_branch + minus_branch
```

**Gate tests**: 9 tests passing (scalar limit, series evaluation, Q-shift correctness)

### Stage 19B: compute_I1_tex_exact_11()

**File**: `src/evaluate.py` (lines 6075-6189)

**Process**:
1. Build CombinedI1Integrand with Q_shifted = Q(x+1)
2. Multiply by P₁(x+u) × P₁(y+u) × prefactor
3. Extract d²/dxdy coefficient
4. Integrate over (u, t)

**Gate tests**: 5 tests passing (finite values, convergence)

---

## Diagnostic Results

### κ Benchmark (R=1.3036, c_target=2.138)

| Component | tex_exact | tex_mirror |
|-----------|-----------|------------|
| I1 | 4.30 | 0.40 |
| I2 | 0.38 | 2.06 |
| S34 | -0.45 | -0.34 |
| **c** | 4.24 | 2.12 |
| **Gap** | +98% | -0.7% |

### κ* Benchmark (R=1.1167, c_target=1.938)

| Component | tex_exact | tex_mirror |
|-----------|-----------|------------|
| I1 | 0.80 | 0.55 |
| I2 | 0.32 | 1.65 |
| S34 | -0.39 | -0.29 |
| **c** | 0.73 | 1.92 |
| **Gap** | -62% | -1.0% |

---

## Root Cause Analysis

### Why tex_exact Differs from tex_mirror

1. **tex_mirror computes**:
   - I1_plus: standard Q and exp at +R
   - I1_minus: standard Q and exp at -R (same Q, just R → -R)
   - I1 = I1_plus + m1 × I1_minus where m1 = exp(R) + 5 ≈ 8.7

2. **tex_exact computes**:
   - plus_branch: Q(arg) × exp(+R×arg)
   - minus_branch: Q(arg+1) × exp(-R×arg) × exp(2R) ≈ Q_shifted × exp(-R×arg) × 13.6
   - combined = plus_branch + minus_branch

### Key Differences

| Aspect | tex_exact | tex_mirror |
|--------|-----------|------------|
| Q in minus | Q(arg+1) shifted | Q(arg) unshifted |
| Multiplier | exp(2R) ≈ 13.6 | m1 ≈ 8.7 |
| Structure | Combined inside | Separate +R/-R evaluations |

### The Interpretation Problem

The TeX formula:
```
(N^{αx+βy} - T^{-α-β}N^{-βx-αy})/(α+β)
= N^{αx+βy} × log(N^{x+y}T) × ∫_0^1 (N^{x+y}T)^{-s(α+β)} ds
```

This integral representation is NOT equivalent to:
```
plus_branch + exp(2R) × minus_branch
```

The log factor and s-integral create a different structure that cannot be decomposed into simple +/- branches.

---

## Files Created/Modified

| File | Changes |
|------|---------|
| `src/term_dsl.py` | Added CombinedI1Integrand class (~150 lines) |
| `src/evaluate.py` | Added compute_I1_tex_exact_11() (~115 lines) |
| `tests/test_tex_exact_gates.py` | Gate tests for Stages 19A-B (14 tests) |
| `run_gpt_run19_diagnostic.py` | Diagnostic comparison script |
| `docs/HANDOFF_GPT_RUN19.md` | This document |

---

## Key Learnings

1. **Q-shift inside combined structure is correct conceptually** - but the exponential combination is wrong

2. **The plus+minus interpretation is too naive** - the TeX formula has log×integral structure

3. **tex_mirror works by empirical calibration** - m1 = exp(R)+5 captures the NET effect without exact structure

4. **The asymmetric benchmark behavior is diagnostic** - 10.7x for κ vs 1.4x for κ* indicates fundamental structural issues

---

## Recommendations for Run 20

### Option A: Investigate log×integral structure

The TeX combined formula involves:
```
log(N^{x+y}T) × ∫_0^1 (N^{x+y}T)^{-s(α+β)} ds
```

This is what Run 18's CombinedMirrorFactor attempted, but Q operators were applied externally. A proper implementation would need to:
1. Compute the s-integral as a series in (x, y)
2. Apply Q operators to this combined series
3. Extract derivatives from the fully assembled structure

### Option B: Derive tex_mirror's m1 from first principles

Instead of guessing m1 = exp(R) + (2K-1), derive it from:
1. The PRZZ asymptotic analysis
2. The relationship between Q(D) and Q(1+D) operators
3. The polynomial degree-dependent normalizations

### Option C: Accept tex_mirror as production

Given that tex_mirror achieves ~1% accuracy on both benchmarks:
- Keep it as the production model
- Document that the "exact" formula is for derivation, not computation
- Focus optimization on improving polynomials rather than kernel formulas

---

## Test Results

```
tests/test_tex_exact_gates.py: 14 passed
```

All gate tests pass - the implementation is correct for the specified structure. The issue is that the structure itself differs from what PRZZ computes.

---

## Clarified Answers (Integrated from GPT Guidance)

1. **I₃/I₄ mirror**: NO (S34 plus-only per TRUTH_SPEC Section 10)
2. **Q-shift value**: Exactly +1 (sigma=1.0, not derived from α+β)
3. **I₂ combined**: NO (use base integral, Run 18 proved this)

These answers remain valid and are reflected in the implementation.
