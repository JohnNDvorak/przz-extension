# Phase 9 Summary: Derived Mirror Term Investigation

**Date:** 2025-12-23
**Status:** Complete
**Tests:** 76 new tests, all passing

## Executive Summary

Phase 9 investigated whether the 3.7% structural gap between the empirical formula `m₁ = exp(R) + 5` and the fitted formula `m₁ = 1.037×exp(R) + 5` could be explained by implementing the "true" TeX mirror transformation.

**Key Finding:** The naive application of the shift identity `Q(1+D)` to compute the TeX mirror gives values that are **100-1000× too large** compared to the empirical formula. This definitively proves that the DSL "minus basis" (R → -R sign flip) is **not** a direct implementation of the TeX mirror term, but rather a practical approximation that works empirically.

---

## Background: The Mirror Assembly Formula

The PRZZ c-constant is computed via mirror assembly:

```
c = S12(+R) + m₁ × S12(-R) + S34(+R)
```

Where:
- `S12 = I₁ + I₂` (terms requiring mirror)
- `S34 = I₃ + I₄` (terms without mirror)
- `m₁` is the mirror multiplier

### The Empirical Formula That Works

From Phase 8, we found:
- **Empirical:** `m₁ = exp(R) + 5` achieves ~1.35% accuracy
- **Fitted:** `m₁ = 1.037×exp(R) + 5` achieves 0% gap (by construction)

The "+5" appears to be `2K - 1` for K=3 mollifier pieces.

### The TeX Mirror Structure

According to PRZZ TeX (lines 1502-1511), the mirror term has structure:

```
I(α,β) + T^{-(α+β)} × I(-β,-α)
```

At the evaluation point `α = β = -R/L`:
- `T^{-(α+β)} = exp(2R)`
- `I(-β,-α)` means the integral evaluated at the mirrored Mellin point

### The Shift Identity

The key algebraic identity is:

```
Q(D_α)[T^{-s}F] = T^{-s} × Q(1+D_α)[F]
```

**Proof sketch:**
- `D_α = -1/L × d/dα`
- `D_α[T^{-s}] = T^{-s}` (eigenvalue = 1)
- By product rule: `D_α[T^{-s}F] = T^{-s}(1 + D_α)[F]`
- Therefore `Q(D)[T^{-s}F] = T^{-s}Q(1+D)[F]`

This identity is mathematically correct and was validated in our tests.

---

## Phase 9 Implementation

### 9.0: Hygiene
- Fixed doc inconsistency about m₁_eff spread (was "50-63%", corrected to "1.6-2.2%")
- Created canonical evaluator entrypoint (`src/canonical_evaluator.py`)

### 9.1: Mirror Transform Algebra Harness
Created `src/mirror_transform_algebra.py` with:
- `AffineOperatorAction` dataclass for eigenvalue representation
- `MirrorTransform` class for T^{-s} transformations
- `validate_shift_identity_analytic()` - proves identity on toy kernels
- `validate_shift_identity_numerical()` - validates with PRZZ Q polynomial

**All shift identity tests pass** - the algebra is correct.

### 9.2: Derived Mirror Implementation
Extended `src/mirror_exact.py` with:
- `compute_I1_mirror_derived()` - computes `exp(2R) × I₁(+R, shifted Q)`
- `compute_I2_mirror_derived()` - computes `exp(2R) × I₂(+R, shifted Q)`
- `compute_S12_mirror_derived()` - full S12 with derived mirror
- `compute_S12_minus_basis()` - DSL minus basis for comparison

### 9.3: Diagnostic Runner
Created `run_phase9_derived_mirror_check.py` producing detailed diagnostics.

### 9.4: R-Sweep Analysis
Created `run_phase9_a_coefficient_sweep.py` to investigate how `a(R)` varies.

---

## Numerical Results

### Channel Values at Benchmarks

#### κ Benchmark (R = 1.3036)

| Quantity | Value |
|----------|-------|
| S12(+R, std Q) | 1.0826 |
| S12(-R, std Q) | 0.3488 (DSL minus basis) |
| S12(+R, shifted Q) | 29.716 |
| exp(2R) | 13.561 |
| S12_mirror_derived | 402.98 |

#### κ* Benchmark (R = 1.1167)

| Quantity | Value |
|----------|-------|
| S12(+R, std Q) | 0.7626 |
| S12(-R, std Q) | 0.2948 (DSL minus basis) |
| S12(+R, shifted Q) | 2.750 |
| exp(2R) | 9.332 |
| S12_mirror_derived | 25.66 |

### The Q(1+·) Shift Effect

The Q polynomial shift `Q(z) → Q(1+z)` causes massive amplification:

| Pair | I1_std(+R) | I1_shifted(+R) | Ratio |
|------|------------|----------------|-------|
| (1,1) κ | 0.4135 | 46.50 | 112× |
| (2,2) κ | 0.1609 | 16.58 | 103× |
| (3,3) κ | 8.79e-5 | 7.51e-3 | 85× |
| (1,2) κ | -0.5681 | -72.01 | 127× |

For κ* benchmark, the amplification is smaller (~5×) but still significant.

### Implied m₁ Values

| Method | κ benchmark | κ* benchmark |
|--------|-------------|--------------|
| **m₁_empirical** | 8.68 | 8.05 |
| m₁ from std mirror (exp(2R)×S12+/S12-) | 42.10 | 24.14 |
| m₁ from shifted mirror | 1155.5 | 87.07 |

### Implied 'a' Coefficient

If `m₁ = a × exp(R) + 5`:

| Benchmark | a_derived (shifted Q) | a_from_std | Expected |
|-----------|----------------------|------------|----------|
| κ | 312.4 | 10.1 | ~1.037 |
| κ* | 26.9 | 6.3 | ~1.037 |

---

## R-Sweep Analysis

Sweeping R from 0.8 to 1.5 with κ polynomials:

| R | exp(R) | S12+/S12- | m₁_emp | m₁_derived | a_derived |
|---|--------|-----------|--------|------------|-----------|
| 0.8 | 2.23 | 1.99 | 7.23 | 177.2 | 77.4 |
| 1.0 | 2.72 | 2.37 | 7.72 | 377.6 | 137.1 |
| 1.2 | 3.32 | 2.83 | 8.32 | 792.6 | 237.2 |
| 1.3036 | 3.68 | 3.10 | 8.68 | 1155.5 | 312.4 |
| 1.5 | 4.48 | 3.70 | 9.48 | 2323.8 | 517.4 |

**Linear fit:** `a_derived = -457.5 + 611.2 × R`

The 'a' coefficient is **strongly R-dependent**, not constant as the fitted formula suggests.

---

## Key Ratio Relationships

### S12(+R) / S12(-R) vs exp(R)

| Benchmark | S12+/S12- | exp(R) | Ratio |
|-----------|-----------|--------|-------|
| κ | 3.10 | 3.68 | 0.84 |
| κ* | 2.59 | 3.05 | 0.85 |

The plus/minus ratio is **~85% of exp(R)**, not equal to it.

### Three Different "Mirror" Quantities

1. **S12(-R):** DSL minus basis - same formula but R → -R in exp attenuation
2. **exp(2R) × S12(+R, std Q):** TeX mirror without Q shift
3. **exp(2R) × S12(+R, shifted Q):** TeX mirror with Q(1+D) shift

These are fundamentally different:
- #1 is what PRZZ code uses (with m₁ multiplier)
- #2 gives m₁ ≈ 42 (5× too large)
- #3 gives m₁ ≈ 1155 (133× too large)

---

## What We Learned

### 1. The Shift Identity is Correct but Misapplied

The shift identity `Q(D)[T^{-s}F] = T^{-s}Q(1+D)[F]` is mathematically valid. Our tests confirm it. However:

- The identity tells us how Q(D) **acts on** a T^{-s} factor
- It does **not** mean the mirror term uses Q(1+D)
- The mirror term `I(-β,-α)` uses the **original Q**, not Q(1+D)

### 2. The DSL Minus Basis is an Approximation

The DSL "minus basis" computes S12 with R → -R sign flip. This is **not** the same as:
- Evaluating I at the mirrored Mellin point
- Applying the T^{-(α+β)} weight

It's a practical approximation that, combined with `m₁ = exp(R) + 5`, gives good numerical results.

### 3. The "+5" Term Absorbs Complex Effects

The formula `m₁ = exp(R) + 5` where 5 = 2K-1:
- Works empirically to ~1.35% accuracy
- Does **not** come from any simple derivation we found
- Absorbs polynomial-dependent and structural effects

### 4. The 3.7% Gap Remains Unexplained

The fitted `a ≈ 1.037` coefficient cannot be derived from:
- The shift identity interpretation
- Simple ratios of S12 values
- The R-sweep analysis

---

## Open Questions for Further Investigation

### Q1: What transformation does PRZZ actually use?

The TeX lines 1502-1511 show `T^{-(α+β)} × I(-β,-α)`, but our naive implementation gives wrong values. What normalization or transformation is missing?

### Q2: Where does "+5" come from?

The `2K-1` pattern suggests counting something (maybe derivative orders or polynomial terms), but we haven't found the derivation.

### Q3: Why does S12+/S12- ≈ 0.85×exp(R)?

If the mirror involves exp(2R) and the ratio is related to exp(R), there should be a theoretical relationship. Why the 0.85 factor?

### Q4: Is there a polynomial-dependent normalization?

The κ vs κ* discrepancy (a = 312 vs a = 27 for derived, but both should be ~1.037) suggests the polynomials affect the mirror structure in ways we don't capture.

### Q5: What is the correct interpretation of the Mellin evaluation?

At `α = β = -R/L`:
- `I(-β,-α)` should evaluate at `(+R/L, +R/L)`
- But our exp attenuation factors use R directly, not R/L

Is there a scale factor (like L or θ) we're missing?

---

## Hypothesis for Future Investigation

The most promising hypothesis is that the PRZZ mirror assembly uses a **different contour or residue structure** than the direct term, leading to different effective weights. The formula `m₁ = exp(R) + 5` might come from:

1. A **discrete sum** over poles rather than continuous integral
2. **Regularization** of divergent contributions
3. **Asymptotic expansion** terms that contribute "+5"

The shift identity is algebraically correct but might not be the right tool for computing the numerical mirror contribution.

---

## Files Created in Phase 9

### Source Files
- `src/canonical_evaluator.py` (294 lines)
- `src/mirror_transform_algebra.py` (~300 lines)
- Extended `src/mirror_exact.py` (+380 lines)

### Test Files
- `tests/test_canonical_evaluator.py` (16 tests)
- `tests/test_mirror_transform_algebra.py` (25 tests)
- `tests/test_mirror_term_derived.py` (25 tests)
- `tests/test_derived_mirror_is_close_to_baseline.py` (10 tests)

### Scripts
- `run_phase9_derived_mirror_check.py`
- `run_phase9_a_coefficient_sweep.py`

### Documentation
- `docs/PHASE9_SUMMARY_FOR_GPT.md` (this file)

---

## Conclusion

Phase 9 successfully implemented the derived mirror infrastructure and proved that **the naive Q(1+D) interpretation does not match the empirical formula**. The DSL minus basis with `m₁ = exp(R) + 5` remains the best practical approach, but its theoretical foundation is still unclear.

The 3.7% gap between empirical and fitted formulas likely comes from structural effects in the PRZZ integral representation that our current analysis doesn't capture. Further investigation should focus on the asymptotic expansion structure and possible discrete contributions to the mirror term.
