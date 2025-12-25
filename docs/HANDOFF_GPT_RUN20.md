# GPT Run 20 Handoff (2025-12-21)

## Executive Summary

Run 20 implemented the **TeX-exact difference quotient → log×integral identity** with Q operators applied AFTER the combined structure. **The result does NOT close the ~1% gap**, but provides diagnostic insight into the structural mismatch.

| Benchmark | I1 (Run 20) | I1 (tex_mirror) | Ratio | c (Run 20) | c Gap |
|-----------|-------------|-----------------|-------|------------|-------|
| κ         | 1.59        | 0.40            | 3.9×  | 1.52       | -29%  |
| κ*        | 1.13        | 0.55            | 2.0×  | 1.06       | -45%  |

**Conclusion**: The TeX combined structure produces I1 values 2-4x larger than tex_mirror's effective I1, confirming that the "combined integral" interpretation differs fundamentally from the empirical mirror assembly.

---

## What Run 20 Implemented

### Stage 20A: TexCombinedMirrorCore Class

**File**: `src/term_dsl.py` (lines 691-856)

**Formula** (PRZZ TeX 1502-1511):
```
(N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)
= N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-s(α+β)} ds
```

At α = β = -R/L:
```
= exp(-Rθ(x+y)) × (1 + θ(x+y)) × ∫₀¹ exp(2sR(1 + θ(x+y))) ds
```

**Key difference from Run 18**: Includes outer `exp(-Rθ(x+y))` factor from N^{αx+βy}.

**Gate tests**: 14 tests passing (scalar limit, series, quadrature convergence, difference quotient identity)

### Stage 20B/20C: compute_I1_tex_combined_core_11()

**File**: `src/evaluate.py` (lines 6212-6330)

**Structure**:
1. Build TexCombinedMirrorCore (NO Q yet)
2. Apply Q operators AFTER combined structure
3. Multiply by P₁ factors and prefactor
4. Extract xy coefficient
5. Integrate

**Key innovation**: Q operators applied at correct time per PRZZ TeX.

**Gate tests**: 7 tests passing (finite, non-zero, converges, differs from Run 18/19)

---

## Diagnostic Results

### κ Benchmark (R=1.3036, c_target=2.138)

| Method | I1 | c | Gap |
|--------|-----|------|------|
| Run 20 (TexCombinedMirrorCore) | 1.59 | 1.52 | -29% |
| Run 18 (CombinedMirrorFactor) | 2.08 | 2.01 | -6% |
| Run 19 (naive plus+minus) | 4.30 | 4.24 | +98% |
| tex_mirror (production) | 0.40 | 2.12 | -0.7% |

### κ* Benchmark (R=1.1167, c_target=1.938)

| Method | I1 | c | Gap |
|--------|-----|------|------|
| Run 20 (TexCombinedMirrorCore) | 1.13 | 1.06 | -45% |
| Run 18 (CombinedMirrorFactor) | 1.43 | 1.37 | -29% |
| Run 19 (naive plus+minus) | 0.80 | 0.73 | -62% |
| tex_mirror (production) | 0.55 | 1.92 | -1.0% |

### I1 Ratios vs tex_mirror

| Benchmark | Run 20 | Run 18 | Run 19 |
|-----------|--------|--------|--------|
| κ         | 3.9×   | 5.1×   | 10.7×  |
| κ*        | 2.0×   | 2.6×   | 1.4×   |

**Observation**: Run 20 is less divergent than Run 18, suggesting the outer exp factor is a step in the right direction.

---

## Root Cause Analysis

### Why "Combined Structure" Differs From tex_mirror

**tex_mirror computes**:
```
I1 = I1_plus + m1 × I1_minus_base
```
Where:
- I1_plus: Standard integral at +R
- I1_minus_base: Standard integral at -R
- m1 ≈ exp(R) + 5: Calibrated weight

**Combined structures compute**:
A single integral that tries to encompass both +R and -R branches via the TeX identity.

**The mismatch**: The TeX identity transforms the difference quotient into a log×integral form, but this is NOT equivalent to simply computing I1 at ±R and weighting.

### The Asymptotic L Factor

The TeX identity includes `log(N^{x+y}T) = L × (1 + θ(x+y))` where L = log T.

In our implementation, we absorb L as asymptotic. But tex_mirror doesn't use this structure at all - it uses separate +R/-R evaluations with empirical weights.

**Hypothesis**: The log×integral identity is for derivation/proof purposes in PRZZ, not for direct numerical computation. The tex_mirror approach (separate +R/-R with weights) may be the intended computational form.

---

## Key Learnings

1. **Run 19 ruled out naive plus+minus** (10x error)
2. **Run 20 improved on Run 18** (outer exp factor helps)
3. **All "combined" approaches produce I1 ~2-5x larger than tex_mirror**
4. **tex_mirror's empirical m1 = exp(R)+5 captures NET effect** without exact structure

### The Fundamental Issue

The PRZZ TeX formula uses the difference quotient → log×integral identity to **remove the 1/(α+β) singularity analytically**. This is a mathematical technique for proving bounds, not necessarily the computational formula.

When PRZZ actually computes c numerically, they likely:
1. Evaluate at +R and -R separately
2. Apply appropriate weights (which become m1, m2)
3. Use the functional equation implicitly through weight calibration

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `src/term_dsl.py` | Added TexCombinedMirrorCore class (~165 lines) |
| `src/evaluate.py` | Added compute_I1_tex_combined_core_11() (~120 lines) |
| `tests/test_tex_combined_gates.py` | Added 21 Stage 20A/B/C tests |
| `run_gpt_run20_diagnostic.py` | Diagnostic comparison script |
| `docs/HANDOFF_GPT_RUN20.md` | This document |

---

## Recommendations

### Option A: Accept tex_mirror as Production

tex_mirror achieves ~1% accuracy on both benchmarks. The ~1% gap may be:
- Quadrature precision
- Polynomial transcription errors
- Missing minor normalization factors

**Pros**: Works now, enables K>3 extension work
**Cons**: exp_R_ref calibration isn't first-principles derived

### Option B: Investigate tex_mirror's Internal Structure

The combined structures produce ~2-5x larger I1 values. This suggests:
1. tex_mirror may have implicit damping factors
2. The m1 weight absorbs asymptotic factors

**Next diagnostic**: Compare intermediate series coefficients (before xy extraction) between combined and tex_mirror approaches.

### Option C: Re-read PRZZ for Computational Form

The TeX identity may be for proof, not computation. Check if PRZZ Section 7 or 10 describes the actual numerical procedure.

---

## Locked Assumptions (Still Valid)

| Question | Answer | Source |
|----------|--------|--------|
| I₃/I₄ mirror? | NO (S34 plus-only) | TRUTH_SPEC Section 10 |
| Q-shift value? | Exactly +1 (σ=1.0) | Run 19 derivation |
| I₂ combined structure? | NO (base integral) | Run 18 proof |

---

## Test Results

```
tests/test_tex_combined_gates.py: 32 passed (includes Run 18 + Run 20 tests)
```

All gate tests pass - the implementations are correct for their specified structures. The issue is that the structures themselves differ from what tex_mirror computes.
