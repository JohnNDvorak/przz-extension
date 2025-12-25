# Session 12 Progress: Series-Backed Evaluator and Ratio Analysis

## Summary

**Phase A Complete**: SeriesBackedEvaluator created and verified
**Key Finding**: The ratio error is in the polynomial structure, NOT derivative computation

---

## Accomplishments

### 1. SeriesBackedEvaluator (Phase A)

Created `src/series_backed_evaluator.py` - All I-terms computed via series:
- I1 → g[1,1] with weight (1-u)^{ell+ellbar}
- I2 → g[0,0] with weight 0
- I3 → -g[1,0] with weight (1-u)^ell
- I4 → -g[0,1] with weight (1-u)^ellbar

**Validation**: Produces IDENTICAL results to HybridEvaluator for all pairs.

### 2. Test Suite (test_series_backed.py)

12 tests covering:
- All pairs match HybridEvaluator ✓
- Integral grid caching ✓
- Symmetric I3/I4 for diagonal pairs ✓
- Quadrature convergence ✓
- Two-benchmark gate (documents expected errors) ✓

### 3. Kernel Research (Phase B.1)

Documented in `docs/SESSION_12_KERNEL_RESEARCH.md`:
- C_alpha/C_beta are likely **coefficients**, not kernel modifiers
- Current (a, b, weight_exp) lookup is probably correct
- Case C (auxiliary integral for ℓ≥2) is a separate concern

### 4. (2,2) Monomial Tracing

Traced all 12 monomials for both benchmarks:

| Monomial Type | κ contrib | κ* contrib | Ratio |
|---------------|-----------|------------|-------|
| D² | +1.818 | +0.681 | 2.67 |
| C_α²×C_β² | -0.909 | -0.341 | 2.67 |
| A×B×D | +4.541 | +1.925 | 2.36 |
| A²×B² | +0.971 | +0.301 | 3.23 |
| **TOTAL (2,2)** | **+1.824** | **+1.154** | **1.58** |

---

## Key Finding: Ratio Error Source

### The Problem

All integral values have κ/κ* ratios > 2.0, but target ratio is 1.10.

| Component | κ value | κ* value | Ratio | Target |
|-----------|---------|----------|-------|--------|
| (2,2) total | 1.824 | 1.154 | 1.58 | 1.10 |
| Full c | 2.325 | 1.166 | 1.99 | 1.10 |
| ||P₂||² L² | 0.725 | 0.318 | 2.28 | - |

### The Mystery

From HANDOFF_SUMMARY:
- **Naive formula gives**: const ratio 1.71 (κ > κ*)
- **PRZZ needs**: const ratio 0.94 (κ < κ*)
- **Ratios are OPPOSITE directions!**

### Interpretation

The κ polynomials are fundamentally larger than κ* polynomials:
- κ P₂ L² norm: 0.851
- κ* P₂ L² norm: 0.564
- Ratio: 1.51

Since ∫P²du is always positive, κ will always exceed κ* in naive integrals.

PRZZ must have **negative correlation** between ||P|| and contribution. Possibilities:
1. Case C kernel suppresses larger polynomials more
2. Derivative term subtraction (larger P → larger P' → more subtraction)
3. Normalization factors we're missing

---

## Disproven Hypotheses

### 1. "Monomials with same (a,b) need different kernels"
**Status**: Unlikely to be the main issue

Evidence: C_alpha/C_beta are coefficient factors from pole residues, not kernel modifiers. The current structure (coefficient × integral) appears correct for the F(x,y) kernel.

### 2. "Series engine derivatives are wrong"
**Status**: Disproven

Evidence: SeriesBackedEvaluator matches HybridEvaluator exactly. The coefficient gate passed. Derivatives are correct.

### 3. "The ratio error is in structural assembly"
**Status**: Partially correct, but more fundamental

The error IS structural, but it's in the fundamental relationship between polynomial magnitudes and contributions, not in derivative computation or monomial collapsing.

---

## What Works

| Component | Status | Evidence |
|-----------|--------|----------|
| Series coefficient extraction | ✓ | Coefficient gate passed |
| I-term mapping (I1→g[1,1], etc.) | ✓ | Matches HybridEvaluator |
| Monomial expansion | ✓ | Correct counts, coefficients |
| Quadrature convergence | ✓ | Stable at n=60/80 |

## What Doesn't Work

| Issue | Magnitude | Root Cause |
|-------|-----------|------------|
| κ/κ* ratio error | 80%+ | Polynomial magnitude correlation |
| c absolute error | 9-40% | Same root cause |

---

## Files Created/Modified

| File | Status |
|------|--------|
| `src/series_backed_evaluator.py` | Created |
| `tests/test_series_backed.py` | Created, 12 tests passing |
| `trace_22_monomials.py` | Created |
| `docs/SESSION_12_KERNEL_RESEARCH.md` | Created |
| `SESSION_12_PROGRESS.md` | This file |

---

## CRITICAL FINDING: Canary Pair Analysis

### GPT's Key Question
"For (2,2), identify two monomials with same (a,b) but different (c_α,c_β,d). Do they map to same or different kernels?"

### Answer: **SAME KERNEL**

From PRZZ TeX analysis (lines 2275-2400):
- **ω (omega) determines the kernel**, not (c_α, c_β, d)
- ω = ℓ - 1 for piece ℓ with d=1
- C_α/C_β are **coefficient factors** from pole residues, not kernel modifiers

### Canary Pair Verdict

| Canary | Monomials | Same Kernel? | Why |
|--------|-----------|--------------|-----|
| 1 | D² vs C_α²×C_β² | YES | Both use (ω=1, ω=1) Case C×C |
| 2 | B×C_β×D vs B×C_α×C_β² | YES | Both use (ω=1, ω=1) with b=1 derivative |

### Implication

**The current (a, b, weight_exp) lookup is structurally correct!**

The ratio error is NOT from monomial collapsing. It's from using raw P(u) instead of Case C kernels K_ω(u; R) for ℓ≥2 pieces.

See: `docs/CANARY_PAIR_VERDICT.md` for full analysis.

---

## Recommended Next Step: Implement Case C Kernels

The path forward is clear:

1. **For P₁ (ω=0)**: Keep using P₁(u) directly (Case B)
2. **For P₂ (ω=1)**: Replace with K₁(u; R) = u × ∫₀¹ P₂((1-a)u) × exp(Rθua) da
3. **For P₃ (ω=2)**: Replace with K₂(u; R) = (u²/1!) × ∫₀¹ P₃((1-a)u) × a × exp(Rθua) da

**Note**: Case C analysis showed this makes c SMALLER (0.57 vs 1.95), which is the WRONG direction. This paradox remains unexplained, but the structure is now clear.

### Alternative: Something Beyond Case C

If Case C alone can't fix the ratio (it pushes c in wrong direction), there may be:
- Additional normalization factors
- Mirror term combination effects (TeX 1502-1511)
- Variable rescaling effects (TeX 2309: x → x log N)

---

## Conclusions

Session 12 confirmed:
1. **Series engine is correct** - derivative computation is not the issue
2. **Monomial structure is correct** - C_alpha/C_beta are coefficient factors
3. **The ratio problem is fundamental** - polynomial magnitudes correlate wrong

The gap between our computation and PRZZ targets requires understanding how PRZZ achieves **negative correlation** between polynomial magnitude and contribution.

---

Date: Session 12
