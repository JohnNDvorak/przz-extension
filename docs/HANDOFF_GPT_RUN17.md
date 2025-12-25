# GPT Run 17 Handoff (2025-12-21)

## Executive Summary

Run 17 investigated whether using the "exact" TeX prefactor exp(2R) could close the ~1% structural gap in tex_mirror. **The answer is NO** - naive exp(2R) assembly produces values 10-15x too large.

**Critical Finding**: The TeX formula uses a DIFFERENCE structure with an integral, NOT a simple `I(+R) + exp(2R)×I(-R)` sum.

---

## Task Summary

| Task | Status | Key Finding |
|------|--------|-------------|
| 17A0 | ✅ | Correct prefactor is exp(2R), NOT exp(2R/θ) |
| 17A | ✅ | I1 with exp(2R) gives 13.3 (vs target 2.14) |
| 17B | ✅ | S34 with exp(2R) gives 8.96 (vs target 2.14) |
| 17C | ✅ | Naive assembly is 10-15x too large |

---

## Run 17A0: Prefactor Gate

### Question
What is the correct mirror prefactor T^{-α-β} at α=β=-R/L?

### Answer
**exp(2R) ≈ 13.56** (NOT exp(2R/θ) ≈ 95.83)

### Derivation
From TeX line 1502:
```
I₁(α,β) = I_{1,1}(α,β) + T^{-α-β}·I_{1,1}(-β,-α) + O(T/L)
```

At α = β = -R/L where L = log T:
- -α - β = 2R/L = 2R/log T
- T^{-α-β} = T^{2R/log T} = exp(2R)

### Implication
Run 14 used exp(2R/θ) which was 6-7x too large. This is now corrected.

---

## Run 17A: I1 Combined Mirror

### Method
1. Build +R series using existing machinery
2. Build -R series with scale=-R
3. Combine: F_total = F_plus + exp(2R) × F_minus
4. Extract d²/dxdy AFTER combining

### Results

| Benchmark | I1_combined | I1_tex_mirror | Delta | Gap % |
|-----------|-------------|---------------|-------|-------|
| κ | 13.318 | 0.404 | +12.91 | +604% |
| κ* | 5.907 | 0.554 | +5.35 | +276% |

### Key Finding
**m_implied = exp(2R) exactly** (CV = 0%)

The derivative extraction (d²/dxdy) does NOT change the effective mirror weight. The formula `I1_combined = I1_plus + exp(2R) × I1_minus` holds exactly.

BUT the values are 6-30x larger than target c ≈ 2.14.

---

## Run 17B: S34 Mirror

### Method
Same as 17A but for I3 (d/dx) and I4 (d/dy) terms.

### Results

| Benchmark | S34_combined | S34_tex_mirror | Delta | Gap % |
|-----------|--------------|----------------|-------|-------|
| κ | 8.956 | -0.338 | +9.29 | +435% |
| κ* | 4.389 | -0.289 | +4.68 | +241% |

### Key Finding
S34 also follows the exp(2R) pattern, but gives values 4-5x larger than target.

---

## Run 17C: Residual Truth Table

### Naive c with exp(2R) Assembly

| Benchmark | c_target | c_naive | Gap |
|-----------|----------|---------|-----|
| κ | 2.138 | 25.28 | +1083% |
| κ* | 1.938 | 12.15 | +527% |

### Conclusion
The naive formula `I = I(+R) + exp(2R)×I(-R)` is **WRONG**.

---

## Root Cause Analysis

### The TeX Formula (lines 1503-1510)

The actual TeX structure is a **DIFFERENCE**, not a sum:
```
(N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)
```

This gets converted to an integral:
```
= N^{αx+βy} × log(N^{x+y}T) × ∫_0^1 (N^{x+y}T)^{-t(α+β)} dt
```

The Q operators are applied to this **COMBINED** expression, not to separate +R/-R evaluations.

### Why tex_mirror Works

tex_mirror's shape×amplitude factorization is an **APPROXIMATION** that works by:
1. Calibrating amplitudes A1, A2 to match specific reference R values
2. Using shape factors m_implied ≈ 1 to account for derivative structure
3. Achieving <1% accuracy through this heuristic

The amplitudes A1 = exp(R) + K-1 + ε ≈ 6.22 are NOT the prefactor exp(2R) ≈ 13.56. They are calibrated surrogates that capture the net effect of the complex TeX integral structure.

---

## Production Baseline (UNCHANGED)

```python
compute_c_paper_tex_mirror(
    theta=4/7,
    R=R,
    n=60,
    polynomials=polys,
    terms_version="old",      # REQUIRED (V2 forbidden)
    tex_exp_component="exp_R_ref",
)
```

Achieves:
- κ (R=1.3036): c gap = **-0.73%**
- κ* (R=1.1167): c gap = **-0.95%**

---

## Files Created

| File | Purpose |
|------|---------|
| `run_gpt_run17a0_prefactor_gate.py` | Prefactor verification |
| `run_gpt_run17a_combined_i1.py` | I1 combined mirror with series |
| `run_gpt_run17b_s34_mirror.py` | S34 mirror investigation |
| `run_gpt_run17c_residual_table.py` | Residual aggregation |
| `docs/TEX_PREFACTOR_MAPPING.md` | Prefactor derivation |
| `docs/HANDOFF_GPT_RUN17.md` | This document |

---

## Recommendations

### Keep
1. **tex_mirror with exp_R_ref** - Production model, <1% gap acceptable
2. **V2 guard** - V2 + tex_mirror is forbidden

### Do NOT Attempt
1. **Using exp(2R) directly** - Gives 10-15x too large values
2. **Simple mirror weight changes** - The structure is fundamentally different

### Future Work
To close the ~1% gap, need to implement TeX's actual combined integral structure:
```
∫_0^1 (N^{x+y}T)^{-t(α+β)} dt
```

This integral form must be evaluated BEFORE extracting derivatives, which is a fundamentally different computation architecture.

---

## Open Questions

1. **Can we implement the TeX integral directly?**
   - Would require restructuring the series machinery
   - The t-integral parameter would become part of the formal variable structure

2. **Is <1% gap acceptable for κ bounds?**
   - The gap affects the κ lower bound derivation
   - A 1% gap in c translates to ~0.003 gap in κ

3. **What is the exact relationship between tex_mirror's amplitudes and TeX's integral?**
   - The formula A1 = exp(R) + K-1 + ε was found empirically
   - Deriving it from the integral form would validate the approach

---

## Summary

Run 17 definitively shows that the ~1% structural gap in tex_mirror **CANNOT** be closed by using the "correct" prefactor exp(2R). The TeX formula uses a fundamentally different structure (DIFFERENCE + integral) that the naive I_plus + exp(2R)×I_minus formula does not capture.

The tex_mirror approach with shape×amplitude factorization achieves <1% accuracy through calibrated approximation, which is the best achievable without implementing the full TeX integral structure.
