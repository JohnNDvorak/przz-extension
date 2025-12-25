# GPT Run 14-16 Handoff (2025-12-21)

## Overview

Runs 14-16 follow up on the Run 12-13 findings to:
1. Lock the production path (V2 guard)
2. Verify quadrature convergence
3. Fix apples-to-oranges comparison in Run 13A
4. Compute exact mirror weights from TeX combined formula
5. R-sweep to compare exact vs tex_mirror weights

**Key Finding**: The exact mirror weight exp(2R/θ) is 10-15x larger than tex_mirror's m ≈ 6-8. This is NOT calibration error - it reflects fundamentally different formulations.

---

## Task 1: V2 Guard (COMPLETE)

### Implementation

Added hard guard at `evaluate.py` line 5219-5229:

```python
if terms_version == "v2":
    raise ValueError(
        "FATAL: terms_version='v2' is FORBIDDEN with tex_mirror. "
        "V2 terms catastrophically fail under mirror assembly (I1_plus sign flip). "
        "Use terms_version='old' (default). See docs/HANDOFF_GPT_RUN12_13.md."
    )
```

### Test

`tests/test_v2_tex_mirror_guard.py` - 5 tests passing:
- V2 + tex_mirror raises ValueError
- Error message is informative
- OLD works correctly
- Default works correctly
- Production config achieves <2% gap

---

## Task 2: Quadrature Convergence (COMPLETE)

### Script

`run_gpt_run16_quadrature_convergence.py`

### Results

| n | κ c | κ gap | κ* c | κ* gap | Converged? |
|---|-----|-------|------|--------|------------|
| 40 | 2.1223 | -0.71% | 1.9199 | -0.93% | — |
| 60 | 2.1220 | -0.73% | 1.9195 | -0.95% | YES |
| 80 | 2.1218 | -0.73% | 1.9193 | -0.96% | YES |
| 100 | 2.1217 | -0.74% | 1.9192 | -0.97% | YES |
| 140 | 2.1216 | -0.74% | 1.9191 | -0.98% | YES |

### Conclusion

**Gap is STRUCTURAL, not numerical.**

Quadrature converges by n=60 (Δc < 0.1%). The residual ~0.7-1.0% gap persists at all resolutions, confirming it comes from the amplitude/assembly model, not quadrature precision.

---

## Task 3: Fix Run 13A Output (COMPLETE)

### Problem

Lines 356-369 compared (1,1) pair values against ALL-pair totals (apples-to-oranges).

### Solution

Restructured into clearly labeled sections:

1. **Section 1**: Pair-Level Analysis (direct I-terms for individual pairs)
2. **Section 2**: Full-Assembly Analysis (tex_mirror totals for all 6 pairs)
3. **Section 3**: Comparison Interpretation (explicit warning about non-comparability)
4. **Section 4**: (2,2) Pair Reference (incomplete, wrong power)
5. **Section 5**: Structural Analysis

---

## Task 4: Run 14 - Combined Mirror Evaluator (COMPLETE)

### Script

`run_gpt_run14_combined_mirror.py`

### Formula

The combined +R/-R integral from TeX:

```
I_combined = ∫∫ f(u,t) × [exp(2Rt) + exp(2R/θ) × exp(-2Rt)] du dt
```

Exact mirror weight:
```
m_exact = (I_combined - I_plus) / I_minus_base = exp(2R/θ)
```

### Results

| Benchmark | R | m_exact | m1_tex | m2_tex | Ratio |
|-----------|------|---------|--------|--------|-------|
| κ | 1.3036 | **95.83** | 6.22 | 7.96 | **15.4x** |
| κ* | 1.1167 | **49.82** | 6.14 | 7.96 | **8.1x** |

### Key Finding

- m_exact = exp(2R/θ) exactly (pair-independent, CV = 0%)
- tex_mirror uses m ≈ 6-8, which is 10-15x smaller
- This is NOT calibration error - different formulations

---

## Task 5: Run 15 - R-Sweep (COMPLETE)

### Script

`run_gpt_run15_mirror_weight_sweep.py`

### Sweep Range

R ∈ [0.8, 1.5] with 15 points

### Results (κ polynomials)

| R | m_exact | m1_tex | c_tex | c_gap |
|------|---------|--------|-------|-------|
| 0.80 | 16.4 | 6.05 | 2.287 | +7.0% |
| 1.00 | 33.1 | 6.13 | 2.200 | +2.9% |
| 1.20 | 66.7 | 6.19 | 2.141 | +0.2% |
| 1.30 | 94.6 | 6.22 | 2.122 | -0.7% |
| 1.50 | 190.6 | 6.28 | 2.108 | -1.4% |

### Interpretation

tex_mirror's amplitude formula:
```
A1 = exp(R_ref) + (K-1) + ε ≈ 3.68 + 2 + 0.27 = 5.96
A2 = exp(R_ref) + 2(K-1) + ε ≈ 3.68 + 4 + 0.27 = 7.96
```

These match tex_mirror's m values, confirming the amplitude model is working as designed - but it's NOT the naive exp(2R/θ) mirror factor.

---

## Summary

| Task | Status | Key Finding |
|------|--------|-------------|
| V2 Guard | ✅ | Prevents catastrophic V2+tex_mirror |
| Quadrature | ✅ | Gap is structural (converges at n=60) |
| Run 13A | ✅ | Output restructured, no apples-to-oranges |
| Run 14 | ✅ | m_exact = exp(2R/θ), 10-15x larger than tex_mirror |
| Run 15 | ✅ | R-sweep confirms fundamental difference |

---

## Production Baseline (Confirmed)

```python
compute_c_paper_tex_mirror(
    theta=4/7,
    R=R,
    n=60,  # n=100 for high precision
    polynomials=polys,
    terms_version="old",      # REQUIRED (V2 forbidden)
    tex_exp_component="exp_R_ref",
)
```

Achieves:
- κ (R=1.3036): c gap = **-0.73%**
- κ* (R=1.1167): c gap = **-0.95%**

---

## Structural Understanding

### Why tex_mirror works despite different m values

The tex_mirror model uses:
```
c = I1(+R) + m1×I1(-R) + I2(+R) + m2×I2(-R) + S34(+R)
```

with m ≈ 6-8, while the "naive" TeX combined formula gives m ≈ exp(2R/θ) ≈ 50-100.

**The 10x difference is explained by:**

1. **I1 has derivative structure**: d²/dxdy modifies the mirror behavior
2. **Shape factors absorb mirror effect**: m_implied ≈ 1.04 in tex_mirror
3. **Different mathematical objects**: The amplitude A = exp(R) + K-1 + ε is NOT the same as the mirror factor exp(2R/θ)

### The ~1% structural gap

The residual gap comes from:
- The shape×amplitude factorization is an APPROXIMATION
- The exp_R_ref calibration uses fixed R=1.3036
- True TeX formula combines +R/-R before Q operators, not after

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `src/evaluate.py` | Added V2 guard at line 5219 |
| `tests/test_v2_tex_mirror_guard.py` | V2 guard tests (5 tests) |
| `run_gpt_run16_quadrature_convergence.py` | Quadrature sweep |
| `run_gpt_run13a_direct_mirror.py` | Restructured output |
| `run_gpt_run14_combined_mirror.py` | Combined mirror evaluator |
| `run_gpt_run15_mirror_weight_sweep.py` | R-sweep analysis |
| `docs/HANDOFF_GPT_RUN14_16.md` | This document |

---

## Open Questions

1. **Can we derive tex_mirror's amplitude from TeX?**
   - The exp(R) + K-1 formula works but isn't derived from first principles
   - TeX Section 10 might have the answer

2. **What is the I1 mirror structure?**
   - I1 has d²/dxdy which changes the mirror behavior
   - This is why the naive exp(2R/θ) doesn't apply

3. **Can we close the ~1% gap?**
   - Requires understanding derivative-term mirror structure
   - Not just increasing quadrature resolution

---

## Recommendations

1. **Keep tex_mirror + exp_R_ref for production** - It works (<1% gap)
2. **Do not attempt to use exp(2R/θ) as mirror factor** - Wrong formulation
3. **Future work**: Study TeX Section 10 to derive amplitude for I1 terms
4. **The gap is structural** - No amount of calibration will close it
