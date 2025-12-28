# Phase 41 Findings: Residual Attribution Analysis

**Created**: 2025-12-27
**Status**: Complete

## Executive Summary

Phase 41 implemented GPT's recommended error budget attribution analysis to determine whether the ±0.15% residual comes from **m** (mirror multiplier) or **S34** (I₃+I₄ terms).

**Key Finding**: Both m and S34 have **opposite-sign shifts** between κ and κ* benchmarks. This confirms Phase 40's conclusion that no single correction can eliminate the residual for both benchmarks.

**Recommendation**: Accept ±0.15% as production accuracy OR implement a polynomial-aware derived functional g(P,Q,R,K,θ).

---

## Phase 41.1: Error Budget Attribution

### Component Values

| Benchmark | S12+ | S12- | S34 | m_derived |
|-----------|------|------|-----|-----------|
| κ | 0.797477 | 0.220121 | -0.600152 | 8.800660 |
| κ* | 0.614582 | 0.216444 | -0.443398 | 8.164345 |

### Computed vs Target

| Benchmark | c_computed | c_target | c_gap % |
|-----------|------------|----------|---------|
| κ | 2.13453265 | 2.13745441 | **-0.1367%** |
| κ* | 1.93830624 | 1.93795241 | **+0.0183%** |

**Observation**: κ undershoots the target, κ* overshoots.

### M Attribution: What m would hit target?

| Benchmark | m_derived | m_needed | delta_m % |
|-----------|-----------|----------|-----------|
| κ | 8.800660 | 8.813933 | **+0.1508%** |
| κ* | 8.164345 | 8.162711 | **-0.0200%** |

**Observation**: κ needs m **increased** by +0.15%, κ* needs m **decreased** by -0.02%.
**Signs are OPPOSITE.**

### S34 Attribution: What S34 would hit target?

| Benchmark | S34 | S34_needed | delta_S34 | delta % |
|-----------|-----|------------|-----------|---------|
| κ | -0.600152 | -0.597230 | +0.00292175 | -0.4868% |
| κ* | -0.443398 | -0.443752 | -0.00035383 | +0.0798% |

**Observation**: κ needs S34 **less negative** (increase), κ* needs S34 **more negative** (decrease).
**Signs are OPPOSITE.**

---

## Phase 41.2: Component Quadrature Convergence

### Convergence Results

All components are converged to **machine precision** (deltas ~10^-15).

| Benchmark | Component | n=40→60 delta | n=120→160 delta | Converged |
|-----------|-----------|---------------|-----------------|-----------|
| κ | S12_plus | 3.62e-15 | 2.65e-15 | YES |
| κ | S12_minus | 1.11e-14 | 8.32e-15 | YES |
| κ | S34 | 7.40e-15 | 1.11e-14 | YES |
| κ* | S12_plus | 9.03e-16 | 4.88e-15 | YES |
| κ* | S12_minus | 8.98e-15 | 1.85e-14 | YES |
| κ* | S34 | 6.01e-15 | 9.01e-15 | YES |

**Conclusion**: The ±0.15% residual is NOT due to numerical quadrature error. All components are stable at n=40.

---

## Decision Gate Analysis

### Shift Direction Summary

| Benchmark | delta_m | delta_S34 |
|-----------|---------|-----------|
| κ | **+0.1508%** | **+0.00292** |
| κ* | **-0.0200%** | **-0.00035** |

- **M shifts same direction?** NO (opposite signs)
- **S34 shifts same direction?** NO (opposite signs)

### Interpretation

**Both m and S34 have opposite-sign shifts between benchmarks.**

This means:
1. No single scalar correction δ_m can work for both benchmarks
2. No single scalar correction δ_S34 can work for both benchmarks
3. The residual is benchmark-specific, not a systematic error

---

## Conclusions

### What Phase 41 Proves

1. **Numerics are not the issue**: All components converge to machine precision at n=40
2. **No universal correction exists**: m and S34 both need opposite adjustments for κ vs κ*
3. **Phase 40's conclusion is confirmed**: A single analytical δ_Q formula cannot eliminate the residual

### Why Opposite Signs Occur

The κ and κ* polynomials have fundamentally different structures:
- κ: Q has degree 5, complex structure
- κ*: Q has degree 1, simple linear structure

This causes the interaction between P and Q polynomials in the integrands to produce different error directions.

### Recommended Next Steps

**Option 1: Accept Current Accuracy (Recommended for Production)**
- ±0.15% is already 10× better than the empirical formula
- Sufficient for κ optimization goals
- No further development needed

**Option 2: Polynomial-Aware Derived Functional**
- Implement g(P,Q,R,K,θ) that varies by polynomial structure
- Compute g from first-principles integrals (not targets)
- Gate: g must equal (1+θ/(2K(2K+1))) when Q=1
- This would allow κ and κ* to get different corrections while remaining first-principles

**Option 3: Investigate I₁/I₂ Ratio Imbalance**
- The ratio S12+/S12- differs between benchmarks
- May indicate structure in how m should be applied differently

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/run_phase41_residual_budget.py` | Error budget attribution analysis |
| `scripts/run_phase41_component_convergence.py` | Quadrature convergence sweep |
| `tests/test_phase41_error_budget.py` | 13 tests validating formulas |
| `docs/PHASE_41_FINDINGS.md` | This document |

## Test Results

```
tests/test_phase41_error_budget.py: 13 passed
```

---

## Raw Data for GPT

### For GPT to Analyze

```
KAPPA BENCHMARK (R=1.3036):
  S12_plus  = 0.797477
  S12_minus = 0.220121
  S34       = -0.600152
  m_derived = 8.800660
  c_computed = 2.13453265
  c_target   = 2.13745441
  c_gap      = -0.1367%

  m_needed   = 8.813933
  delta_m    = +0.1508%

  S34_needed = -0.597230
  delta_S34  = +0.00292175

KAPPA* BENCHMARK (R=1.1167):
  S12_plus  = 0.614582
  S12_minus = 0.216444
  S34       = -0.443398
  m_derived = 8.164345
  c_computed = 1.93830624
  c_target   = 1.93795241
  c_gap      = +0.0183%

  m_needed   = 8.162711
  delta_m    = -0.0200%

  S34_needed = -0.443752
  delta_S34  = -0.00035383

DECISION GATE:
  M shifts opposite sign: YES (κ needs increase, κ* needs decrease)
  S34 shifts opposite sign: YES (κ needs less negative, κ* needs more negative)

  CONCLUSION: No single correction can work for both benchmarks
  RECOMMENDATION: Accept ±0.15% OR implement polynomial-aware functional
```

---

## Phase 41.3: I1/I2 Component Attribution (NEW)

### Key Finding: I1 and I2 Scale Oppositely!

| Component | κ value | κ* value | Ratio (κ/κ*) |
|-----------|---------|----------|--------------|
| c_I1 | 0.536049 | 0.696875 | **0.7692** |
| c_I2 | 2.198635 | 1.684830 | **1.3050** |
| c_S12 | 2.734684 | 2.381704 | 1.1482 |

**Critical Observation**: I1 and I2 scale in **opposite directions**:
- c_I1: κ has **less** than κ* (ratio 0.77)
- c_I2: κ has **more** than κ* (ratio 1.31)

This 69% difference in scaling behavior explains why:
1. κ needs m increased (I2-dominated → needs more mirror contribution)
2. κ* needs m decreased (I1 has higher fraction → different balance)

### I1/I2 Fraction Breakdown

| Benchmark | I1 fraction of c_S12 | I2 fraction of c_S12 |
|-----------|---------------------|---------------------|
| κ | 19.60% | 80.40% |
| κ* | 29.26% | 70.74% |

κ* has a higher I1 fraction (29% vs 20%), which changes the balance.

---

## Infrastructure Implemented

### New Files

| File | Purpose |
|------|---------|
| `scripts/run_phase41_i1_i2_attribution.py` | I1/I2 component breakdown |
| `src/evaluator/g_functional.py` | Polynomial-aware g(P,Q,R,K,θ) |
| `tests/test_g_functional.py` | 10 validation tests |

### New Features

1. **`compute_g_functional()`**: Computes polynomial-aware correction factor
2. **`validate_g_functional_Q1_gate()`**: Validates g=baseline when Q=1
3. **`derived_functional` mode**: Added to `compute_mirror_multiplier()`

---

## Questions for GPT

1. **Route A Implementation**: Given that I1 and I2 scale oppositely (0.77 vs 1.31), how should g(P,Q,R,K,θ) weight the I1 and I2 contributions? Should we compute:
   ```
   g = g_baseline * f(I1_fraction, I2_fraction)
   ```
   where f adjusts based on the I1/I2 balance?

2. **I1/I2 Ratio Insight**: The I1 ratio (0.77) and I2 ratio (1.31) bracket 1.0. This suggests:
   - I1 is **under-represented** in κ relative to κ*
   - I2 is **over-represented** in κ relative to κ*

   Is there a polynomial-structure explanation for why I1 (derivative term) scales down while I2 (non-derivative term) scales up?

3. **S12 Ratio Analysis**: The ratio S12+/S12- differs between benchmarks:
   - κ: 0.797477/0.220121 = 3.623
   - κ*: 0.614582/0.216444 = 2.839

   Does this 27% difference suggest m should incorporate the S12 ratio?

4. **Physical Interpretation**: κ has degree-5 Q (complex), κ* has degree-1 Q (linear). Does the Q polynomial's complexity affect the I1/I2 balance? I1 involves d²/dxdy derivatives while I2 has no derivatives - could Q's derivative structure be the driver?

5. **K=4 Implications**: Given the benchmark-specific nature of the residual, should we expect similar ±0.15% accuracy for K=4, or might the K-dependence change the picture?
