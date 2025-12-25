# Phase 8 Summary: m₁ Derivation Investigation

**Date:** 2025-12-23
**Purpose:** Complete summary of Phase 8 findings for GPT analysis
**Context:** PRZZ κ-optimization for Riemann zeta zeros on critical line

---

## Executive Summary

We have an empirical mirror multiplier m₁ that achieves ~1.35% accuracy on benchmark targets, but we cannot derive it from first principles. Phase 8 was designed to investigate this gap.

**The Core Problem:**
```
Empirical:  m₁ = exp(R) + 5        → -1.35% gap (undershoot)
Fitted:     m₁ = 1.037×exp(R) + 5  → 0.00% gap (exact match)
```

**Key Discovery:** The 3.7% correction factor (a ≈ 1.037) is **structural, not numerical**. It is stable across quadrature precision levels n = 40, 60, 80, 100.

---

## 1. The Mirror Assembly Formula

The main computational formula is:
```
c = S12(+R) + m₁ × S12(-R) + S34(+R)
```

Where:
- `S12 = I₁ + I₂` (terms requiring mirror assembly per PRZZ TeX Section 10)
- `S34 = I₃ + I₄` (terms NOT requiring mirror)
- `m₁` is the mirror multiplier (empirically exp(R) + 5 for K=3)
- R is the shift parameter (R = 1.3036 for κ benchmark, R = 1.1167 for κ* benchmark)

**PRZZ TeX Prediction:**
- Section 10 says mirror assembly is: `I(α,β) + T^{-α-β} × I(-β,-α)`
- At α = β = -R/L: `T^{-α-β} = exp(2R) ≈ 13.56` for R = 1.3036

**Actual Empirical Value:**
- m₁ = exp(R) + 5 ≈ 8.68 for R = 1.3036
- This is NOT exp(2R)!

**The Mystery:** Why does m₁ = exp(R) + 5 work instead of exp(2R)?

---

## 2. Phase 8 Completed Tasks

### Phase 8.0: S34 Triangle Convention Lock ✓
- S34 uses triangle×2 convention (6 pairs with symmetry factor)
- NOT 9 ordered pairs (which caused +11% overshoot)
- Enforced by `tests/test_s34_triangle_spec_lock.py`

### Phase 8.1: Quarantine Fitted m₁ ✓
- Added `M1Mode.DIAGNOSTIC_FITTED` to `src/m1_policy.py`
- Hard guard: raises `M1DiagnosticError` if used without `allow_diagnostic=True`
- Prevents "calibration creep" — using fitted values that mask derivation problems
- 10 tests in `tests/test_no_fitted_m1_in_production_paths.py`

### Phase 8.2: Semantic Channel Diagnostics ✓
- Created `run_channel_diagnostics_v2.py`
- Computes per-channel and per-pair breakdown
- **Key finding:** m₁_ideal/m₁_empirical ≈ 1.015 (NOT 2.6× as previously thought)
- The "2.6× mystery" was an artifact of the S34 bug (using ordered pairs)

### Phase 8.3a: Object Identification ✓
- Created `run_identify_minus_basis.py`
- **Finding:** DSL "minus basis" is the SAME integral formula with R sign-flipped
- It is NOT directly the TeX mirror term I(-β,-α)
- The exp_factors.scale changes sign: +R → -R

### Phase 8.3d: Pair Consistency Test ✓
- Created `tests/test_m1_eff_pair_consistency.py`
- m₁_eff spread across pairs: **~1.6-2.2%** (where spread = (max-min)/mean)
  - κ benchmark: 2.2% spread (values 2.79-2.86, mean 2.82)
  - κ* benchmark: 1.6% spread (values 2.25-2.29, mean 2.27)
- Scalar m₁ is a reasonable approximation (low spread validates scalar architecture)

### Phase 8.5: Convert XFAIL to Real Test ✓
- Added `TestEmpiricalBaselineGate` to `tests/test_two_benchmark_gate.py`
- 3 real tests (not XFAIL) that enforce:
  - κ gap within [-3%, +0.5%]
  - κ* gap within [-3%, +0.5%]
  - Ratio error within ±1%

---

## 3. Numerical Results

### 3.1 Benchmark Gaps with Empirical m₁

| Benchmark | R | c_target | c_computed | Gap |
|-----------|------|----------|------------|-----|
| κ | 1.3036 | 2.13745 | 2.1087 | **-1.35%** |
| κ* | 1.1167 | 1.93795 | 1.9146 | **-1.21%** |
| Ratio | - | 1.1029 | 1.1014 | **-0.14%** |

### 3.2 The "a" Coefficient Analysis

The ideal m₁ can be written as: `m₁_ideal = a × exp(R) + b`

| Benchmark | a_coefficient | b_coefficient |
|-----------|---------------|---------------|
| κ | 1.0357 | 4.9938 |
| κ* | 1.0353 | 4.9938 |

**Empirical uses:** a = 1.0, b = 5.0

### 3.3 Stability Across Quadrature Precision

| n | a_kappa | a_kappa_star |
|---|---------|--------------|
| 40 | 1.035715 | 1.035296 |
| 60 | 1.035715 | 1.035296 |
| 80 | 1.035715 | 1.035296 |
| 100 | 1.035715 | 1.035296 |

**Conclusion:** The 3.7% correction is **STRUCTURAL**, not a numerical artifact.

### 3.4 m₁_eff Per-Pair Analysis

m₁_eff = S12_plus / S12_minus for each pair:

**κ benchmark (R=1.3036):**
| Pair | m₁_eff |
|------|--------|
| 11 | 2.8036 |
| 12 | 2.8283 |
| 13 | 2.7975 |
| 22 | 2.8562 |
| 23 | 2.8355 |
| 33 | 2.7947 |
| Mean | 2.8193 |
| Spread | 2.2% |

**κ* benchmark (R=1.1167):**
| Pair | m₁_eff |
|------|--------|
| 11 | 2.2573 |
| 12 | 2.2710 |
| 13 | 2.2541 |
| 22 | 2.2872 |
| 23 | 2.2746 |
| 33 | 2.2520 |
| Mean | 2.2660 |
| Spread | 1.6% |

**Observation:** The per-pair ratios are very consistent (~2% spread), suggesting scalar m₁ is well-justified.

---

## 4. What the DSL Minus Basis Actually Computes

### 4.1 Term Structure Comparison

For pair (1,1), I₁ term:

**+R branch:**
```
name: I1_11
pair: (1, 1)
vars: ('x', 'y')
deriv_orders: {'x': 1, 'y': 1}
numeric_prefactor: 1.0
exp_factors: scale = +R (positive)
```

**-R branch:**
```
name: I1_11
pair: (1, 1)
vars: ('x', 'y')
deriv_orders: {'x': 1, 'y': 1}
numeric_prefactor: 1.0
exp_factors: scale = -R (negative)
```

### 4.2 Key Insight

The DSL minus basis is computed by:
1. Building terms with `kernel_regime='paper'` at R = -R
2. Evaluating the SAME integrand structure but with R sign-flipped

This is **NOT** the TeX mirror term I(-β,-α) directly. The TeX mirror involves:
1. Swapping (α,β) → (-β,-α)
2. Multiplying by T^{-α-β} = exp(2R)

Our DSL minus branch only does the R sign flip, not the full TeX mirror transform.

### 4.3 Ratio Analysis

| Quantity | κ (R=1.3036) | κ* (R=1.1167) |
|----------|--------------|---------------|
| S12(+R)/S12(-R) | 2.82 | 2.27 |
| exp(2R) | 13.56 | 9.37 |
| exp(R) | 3.68 | 3.05 |

**The ratio is closer to exp(R) than exp(2R)!**

This suggests the Q polynomial contributions break the simple exp(2R) relationship.

---

## 5. The "+5" Mystery

The empirical formula has m₁ = exp(R) + 5, where 5 = 2K - 1 for K = 3.

**Possible interpretations:**
- (2K - 1) = 2×3 - 1 = 5
- (K + 2) = 3 + 2 = 5
- (K!) - 1 = 6 - 1 = 5
- Coincidence

**No derivation exists for why "+5" appears.**

Only K=3 has been validated. The pattern (2K-1) is extrapolated for K>3 but untested.

---

## 6. What Remains Unknown

### 6.1 The 3.7% Structural Factor

The ideal m₁ has a ≈ 1.037 instead of a = 1.0.

**Possible sources:**
1. Missing θ-dependent normalization factor
2. Chain rule effects from the (-β,-α) substitution (though GPT demoted this since at α=β the swap is trivial)
3. A structural factor in the combined identity that our DSL doesn't capture
4. Polynomial-degree-dependent normalization we're missing

### 6.2 Why m₁ ≠ exp(2R)

PRZZ TeX Section 10 predicts T^{-α-β} = exp(2R) as the mirror multiplier.

Our empirical m₁ = exp(R) + 5 ≈ 8.68 is NOT exp(2R) ≈ 13.56.

**GPT's Hypothesis (from prior session):**
> "TeX mirror factor is exp(2R), but our '−R branch term' is NOT proven equal to I(-β,-α) in TeX's sense. The +49% gap with exp(2R) indicates a BASIS/SEMANTICS MISMATCH, not that TeX's factor is wrong."

### 6.3 The Relationship Between DSL -R and TeX Mirror

Three possibilities (from Phase 8.3a):

**Option A:** DSL minus = I evaluated at -R in TeX sense
- Partially true — it's I evaluated at -R, but TeX mirror includes additional structure

**Option B:** DSL minus = base object with some factor stripped
- No evidence for this — we use full exp factors

**Option C:** DSL minus = computational proxy requiring empirical calibration
- Most likely — it captures PART of the mirror effect, with m₁ absorbing the rest

---

## 7. Failed Derivation Attempts

| Attempt | Approach | Result | Why It Failed |
|---------|----------|--------|---------------|
| Finite-L | Combined identity at finite L | m₁_eff ≈ -9.15×L (diverges) | L-dependence not controlled |
| Unified-t | Unify t-parameters across branches | 24% amplification (need 8-9×) | Wrong object level |
| Q-shift | T^{-s}Q(D_α)F = T^{-s}Q(1+D_α)F | 85-127× (way too large) | Applied to wrong basis |
| exp(2R) direct | Use T^{-α-β} = exp(2R) literally | +49% gap | Basis/semantics mismatch |

---

## 8. Hypothesis for Investigation

### Hypothesis: The Missing Factor is in Q Polynomial Evaluation

The Q polynomial is evaluated at different points for +R vs -R:
```
Q(1 + A_α) where A_α = t + θ(t-1)x + θt·y
```

When R → -R:
- The eigenvalue structure changes
- The xy-coefficient extraction picks up different contributions
- This could explain why the ratio is ~2.8 instead of ~13.6

### Test: Compute Q Contribution Separately

If we could isolate the Q polynomial contribution to the +R/-R ratio:
```
ratio_Q = Q_contribution(+R) / Q_contribution(-R)
ratio_exp = exp(2R)
ratio_total = ratio_Q × ratio_exp  (should equal ~2.8)
```

This would tell us if Q absorbs a factor of ~exp(R)/ratio_total ≈ 4.8.

---

## 9. Code References

### Key Files

| File | Purpose |
|------|---------|
| `src/m1_policy.py` | m₁ mode selection and guards |
| `src/evaluate.py` | Main evaluator with `compute_c_paper_ordered()` |
| `src/terms_k3_d1.py` | Term DSL for K=3 |
| `run_channel_diagnostics_v2.py` | Semantic channel diagnostics |
| `run_identify_minus_basis.py` | Object identification for minus basis |

### Key Functions

```python
# Production evaluator (uses empirical m₁)
compute_c_paper_ordered(theta, R, n, polynomials, K, s12_pair_mode="triangle")

# Mirror multiplier formula
m1_formula(K, R, policy)  # Returns exp(R) + (2K-1) for K3_EMPIRICAL mode

# Diagnostic fitted (quarantined)
m1_diagnostic_fitted(R)  # Returns 1.037×exp(R) + 5 with warning
```

### Key Tests

```python
# Phase 8.1: Production path guards
tests/test_no_fitted_m1_in_production_paths.py  # 10 tests

# Phase 8.3d: Pair consistency
tests/test_m1_eff_pair_consistency.py  # 5 tests

# Phase 8.5: Real baseline tests
tests/test_two_benchmark_gate.py::TestEmpiricalBaselineGate  # 3 tests
```

---

## 10. Open Questions for GPT

1. **Why is a ≈ 1.037 instead of 1.0?**
   - The coefficient is stable across quadrature, so it's structural
   - What mathematical factor would produce exactly 1.037?
   - Is there a θ-dependent correction? (θ = 4/7 ≈ 0.5714)

2. **Why does the S12(+R)/S12(-R) ratio equal ~2.8 instead of exp(2R) ≈ 13.6?**
   - The Q polynomial evaluation must absorb a factor of ~4.8
   - How does Q(1+A_α) behave when R → -R?

3. **What is the correct theoretical relationship between DSL -R branch and TeX mirror?**
   - DSL computes I at -R
   - TeX mirror is T^{-α-β} × I(-β,-α)
   - What factor relates these?

4. **Is there a closed-form derivation for m₁ = exp(R) + (2K-1)?**
   - The "+5" for K=3 has no derivation
   - Is (2K-1) the correct pattern for general K?

5. **Can the 3.7% gap be explained by a known mathematical constant?**
   - 1.037 ≈ 1 + 1/27?
   - 1.037 ≈ exp(θ/16)?
   - 1.037 ≈ 1 + θ²/12? (No, that's 1.027)

---

## 11. Appendix: Raw Diagnostic Output

### Channel Diagnostics v2 (κ benchmark, n=60)

```
======================================================================
KAPPA Benchmark (R = 1.3036, n = 60)
======================================================================

--- TARGET ---
  c_target = 2.1374544061

--- CHANNEL TOTALS ---
  S12(+R) = +1.22174182
  S12(-R basis) = +0.43335058
  S34(+R) = +0.51047188

--- MIRROR MULTIPLIERS ---
  m_empirical = exp(R)+5 = 8.683109
  m_fitted = 1.037*exp(R)+5 = 8.819839
  m_needed (for 0% gap) = 8.813843
  Ratio m_needed/m_empirical = 1.015063

--- A COEFFICIENT ---
  m_needed = a × exp(R) + 5
  a = 1.035715  (empirical uses a=1.0, fitted uses a=1.0374)

--- GAPS ---
  Gap with m_empirical: -1.3540%
  Gap with m_fitted: -0.0068%
```

### Channel Diagnostics v2 (κ* benchmark, n=60)

```
======================================================================
KAPPA_STAR Benchmark (R = 1.1167, n = 60)
======================================================================

--- TARGET ---
  c_target = 1.9379524125

--- CHANNEL TOTALS ---
  S12(+R) = +1.02810813
  S12(-R basis) = +0.45363618
  S34(+R) = +0.43320588

--- MIRROR MULTIPLIERS ---
  m_empirical = exp(R)+5 = 8.055369
  m_fitted = 1.037*exp(R)+5 = 8.163262
  m_needed (for 0% gap) = 8.163048
  Ratio m_needed/m_empirical = 1.013372

--- A COEFFICIENT ---
  m_needed = a × exp(R) + 5
  a = 1.035296  (empirical uses a=1.0, fitted uses a=1.0374)

--- GAPS ---
  Gap with m_empirical: -1.2061%
  Gap with m_fitted: +0.0026%
```

---

## 12. Conclusion

Phase 8 has successfully:
1. Quarantined the fitted m₁ to prevent calibration creep
2. Identified that the 3.7% correction is structural (stable across n)
3. Shown that scalar m₁ is justified (low per-pair spread)
4. Established baseline regression tests with real (non-XFAIL) assertions
5. Identified that DSL minus basis ≠ TeX mirror directly

The remaining mystery is **why a ≈ 1.037** instead of 1.0, and **why m₁ = exp(R) + 5** instead of exp(2R).

The most promising hypothesis is that the Q polynomial evaluation absorbs a factor when R → -R, and the 3.7% represents a structural correction we haven't identified.
