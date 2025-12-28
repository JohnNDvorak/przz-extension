# Phase 25/26: Factor-of-3 Investigation Summary

**Date:** 2025-12-25
**Status:** Investigation Complete, Partial Fix Applied
**Key Finding:** Missing `(1-u)^{ℓ₁+ℓ₂}` prefactor in unified evaluator

---

## Executive Summary

Phase 25 identified a **factor-of-3 discrepancy** between the unified S12 evaluator and the empirical DSL evaluator in the P=Q=1 microcase. Investigation revealed the root cause: the unified bracket was missing the `(1-u)^{ℓ₁+ℓ₂}` prefactor required by PRZZ.

**After fix:**
- P=Q=1 microcase for (1,1): ratio = 1.0 (was 3.0) ✓
- Full evaluation with actual polynomials: (1,1) matches, other pairs have structural differences

---

## 1. The Factor-of-3 Discovery (Phase 25.3)

### Original Finding

The P=Q=1 microcase ladder revealed:

| Benchmark | Unified I1 | Empirical I1 | Ratio |
|-----------|------------|--------------|-------|
| kappa     | 4.678e+00  | 1.559e+00    | **3.0** |
| kappa*    | 2.658e+00  | 8.861e-01    | **3.0** |

The factor of 3 was **constant across both benchmarks**, indicating a structural issue in the bracket, not polynomial interactions.

### Root Cause Analysis

**Empirical DSL Path** (`src/terms_k3_d1.py`):
```python
# make_I1_11():
poly_prefactors=[make_poly_prefactor_11()]  # Returns (1-u)²
```

**Unified Bracket** (`src/unified_s12_evaluator_v3.py`):
- Builds: `exp(...) × (1/θ + x + y) × P_factors × Q_factors`
- **NO `(1-u)^{ℓ₁+ℓ₂}` factor anywhere**

### Mathematical Verification

The average of `(1-u)²` over [0,1]:
```
∫₀¹ (1-u)² du = [-(1-u)³/3]₀¹ = 1/3
```

**Impact:**
- Empirical path: applies `(1-u)²` → integrand weighted by ~1/3
- Unified path: NO such factor → integrand weighted by ~1
- **Ratio: unified / empirical = 1 / (1/3) = 3.0** ✓

---

## 2. The Fix

### Code Change

Modified `compute_I1_unified_v3()` in `src/unified_s12_evaluator_v3.py`:

```python
# PRZZ (1-u) power from Euler-Maclaurin: (1-u)^{ℓ₁+ℓ₂} for I₁
one_minus_u_power = ell1 + ell2

total = 0.0
for u, u_w in zip(u_nodes, u_weights):
    # Compute (1-u)^{ℓ₁+ℓ₂} prefactor
    one_minus_u_factor = (1.0 - u) ** one_minus_u_power

    for t, t_w in zip(t_nodes, t_weights):
        series = build_unified_bracket_series(...)
        xy_coeff = series.coeffs.get(xy_mask, 0.0)
        # Apply (1-u)^{ℓ₁+ℓ₂} prefactor
        total += xy_coeff * one_minus_u_factor * u_w * t_w
```

### Verification

After fix, P=Q=1 microcase:
```
KAPPA (R=1.3036):
  Unified I1:    1.559481e+00
  Empirical I1:  1.559481e+00
  Relative diff:       0.0000%
  Agreement:     True

KAPPA* (R=1.1167):
  Unified I1:    8.860791e-01
  Empirical I1:  8.860791e-01
  Relative diff:       0.0000%
  Agreement:     True

RATIO ANALYSIS:
  kappa:   unified/empirical = 1.000000
  kappa*:  unified/empirical = 1.000000
```

---

## 3. Per-Pair Analysis with Actual Polynomials

### Comparison: Unified vs OLD DSL

| Pair | Unified | OLD DSL | Ratio | Notes |
|------|---------|---------|-------|-------|
| 11   | 0.4129  | 0.4135  | 0.999 | ✓ Match |
| 22   | 0.5591  | 3.8839  | 0.144 | 7x gap |
| 33   | 0.0283  | 2.8610  | 0.010 | 100x gap |
| 12   | 0.4136  | -0.5650 | -0.73 | Sign flip |
| 13   | 0.0383  | -0.5816 | -0.07 | Sign flip |
| 23   | 0.0938  | 3.5715  | 0.026 | 38x gap |

**Root Cause:** The unified evaluator and OLD DSL use **different mathematical structures**:

| Aspect | Unified | OLD DSL |
|--------|---------|---------|
| Variables | 2 (x, y) | ℓ₁+ℓ₂ variables |
| Derivative | d²/dxdy | d^{ℓ₁+ℓ₂}/dx₁...dy_{ℓ₂} |
| (1-u) power | ℓ₁+ℓ₂ | ℓ₁+ℓ₂ |

For (1,1), both happen to extract the same quantity (d²/dxdy with power 2).
For other pairs, they extract **fundamentally different** mathematical quantities.

### Comparison: Unified vs V2 DSL

The V2 DSL also uses 2-variable structure but with different (1-u) power:

| Pair | Unified Power | V2 Power | Formula |
|------|---------------|----------|---------|
| 11   | 2             | 2*       | V2 special-cases (1,1) |
| 22   | 4             | 2        | V2: (ℓ₁-1)+(ℓ₂-1) |
| 33   | 6             | 4        | |
| 12   | 3             | 1        | |
| 13   | 4             | 2        | |
| 23   | 5             | 3        | |

*V2 uses `max(0, (ℓ₁-1) + (ℓ₂-1))` except for (1,1) which is hardcoded to 2.

---

## 4. Understanding the Three DSL Structures

### OLD DSL (Production, "TeX-Truth")
- Uses ℓ₁+ℓ₂ variables for pair (ℓ₁, ℓ₂)
- Extracts d^{ℓ₁+ℓ₂}/dx₁...dx_{ℓ₁}dy₁...dy_{ℓ₂}
- (1-u) power: ℓ₁+ℓ₂
- **Status:** Works with mirror assembly, matches PRZZ targets

### V2 DSL (Simplified, Problematic)
- Uses 2 variables (x, y) for all pairs
- Extracts d²/dxdy
- (1-u) power: max(0, (ℓ₁-1)+(ℓ₂-1))
- **Status:** Breaks under mirror assembly (sign flips)

### Unified Bracket (After Fix)
- Uses 2 variables (x, y) for all pairs
- Extracts d²/dxdy
- (1-u) power: ℓ₁+ℓ₂ (OLD formula, not V2 formula)
- **Status:** Matches (1,1), differs from both OLD and V2 for other pairs

---

## 5. Key Insights

### 5.1 The (1-u) Power Has Two Roles

1. **Euler-Maclaurin contribution:** From converting sums to integrals
2. **Derivative order compensation:** Higher derivatives extract higher polynomial derivatives

In the OLD DSL, both roles align: ℓ₁+ℓ₂ variables with ℓ₁+ℓ₂ power.

In the 2-variable unified structure, the derivative order is always 2, so the (1-u) power may need adjustment to compensate.

### 5.2 Why V2 Uses Different Power

V2 uses `(ℓ₁-1)+(ℓ₂-1)` because:
- The 2-variable d²/dxdy extracts P'(u)×P'(u)
- The OLD ℓ₁+ℓ₂-variable derivative extracts P^{(ℓ₁)}(u)×P^{(ℓ₂)}(u)
- The (1-u) power difference compensates for the derivative difference

### 5.3 Why V2 Breaks Under Mirror Assembly

From `docs/TEX_VERIFICATION_1_MINUS_U.md`:
> V2's different (1-u) power formula causes I1_plus to flip sign from positive (OLD) to negative (V2). This single change collapses the entire assembly.

---

## 6. Current Status

### Fixed
- [x] P=Q=1 microcase for (1,1) matches (ratio = 1.0)
- [x] Individual I1 for (1,1) with actual polynomials matches (ratio = 0.999)

### Outstanding Issues
- [ ] Other pairs (2,2), (3,3), (1,2), (1,3), (2,3) have large discrepancies
- [ ] Unified uses OLD's (1-u) power with V2's derivative structure - hybrid approach
- [ ] Gap attribution still shows large differences at S12 level

### Possible Paths Forward

1. **Match V2 structure completely:**
   - Change unified (1-u) power to `max(0, (ℓ₁-1)+(ℓ₂-1))`
   - Will match V2 DSL but NOT production (OLD)
   - May require new mirror assembly formula

2. **Accept structural difference:**
   - Unified computes a different (but potentially equivalent) quantity
   - Develop transformation between unified and OLD results
   - Use unified for theoretical analysis, OLD for production

3. **Derive the correct 2-variable formula from PRZZ:**
   - Go back to PRZZ Section 7 derivation
   - Determine the mathematically correct (1-u) power for 2-variable structure
   - May resolve V2 vs unified discrepancy

---

## 7. PRZZ Reference

### (1-u) Power in PRZZ TeX

| Term | PRZZ TeX Line | Power Pattern |
|------|---------------|---------------|
| I₁   | 1435          | (1-u)^{ℓ₁+ℓ₂} |
| I₂   | 1548          | none |
| I₃   | 1484          | (1-u)^{ℓ₁} |
| I₄   | 1488          | (1-u)^{ℓ₂} |

### Key PRZZ Lines
- **Lines 2391-2396:** Euler-Maclaurin derivation of (1-u) factor
- **Lines 1502-1511:** Difference quotient identity (unified bracket basis)
- **Lines 1529-1533:** I₁ formula structure

---

## 8. Files Modified

### Modified
- `src/unified_s12_evaluator_v3.py`: Added (1-u)^{ℓ₁+ℓ₂} factor in `compute_I1_unified_v3()`

### Created (Phase 25)
- `src/evaluator/gap_attribution.py`
- `src/unified_s12_microcases.py`
- `scripts/run_phase25_gap_attribution.py`
- `tests/test_phase25_gap_attribution.py`
- `tests/test_phase25_microcases.py`
- `tests/test_phase25_eigenvalue_mapping.py`
- `docs/PHASE_25_SUMMARY.md`
- `docs/PHASE_25_26_FACTOR_OF_3_INVESTIGATION.md` (this file)

---

## 9. Recommendations for Phase 26

Based on GPT's guidance, Phase 26 should:

1. **Define microcase contract explicitly** - clarify what unified vs empirical should compute
2. **Add I2 P=Q=1 microcase** - verify I2 ratio (may be 1/3, canceling the I1 factor of 3)
3. **Add debug breakdown outputs** - u_weight_integral, denominator placement, log-factor split
4. **Run pinpoint tests** - profile-weight test, denom placement test, factorial norm test
5. **Determine correct (1-u) formula** - for 2-variable structure from first principles

---

*Investigation completed 2025-12-25*
