# Phase 14C Summary: Main-Term Reductions for J₁₂–J₁₄

**Date:** 2025-12-23
**Status:** MAIN-TERM REDUCTIONS IMPLEMENTED, GATE TESTS IN PLACE

---

## Executive Summary

Phase 14C addressed the root cause of the B≈58 bridge discrepancy identified in Phase 14B:

**Problem:** J12-J14 were evaluated "literally" by multiplying raw ζ'/ζ values.

**Solution:** Implemented the paper's Laurent/contour-lemma reductions that:
1. Collapse J12 to Laurent coefficient extraction
2. Give J13/J14 leading MINUS signs from residue calculus
3. Provide cutoff-free Euler-Maclaurin integral forms

---

## Tasks Completed

### Task C1: Lock Sign Conventions ✓

**Files Modified:**
- `src/ratios/arithmetic_factor.py` - Added sign convention documentation
- `tests/test_arithmetic_factor_A11.py` - Added `test_A11_sign_convention_locked()`

**Key Decision:**
- `A11_prime_sum()` returns POSITIVE ~1.3856
- This matches PRZZ TeX Lines 1377-1389

### Task C2: Implement Laurent Series ✓

**New File:** `src/ratios/zeta_laurent.py`

**Functions Implemented:**
```python
def zeta_series(order) -> LaurentSeries           # ζ(1+s) = 1/s + γ + ...
def inv_zeta_series(order) -> Tuple               # 1/ζ(1+s) = s - γs² + ...
def zeta_logderiv_series(order) -> LaurentSeries  # ζ'/ζ(1+s) = -1/s + γ + ...
def inv_zeta_times_logderiv_series(order) -> Tuple
def logderiv_product_series(alpha, beta, order)   # For J12 reductions
```

**Key Insight:** The pole coefficient of ζ'/ζ is -1 (negative).

**Tests:** 18 tests in `tests/test_zeta_laurent.py`

### Task C3: Rebuild J12-J14 Main-Term Engines ✓

**File Modified:** `src/ratios/j1_k3_decomposition.py`

**New Functions:**
```python
def bracket_j12_main(...)  # Uses Laurent coefficient, not raw product
def bracket_j13_main(...)  # Has NEGATIVE sign from residue calculus
def bracket_j14_main(...)  # Has NEGATIVE sign (symmetric with J13)
def build_J1_pieces_K3_main_terms(...)  # Main-term builder
```

**Key Difference from Literal:**
- Literal J13: `+ (ζ'/ζ) × log(n) sum`
- Main-term J13: `- (ζ'/ζ) × log(n) sum` (NEGATIVE)

**Tests:** 11 tests in `tests/test_j1_k3_main_terms.py`

### Task C4: Euler-Maclaurin Integrals ✓

**New File:** `src/ratios/j1_euler_maclaurin.py`

**Functions Implemented:**
```python
def j11_as_integral(R, theta, ...)
def j12_as_integral(R, theta, ...)  # With 1/(α+β) factor
def j13_as_integral(R, theta, ...)  # With (1-u) weight and -1/θ prefactor
def j14_as_integral(R, theta, ...)
def j15_as_integral(R, theta, ...)
def compute_J1_as_integrals(R, theta, ...)
def compute_S12_from_J1_integrals(theta, R, ...)
```

**Note:** Uses simplified default polynomials. Full pipeline needs actual PRZZ polynomials.

### Task C5: +5 Gate Test ✓

**New File:** `tests/test_plus5_gate.py`

**Tests Implemented:**
- `test_j15_contributes_approximately_5()` - PASSING
- `test_main_term_j13_j14_are_negative()` - PASSING
- `test_literal_vs_main_term_difference()` - PASSING
- `test_constant_offset_is_5()` - XFAIL (needs PRZZ polynomials)
- `test_exp_coefficient_is_positive()` - XFAIL (needs PRZZ polynomials)

---

## Test Results

```
Phase 14C Tests: 47 passed, 2 xfailed
Phase 14B Tests: 40+ passed

Total New Tests: 49 (18 + 11 + 8 + 12)
```

The xfailed tests are EXPECTED to fail until the full pipeline connects
the main-term reductions to actual PRZZ polynomials.

---

## What Changed vs Phase 14B

### Before (Phase 14B)
```python
# bracket_j12 (literal):
zeta1 = zeta_log_deriv(1.0 + alpha + s)
zeta2 = zeta_log_deriv(1.0 + beta + u)
return A00 * dirichlet_sum * zeta1 * zeta2
```

### After (Phase 14C)
```python
# bracket_j12_main:
s_coeffs, u_coeffs = logderiv_product_series(alpha, beta)
c00 = s_coeffs[0] * u_coeffs[0]  # Laurent [s^0 u^0]
return c00 * dirichlet_sum / (alpha + beta)

# bracket_j13_main:
sign = -1  # FROM LAURENT REDUCTION
return sign * beta_logderiv * dirichlet_sum
```

---

## Key Mathematical Insights

### 1. Laurent Pole Structure
```
ζ'/ζ(1+s) = -1/s + γ_E + γ₁s + O(s²)
```
The -1/s pole drives the residue calculus.

### 2. J12 Main-Term Collapse
At PRZZ point α=β=-R:
```
(ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u) → (1/R + γ)² at s=u=0
```
The raw product becomes a constant (Laurent coefficient).

### 3. J13/J14 Sign Flip
The residue extraction gives a leading minus sign:
- Literal J13: positive
- Main-term J13: NEGATIVE

This is consistent with PRZZ I₃ prefactor -1/θ (TeX lines 1551-1564).

---

## Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `src/ratios/zeta_laurent.py` | CREATE | Laurent series machinery |
| `src/ratios/j1_euler_maclaurin.py` | CREATE | Euler-Maclaurin integrals |
| `src/ratios/arithmetic_factor.py` | MODIFY | Sign convention docs |
| `src/ratios/j1_k3_decomposition.py` | MODIFY | Main-term variants |
| `tests/test_zeta_laurent.py` | CREATE | 18 tests |
| `tests/test_j1_k3_main_terms.py` | CREATE | 11 tests |
| `tests/test_plus5_gate.py` | CREATE | 8 tests (2 xfail) |
| `tests/test_arithmetic_factor_A11.py` | MODIFY | Sign convention test |

---

## What's Still Needed

1. **Connect to PRZZ Polynomials:** The Euler-Maclaurin integrals use simplified default polynomials. Need to load actual PRZZ polynomial coefficients.

2. **Verify B ≈ 5:** The gate test `test_constant_offset_is_5()` is marked xfail. When it passes, Phase 14C is complete.

3. **Verify A ≈ 1:** The gate test `test_exp_coefficient_is_positive()` is marked xfail. Should flip from -2.89 to +O(1).

---

## Next Steps (Phase 14D?)

1. Load PRZZ polynomial coefficients into Euler-Maclaurin integrals
2. Verify the two gate tests pass
3. Compare bridge analysis with new main-term reductions
4. Document the final m₁ = exp(R) + 5 derivation

---

## Conclusion

Phase 14C successfully implemented the infrastructure for main-term reductions:

- **Laurent series machinery** for coefficient extraction
- **Main-term bracket functions** with correct signs
- **Euler-Maclaurin integral forms** without cutoff artifacts
- **Gate tests** that will verify B ≈ 5 when pipeline is complete

The "+5" signal from J15 is preserved. The excess contribution from J11-J14
should now cancel once the full polynomial pipeline is connected.

**87+ tests passing with main-term reductions in place.**
