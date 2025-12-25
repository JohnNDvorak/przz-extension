# Phase 14D Summary: Wire PRZZ Polynomials to +5 Gate Tests

**Date:** 2025-12-23
**Status:** WIRING COMPLETE, GATE TESTS STILL XFAIL

---

## Executive Summary

Phase 14D successfully wired real PRZZ polynomials into the Euler-Maclaurin integral path. All infrastructure is now in place, but the gate tests remain XFAIL because the Euler-Maclaurin formulas are simplified approximations that don't yet produce B ≈ 5.

**Key Finding:** The "+5" in m₁ = exp(R) + 5 is a **combinatorial factor** (2K-1 for K=3 pieces), not a result of polynomial integrals. The current formulas give B ≈ -0.27 with real polynomials.

---

## Tasks Completed

### D1: Polynomial Loader ✓

**New File:** `src/ratios/przz_polynomials.py`

Provides a clean, single-source API for loading PRZZ polynomials:

```python
@dataclass(frozen=True)
class PrzzK3Polynomials:
    P1: P1Polynomial
    P2: PellPolynomial
    P3: PellPolynomial
    Q: QPolynomial
    benchmark: str  # "kappa" or "kappa_star"
    R: float        # 1.3036 or 1.1167

def load_przz_k3_polynomials(benchmark: str) -> PrzzK3Polynomials
```

**Tests:** 28 tests in `tests/test_przz_polynomials_loader.py`

### D2: Wire Polynomials into Euler-Maclaurin ✓

**Modified:** `src/ratios/j1_euler_maclaurin.py`

- Added `_extract_poly_funcs()` helper
- Updated `compute_S12_from_J1_integrals()` to accept `polys` parameter
- Updated `decompose_m1_using_integrals()` to accept `polys` parameter
- Added `using_real_polynomials` flag to output

### D3: Update Gate Tests ✓

**Modified:** `tests/test_plus5_gate.py`

- Updated gate tests to use `decompose_m1_using_integrals()` with real polynomials
- Added new `TestPhase14DPolynomialWiring` test class
- Gate tests remain XFAIL (formulas need correction)

### D4: Microcase Diagnostic Script ✓

**New File:** `src/ratios/microcase_plus5_signature_k3.py`

Minimal diagnostic script that:
1. Loads PRZZ polynomials for κ and κ*
2. Computes (A, B) decomposition from J1 integrals
3. Prints per-piece contributions
4. Shows polynomial integrals and key parameters

**Tests:** 15 tests in `tests/test_microcase_plus5_signature_k3.py`

### D5: Sign Convention Lock Tests ✓

**New File:** `tests/test_przz_sign_conventions.py`

Locks Phase 14C sign corrections:
- ζ'/ζ(1+s) pole coefficient is -1
- J13/J14 main-terms are NEGATIVE
- A^{(1,1)}(0) is POSITIVE (~1.3856)

**Tests:** 13 tests

---

## Test Results

```
Phase 14D Tests: 64 passed, 5 xfailed
Phase 14C/14D Combined: 136 passed, 5 xfailed
```

### XFAIL Tests (Expected)

1. `test_constant_offset_is_5()` - B ≈ -0.27, not ≈ 5
2. `test_exp_coefficient_is_positive()` - A ≈ 0.15, not ≈ 1
3. `test_kappa_B_is_approximately_5()` - Same as #1
4. `test_kappa_A_is_approximately_1()` - A ≈ 0.15, not ≈ 1
5. `test_kappa_star_B_is_approximately_5()` - B ≈ -0.50, not ≈ 5

---

## Current Output (With Real Polynomials)

```
KAPPA (R=1.3036):
  A (exp coefficient): 0.151123
  B (constant offset): -0.272078
  Target B: 5 (= 2K-1 for K=3)
  Gap from target: -5.272078

KAPPA* (R=1.1167):
  A (exp coefficient): 0.183954
  B (constant offset): -0.502923
  Target B: 5 (= 2K-1 for K=3)
  Gap from target: -5.502923
```

### Per-Piece Breakdown (κ benchmark)

| Piece | exp coefficient | constant |
|-------|-----------------|----------|
| j11   | +0.000          | +0.470   |
| j12   | +0.104          | -0.707   |
| j13   | +0.024          | -0.342   |
| j14   | +0.024          | -0.342   |
| j15   | +0.000          | +0.650   |
| **Total** | **+0.151**  | **-0.272** |

---

## Gap Analysis

The "+5" constant is NOT coming from J15 in the expected way:

```
Expected: J15 = A^{(1,1)}(0) × [large factor] ≈ 5
Actual:   J15 = A^{(1,1)}(0) × ∫P₁P₂ du = 1.3837 × 0.4697 = 0.65
```

**Root Cause:** The Euler-Maclaurin formulas in `j1_euler_maclaurin.py` are simplified approximations. They don't capture the full PRZZ main-term structure that produces the "+5" combinatorial factor.

---

## Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `src/ratios/przz_polynomials.py` | CREATE | Polynomial loader for ratios pipeline |
| `src/ratios/j1_euler_maclaurin.py` | MODIFY | Accept polys parameter |
| `src/ratios/microcase_plus5_signature_k3.py` | CREATE | Diagnostic script |
| `tests/test_przz_polynomials_loader.py` | CREATE | 28 tests |
| `tests/test_plus5_gate.py` | MODIFY | Use real polynomials, add 5 new tests |
| `tests/test_microcase_plus5_signature_k3.py` | CREATE | 15 tests |
| `tests/test_przz_sign_conventions.py` | CREATE | 13 tests |

---

## What's Still Needed

1. **Fix Euler-Maclaurin Formulas:** The formulas in `j1_euler_maclaurin.py` need to match the paper's main-term structure. The "+5" is a combinatorial factor from the 5 pieces (2K-1 for K=3), not from polynomial integrals.

2. **Investigate Paper Formulas:** PRZZ TeX lines 1502-1511 describe the m₁ structure. The "+5" should emerge from the correct formula, not be calibrated.

3. **Un-XFAIL Gate Tests:** Once formulas are corrected, the 5 XFAIL tests should pass.

---

## What's Working

✅ Real PRZZ polynomials are correctly loaded and wired
✅ Polynomial constraints are enforced (P1(0)=0, P1(1)=1, etc.)
✅ Sign conventions are locked (J13/J14 negative, A^{(1,1)} positive)
✅ Microcase diagnostic shows per-piece contributions
✅ κ and κ* benchmarks give different (expected) results

---

## Next Steps (Phase 14E?)

1. **Study PRZZ TeX 1502-1511:** Understand the exact formula for m₁
2. **Identify missing factor:** What produces the "+5" combinatorially?
3. **Update J1 formulas:** Match the paper's main-term structure
4. **Remove XFAIL markers:** When B ≈ 5 and A ≈ 1

---

## Conclusion

Phase 14D successfully completed the infrastructure work:
- Real PRZZ polynomials are now wired into the Euler-Maclaurin path
- Comprehensive tests verify the wiring
- Sign conventions are locked
- Diagnostic tools show exactly where the gap is

The remaining work is mathematical: the Euler-Maclaurin formulas need to be corrected to produce B ≈ 5.

**136 tests passing, 5 xfailed (expected).**
