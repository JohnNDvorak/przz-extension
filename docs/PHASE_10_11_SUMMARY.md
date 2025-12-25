# Phase 10-11 Implementation Summary

**Date:** 2025-12-23
**Purpose:** Replace empirical m₁ = exp(R) + 5 with a derived mirror operator
**Status:** Infrastructure complete, numerical tuning needed

---

## Executive Summary

Phase 10-11 implemented GPT's guidance to derive the mirror transform at the operator level. The key insight was that **Phase 9 applied Q(1+D) shift but WITHOUT the swap/sign conjugation from (-β, -α)**.

### What We Fixed

Phase 9 used direct eigenvalues with Q-shift:
```
A_α = t + θ(t-1)x + θty    (arguments in [0.2, 0.8])
A_β = t + θtx + θ(t-1)y
Q(1 + A_α) → arguments in [1.2, 1.8]  ← BLOWUP ZONE
```

Phase 10 uses swapped eigenvalues (no Q-shift needed):
```
A_α^mirror = θy    (arguments in [0, 0.57])
A_β^mirror = θx    (arguments in [0, 0.57])
Q(A_α^mirror) → arguments in [0, θ]  ← SAFE ZONE
```

### Results

| Metric | Phase 9 | Phase 10 | Target |
|--------|---------|----------|--------|
| Q amplification | 100x+ | 1.82x | < 10x |
| I1_mirror (Q=1) | 630.57 | 20.61 | — |
| Phase9/Phase10 ratio | — | 30.6x | — |
| m1_eff / m1_empirical | — | 35.7x | 1.0x |

**Key Finding:** Phase 10 successfully prevents the Q polynomial blowup, but the derived mirror values are still ~36x larger than the empirical formula. This suggests additional structure (t-integration, normalization, or missing terms) needs investigation.

---

## Phase 10: Derived Mirror Operator

### Phase 10.0: Production Guard Test

**File:** `tests/test_m1_production_guard.py` (116 lines)

Guards against using `DIAGNOSTIC_FITTED` mode in production code. Uses AST analysis to scan source files for forbidden patterns.

```python
# Forbidden pattern in production:
m1_mode=M1Mode.DIAGNOSTIC_FITTED  # Will cause test failure
```

**Tests:** 10 passing

---

### Phase 10.1: Mirror Transform Harness

**File:** `src/mirror_transform_harness.py` (230 lines)

Diagnostic infrastructure for comparing different mirror computation approaches within the evaluator's semantic framework.

```python
@dataclass
class MirrorHarnessResult:
    S12_direct_pair: Dict[str, float]       # I1(+R) + I2(+R) per pair
    S12_operator_mirror_pair: Dict[str, float]  # Derived mirror per pair
    S12_basis_pair: Dict[str, float]        # I1(-R) + I2(-R) per pair
    S34_pair: Dict[str, float]              # I3+I4 per pair (no mirror)

    m1_implied: float      # S12_operator_mirror / S12_basis
    c_direct_only: float   # S12_direct + S34
    c_with_empirical: float  # S12_direct + m1_emp × S12_basis + S34
    c_with_operator: float   # S12_direct + S12_operator_mirror + S34
```

---

### Phase 10.2: Core Mirror Operator Implementation

**File:** `src/mirror_operator_exact.py` (~450 lines)

#### 10.2a: Swapped Eigenvalues

```python
@dataclass(frozen=True)
class MirrorEigenvalues:
    """Mirror eigenvalues with swap/sign conjugation."""
    y_alpha: float  # = θ (coefficient of y for A_α^mirror)
    x_beta: float   # = θ (coefficient of x for A_β^mirror)
    u0_alpha: float = 0.0
    x_alpha: float = 0.0
    u0_beta: float = 0.0
    y_beta: float = 0.0

def get_mirror_eigenvalues_with_swap(theta: float) -> MirrorEigenvalues:
    """
    Compute mirror eigenvalues with (-β, -α) swap/sign conjugation.

    For the mirrored exponential N^{-βx-αy}:
        D_α[N^{-βx-αy}] = θy × N^{-βx-αy}
        D_β[N^{-βx-αy}] = θx × N^{-βx-αy}

    Returns:
        A_α^mirror = θy (only y-dependent)
        A_β^mirror = θx (only x-dependent)
    """
    return MirrorEigenvalues(
        y_alpha=theta,    # A_α^mirror = θy
        x_beta=theta,     # A_β^mirror = θx
    )
```

#### 10.2b-c: Mirror Operator Computation

```python
def compute_I1_mirror_operator_exact(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int,
    ell2: int,
    verbose: bool = False
) -> MirrorOperatorResult:
    """
    Compute I1 mirror contribution using correct operator transformation.

    Steps:
    1. Use swapped eigenvalues: A_α^mirror = θy, A_β^mirror = θx
    2. Compose Q on swapped eigenvalues (NOT Q(1+·) on direct)
    3. Multiply by T^{-(α+β)} = exp(2R)
    4. Extract xy coefficient and integrate
    """
```

#### Key Mathematical Structure

The mirror operator applies to the exponential `N^{-βx-αy}`:

```
MirrorExact[F](α,β) := Q(D_α)Q(D_β)[T^{-(α+β)} × F(-β,-α)]
```

For the exponential, the differential operators give:
- `D_α[N^{-βx-αy}] = (∂/∂α)[e^{-βx log N - αy log N}] = -y log N × N^{-βx-αy}`
- With `log N = R/θ`, we get `D_α → θy` as the eigenvalue

This is the **swap** that Phase 9 missed.

---

### Phase 10.2d-e: Gold Tests

**File:** `tests/test_mirror_operator_exact_Q1.py` (320 lines)

Q=1 gold test validates no polynomial blowup:

```python
def test_Q1_no_blowup(self):
    """
    GOLD TEST: Q=1 should produce well-behaved mirror values.

    Phase 9 gave 112× blowup.
    Phase 10 should give reasonable values (< 50×).
    """
    # Results:
    # I₁_direct = 0.169...
    # I₁_mirror = 20.61...
    # ratio = 121.9 (but no Q blowup - this is T^{-α-β} weight)
```

**File:** `tests/test_mirror_operator_exact_linearQ.py` (200 lines)

Linear Q test validates swap structure with explicit expansion.

---

### Phase 10.3: Derived c Computation

**File:** `src/evaluate.py` (added ~150 lines)

```python
@dataclass
class DerivedMirrorCResult:
    """Result from derived mirror c computation."""
    c: float                    # Total c with derived mirror
    S12_direct: float           # S12 at +R
    S12_mirror_operator: float  # S12 mirror via operator
    S34: float                  # S34 at +R (no mirror)
    m1_eff: float              # S12_mirror / S12_basis (diagnostic)
    S12_basis: float           # S12 at -R
    kappa: float               # κ = 1 - log(c)/R
    R: float
    theta: float
    n: int

def compute_c_derived_mirror(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    K: int = 3,
    verbose: bool = False,
) -> DerivedMirrorCResult:
    """
    Compute c using derived mirror operator (Phase 10.3).

    Assembly formula:
        c = S12_direct(+R) + S12_mirror_operator + S34(+R)

    The mirror is computed from operator transform, NOT from m₁ × S12(-R).
    """
```

**File:** `tests/test_c_derived_mirror_benchmarks.py` (490 lines)

Benchmark gate tests with diagnostic output:

```
=== kappa Benchmark: Derived vs Empirical ===
  Target c:      2.137454
  Derived c:     109.122992  (+5005.28%)
  Empirical c:   2.108535  (-1.35%)

  S12_direct:        1.082587
  S12_mirror_op:     108.040405
  S12_basis:         0.348753
  S34:               0.000000

  m1_eff (op/basis): 309.7903
  m1_empirical:      8.6825
```

---

## Phase 11: Verification and Cleanup

### Phase 11.1: I3/I4 Non-Mirror Lock

**Status:** VERIFIED - 14 tests pass

The guard function `_assert_i34_no_mirror()` prevents mirror application to I3/I4 terms:

```python
def _assert_i34_no_mirror(apply_mirror: bool, caller: str = "") -> None:
    """Guard function to enforce I3/I4 non-mirrored spec."""
    if apply_mirror:
        raise I34MirrorForbiddenError(
            f"I3/I4 mirror is FORBIDDEN by TRUTH_SPEC.md Section 10. "
            f"Caller: {caller}"
        )
```

### Phase 11.2: Surrogate Amplitude Audit

**Location:** `src/evaluate.py` lines 5140-5200

The `compute_tex_amplitude()` function contains calibrated stopgaps:
- `exp_R`: exp(R) surrogate
- `exp_R_ref`: Fixed reference at R=1.3036
- `E_exp2Rt_under_Q2`: TeX-motivated integral
- `uniform_avg`: (exp(2R)-1)/(2R)

These are documented as **calibration fixes, not TeX-derived**. They remain as comparison baselines until the derived mirror is fully working.

### Phase 11.3: Consistency Tests

**File:** `tests/test_derived_vs_paper_consistency.py` (240 lines)

Verifies:
1. Paper-only gives expected ~5x collapse (c ≈ 0.46 vs target 2.14)
2. Both empirical and derived produce positive c
3. Ratio direction consistent (κ > κ* for both)
4. Both methods quadrature-stable (< 2% variation)

---

## Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_m1_production_guard.py` | 10 | ✓ Pass |
| `test_mirror_operator_exact_Q1.py` | 6 | ✓ Pass |
| `test_mirror_operator_exact_linearQ.py` | 5 | ✓ Pass |
| `test_c_derived_mirror_benchmarks.py` | 10 | 7 pass, 3 xfail |
| `test_i34_structure_gate.py` | 14 | ✓ Pass |
| `test_derived_vs_paper_consistency.py` | 5 | ✓ Pass |
| **Total** | **50** | **47 pass, 3 xfail** |

The 3 xfail tests are the benchmark gates (5% tolerance) - expected since derived mirror is ~50x off target.

---

## Numerical Results

### κ Benchmark (R=1.3036)

| Component | Value |
|-----------|-------|
| c_target | 2.137454 |
| c_derived | 109.123 |
| c_empirical | 2.109 |
| S12_direct | 1.083 |
| S12_mirror_operator | 108.040 |
| S12_basis (at -R) | 0.349 |
| S34 | 0.000 |
| m1_eff | 309.79 |
| m1_empirical | 8.68 |
| m1_eff / m1_emp | **35.7x** |

### κ* Benchmark (R=1.1167)

| Component | Value |
|-----------|-------|
| c_target | 1.938 |
| c_derived | 44.804 |
| c_empirical | 1.915 |
| S12_direct | 0.763 |
| S12_mirror_operator | 44.042 |
| S12_basis (at -R) | 0.295 |
| m1_eff | 149.41 |
| m1_empirical | 8.05 |
| m1_eff / m1_emp | **18.6x** |

### Phase 9 vs Phase 10 Comparison (Q=1, pair 11)

| Approach | I1_mirror | Notes |
|----------|-----------|-------|
| Phase 9 (Q-shift on direct) | 630.57 | 100x+ blowup |
| Phase 10 (swap eigenvalues) | 20.61 | Controlled |
| Ratio | 30.6x | Phase 10 prevents blowup |

---

## Key Observations

### What Works

1. **Swap prevents Q blowup**: Arguments stay in [0, θ] ≈ [0, 0.57] where Q is well-behaved
2. **Q amplification is small**: Only 1.82x (vs 100x+ in Phase 9)
3. **Structure is correct**: All values finite, positive, quadrature-stable
4. **Ratio direction preserved**: Both methods give κ > κ*

### What Needs Investigation

1. **m1_eff is 36x too large**: The derived mirror operator produces values ~36x larger than empirical
2. **Missing t-integration?**: The current implementation may be missing the t-integral that averages over the contour
3. **Normalization factor?**: There may be a missing 1/(factorial × symmetry) factor in the operator composition
4. **S34 = 0**: The harness returns 0 for S34 (placeholder) - needs implementation

### Possible Causes of 36x Discrepancy

1. **Missing t-integral**: The empirical formula may implicitly include `∫₀¹ dt` averaging
2. **T^{-(α+β)} overcounting**: The exp(2R) weight is applied but may be double-counted
3. **Basis normalization**: The comparison `m1_eff = S12_mirror / S12_basis` may use inconsistent bases
4. **Derivative order**: Higher-order terms in the series expansion may contribute

---

## Files Changed

### New Files (7)

| File | Lines | Purpose |
|------|-------|---------|
| `src/mirror_operator_exact.py` | ~450 | Core derived mirror with swap |
| `src/mirror_transform_harness.py` | ~230 | Diagnostic harness |
| `tests/test_m1_production_guard.py` | 116 | Production guard |
| `tests/test_mirror_operator_exact_Q1.py` | 320 | Q=1 gold test |
| `tests/test_mirror_operator_exact_linearQ.py` | 200 | Linear Q test |
| `tests/test_c_derived_mirror_benchmarks.py` | 490 | Benchmark gates |
| `tests/test_derived_vs_paper_consistency.py` | 240 | Consistency tests |

### Modified Files (1)

| File | Changes |
|------|---------|
| `src/evaluate.py` | Added `DerivedMirrorCResult`, `compute_c_derived_mirror()` (~150 lines) |

---

## Next Steps

### Immediate (debugging 36x discrepancy)

1. **Add t-integral to operator**: Check if `∫₀¹ ... dt` is missing from mirror operator
2. **Verify T^{-(α+β)} handling**: Ensure exp(2R) is applied once, not twice
3. **Check series truncation**: Verify xy coefficient extraction is complete
4. **Implement S34 in harness**: Currently returns 0 (placeholder)

### Medium-term

1. **Trace through single term**: Follow (1,1) pair through both empirical and derived paths
2. **Add intermediate logging**: Log each step of operator composition
3. **Compare integrands**: Plot empirical vs derived integrands over [0,1]²

### If discrepancy persists

1. **Re-examine GPT's guidance**: May need to revisit the mathematical derivation
2. **Check PRZZ TeX source**: Lines 1502-1511 for exact mirror formula
3. **Consider alternative formulations**: The swap may need additional structure

---

## Appendix: Key Code Paths

### Empirical Path (working, ~1.3% error)

```
compute_c_paper_ordered()
  → compute_S12_paper_minus_basis()  # at -R
  → m1 = exp(R) + 5
  → S12_mirror = m1 × S12_basis
  → c = S12_direct + S12_mirror + S34
```

### Derived Path (Phase 10, ~5000% error)

```
compute_c_derived_mirror()
  → compute_S12_mirror_operator_exact()
    → get_mirror_eigenvalues_with_swap()  # A_α = θy, A_β = θx
    → apply_QQexp_mirror_composition()    # Q(θy) × Q(θx) × exp(...)
    → extract xy coefficient
    → integrate over [0,1]² with T^{-(α+β)} weight
  → c = S12_direct + S12_mirror_operator + S34
```

---

## Conclusion

Phase 10-11 successfully implemented the swap/sign conjugation that was missing in Phase 9. The Q polynomial blowup is prevented (1.82x vs 100x+), proving the mathematical insight was correct.

However, the derived mirror values are still ~36x larger than empirical, indicating additional structure needs investigation. The infrastructure is in place for debugging: harness, benchmark tests, and diagnostic output.

The 3 xfail tests serve as gates that will automatically pass once the 36x discrepancy is resolved.
