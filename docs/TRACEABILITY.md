# TRACEABILITY.md — TeX → Spec → Code → Test Mapping

**Purpose:** Map PRZZ TeX sections to spec paragraphs to code functions to test files.
This is how K=5 won't become archaeology.

---

## I-Term Structure (Mirror/Non-Mirror)

### TeX Source

**RMS_PRZZ.tex Section 6.2.1** (lines ~1499-1530):
- Combined identity structure
- Mirror terms for I₁/I₂
- No mirror for I₃/I₄

### Spec Reference

**TRUTH_SPEC.md Section 10** (lines 370-388):
```
The exact PRZZ formula producing published c:

From Section 6.2.1, after combining:
- I₁(α,β) + T^{-α-β}I₁(-β,-α) (mirror terms)  ← HAS MIRROR
- I₂(α,β) + T^{-α-β}I₂(-β,-α)                 ← HAS MIRROR
- I₃(α,β) and I₄(α,β)                          ← NO MIRROR
```

### Code Implementation

**File:** `src/evaluate.py`

| Function | Purpose | Mirror? |
|----------|---------|---------|
| `compute_I12_channels()` | I₁+I₂ combined | YES (via m1 weight) |
| `compute_S34_tex_combined_11()` | I₃+I₄ combined | NO (raises if True) |
| `compute_S34_base_11()` | I₃+I₄ base | NO (raises if True) |
| `_assert_i34_no_mirror()` | Guard function | — |
| `I34MirrorForbiddenError` | Exception class | — |

### Test Coverage

**File:** `tests/test_i34_structure_gate.py` (14 tests)

| Test Class | Purpose |
|------------|---------|
| `TestI34MirrorForbiddenError` | Exception is ValueError subclass |
| `TestAssertI34NoMirror` | Guard raises on mirror=True |
| `TestComputeS34TexCombined11SpecLock` | S34_tex raises on mirror=True |
| `TestComputeS34Base11SpecLock` | S34_base raises on mirror=True |
| `TestSpecLockDocumentation` | SPEC LOCK comment exists in source |

---

## Mirror Weight m1

### TeX Source

**RMS_PRZZ.tex Section 6.2.1** (lines ~1502-1511):
- Mirror term: `T^{-α-β}·I(-β,-α)`
- At α = β = -R/L: `T^{2R/L}` ≈ `exp(2R)` (naive)

### Spec Reference

**TRUTH_SPEC.md Section 10** + **K_SAFE_BASELINE_LOCKDOWN.md**:
- Naive: m1 = exp(2R)
- Empirical: m1 = exp(R) + 5 for K=3
- Generalized: m1 = exp(R) + (2K-1) (extrapolated)

### Code Implementation

**File:** `src/m1_policy.py`

| Component | Purpose |
|-----------|---------|
| `M1Mode` enum | K3_EMPIRICAL, K_DEP_EMPIRICAL, PAPER_NAIVE, OVERRIDE |
| `M1Policy` dataclass | mode, allow_extrapolation, override_value |
| `m1_formula(K, R, policy)` | Compute m1 with safety gates |
| `M1ExtrapolationError` | Raised on K>3 without opt-in |
| `M1_EMPIRICAL_KAPPA` | Reference value at R=1.3036 |
| `M1_EMPIRICAL_KAPPA_STAR` | Reference value at R=1.1167 |

**File:** `src/m1_calibration.py`

| Function | Purpose |
|----------|---------|
| `solve_m1_from_channels()` | Solve for m1 from c_target and channels |
| `validate_m1_formula_at_k3()` | Validate K=3 at both benchmarks |
| `estimate_m1_for_k4()` | Ready for K=4 validation |
| `CalibrationResult` | Dataclass with solved/empirical m1 and ratio |

### Test Coverage

**File:** `tests/test_m1_policy_gate.py` (27 tests)

| Test Class | Purpose |
|------------|---------|
| `TestM1ModeEnum` | All modes defined and distinct |
| `TestM1PolicyDataclass` | Defaults, frozen |
| `TestK3Empirical` | K=3 works, K≠3 raises |
| `TestKDepEmpirical` | K=3 works, K>3 raises without opt-in, works with opt-in |
| `TestPaperNaive` | exp(2R) works, K>3 raises without opt-in |
| `TestOverride` | override_value works |
| `TestBenchmarkValidation` | κ and κ* match expected values |
| `TestReferenceValues` | get_m1_reference_values() structure |
| `TestM1ExtrapolationError` | Exception is ValueError subclass |

**File:** `tests/test_m1_sensitivity.py` (11 tests)

| Test Class | Purpose |
|------------|---------|
| `TestM1Monotonicity` | m1 increases with R and K |
| `TestM1Linearity` | m1 is linear in K with delta=2 |
| `TestM1Sensitivity` | Small perturbations give finite results |
| `TestM1ExtrapolationRisk` | K=4 extrapolation characterization |
| `TestM1NaiveComparison` | naive > empirical at benchmark R |
| `TestM1DocumentedFormula` | exp(R)+5 and exp(R)+(2K-1) correct |

---

## Polynomial Structure

### TeX Source

**RMS_PRZZ.tex Section 7** (lines ~2550-2600):
- P₁, P₂, P₃ polynomial coefficients
- Q polynomial (normalization)
- Constraint: Q(0) = 1

### Spec Reference

**CLAUDE.md — Indexing of mollifier pieces**:
- Piece index ℓ ∈ {1,…,K}
- ℓ=1 corresponds to μ piece

### Code Implementation

**File:** `src/polynomials.py`

| Function | Purpose |
|----------|---------|
| `load_przz_polynomials()` | Load κ polynomials (R=1.3036) |
| `load_przz_polynomials_kappa_star()` | Load κ* polynomials (R=1.1167) |
| `PRZZPolynomial` class | Polynomial evaluation |

**File:** `data/przz_parameters.json`
**File:** `data/przz_parameters_kappa_star.json`

### Test Coverage

**File:** `tests/test_polynomials.py`

---

## Quadrature (Integration)

### TeX Source

**RMS_PRZZ.tex Section 6** (general):
- Integrals over [0,1]² domain
- Various u,v integrations

### Code Implementation

**File:** `src/quadrature.py`

| Function | Purpose |
|----------|---------|
| `gauss_legendre_1d()` | 1D Gauss-Legendre quadrature |
| `gauss_legendre_2d()` | 2D Gauss-Legendre quadrature |
| `integrate_2d()` | Wrapper for 2D integration |

### Test Coverage

**File:** `tests/test_quadrature.py`

---

## Complete Mapping Table

| TeX Section | Spec Location | Code File | Function/Class | Test File |
|-------------|---------------|-----------|----------------|-----------|
| 6.2.1 I-terms | TRUTH_SPEC §10 | evaluate.py | I12/S34 compute | test_i34_structure_gate.py |
| 6.2.1 mirror | TRUTH_SPEC §10 | m1_policy.py | m1_formula | test_m1_policy_gate.py |
| 6.2.1 mirror | K_SAFE docs | m1_calibration.py | calibration | test_m1_sensitivity.py |
| 7 polynomials | CLAUDE.md | polynomials.py | load_przz_* | test_polynomials.py |
| 6 integrals | CLAUDE.md | quadrature.py | integrate_* | test_quadrature.py |

---

## Traceability for Future K Values

### K=4 (Not Yet Implemented)

| Aspect | TeX Source | Spec Location | Code Location | Tests |
|--------|------------|---------------|---------------|-------|
| P₄ polynomial | TBD | TBD | polynomials.py | TBD |
| Pair enumeration | TBD | TBD | terms_k4_d1.py | TBD |
| m1 formula | — | m1_policy.py (extrapolated) | m1_formula() | test_m1_*.py |
| Mirror structure | TRUTH_SPEC §10 | Same | evaluate.py | Same |

### K=5 (Not Yet Implemented)

Follow same pattern as K=4.

---

## How to Use This Document

### When Adding New Features

1. Identify TeX source (section + line numbers)
2. Add to TRUTH_SPEC.md or update existing spec
3. Implement in code
4. Add tests that reference spec
5. Update this traceability matrix

### When Debugging Mismatches

1. Find the feature in this matrix
2. Check TeX source for authoritative formula
3. Check spec for interpretation
4. Check code matches spec
5. Check tests verify code

### When Reviewing Changes

1. Verify TeX source is cited
2. Verify spec is updated if needed
3. Verify tests cover new functionality
4. Verify traceability is maintained

---

## Document History

| Date | Change |
|------|--------|
| 2025-12-22 | Initial creation with I-term and m1 traceability |
