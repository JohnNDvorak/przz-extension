# ASSUMPTIONS.md — Empirical Assumptions and Falsifiability

**Purpose:** List every empirical choice, what would falsify it, and what test catches it.
This is how K=5 won't become archaeology.

---

## Assumption 1: m1 = exp(R) + 5 for K=3

**Type:** EMPIRICAL CALIBRATION
**Confidence:** HIGH (validated at two independent benchmarks)

### Statement

For K=3 mollifier pieces, the mirror weight is:
```
m1 = exp(R) + 5
```

### Evidence

| Benchmark | R | c target | c computed | Gap |
|-----------|------|----------|------------|-----|
| κ | 1.3036 | 2.137 | 2.109 | -1.35% |
| κ* | 1.1167 | 1.938 | 1.915 | -1.20% |

Both benchmarks achieved within ~1.5% using m1 = exp(R) + 5.

### Falsification Criteria

- If m1 = exp(R) + 5 fails to reproduce c_target within 5% at BOTH benchmarks
- If a different m1 formula reproduces targets MORE accurately

### Test Coverage

- `test_m1_policy_gate.py::TestBenchmarkValidation::test_kappa_benchmark`
- `test_m1_policy_gate.py::TestBenchmarkValidation::test_kappa_star_benchmark`
- `test_m1_sensitivity.py::TestM1SensitivityRange::test_m1_sensitivity_range`

### What Happens if Falsified

1. Check for transcription errors in polynomial coefficients
2. Check for missing normalization factors in c formula
3. Explore alternative m1 formulas using calibration harness

---

## Assumption 2: m1 = exp(R) + (2K-1) for General K

**Type:** EXTRAPOLATION (UNVALIDATED)
**Confidence:** LOW (no K>3 reference data)

### Statement

The K=3 formula generalizes to:
```
m1 = exp(R) + (2K - 1)
```

This gives:
- K=3: m1 = exp(R) + 5 ✓ (validated)
- K=4: m1 = exp(R) + 7 (extrapolated)
- K=5: m1 = exp(R) + 9 (extrapolated)

### Evidence

Pattern observed: 2K-1 = {5, 7, 9, ...} for K = {3, 4, 5, ...}

No direct evidence for K>3 — this is CONJECTURED.

### Falsification Criteria

- If K=4 reference target appears and exp(R) + 7 fails to reproduce it
- If theoretical derivation shows different K-dependence

### Test Coverage

- `test_m1_policy_gate.py::TestKDepEmpirical::test_k4_works_with_optin`
- `test_m1_sensitivity.py::TestM1Monotonicity::test_m1_formula_monotone_in_K`
- `test_m1_sensitivity.py::TestM1Linearity::test_m1_linear_in_K`
- `test_m1_sensitivity.py::TestM1ExtrapolationRisk::test_k3_to_k4_delta_is_2`

### Safety Gate

K>3 extrapolation requires explicit opt-in:
```python
policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL, allow_extrapolation=True)
```

Without opt-in, `M1ExtrapolationError` is raised.

### What Happens if Falsified

1. Use calibration harness to solve for correct K=4 m1
2. Check if a different pattern fits (e.g., K², K, constants)
3. Possibly need K-specific empirical calibration

---

## Assumption 3: I3/I4 Have NO Mirror Structure

**Type:** SPEC LOCK (from TeX)
**Confidence:** VERY HIGH (explicit in TRUTH_SPEC.md)

### Statement

In the combined PRZZ identity:
- I₁ and I₂ combine with mirror: `I(α,β) + T^{-α-β}·I(-β,-α)`
- I₃ and I₄ do NOT have mirror terms

### Evidence

TRUTH_SPEC.md Section 10 (lines 370-388):
```
- I₁(α,β) + T^{-α-β}I₁(-β,-α)  ← HAS MIRROR
- I₂(α,β) + T^{-α-β}I₂(-β,-α)  ← HAS MIRROR
- I₃(α,β) and I₄(α,β)          ← NO MIRROR
```

Empirical confirmation: Adding mirror to I3/I4 causes 350% overshoot.

### Falsification Criteria

- If PRZZ TeX reinterpretation shows I3/I4 DO have mirror
- If a different mirror structure for I3/I4 produces better results

### Test Coverage

- `test_i34_structure_gate.py::TestAssertI34NoMirror::test_raises_when_mirror_true`
- `test_i34_structure_gate.py::TestComputeS34TexCombined11SpecLock::test_raises_when_mirror_true`
- `test_i34_structure_gate.py::TestComputeS34Base11SpecLock::test_raises_when_mirror_true`

### Safety Gate

`I34MirrorForbiddenError` raised if `apply_mirror=True` is passed to I3/I4 functions.

### What Happens if Falsified

1. Remove the guard in `_assert_i34_no_mirror()`
2. Update TRUTH_SPEC.md with corrected interpretation
3. Implement correct mirror structure for I3/I4

---

## Assumption 4: Paper Regime Kernels Are Correct

**Type:** STRUCTURAL (from TeX)
**Confidence:** HIGH (ratio test confirms)

### Statement

The "paper regime" kernels (Case A/B/C from PRZZ) with `kernel_regime="paper"` correctly implement the PRZZ formula structure.

### Evidence

Ratio test at two benchmarks:
```
Target ratio: c(κ)/c(κ*) = 2.137/1.938 = 1.103
Paper ratio:  2.109/1.915 = 1.101 (within 0.2%)
Raw ratio:    1.960/0.937 = 2.09 (wrong by 90%)
```

Paper regime preserves benchmark ratio; raw mode does not.

### Falsification Criteria

- If paper regime + mirror assembly fails to reproduce c_target within 5%
- If a different kernel regime produces better results

### Test Coverage

- Integration tests in `tests/test_evaluate.py` (paper regime mode)
- Ratio tests comparing κ and κ* benchmarks

### What Happens if Falsified

1. Recheck Case A/B/C kernel implementations against TeX
2. Look for missing attenuation factors
3. Consider polynomial-dependent kernels

---

## Assumption 5: Naive m1 = exp(2R) is Too Large

**Type:** COMPARISON REFERENCE
**Confidence:** HIGH (documented ratios)

### Statement

The naive formula from combined identity, m1 = exp(2R), overestimates by:
- κ (R=1.3036): naive/empirical = 1.56
- κ* (R=1.1167): naive/empirical = 1.16

### Evidence

```python
# At R=1.3036:
m1_naive = exp(2 * 1.3036) ≈ 13.56
m1_empirical = exp(1.3036) + 5 ≈ 8.68
ratio = 13.56 / 8.68 ≈ 1.56

# At R=1.1167:
m1_naive = exp(2 * 1.1167) ≈ 9.33
m1_empirical = exp(1.1167) + 5 ≈ 8.05
ratio = 9.33 / 8.05 ≈ 1.16
```

### Falsification Criteria

- If naive formula produces BETTER results than empirical at ANY benchmark

### Test Coverage

- `test_m1_policy_gate.py::TestPaperNaive::test_naive_is_larger_than_empirical`
- `test_m1_sensitivity.py::TestM1NaiveComparison::test_naive_is_larger_at_benchmark_R`
- `test_m1_sensitivity.py::TestM1NaiveComparison::test_naive_ratio_documented`

### What Happens if Falsified

This would indicate fundamental misunderstanding of mirror term structure.

---

## Assumption 6: The "+5" in m1 Equals (2K-1) for K=3

**Type:** PATTERN OBSERVATION
**Confidence:** MEDIUM (single data point)

### Statement

For K=3: m1 = exp(R) + **5**, where 5 = 2×3 - 1 = 2K - 1

### Evidence

Only one K value (K=3) is validated. The pattern 2K-1 is observed but not proven.

Alternative explanations:
- Could be (K+2) = 3+2 = 5
- Could be (K!) - 1 = 6 - 1 = 5
- Could be coincidence

### Falsification Criteria

- If K=4 reference shows m1 ≠ exp(R) + 7
- If theoretical derivation shows different structure

### Test Coverage

- `test_m1_sensitivity.py::TestM1Linearity::test_m1_linear_in_K` (tests pattern)
- `test_m1_sensitivity.py::TestM1ExtrapolationRisk::test_k3_to_k4_delta_is_2` (tests delta)

### What Happens if Falsified

1. Determine correct pattern from K=4 data
2. Update `K_DEP_EMPIRICAL` formula
3. Rerun all K>3 computations

---

## Summary Table

| # | Assumption | Type | Confidence | Gate |
|---|------------|------|------------|------|
| 1 | m1 = exp(R) + 5 for K=3 | EMPIRICAL | HIGH | test_m1_policy_gate |
| 2 | m1 = exp(R) + (2K-1) | EXTRAPOLATION | LOW | M1ExtrapolationError |
| 3 | I3/I4 NO mirror | SPEC LOCK | VERY HIGH | I34MirrorForbiddenError |
| 4 | Paper kernels correct | STRUCTURAL | HIGH | ratio tests |
| 5 | Naive exp(2R) too large | COMPARISON | HIGH | test_m1_sensitivity |
| 6 | "+5" = (2K-1) pattern | PATTERN | MEDIUM | test_m1_sensitivity |

---

## Adding New Assumptions

When adding a new empirical assumption:

1. **Document it here** with:
   - Statement
   - Evidence
   - Falsification criteria
   - Test coverage

2. **Add tests** that would catch falsification

3. **Add safety gates** if the assumption is risky (raises on violation)

4. **Update TRACEABILITY.md** with spec→code→test mapping
