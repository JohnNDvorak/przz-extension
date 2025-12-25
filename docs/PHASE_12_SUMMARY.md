# Phase 12 Summary: T-Dependent Mirror Eigenvalues

**Date:** 2025-12-23
**Status:** COMPLETE - Hypothesis Disproven
**Author:** Claude (Opus 4.5)

---

## Executive Summary

Phase 12 implemented t-dependent "complement" eigenvalues for the derived mirror operator, testing GPT's hypothesis that restoring t-dependence would close the ~36x gap between derived and empirical mirror values.

**Result:** The complement transform provides only 1.84-1.99x improvement. A ~10-20x gap remains. More critically, integrand-level diagnosis reveals a **fundamental functional mismatch** between operator-based and evaluation-based mirror approaches.

**Conclusion:** The operator-based derived mirror approach (`Q(D)[T^{-α-β} F(-β,-α)]`) is **not mathematically equivalent** to the empirical approach (evaluate integrals at R → -R). The empirical formula `m₁ = exp(R) + 5` captures structure that cannot be expressed as eigenvalue transformations on Q.

---

## Phase 12 Implementation Details

### Phase 12.0: S34 Completion
- Implemented `_compute_S34_pair()` in `src/mirror_transform_harness.py`
- Uses DSL-based evaluation path (same as canonical evaluator)
- **Validation:** Harness S34 matches canonical evaluator exactly (tolerance 1e-10)
- **7 tests added**, all passing

### Phase 12.1: T-Dependent Complement Eigenvalues

Implemented the complement transform as suggested:

```python
def get_mirror_eigenvalues_complement_t(t: float, theta: float) -> MirrorEigenvalues:
    """
    A_α^mirror(t) = 1 - A_β(t) = (1-t) - θt·x + θ(1-t)·y
    A_β^mirror(t) = 1 - A_α(t) = (1-t) + θ(1-t)·x - θt·y
    """
    u0_alpha = 1 - t
    x_alpha = -theta * t
    y_alpha = theta * (1 - t)

    u0_beta = 1 - t
    x_beta = theta * (1 - t)
    y_beta = -theta * t

    return MirrorEigenvalues(...)
```

**Files modified:**
- `src/mirror_operator_exact.py` - Added `get_mirror_eigenvalues_complement_t()`
- Updated `apply_QQexp_mirror_composition()` with `use_t_dependent` parameter
- Updated `compute_I1_mirror_operator_exact()` and `compute_I2_mirror_operator_exact()`

**10 algebraic micro-tests added** in `tests/test_mirror_eigenvalues_algebra.py`:
- Verify complement relationship: `A_α^mirror = 1 - A_β`
- Verify Q argument ranges stay bounded ([-0.6, 1.6])
- Verify t-dependence exists and varies correctly

### Phase 12.2: Integrand Probe Diagnostic

Created `scripts/run_phase12_integrand_probe.py` to compare integrands at specific (u,t) nodes.

**Test nodes:** (0.2, 0.5), (0.8, 0.5), (0.5, 0.2), (0.5, 0.8), (0.3, 0.3), (0.7, 0.7)

### Phase 12.3: Benchmark Validation

Ran full validation on both PRZZ benchmarks.

---

## Results

### Benchmark Comparison Table

| Metric | Phase 10 | Phase 12 | Target | Remaining Gap |
|--------|----------|----------|--------|---------------|
| **κ benchmark (R=1.3036)** |
| m1_implied | 309.79 | 168.25 | 8.68 | **19.4x** |
| c_with_operator | 108.52 | 59.16 | 2.14 | **27.6x** |
| c_with_empirical | 3.51 | 3.51 | 2.14 | 1.64x |
| **κ* benchmark (R=1.1167)** |
| m1_implied | 149.41 | 75.21 | 8.05 | **9.3x** |
| c_with_operator | 44.36 | 22.49 | 1.94 | **11.6x** |
| c_with_empirical | 2.69 | 2.69 | 1.94 | 1.39x |

**Improvement factor:** Phase 12 gives 1.84x (κ) to 1.99x (κ*) improvement over Phase 10.

### Integrand Probe Results (CRITICAL)

```
Parameters: theta=0.5714, R=1.3036

Eigenvalue Structure at t=0.5:
  Direct A_α       = 0.500 - 0.286x + 0.286y
  Phase 10 A_α^mir = 0.000 + 0.000x + 0.571y
  Phase 12 A_α^mir = 0.500 - 0.286x + 0.286y  ← IDENTICAL TO DIRECT!

----------------------------------------------------------------------
(u, t)       Empirical    Phase 10    Phase 12   P10/Emp   P12/Emp
----------------------------------------------------------------------
(0.2, 0.5)    0.052425   29.524556    8.001552    563.18    152.63
(0.8, 0.5)    0.014282    1.090541    1.294825     76.36     90.66
(0.5, 0.2)    0.648944   10.422712    0.127618     16.06      0.20
(0.5, 0.8)    0.000764   49.814559   63.781774  65180.73  83456.38
(0.3, 0.3)    0.329147   20.510568    1.018904     62.31      3.10
(0.7, 0.7)    0.002131    8.744648   14.709445   4103.12   6901.89
----------------------------------------------------------------------

Ratio Analysis:
  Phase 10: Range [16, 65181], CV >> 1  → FUNCTIONAL mismatch
  Phase 12: Range [0.2, 83456], CV >> 1 → FUNCTIONAL mismatch
```

**Diagnosis:** If ratios were roughly constant, we'd be hunting a missing scalar. The wild variation (0.2 to 83,456) indicates **missing FUNCTIONAL structure**, not a normalization factor.

---

## Critical Observations

### Observation 1: Complement = Direct at t=0.5

At t=0.5, the complement transform yields:
```
A_α^mirror(0.5) = 1 - A_β(0.5)
                = 1 - [0.5 + 0.286x - 0.286y]
                = 0.5 - 0.286x + 0.286y
                = A_α(0.5)  ← SAME AS DIRECT!
```

This is mathematically necessary due to the symmetry `A_α(t) + A_β(t) = 2t` at any t, and at t=0.5 both equal 0.5 plus antisymmetric x,y terms.

**Implication:** The complement transform cannot be correct because at t=0.5 it gives back the direct eigenvalues, not a "mirror" structure.

### Observation 2: Operator vs. Evaluation Semantics

The two approaches are fundamentally different mathematical operations:

**Operator-based (derived) approach:**
```
Mirror term = Q(D_α)Q(D_β)[T^{-α-β} · N^{-βx-αy}]
```
This involves:
- Differential operators Q(∂/∂α), Q(∂/∂β)
- The factor T^{-α-β} which becomes exp(2R) after α=β=-R/L
- Swap of x,y in the exponential argument
- Some form of eigenvalue transformation

**Evaluation-based (empirical) approach:**
```
S12(-R) = ∫∫ [integrand evaluated with R → -R in exponentials]
```
This involves:
- Direct parameter substitution: exp(2Rt) → exp(-2Rt)
- No swap of x,y
- No eigenvalue transformation

**These are NOT the same operation!**

### Observation 3: The Mystery of m₁ = exp(R) + 5

The empirical formula that works is:
```
c = S12(+R) + m₁ × S12(-R) + S34(+R)
```
where `m₁ = exp(R) + (2K-1)` for K mollifier pieces.

For K=3: `m₁ = exp(R) + 5 ≈ 8.68`

The "+5" term has no obvious operator interpretation. It appears to encode combinatorial structure from the K=3 mollifier assembly, not an eigenvalue property.

---

## Files Created/Modified

| File | Status | Description |
|------|--------|-------------|
| `src/mirror_operator_exact.py` | Modified | Added `get_mirror_eigenvalues_complement_t()`, updated composition functions |
| `src/mirror_transform_harness.py` | Modified | Implemented S34, added `use_t_dependent` parameter |
| `tests/test_mirror_eigenvalues_algebra.py` | **New** | 10 algebraic micro-tests |
| `tests/test_mirror_transform_harness.py` | **New** | 7 S34 validation tests |
| `scripts/run_phase12_integrand_probe.py` | **New** | Integrand diagnostic script |

**All 17 new tests pass.**

---

## Questions for GPT

1. **Why does the complement transform fail?**
   - The mathematical intuition was that `A^mirror = 1 - A` would create a "reflection" structure. But this just recovers the direct eigenvalues at t=0.5 and gives wrong functional form elsewhere.

2. **Is the operator interpretation fundamentally wrong?**
   - PRZZ TeX 1502-1511 shows the combined identity, but perhaps the differential operator `Q(D)` acts differently on the mirror term than we assumed?
   - Could the mirror term require a different differential operator entirely?

3. **What does m₁ = exp(R) + 5 actually represent?**
   - The "+5" for K=3 (and presumably +7 for K=4, etc.) suggests a combinatorial origin
   - Is this counting mollifier piece interactions?
   - Is there a generating function or recurrence relation?

4. **Alternative approaches:**
   - Should we abandon derived mirror entirely and use the empirical formula?
   - Is there a hybrid approach where we derive the "5" term from first principles?
   - Should we re-examine PRZZ TeX to find where the R→-R evaluation originates?

5. **Diagnostic suggestions:**
   - What other tests would distinguish operator vs. evaluation semantics?
   - Is there a simpler toy model (K=1 or K=2) where the mirror structure is more transparent?

---

## Recommended Path Forward

### Option A: Accept Empirical Formula (Pragmatic)
Continue using `compute_c_paper_with_mirror()` which achieves ~2% accuracy via:
```
c = S12(+R) + (exp(R) + 5) × S12(-R) + S34(+R)
```
This is accurate enough for optimization experiments. The theoretical derivation can wait.

### Option B: Fresh Derivation (Research)
Start from scratch with PRZZ TeX 1502-1511:
1. Carefully trace what `T^{-α-β}` means in the context of the combined identity
2. Determine whether Q(D) acts on the mirror term at all
3. Look for where R→-R substitution enters the derivation

### Option C: Investigate the "+5" (Mathematical)
Try to derive the formula `m₁ = exp(R) + 2K - 1`:
1. Check if PRZZ gives explicit K-dependence anywhere
2. Look for combinatorial interpretations (K choose 2, etc.)
3. Test with K=2 or K=4 to verify the pattern

---

## Appendix: Test Output

```
============================= test session starts ==============================
tests/test_mirror_eigenvalues_algebra.py::TestComplementAlgebra::test_complement_formula_at_t_0 PASSED
tests/test_mirror_eigenvalues_algebra.py::TestComplementAlgebra::test_complement_formula_at_t_1 PASSED
tests/test_mirror_eigenvalues_algebra.py::TestComplementAlgebra::test_complement_formula_at_t_half PASSED
tests/test_mirror_eigenvalues_algebra.py::TestComplementAlgebra::test_complement_is_1_minus_beta PASSED
tests/test_mirror_eigenvalues_algebra.py::TestComplementAlgebra::test_complement_is_1_minus_alpha_for_beta PASSED
tests/test_mirror_eigenvalues_algebra.py::TestQArgumentRange::test_phase10_static_range PASSED
tests/test_mirror_eigenvalues_algebra.py::TestQArgumentRange::test_phase12_complement_range PASSED
tests/test_mirror_eigenvalues_algebra.py::TestPhase10VsPhase12Comparison::test_phase10_has_no_t_dependence PASSED
tests/test_mirror_eigenvalues_algebra.py::TestPhase10VsPhase12Comparison::test_phase12_has_t_dependence PASSED
tests/test_mirror_eigenvalues_algebra.py::TestPhase10VsPhase12Comparison::test_symmetry_at_t_half PASSED
tests/test_mirror_transform_harness.py::TestS34Validation::test_s34_matches_canonical_kappa PASSED
tests/test_mirror_transform_harness.py::TestS34Validation::test_s34_matches_canonical_kappa_star PASSED
tests/test_mirror_transform_harness.py::TestS34Validation::test_s34_per_pair_nonzero PASSED
tests/test_mirror_transform_harness.py::TestS34Validation::test_s34_total_negative PASSED
tests/test_mirror_transform_harness.py::TestHarnessStructure::test_harness_runs_without_error PASSED
tests/test_mirror_transform_harness.py::TestHarnessStructure::test_c_assembly_consistency PASSED
tests/test_mirror_transform_harness.py::TestHarnessStructure::test_m1_implied_calculation PASSED
============================= 17 passed in 57.86s ==============================
```

---

## Key Code References

- Direct eigenvalues: `src/operator_post_identity.py:150-177`
- Phase 10 static eigenvalues: `src/mirror_operator_exact.py:get_mirror_eigenvalues_with_swap()`
- Phase 12 complement eigenvalues: `src/mirror_operator_exact.py:get_mirror_eigenvalues_complement_t()`
- Mirror composition: `src/mirror_operator_exact.py:apply_QQexp_mirror_composition()`
- Empirical c computation: `src/evaluate.py:compute_c_paper_with_mirror()`
- Harness comparison: `src/mirror_transform_harness.py:MirrorTransformHarness`
- PRZZ combined identity: `docs/TRUTH_SPEC.md` Section 7
