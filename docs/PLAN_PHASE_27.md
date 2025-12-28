# Phase 27: Derived Mirror Transform Using Bivariate Engine

**Date:** 2025-12-26
**Status:** PLANNING
**Prerequisite:** Phase 26B complete (unified evaluator matches OLD DSL exactly)

---

## Executive Summary

Phase 26B resolved the I₁/I₂ semantic mismatch. The unified evaluator now matches OLD DSL to ~1e-13 relative error for all 6 pairs. The remaining ~1-2% gap on c is **not from I₁/I₂ evaluation**—it's from the **mirror assembly**.

**Phase 27 Goal:** Replace the empirical mirror multiplier `m = exp(R) + 5` with a **derived mirror transform** computed directly from PRZZ TeX structure.

---

## Background: Why the Mirror Term Matters

### Current State (Post-Phase 26B)

| Benchmark | c target | c computed | c gap |
|-----------|----------|------------|-------|
| κ (R=1.3036) | 2.1375 | 2.1085 | **-1.35%** |
| κ* (R=1.1167) | 1.9384 | 1.9146 | **-1.23%** |

The ratio test passes (gap < 0.2%), confirming no major structural bugs remain.

### The Mirror Assembly Problem

Current `compute_c_paper_with_mirror()` uses:
```python
c = S12(+R) + m × S12(-R) + S34(+R)
```
where `m = exp(R) + 5` is **empirical**.

But PRZZ TeX (lines 1502-1511) defines the mirror term as:
```
I(α,β) + T^{-(α+β)} × I(-β,-α)
```

This is NOT the same as `I(+R) + scalar × I(-R)`. The (-β,-α) substitution involves:
1. **Swapped eigenvalues** for Q operators
2. **Different exponential structure** (N^{-βx-αy} vs N^{αx+βy})
3. **Chain rule effects** on derivative operators

---

## Task Breakdown

### Task 27.1: Make Unified General the Canonical Backend

**Goal:** Establish `unified_i1_general.py` and `unified_i2_general.py` as the production backend, with OLD DSL as a validation gate.

**Files to create:**
- `src/evaluator/s12_backend.py` - Backend abstraction layer

**Implementation:**
```python
def compute_I1_backend(
    R, theta, ell1, ell2, polynomials,
    backend: str = "unified_general",  # or "dsl"
    **kwargs
) -> float:
    """Dispatch I₁ computation to chosen backend."""
    if backend == "unified_general":
        from src.unified_i1_general import compute_I1_unified_general
        result = compute_I1_unified_general(R, theta, ell1, ell2, polynomials, **kwargs)
        return result.I1_value
    elif backend == "dsl":
        # Existing DSL path
        ...
```

**Tests:**
- `tests/test_s12_backend_equivalence.py`
  - Assert unified_general == OLD DSL for all 6 pairs
  - Both benchmarks (κ and κ*)
  - Tolerance: < 1e-12 relative error

**Acceptance criteria:**
- [ ] Backend abstraction created
- [ ] All 6 pairs match between backends for both benchmarks
- [ ] Production evaluator defaults to unified_general

---

### Task 27.2: Implement Mirror Transform at Operator Level (CORE TASK)

**Goal:** Implement the PRZZ mirror transform directly using the bivariate series engine, without using `R → -R` as a proxy.

**Files to create:**
- `src/mirror_transform_exact.py` - Core mirror transform implementation

**Mathematical Structure:**

The TeX mirror term (PRZZ lines 1502-1511) is:
```
I(α,β) + T^{-(α+β)} × I(-β,-α)
```

For I₁, the "mirrored" integrand I(-β,-α) requires:

1. **Swapped exponential:** N^{-βx-αy} instead of N^{αx+βy}
   - At α=β=-R/L: N^{Rx/L+Ry/L} vs N^{-Rx/L-Ry/L}
   - After θL normalization: exp(-θR(x+y)) vs exp(+θR(x+y)) ← **sign flip!**

2. **Swapped Q eigenvalues:**
   - Direct: A_α = t + θ(t-1)x + θty, A_β = t + θtx + θ(t-1)y
   - Mirror: A_α^M = t + θty + θ(t-1)x → swap x↔y roles

   Actually, the (-β,-α) substitution means:
   - D_α acts on the -β variable slot
   - D_β acts on the -α variable slot
   - This swaps which eigenvalue couples to which variable

3. **T weight factor:** T^{-(α+β)} = exp(2R) at α=β=-R/L

**Implementation approach:**

Use the Phase 26B `BivariateSeries` engine with a "mirror mode":

```python
def build_I1_mirror_integrand(
    R, theta, t, u, ell1, ell2, P_coeffs, Q_coeffs,
    max_dx, max_dy
) -> BivariateSeries:
    """Build the MIRROR integrand for I₁.

    Key differences from direct:
    1. Exp factor: exp(2Rt - θR(x+y)) instead of exp(2Rt + θR(2t-1)(x+y))
       Wait, need to verify this from TeX...

    Actually, the full structure is:
    - Direct: N^{αx+βy} = exp(θL(αx+βy)) → at α=β=-R/L: exp(-θR(x+y))
    - T factor: T^{-t(α+β)} = exp(2Rt)
    - Combined: exp(2Rt - θR(x+y))... wait, that's not matching Phase 26B

    Let me re-derive from Phase 26B structure:
    Phase 26B direct exp is: exp(2Rt + Rθ(2t-1)(x+y))

    For mirror, we need N^{-βx-αy} instead of N^{αx+βy}:
    - Direct N factor: N^{αx+βy} at α=β=-R/L → exp(-θR(x+y))
    - Mirror N factor: N^{-βx-αy} at α=β=-R/L → exp(θR(x+y)) ← sign flip

    But we also need the T^{-t(α+β)} factor which is same for both.

    So the exp structure needs careful derivation...
    """
```

**Key insight from existing `mirror_operator_exact.py`:**

Phase 10-13 attempted various mirror eigenvalue approaches:
- Phase 10: Static swapped eigenvalues (A_α^M = θy, A_β^M = θx)
- Phase 12: t-dependent complement eigenvalues
- Phase 13: t-flip consistent exp

But none of these were integrated with the Phase 26B bivariate engine.

**Phase 27 approach:**

Build a **transform layer** that operates on the bivariate integrand builder:

```python
def compute_I1_mirror_exact(
    R, theta, ell1, ell2, polynomials,
    n_quad_u=60, n_quad_t=60
) -> float:
    """Compute I₁ mirror term using derived operator transform.

    NOT using R → -R proxy. Instead:
    1. Build exp factor with mirror structure
    2. Build Q factors with swapped eigenvalue coupling
    3. Build P factors (same as direct)
    4. Extract x^ℓ₁ y^ℓ₂ coefficient
    5. Multiply by T weight = exp(2R)
    """
    ...
```

**Tests (before assembly):**
- `tests/test_mirror_transform_microcases.py`
  1. P=Q=1 oracle checks in mirror mode
  2. Symmetry checks under swapping ell1/ell2
  3. Sanity: finite, stable, no sign flips

**Acceptance criteria:**
- [ ] Mirror integrand builder implemented using BivariateSeries
- [ ] P=Q=1 microcase oracle derived and matches implementation
- [ ] All symmetry properties validated

---

### Task 27.3: Compute Diagnostic m_eff (NOT a knob)

**Goal:** Once mirror is computed directly, measure what "effective m" it implies.

**Files to create:**
- `scripts/run_phase27_meff_report.py`

**Diagnostic computation:**
```python
# Compute components
S12_direct_plusR = compute_S12_direct(theta, +R, polynomials)
S12_mirror_exact = compute_S12_mirror_exact(theta, R, polynomials)  # derived
S12_basis_minusR = compute_S12_direct(theta, -R, polynomials)  # current proxy basis

# Effective multiplier
m_eff = S12_mirror_exact / S12_basis_minusR  # if S12_basis_minusR != 0

# Compare to empirical
m_empirical = math.exp(R) + 5
print(f"m_eff = {m_eff:.6f}")
print(f"m_empirical = {m_empirical:.6f}")
print(f"ratio = {m_eff / m_empirical:.6f}")
```

**Gate tests:**
- [ ] m_eff is finite and stable under quadrature refinement
- [ ] No catastrophic sign flips
- [ ] R→0 limit behavior is stable

**Acceptance criteria:**
- [ ] Diagnostic script runs for both benchmarks
- [ ] Results logged and compared to empirical m

---

### Task 27.4: Re-run c with Derived Mirror Term

**Goal:** Compute c using the derived mirror transform and measure accuracy gap.

**Files to create:**
- `src/evaluator/c_assembly_tex_mirror.py`

**Implementation:**
```python
def compute_c_tex_mirror_exact(
    theta, R, n, polynomials, K=3
) -> EvaluationResult:
    """Compute c using derived mirror transform.

    Assembly:
        c = S12_direct(+R) + S12_mirror_exact(R) + S34(+R)

    Where S12_mirror_exact is the DERIVED mirror term,
    NOT m × S12_basis(-R).
    """
    # Direct S12 at +R
    s12_direct = compute_S12_direct(theta, R, polynomials, n)

    # Derived mirror S12 (includes T^{-(α+β)} = exp(2R) weight)
    s12_mirror = compute_S12_mirror_exact(theta, R, polynomials, n)

    # S34 at +R (no mirror required per TRUTH_SPEC Section 10)
    s34 = compute_S34(theta, R, polynomials, n)

    c = s12_direct + s12_mirror + s34
    kappa = 1 - math.log(c) / R

    return EvaluationResult(c=c, kappa=kappa, ...)
```

**Tests:**
- `tests/test_phase27_c_accuracy_derived_mirror.py`
  - κ benchmark (R=1.3036)
  - κ* benchmark (R=1.1167)
  - Measure gap from targets

**Acceptance criteria:**
- [ ] Derived mirror c computation implemented
- [ ] Gap measured and documented
- [ ] No tuning knobs used

---

### Task 27.5: Quadrature Convergence Sweep

**Goal:** Confirm the remaining gap is structural (mirror/inputs), not numerical.

**Files to create:**
- `tests/test_phase27_quadrature_convergence.py` (mark slow)

**Sweep:**
```python
@pytest.mark.slow
def test_quadrature_convergence():
    for n in [40, 60, 80, 120]:
        result = compute_c_tex_mirror_exact(theta, R, n, polynomials)
        # Log result
        ...

    # Assert convergence (values stabilize)
    assert abs(results[80] - results[120]) / abs(results[120]) < 0.001
```

**Acceptance criteria:**
- [ ] Convergence confirmed for both benchmarks
- [ ] Gap does not move materially with n increase

---

## Implementation Order

1. **Task 27.1** - Backend abstraction (foundation)
2. **Task 27.2** - Mirror transform (core derivation)
3. **Task 27.5** - Quadrature sweep (cheap sanity check)
4. **Task 27.3** - m_eff diagnostic (insight into gap)
5. **Task 27.4** - Full c with derived mirror (final integration)

---

## Key Mathematical Derivations Needed

### Derivation 1: Mirror Exponential Structure

From PRZZ TeX 1502-1511:
```
(N^{αx+βy} - T^{-α-β}N^{-βx-αy})/(α+β)
```

At α=β=-R/L with N=T^θ:
- Direct N factor: N^{αx+βy} = T^{θ(αx+βy)} = exp(θL(αx+βy)) = exp(-θR(x+y))
- Mirror N factor: N^{-βx-αy} = exp(-θL(βx+αy)) = exp(θR(x+y)) ← **sign flip!**

But Phase 26B uses exp(2Rt + Rθ(2t-1)(x+y))... need to reconcile.

The (2t-1) factor comes from the T^{-t(α+β)} integration structure.
For mirror, this needs re-derivation.

### Derivation 2: Mirror Q Eigenvalues

Direct eigenvalues (from operator_post_identity.py):
```
A_α(t) = t + θ(t-1)x + θty
A_β(t) = t + θtx + θ(t-1)y
```

Under (-β,-α) → (α,β) substitution in operator action:
- D_α acting on N^{-βx-αy} gives eigenvalue depending on -α position
- This swaps the x↔y coupling structure

Need to derive: A_α^mirror(t), A_β^mirror(t)

### Derivation 3: P=Q=1 Mirror Oracle

With P=Q=1, derive closed-form for mirror coefficient of x^ℓ₁y^ℓ₂.
This validates the implementation.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/evaluator/s12_backend.py` | Backend abstraction |
| `src/mirror_transform_derived.py` | Core mirror transform using BivariateSeries |
| `src/evaluator/c_assembly_tex_mirror.py` | Full c assembly with derived mirror |
| `scripts/run_phase27_meff_report.py` | m_eff diagnostic |
| `tests/test_s12_backend_equivalence.py` | Backend validation |
| `tests/test_mirror_transform_microcases.py` | Mirror P=Q=1 oracle |
| `tests/test_phase27_c_accuracy_derived_mirror.py` | Final accuracy gate |
| `tests/test_phase27_quadrature_convergence.py` | Convergence sweep |

---

## Success Criteria

Phase 27 is successful if:

1. **Derived mirror transform implemented** without using `R → -R` proxy
2. **P=Q=1 microcase oracle validates** mirror implementation
3. **m_eff diagnostic reveals** relationship to empirical m = exp(R) + 5
4. **Gap on c is measured** using derived approach (may or may not close gap)
5. **No tuning knobs used** - everything derived from TeX

Even if the gap doesn't close, we will have:
- Identified exactly what the derived mirror gives
- Determined whether gap is from mirror derivation vs polynomial precision
- Clear next steps based on diagnostic results

---

## Risk Assessment

**High confidence:**
- Task 27.1 (backend abstraction) - straightforward refactor
- Task 27.5 (quadrature sweep) - pure numerical check

**Medium confidence:**
- Task 27.2 (mirror transform) - mathematical derivation required
- Task 27.3 (m_eff diagnostic) - depends on 27.2 working

**Low confidence:**
- Task 27.4 (full c accuracy) - gap may not close even with correct derivation

If the gap doesn't close with correct derivation, next steps are:
1. Polynomial coefficient precision audit
2. Check for missing normalization factors at assembly level
3. Compare against PRZZ numerical intermediate values (if available)

---

## Questions for User Before Implementation

1. Should I prioritize getting the P=Q=1 mirror oracle working first as a validation anchor?
2. Is there any additional PRZZ TeX context that would help derive the exact mirror eigenvalue structure?
3. Should the backend abstraction (Task 27.1) include backward compatibility with existing tests?
