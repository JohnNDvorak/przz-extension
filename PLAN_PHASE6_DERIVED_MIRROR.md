# Phase 6: Replace Empirical m1 with Derived Mirror-Operator Transformation

**Date:** 2025-12-22
**Status:** PLANNING
**Predecessor:** Phase 5 (COMPLETE - u-regularized path matches post-identity)

---

## Executive Summary

Phase 5 proved that the u-regularized and post-identity paths compute **the same TeX I₁ object** to machine precision. This success unlocks a path to replace the empirical mirror weight `m1 = exp(R) + 5` with a **derived mirror-operator transformation**.

The key insight (from GPT 2025-12-22):
> Under D_α, D_α(T^{-s}F) = T^{-s}(1 + D_α)F
> Hence: Q(D_α)(T^{-s}F) = T^{-s} Q(1 + D_α)F

This means the mirror term is **not** "same integrand with R→-R times m₁" but rather "mirror integrand with Q shifted by 1."

**Goal:** Eliminate m₁ calibration by implementing the mirror contribution exactly via operator shift Q→Q(1+·), and verify via decomposition gate.

---

## Current State Assessment

### What We Have (Phase 5 Validated)

| Component | Status | Location |
|-----------|--------|----------|
| Post-identity I₁ computation | **WORKING** | `src/operator_post_identity.py` |
| U-regularized I₁ computation | **WORKING** | `src/combined_identity_regularized.py` |
| Machine-precision agreement | **VERIFIED** | 36 gate tests in `test_regularized_matches_post_identity.py` |
| L-invariance | **VERIFIED** | Zero L-dependence confirmed |
| TruncatedSeries extraction | **WORKING** | `src/composition.py` |

### What Is Still Empirical

| Component | Current Value | Location | Status |
|-----------|---------------|----------|--------|
| m1 formula | `exp(R) + (2K-1)` | `src/m1_policy.py` | **CALIBRATED, NOT DERIVED** |
| Mirror assembly | `c = I₁I₂(+R) + m1 × I₁I₂(-R) + I₃I₄(+R)` | `src/evaluate.py` | Scalar approximation |
| tex_mirror amplitude | Surrogate model | `src/evaluate.py` | **ASPIRATIONAL** |

---

## Mathematical Foundation

### The Bracket Structure

The PRZZ bracket has direct and mirror terms:
```
B(α,β;x,y) = [N^{αx+βy} - T^{-(α+β)}N^{-βx-αy}] / (α+β)
           = B_direct - B_mirror / (α+β)
```

where:
- `B_direct = N^{αx+βy}` (direct term)
- `B_mirror = T^{-(α+β)}N^{-βx-αy}` (mirror term, has T^{-s} weight)

### The Operator Shift Identity

When applying Q(D_α)Q(D_β) to the mirror term:

**Theorem (Operator Shift):**
For any polynomial Q and function F:
```
Q(D_α)(T^{-s}F) = T^{-s} × Q(1 + D_α)F
```

where s = α + β.

**Proof sketch:**
- D_α = -1/L × ∂/∂α
- D_α(T^{-s}) = D_α(exp(-sL)) = -L × exp(-sL) × (-1/L) = exp(-sL) = T^{-s}
- So D_α(T^{-s}F) = T^{-s}(1 + D_α)F
- By induction: D_α^n(T^{-s}F) = T^{-s}(1 + D_α)^n F
- Therefore: Q(D_α)(T^{-s}F) = T^{-s}Q(1 + D_α)F

### Application to Mirror Term

The mirror contribution to I₁ is:
```
I₁_mirror = -1/(α+β) × Q(D_α)Q(D_β) [T^{-(α+β)}N^{-βx-αy}]
          = -1/(α+β) × T^{-(α+β)} × Q(1+D_α)Q(1+D_β) [N^{-βx-αy}]
```

The **shifted polynomial** Q_shifted(z) = Q(1+z) has different coefficients that can be computed exactly.

### Why Scalar m1 Is Insufficient

The scalar m1 approach treats mirror as:
```
I₁_mirror ≈ m1 × I₁(-R)
```

But this ignores that Q itself must be **shifted**. The shift changes the polynomial coefficients, not just the amplitude. This is why no finite-L sweep could derive m1 from first principles.

---

## Implementation Plan

### 6.1 Create Derivation Note: `docs/TEX_MIRROR_OPERATOR_SHIFT.md`

**Purpose:** Document the mathematical proof and TeX line references.

**Contents:**
1. Operator definitions: D_α = -1/L × ∂/∂α
2. Proof of shift identity: D_α(T^{-s}F) = T^{-s}(1 + D_α)F
3. Application to PRZZ bracket
4. Derivation of eigenvalues for shifted operators
5. TeX line references (PRZZ 1502-1511)

**File location:** `docs/TEX_MIRROR_OPERATOR_SHIFT.md`

### 6.2 Implement Shifted Polynomial Computation

**New function:** `compute_shifted_polynomial(Q, shift=1.0)`

```python
def compute_shifted_polynomial(Q_poly, shift: float = 1.0):
    """
    Compute Q_shifted(z) = Q(shift + z).

    If Q(z) = Σ q_k z^k (monomial form)
    then Q(shift + z) = Σ q_k (shift + z)^k
                      = Σ q_k Σ_j C(k,j) shift^{k-j} z^j

    Returns: New polynomial object with shifted coefficients
    """
```

**File location:** `src/q_operator.py` (new or extend existing)

**Tests:**
- `Q_shifted(0) = Q(1)` for shift=1
- `Q_shifted'(0) = Q'(1)` for shift=1
- Verify against direct evaluation at sample points

### 6.3 Implement Mirror Eigenvalue Functions

The mirror term uses **flipped** coordinates (-βx - αy instead of αx + βy) and the shifted Q.

**New functions in `src/operator_post_identity.py` or new file:**

```python
def get_A_alpha_mirror_affine_coeffs(t: float, theta: float) -> Tuple[float, float, float]:
    """
    Affine coefficients for A_α on mirror eigenvalue N^{-βx-αy}.

    D_α(exp(-θL(βx+αy))) = θy × exp(...)
    So A_α_mirror = θy (constant in x,y... needs careful derivation)

    NOTE: This needs careful derivation from the TeX.
    """

def get_A_beta_mirror_affine_coeffs(t: float, theta: float) -> Tuple[float, float, float]:
    """Affine coefficients for A_β on mirror eigenvalue."""
```

### 6.4 Implement `compute_I1_mirror_exact()`

**New function:**

```python
def compute_I1_mirror_exact(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
) -> float:
    """
    Compute the EXACT mirror contribution to I₁.

    Uses operator shift: Q(D_α)(T^{-s}F) = T^{-s}Q(1+D_α)F

    This replaces the scalar approximation m1 × I(-R).

    Returns:
        The exact mirror I₁ contribution
    """
    # 1. Compute shifted polynomial Q_shifted(z) = Q(1+z)
    Q_shifted = compute_shifted_polynomial(polynomials['Q'], shift=1.0)

    # 2. Compute T^{-(α+β)} weight at evaluation point
    #    At α=β=-R/L: T^{2R/L} = exp(2R/θ) approximately... needs derivation

    # 3. Apply Q_shifted(A_α_mirror)Q_shifted(A_β_mirror) via TruncatedSeries

    # 4. Extract xy coefficient and integrate
```

**File location:** `src/mirror_exact.py` (new file)

### 6.5 Implement `compute_I1_direct()`

The direct term is the N^{αx+βy} contribution **without** the mirror:

```python
def compute_I1_direct(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    ell1: int = 1,
    ell2: int = 1,
) -> float:
    """
    Compute the DIRECT contribution to I₁ (no mirror).

    This is the N^{αx+βy} term only, before combining with mirror.
    """
```

**File location:** `src/mirror_exact.py`

### 6.6 Create Decomposition Gate Tests

**New test file:** `tests/test_mirror_decomposition_gate.py`

**Test structure:**
```python
class TestDecompositionGate:
    """
    GATE: I₁_combined = I₁_direct + I₁_mirror_exact

    This gate verifies that the derived mirror operator shift
    correctly decomposes the combined I₁.
    """

    @pytest.mark.parametrize("ell1,ell2", [
        (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)
    ])
    def test_decomposition_kappa(self, polys_kappa, ell1, ell2):
        """I₁_combined = I₁_direct + I₁_mirror_exact for kappa benchmark."""
        # LHS: compute via existing validated post-identity
        I1_combined = compute_I1_operator_post_identity_pair(...)

        # RHS: compute via decomposition
        I1_direct = compute_I1_direct(...)
        I1_mirror = compute_I1_mirror_exact(...)

        assert np.isclose(I1_combined, I1_direct + I1_mirror, rtol=1e-10)
```

**Acceptance criteria:**
- All 6 pairs pass for κ benchmark (R=1.3036)
- All 6 pairs pass for κ* benchmark (R=1.1167)
- Tolerance: < 1e-8 relative error (or tighter if achievable)

### 6.7 Update Evaluator Default Path

Once the decomposition gate passes:

1. Add new `mirror_mode="operator_q_shift_exact"` option
2. Update `compute_c_paper_with_mirror()` to use exact mirror when mode selected
3. Keep `mirror_mode="empirical_scalar"` as fallback
4. Refuse to run empirical mode unless explicitly requested (similar to K>3 opt-in)

**File:** `src/evaluate.py`

---

## Testing Strategy

### Unit Tests

| Test | Location | Purpose |
|------|----------|---------|
| `test_shifted_polynomial_basic` | `tests/test_q_operator.py` | Q(1+z) evaluated correctly |
| `test_shifted_polynomial_derivatives` | `tests/test_q_operator.py` | Q'(1+z) matches |
| `test_mirror_eigenvalue_coeffs` | `tests/test_mirror_exact.py` | Affine coefficients correct |

### Integration Tests

| Test | Location | Purpose |
|------|----------|---------|
| `test_I1_direct_nonzero` | `tests/test_mirror_exact.py` | Direct term computes |
| `test_I1_mirror_nonzero` | `tests/test_mirror_exact.py` | Mirror term computes |
| `test_direct_plus_mirror_finite` | `tests/test_mirror_exact.py` | Sum is finite |

### Gate Tests

| Test | Location | Purpose |
|------|----------|---------|
| `test_decomposition_kappa` | `tests/test_mirror_decomposition_gate.py` | Decomposition identity holds (κ) |
| `test_decomposition_kappa_star` | `tests/test_mirror_decomposition_gate.py` | Decomposition identity holds (κ*) |

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `docs/TEX_MIRROR_OPERATOR_SHIFT.md` | Mathematical derivation document |
| `src/mirror_exact.py` | Exact mirror computation via operator shift |
| `tests/test_mirror_exact.py` | Unit tests for mirror computation |
| `tests/test_mirror_decomposition_gate.py` | Decomposition gate tests |

### Modified Files

| File | Changes |
|------|---------|
| `src/q_operator.py` | Add `compute_shifted_polynomial()` |
| `src/evaluate.py` | Add `mirror_mode="operator_q_shift_exact"` option |
| `src/m1_policy.py` | Add note that derived path exists |
| `docs/DECISIONS.md` | Document Decision 6: Mirror derived via operator shift |
| `docs/TRACEABILITY.md` | Add mirror operator traceability |

---

## Risks and Mitigations

### Risk 1: Mirror Eigenvalue Derivation Complexity

The mirror term has flipped coordinates (-βx - αy) which changes the eigenvalue structure.

**Mitigation:** Carefully derive eigenvalues step-by-step in TEX_MIRROR_OPERATOR_SHIFT.md before coding.

### Risk 2: T^{-(α+β)} Weight at Evaluation Point

At α = β = -R/L, the weight T^{2R/L} needs careful handling.

**Mitigation:** Cross-check numerically against known m1 values. The exact weight should produce results consistent with empirical m1.

### Risk 3: Quadrature Tolerance

The decomposition gate requires very precise quadrature to distinguish direct from mirror.

**Mitigation:** Use high quadrature (n=40+) for gate tests. Accept 1e-8 tolerance if 1e-10 is not achievable.

---

## Success Criteria

1. **Decomposition gate passes:** I₁_combined = I₁_direct + I₁_mirror_exact for all pairs
2. **Both benchmarks:** κ (R=1.3036) and κ* (R=1.1167) pass
3. **Tolerance:** Relative error < 1e-8
4. **m1 consistency:** When using exact mirror, implied m1 should match exp(R)+5 (or explain discrepancy)
5. **Documentation complete:** TEX_MIRROR_OPERATOR_SHIFT.md proves the identity

---

## What This Does NOT Do

- **Does not touch K=4:** Phase 6 is K=3 only
- **Does not derive amplitude/I₂ structure:** That is Phase 7
- **Does not remove m1_policy.py:** Kept as fallback mode
- **Does not change I₃/I₄:** They have no mirror (spec-locked)

---

## Implementation Order

1. **Week A: Derivation**
   - Write TEX_MIRROR_OPERATOR_SHIFT.md
   - Derive mirror eigenvalue formulas
   - Verify against TeX lines

2. **Week B: Implementation**
   - Implement `compute_shifted_polynomial()`
   - Implement `compute_I1_direct()` and `compute_I1_mirror_exact()`
   - Unit tests for each

3. **Week C: Gate Testing**
   - Create decomposition gate tests
   - Debug until gate passes
   - Document findings

4. **Week D: Integration**
   - Update evaluator default path
   - Update DECISIONS.md and TRACEABILITY.md
   - Final documentation

---

## Open Questions (To Resolve During Implementation)

1. **Exactly how does T^{-(α+β)} interact with the t-integral?**
   - The combined identity rewrites the bracket as an integral
   - The T^{-s} factor may need to be handled inside the integral

2. **Is the shift exactly 1, or is there an L-dependence?**
   - The derivation shows shift = 1, but verify numerically

3. **What is the sign convention for the mirror term?**
   - Bracket has (direct - mirror) / (α+β)
   - Ensure sign is correct in decomposition

---

## References

- Phase 5 documentation: `docs/PHASE5_REGULARIZED_FIX_COMPLETE.md`
- Current m1 policy: `src/m1_policy.py`
- Post-identity operator: `src/operator_post_identity.py`
- Regularized path: `src/combined_identity_regularized.py`
- TRUTH_SPEC.md Section 10: Mirror structure
- PRZZ TeX lines 1502-1511: Combined identity
