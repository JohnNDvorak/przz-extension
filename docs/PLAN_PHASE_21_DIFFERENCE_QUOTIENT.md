# PHASE 21 IMPLEMENTATION PLAN: PRZZ Difference Quotient Identity for D = 0

**Created:** 2025-12-24
**Goal:** Achieve D = 0 analytically by implementing the PRZZ difference quotient identity

---

## Executive Summary

Based on research of the codebase, I have identified the key mathematical structure and implementation path to achieve D = 0 analytically by implementing the PRZZ difference quotient identity (Lines 1502-1511).

### Current State

The production pipeline uses an **empirical shim** `m = exp(R) + 5` for mirror term assembly:
```python
c = I₁I₂(+R) + m × I₁I₂(-R) + I₃I₄(+R)
```

**Current metrics (from Phase 20.3):**
- A (exp(R) coefficient) is ~10% below target (A_ratio ≈ 0.89)
- D = I₁₂(+R) + I₃₄(+R) ≠ 0 (should be 0 for exact B/A = 5)
- c accuracy is ~1.3% (acceptable but not first-principles)

### The Key Insight

The PRZZ difference quotient identity (Lines 1502-1511) transforms:
```latex
(N^{αx+βy} - T^{-α-β}N^{-βx-αy})/(α+β)
= N^{αx+βy} log(N^{x+y}T) ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

This identity:
1. **Removes the 1/(α+β) singularity analytically** via t-integral representation
2. **Pre-combines direct and mirror terms** BEFORE operator application
3. **Requires operator shift** Q(D) → Q(1+D) for mirror terms

---

## Mathematical Foundation

### 1. The Difference Quotient Structure

From `docs/TEX_MIRROR_OPERATOR_SHIFT.md` and `docs/TRUTH_SPEC.md`:

**Bracket Definition (Lines 1499-1501):**
```
B(α,β;x,y) = [N^{αx+βy} - T^{-(α+β)}N^{-βx-αy}] / (α+β)
```

**Difference Quotient Identity (Lines 1502-1511):**
```
[A - B]/s = A × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t×s} dt
```

### 2. The Operator Shift Identity

From `docs/TEX_MIRROR_OPERATOR_SHIFT.md`:

**Theorem:** For any polynomial Q and the T-weight factor T^{-s}:
```
Q(D_α)(T^{-s}F) = T^{-s} × Q(1 + D_α)F
```

**Consequence for mirror terms:**
- Direct term: Uses Q(A_α) where A_α = t + θ(t-1)x + θty
- Mirror term: Uses Q(1 + A_α^{mirror}) where A_α^{mirror} = θy (swapped/flipped)

### 3. Why Current Implementation Has D ≠ 0

**Current approach** (from `src/evaluate.py:compute_c_paper_with_mirror`):
```python
c = I₁I₂(+R) + m × I₁I₂(-R) + I₃I₄(+R)
```

This applies the mirror as a **post-hoc scalar multiplier** AFTER evaluating integrals separately.

**Correct PRZZ approach:**
The difference quotient identity combines direct and mirror **WITHIN the integral**, producing automatic cancellation that gives D = 0.

---

## Implementation Architecture

### Overview of Modules Needed

```
src/
├── difference_quotient.py       # NEW: Core difference quotient evaluator
├── q_shift_exact.py             # NEW: Exact Q(1+z) polynomial computation
├── unified_bracket_evaluator.py # NEW: Unified I₁/I₂ with built-in mirror
└── evaluate.py                  # MODIFY: Add new mirror_mode
```

### Component 1: Difference Quotient Core

**New File:** `src/difference_quotient.py`

**Purpose:** Implement the t-integral representation that combines direct and mirror terms as a single unit.

**Key Structure:**
```python
class DifferenceQuotientBracket:
    """
    PRZZ difference quotient bracket (TeX Lines 1502-1511).

    Computes:
        [N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
        = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt

    This is evaluated at α = β = -R/L (PRZZ evaluation point).
    """

    def evaluate_as_series(
        self,
        t: float,
        theta: float,
        R: float,
        var_names: Tuple[str, ...]
    ) -> TruncatedSeries:
        """
        Evaluate the bracket as a TruncatedSeries for series algebra.

        The t-integral becomes a single integration variable that enters
        the Q operator arguments.
        """
        pass
```

### Component 2: Exact Q(1+z) Polynomial

**New File:** `src/q_shift_exact.py` (or extend `src/q_operator.py`)

**Purpose:** Compute the **binomial-shifted polynomial** Q(1+z) exactly.

**Enhancement Needed:**
- Add validation that Q(1+z) coefficients are computed correctly
- Add gate tests comparing Q(1+0) = Q(1), Q'(1+0) = Q'(1)

### Component 3: Unified Bracket Evaluator

**New File:** `src/unified_bracket_evaluator.py`

**Purpose:** Evaluate I₁ and I₂ with the difference quotient structure built-in, not post-hoc.

**Key Insight:** The unified approach means:
1. The t-integral from the difference quotient IS the same t in Q(A_α), Q(A_β)
2. Both direct and mirror contributions emerge from a single integral
3. The operator shift Q → Q(1+·) is applied automatically for mirror eigenvalues

### Component 4: Integration with evaluate.py

**Modify:** `src/evaluate.py`

**Add new mirror_mode:**
```python
mirror_mode: str = "difference_quotient"  # NEW option
```

---

## Implementation Steps

### Step 1: Mathematical Derivation Document (1-2 hours)

Create `docs/PHASE_21_DIFFERENCE_QUOTIENT_DERIVATION.md`:

1. Derive the exact form of the t-integrand after α=β=-R/L substitution
2. Derive the mirror eigenvalues A_α^{mirror}, A_β^{mirror}
3. Show that the operator shift produces Q(1+·) exactly
4. Prove that D = 0 follows from the unified structure

### Step 2: Implement DifferenceQuotientBracket (4-6 hours)

**File:** `src/difference_quotient.py`

### Step 3: Enhance Q Shift Computation (2-3 hours)

**File:** `src/q_operator.py` (enhance existing)

### Step 4: Implement Unified Bracket Evaluator (6-8 hours)

**File:** `src/unified_bracket_evaluator.py`

### Step 5: Decomposition Gate Tests (3-4 hours)

**File:** `tests/test_difference_quotient_gate.py`

### Step 6: Integration with Production Pipeline (2-3 hours)

**Modify:** `src/evaluate.py`

### Step 7: Validation and Documentation (2-3 hours)

---

## Expected Outcomes

### Success Criteria

1. **D = 0 to numerical precision** (< 1e-10 for all pairs)
2. **B/A = 5 exactly** (< 1e-8 relative error)
3. **c accuracy ≤ 0.5%** (improved from current ~1.3%)
4. **Both benchmarks pass** (κ and κ*)

### Verification Metrics

| Metric | Current (empirical shim) | Expected (difference quotient) |
|--------|--------------------------|--------------------------------|
| D | +0.20 | ~0 |
| B/A | 5.85 | 5.00 |
| A_ratio | 0.89 | 1.00 |
| c gap | -1.35% | < 0.5% |

---

## Test Strategy

### Unit Tests

| Test | Purpose |
|------|---------|
| `test_bracket_scalar_limit` | Verify (exp(2R)-1)/(2R) at x=y=0 |
| `test_bracket_series_convergence` | Series coefficients converge with n_quad |
| `test_q_shift_basic` | Q(1+0) = Q(1) |
| `test_q_shift_derivative` | Q'(1+0) = Q'(1) |

### Integration Tests

| Test | Purpose |
|------|---------|
| `test_unified_I1_nonzero` | Unified I1 computes finite value |
| `test_unified_I1_matches_structure` | I1 has expected order of magnitude |

### Gate Tests (Critical)

| Test | Purpose |
|------|---------|
| `test_D_is_zero_all_pairs` | D < 1e-10 for all 6 pairs |
| `test_B_over_A_is_five` | B/A = 5.00 ± 1e-8 |
| `test_c_accuracy_improved` | c gap < 0.5% for both benchmarks |

---

## Files Summary

### New Files to Create

1. `src/difference_quotient.py` - Core difference quotient evaluator
2. `src/unified_bracket_evaluator.py` - Unified I₁/I₂ evaluation
3. `tests/test_difference_quotient_gate.py` - Gate tests
4. `docs/PHASE_21_DIFFERENCE_QUOTIENT_DERIVATION.md` - Mathematical derivation

### Files to Modify

1. `src/evaluate.py` - Add `mirror_mode="difference_quotient"`
2. `src/q_operator.py` - Enhance Q shift validation
3. `docs/TRUTH_SPEC.md` - Add Section 12 on difference quotient
4. `docs/DECISIONS.md` - Add Decision 7

### Files to Reference (Read-Only)

1. `src/combined_identity_unified_t.py` - Existing t-parameterization
2. `src/operator_post_identity.py` - Eigenvalue derivation
3. `docs/TEX_MIRROR_OPERATOR_SHIFT.md` - Operator shift theorem

---

## Risks and Mitigations

### Risk 1: Numerical Instability in t-Integral

The t-integral representation may require high quadrature precision.

**Mitigation:** Start with n_quad_t = 40, increase if needed.

### Risk 2: Complexity of Mirror Eigenvalue Derivation

The mirror eigenvalues A_α^{mirror} = θy (swapped) need careful derivation.

**Mitigation:** Derive step-by-step in documentation before coding.

### Risk 3: Run 20 Precedent Shows Structural Mismatch

Run 20 attempted a similar combined structure but got 2-4x larger I1 values.

**Mitigation:** Key insight from Run 20: "The outer exp(-Rθ(x+y)) factor was missing."

---

*Generated 2025-12-24 as part of Phase 21 planning.*
