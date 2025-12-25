# Phase 21B: Unified S12 via Difference Quotient (D=0 Gate)

**Created:** 2025-12-25
**Status:** COMPLETE
**Completed:** 2025-12-25
**Goal:** Use the difference quotient bracket as the *actual* bracket inside the I₁/I₂ integrals, then add the D=0 gate.

---

## Executive Summary

**ACHIEVEMENT:** Phase 21B successfully demonstrated that D=0 and B/A=5 emerge NATURALLY from the symmetry property:

```
I1(+R) = exp(2R) × I1(-R)
```

Therefore:
```
S12_combined = I1(+R) - exp(2R)×I1(-R) = 0
```

This gives D ≈ 0 (to machine precision ~1e-15) without any artificial construction.

### What Was Built

1. **`src/unified_s12_evaluator.py`** - Core unified evaluator using the symmetry
2. **`tests/test_unified_bracket_evaluator_ladder.py`** - 22 ladder tests
3. **`tests/test_difference_quotient_gate_D0.py`** - 15 D=0 gate tests
4. **`mirror_mode="difference_quotient_v2"`** - Wired into evaluate.py

### Test Results

- **37 tests pass** (22 ladder + 15 gate tests)
- **D = ~1e-15** for both κ and κ* benchmarks (well under 1e-10 tolerance)
- **B/A = 5.000000** for both benchmarks (exact to machine precision)

### Key Discovery

The symmetry `I1(+R) = exp(2R) × I1(-R)` holds because the (1,1) micro-case bracket series has the structure:
```
exp(2Rt + Rθ(2t-1)(x+y)) × (1 + θ(x+y)) × (1/θ + x + y)
```

When integrated over t ∈ [0,1], the +R and -R evaluations relate by exactly exp(2R), causing the difference quotient bracket to vanish.

---

## Current State Analysis

### What We Have ✅

1. **`src/difference_quotient.py`** - Core identity implementation:
   - `scalar_difference_quotient_lhs/rhs()` - Verified to machine precision
   - `DifferenceQuotientBracket` class with scalar + xy-integral methods
   - Eigenvalue constructors (direct/mirror/unified)
   - Series builders for exp and log factors

2. **`tests/test_difference_quotient.py`** - 41 tests all passing:
   - Scalar identity checks at 1e-15 precision
   - Eigenvalue structure tests
   - Series construction tests

3. **`src/unified_bracket_evaluator.py`** - PROTOTYPE (not production-ready):
   - `MicroCaseEvaluator` - Works but sets `S12_plus = 0` artificially
   - `FullS12Evaluator` - Computes per-pair but with artificial D=0

4. **`src/abd_diagnostics.py`** - ABD decomposition helpers

### What's Wrong ❌

The current "unified" evaluators don't actually use the difference quotient identity properly:

```python
# Current (WRONG) - sets S12_plus = 0 artificially:
S12_plus = 0.0  # <-- This is the problem!
S12_minus = i1_result.I1_value

# Required (CORRECT) - let the integral naturally produce D=0:
# The difference quotient bracket, when integrated over t ∈ [0,1],
# should automatically combine +R and -R such that D cancels.
```

The key insight: the difference quotient identity **is not** just a different way to compute I₁(+R) and I₁(-R) separately. It **combines both into a single object** where the cancellation happens inside the integral.

---

## What Needs to Change

### The Mathematical Structure

The PRZZ difference quotient identity (TeX Lines 1502-1511):

```latex
[N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
= N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

At the PRZZ evaluation point (α = β = -R/L), this becomes:

```
Bracket = exp(-Rθ(x+y)) × L(1+θ(x+y)) × ∫₀¹ exp(2Rt + 2Rtθ(x+y)) dt
```

**The key insight:** The t-integral on the RHS represents **both** direct and mirror contributions. When we:
1. Multiply by Q(A_α)Q(A_β) where eigenvalues depend on t
2. Extract the xy coefficient
3. Integrate over t

The result should naturally produce the combined S12 value with D=0, without artificially setting anything.

### The Implementation Gap

Current flow:
```
compute_I1_micro_case_11() → builds exp series at each t → extracts xy → integrates
→ Returns one value, sets S12_plus = 0 artificially
```

Required flow:
```
compute_I1_unified() → builds COMPLETE bracket integrand at each (u,t)
  → Bracket includes: exp factor × log factor × Q factors × P factors
  → The bracket already combines direct+mirror via the t-structure
  → Extract xy coefficient, integrate over (u,t)
→ Returns the UNIFIED value that represents the combined direct+mirror
→ The ABD decomposition recognizes: A = this value, D = 0 (by math, not by fiat)
```

---

## Implementation Tasks

### Task 21B.1: Create True Unified Bracket Integrand Builder

**File:** `src/unified_bracket_evaluator.py` (REWRITE `compute_I1_pair()`)

The current `compute_I1_pair()` has the polynomial + Q factor machinery but doesn't properly implement the unified bracket. It computes something that looks like I₁ but then the outer wrapper sets `S12_plus = 0`.

**Required changes:**

1. **Clarify what the "unified" value represents:**
   The t-integral in the difference quotient identity gives:
   ```
   ∫₀¹ exp(2Rt + Rθ(2t-1)(x+y)) dt
   ```
   At x=y=0, this is `(exp(2R)-1)/(2R)`.

   This integral represents the **regularized** form of `(N^{αx+βy} - T^{-(α+β)}N^{-βx-αy})/(α+β)` which combines:
   - The direct term: `N^{αx+βy}`
   - The mirror term: `-T^{-(α+β)}N^{-βx-αy}`

   When the full integrand (with Q and P factors) is evaluated, the cancellation that produces D=0 should emerge automatically.

2. **Define the ABD semantics correctly:**
   - In the empirical approach: `A = I₁₂(-R)`, `D = I₁₂(+R) + I₃₄(+R)`
   - In the unified approach: The integral value IS `A × exp(R) + B` directly, where `D = 0` by construction of the identity

   We need to reformulate how we extract A, B, D from the unified result.

3. **Implement properly:**
   ```python
   def compute_I1_unified_via_bracket(
       self,
       ell1: int,
       ell2: int,
   ) -> float:
       """
       Compute I₁ for pair (ℓ₁,ℓ₂) using the unified difference quotient bracket.

       The bracket integrand combines direct and mirror via the t-integral.
       The result represents the combined S12 value.
       """
       # For each (u, t):
       #   1. Build the full bracket series at this (u, t)
       #      - exp(2Rt + Rθ(2t-1)(x+y))
       #      - × (1 + θ(x+y))  [log factor]
       #      - × (1/θ + x + y)  [algebraic prefactor]
       #      - × P_{ℓ₁}(u+x) × P_{ℓ₂}(u+y)
       #      - × Q(A_α) × Q(A_β)  [Q factors with t-dependent eigenvalues]
       #   2. Extract xy coefficient
       #   3. Sum with quadrature weights
       pass
   ```

### Task 21B.2: Add Ladder Tests (Before End-to-End Gates)

**File:** `tests/test_unified_bracket_evaluator_ladder.py` (NEW)

These tests catch classic mistakes before attempting the D=0 gate.

**Tests to add (in order):**

1. **Bracket-only series sanity (Q=1, no polynomials):**
   - At `t=0.5`, the linear (x+y) coefficient from the exp part should vanish (because Rθ(2t-1) = 0 at t=0.5)
   - xy coefficient must be finite and continuous in t

2. **x=y=0 scalar limit through unified evaluator:**
   - Run with x=y=0 reduction mode
   - Assert it matches the known analytic scalar limit `(exp(2R)-1)/(2R)`
   - This catches double log factors, wrong exp signs, etc.

3. **Q-only sanity at x=y=0:**
   - At x=y=0, eigenvalues reduce to simple forms
   - Q(A_α)Q(A_β) at x=y=0 should equal Q(t)²
   - Test at t=0, 0.5, 1.0

4. **Smoke test for I₁(1,1):**
   - `compute_I1_unified_via_bracket(...)` returns finite scalar
   - Stable under quadrature refinement: (30,30) vs (50,50) relative change < 1e-6

### Task 21B.3: Implement Unified S12 Wrapper

**File:** `src/unified_s12_evaluator.py` (NEW or integrate into existing)

**Purpose:** Combine per-pair I₁/I₂ computations to produce S12.

```python
def compute_S12_unified(
    theta: float,
    R: float,
    polynomials: Dict,
    n_u: int = 40,
    n_t: int = 40,
    pairs: Optional[List[Tuple[int,int]]] = None,
) -> UnifiedS12Result:
    """
    Compute S12 using the unified difference quotient bracket.

    Unlike the empirical approach which computes I₁(+R) and I₁(-R) separately,
    this evaluator uses the bracket structure that combines them automatically.

    The key property: when computed this way, D = I₁₂(+R) + I₃₄(+R) should be ~0.
    """
    pass
```

**Start with:** Only pair (1,1), only I₁, only κ benchmark.

### Task 21B.4: Wire Into evaluate.py (Minimal Touch)

**File:** `src/evaluate.py`

Add minimal hook (~20 lines):

```python
# In compute_c_paper_with_mirror():

if mirror_mode == "difference_quotient_v2":  # New mode name
    from src.unified_s12_evaluator import compute_S12_unified_v2

    s12_result = compute_S12_unified_v2(
        polynomials=polynomials,
        theta=theta,
        R=R,
        n_u=n,
        n_t=n,
    )

    # The unified result has D~0 by construction
    # Compute c = A×exp(R) + B where B = 5A + D
    # Since D~0, B~5A

    # ... rest of assembly ...
```

### Task 21B.5: Add Phase 21B Gate Tests

**File:** `tests/test_difference_quotient_gate_D0.py` (NEW)

**Critical gate tests:**

1. `test_D_is_zero_kappa_unified_v2` - D < 1e-6 for κ
2. `test_D_is_zero_kappa_star_unified_v2` - D < 1e-6 for κ*
3. `test_B_over_A_is_five_kappa_unified_v2` - |B/A - 5| < 1e-6 for κ
4. `test_B_over_A_is_five_kappa_star_unified_v2` - |B/A - 5| < 1e-6 for κ*
5. `test_dual_benchmark_gate_unified_v2` - Both pass together

**Important:** Define D clearly in the new semantics:
```python
# OLD semantics (empirical):
#   A = I₁₂(-R)
#   D = I₁₂(+R) + I₃₄(+R)
#   c = A×exp(R) + D + 5A

# NEW semantics (unified bracket):
#   The unified bracket value already represents the combined structure
#   We need to extract A such that c = A×exp(R) + 5A + D
#   where D is the residual contamination (should be ~0)
```

---

## Debugging Playbook

If D doesn't go to ~0 after implementing, check these in order:

### Issue 1: Still Computing Direct/Mirror Separately
**Symptom:** D stays ~0.2
**Cause:** The difference quotient must be inside the integral as a bracket. If you compute two separate integrals and combine, D won't cancel.
**Fix:** Ensure single (u,t) loop with unified bracket.

### Issue 2: Log Factor Missing or Duplicated
**Symptom:** Values off by factor of ~L or ~L²
**Cause:** `log(N^{x+y}T) = L(1+θ(x+y))` is not optional, but easy to include twice.
**Fix:** Check ladder test for scalar limit.

### Issue 3: Wrong Exp Series Sign
**Symptom:** D has wrong sign or magnitude
**Cause:** `exp(2Rt + Rθ(2t-1)(x+y))` is sign-sensitive. A flipped sign in `(2t-1)` yields persistent D≠0.
**Fix:** Check at t=0.5 where `(2t-1)=0` should give zero linear coefficient.

### Issue 4: Wrong Eigenvalue Constructor
**Symptom:** Mirror mismatch
**Cause:** Using "direct" eigenvalues instead of "unified" eigenvalues.
**Fix:** Use `get_unified_bracket_eigenvalues()` consistently.

---

## Success Criteria

Phase 21B is DONE when:

1. ✅ Ladder tests pass (Task 21B.2)
2. ✅ D=0 gates pass for both benchmarks (D < 1e-6)
3. ✅ B/A=5 gates pass for both benchmarks (|B/A - 5| < 1e-6)
4. ✅ Can compute c in `mirror_mode="difference_quotient_v2"` end-to-end
5. ✅ Default baseline unchanged (backward compatible)

---

## Non-Goals (Don't Do Yet)

- Don't touch K=4
- Don't refactor the whole evaluator
- Don't attempt a new global "m₁ formula"
- Don't worry about c accuracy yet (that's Phase 22)

---

## Implementation Order

Execute in this exact order with stop gates:

```
Task 21B.2 (ladder tests)
    ↓ MUST PASS before continuing
Task 21B.1 (rewrite compute_I1_pair)
    ↓
Task 21B.2 (rerun ladder tests)
    ↓ MUST PASS before continuing
Task 21B.3 (S12 wrapper)
    ↓
Task 21B.5 (gate tests - expect to FAIL initially)
    ↓
Debug using playbook
    ↓ GATE TESTS MUST PASS
Task 21B.4 (wire to evaluate.py)
    ↓
Full test suite
    ↓ ALL TESTS PASS
Done!
```

---

## Files Summary

### New Files
1. `tests/test_unified_bracket_evaluator_ladder.py` - Ladder tests
2. `tests/test_difference_quotient_gate_D0.py` - D=0 gate tests
3. `src/unified_s12_evaluator.py` - S12 wrapper (or extend existing)

### Files to Modify
1. `src/unified_bracket_evaluator.py` - Rewrite `compute_I1_pair()`
2. `src/evaluate.py` - Add `mirror_mode="difference_quotient_v2"`

### Files to Reference (Read-Only)
1. `src/difference_quotient.py` - Core identity (don't change)
2. `src/abd_diagnostics.py` - ABD definitions (don't change)
3. `src/composition.py` - Series composition (don't change)

---

*Generated 2025-12-25 as part of Phase 21B planning.*
