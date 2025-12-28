# K=4 Implementation Plan

**Date:** 2025-12-26 (Updated 2025-12-27 with Phase 40 GPT guidance)
**Status:** PLANNED (awaiting K=4 polynomials)
**Phase:** 40 complete, K-generic infrastructure ready

---

## Prerequisites

Before implementing K=4:
1. ✅ Derived formula locked (Phase 36)
2. ✅ Q residual diagnostic in place (Phase 36 Priority 2)
3. ✅ K-sweep validation passed for K=4 proxy (Phase 39)
4. ✅ K-generic pairs module created (Phase 40) - `src/evaluator/pairs.py`
5. ⏳ Need: K=4 polynomial coefficients from PRZZ or optimization

**IMPORTANT CAVEAT (GPT guidance):**
Phase 39 validated K=4 structure using **K=3 polynomials as proxy**. This is useful
structurally but NOT a substitute for real K=4 polynomials. "Production-ready" is
conditional on real K=4 polynomials passing Steps 0-3.

---

## Formula for K=4

```
m(K=4, R) = [1 + θ/(2×4×9)] × [exp(R) + 7]
          = [1 + θ/72] × [exp(R) + 7]
          = 1.00794 × [exp(R) + 7]   (for θ=4/7)
```

Compared to K=3:
- K=3: 1 + θ/42 = 1.01361
- K=4: 1 + θ/72 = 1.00794

The correction shrinks by ~45% (0.794/1.361 ≈ 0.58).

---

## Microcase Progression (MANDATORY)

GPT's guidance: Do NOT go straight to P=real, Q=real at K=4.
Use the microcase ladder to validate each component.

### Step 0: P=Q=1 (Kernel-Only Sanity) [NEW - Phase 40]

**What:** Implement K=4 with P=1 and Q=1 (all constant polynomials).

**Why:** This validates the **K=4 plumbing** before polynomials enter:
- Pairs: 10 pairs generated correctly
- Variable counts: ℓ₁ + ℓ₂ variables per pair
- Factorial weights: 1/(ℓ₁! × ℓ₂!)
- Quadrature: converges correctly

**Expected:** Beta correction should appear cleanly = 1 + θ/72 = 1.00794

**Rationale:** Any weirdness here is a plumbing bug, not an optimization artifact.
This takes almost no time and prevents a ton of misdiagnosis later.

**Test:**
```python
def test_k4_p_one_q_one():
    # All constant polynomials
    P_one = Polynomial([1.0])
    Q_one = QPolynomial({0: 1.0})
    polys = {"P1": P_one, "P2": P_one, "P3": P_one, "P4": P_one, "Q": Q_one}

    # Compute ratio
    # Should give clean Beta correction without Q/P effects
```

### Step 1: P=real, Q=1

**What:** Implement K=4 with real P polynomials but Q=1 (constant).

**Why:** This validates:
- The Beta correction formula (1 + θ/72) is wired correctly
- The base term (exp(R) + 7) is correct
- The K=4 polynomial structure works

**Expected:** Ratio should be ~1.00794 (±0.1% from Beta moment)

**Test:**
```python
def test_k4_p_real_q_one():
    # Load K=4 polynomials (when available)
    P1, P2, P3, P4 = load_k4_polynomials()
    Q_one = Polynomial([1.0])
    polys = {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "Q": Q_one}

    diag = compute_q_residual_diagnostic(R=..., K=4, polynomials=polys)

    # With Q=1, the Q effect should be ~0
    assert abs(diag.Q_effect_on_correction_pct) < 0.1
```

### Step 2: P=1, Q=real

**What:** Use constant P=1 polynomials with real K=4 Q polynomial.

**Why:** This isolates the Q polynomial effect:
- Does the Q effect amplify at K=4?
- Is the Q effect still negative?
- Is the magnitude still controlled?

**Expected:** Q effect should be negative and bounded

**Dual Gate Thresholds (GPT guidance, Phase 40):**

| Gate | Threshold | Purpose |
|------|-----------|---------|
| **Safety** | `abs(Q_effect) < 5%` | Prevents blow-ups / sign catastrophe |
| **Precision** | `abs(Q_effect) < 0.5%` | Keeps near K=3 regime (~0.15% residual) |

K=4 doesn't get blocked by tiny increase, but you know immediately if Q dominates.

**Test:**
```python
def test_k4_p_one_q_real():
    # Use constant P=1
    P_one = Polynomial([1.0])
    Q = load_k4_q_polynomial()
    polys = {"P1": P_one, "P2": P_one, "P3": P_one, "P4": P_one, "Q": Q}

    diag = compute_q_residual_diagnostic(R=..., K=4, polynomials=polys)

    # Safety gate: Q effect should be negative and < 5%
    assert diag.Q_effect_on_correction_pct < 0
    assert abs(diag.Q_effect_on_correction_pct) < 5.0  # Safety

    # Precision gate: ideally < 0.5%
    if abs(diag.Q_effect_on_correction_pct) > 0.5:
        warnings.warn(f"Q effect {diag.Q_effect_on_correction_pct:.2f}% exceeds precision threshold")
```

### Step 3: P=real, Q=real

**What:** Full K=4 with all real polynomials.

**Why:** This is the production case. Only run after Steps 1-2 pass.

**Expected:**
- All Q residual gates pass
- Accuracy similar to K=3 (±0.15% or better)

---

## Infrastructure Requirements

### 1. New Polynomial Loader

```python
def load_przz_polynomials_k4():
    """Load K=4 polynomials from PRZZ or optimization."""
    # P4 polynomial structure
    # Q polynomial (may have different structure at K=4)
    pass
```

### 2. K-Generic Pairs Module [COMPLETE - Phase 40]

**DO NOT create `terms_k4_d1.py` as a fork!** (GPT guidance)

Instead, use `src/evaluator/pairs.py` which provides K-generic functions:

```python
from src.evaluator.pairs import get_triangle_pairs, factorial_norm, symmetry_factor

# K=4 has 10 pairs
pairs = get_triangle_pairs(K=4)
# [(1,1), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,3), (3,4), (4,4)]

# Normalization is computed, not hardcoded
norm = factorial_norm(1, 4)  # Returns 1/24
sym = symmetry_factor(1, 4)  # Returns 2.0
```

This prevents:
- Code path divergence between K=3 and K=4
- Silent semantic drift
- Technical debt (6,700-line evaluate.py style)

### 3. Factorial Normalization [COMPLETE - Phase 40]

No longer hardcoded. Computed by `factorial_norm(l1, l2)`:

| Pair | 1/(ℓ₁!×ℓ₂!) |
|------|-------------|
| (1,1) | 1.0 |
| (1,2) | 0.5 |
| (1,3) | 1/6 |
| (1,4) | 1/24 |
| (2,2) | 0.25 |
| (2,3) | 1/12 |
| (2,4) | 1/48 |
| (3,3) | 1/36 |
| (3,4) | 1/144 |
| (4,4) | 1/576 |

### 4. Q Residual Diagnostic

The diagnostic already supports arbitrary K. Just pass K=4:
```python
diag = compute_q_residual_diagnostic(R=..., K=4, polynomials=polys_k4)
```

---

## Validation Checklist

Before declaring K=4 production-ready:

- [ ] **Step 0:** P=Q=1 gives clean Beta correction (1.00794)
- [ ] **Step 1:** P=real, Q=1 passes (ratio ~1.00794, Q effect ~0)
- [ ] **Step 2:** P=1, Q=real passes:
  - Safety gate: Q effect negative and < 5%
  - Precision gate: Q effect < 0.5% (warning if exceeded)
- [ ] **Step 3:** P=real, Q=real passes all gates
- [ ] K-sweep still shows "shrinking" trend at K=4
- [ ] Accuracy on K=4 benchmark (when available) is ±0.15% or better
- [ ] Pair count gate: 10 pairs with correct normalization (use `validate_k_pairs(4)`)

---

## Known Risks

### 1. Q Polynomial Structure May Differ

At K=4, the PRZZ Q polynomial may have:
- Different degree
- Different coefficient structure
- Different Q(0)/Q(1) boundary values

The Q residual diagnostic will catch amplification.

### 2. I1/I2 Split May Shift

At K=4, the (1-u)^{2K} weight becomes (1-u)^8 instead of (1-u)^6.
This may shift the I1/I2 balance slightly.

The diagnostic reports I1 share, so we'll see if this changes.

### 3. 10 Pairs Instead of 6

More terms means more quadrature time.
Consider n_quad=40 for exploratory runs, n_quad=60 for production.

---

## Timeline

1. **When K=4 polynomials available:** Run microcase ladder (Steps 1-3)
2. **If all pass:** K=4 is production-ready
3. **If gates fail:** Investigate Q polynomial structure at K=4

---

## References

- Phase 36: Derived formula locked
- Phase 37: Frozen-Q experiment (Q effect mechanism)
- Phase 38: Q moment analysis
- Phase 39: K=4 safety check (formula validated structurally)
- **Phase 40: Q correction analysis + K-generic infrastructure**
- GPT Priority 3 guidance (microcase progression)
- GPT Phase 40 guidance (Step 0, dual gates, K-generic design)

---

## Files Related to K=4

| File | Status | Purpose |
|------|--------|---------|
| `src/evaluator/pairs.py` | ✅ COMPLETE | K-generic pairs and normalization |
| `tests/test_pairs_k_generic.py` | ✅ COMPLETE | 26 tests for K-generic pairs |
| `src/terms/build_terms.py` | ⏳ TODO | K-generic terms generation |
| `tests/test_k4_microcase_gates.py` | ⏳ TODO | K=4 gate tests (skip if no polys) |
| `scripts/run_k4_microcase_ladder.py` | ⏳ TODO | K=4 microcase runner |
