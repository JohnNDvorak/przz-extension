# Phase 7: TeX-Exact Evaluation — Eliminate m₁ Scalar Dependency

**Date:** 2025-12-22
**Status:** APPROVED
**Goal:** Build a TeX-faithful evaluation path that computes c(R,θ) directly from PRZZ formulas WITHOUT the scalar m₁ mirror multiplier.

---

## Executive Summary

GPT's Phase 6 analysis revealed that the Q-shift ratio (85-127×) differs dramatically from the empirical m₁ (~8.68). This is **not a derivation failure** — it's evidence that **"m₁ as a single scalar" is the wrong target**.

The combined identity (TeX lines 1502-1511) already INTERNALIZES the mirror contribution. Once we use that object, there is no external scalar m₁ to derive. The current m₁ is a projection weight induced by our chosen decomposition — not a TeX constant.

**The derive-first move:** Build an evaluator that computes c directly from TeX formulas, with no m₁ concept. If this matches the published κ=0.417293962, then m₁ is provably non-fundamental.

---

## Phase 7 Overview

| Phase | Goal | Deliverables |
|-------|------|--------------|
| **7A** | TeX-exact I₁-I₄ evaluator (no m₁) | `src/tex_exact_k3.py`, validation tests |
| **7B** | Prefactor/normalization audit | `tests/test_logN_logT_consistency.py`, numerical identity checks |
| **7C** | Channel projection diagnostic | `run_channel_projection_diagnostics.py`, prove scalar m₁ is not TeX-exact |
| **7D** | Polynomial regression anchor | `tests/test_tex_polynomials_match_paper.py` |

---

## Phase 7A: TeX-Exact K=3 Evaluator

### Objective

Create `src/tex_exact_k3.py` that implements PRZZ formulas DIRECTLY with NO mirror multiplier.

### Mathematical Basis

From PRZZ TeX lines 1502-1511, the combined identity is:

```
[N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
= N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

This is a SINGLE integral representation that includes BOTH direct and mirror contributions. After applying Q(D_α)Q(D_β) and setting α=β=-R/L, we get the post-identity exponential core with t-dependent eigenvalues.

**Key insight:** The combined identity IS the TeX formula. There's nothing to "add back."

### Current Post-Identity Implementation

`src/operator_post_identity.py` already implements this correctly:

- Eigenvalues: `A_α = t + θ(t-1)x + θt·y`, `A_β = t + θt·x + θ(t-1)·y`
- Exp factor: `exp(R*(Arg_α + Arg_β))` with affine x,y dependence
- Series algebra via `TruncatedSeries`

**Question:** Does this ALONE (without mirror assembly) match the published κ?

### Implementation Plan

#### File: `src/tex_exact_k3.py`

```python
"""
TeX-Exact K=3 Evaluator — NO MIRROR MULTIPLIER.

This module computes c(R,θ) directly from PRZZ TeX formulas (lines 1502-1533).
The combined identity already includes both direct and mirror contributions —
there is no external scalar m₁ in the TeX.

PURPOSE: Prove that either:
(a) TeX-exact evaluation matches κ=0.417293962 → m₁ is non-fundamental
(b) TeX-exact evaluation differs → identify which term's transcription is off

ARCHITECTURE:
- I₁: (u,t) integral with Q(A_α)Q(A_β)exp, polynomial profiles, d²/dxdy
- I₂: t-integral with Q(t)²exp, polynomial integral over u
- I₃: (u,t) integral with d/dx derivative
- I₄: (u,t) integral with d/dy derivative

KEY INVARIANT: No `m1_formula`, no `mirror_multiplier`, no separate ±R channels.
"""
```

**Functions to implement:**

1. `compute_I1_tex_exact(pair, theta, R, n, polynomials)` — Post-identity formula
2. `compute_I2_tex_exact(pair, theta, R, n, polynomials)` — ∫Q(t)²e^{2Rt}dt form
3. `compute_I3_tex_exact(pair, theta, R, n, polynomials)` — d/dx derivative
4. `compute_I4_tex_exact(pair, theta, R, n, polynomials)` — d/dy derivative
5. `compute_c_tex_exact(theta, R, n, polynomials)` — Full assembly

**Key differences from current code:**
- NO call to `m1_formula()` or `m1_policy`
- NO separate `+R` and `-R` evaluations
- Single integral path using post-identity eigenvalues
- Assembly: `c = Σ_pairs (I₁ + I₂ + I₃ + I₄)` with factorial normalization

#### File: `tests/test_tex_exact_k3_matches_existing.py`

**Tests:**
1. Per-pair I₁ matches `compute_I1_operator_post_identity_pair` (tight tolerance 1e-10)
2. Per-pair I₂ matches existing DSL I₂ (from `terms_k3_d1.py`)
3. Full c comparison with known baselines
4. Both κ and κ* benchmarks tested

### Acceptance Criteria (7A)

- [ ] `tex_exact_k3.py` computes c without any m₁ concept
- [ ] Per-pair outputs match trusted paths to 1e-10
- [ ] Document gap (if any) from target c=2.137

---

## Phase 7B: Prefactor and Normalization Audit

### Objective

TeX is specific about where log N, log T, and θ sit. Eliminate "mysterious amplitude drift."

### Implementation Plan

#### File: `tests/test_logN_logT_consistency.py`

```python
"""
Normalization Contract Tests.

TeX defines:
- N = T^θ, so log N = θ × log T
- Combined identity introduces log(N^{x+y}T) = (1+θ(x+y)) × log T
- Operators D_α = -1/L × ∂/∂α where L = log T
"""

# Test 1: Verify logN == theta*logT in all eigenvalue computations
# Test 2: Verify combined identity prefactor is L(1+θ(x+y))
# Test 3: Verify exp factors use consistent L definition
```

#### Numerical Identity Check:

```python
def test_combined_identity_integral_equals_difference_quotient():
    """
    Verify the combined identity holds numerically.

    For random small α, β and nilpotent (x, y) coefficients:
    LHS ≈ RHS to machine precision
    """
```

### Acceptance Criteria (7B)

- [ ] All log N / log T relationships verified in tests
- [ ] Combined identity numerically validated
- [ ] Any normalization bugs found and fixed

---

## Phase 7C: Channel Projection Diagnostic

### Objective

If scalar m₁ is not TeX-exact, prove it by computing pair-dependent effective m₁ values.

### Mathematical Setup

```
m₁_eff(ℓ₁, ℓ₂; R) = [I₁_TeX(R) - I₁_chan+(R)] / I₁_chan-(R)
```

If m₁_eff varies significantly across pairs/benchmarks, then **a scalar m₁ cannot be TeX-exact**.

### Implementation Plan

#### File: `run_channel_projection_diagnostics.py`

```python
"""
Channel Projection Diagnostics — Is scalar m₁ TeX-exact?

OUTPUT: Table of m₁_eff values with variance statistics
"""
```

**Expected result based on Phase 6:** m₁_eff will vary significantly.

### Acceptance Criteria (7C)

- [ ] m₁_eff computed for all 6 pairs at both benchmarks
- [ ] Variance statistics printed
- [ ] If variance > 10%, document "scalar m₁ is not TeX-exact"

---

## Phase 7D: Polynomial Regression Anchor

### Objective

PRZZ TeX includes explicit polynomial coefficients. Make this a strict regression gate.

### Implementation Plan

#### File: `tests/test_tex_polynomials_match_paper.py`

```python
"""
TeX Polynomial Regression Tests.

Sources:
- κ polynomials: PRZZ TeX
- κ* polynomials: Lines 2587-2598

Constraints:
- P₁(0) = 0, P₁(1) = 1
- P₂(0) = P₃(0) = 0
- Q(0) = 1
"""
```

### Acceptance Criteria (7D)

- [ ] All polynomial coefficients verified against TeX
- [ ] Q(0) = 1 confirmed for both benchmarks
- [ ] Any transcription errors documented

---

## Implementation Order

1. **Phase 7D first** — Verify polynomials before building new evaluator
2. **Phase 7B second** — Ensure log N / log T consistency
3. **Phase 7A third** — Build tex_exact_k3.py
4. **Phase 7C last** — Diagnostic only meaningful after 7A

---

## Files to Create

| File | Purpose | Lines (est) |
|------|---------|-------------|
| `src/tex_exact_k3.py` | TeX-exact evaluator | 400-500 |
| `tests/test_tex_exact_k3_matches_existing.py` | Validation | 200-300 |
| `tests/test_logN_logT_consistency.py` | Normalization | 100-150 |
| `tests/test_combined_identity_numerical.py` | Identity check | 80-100 |
| `tests/test_tex_polynomials_match_paper.py` | Polynomial gate | 150-200 |
| `run_channel_projection_diagnostics.py` | m₁_eff analysis | 150-200 |

**Total: ~1100-1450 new lines**

---

## Success Criteria Summary

**Phase 7 is SUCCESSFUL if:**

1. TeX-exact evaluator produces consistent results
2. We can definitively say whether scalar m₁ is:
   - (a) Non-fundamental (TeX-exact matches target), OR
   - (b) A structurally approximate projection (variance in m₁_eff)
3. All normalization/prefactor relationships are tested
4. Polynomial coefficients are verified against TeX source

---

## References

- GPT Phase 7 guidance (2025-12-22)
- PRZZ TeX lines 1502-1511 (combined identity)
- PRZZ TeX lines 1529-1533 (I₁ formula)
- PRZZ TeX lines 2587-2598 (κ* polynomials)
- `docs/TRUTH_SPEC.md` Section 5 (mirror combination)
- `docs/TEX_MIRROR_OPERATOR_SHIFT.md` (Phase 6 derivation)
- `docs/DECISIONS.md` Decision 6 (Q-shift investigation)
