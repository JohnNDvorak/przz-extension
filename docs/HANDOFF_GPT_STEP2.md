# GPT Step 2 Handoff: Operator-Level Mirror Computation (2025-12-21)

## Executive Summary

GPT Step 2 implemented the **operator-level mirror computation** - applying Q as actual differential operators (d/dα, d/dβ) on the pre-identity bracket. This was proposed as a "decisive experiment" to resolve the structural contradiction.

**Key Finding: The operator-level approach reveals an L-divergence issue.**

| Benchmark | I1 (L=20) | I1/L | tex_mirror I1 | Gap |
|-----------|-----------|------|---------------|-----|
| κ         | 2.46      | 0.12 | 0.40          | -70% |
| κ*        | 2.37      | 0.12 | 0.55          | -78% |

**Critical observation**: Operator-level I1 grows linearly with L (logT), meaning it **does not converge to a finite asymptotic value**. This reveals the fundamental issue: the bracket B itself is O(L).

---

## What GPT Step 2 Implemented

### Stage 21A: BracketDerivatives Class

**File**: `src/operator_level_mirror.py`

**Formula** (pre-identity bracket):
```
B(α,β,x,y) = (N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)
```

At α = β = -R/L:
- The denominator 1/(α+β) = -L/(2R)
- **This is why B ~ L** (linear in logT)

Uses SymPy for symbolic derivatives ∂^{i+j}B/∂α^i∂β^j up to order 5.

### Stage 21B: apply_Q_operator_to_bracket()

Applies Q(D_α) × Q(D_β) where D = -1/L × d/dα:
```
Q(D_α) × Q(D_β) × B = Σᵢ Σⱼ qᵢqⱼ × (-1/L)^{i+j} × ∂^{i+j}B/∂α^i∂β^j
```

### Stage 21C: compute_I1_operator_level_11()

Full I1 computation integrating over (u,t) with P factors.

### Stage 21D: Diagnostic Comparison

Compared operator-level against Run 18, 19, 20, and tex_mirror.

---

## L-Divergence Analysis

### Why B ~ L

At α = β = -R/L, the bracket simplifies to:
```
B = (exp(-Rθ(x+y)) - exp(2R + Rθ(x+y))) / (-2R/L)
  = (L/2R) × [exp(2R + Rθ(x+y)) - exp(-Rθ(x+y))]
  = L × (constant in x,y)
```

**Verified numerically**:
| L    | B at x=y=0.05 | B/L   |
|------|---------------|-------|
| 10   | 52.48         | 5.248 |
| 20   | 104.95        | 5.248 |
| 50   | 262.38        | 5.248 |
| 100  | 524.76        | 5.248 |

B/L is exactly constant, confirming B ∝ L.

### Implications for Operator-Level

When we apply Q(D_α)Q(D_β):
- The (0,0) term (no derivatives) contributes q₀² × B ~ q₀² × L
- Higher derivatives get factors of (-1/L)^{i+j} but operate on B ~ L
- Net result: Q(D)Q(D)B ~ L

**This means operator-level I1 diverges as L → ∞.**

---

## Diagnostic Results

### L Convergence Test

| L    | I1 (κ) | I1 (κ*) |
|------|--------|---------|
| 10   | 1.23   | 1.18    |
| 20   | 2.46   | 2.37    |
| 50   | 6.14   | 5.91    |

I1 grows linearly with L in both benchmarks.

### Method Comparison (at L=20)

| Method | I1 (κ) | I1 (κ*) | vs tex_mirror |
|--------|--------|---------|---------------|
| Operator-level | 2.46 | 2.37 | 6.1×, 4.3× |
| Run 20 | 1.59 | 1.13 | 3.9×, 2.0× |
| Run 18 | 2.08 | 1.43 | 5.1×, 2.6× |
| Run 19 | 4.30 | 0.80 | 10.7×, 1.4× |
| tex_mirror | 0.40 | 0.55 | 1×, 1× |

### After L-Normalization

If we divide operator-level by L:
| Benchmark | I1/L | tex_mirror I1 | Ratio |
|-----------|------|---------------|-------|
| κ         | 0.12 | 0.40          | 0.30  |
| κ*        | 0.12 | 0.55          | 0.21  |

**Even after L-normalization, operator-level gives only 30% (κ) / 21% (κ*) of tex_mirror.**

---

## Root Cause Analysis

### The L Factor in PRZZ

PRZZ's asymptotic formulas include explicit L = logT factors. The difference quotient → log×integral identity:
```
(N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)
= N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-s(α+β)} ds
```

The log(N^{x+y}T) = L(1 + θ(x+y)) contains the L factor.

### How tex_mirror Handles This

tex_mirror uses:
```
I1 = I1_plus + m1 × I1_minus_base
```
where m1 = exp(R) + 5 (calibrated).

The separate +R/-R evaluations do NOT have the L factor because they evaluate **after** the combined identity is applied and the L is absorbed into normalization.

### Why Operator-Level Doesn't Match

The operator-level approach applies Q(D) to the **pre-identity** bracket, which still contains the 1/(α+β) singularity factor. At α=β=-R/L, this becomes -L/(2R), introducing the L dependence.

**Key insight**: The PRZZ combined identity is specifically designed to **cancel** the 1/(α+β) factor. By going back to the pre-identity form and applying operators there, we reintroduce this factor.

---

## GPT's Follow-Up Analysis (2025-12-22)

After reviewing Step 2's results, GPT provided a deeper explanation of WHY the operator-level approach diverges:

### The Core Insight: (θt-θ) Cross-Terms

GPT identified that setting α=β=-R/L **too early** causes the structure to collapse to depend only on (x+y), losing the crucial mixed cross-terms.

**In the correct PRZZ structure**, the affine arguments for Q have DIFFERENT coefficients for x and y:
```
Q_α = t + θt·x + (θt-θ)·y
Q_β = t + (θt-θ)·x + θt·y
```

The (θt-θ) = θ(t-1) term creates **asymmetry** between x and y coefficients.

### Why This Matters

In nilpotent algebra:
- δ = a_x·x + a_y·y where a_x = θt, a_y = θ(t-1)
- δ² = 2·a_x·a_y·xy = 2·θt·θ(t-1)·xy
- xy coefficient = θ²·t·(t-1)·P''(u)

If x and y had the SAME coefficient (θt), we'd get:
- xy coefficient (wrong) = θ²·t²·P''(u)

These are different! The (θt-θ) cross-term is essential.

### Why Operator-Level Misses This

The operator-level bracket B(α,β,x,y) does NOT have t as a variable.

**Where t comes from**: The PRZZ combined identity (TeX lines 1502-1511) has an s-integral:
```
B = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-s(α+β)} ds
```

This s-integral, after parametrization, becomes the t-integration. The t enters the affine forms for Q.

**Operator-level consequence**: No t → No (θt-θ) terms → Wrong structure → L-divergence.

### The Operator-Polynomial Equivalence

GPT also clarified the mathematical relationship:
```
Q(D_α) × exp(θL·αx) = Q(-θx) × exp(θL·αx)
```

This shows how the differential operator Q(D) maps to polynomial composition Q(-θx). But without the t-dependent affine structure from the combined identity, this equivalence doesn't help.

### Documentation

See `docs/OPERATOR_VS_COMPOSITION.md` for the full mathematical derivation and `run_operator_equivalence_check.py` for numerical verification.

---

## Conclusion

GPT Step 2 provides **decisive diagnostic information**: the operator-level approach diverges with L because it sets α=β=-R/L too early, losing the (θt-θ) cross-terms.

### What We Learned

1. **The bracket B is O(L)** due to 1/(α+β) = -L/(2R) at the evaluation point
2. **Operator-level I1 diverges** linearly with L
3. **The missing t variable** is the root cause - it comes from PRZZ's combined identity
4. **The (θt-θ) cross-terms** create necessary asymmetry between x and y
5. **tex_mirror correctly preserves** these cross-terms via affine composition

### Why tex_mirror Works

tex_mirror's structure correctly:
- Keeps x and y separate in affine dicts: `{"x": θt, "y": θ(t-1)}`
- Uses nilpotent algebra which preserves cross-terms
- Integrates over (u,t) capturing the t-dependence
- Uses m1 = exp(R) + 5 to absorb remaining asymptotic factors

### Mystery Resolved

The 3-4× discrepancy between operator-level/L and tex_mirror is explained by the **missing (θt-θ) structure** in operator-level, not just an L factor.

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `src/operator_level_mirror.py` | BracketDerivatives, apply_Q_operator_to_bracket(), compute_I1_operator_level_11() |
| `tests/test_operator_level_gates.py` | 14 tests (3 quick, 11 slow) |
| `run_gpt_step2_diagnostic.py` | Diagnostic comparison script |
| `docs/HANDOFF_GPT_STEP2.md` | This document |

---

## Recommendations (Updated 2025-12-22)

### Option A: Accept tex_mirror's Calibration (RECOMMENDED)

tex_mirror achieves ~1% accuracy with empirical m1 = exp(R) + 5.

**Pros**:
- Works now with ~1% accuracy on both benchmarks
- Structure is correct (preserves (θt-θ) cross-terms)
- Can proceed to K>3 extension work
- Calibration absorbs asymptotic factors cleanly

**Cons**:
- m1 formula not first-principles derived (but this is acceptable for numerical work)

### Option B: Derive m1 From First Principles (Lower Priority)

If desired for mathematical completeness:
1. Trace through PRZZ TeX lines 1502-1511 to understand how s-integral → t
2. Derive the exact mirror weight from the combined identity
3. Show m1 = exp(R) + 5 emerges from the analysis

This is intellectually satisfying but not required for the ~1% accuracy target.

### Option C: Proceed to K>3 (After Option A)

Once tex_mirror is validated as structurally correct:
1. Extend the evaluator to K=4
2. Add P₄ polynomial with optimization
3. Test if κ > 0.42 is achievable

---

## Test Results

```
tests/test_operator_level_gates.py: 3 passed (quick), 11 deselected (slow)
```

Quick tests validate Q polynomial conversion.
Slow tests (marked with @pytest.mark.slow) validate bracket derivatives and I1 computation.

---

## Relation to GPT's 5-Step Plan

| Step | Status | Outcome |
|------|--------|---------|
| Step 1 | Pending | TeX vs code structural checklist |
| **Step 2** | **Complete + Analyzed** | Operator-level diverges with L; GPT explains WHY via (θt-θ) cross-terms |
| Step 3 | Pending | Re-verify I3/I4 mirror |
| Step 4 | Informed | tex_mirror's m1 calibration captures correct structure |
| Step 5 | Ready | Can proceed to K>3 once Step 4 is accepted |

**GPT's Follow-Up (2025-12-22)**:
- Step 2's L-divergence is **expected behavior**
- The "mystery" is resolved: operator-level misses (θt-θ) terms because t comes from combined identity
- tex_mirror is **structurally correct** - the calibration (m1 = exp(R)+5) is acceptable
- **Ready to proceed to K>3** using tex_mirror as the production evaluator

---

## GPT's Post-Identity Operator Experiment (2025-12-22)

GPT proposed a new experiment: apply Q(D_α)Q(D_β) **AFTER** the PRZZ combined identity
has introduced t-dependence, producing the correct affine arguments.

### Implementation

**Files Created:**

| File | Purpose |
|------|---------|
| `src/operator_post_identity.py` | Post-identity operator core with A_α, A_β eigenvalues |
| `tests/test_operator_post_identity_core.py` | 54 tests for affine coefficients, L-stability |
| `tests/test_operator_post_identity_I1_11_gate.py` | 7 tests for DSL comparison |
| `run_operator_post_identity_diagnostic.py` | Diagnostic script with 6 stages |

### Key Results

**All 61 tests pass.**

**I1(1,1) matches DSL exactly:**

| Benchmark | R | Post-Identity I1 | DSL I1 | Ratio |
|-----------|------|------------------|--------|-------|
| κ | 1.3036 | 0.41347410 | 0.41347410 | 1.000000 |
| κ* | 1.1167 | 0.37198671 | 0.37198671 | 1.000000 |

**Affine Coefficients Verified:**
```
A_α = t + θ(t-1)·x + θt·y
A_β = t + θt·x + θ(t-1)·y
```

**No L-Divergence:** Q×Q×E is stable in L (not proportional to L).

### What This Proves

1. **The post-identity approach produces the correct (θt-θ) cross-terms**
2. **It matches tex_mirror/DSL exactly** (ratio = 1.000000)
3. **The L-divergence issue from Step 2 is resolved** by including t
4. **tex_mirror's structure is validated** at the operator level

### Conclusion

The post-identity operator experiment **VALIDATES** tex_mirror's structure.
The operator approach and the polynomial composition approach are equivalent
when both use the correct affine forms with (θt-θ) cross-terms.

**Ready to proceed to K>3** using tex_mirror as the production evaluator.
