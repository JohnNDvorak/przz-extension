# Phase 20.3 Findings: exp(R) Coefficient Residual Analysis

**Date:** 2025-12-24
**Status:** Research Complete

---

## Executive Summary

**Key Finding:** The production pipeline's exp(R) coefficient A is ~10% below target, but this is compensated by D > 0, resulting in ~1.3% c accuracy. A single global factor of ~1.11× would fix A for both benchmarks.

**Key Insight from PRZZ Research:** Achieving D = 0 analytically requires implementing the full PRZZ difference quotient identity (Lines 1502-1511) with proper operator shifts, rather than the current post-hoc scalar `m = exp(R) + 5`.

---

## 1. Production Pipeline exp(R) Coefficient Analysis

### 1.1 Summary Table

| Metric | κ (R=1.3036) | κ* (R=1.1167) | Ratio |
|--------|--------------|---------------|-------|
| A (production) | 0.2201 | 0.2164 | 1.017 |
| A (target) | 0.2462 | 0.2406 | 1.023 |
| A/A_target | 0.8942 | 0.8996 | - |
| A gap | -10.58% | -10.04% | - |
| D = S12+ + S34+ | +0.197 | +0.171 | - |
| B/A | 5.90 | 5.79 | - |
| c gap | -1.35% | -1.21% | - |

### 1.2 Key Observations

1. **A is ~10% too small** for both benchmarks (A_ratio ≈ 0.89)
2. **D is positive** (D ≈ +0.20), causing B/A > 5
3. **c accuracy is good** (~1.3%) because A and D errors partially compensate
4. **Single factor could work**: A/A_target differs by only 0.6% between benchmarks

### 1.3 Per-Piece Contributions to A

The exp(R) coefficient A = I₁₂(-R) = J11(-R) + J12(-R):

| Piece | κ | κ* |
|-------|---|----|
| J11(-R) | 46.5% | 41.4% |
| J12(-R) | 53.5% | 58.6% |

J12 dominates because it contains the (ζ'/ζ)² factor.

---

## 2. PRZZ Difference Quotient Analysis

### 2.1 The Difference Quotient Identity (Lines 1502-1511)

```latex
(N^{αx+βy} - T^{-α-β}N^{-βx-αy})/(α+β)
= N^{αx+βy} log(N^{x+y}T) ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

This transformation:
- Removes the singularity at α+β=0 analytically
- Converts the 1/(α+β) factor to a t-integral representation
- Pre-combines direct and mirror terms BEFORE operator application

### 2.2 Why Current Implementation Has D ≠ 0

**Current approach:**
```
I_mirror ≈ m × I(-R)   [post-hoc scalar multiplication]
```

This applies the mirror as a separate scalar multiplier AFTER evaluating integrals separately.

**Correct PRZZ approach:**
```
I_combined = [difference quotient evaluation with mirror built-in]
```

The difference quotient identity combines direct and mirror WITHIN the integral, producing automatic cancellation that gives D = 0.

### 2.3 The Operator Shift Identity

For mirror terms, the polynomial Q transforms:
```
Q(D_α)(T^{-s}F) = T^{-s} × Q(1 + D_α)F
```

This means the mirror term uses Q(1+z), not just Q(z) with R→-R.

### 2.4 What's Needed for D = 0

1. **Implement difference quotient evaluation**: Treat the bracket as a single unit with t-integral
2. **Implement operator shift**: Use binomial-shifted polynomial Q(1+z) for mirror
3. **Unify I₁/I₂ evaluation**: Build mirror into the integral structure
4. **Remove empirical shim**: Replace `m = exp(R) + 5` with first-principles derivation

---

## 3. Why c Accuracy is Good Despite Residuals

### 3.1 Error Compensation

The ~10% underestimate in A is compensated by D > 0:
- A too small → c = A×exp(R) + B pulls down
- D > 0 → B = D + 5A pulls up
- Net effect: partial cancellation, ~1.3% c gap

### 3.2 Formula Structure

```
c = A × exp(R) + B
  = A × exp(R) + D + 5A
  = A × (exp(R) + 5) + D
```

When A is 10% low but D is +0.2 (positive), the errors partially cancel.

---

## 4. Recommendations

### 4.1 Short-term: Accept Current Pipeline

The current production pipeline with `m = exp(R) + 5` achieves ~1.3% accuracy on c, which is acceptable for most purposes.

### 4.2 Long-term: Implement Exact Difference Quotient

For exact B/A = 5 (D = 0):
1. Implement PRZZ Lines 1502-1511 difference quotient → integral representation
2. Apply operator shift Q(D) → Q(1+D) for mirror terms
3. Evaluate direct+mirror as unified structure
4. Remove empirical `m = exp(R) + 5` shim

**Estimated effort:** Significant implementation work required.

---

## 5. Test Coverage

22 tests created in `tests/test_amplitude_analysis.py`:
- `TestExpCoefficientAnalysis`: Single-benchmark A analysis
- `TestPerPieceContributions`: J11/J12 breakdown
- `TestCrossBenchmarkAnalysis`: κ vs κ* comparison
- `TestSingleFactorAnalysis`: Global factor feasibility
- `TestPhase20_3KeyFindings`: Key findings validation

All tests pass.

---

## References

- PRZZ TeX Lines 1502-1511: Difference quotient identity
- PRZZ TeX Lines 1514-1517: Q operator structure
- `docs/TEX_MIRROR_OPERATOR_SHIFT.md`: Operator shift theorem
- `PLAN_PHASE6_DERIVED_MIRROR.md`: Implementation roadmap
- `src/combined_identity_unified_t.py`: Existing t-parameterization code

---

*Generated 2025-12-24 as part of Phase 20.3 implementation.*
