# Phase 20.2 Findings: Main-Only B/A Gap Analysis

**Date:** 2025-12-24
**Status:** Research Complete

---

## Executive Summary

**Finding:** The main-term-only B/A ratio deviates from target 5 in BOTH directions depending on which pipeline is used:

| Pipeline | D = I₁₂(+R) + I₃₄(+R) | δ = D/A | B/A = 5 + δ | Gap |
|----------|----------------------|---------|-------------|-----|
| Simplified J1 (j1_euler_maclaurin) | -0.73 | -0.72 | 4.28 | -14.4% |
| Production Case C (evaluate.py) | +0.20 | +0.90 | 5.90 | +17.9% |
| **Target (PRZZ exact)** | **0** | **0** | **5** | **0%** |

**Conclusion:** The "+5" in the mirror formula `m = exp(R) + 5` is an empirical shim that produces accurate c values, but the internal B/A structure is off because neither pipeline uses the exact PRZZ formulas.

---

## 1. Problem Statement

Phase 20.2 sought to understand why main-only B/A ≈ 4.28 instead of 5 in the simplified J1 diagnostic. The target from PRZZ is:

```
B/A = 2K - 1 = 5 for K = 3
```

This should hold for the **main term only**, without needing J₁,₅ (which is an error term per TRUTH_SPEC Lines 1621-1628).

---

## 2. The Two Pipelines

### 2.1 Simplified J1 Pipeline (`j1_euler_maclaurin.py`)

This diagnostic pipeline uses:
- Simple polynomial products: ∫P₁(u)P₂(u) du
- Laurent approximations for (ζ'/ζ) factors
- NO Case C kernel structure

**Result at κ benchmark (R=1.3036):**
```
I₁₂(+R) = -0.07  (j11: +0.47, j12: -0.54)
I₁₂(-R) = +1.01  (j11: +0.47, j12: +0.54)
I₃₄(+R) = -0.66  (j13: -0.33, j14: -0.33)

A = I₁₂(-R) = 1.01
D = I₁₂(+R) + I₃₄(+R) = -0.73
δ = D/A = -0.72
B/A = 5 + δ = 4.28
```

### 2.2 Production Case C Pipeline (`evaluate.py`)

This pipeline uses:
- Case C kernel Taylor coefficients for P₂/P₃
- Proper Term DSL structure
- Auxiliary a-integrals per PRZZ Section 7

**Result at κ benchmark (R=1.3036):**
```
S12(+R) = +0.80
S12(-R) = +0.22  <- This is A
S34(+R) = -0.60

A = S12(-R) = 0.22
D = S12(+R) + S34(+R) = +0.20
δ = D/A = +0.90
B/A = 5 + δ = 5.90
```

---

## 3. Key Observations

### 3.1 The Gap is in D, Not the Mirror Formula

Both pipelines correctly implement the mirror formula:
```
c = I₁₂(+R) + [exp(R) + 5] × I₁₂(-R) + I₃₄(+R)
c = A × exp(R) + B  where B = D + 5 × A
```

The "+5" contribution comes from the mirror multiplier `m = exp(R) + 5`, not from the integrals themselves. The gap arises because D ≠ 0.

### 3.2 Case C Kernels Flip the Sign of δ

| Pipeline | D | δ | B/A |
|----------|---|---|-----|
| Simplified | -0.73 | -0.72 | 4.28 (undershoot) |
| Case C | +0.20 | +0.90 | 5.90 (overshoot) |

The Case C kernels significantly change the I₁₂ and I₃₄ values, flipping the sign of the deviation.

### 3.3 The "+5" is an Empirical Shim

The formula `m = exp(R) + 5` was discovered empirically to produce c ≈ target. It was **not** derived from first principles. The true PRZZ derivation should give D = 0 exactly.

From PRZZ TeX Lines 1502-1511, the mirror combination is:
```latex
(N^{αx+βy} - T^{-α-β}N^{-βx-αy})/(α+β)
= N^{αx+βy} log(N^{x+y}T) ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

This difference quotient representation should **cancel** the D term analytically.

---

## 4. Why D ≠ 0 in Our Implementation

### 4.1 Simplified Pipeline Issues

The simplified J1 pipeline has several approximations:
1. Uses polynomial products instead of Case C kernels
2. Uses Laurent approximation (1/R + γ)² instead of actual (ζ'/ζ)²
3. Doesn't implement the full PRZZ integral structure

### 4.2 Production Pipeline Issues

The production Case C pipeline is closer to PRZZ but:
1. The "+5" shim is empirical, not derived
2. The mirror assembly is done **after** evaluating I-terms, not analytically combined
3. Missing normalization factors or combinatorial terms may exist

---

## 5. Path Forward

### 5.1 Option A: Accept Empirical Shim

Keep the current `m = exp(R) + 5` formula since it produces c ≈ target with ~1.3% accuracy. Document that B/A ≠ 5 is expected with the empirical approach.

**Pros:** Simple, works, minimal code changes.
**Cons:** Doesn't match PRZZ exact formulas, harder to extend.

### 5.2 Option B: Derive True Mirror Formula

Re-derive the mirror combination using PRZZ Lines 1502-1511 difference quotient representation. This would analytically combine terms **before** evaluating integrals.

**Pros:** Would give D = 0 and B/A = 5 exactly.
**Cons:** Significant implementation effort, may reveal other issues.

### 5.3 Option C: Calibrate D to Zero

Add a normalization factor to make D → 0. This is another shim but explicitly targets B/A = 5.

**Pros:** Could achieve B/A = 5.
**Cons:** Ad-hoc, doesn't match PRZZ derivation.

---

## 6. Recommendation

**Short-term:** Accept Option A. The empirical `m = exp(R) + 5` formula works and achieves ~1.3% accuracy on c. The B/A ≠ 5 is documented as expected behavior with the current implementation.

**Long-term:** Consider Option B for a proper PRZZ extension. The difference quotient representation (PRZZ Lines 1502-1511) is the key to achieving D = 0 analytically.

---

## 7. Test Status

The Phase 20.2 gate tests remain xfail because:
- Main-only B/A is 4.28 (simplified) or 5.90 (Case C), not 5
- J₁,₅ exclusion is correctly implemented
- The gap is in D, not in J₁,₅ handling

```bash
pytest tests/test_plus5_main_only_gate.py -v
# All xfail tests fail as expected
```

---

## References

- PRZZ TeX Lines 1502-1511: Difference quotient → integral representation
- TRUTH_SPEC.md Section 5: Mirror Combination Identity
- J15_VS_I5_RECONCILIATION.md: Confirms J15 = I5 (same object)
- Phase 19 Summary: Original "+5" discovery

---

*Generated 2025-12-24 as part of Phase 20.2 implementation.*
