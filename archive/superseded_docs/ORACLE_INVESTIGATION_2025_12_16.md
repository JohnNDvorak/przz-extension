# Oracle Investigation Findings (2025-12-16)

**Status:** Multi-variable structure identified as critical for valid F_d kernel comparison

---

## Summary

Attempted to build a first-principles oracle for the (2,2) pair to compare PRZZ's F_d kernel
structure against our DSL's direct P(u) evaluation. The investigation revealed that a simplified
oracle approach cannot capture the correct derivative structure.

---

## Key Finding: Multi-Variable Structure for ℓ > 1

The DSL uses **4 variables** for the (2,2) pair: `(x1, x2, y1, y2)`

This is because:
- ℓ = 2 corresponds to μ ⋆ Λ ⋆ Λ (two Λ-convolutions)
- Each Λ-convolution introduces a **separate** formal variable for residue extraction
- The polynomial argument is P(x₁ + x₂ + u), not P(x + u)

### DSL Term Structure (2,2):

| Term | Variables | Derivative Orders | Numeric Prefactor |
|------|-----------|-------------------|-------------------|
| I1_22 | (x1, x2, y1, y2) | ∂⁴/∂x₁∂x₂∂y₁∂y₂ | 1.0 |
| I2_22 | () | none | 1.75 (= 1/θ) |
| I3_22 | (x1, x2) | ∂²/∂x₁∂x₂ | -1.75 (= -1/θ) |
| I4_22 | (y1, y2) | ∂²/∂y₁∂y₂ | -1.75 (= -1/θ) |

### Oracle vs DSL Comparison (R = 1.3036)

| Term | DSL (correct) | Oracle (simplified) | Issue |
|------|---------------|---------------------|-------|
| I1 | 0.9710 | 0.1399 | Oracle uses ∂²/∂x∂y instead of ∂⁴/∂x₁∂x₂∂y₁∂y₂ |
| I2 | 0.2272 | 0.2272 | **MATCHES** (no derivatives) |
| I3 | +0.0315 | -0.0524 | Oracle uses ∂/∂x instead of ∂²/∂x₁∂x₂, wrong sign |
| I4 | +0.0315 | -0.0524 | Same issue as I3 |

The I2 term matches because it has no derivatives - only the polynomial values at x=y=0.

---

## Implications for Case C Investigation

### What the F_d Kernel Comparison Shows (with caveats)

| u = 0.5, R = 1.3036 | Value |
|---------------------|-------|
| F_d(P₂, u, ω=1) | 0.194 |
| P₂(u) | 0.737 |
| Ratio F_d / P | 0.26 |

The F_d kernel is ~26% of the raw polynomial value at u=0.5.

### Why Simple Oracle Can't Diagnose the Gap

1. **Structural mismatch**: Simplified (x, y) oracle cannot reproduce DSL's 4-variable structure
2. **Derivative coupling**: The way derivatives interact with F_d vs P is more complex with multiple variables
3. **Polynomial arguments**: DSL uses P(x₁ + x₂ + u), F_d would apply to a different structure

### What Would Be Needed

A proper F_d kernel comparison requires:
1. Understanding how F_d applies to multi-convolution structures
2. Implementing the full 4-variable derivative extraction with F_d kernels
3. Matching PRZZ's exact assembly at the α,β level before any simplifications

---

## The Fundamental Question

PRZZ line 1726 states:
> "It is now clear that the process to achieve the main terms from here is
> **the same as in the case ℓ₁ = ℓ₂ = 1**, only that there are lot more of them."

This suggests PRZZ uses the **same simplified structure** for all pairs, without Case C kernels.

But then why is there an R-dependent gap?

### Two Interpretations

1. **PRZZ numerical code differs from displayed formulas**
   - Line 2566: "we had to rely on our code of the main terms (which matched Feng's)"
   - Feng's code may have additional terms or different assembly

2. **We are at a different stage of the asymptotic expansion**
   - Our DSL evaluates at α = β = -R/L already substituted
   - PRZZ may combine terms analytically before this substitution

---

## Recommendations

### Short-term
1. **Don't pursue F_d oracle further** - the simplified structure cannot give valid comparison
2. **Focus on (1,1) pair** - this has only 2 variables (x, y) and we have validated it term-by-term

### Medium-term
3. **Investigate mirror combination** (PRZZ lines 1511-1527)
   - Check if analytic combination before α,β substitution produces different constants

4. **Look for Feng's code or formula derivation**
   - PRZZ references matching Feng's code specifically

### Long-term
5. **Consider that gap may be acceptable**
   - Our c = 1.95 gives κ = 0.488 > PRZZ κ = 0.417
   - If we're computing a valid lower bound on κ (just different), we're actually BETTER
   - The paradox is: PRZZ optimized to maximize κ, so how could we get higher κ with same polynomials?

---

## Files Created

| File | Purpose |
|------|---------|
| `src/przz_22_oracle.py` | Simplified oracle (educational, but doesn't match DSL) |
| `docs/ORACLE_INVESTIGATION_2025_12_16.md` | This document |

---

## Appendix: Variable Count by Pair

| Pair (ℓ₁, ℓ₂) | # x-vars | # y-vars | Total | I1 derivative order |
|---------------|----------|----------|-------|---------------------|
| (1,1) | 1 | 1 | 2 | ∂²/∂x∂y |
| (1,2) | 1 | 2 | 3 | ∂³/∂x∂y₁∂y₂ |
| (1,3) | 1 | 3 | 4 | ∂⁴/∂x∂y₁∂y₂∂y₃ |
| (2,2) | 2 | 2 | 4 | ∂⁴/∂x₁∂x₂∂y₁∂y₂ |
| (2,3) | 2 | 3 | 5 | ∂⁵/∂x₁∂x₂∂y₁∂y₂∂y₃ |
| (3,3) | 3 | 3 | 6 | ∂⁶/∂x₁∂x₂∂x₃∂y₁∂y₂∂y₃ |

The (1,1) pair is the only one where a simple (x, y) oracle can potentially match the DSL exactly.
