# Ψ Formula Implementation Plan

**Date**: 2025-12-17
**Status**: Key insights validated, implementation path clarified

---

## Summary of Discovery

The "I₁-I₄" structure from PRZZ Section 7 is **only valid for (ℓ,ℓ̄)=(1,1)**. For higher pairs, PRZZ uses the Ψ combinatorial expansion which produces:

| Pair | Monomials | Current DSL | Gap |
|------|-----------|-------------|-----|
| (1,1) | 4 | 4 | 0 |
| (2,2) | 12 | 4 | 8 missing |
| (3,3) | 27 | 4 | 23 missing |
| (1,2) | 7 | 4 | 3 missing |
| (1,3) | 10 | 4 | 6 missing |
| (2,3) | 18 | 4 | 14 missing |

**Total for K=3**: 78 monomials needed, DSL has 24 (4×6 pairs) = 69% missing

This explains the two-benchmark failure: the DSL computes ~31% of the required structure.

---

## The Ψ Formula

```
Ψ_{ℓ,ℓ̄}(A,B,C,D) = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × (D-C²)^p × (A-C)^{ℓ-p} × (B-C)^{ℓ̄-p}
```

Where the building blocks are:
- **A** = ∂/∂z [log ξ_P] at z=s₀
- **B** = ∂/∂w [log ξ_P] at w=s₀
- **C** = log ξ_P(s₀) (no derivative)
- **D** = ∂²/∂z∂w [log ξ_P] at z=w=s₀

Each monomial `(k₁, k₂, l₁, m₁)` with coefficient `coeff` represents:
- `C^k₁`: k₁ factors of base evaluation
- `D^k₂`: k₂ factors of mixed ∂²/∂z∂w
- `A^l₁`: l₁ factors of ∂/∂z
- `B^m₁`: m₁ factors of ∂/∂w

---

## Implementation Strategy

### Option A: Term-by-Term Expansion (Recommended)

For each pair (ℓ,ℓ̄):
1. Call `psi_d1_configs(ell, ellbar)` to get monomials
2. For each monomial (k₁,k₂,l₁,m₁) → coeff:
   - Create a `Term` with appropriate derivative structure
   - The total z-derivative order is `l₁ + k₂`
   - The total w-derivative order is `m₁ + k₂`
   - Include coefficient in `numeric_prefactor`

### Challenge: Mapping Powers to Derivatives

For (1,1), the mapping is straightforward:
- A¹B¹ → ∂²/∂z∂w (I₁ structure)
- A¹C⁰ → ∂/∂z (I₃ structure, multiplied by C⁰)
- B¹C⁰ → ∂/∂w (I₄ structure, multiplied by C⁰)
- D¹ → the "base" term with no derivatives (I₂)

For (2,2), we have terms like:
- A²B² → ∂⁴/∂z²∂w² (4th order total)
- A¹B²C¹ → ∂³/∂z∂w² with one C factor
- D² → two mixed derivatives multiplied together

**Key insight from GPT**: The powers in the Ψ expansion correspond to derivative orders. For example:
- `A^2` means we need the coefficient of x² in the Taylor expansion of log ξ(s₀+x)
- This is ½ × d²/dx² [log ξ]|_{x=0}

### Implementation Steps

1. **Extend `Term` to support variable derivative orders**
   - Currently: `deriv_orders = {"x": 1, "y": 1}`
   - Needed: `deriv_orders = {"x": 2, "y": 3}` for higher powers

2. **Create a Ψ-to-Term translator**
   ```python
   def psi_monomial_to_term(
       ell: int, ellbar: int,
       k1: int, k2: int, l1: int, m1: int,
       coeff: int,
       theta: float, R: float
   ) -> Term:
       """Convert a Ψ monomial to a Term."""
       # Total derivative orders
       x_deriv = l1 + k2  # A powers + D powers
       y_deriv = m1 + k2  # B powers + D powers

       # Number of "base" (C) factors to include
       c_power = k1

       # Build term with appropriate structure...
   ```

3. **Validate on (1,1)** - should produce same 4 terms as current DSL

4. **Extend to (2,2)** - should produce 12 terms

5. **Run two-benchmark test** - should show improved stability

---

## Alternative: Direct Section 7 Oracle

Instead of rewriting the DSL, implement PRZZ Section 7 directly:

1. Use Faà-di-Bruno for derivative structure
2. Use F_d factors as in PRZZ
3. Apply Euler-Maclaurin for n-sum → integral

This bypasses the Ψ expansion but requires understanding more of PRZZ's machinery.

---

## Questions for GPT/User

1. **Derivative interpretation**: For A², is this ½×∂²/∂z² or (∂/∂z)²?
2. **Product structure**: For D², is this the product of two separate ∂²/∂z∂w evaluations?
3. **C factor handling**: How does C^k₁ enter the integrand?

---

## Validation Plan

1. **Check (1,1) is unchanged**
   - Current DSL c₁₁ should match Ψ-based c₁₁

2. **Compare (2,2) with extended structure**
   - Should have 12 terms instead of 4
   - Check if ratio κ/κ* improves toward 1.10

3. **Full K=3 two-benchmark test**
   - Target: both benchmarks within 10% of PRZZ

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/psi_combinatorial.py` | EXISTS | Ψ expansion |
| `src/psi_term_builder.py` | CREATE | Translate Ψ → Terms |
| `src/terms_k3_psi.py` | CREATE | K=3 terms using Ψ |
| `tests/test_psi_terms.py` | CREATE | Validate Ψ → DSL |

---

## Key Numerical Validation Results

### (1,1) Comparison: Oracle vs Log-Derivative Integrals

| Quantity | Value | Notes |
|----------|-------|-------|
| Oracle I₁ | 0.426 | Mixed derivative contribution |
| Oracle I₂ | 0.385 | Base integral (no derivatives) |
| Oracle I₃ | -0.226 | z-derivative contribution |
| Oracle I₄ | -0.226 | w-derivative contribution |
| **Oracle Total** | **0.359** | Sum of I₁+I₂+I₃+I₄ |
| | | |
| ∫ F₀ du dt | 0.385 | Matches I₂ exactly ✓ |
| ∫ (D+AB)×F₀ du dt | 2.265 | Does NOT match I₁ |

### Key Insight: Structure vs Values

The Ψ formula provides **combinatorial structure**, not direct numerical mapping:

1. **Ψ = AB - AC - BC + D** tells us there are 4 monomial contributions
2. **I₁, I₂, I₃, I₄** are the 4 integral terms that capture these contributions
3. But I₁ ≠ ∫(AB×F₀), etc. - the mapping involves PRZZ's prefactor/derivative machinery

The validation confirms:
- I₂ = ∫F₀ (base integral, "D" piece) ✓
- I₃, I₄ < 0 (negative contributions, "-AC", "-BC" pieces) ✓
- The DSL 4-term structure captures Ψ_{1,1} combinatorics ✓

### What This Means for (2,2)

For (2,2), Ψ has 12 monomials organized as 3 p-configs:
- p=0: X²Y² = (A-C)²(B-C)² → needs **A², B²** structures
- p=1: 4ZXY = 4(D-C²)(A-C)(B-C) → needs **mixed** structures
- p=2: 2Z² = 2(D-C²)² → needs **D²** structure

The current DSL only has I₁-I₄, which capture:
- I₁: AB structure
- I₂: D structure (base)
- I₃: A structure (with -C subtraction)
- I₄: B structure (with -C subtraction)

Missing for (2,2): A², B², D², A²B, AB², mixed products with D, etc.

---

## GPT's Recommended Implementation Approach

### Don't Expand to 12 Monomials - Use 3 p-Configs

Instead of building 12 separate terms, build 3 evaluators:

```python
def eval_X(ell, u, t):
    """Connected singleton z-block = (A - C) for piece ℓ"""
    ...

def eval_Y(ellbar, u, t):
    """Connected singleton w-block = (B - C) for piece ℓ̄"""
    ...

def eval_Z(u, t):
    """Connected paired block = (D - C²)"""
    ...
```

Then for each (ℓ,ℓ̄) pair:
```python
total = 0
for p_config in psi_p_configs(ell, ellbar):
    contrib = p_config.coeff
    contrib *= eval_Z(u, t) ** p_config.z_exp
    contrib *= eval_X(ell, u, t) ** p_config.x_exp
    contrib *= eval_Y(ellbar, u, t) ** p_config.y_exp
    total += contrib
return integral(total × base_factors)
```

### The Challenge: What Are X, Y, Z Concretely?

GPT's guidance:
- A² = (∂L/∂z)² = **product of two singleton evaluations**
- D² = (∂²L/∂z∂w)² = **product of two paired evaluations**
- C enters only through (A-C), (B-C), (D-C²) - never directly

The exact implementation requires understanding PRZZ Section 7's F_d factors
and Euler-Maclaurin machinery.

---

## Risk Assessment

**High confidence**:
- Ψ formula produces correct monomial counts
- (1,1) maps correctly to I₁-I₄

**Medium confidence**:
- Derivative order interpretation is correct
- C factor handling is understood

**Low confidence**:
- Full integration will match PRZZ targets
- No other missing pieces in the pipeline

---

## Recommendation

**Proceed with Option A (Term-by-Term Expansion)** because:
1. Ψ formula is validated (4/12/27 counts match GPT's expectation)
2. Builds on existing DSL infrastructure
3. Incremental: can validate (1,1) first, then extend
4. If Ψ approach fails, falling back to direct Section 7 implementation is still possible

**Next immediate step**: Clarify with GPT how to map C^k₁ × D^k₂ × A^l₁ × B^m₁ to a concrete integral structure.
