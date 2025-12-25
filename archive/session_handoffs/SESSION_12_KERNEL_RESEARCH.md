# Session 12: Kernel Structure Research

## Executive Summary

**Problem**: Current implementation assigns the same integral to monomials with identical (a,b) but different (c_alpha, c_beta, d) structure. This collapses distinct monomial types.

**Finding**: The issue is NOT simply "more dict keys" - the **integrand kernel itself** may differ for different monomial types.

---

## 1. Current Implementation Structure

### SeriesBackedEvaluator (Phase A - Complete)

All I-terms use `integral_grid[(a, b, weight_exp)]`:
- I1 → g[1,1] with weight (1-u)^{ell+ellbar}
- I2 → g[0,0] with weight 0
- I3 → -g[1,0] with weight (1-u)^ell
- I4 → -g[0,1] with weight (1-u)^ellbar

This is mathematically correct for the kernel F(x,y) = P_L(u-x) × P_R(u-y) × Q(α) × Q(β) × exp(R(α+β)).

### The Monomial Collapsing Problem

For pair (2,2), there are 12 monomials with structure (a, b, c_alpha, c_beta, d):

| Monomial Type | (a,b,c_α,c_β,d) | Weight | Current Key |
|---------------|-----------------|--------|-------------|
| A²×B² | (2,2,0,0,0) | (1-u)^4 | (2,2,4) |
| A×B×C_α | (1,1,1,0,0) | (1-u)^2 | (1,1,2) |
| A×B×C_β | (1,1,0,1,0) | (1-u)^2 | (1,1,2) ← **Same!** |
| C_α×C_β | (0,0,1,1,0) | (1-u)^0 | (0,0,0) |
| D | (0,0,0,0,1) | (1-u)^0 | (0,0,0) ← **Same!** |
| ... | ... | ... | ... |

**Problem**: Monomials like `A×B×C_α` and `A×B×C_β` get the SAME integral, but they may represent different pole contributions.

---

## 2. PRZZ Block Definitions (from psi_expansion.py)

### Singleton Blocks (A, B)
- **A** = ζ'/ζ(1+α+s) with z-derivative (x-side)
- **B** = ζ'/ζ(1+β+u) with w-derivative (y-side)

These enter the integrand as P_L and P_R polynomial evaluations.

### Pole Contributions (C_alpha, C_beta)
- **C_α** = contribution from 1/ζ(1+α+s) pole
- **C_β** = contribution from 1/ζ(1+β+u) pole

**Key Question**: Do these require separate integrals, or are they coefficients?

### Paired Block (D)
- **D** = (ζ'/ζ)'(1+s+u) = mixed z,w derivative

**Key Question**: How does D differ from the A×B structure in the integrand?

---

## 3. Case C Kernel Structure (PRZZ TeX 2370-2384)

### When Case C Applies
- **Case B** (ω=0): ℓ=1 → P₁(u) evaluated directly
- **Case C** (ω>0): ℓ≥2 → Auxiliary integral K_ω(u; R)

### K_ω Formula
```
K_ω(u; R) = [u^ω / (ω-1)!] × ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθua) da
```

For K=3:
- P₁: ω=0 → Case B (direct)
- P₂: ω=1 → K₁(u; R)
- P₃: ω=2 → K₂(u; R)

### Numerical Effect (from CASE_C_ANALYSIS.md)
- Case C kernels are 50-75% SMALLER than raw polynomials
- Applying Case C makes c DECREASE, not increase
- This cannot explain the gap alone

---

## 4. Key Research Questions

### Q1: What integral does C_α × C_β use?

**Hypothesis A**: C_α and C_β are just multiplicative factors.
- They don't change the kernel
- The coefficient accounts for their contribution
- Current implementation would be correct

**Hypothesis B**: C_α and C_β require pole residue integrals.
- Pole contributions have different integrand structure
- May involve S(z) zeta ratio factors (PRZZ Section 7)
- Would need separate kernel evaluation

### Q2: How does D differ from A×B?

**Current understanding**:
- D is (ζ'/ζ)' = second derivative
- A×B involves ζ'/ζ × ζ'/ζ

**In terms of integrand**:
- A×B → f[1,1] from series expansion (mixed x,y derivative)
- D → f[0,0] base term (no derivatives)

**This is backwards!** D should be the derivative structure, but we're treating it as the base.

### Q3: Is the current I-term mapping correct?

From PRZZ/GenEval convention:
- I₁: mixed derivative d²/dxdy
- I₂: base term (no derivatives)
- I₃: x-derivative d/dx
- I₄: y-derivative d/dy

From monomial structure:
- A×B (a=1, b=1) → I₁ mixed derivative ✓
- D (d=1, a=b=0) → I₂ base term ✓
- A×C (a=1, c>0) → I₃ x-derivative (C is pole factor) ?
- B×C (b=1, c>0) → I₄ y-derivative (C is pole factor) ?

The mapping seems consistent, but C is being treated as a coefficient rather than changing the kernel.

---

## 5. What the Plan Calls For

### Phase B: MonomialKernelEvaluator

Per the Session 12 plan:
```python
def select_kernel(self, mono):
    """Select kernel class based on monomial structure."""
    if mono.d > 0 and mono.a == 0 and mono.b == 0:
        return "D-block"  # Base integral, no derivatives
    elif mono.c_alpha > 0 or mono.c_beta > 0:
        return "C-type"   # Pole contribution kernel (Case C?)
    else:
        return "AB-block" # Singleton derivative kernel
```

### The Key Insight

> "Keeping more indices in integral_grid doesn't fix anything unless the *kernel* depends on those exponents."

If C_alpha and C_beta are just coefficients (Hypothesis A), then:
- Current implementation is mathematically correct
- The ratio error must come from something else (Case C, normalization, etc.)

If C_alpha and C_beta change the kernel (Hypothesis B), then:
- We need to identify what integral formula they use
- This would require significant restructuring

---

## 6. Evidence for Hypothesis A (Coefficients Only)

### From psi_expansion.py
The weight formula is:
```
weight_exponent = a + b  (NOT dependent on c_alpha, c_beta, d)
```

This suggests C_alpha, C_beta, and D are treated as coefficients that multiply the (a,b)-dependent integral.

### From PRZZ structure
The pole contributions -C_α and -C_β subtract from the singleton blocks:
- X = A - C_β
- Y = B - C_α

This is a **linear combination**, not a different integral structure.

### From (1,1) validation
The (1,1) oracle match works because:
- +1×AB → I₁
- +1×D → I₂
- -1×AC_α → I₃ (A derivative with C_α coefficient)
- -1×BC_β → I₄ (B derivative with C_β coefficient)

The signs are handled by coefficients, not different integrals.

---

## 7. Evidence Against Hypothesis A

### The Ratio Mystery
If the current structure is correct, why does the ratio error persist at 80%?

Possibilities:
1. Case C auxiliary integral (K_ω) is missing for ℓ≥2
2. Polynomial normalization factors
3. Something else entirely

### Session 11 Finding
> "The failure is dominated by 'structural assembly' effects, not local derivative algebra."

But if C_alpha/C_beta are just coefficients, where does the structural assembly error come from?

---

## 8. Recommended Path Forward

### Option 1: Accept Hypothesis A and Focus on Case C
- Assume current (a,b,weight) structure is correct for coefficient extraction
- Implement Case C kernel K_ω for ℓ≥2 pieces
- See if this fixes the ratio error

**Risk**: Case C analysis shows it makes c SMALLER, wrong direction.

### Option 2: Build Full Section 7 Oracle
- Implement PRZZ's exact formula from TeX
- Bypass DSL assumptions entirely
- Compare to identify structural differences

**Cost**: Significant implementation effort.

### Option 3: Trace (2,2) Monomial-by-Monomial
- For each of 12 monomials, compute its contribution
- Verify each uses the correct kernel
- Look for the source of ratio error

**Benefit**: Most diagnostic value.

---

## 9. Conclusion

The kernel structure research reveals that:

1. **Current implementation treats C_alpha/C_beta as coefficients** - this may be correct
2. **Case C (K_ω auxiliary integral)** is a separate concern from C_alpha/C_beta structure
3. **The ratio error source remains unclear** - neither C monomial structure nor Case C alone explains it

**Next step**: Implement Option 3 (trace (2,2) monomial-by-monomial) to understand where the ratio error originates.

---

Date: Session 12
