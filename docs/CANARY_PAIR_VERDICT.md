# Canary Pair Verdict: Do Same-(a,b) Monomials Need Different Kernels?

## Answer: **NO** - C_α/C_β are coefficient factors, not kernel modifiers

Based on PRZZ TeX analysis (lines 2275-2400), **case selection (A/B/C)** is determined by **ω(d,l)**. C_α/C_β do not by themselves select a different kernel family; within a case, prefactors (e.g. 𝒲(d,l)) and the specific polynomial P_{d,l} can still depend on the full l-vector.

---

## 1. PRZZ Kernel Classification (TeX 2301-2384)

### ω Determines the Kernel

For piece ℓ with d=1:
```
ω = ℓ - 1
```

- **Case B** (ω=0): ℓ=1 → P(u) evaluated directly
- **Case C** (ω>0): ℓ≥2 → Auxiliary integral K_ω(u; R)

### For K=3 Pieces

| Piece | ω | Kernel |
|-------|---|--------|
| P₁ | 0 | Case B: P₁(u) |
| P₂ | 1 | Case C: K₁(u; R) |
| P₃ | 2 | Case C: K₂(u; R) |

### For Pairs (ℓ₁, ℓ₂)

The kernel for a pair depends on (ω₁, ω₂) = (ℓ₁-1, ℓ₂-1):

| Pair | (ω₁, ω₂) | Cases |
|------|----------|-------|
| (1,1) | (0,0) | B×B |
| (2,2) | (1,1) | C×C |
| (3,3) | (2,2) | C×C |

---

## 2. What About C_α and C_β?

### Source in TeX (Lines 2279-2283)

The pole factors come from:
```
1/ζ(1+α+s) = (α+s)(1 + O(α+s))
```

These are **residue coefficients** from the contour integral around the pole at s=-α (or s=-β).

### Key Insight

C_α and C_β are NOT part of the l-vector (ζ derivative powers). They're:
- **Multiplicative factors** from pole residues
- **Don't change ω** (which determines Case A/B/C)
- **Don't require different kernel evaluation**

---

## 3. Canary Pair Analysis

### CANARY 1: D² vs C_α²×C_β² (both have a=b=0)

| Monomial | Full Key | Meaning |
|----------|----------|---------|
| D² | (0,0,0,0,2) | Two (ζ'/ζ)' paired blocks |
| C_α²×C_β² | (0,0,2,2,0) | Pure pole factors |

**Do they use the same kernel?**

For (2,2) pair with (ω₁, ω₂) = (1, 1):
- D² contributes to the base F(x,y) integrand
- C_α²×C_β² are coefficient factors from pole residues

**Verdict**: They use the SAME kernel (F_d Case C×C), but with different coefficient structures. The C factors multiply the integral result.

### CANARY 2: B×C_β×D vs B×C_α×C_β² (both have a=0, b=1)

| Monomial | Full Key | Meaning |
|----------|----------|---------|
| B×C_β×D | (0,1,0,1,1) | B singleton + pole + paired |
| B×C_α×C_β² | (0,1,1,2,0) | B singleton + poles only |

**Verdict**: Same kernel - B contributes a y-derivative (b=1), and C factors are coefficients.

---

## 4. Implications for Implementation

### Current Structure is Correct (for kernel selection)

The lookup `integral_grid[(a, b, weight_exp)]` is correct because:
- (a, b) determines the derivative structure
- All monomials with same (a, b) use the same integrand kernel
- C_α, C_β, d contribute through coefficients, not kernel choice

### The Ratio Error is NOT from Monomial Collapsing

Since same-(a,b) monomials correctly share integrals, the 80% ratio error must come from:

1. **Case C auxiliary integral** - We're not using K_ω(u; R) for ℓ≥2 pieces
2. **Normalization factors** - PRZZ may have degree-dependent factors we're missing
3. **Something else entirely** - R-dependent scaling issue

---

## 5. Updated Diagnosis

### What's Correct
- Coefficient extraction via series engine ✓
- Monomial expansion with full (a,b,c_α,c_β,d) structure ✓
- Lookup by (a, b, weight_exp) ✓ (per this analysis)

### What's Missing

**The kernel itself is wrong for ℓ≥2 pieces.**

For (2,2) pair, we're using:
```
F(x,y) = P₂(u-x) × P₂(u-y) × Q(α) × Q(β) × exp(R(α+β))
```

But PRZZ says for ω=1 (P₂), we should use:
```
K₁(u; R) = u × ∫₀¹ P₂((1-a)u) × exp(Rθua) da
```

This is the **Case C auxiliary integral** that replaces P(u).

---

## 6. Recommended Action

### Implement Case C Kernel for ℓ≥2 Pieces

Instead of searching for monomial-specific kernels, implement the PRZZ Case C structure:

1. For P₁ (ω=0): Keep using P₁(u) directly
2. For P₂ (ω=1): Replace with K₁(u; R)
3. For P₃ (ω=2): Replace with K₂(u; R)

The series engine can still extract coefficients, but the base polynomials need to be replaced with their Case C kernels before derivative extraction.

---

## 7. Conclusion

**The canary pairs DO share the same kernel** because:
- ω is determined by piece index ℓ, not by (c_α, c_β, d)
- C_α/C_β are coefficient factors from pole residues
- The current (a, b, weight_exp) lookup is structurally correct

**The ratio error comes from using raw P(u) instead of Case C kernels K_ω(u; R) for ℓ≥2.**

---

Date: Session 12
