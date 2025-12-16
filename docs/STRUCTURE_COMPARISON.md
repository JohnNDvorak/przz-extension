# Structure Comparison: Our Implementation vs PRZZ

**Date:** 2025-12-16
**Status:** Analysis complete — fundamental structural mismatch identified

## Executive Summary

Our I1+I2+I3+I4 implementation computes a **different mathematical object** than PRZZ.
The gap is NOT from a missing factor or incorrect prefactor, but from how polynomials
enter the integrand.

| Component | Our Implementation | PRZZ |
|-----------|-------------------|------|
| P₁ (ω=0, Case B) | P₁(u) directly | P₁(u) directly |
| P₂ (ω=1, Case C) | P₂(u) directly | F_d kernel with a-integral |
| P₃ (ω=2, Case C) | P₃(u) directly | F_d kernel with a-integral |
| Result | c = 1.95 | c = 2.14 |
| With Case C kernel | c = 0.57 | — |

## What We Validated as Correct

### 1. Q-Operator Substitution (TeX 1514-1517)

**Status: VALIDATED**

Q-operator oracle confirms:
- Q arguments match PRZZ exactly
- arg_α = t + θt·x + θ(t-1)·y
- arg_β = t + θ(t-1)·x + θt·y
- Symmetric under x↔y swap
- R-independent (R only enters exp factors)

### 2. Mirror Combination Structure (TeX 1502-1511)

**Status: VALIDATED**

The mirror combination gives:
```
(N^{αx+βy} - T^{-α-β}N^{-βx-αy})/(α+β)
= N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

Our implementation includes:
- log(N^{x+y}T) = [1+θ(x+y)] × log(T) via algebraic prefactor (θS+1)/θ
- The t-integral via quadrature over [0,1]²
- Proper derivative extraction

### 3. I3/I4 Prefactor (TeX 1551-1570)

**Status: VALIDATED**

PRZZ formula: I₃ = -[(1+θx)/θ] × d/dx[...] at x=0

At x=0: (1+θx)/θ = 1/θ

Our implementation: `numeric_prefactor = -1.0/theta`

FD oracle validated this is correct.

### 4. Derivative Extraction

**Status: VALIDATED**

Multi-variable Taylor series extraction correctly computes:
- ∂²/∂x∂y for I1 (both derivatives)
- constant term for I2 (no derivatives)
- ∂/∂x for I3 (x-only derivatives)
- ∂/∂y for I4 (y-only derivatives)

## What Differs: Case C Structure

### PRZZ Case C Definition (TeX 2369-2384)

For ω > 0 (P₂, P₃), PRZZ uses kernel F_d, not raw polynomial P:

```
F_d(k,α,n) = u^ω / (ω-1)! × ∫₀¹ P((1-a)u) × a^{ω-1} × (N/n)^{-αa} da
```

At α = -R/L with (N/n)^{-αa} → exp(Rθua):

```
K_ω(u; R) = u^ω / (ω-1)! × ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθua) da
```

This kernel is **smaller** than raw P(u):
- K₁ / P₂ ≈ 0.47 at R=1.3036
- K₂ / P₃ ≈ 0.24 at R=1.3036

### Our Implementation

We use raw polynomial P(u) for all polynomials:

```python
PolyFactor("P2", P_arg)  # Evaluates P₂(argument)
```

This is **incorrect** for Case C (ω > 0) pairs.

### Why Naive Case C Correction Fails

When we replace P with K in our I1-I4 structure:

| Metric | Raw | With Case C K | Target |
|--------|-----|---------------|--------|
| c(R₁) | 1.95 | 0.57 | 2.14 |
| R-sensitivity | 18.7% | 14.8% | 10.3% |

The Case C kernel:
1. **Reduces c by 70%** (wrong direction)
2. **Improves R-sensitivity** (right direction)

This proves Case C alone cannot explain the gap.

### The Paradox

- We need c to INCREASE from 1.95 to 2.14 (+10%)
- Case C makes c DECREASE from 1.95 to 0.57 (-70%)
- PRZZ uses Case C and gets c = 2.14

**Resolution:** PRZZ's c = 2.14 must come from a different computation structure, not I1+I2+I3+I4 with Case C kernels.

## Per-Pair Structure Comparison

### (1,1) — Case B × Case B

| Aspect | Our Implementation | PRZZ Expected | Match? |
|--------|-------------------|---------------|--------|
| P factors | P₁(x+u) × P₁(y+u) | P₁(x+u) × P₁(y+u) | ✓ |
| Q factors | Q(arg_α) × Q(arg_β) | Q(arg_α) × Q(arg_β) | ✓ |
| Exp factors | exp(R·arg_α) × exp(R·arg_β) | exp(R·arg_α) × exp(R·arg_β) | ✓ |
| poly_prefactor | (1-u)² | (1-u)² | ✓ |
| Case C | N/A (ω=0) | N/A (ω=0) | ✓ |

### (1,2) — Case B × Case C

| Aspect | Our Implementation | PRZZ Expected | Match? |
|--------|-------------------|---------------|--------|
| Left P | P₁(x+u) | P₁(x+u) | ✓ |
| Right P | P₂(y+u) | K₁(y+u; R) with a-integral | **✗** |
| poly_prefactor | (1-u) | ??? | ? |

### (2,2) — Case C × Case C

| Aspect | Our Implementation | PRZZ Expected | Match? |
|--------|-------------------|---------------|--------|
| Left P | P₂(x+u) | K₁(x+u; R) with a-integral | **✗** |
| Right P | P₂(y+u) | K₁(y+u; R) with a-integral | **✗** |
| poly_prefactor | (1-u)² | ??? | ? |

## Hypothesis: Missing Main Term Family

Given:
1. Raw computation: c = 1.95
2. Case C correction: c = 0.57
3. PRZZ target: c = 2.14

Something must add ~1.5 to c (beyond our I1+I2+I3+I4).

Possibilities:
1. **Additional term families** we haven't identified
2. **Different integrand assembly** in PRZZ
3. **Cross-terms from mirror combination** before constant extraction
4. **Our poly_prefactor powers** are wrong for Case C pairs

## Key TeX References

| Lines | Content | Status |
|-------|---------|--------|
| 1502-1511 | Mirror combination | Validated |
| 1514-1517 | Q-operator substitution | Validated |
| 1551-1570 | I₃/I₄ prefactor | Validated |
| 2301-2310 | ω definition, x→x·log(N) | Needs trace |
| 2369-2384 | Case C F_d kernel | **MISMATCH** |
| 2387-2388 | Cross-term bookkeeping | Needs trace |

## Next Steps

1. **Trace PRZZ's exact integrand definition**
   - What factors multiply the F_d × F_d product?
   - What is the analog of our poly_prefactor for Case C?

2. **Check if PRZZ has additional term families**
   - Are there terms beyond I1-I4 in the main constant?
   - What about cross-terms from mirror combination?

3. **Variable rescaling x → x·log(N)**
   - Does this affect the relative magnitudes?
   - Are our derivative counts correct after rescaling?

4. **Consider that our object ≠ PRZZ's object**
   - Maybe our I1+I2+I3+I4 structure is only valid for Case B×B
   - Case C pairs may need completely different formulation
