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

## Update 2025-12-16: (1-u) Power Verification

### (1-u) Powers Are Correct

Verified all (1-u) powers against PRZZ TeX:

| Term | PRZZ Pattern | Our Power | Pair (1,1) | Pair (2,2) | Pair (1,2) |
|------|--------------|-----------|------------|------------|------------|
| I₁ | (1-u)^{ℓ₁+ℓ₂} | ✓ | (1-u)² | (1-u)⁴ | (1-u)³ |
| I₂ | none | ✓ | none | none | none |
| I₃ | (1-u)^{ℓ₁} | ✓ | (1-u)¹ | (1-u)² | (1-u)¹ |
| I₄ | (1-u)^{ℓ₂} | ✓ | (1-u)¹ | (1-u)² | (1-u)² |

**Source:** PRZZ TeX lines 1435, 1503, 1562, 1568

### I₂ Has No (1-u) Factor (Confirmed)

PRZZ line 1548:
```
I₂ = T·Φ̂(0)/θ × ∫∫ Q(t)² e^{2Rt} P₁(u)P₂(u) dt du
```

No (1-u) factor — matches our implementation exactly.

### PRZZ Notation Clarification (Important!)

**Line 1396-1400:** In Section 6.2.1, PRZZ writes:
> "For brevity, we will write P₁(u) and P₂(u) for P_{1,ℓ₁}(u) and P_{1,ℓ₂}(u)."

This means:
- "P₁" in 6.2.1 = **left polynomial** (generic)
- "P₂" in 6.2.1 = **right polynomial** (generic)

For ℓ₁=ℓ₂=1 (our (1,1) pair): both are our actual P₁ polynomial.

### Line 1726 Key Quote

> "It is now clear that the process to achieve the main terms from here is
> **the same as in the case ℓ₁ = ℓ₂ = 1**, only that there are lot more of them."

This suggests PRZZ uses the **same simplified structure** for all pairs, not Case C kernels.

### The Deepened Paradox

If PRZZ's numerical computation uses simplified formulas (P directly, like us):
- We should get c ≈ 2.14
- We get c ≈ 1.95 (-9% gap)

If PRZZ requires Case C kernels:
- We should get c ≈ 0.57 (-73% gap)

**Neither interpretation matches!**

### Critical Observation: κ Implications

Our computed c implies:
```
c_ours = 1.95  →  κ_ours = 1 - log(1.95)/1.3036 = 0.488
c_PRZZ = 2.14  →  κ_PRZZ = 1 - log(2.14)/1.3036 = 0.417
```

**Our c < PRZZ c, but κ = 1 - log(c)/R means smaller c → HIGHER κ**

This is logically impossible if we computed the same object:
- PRZZ optimized polynomials to maximize κ
- If our formula gave κ = 0.488, PRZZ would have found it
- Our κ = 0.488 would be a major mathematical breakthrough

**Conclusion: We're computing a fundamentally different mathematical object.**

### Possible Resolutions

1. **Hidden normalization in Φ̂(0)**: PRZZ may define Φ̂(0) differently
2. **Asymptotic limit not fully taken**: Our computation may be at different stage
3. **Cross-term contributions**: Mirror combination may generate terms we miss
4. **Numerical precision in PRZZ**: Their optimization found different polynomial coefficients than stated
5. **Our I₁+I₂+I₃+I₄ is NOT PRZZ's main constant**: Most likely explanation

## Key TeX References

| Lines | Content | Status |
|-------|---------|--------|
| 1396-1400 | P₁/P₂ notation (generic) | Clarified |
| 1502-1511 | Mirror combination | Validated |
| 1514-1517 | Q-operator substitution | Validated |
| 1529-1533 | I₁ formula structure | Validated |
| 1548 | I₂ formula (no (1-u)) | Validated |
| 1551-1570 | I₃/I₄ prefactor | Validated |
| 1726 | "Same process" for all pairs | Key quote |
| 2301-2310 | ω definition, x→x·log(N) | Needs trace |
| 2369-2384 | Case C F_d kernel | **NOT USED in 6.2.1?** |
| 2391-2396 | Euler-Maclaurin (1-u) | Validated |

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

## Update 2025-12-16: First-Principles Oracle Validation

### Term-Level Implementation is CORRECT

Built first-principles oracle for (1,1) pair (Case B × Case B):

| Term | First-Principles | DSL | Match? |
|------|-----------------|-----|--------|
| I1_11 | 0.4134741060 | 0.4134741024 | ✓ (0.0000% diff) |
| I2_11 | 0.3846294634 | 0.3846294634 | ✓ |
| I3_11 | -0.1780863845 | -0.1780863845 | ✓ |
| I4_11 | -0.1780863845 | -0.1780863845 | ✓ |

Verified components:
- Q(arg_α) × Q(arg_β) arguments are correct
- exp(R×arg_α) × exp(R×arg_β) factors are correct
- Derivative extraction via series expansion is accurate
- Algebraic prefactor cross-terms (F_x, F_y from (θS+1)/θ) are included

### The κ Paradox (Definitive)

| Metric | Our Computation | PRZZ Target |
|--------|-----------------|-------------|
| c (R=1.3036) | 1.950 | 2.137 |
| Implied κ | 0.488 | 0.417 |

**Critical insight:** Our c < PRZZ c implies our κ > PRZZ κ.

If PRZZ optimized polynomials to MAXIMIZE κ and got 0.417, but our formula
with the SAME polynomials gives 0.488, we are computing a fundamentally
different mathematical object.

This is NOT a normalization issue - it's a structural mismatch.

### Per-Pair Analysis at Both Benchmarks

| Pair | R=1.3036 | R=1.1167 | Change |
|------|----------|----------|--------|
| (1,1) | +0.442 | +0.390 | -12% |
| (2,2) | +1.261 | +1.086 | -14% |
| (3,3) | +0.080 | +0.072 | -10% |
| (1,2) | -0.201 | -0.224 | +11% |
| (1,3) | -0.218 | -0.196 | -10% |
| (2,3) | +0.586 | +0.514 | -12% |

**All pairs show R-dependent gaps**, not just Case C pairs.
This rules out Case C auxiliary integral as the sole explanation.

### PRZZ Line 2566 Key Quote

> "we had to rely on our code of the main terms (which matched Feng's)"

This suggests PRZZ used Feng's specific numerical implementation.
Our derivation from first principles may be computing a different
stage of the asymptotic expansion.

### Remaining Hypotheses

1. **Stage mismatch**: PRZZ evaluates at a different point in the
   asymptotic expansion (before some limit is taken).

2. **Log factor absorption**: The (log N)^ω factors from Case C may
   partially cancel with something we're not including.

3. **Feng's numerical procedure**: There may be additional terms or
   normalization in Feng's original code that we don't have access to.

4. **Mirror combination timing**: PRZZ does analytic combination
   BEFORE extracting constants. Our separate evaluation may miss
   subtle cross-terms.
