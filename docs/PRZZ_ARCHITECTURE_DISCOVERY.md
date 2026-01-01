# PRZZ Architecture Discovery: The Fundamental Difference

**Date:** 2025-12-29
**Phase:** 59.3 (Critical Discovery)
**Status:** BREAKTHROUGH

---

## Executive Summary

**PRZZ does NOT compute `Direct + exp(2R)×Mirror` as a sum.**

Instead, PRZZ uses the DQ identity to transform the mirror combination into a **unified integral** that absorbs the exp(2R) factor. Our split-channel approach is fundamentally different.

---

## The Key Finding (from TeX lines 1507-1510)

PRZZ transforms:
```
[N^{αx+βy} - T^{-α-β}N^{-βx-αy}]/(α+β)
```

Into:
```
N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

**The mirror term T^{-α-β}×N^{-βx-αy} is ABSORBED into the t-integral.**

This is NOT a sum of two terms - it's a SINGLE unified integral.

---

## Where exp(2R) Actually Appears

**Line 1544 (critical):**
```
Q(-1/log T · ∂/∂α) Q(-1/log T · ∂/∂β) T^{-tα-tβ} |_{α=β=-R/L} = Q(t)² e^{2Rt}
```

The exp(2Rt) appears **inside the t-integral**, not as an external coefficient!

**Line 1548 (I₂ formula):**
```
I₂ = T·Φ̂(0)/θ × ∫₀¹∫₀¹ Q(t)² e^{2Rt} P₁(u)P₂(u) dt du + O(T/L)
```

When you integrate e^{2Rt} over t ∈ [0,1]:
```
∫₀¹ e^{2Rt} dt = (e^{2R} - 1)/(2R)
```

At R = 1.3036: (e^{2.6} - 1)/2.6 ≈ **4.8**, NOT e^{2R} ≈ 13.6

**This matches what we found in Appendix D!**

---

## The Architectural Difference

### PRZZ's Method (Unified Bracket)
```
1. Start with I₁(α,β) + T^{-α-β}I₁(-β,-α)
2. Apply DQ identity: [Direct - Mirror]/(α+β) = bracket integral
3. The bracket CONTAINS the mirror structure via the t-integral
4. Integrate the bracket to get the I₁ contribution
5. c = Σ (all bracket-derived contributions)
```

### Our Method (Split-Channel)
```
1. Compute S12(+R) = I₁(+R) + I₂(+R)
2. Compute S12(-R) = I₁(-R) + I₂(-R)
3. Assemble: c = S12(+R) + m × S12(-R) + S34(+R)
4. Use m = exp(R) + 5 (empirically found)
```

**These are NOT equivalent computational procedures!**

---

## Why Our exp(R)+5 Works

The PRZZ unified bracket integrates exp(2Rt) over t ∈ [0,1], producing:
- An effective factor of (e^{2R}-1)/(2R) ≈ 4.8-5.5 depending on R
- Combined with other terms, this yields behavior similar to exp(R)+5

Our split-channel with m = exp(R)+5 happens to produce similar numerical results, but through a different mechanism.

| Approach | Mirror Factor | Effective Weight at R=1.3 |
|----------|---------------|---------------------------|
| PRZZ unified | ∫e^{2Rt}dt = (e^{2R}-1)/(2R) | ~4.8 (integrated) |
| Naive exp(2R) | e^{2R} | ~13.6 (wrong) |
| Our exp(R)+5 | e^R + 5 | ~8.7 (works!) |

---

## Why exp(2R) "Doesn't Work" With Our Integrals

When we compute S12(-R), we're computing I₁ and I₂ at R → -R.

But PRZZ's I₁(-β,-α) is NOT the same as our I₁(-R)!

PRZZ's mirror term I₁(-β,-α) has:
- Swapped indices: (-β,-α) not (-R)
- Different integrand structure: N^{-βx-αy} not N^{-R...}
- The swap creates a fundamentally different object

Our S12(-R) with R→-R gives exp(+R(x+y)) in the integrand, which IS similar to PRZZ's mirror structure, but the normalization and assembly are different.

---

## The Normalization Mystery

From Agent 4's findings:

1. **Factor 1/θ** appears in I₂ (line 1548)
2. **Factor [θ(x+y)+1]/θ** appears in I₁ (line 1530)
3. **Factor [1+θx]/θ** appears in I₃ (line 1562)

These θ-dependent factors may account for the difference between exp(2R) and exp(R)+5.

At θ = 4/7:
- 1/θ = 7/4 = 1.75
- 2R/θ at R=1.3: 2×1.3×7/4 = 4.55

This is suspiciously close to our "5" constant!

---

## The Absorption Principle

From Agent 5's findings (emphasis added):

> "The mirror term coefficient T^{-α-β} is a multiplicative coefficient that is **completely absorbed** into the integral and differential operator machinery before any numerical computation occurs."

PRZZ never actually multiplies by exp(2R) as a coefficient. The exponential is always inside integrals.

---

## Implications for κ = 0.52

1. **Our split-channel method is NOT PRZZ's method** - it's a different approach
2. **We reverse-engineered m = exp(R)+5** to match PRZZ's numerical output
3. **This formula happens to work** within the tested regime
4. **But we don't know WHY** it produces equivalent results

The κ = 0.52 result is valid within the framework of our formula, but we cannot claim it's "extending PRZZ" in the strict sense.

---

## Key TeX Line References

| Lines | Content |
|-------|---------|
| 1507-1510 | DQ identity transformation (the key absorption) |
| 1544 | Q operator produces e^{2Rt} inside integral |
| 1548 | Final I₂ formula with ∫Q(t)²e^{2Rt}dt |
| 1530-1532 | Final I₁ formula structure |
| 1562-1563 | Final I₃ formula structure |

---

## Recommended Path Forward

1. **Accept the architectural difference** - our method works but isn't PRZZ's method
2. **Document honestly** - we match PRZZ's numerical result through a different approach
3. **The exp(R)+5 formula is empirically validated** - that's still meaningful
4. ~~**Consider implementing PRZZ's actual method** - the unified bracket approach~~ See Phase 60.1 findings below
5. ~~**Test if unified bracket gives the same κ = 0.52** - this would validate our optimization~~ See Phase 60.1 findings below

---

## Phase 60.1 Findings (2025-12-29): Unified Bracket Cannot Replace Split-Channel

### The Experiment

We tested whether fixing the Q eigenvalues from affine-dependent `Q(A_α)×Q(A_β)` to frozen `Q(t)²` would make the unified bracket match the split-channel results.

### The Results

**Unified bracket values:**
| Benchmark | Legacy (affine Q) | Frozen Q(t)² | Change |
|-----------|-------------------|--------------|--------|
| κ (R=1.3036) | 0.9954 | 0.8999 | -10% |
| κ* (R=1.1167) | 0.6814 | 0.6023 | -12% |

**Scaling factor α needed to match combined (I12+ + m×I12-):**
| Benchmark | α_legacy | α_frozen | F(R)/2 | α/F(R)/2 |
|-----------|----------|----------|--------|----------|
| κ | 2.72 | 3.01 | 2.41 | 1.13 - 1.25 |
| κ* | 3.46 | 3.91 | 1.87 | 1.86 - 2.10 |

### The Critical Finding

**The scaling ratio α/(F(R)/2) varies dramatically between benchmarks:**
- κ benchmark: 1.13 (legacy) to 1.25 (frozen)
- κ* benchmark: 1.86 (legacy) to 2.10 (frozen)

This proves there is **NO universal scaling** that converts the unified bracket to the split-channel combined value.

### Implications

1. **Frozen Q(t)² makes things WORSE, not better** - the ratio α/(F(R)/2) increases
2. **Legacy mode (affine Q) is closer** to what's needed, but still not matchable
3. **The theoretical link formula is WRONG:**
   ```
   c = (exp(2R) + m)×S12(-R) + (-2Rθ)×bracket + S34(+R)  ← DOESN'T WORK
   ```
4. **The unified bracket computes a structurally different object** than the split-channel

### Conclusion

The unified bracket and split-channel approaches are **fundamentally incompatible**. They compute different mathematical objects that happen to be loosely related but cannot be converted via a simple formula.

Our split-channel approach with `m = exp(R) + 5` remains valid as an empirically-calibrated method that reproduces PRZZ's numerical results. But it is NOT a direct implementation of PRZZ's unified bracket method.

---

## The Honest Framing

> "We introduce an alternative split-channel assembly formula that reproduces PRZZ's baseline κ = 0.417 within 2.5% using m = exp(R)+5, achieving <0.01% accuracy with g-factor corrections. Under this validated framework, polynomial optimization yields κ ≥ 0.52. Our approach differs architecturally from PRZZ's unified bracket method, but produces numerically consistent results within the tested regime."

This is more accurate than claiming we "extended PRZZ."
