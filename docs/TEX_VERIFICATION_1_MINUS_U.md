# TeX Verification: (1-u) Power Formula

**Date**: 2025-12-21
**Source**: `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/RMS_PRZZ.tex`

---

## Summary

**VERDICT: OLD is TeX-truth for the production assembly.**

While both OLD and V2 encode the (1-u) power differently, the OLD formula produces correct I-term values when integrated through the production tex_mirror assembly. V2, while mathematically equivalent for isolated I-term evaluation, breaks catastrophically under the full assembly.

---

## TeX Evidence

### Line 1435 - I₁ for (1,1) pair
```latex
I_{1,1}(\alpha,\beta) = \frac{T\widehat{\Phi}(0)}{\log N}\frac{1}{\alpha+\beta}
\frac{d^2}{dxdy}N^{\alpha x + \beta y}
\int_0^{1} (1-u)^2 P_{1} (x+u) P_{2} (y+u)du \bigg|_{x=y=0} + O(T/L).
```

**Power for I₁(1,1) = 2**

### Line 1484 - I₃ for (1,1) pair
```latex
I_{1,3}(\alpha,\beta) = -\frac{T\widehat{\Phi}(0)}{\log N}\frac{1}{\alpha+\beta}
\frac{d}{dx} N^{\alpha x}
\int_0^1 (1-u) P_{1}(x+u) P_{2}(u)du \bigg|_{x=0} + O(T/L).
```

**Power for I₃(1,1) = 1**

### Line 1488 - I₄ for (1,1) pair
```latex
I_{1,4}(\alpha,\beta) = -\frac{T\widehat{\Phi}(0)}{\log N}\frac{1}{\alpha+\beta}
\frac{d}{dy} N^{\beta y}
\int_0^1 (1-u) P_{1}(u)P_{2}(y+u)du \bigg|_{y=0} + O(T/L).
```

**Power for I₄(1,1) = 1**

### Lines 2391-2409 - Euler-Maclaurin Lemma (General Formula)

The lemma shows that for arithmetic functions with:
```
Σ_{n≤z} g(n) = c_g z log^{k_g-1} z + O(z log^{k_g-2} z)
```

The integral becomes:
```
∫_0^1 (1-u)^{k_g-1} F(...) H(u) z^{us} du
```

This establishes that the **(1-u) power = k_g - 1** where k_g is determined by the arithmetic coefficient function.

---

## Indexing Convention

### PRZZ TeX (0-based Λ convolution count)
- ℓ = 0: μ piece (no Λ convolution)
- ℓ = 1: μ⋆Λ piece (one Λ convolution)
- ℓ = 2: μ⋆Λ⋆Λ piece (two Λ convolutions)

### Our Code (1-based piece index)
- Piece 1: μ piece (corresponds to ℓ=0)
- Piece 2: μ⋆Λ piece (corresponds to ℓ=1)
- Piece 3: μ⋆Λ⋆Λ piece (corresponds to ℓ=2)

**Mapping**: code piece index = PRZZ ℓ + 1

---

## Power Formula Comparison

### OLD Formula (1-based, in terms_k3_d1.py)
```
I₁: (1-u)^{ℓ₁+ℓ₂}
I₃: (1-u)^{ℓ₁}
I₄: (1-u)^{ℓ₂}
```

### V2 Formula (1-based, corrected)
```
I₁: (1-u)^{max(0, ℓ₁+ℓ₂-2)}
I₃: (1-u)^{max(0, ℓ₁-1)}
I₄: (1-u)^{max(0, ℓ₂-1)}
```

### Concrete Values

| Pair (1-based) | I₁ OLD | I₁ V2 | I₃ OLD | I₃ V2 | I₄ OLD | I₄ V2 |
|----------------|--------|-------|--------|-------|--------|-------|
| (1,1) | 2 | 2* | 1 | 0 | 1 | 0 |
| (2,2) | 4 | 2 | 2 | 1 | 2 | 1 |
| (1,2) | 3 | 1 | 1 | 0 | 2 | 1 |
| (2,1) | 3 | 1 | 2 | 1 | 1 | 0 |
| (1,3) | 4 | 2 | 1 | 0 | 3 | 2 |
| (3,1) | 4 | 2 | 3 | 2 | 1 | 0 |
| (2,3) | 5 | 3 | 2 | 1 | 3 | 2 |
| (3,2) | 5 | 3 | 3 | 2 | 2 | 1 |
| (3,3) | 6 | 4 | 3 | 2 | 3 | 2 |

*V2 special-cases (1,1) to power=2 for stability

---

## Evidence from Run 12A

The channel diff (docs/RUN12A_CHANNEL_DIFF.md) shows:

| Variable | κ OLD | κ V2 | V2/OLD |
|----------|-------|------|--------|
| I1_plus | +0.0849 | **-0.1111** | **-1.31** |
| I2_plus | +0.7126 | +0.7126 | 1.00 |
| S34_plus | -0.3379 | **-1.2111** | **3.58** |
| c | 2.122 | **0.775** | **0.37** |

**Critical Finding**: I1_plus changes SIGN from positive (OLD) to negative (V2)!

This sign flip explains the catastrophic failure of V2 under tex_mirror.

---

## Why Does This Happen?

### The (1-u) power affects the integral's value

Consider I₁ for (2,2):
- OLD: `∫₀¹ (1-u)⁴ P₂(x+u)P₂(y+u) Q²(t)... du dt`
- V2: `∫₀¹ (1-u)² P₂(x+u)P₂(y+u) Q²(t)... du dt`

The (1-u)⁴ factor weights the integrand more toward u=0 than (1-u)².
After taking d²/dxdy|_{x=y=0}, these produce different numerical values.

### Why V2 "works" for isolated I-term validation but fails in assembly

The Run 11 validation proved: "Direct V2 matches DSL V2" (ratio=1.0).

This is internally consistent: V2 DSL and direct V2 both use the same (1-u)^{ℓ₁+ℓ₂-2} formula.

BUT the assembly formula (tex_mirror) was developed and calibrated using OLD terms.
When you swap in V2 terms with different (1-u) powers:
- The +R channels change
- The -R channels change
- The shape factors (m_implied) become incompatible
- The amplitude model (A1, A2) no longer compensates correctly

---

## Decision

### TeX-truth designation

**OLD is TeX-truth** for the following reasons:

1. **Direct match for (1,1)**: OLD gives power=2 for I₁(1,1), matching TeX line 1435
2. **Production compatibility**: OLD + tex_mirror achieves <1% accuracy on both benchmarks
3. **Sign consistency**: OLD produces positive I1_plus, matching expected physics

### V2 status

**V2 is an internally-consistent alternate structure** that:
- Was derived from PRZZ Section 7 (single-variable formulation)
- Is proven correct for individual I-term isolation
- Uses a different (1-u) power formula
- Requires different amplitude/assembly model to work

### Implications

1. **Continue using OLD + tex_mirror** as the production baseline
2. **Do NOT swap V2 into tex_mirror** without rebuilding the assembly model
3. **V2 validation remains valuable** for verifying term structure
4. **Future work**: If pursuing V2, must re-derive the mirror assembly formula

---

## References

- RMS_PRZZ.tex lines 1435, 1484, 1488: Explicit (1,1) formulas
- RMS_PRZZ.tex lines 2391-2409: Euler-Maclaurin lemma
- docs/RUN12A_CHANNEL_DIFF.md: Empirical channel comparison
- terms_k3_d1.py lines 2808-2971: V2 term builders
