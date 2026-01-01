# Paper Appendix Outline: Combinatorial Derivations

## Overview

This document outlines the appendices needed to show the "ugly work" that proves our κ improvement is mathematically valid. The goal is to make the combinatorial wall transparent and verifiable.

---

## Appendix A: The Difference Quotient Identity and I₁ Assembly

### A.1 Statement of the Identity

**Proposition A.1** (PRZZ Lines 1508-1511): For N = T^θ and complex α, β with Re(α), Re(β) > -1:

```
[N^{αx+βy} - T^{-α-β} N^{-βx-αy}] / (α+β)
= N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

### A.2 Proof of the Identity

**(Include full proof with all steps)**

### A.3 Specialization to α = β = -R/L

At α = β = -R/L, the identity becomes (Line 1522):
```
(1 + θ(x+y)) × ∫₀¹ e^{R[2t + θ(2t-1)(x+y)]} dt
```

### A.4 Application of the Q Operator

**Proposition A.2** (Lines 1512-1518): After applying Q(-∂/∂α) Q(-∂/∂β):
```
Q(θt(x+y) - θy + t) × Q(θt(x+y) - θx + t) × e^{R[...]}
```

### A.5 The Complete I₁ Formula

**Theorem A.3** (Lines 1529-1533):
```
I₁ = T × (θ(x+y)+1)/θ × d²/dxdy ∫∫ (1-u)² P₁(x+u) P₂(y+u)
     × Q(...)Q(...) × exp(...) du dt |_{x=y=0}
```

**(Show all cancellations in the derivation)**

---

## Appendix B: The ω-Classification and Cases A, B, C

### B.1 Definition of ω

**Definition B.1** (Line 2303):
```
ω(d, l) := 1×l₁ + 2×l₂ + ... + d×l_d - 1
```

For K=3, d=1, the possible values are ω ∈ {-1, 0, 1, 2}.

### B.2 Case A: ω = -1 (Derivative Terms)

**Proposition B.2** (Lines 2305-2323): When ω = -1:
```
Υ_A = U(d,l) × (1/i!) × d/dx [e^{αx} × polynomial] |_{x=0}
```

where U(d,l) = (1!(-1)¹)^{l₁} × (2!(-1)²)^{l₂} × ...

### B.3 Case B: ω = 0 (No Attenuation)

**Proposition B.3** (Lines 2324-2335): When ω = 0:
```
Υ_B = -V(d,l) × polynomial evaluation (no derivatives)
```

### B.4 Case C: ω > 0 (Auxiliary Integral)

**Proposition B.4** (Lines 2336-2362): When ω > 0, an auxiliary integral appears:
```
Υ_C = W(d,l) × (-1)^{1-ω}/((ω-1)!) × (log N)^ω × ∫₀¹ polynomial × a^{ω-1} × (N/n)^{-αa} da
```

This integral arises from the identity (Line 2347):
```
∫_{1/q}¹ t^{α+s-1} log^τ t dt = (-1)^τ τ! / (α+s)^{τ+1} - ...
```

### B.5 The Factorial and Sign Tracking

**Lemma B.5**: The constants U, V, W involve:
- Sign factors: (-1)^{l₁ + 2l₂ + ... + dl_d}
- Factorial factors: (1!)^{l₁} × (2!)^{l₂} × ... × (d!)^{l_d}

**(Show explicit calculation for each case)**

---

## Appendix C: The Six Pair Contributions for K=3

### C.1 Table of Pair Types

| Pair (ℓ₁,ℓ₂) | ω(ℓ₁) | ω(ℓ₂) | Case Type | Contribution Sign |
|--------------|-------|-------|-----------|-------------------|
| (1,1) | -1 | -1 | A×A | + |
| (1,2) | -1 | 0 | A×B | + |
| (1,3) | -1 | 1 | A×C | **can be ±** |
| (2,2) | 0 | 0 | B×B | + |
| (2,3) | 0 | 1 | B×C | **can be ±** |
| (3,3) | 1 | 1 | C×C | + |

### C.2 Explicit Formulas for Each Pair

#### Pair (1,1): A×A

```
S₁₁ = ∫∫ (1-u)² P₁(u)² × Q(t)² × e^{2Rt} × [derivative structure] du dt
```

#### Pair (1,2): A×B

```
S₁₂ = ∫∫ (1-u) P₁(u) P₂(u) × Q(t)² × e^{2Rt} × [mixed structure] du dt
```

#### Pair (1,3): A×C

```
S₁₃ = ∫∫∫ (1-u) P₁(u) P₃((1-a)u) × a^0 × Q(...)² × e^{...} × [Case C integral] da du dt
```

#### Pair (2,2): B×B

```
S₂₂ = ∫∫ P₂(u)² × Q(t)² × e^{2Rt} du dt
```

#### Pair (2,3): B×C

```
S₂₃ = ∫∫∫ P₂(u) P₃((1-a)u) × a^0 × Q(...)² × e^{...} da du dt
```

#### Pair (3,3): C×C

```
S₃₃ = ∫∫∫∫ P₃((1-a)u) P₃((1-b)u) × a^0 × b^0 × Q(...)² × e^{...} da db du dt
```

### C.3 Numerical Values (Baseline κ-polynomials)

| Pair | Contribution | % of Total |
|------|--------------|------------|
| (1,1) | 0.4129 | 41.5% |
| (1,2) | 0.4136 | 41.6% |
| (1,3) | 0.0128 | 1.3% |
| (2,2) | 0.1398 | 14.0% |
| (2,3) | 0.0156 | 1.6% |
| (3,3) | 0.0008 | 0.1% |
| **Total** | **0.9954** | **100%** |

---

## Appendix D: Why Negative Pair Contributions Are Valid

### D.1 Statement of the Problem

With optimized polynomials (α=61 perturbation):
- Pair (1,3): 0.0128 → **-0.0991** (sign flip)
- Pair (2,3): 0.0156 → **-0.0667** (sign flip)

**Question**: Is this mathematically valid?

### D.2 Theoretical Analysis

**Theorem D.1**: Individual pair contributions S_{ℓ₁,ℓ₂} are NOT constrained to be positive.

**Proof sketch**:
1. The PRZZ integral structure involves products P_{ℓ₁}(u) × P_{ℓ₂}(u)
2. For ℓ₁ ≠ ℓ₂, the polynomial product can be negative on parts of [0,1]
3. The Case C auxiliary integral can produce sign changes
4. Only the TOTAL c = exp(R(1-κ)) must be positive

### D.3 Physical Interpretation

- Pairs (1,3) and (2,3) involve cross-terms between different mollifier pieces
- These cross-terms can interfere destructively
- The optimization found a direction where destructive interference reduces c

### D.4 Numerical Verification

| α | S₁₃ | S₂₃ | Total S₁₂ | Valid? |
|---|-----|-----|-----------|--------|
| 0 | +0.0128 | +0.0156 | 0.9954 | ✓ |
| 61 | -0.0991 | -0.0667 | 0.7961 | ✓ |

Total remains positive, confirming validity.

---

## Appendix E: Mirror Term Assembly

### E.1 The Mirror Identity

**Proposition E.1** (Line 1502):
```
I₁(α,β) = I_{1,1}(α,β) + T^{-α-β} × I_{1,1}(-β,-α) + O(T/L)
```

At α = β = -R/L, the factor T^{-α-β} becomes T^{2R/L} ≈ exp(2R/θ).

### E.2 Why S₁₂ Needs Mirror but S₃₄ Doesn't

**Theorem E.2**:
- I₁ and I₂ involve the (N^{αx+βy} - T^{-α-β}N^{-βx-αy})/(α+β) structure
- I₃ and I₄ have the structure N^{αx}/(α+β) which lacks the T^{-α-β} coupling

Therefore: c = S₁₂(+R) + m × S₁₂(-R) + S₃₄(+R)

### E.3 The m = exp(R) + (2K-1) Formula

**Proposition E.3**: For K mollifier pieces:
```
m = (f_{I₁} × g_{I₁} + (1-f_{I₁}) × g_{I₂}) × (exp(R) + 2K - 1)
```

where g_{I₁}, g_{I₂} are correction factors.

**Derivation** (sketch):
- The (2K-1) term arises from the number of cross-terms
- For K=3: 2×3 - 1 = 5
- This counts the interference pattern between K pieces

### E.4 Correction Factors g_{I₁}, g_{I₂}

**Theorem E.4** (First-principles formulas):
```
g_{I₁} = 1 + θ(1-θ)(2(K-1)+θ) / (8K(2K+1)²)
g_{I₂} = 1 + θ(2-θ) / (2K(2K+1))
```

For K=3, θ=4/7:
- g_{I₁} = 1.000952
- g_{I₂} = 1.019436

---

## Appendix F: Numerical Verification Summary

### F.1 Golden Values (Locked)

| Quantity | Value | Tolerance |
|----------|-------|-----------|
| κ (R=1.3036) | 0.417293962 | ±0.001% |
| κ* (R=1.1167) | 0.407511457 | ±0.001% |
| c | 2.137454406... | ±0.001% |
| g_{I₁} | 1.0009519843 | exact |
| g_{I₂} | 1.0194363460 | exact |

### F.2 Quadrature Convergence

| n_quad | κ (baseline) | κ (optimized) | Improvement |
|--------|--------------|---------------|-------------|
| 40 | 0.41729593 | 0.44933435 | +7.6776% |
| 60 | 0.41729593 | 0.44933435 | +7.6776% |
| 80 | 0.41729593 | 0.44933435 | +7.6776% |
| 100 | 0.41729593 | 0.44933435 | +7.6776% |

**Variation**: < 10⁻⁶% (stable)

### F.3 R-Sweep Validation

| R | κ (baseline) | κ (optimized) | Improvement |
|---|--------------|---------------|-------------|
| 1.10 | 0.338 | 0.378 | +11.9% |
| 1.20 | 0.382 | 0.418 | +9.4% |
| 1.3036 | 0.417 | 0.449 | +7.7% |
| 1.35 | 0.430 | 0.461 | +7.1% |
| 1.40 | 0.443 | 0.472 | +6.5% |

**Conclusion**: Improvement persists across R values (not overfitting).

### F.4 Test Suite Summary

| Test File | Tests | Purpose |
|-----------|-------|---------|
| test_golden_regression.py | 10 | Drift detection |
| test_production_guards.py | 19 | Prevent calibration creep |
| test_out_of_sample_smoke.py | 18 | Robustness |
| test_kappa_engine.py | 14 | Core engine |
| **Total** | **61** | **All passing** |

---

## Writing Schedule

| Appendix | Content | Priority | Estimated Effort |
|----------|---------|----------|------------------|
| A | I₁ derivation | HIGH | 1 week |
| B | Case A/B/C | HIGH | 1 week |
| C | Six pairs | MEDIUM | 3 days |
| D | Negative validity | HIGH | 2 days |
| E | Mirror assembly | MEDIUM | 3 days |
| F | Numerical verification | LOW | 2 days |

**Total estimated effort**: 4-5 weeks of focused writing.
