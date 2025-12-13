# TECHNICAL_ANALYSIS.md — PRZZ Framework Extension Notes (κ Optimization)

## Scope

This document is a handoff for implementing the PRZZ (2019) unconditional framework to compute κ,
the proportion of zeta zeros on the critical line, using multi-piece (Feng/Levinson) mollifiers.

It is written for implementers. Where PRZZ has multiple equivalent analytic reductions,
we prioritize implementation-ready forms and emphasize validation strategy.

---

## Table of Contents

1. [Background and History](#1-background-and-history)
2. [The κ Bound and the Constant c](#2-the-κ-bound-and-the-constant-c)
3. [Mollifier Structure](#3-mollifier-structure)
4. [Mean Value Decomposition](#4-mean-value-decomposition)
5. [Autocorrelation Ratio and Residue Derivatives](#5-autocorrelation-ratio-and-residue-derivatives)
6. [Main Bracket Formulas](#6-main-bracket-formulas)
7. [Arithmetic Factor A and its Derivatives](#7-arithmetic-factor-a-and-its-derivatives)
8. [Variable Structure](#8-variable-structure)
9. [The (1,2) Pilot Derivation](#9-the-12-pilot-derivation)
10. [Mapping to Final Integral Templates](#10-mapping-to-final-integral-templates)
11. [PRZZ Published Parameters](#11-przz-published-parameters)

---

## 1. Background and History

### 1.1 The Problem

The Riemann Hypothesis (RH) asserts that all non-trivial zeros of ζ(s) lie on the 
critical line Re(s) = 1/2. While RH remains unproven, we can prove that a positive 
proportion κ of zeros lie on the critical line.

### 1.2 Historical Progress

| Year | Authors | Method | κ bound |
|------|---------|--------|---------|
| 1942 | Selberg | First positive proportion | κ > 0 |
| 1974 | Levinson | Mollifier method | κ > 1/3 |
| 1989 | Conrey | Improved mollifier | κ > 0.40 |
| 2010 | Feng (v1) | K-piece mollifier, θ=4/7 all pieces | κ > 0.4173 (RETRACTED) |
| 2010 | Feng (v2) | Corrected, θ₁=4/7, θ₂=1/2 | κ > 0.4128 |
| 2014 | Bui survey | Conservative bound | κ > 0.410725 |
| 2019 | PRZZ | New exponential sum bounds | κ ≥ 0.417293962 |

### 1.3 The Feng Gap and PRZZ Resolution

**The Problem (2010):**
Feng's original claim used θ = 4/7 for ALL mollifier pieces. Conrey identified that 
the Balasubramanian et al. constraint **θ₁ + θ₂ < 1** was violated in cross-terms 
(4/7 + 4/7 = 8/7 > 1). Feng retracted and published a weaker result.

**PRZZ's Solution (2019):**
PRZZ provided new exponential sum bounds (Theorem 4.1, Case 3) specifically for 
coefficients of the form (μ ⋆ Λ₁^{⋆k₁} ⋆ ...). The error bound:

```
E = T^ε(N^{7/4} + N^{7/8}T^{1/2})
```

allows N = T^{4/7-ε}, which permits θ = 4/7 for ALL mollifier pieces including cross-terms.

**Verification:** We have confirmed through detailed analysis that PRZZ:
1. Explicitly acknowledges the historical 3/7 obstruction
2. Provides Theorem 4.1 Case 3 as the resolution
3. Handles cross-terms explicitly ("cross 3×3 = 9 cases")

This is NOT Feng's retracted result reused—it's a rigorous new framework.

---

## 2. The κ Bound and the Constant c

### 2.1 Levinson-type inequality

```
κ ≥ 1 − (1/R)·log(c)
```

where:
- R is the shift parameter: σ₀ = 1/2 − R/log(T)
- c is the main-term constant in the mollified mean square asymptotic

### 2.2 PRZZ benchmark values

```
R = 1.3036
κ = 0.417293962
c = exp(R(1−κ)) = 2.13745440613217263636...
```

### 2.3 Sensitivity analysis

```
∂κ/∂c = −1/(R·c) ≈ −0.359

Relative sensitivity:
δκ ≈ −(1/R)·(δc/c) ≈ −0.767·(δc/c)
```

**Critical engineering insight:**
```
To achieve κ = 0.42 (from 0.41729):
  c_target = exp(1.3036 × 0.58) ≈ 2.12993
  Required reduction: (2.1375 - 2.1299)/2.1375 ≈ 0.35%
```

A 0.35% improvement in c is achievable through optimization—this is not a moonshot.

---

## 3. Mollifier Structure

### 3.1 Piece indexing convention

For this project:
- Piece index ℓ ∈ {1,…,K} corresponds to polynomial P_ℓ
- The ℓ-th piece uses coefficients ∝ (μ ⋆ Λ^{⋆(ℓ−1)})(n) / (log N)^{ℓ−1}

Thus:
- ℓ=1: μ piece (no Λ factor)
- ℓ=2: μ⋆Λ piece
- ℓ=3: μ⋆Λ⋆Λ piece

### 3.2 The d=1 mollifier

```
ψ(s) = Σ_{ℓ=1}^K (−1)^{ℓ−1} Σ_{n≤N} a_ℓ(n) n^{σ₀−1/2} / n^s · P_ℓ(log(N/n)/log N)
```

with a_ℓ(n) ∝ (μ⋆Λ^{⋆(ℓ−1)})(n) / (log N)^{ℓ−1}.

### 3.3 The differential operator V

```
V(s) = Q(−(1/L)d/ds) ζ(s) = ζ(s) + c₁ζ'(s)/L + c₂ζ''(s)/L² + ...
```

where L = log T and Q(x) = Σ qⱼxʲ with Q(0) = 1.

### 3.4 Polynomial constraints

**Verified constraints from PRZZ:**
- P_ℓ(0) = 0 for all ℓ ≥ 1
- P₁(1) = 1 (normalization)
- Q(0) = 1

**Note:** There is NO evidence that P₂(1) = 0 or P₃(1) = 0 in the published example.

---

## 4. Mean Value Decomposition

### 4.1 Structure of c

The mean value c decomposes as:

```
c = Σᵢ c_{ii} + 2·Σ_{i<j} c_{ij}
```

For K = 3:
- Diagonal terms: c₁₁, c₂₂, c₃₃
- Cross terms: c₁₂, c₁₃, c₂₃ (with factor of 2)

Total: 6 distinct pair computations.

For K = 4:
- 4 diagonal + 6 cross = 10 pair types

### 4.2 Pair sign convention

With the standard alternating sign (−1)^{ℓ−1} in the mollifier, the pair contribution 
carries sign (−1)^{ℓ₁+ℓ₂}.

---

## 5. Autocorrelation Ratio and Residue Derivatives

### 5.1 The autocorrelation ratio structure

PRZZ expresses the mean value via an "autocorrelation of ratios" Euler product:
- A ratio of zeta factors
- Times an arithmetic Euler product A_{α,β}

### 5.2 Residue extraction

Residue extraction uses contour integrals:
```
∮ dz_i/z_i² and ∮ dw_j/w_j²
```

which correspond to partial derivatives at 0:
```
∂_{z_i} and ∂_{w_j} evaluated at z=w=0
```

### 5.3 The F = exp(G) trick

**Key computational method:**
Write the integrand as F = exp(G), with G = log(zeta-ratio) + log(A_{α,β}).

Then derivatives of F follow from Bell/partition formulas. For order 3 (the (1,2) case):

```
F_{zw₁w₂}(0) = F(0) × [
    G_{zw₁w₂} + 
    G_{zw₁}G_{w₂} + 
    G_{zw₂}G_{w₁} + 
    G_{w₁w₂}G_z + 
    G_z G_{w₁}G_{w₂}
]
```

This avoids product-rule explosion.

---

## 6. Main Bracket Formulas

### 6.1 Zeta log-derivative primitives

Define (at the evaluation point):
```
A₁ := (ζ'/ζ)(1+α+s) − (ζ'/ζ)(1+s+u)
B  := (ζ'/ζ)(1+β+u) − (ζ'/ζ)(1+s+u)
C  := ((ζ'/ζ)'(1+s+u) = ζ''/ζ(1+s+u) − (ζ'/ζ(1+s+u))²
```

### 6.2 Universal bracket formula

For pair (ℓ₁, ℓ₂), the ratio-only bracket is:

```
ℬ^{main}_{ℓ₁,ℓ₂} = Σ_{k=0}^{min(ℓ₁,ℓ₂)} [ℓ₁!ℓ₂! / ((ℓ₁−k)!(ℓ₂−k)!k!)] · A₁^{ℓ₁−k} · B^{ℓ₂−k} · C^k
```

**Interpretation:** Choose k cross-pairings between z's and w's (each contributes C),
remaining unmatched z's contribute A₁, remaining unmatched w's contribute B.

### 6.3 Explicit bracket table for K=3

| Pair | Bracket ℬ^{main}_{ℓ₁,ℓ₂} | Sign (−1)^{ℓ₁+ℓ₂} |
|------|--------------------------|-------------------|
| (1,1) | A₁B + C | +1 |
| (1,2) | A₁B² + 2BC | −1 |
| (1,3) | A₁B³ + 3B²C | +1 |
| (2,2) | A₁²B² + 4A₁BC + 2C² | +1 |
| (2,3) | A₁²B³ + 6A₁B²C + 6BC² | −1 |
| (3,3) | A₁³B³ + 9A₁²B²C + 18A₁BC² + 6C³ | +1 |

**Important:** (2,2) is NOT (A₁B + C)². The formula is combinatorial matching, not algebraic squaring.

### 6.4 Extension to K=4

Additional brackets needed:
| Pair | Bracket |
|------|---------|
| (1,4) | A₁B⁴ + 4B³C |
| (2,4) | A₁²B⁴ + 8A₁B³C + 12B²C² |
| (3,4) | A₁³B⁴ + 12A₁²B³C + 36A₁B²C² + 24BC³ |
| (4,4) | A₁⁴B⁴ + 16A₁³B³C + 72A₁²B²C² + 96A₁BC³ + 24C⁴ |

---

## 7. Arithmetic Factor A and its Derivatives

### 7.1 Special evaluation point

At (s,u) = (β,α), the three key exponents align:
```
1+s+u = 1+α+s = 1+β+u = E = 1+α+β
```

This is where PRZZ evaluates A-derivative patterns.

### 7.2 Verified vanishing pattern

At z_i = w_j = 0 and (s,u) = (β,α):

| Derivative Type | Value | Reason |
|-----------------|-------|--------|
| All first derivatives | 0 | Cancellation in log(1-q) terms |
| (log A)_{z_i z_{i'}} | 0 | No terms couple z's together |
| (log A)_{w_j w_{j'}} | 0 | No terms couple w's together |
| (log A)_{z_i w_j} | −S(α+β) | Universal cross-term |
| Higher mixed | 0 | No three-variable couplings |

where:
```
S(α+β) = Σ_p (log p / (p^{1+α+β} − 1))²
```

### 7.3 Explicit verification for (1,2)

From PRZZ's logderivativeA formula specialized to (1,2) at (β,α):

```
(log A)_z = 0
(log A)_{w₁} = (log A)_{w₂} = 0
(log A)_{zw₁} = (log A)_{zw₂} = −S(α+β)
(log A)_{w₁w₂} = 0
(log A)_{zw₁w₂} = 0
```

**All verified by direct computation from the Euler product.**

### 7.4 Implementation consequence

The A-factor corrections appear as "I5-type" terms that are lower-order (suppressed by 1/(log N)²).

**Strategy:**
- Start by implementing ratio-only brackets (A ≡ 1)
- Add A-correction terms as optional "audit" to confirm they match what PRZZ included

---

## 8. Variable Structure

### 8.1 Why multi-variable is mandatory

Even though there is no ζ(…+w₁+w₂) coupling, you CANNOT compress w₁, w₂ into y.

**Reason:** Setting H(y) = F(y,y), we get:
```
H''(0) = F_{w₁w₁}(0) + 2F_{w₁w₂}(0) + F_{w₂w₂}(0)
```

This mixes the desired mixed derivative with pure second derivatives.

### 8.2 Complete variable structure for all pairs

| Pair | Variables | deriv_orders | Total vars |
|------|-----------|--------------|------------|
| (1,1) | (x, y) | {x:1, y:1} | 2 |
| (1,2) | (x, y1, y2) | {x:1, y1:1, y2:1} | 3 |
| (1,3) | (x, y1, y2, y3) | all 1's | 4 |
| (2,2) | (x1, x2, y1, y2) | all 1's | 4 |
| (2,3) | (x1, x2, y1, y2, y3) | all 1's | 5 |
| (3,3) | (x1, x2, x3, y1, y2, y3) | all 1's | 6 |

For K=4, maximum is (4,4) with 8 variables.

### 8.3 Bitset optimization

Since all derivative orders are 1, each monomial in the series is a product of distinct variables.
We can represent monomials as **bitsets**:
- vars = (v₀, v₁, ..., v_{m-1})
- monomial key = integer mask in [0, 2^m)

Maximum monomials: 2^6 = 64 for K=3, 2^8 = 256 for K=4.

Series multiplication: for masks a, b with (a & b) == 0:
```
out[a | b] += A[a] * B[b]
```

---

## 9. The (1,2) Pilot Derivation

### 9.1 Zeta-factor list

**Numerator:**
- ζ(1+s+u)^6
- ζ(1+α+s+z)
- ζ(1+β+u+w₁), ζ(1+β+u+w₂)
- ζ(1+s+u+z+w₁), ζ(1+s+u+z+w₂)

**Denominator:**
- ζ(1+α+s)^2
- ζ(1+β+u)^3
- ζ(1+s+u+w₁)^2, ζ(1+s+u+w₂)^2
- ζ(1+s+u+z)^3

**Critical observation:** No ζ(…+w₁+w₂) factor exists.

### 9.2 G = log(ratio) + log A

```
G_rat = 6·log ζ(1+s+u) − 2·log ζ(1+α+s) − 3·log ζ(1+β+u)
      + log ζ(1+α+s+z) + log ζ(1+β+u+w₁) + log ζ(1+β+u+w₂)
      + log ζ(1+s+u+z+w₁) + log ζ(1+s+u+z+w₂)
      − 2·log ζ(1+s+u+w₁) − 2·log ζ(1+s+u+w₂) − 3·log ζ(1+s+u+z)
```

### 9.3 Ratio-only derivatives at z=w₁=w₂=0

```
G_z|₀ = A₁
G_{w₁}|₀ = G_{w₂}|₀ = B
G_{zw₁}|₀ = G_{zw₂}|₀ = C
G_{w₁w₂}|₀ = 0
G_{zw₁w₂}|₀ = 0
```

### 9.4 Assembling via F = exp(G)

```
F_{zw₁w₂}(0)/F(0) = G_{zw₁w₂} + G_{zw₁}G_{w₂} + G_{zw₂}G_{w₁} + G_{w₁w₂}G_z + G_z G_{w₁}G_{w₂}
                  = 0 + C·B + C·B + 0·A₁ + A₁·B·B
                  = A₁B² + 2BC  ✓
```

**Matches our bracket table.**

### 9.5 Full bracket with A-corrections

```
Full = A₁B² + 2B(C − S(α+β)) = (A₁B² + 2BC) − 2B·S(α+β)
```

The −2B·S term is the I5-type arithmetic correction.

---

## 10. Mapping to Final Integral Templates

### 10.1 Structural pattern

| Component | (1,1) | (1,2) | General (ℓ₁, ℓ₂) |
|-----------|-------|-------|-------------------|
| Poly prefactor | (1−u)² | (1−u)³ | (1−u)^{ℓ₁+ℓ₂} |
| P_left arg | x + u | x + u | (Σᵢ xᵢ) + u |
| P_right arg | y + u | y₁+y₂ + u | (Σⱼ yⱼ) + u |
| Total sum | x + y | x + y₁ + y₂ | S = (Σᵢ xᵢ) + (Σⱼ yⱼ) |
| Domain | [0,1]² | [0,1]² | [0,1]² (main terms) |

### 10.2 Q and exponential arguments

Let S = sum of all formal variables, X = sum of x-vars, Y = sum of y-vars.

```
Arg_α = θ·t·S − θ·Y + t
Arg_β = θ·t·S − θ·X + t
```

Q factors: Q(Arg_α), Q(Arg_β)
Exp factors: exp(R·Arg_α), exp(R·Arg_β)

### 10.3 The (1,1) terms from PRZZ Section 6.2.1

PRZZ gives four explicit terms for (1,1):

**I₁** (main coupled term, ∂²/∂x∂y):
```
(θ(x+y)+1)/θ × ∫∫ (1−u)² P₁(x+u) P₂(y+u) Q(...)² exp(2R·...) du dt
```

**I₂** (decoupled, no derivatives):
```
(1/θ) × ∫∫ P₁(u) P₂(u) Q(t)² exp(2Rt) du dt
```

**I₃, I₄** (single derivative terms with leading minus signs)

**I₅** (arithmetic correction, lower order)

### 10.4 Warning about integration domains

Most main terms reduce to [0,1]², but:
- Some lower-order/arithmetic terms may need 3D integrals
- Architecture should support at least 3D quadrature

---

## 11. PRZZ Published Parameters

### 11.1 Configuration

```
K = 3 (number of mollifier pieces)
d = 1 (derivative depth)
θ = 4/7 ≈ 0.571428571428...
R = 1.3036
```

### 11.2 Polynomials

```python
# P₁: satisfies P₁(0)=0, P₁(1)=1
# Written as P₁(x) = x + x(1-x)·(...) 
P1_coeffs = {
    # x + 0.261076·x(1-x) - 1.071007·x(1-x)² - 0.236840·x(1-x)³ + 0.260233·x(1-x)⁴
    "form": "x + x(1-x)*P_tilde",
    "P_tilde_coeffs": [0.261076, -1.071007, -0.236840, 0.260233]  # in (1-x) powers
}

# P₂: satisfies P₂(0)=0
# P₂(x) = 1.048274·x + 1.319912·x² - 0.940058·x³
P2_coeffs = [0, 1.048274, 1.319912, -0.940058]

# P₃: satisfies P₃(0)=0
# P₃(x) = 0.522811·x - 0.686510·x² - 0.049923·x³
P3_coeffs = [0, 0.522811, -0.686510, -0.049923]

# Q: satisfies Q(0)=1, odd symmetry in (1-2x)
# Q(x) = 0.490464 + 0.636851·(1-2x) - 0.159327·(1-2x)³ + 0.032011·(1-2x)⁵
Q_coeffs = {
    "basis": "(1-2x)^k for odd k, plus constant",
    "values": {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
}
```

### 11.3 Target outputs

```
c = 2.13745440613217263636...
κ = 0.417293962
```

---

## Appendix A: PRZZ Paper Reference Map

| Our Concept | PRZZ Section | Key Equation/Label |
|-------------|--------------|-------------------|
| Mollifier ψ | §3 | (3.1)-(3.5) |
| θ = 4/7 justification | §4 | Theorem 4.1, Case 3 |
| Autocorrelation ratio | §6 | autocorrelationratio |
| log A formula | §6 | (6.11) logderivativeA |
| (1,1) worked example | §6.2.1 | I₁-I₅ |
| (2,2) A-derivative vanishing | §6.2.2 | Explicit verification |
| Numerical values | §8 | Final parameters |

## Appendix B: Quick Reference

```
κ = 1 - log(c)/R
c = exp(R(1-κ))
∂κ/∂c = -1/(R·c) ≈ -0.359

Current best: κ = 0.417293962, c = 2.1374544...
Target κ=0.42: c ≈ 2.1299, improvement needed ≈ 0.35%
```
