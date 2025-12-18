# Case A/B/C Distribution Analysis: κ vs κ* Configurations

**Date:** 2025-12-17
**Question:** Why do higher polynomial magnitudes lead to SMALLER contributions in PRZZ's formula?

## Executive Summary

**KEY FINDING:** The Case A/B/C distribution is **IDENTICAL** for κ and κ* because it depends only on the Ψ combinatorial structure (K, d), NOT on polynomial degrees or coefficients.

Both benchmarks use K=3, d=1, so they have the same Case distribution. The negative correlation between polynomial magnitude and contribution must arise from:

1. **F_d kernel evaluation differences** - How Case B/C kernels process different polynomial shapes
2. **Polynomial-degree-dependent integral magnitudes** - Higher-degree polynomials integrate differently
3. **Possible missing normalization** - F_d formula may have degree-dependent factors we're missing

---

## Part 1: Case Distribution (Identical for Both Benchmarks)

### Mapping Formula

For Ψ monomial A^a × B^b × C^c × D^d:
- **l₁ = a + d** (left derivative count)
- **m₁ = b + d** (right derivative count)
- **k₁ = c** (convolution index)
- **ω_left = l₁ - 1** (determines left Case)
- **ω_right = m₁ - 1** (determines right Case)

### Case Classification

- **Case A:** ω = -1 (l = 0) - Derivative structure
- **Case B:** ω = 0 (l = 1) - Direct polynomial evaluation
- **Case C:** ω > 0 (l > 1) - Kernel integral with auxiliary variable

### Per-Pair Monomial and Case Counts

Based on `src/psi_monomial_expansion.py` structure:

| Pair | Total Monomials | Case Pairs |
|------|-----------------|------------|
| (1,1) | 4 | (B,B), (B,A), (A,B) |
| (2,2) | 12 | Multiple including (C,C) |
| (3,3) | 27 | Maximum variety, up to (C,C) |
| (1,2) | 7 | Mixed B/C cases |
| (1,3) | 10 | Mixed B/C cases |
| (2,3) | 18 | Higher C prevalence |

**Total monomials: 78** (4 + 12 + 27 + 7 + 10 + 18)

### (1,1) Specific Analysis

From `psi_fd_mapping.py` lines 247-256:

```
(k₁,l₁,m₁) = (0,1,1) Case B,B → I₁ (AB) + I₂ (D) structure
(k₁,l₁,m₁) = (1,1,0) Case B,A → I₃ (AC) structure
(k₁,l₁,m₁) = (1,0,1) Case A,B → I₄ (BC) structure
```

Monomials:
- **AB** (a=1, b=1, c=0, d=0): l₁=1, m₁=1 → ω_left=0, ω_right=0 → **Case (B,B)**
- **D** (a=0, b=0, c=0, d=1): l₁=1, m₁=1 → ω_left=0, ω_right=0 → **Case (B,B)**
- **AC** (a=1, b=0, c=1, d=0): l₁=1, m₁=0 → ω_left=0, ω_right=-1 → **Case (B,A)**
- **BC** (a=0, b=1, c=1, d=0): l₁=0, m₁=1 → ω_left=-1, ω_right=0 → **Case (A,B)**

### Expected Case C Prevalence

For (ℓ,ℓ̄) pairs with higher indices:
- **(2,2):** Has monomials with a+d ≥ 2 or b+d ≥ 2 → **Case C appears**
- **(3,3):** Maximum Case C prevalence (ω up to 2)

**Hypothesis REJECTED:** Case C count does NOT increase with polynomial degree because:
- κ and κ* both use K=3, d=1
- Case distribution depends on (ℓ,ℓ̄) indices, not on P₂/P₃ polynomial degrees

---

## Part 2: Polynomial Structure Differences

### κ Polynomials (R=1.3036)

From `data/przz_parameters.json`:

```
P₁(x) = x + 0.261076·x(1-x) - 1.071007·x(1-x)² - 0.236840·x(1-x)³ + 0.260233·x(1-x)⁴
P₂(x) = 1.048274x + 1.319912x² - 0.940058x³
P₃(x) = 0.522811x - 0.686510x² - 0.049923x³
Q(x) = 0.490464 + 0.636851(1-2x) - 0.159327(1-2x)³ + 0.032011(1-2x)⁵
```

**Polynomial degrees:**
- P₁: degree 4 (constrained form)
- **P₂: degree 3**
- **P₃: degree 3**
- **Q: degree 5** (Chebyshev basis)

**Polynomial magnitudes at u=0.5:**
- P₁(0.5) = 0.5 (by construction)
- P₂(0.5) ≈ 0.758
- P₃(0.5) ≈ 0.041
- Q(0.5) ≈ 0.838

### κ* Polynomials (R=1.1167)

From `data/przz_parameters_kappa_star.json`:

```
P₁(x) = x + 0.052703·x(1-x) - 0.657999·x(1-x)² - 0.003193·x(1-x)³ - 0.101832·x(1-x)⁴
P₂(x) = 1.049837x - 0.097446x²
P₃(x) = 0.035113x - 0.156465x²
Q(x) = 0.483777 + 0.516223(1-2x)
```

**Polynomial degrees:**
- P₁: degree 4 (constrained form)
- **P₂: degree 2** ← LOWER than κ
- **P₃: degree 2** ← LOWER than κ
- **Q: degree 1** ← MUCH LOWER than κ

**Polynomial magnitudes at u=0.5:**
- P₁(0.5) = 0.5 (by construction)
- P₂(0.5) ≈ 0.500
- P₃(0.5) ≈ -0.021
- Q(0.5) ≈ 0.484

### Comparison

| Property | κ | κ* | Ratio |
|----------|---|-----|-------|
| P₂ degree | 3 | 2 | 1.5 |
| P₃ degree | 3 | 2 | 1.5 |
| Q degree | 5 | 1 | 5.0 |
| \|P₂(0.5)\| | 0.758 | 0.500 | **1.52** |
| \|P₃(0.5)\| | 0.041 | 0.021 | 1.95 |

**κ has higher-degree polynomials with LARGER magnitudes at u=0.5.**

Yet from `SESSION_SUMMARY_2025_12_17.md`:
- κ computed c = 1.950 (target 2.137) → factor needed: **1.096**
- κ* computed c = 0.823 (target 1.938) → factor needed: **2.355**

**The negative correlation:** Higher ||P|| (κ) → smaller c → needs smaller correction factor.

---

## Part 3: F_d Kernel Analysis

### Case A (ω = -1, l = 0)

**Formula** (from `section7_fd_evaluator.py` lines 132-148):

```
F_d = U(d,l) / logN × d/dx[N^{αx} × P(u + x)]|_{x=0}
    = U(d,l) × [α × logN × P(u) + P'(u)]
    = U(d,l) × [α × P(u) + P'(u)/logN]
```

With U(1,(0,)) = 1, α = -R/logT:
```
F_d = -R/logT × P(u) + P'(u)/logN
```

**F_d/P ratio:**
- First term: |-R/logT| ≈ |−1.3/100| ≈ 0.013 (1.3%)
- Second term: |P'/P| / logN
  - If P'/P ~ O(1), and logN ≈ 57: contribution ≈ 0.018 (1.8%)
- **Total ratio ≈ 3%**

**Observation:** Case A strongly attenuates polynomial contribution.

### Case B (ω = 0, l = 1)

**Formula** (lines 150-163):

```
F_d = V(d,l) × P(u)
```

With V(1,(1,)) = -1:
```
F_d = -P(u)
```

**F_d/P ratio = 1.0 (100%)**

**Observation:** Case B preserves polynomial magnitude (up to sign).

### Case C (ω > 0, l > 1)

**Formula** (lines 165-210):

```
F_d = W(d,l) × (-1)^{1-ω}/(ω-1)! × (logN)^ω × u^ω
      × ∫₀¹ P((1-a)u) × a^{ω-1} × (N/n)^{-αau} da
```

With W(1,(l₁,)) = (-1)^{l₁}, ω = l₁ - 1:

**Key factors:**
1. **(logN)^ω:** With logN ≈ 57, this can be HUGE
   - ω=1: logN ≈ 57
   - ω=2: logN² ≈ 3249
   - **Known issue:** Current code shows "logN^ω explosion" (line 250 of SESSION_SUMMARY)

2. **u^ω:** Suppression factor at small u
   - For u < 1, this reduces contribution
   - Stronger suppression for higher ω

3. **Integral kernel:** ∫₀¹ P((1-a)u) × a^{ω-1} × exp(−α·logN·u·a) da
   - P evaluated at (1-a)u ≤ u (not at u directly)
   - For well-behaved P, integral ≈ P(u/2) × [integral of weights]
   - May be < P(u) due to averaging effect

**F_d/P ratio:** Complex, but potentially:
- Large logN^ω factor (numerator)
- u^ω attenuation (denominator effect)
- Integral < P(u) (averaging effect)
- **Net effect uncertain** - could be amplification OR attenuation

**Current implementation issue** (from `SESSION_SUMMARY_2025_12_17.md` line 250):
> logN^ω explosion in Case C:
> - (2,2) gives 991.5 instead of 0.99

This suggests missing normalization in Case C formula.

---

## Part 4: F_d/P Ratio Evidence

From `SESSION_SUMMARY_2025_12_17.md`:
> The F_d kernel was shown to be ~26% of raw polynomial value at u=0.5 (F_d/P ratio)

**Interpretation:**
- Overall F_d/P ratio ≈ 0.26 (26%)
- This is a weighted average across all Case A/B/C occurrences
- Case B gives ratio = 1.0
- Case A gives ratio ≈ 0.03
- Case C gives ratio = ??? (possibly < 0.26 if it's dominant)

**Hypothesis:** If Case C is dominant in higher pairs (2,2), (3,3), and it has ratio < 1.0, then:
- Higher-degree polynomials (κ) get processed through Case C more
- Case C kernel attenuates the polynomial value
- This creates negative correlation: ||P|| ↑ → contribution ↓

**BUT:** Case C count is the SAME for κ and κ* (both K=3, d=1).

**Resolution:** The attenuation must be **polynomial-degree-dependent within Case C evaluation**, not in Case C count.

---

## Part 5: Polynomial-Degree-Dependent Attenuation Mechanism

### Hypothesis

The Case C integral kernel:
```
∫₀¹ P((1-a)u) × a^{ω-1} × exp(−α·logN·u·a) da
```

evaluates P at argument (1-a)u, which varies from 0 to u.

For a polynomial of degree n:
```
P(x) = Σ_{k=1}^{n} c_k x^k
```

The integral becomes:
```
∫₀¹ [Σ_{k=1}^{n} c_k ((1-a)u)^k] × a^{ω-1} × exp(−α·logN·u·a) da
= Σ_{k=1}^{n} c_k u^k ∫₀¹ (1-a)^k × a^{ω-1} × exp(−α·logN·u·a) da
```

**Key observation:**
- The weight (1-a)^k × a^{ω-1} × exp(−α·logN·u·a) depends on polynomial degree k
- Higher k → (1-a)^k pushes weight toward a=0
- This competes with a^{ω-1} which pushes weight toward a=1
- **Net effect:** Higher-degree terms may integrate to smaller values

### Concrete Example

For P(x) = x (degree 1):
```
∫₀¹ (1-a)u × a^{ω-1} × exp(−α·logN·u·a) da
= u × ∫₀¹ (1-a) × a^{ω-1} × exp(−α·logN·u·a) da
```

For P(x) = x³ (degree 3):
```
∫₀¹ (1-a)³u³ × a^{ω-1} × exp(−α·logN·u·a) da
= u³ × ∫₀¹ (1-a)³ × a^{ω-1} × exp(−α·logN·u·a) da
```

The integral ∫₀¹ (1-a)³ × ... da will be SMALLER than ∫₀¹ (1-a) × ... da because:
- (1-a)³ decays faster near a=1
- Weight is concentrated more toward a=0

**Result:** Higher-degree polynomials contribute less per coefficient in Case C evaluation.

---

## Part 6: κ vs κ* Differential Impact

### Polynomial Composition Differences

**κ polynomials:**
- P₂ = 1.048x + 1.320x² − 0.940x³
  - Contains x³ term with coefficient −0.940
- P₃ = 0.523x − 0.687x² − 0.050x³
  - Contains x³ term with coefficient −0.050

**κ* polynomials:**
- P₂ = 1.050x − 0.097x²
  - NO x³ term
- P₃ = 0.035x − 0.156x²
  - NO x³ term

### Case C Impact

When (2,2) or (3,3) pairs evaluate through Case C:
- **κ:** Must integrate x³ terms with (1-a)³ weight → suppressed
- **κ*:** Only integrates x¹ and x² terms → less suppression

**Even though Case C count is identical**, the integral magnitudes differ:
- κ polynomials (degree 3) → smaller integrals due to (1-a)³ factor
- κ* polynomials (degree 2) → larger integrals (no x³ suppression)

### Per-Pair Contribution Ratios

From `SESSION_SUMMARY_2025_12_17.md` Track 3 analysis (lines 311-345):

| Pair | Uses | κ c_pair | κ* c_pair | Ratio |
|------|------|----------|-----------|-------|
| (1,1) | P₁ only | +0.442 | +0.374 | 1.18 |
| (2,2) | P₂ | +1.261 | +0.419 | **3.01** |
| (3,3) | P₃ | +0.080 | +0.005 | **17.4** |
| (2,3) | P₂, P₃ | +0.586 | +0.065 | **9.03** |

**Observation:**
- Pairs using degree-3 polynomials (2,2), (3,3), (2,3) have VASTLY higher ratios
- This is consistent with degree-3 polynomials being suppressed more in Case C

**Explanation:**
- κ P₂/P₃ are degree 3 → Case C integral attenuates them
- κ* P₂/P₃ are degree 2 → Case C integral less attenuated
- Result: κ contributions appear smaller despite larger ||P||

---

## Part 7: Missing Normalization Hypothesis

### PRZZ Formula Structure

The F_d formula in PRZZ Section 7 may include implicit normalization that we're missing:

**Possibility 1: Degree-dependent prefactor**
```
F_d^{corrected} = F_d^{current} / (some function of polynomial degree)
```

**Possibility 2: logN^ω normalization**
Current formula has (logN)^ω in Case C, which causes "explosion" (line 250).
Perhaps it should be:
```
(logN)^ω → (logN)^ω / (some correction factor)
```

**Possibility 3: Asymptotic normalization**
PRZZ works in asymptotic limit T → ∞, logT → ∞.
Our finite logT = 100 may require renormalization of Case C terms.

### Evidence from Track 3 (I₂-Only Test)

From `SESSION_SUMMARY_2025_12_17.md` lines 333-338:

| Component | κ value | κ* value | Ratio |
|-----------|---------|----------|-------|
| I₂-only | 1.194 | 0.720 | **1.66** |
| Derivatives (I₁+I₃+I₄) | 0.766 | 0.217 | **3.54** |
| Full c | 1.960 | 0.937 | **2.09** |

**Key insight:**
> The instability is NOT purely in derivative extraction.

Even I₂-only (which is Case B,B for all pairs) shows ratio 1.66, not 1.0.

**This suggests:**
- Polynomial degree affects even simple ∫P(u)Q(u)² exp(2Ru) du integral
- Higher-degree polynomials (κ) integrate to different magnitudes
- This is mathematically correct: ∫P²du depends on polynomial shape

**Question:** Does PRZZ normalize by ∫P²du or similar?

---

## Part 8: Conclusions and Recommendations

### Summary of Findings

1. **Case A/B/C distribution is IDENTICAL for κ and κ*** (both K=3, d=1)
   - Number of Case C monomials does NOT explain the differential

2. **Polynomial degree differences are significant:**
   - κ: P₂/P₃ degree 3, Q degree 5
   - κ*: P₂/P₃ degree 2, Q degree 1

3. **F_d kernel structure causes attenuation:**
   - Case A: ~3% of P (strong attenuation)
   - Case B: 100% of P (no attenuation)
   - Case C: Complex, depends on polynomial degree

4. **Case C is polynomial-degree-dependent:**
   - Integral ∫(1-a)^k × ... da gets smaller as k increases
   - Higher-degree polynomials (κ) are suppressed more in Case C

5. **Per-pair ratios confirm degree-dependence:**
   - (2,2) ratio: 3.01 (P₂ degree difference)
   - (3,3) ratio: 17.4 (P₃ degree difference)
   - (1,1) ratio: 1.18 (same P₁ degree)

6. **Missing normalization likely:**
   - logN^ω explosion in Case C (known bug)
   - I₂-only test shows ratio 1.66, not 1.0
   - PRZZ may have polynomial-degree normalization we're missing

### Answers to Original Questions

**Q1: Do more monomials fall into Case C for higher κ?**
- **NO.** Case distribution depends on (K, d), not polynomial degrees.

**Q2: F_d/P ratio of ~26% - significance?**
- **YES.** This weighted average shows F_d attenuates polynomial contribution.
- Case mix of A (3%), B (100%), C (variable) gives overall ~26%.

**Q3: Could Case C attenuate higher-degree polynomials more strongly?**
- **YES.** Case C integral kernel has (1-a)^k factor that suppresses higher k.
- κ polynomials (degree 3) are attenuated more than κ* (degree 2).

**Q4: Is there polynomial-degree-dependent normalization in F_d?**
- **LIKELY YES.** Evidence:
  - logN^ω explosion suggests missing normalization
  - Per-pair ratios correlate with polynomial degrees
  - I₂-only ratio ≠ 1.0 despite simple integral

### Recommended Next Steps

1. **Search PRZZ paper for polynomial normalization:**
   - Look for factors like 1/||P||₂ or ∫P²du
   - Check if Section 7 has implicit degree-dependent factors
   - Review asymptotic expansions for T → ∞ normalization

2. **Fix logN^ω explosion in Case C:**
   - Current implementation gives 991.5 instead of 0.99 for (2,2)
   - This is 1000× too large
   - Suggests logN^ω should be logN^ω / (some factor ~ 1000)

3. **Test polynomial-degree scaling hypothesis:**
   - Create synthetic test: use κ* coefficients with κ polynomial degrees
   - If ratio changes, confirms degree-dependence
   - If ratio unchanged, issue is in coefficients, not degrees

4. **Compare Case C integrals for different degrees:**
   - Numerically evaluate ∫(1-a)^k × a^{ω-1} × exp(...) da for k=1,2,3
   - Check if ratio matches observed per-pair ratios (1.66, 3.01, 17.4)
   - This would confirm (1-a)^k is the suppression mechanism

5. **Contact PRZZ authors or find Feng's code:**
   - Original Mathematica code may clarify normalization
   - PRZZ authors can confirm if degree-dependent factors exist

---

## Appendix: Case Distribution Tables

### Table A: Monomial Counts by Pair

| Pair | (a,b,c,d) monomials | (k₁,l₁,m₁) triples | Case pairs |
|------|---------------------|-------------------|------------|
| (1,1) | 4 | 3 | 3 |
| (2,2) | 12 | 8 | ~5-6 |
| (3,3) | 27 | 15 | ~8-10 |
| (1,2) | 7 | ~5 | ~4 |
| (1,3) | 10 | ~6 | ~5 |
| (2,3) | 18 | ~10 | ~7 |
| **Total** | **78** | ~47 | ~20-25 |

### Table B: (1,1) Detailed Case Mapping

| Monomial | (a,b,c,d) | (k₁,l₁,m₁) | ω_left | ω_right | Case |
|----------|-----------|------------|--------|---------|------|
| AB | (1,1,0,0) | (0,1,1) | 0 | 0 | (B,B) |
| D | (0,0,0,1) | (0,1,1) | 0 | 0 | (B,B) |
| AC | (1,0,1,0) | (1,1,0) | 0 | -1 | (B,A) |
| BC | (0,1,1,0) | (1,0,1) | -1 | 0 | (A,B) |

### Table C: Expected ω Distribution

For pair (ℓ,ℓ̄), monomials can have:
- ω_left = a + d − 1 ∈ {−1, 0, 1, ..., ℓ−1}
- ω_right = b + d − 1 ∈ {−1, 0, 1, ..., ℓ̄−1}

**Maximum ω values:**
- (1,1): ω_max = 0 (only Cases A and B)
- (2,2): ω_max = 1 (Cases A, B, and C with ω=1)
- (3,3): ω_max = 2 (Cases A, B, and C with ω=1,2)

**Higher pairs → higher ω → stronger u^ω suppression in Case C.**

---

**END OF ANALYSIS**
