# PRZZ Case C Kernel: Deep Analysis and Implications

**Created:** 2025-12-17
**Purpose:** Comprehensive analysis of PRZZ's Case C kernel structure and its role in the ratio reversal problem
**Status:** Complete synthesis of existing findings + new insights

---

## Executive Summary

### What is Case C?

For mollifier pieces with ω > 0 (where ω = ℓ - 1 for d=1), PRZZ replaces the direct polynomial evaluation P(u) with a **kernel function** K_ω(u; R) that involves an auxiliary integral over a ∈ [0,1].

### Key Finding: Case C is Not a Post-Hoc Correction

The kernel is **part of the integrand definition** in PRZZ's formula. It cannot be applied as a multiplicative correction factor—it must be integrated at the correct stage of the calculation.

### Critical Paradox

- **We need:** c to INCREASE from 1.95 → 2.14 (+9.6%)
- **Case C does:** c DECREASES from 1.95 → 0.57 (-70%)

This reveals that Case C alone cannot explain the gap, and suggests fundamental structural differences between our implementation and PRZZ's formula.

---

## 1. PRZZ Formulation (TeX Lines 2301-2384)

### 1.1 The ω Parameter

**Definition (TeX 2302-2304):**
```
ω(d, l) = 1×l₁ + 2×l₂ + ⋯ + d×l_d - 1
```

**For d=1:**
```
ω = l₁ - 1 = ℓ - 1
```

where ℓ is the piece index (number of Λ convolutions + 1).

**Case classification:**
- **Case A:** ω = -1 (Conrey piece, ℓ=0) — derivative structure
- **Case B:** ω = 0 (ℓ=1) — direct polynomial evaluation P(u)
- **Case C:** ω > 0 (ℓ≥2) — kernel with auxiliary a-integral

### 1.2 K=3 Piece Mapping

| Polynomial | Piece ℓ | ω = ℓ-1 | Case | Structure |
|------------|---------|---------|------|-----------|
| P₁ | 1 | 0 | B | P₁(u) directly |
| P₂ | 2 | 1 | C | K₁(u; R) kernel |
| P₃ | 3 | 2 | C | K₂(u; R) kernel |

### 1.3 Cross-Term Cases

| Pair | ω₁ | ω₂ | Cases | a-integrals |
|------|----|----|-------|-------------|
| (1,1) | 0 | 0 | B×B | 0 |
| (1,2) | 0 | 1 | B×C | 1 |
| (1,3) | 0 | 2 | B×C | 1 |
| (2,2) | 1 | 1 | C×C | 2 |
| (2,3) | 1 | 2 | C×C | 2 |
| (3,3) | 2 | 2 | C×C | 2 |

**Key insight:** 5 out of 6 K=3 pairs involve at least one Case C polynomial.

---

## 2. The Case C Kernel Formula

### 2.1 Exact PRZZ Formula (TeX 2370-2375)

For ω > 0, the F_d factor is:

```
F_d(l, α, n) = W(d,l) × [(-1)^(1-ω) / (ω-1)!] × (log N)^ω × [log(N/n)/log N]^ω
              × ∫₀¹ P_ℓ((1-a)·log(N/n)/log N) × a^(ω-1) × (N/n)^(-αa) da
```

**Simplifying for our normalized u = log(N/n)/log N:**

```
F_d(l, α, n) ∝ u^ω × ∫₀¹ P((1-a)u) × a^(ω-1) × exp(-Rθua) da
```

(The exp factor comes from (N/n)^(-αa) with α ≈ -Rθ in the critical strip.)

### 2.2 Kernel Definition

We define the Case C kernel as:

```
K_ω(u; R) = [u^ω / (ω-1)!] × ∫₀¹ P((1-a)u) × a^(ω-1) × exp(Rθua) da
```

**For specific ω values:**
- **ω=1 (P₂):** K₁(u; R) = u × ∫₀¹ P₂((1-a)u) × exp(Rθua) da
- **ω=2 (P₃):** K₂(u; R) = (u²/1!) × ∫₀¹ P₃((1-a)u) × a × exp(Rθua) da

### 2.3 R-Dependence Structure

The kernel has **explicit R-dependence** through:
1. **Exponential factor:** exp(Rθua) inside the integral
2. **Argument compression:** P((1-a)u) with a ∈ [0,1] evaluates P at smaller arguments
3. **Interaction:** The exp factor and a-power weight compete

**Key property:** K_ω(u; R) → smaller values than P(u) due to argument compression, but exp(Rθua) provides some compensation.

---

## 3. Implementation and Numerical Results

### 3.1 Kernel Evaluation

Implemented in `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/src/case_c_kernel.py`:

```python
def compute_case_c_kernel(
    P_eval: Callable,
    u_grid: np.ndarray,
    omega: int,
    R: float,
    theta: float,
    n_quad_a: int = 30
) -> np.ndarray:
    """Compute K_ω(u; R) using Gauss-Legendre quadrature."""
```

**Validation tests:**
- For constant P=1, R=0: K₁ = 1.0, K₂ = 0.5 ✓ (exact)
- Vectorized vs loop implementations match to machine precision ✓

### 3.2 Kernel vs Raw Polynomial (RMS comparison)

Using PRZZ κ polynomials at both benchmarks:

| Polynomial | ω | K/P ratio (R₁=1.3036) | K/P ratio (R₂=1.1167) | K ratio R₁/R₂ |
|------------|---|----------------------|----------------------|---------------|
| P₁ | 0 | 1.000 | 1.000 | 1.000 |
| P₂ | 1 | **0.470** | **0.455** | 1.033 |
| P₃ | 2 | **0.240** | **0.226** | 1.063 |

**Key observations:**
1. Case C kernels are ~50-75% smaller than raw polynomials
2. Higher ω → stronger suppression
3. Kernels have intrinsic R-dependence (ratio R₁/R₂ ≈ 1.03-1.06)

### 3.3 Case C Kernel Derivative

For derivative terms (I₁, I₃, I₄), we need dK_ω/darg|_{arg=u}.

**Formula (product rule):**
```
K'_ω(u; R) = [ω·u^(ω-1) / (ω-1)!] × ∫₀¹ P((1-a)u) × a^(ω-1) × exp(Rθua) da
           + [u^ω / (ω-1)!] × ∫₀¹ [(1-a)P'((1-a)u) + Rθa·P((1-a)u)] × a^(ω-1) × exp(Rθua) da
```

The second integral includes an extra Rθa·P term that doesn't appear in the naive derivative.

---

## 4. Effect on I₂ Terms (No Derivatives)

### 4.1 I₂ Structure

For the decoupled term (no mixed derivatives):
```
I₂ = (1/θ) × ∫∫ [P_left × P_right]_(u) × Q(t)² × exp(2Rt) du dt
```

With Case C, P is replaced by K for ω > 0 polynomials.

### 4.2 I₂ Values with Case C

Using κ polynomials at R₁=1.3036:

| Pair | Raw I₂ | Case C I₂ | Reduction |
|------|--------|-----------|-----------|
| (1,1) | 0.225 | 0.225 | 0% (both Case B) |
| (1,2) | 0.239 | 0.108 | -55% |
| (1,3) | 0.257 | 0.057 | -78% |
| (2,2) | 0.227 | 0.050 | -78% |
| (2,3) | 0.230 | 0.027 | -88% |
| (3,3) | 0.233 | 0.013 | -94% |
| **Total** | **1.19** | **0.71** | **-40%** |

**Critical finding:** Case C reduces I₂ total by 40%, making c smaller (wrong direction for gap closure).

### 4.3 R-Sensitivity of I₂

| Pair | Raw I₂ R-sens | Case C I₂ R-sens |
|------|---------------|------------------|
| (1,1) | 12.90% | 12.90% |
| (1,2) | 12.90% | 16.46% |
| (1,3) | 12.90% | 19.68% |
| (2,2) | 12.90% | 20.40% |
| (2,3) | 12.90% | 23.90% |
| (3,3) | 12.90% | 27.50% |
| **Total** | **12.90%** | **14.81%** |

**Problem:** Case C makes I₂ R-sensitivity WORSE (target is 10.29%, we get 14.81%).

---

## 5. Effect on Full Terms (Including Derivatives)

### 5.1 Full Case C Computation

Applied Case C to all terms (I₁, I₂, I₃, I₄) using kernel derivatives for I₁/I₃/I₄.

**Results at R₁=1.3036 (κ polynomials):**

| Pair | Raw c | Case C c | Change |
|------|-------|----------|--------|
| (1,1) | +0.442 | +0.317 | -28% |
| (1,2) | -0.201 | **+0.210** | **Sign flip!** |
| (1,3) | -0.218 | +0.003 | Near zero |
| (2,2) | +1.261 | +0.043 | **-97%** |
| (2,3) | +0.586 | +0.001 | -99.7% |
| (3,3) | +0.080 | +0.00001 | -99.98% |
| **Total** | **1.950** | **0.574** | **-71%** |

### 5.2 (2,2) Pair Breakdown

| Term | Raw | Case C | Ratio |
|------|-----|--------|-------|
| I₁ | +0.971 | +0.008 | 0.8% |
| I₂ | +0.227 | +0.050 | 22% |
| I₃ | +0.031 | -0.008 | Sign flip |
| I₄ | +0.031 | -0.008 | Sign flip |
| **Total** | **+1.261** | **+0.043** | **3.4%** |

**Observations:**
1. I₁ (double derivative) most suppressed: K'×K' compounds the reduction
2. I₃/I₄ flip sign but remain small
3. Overall pair nearly vanishes under Case C

### 5.3 Sign Flip in (1,2) Pair

The (1,2) pair flips from -0.201 to +0.210, which INCREASES c by ~0.41.

**Why this happens:**
- I₁ is negative in raw computation
- Case C derivative structure changes the balance
- Extra Rθa·P term in K' shifts the sign

**But:** The (2,2) collapse (-1.22) overwhelms the (1,2) improvement (+0.41).

---

## 6. The Ratio Reversal Problem

### 6.1 Target const Ratios

From HANDOFF_SUMMARY.md Section 15:

```
c = const × ∫Q²e^{2Rt}dt
```

| Component | κ (R=1.3036) | κ* (R=1.1167) | Ratio |
|-----------|--------------|---------------|-------|
| t-integral | 0.7163 | 0.6117 | 1.171 |
| **const (target)** | **2.984** | **3.168** | **0.942** |
| c total | 2.137 | 1.938 | 1.103 |

**Key requirement:** const_κ / const_κ* ≈ 0.94 (κ < κ*)

### 6.2 Naive ∫P²du Gives Wrong Direction

Our naive (1/θ) × Σ ∫P_i·P_j du gives:
- κ sum: 3.38
- κ* sum: 1.97
- **Ratio: 1.71** (κ > κ*)

**THE RATIOS ARE BACKWARDS!**

PRZZ needs κ < κ* (ratio 0.94), but naive integral gives κ > κ* (ratio 1.71).

### 6.3 Case C Effect on Ratio

Testing Case C I₂ only (from Section 4.2):

| Benchmark | Polynomials | Case C I₂ total | Per t-integral |
|-----------|-------------|-----------------|----------------|
| κ | R=1.3036 | 0.71 | 0.71 / 0.716 = 0.991 |
| κ* | R=1.1167 | 0.33 | 0.33 / 0.612 = 0.539 |
| **Ratio** | | | **0.991 / 0.539 = 1.84** |

Wait, this is confusing. Let me recalculate correctly:

| Benchmark | Polynomials | I₂ (Case C) | Needs ratio |
|-----------|-------------|-------------|-------------|
| κ | degree 3 P₂/P₃ | 0.71 | 0.94 ÷ 1.17 = 0.80? |
| κ* | degree 2 P₂/P₃ | 0.33 | Need smaller I₂ |

Actually, Case C I₂ ratio is:
```
0.71 / 0.33 = 2.15
```

**Still wrong direction!** We need κ < κ*, but Case C gives κ > κ* for I₂.

---

## 7. Why Case C Cannot Fix the Gap Alone

### 7.1 The Triple Constraint Problem

1. **Gap magnitude:** We need c to increase from 1.95 → 2.14 (+9.6%)
2. **Gap direction:** Case C makes c decrease from 1.95 → 0.57 (-71%)
3. **R-sensitivity:** Case C worsens R-sensitivity (12.9% → 14.8%)

**Conclusion:** Case C pushes in the wrong direction for both magnitude and R-sensitivity.

### 7.2 Negative Correlation Paradox

PRZZ's formula must have **negative correlation** between ||P|| and contribution:
- κ polynomials have larger ||P|| (degree 3)
- κ* polynomials have smaller ||P|| (degree 2)
- But PRZZ needs const_κ < const_κ* (smaller contribution from larger polynomials!)

**This is opposite to naive ∫P²du.**

Case C provides some negative correlation (larger P → more suppression via (1-a) argument), but:
1. The effect is too strong (71% reduction, not 9%)
2. The ratio still goes wrong direction for κ vs κ*

### 7.3 What Case C Does Well

Despite failing to fix the gap, Case C correctly:
1. **Introduces R-dependence** in pairs with ω > 0
2. **Reduces higher pairs more** (ω=2 suppressed more than ω=1)
3. **Changes derivative structure** through the extra Rθa·P term

These are all features PRZZ's formula should have, but the magnitudes are wrong.

---

## 8. Structural Insights

### 8.1 Case C is Not a Multiplicative Correction

Early attempts tried applying Case C as a multiplicative factor:
```
c_corrected = Σ (correction_ratio × raw_pair_value)
```

**This fails because:**
1. Case C changes how P interacts with Q, exp, and derivative operators
2. The kernel structure creates new cross-terms that aren't in raw computation
3. The a-integral is integrated over the same domain as u, creating correlations

**Proper implementation:** Case C must be part of the integrand definition from the start.

### 8.2 The Variable Rescaling (TeX 2309)

PRZZ uses the change of variable:
```
x → x log N
```

This rescaling affects:
1. How polynomial arguments are formed
2. Relative magnitudes of derivative terms
3. The u-integral domain interpretation

**Question:** Does our implementation correctly account for this rescaling?

### 8.3 The Mirror Combination (TeX 1502-1511)

PRZZ analyzes I(α,β) + T^(-α-β)I(-β,-α) together before extracting constants.

**Potential issue:** We expand each piece separately, then sum. PRZZ may combine analytically first, creating cross-terms we miss.

---

## 9. Candidate Explanations for Ratio Reversal

Given that Case C alone cannot explain the ratio reversal, what else could cause κ < κ*?

### 9.1 Ψ Combinatorial Structure

PRZZ uses a full Ψ expansion with many monomials (HANDOFF Section 14):

| Pair | Ψ monomials | Our DSL terms | Coverage |
|------|-------------|---------------|----------|
| (1,1) | 4 | 4 | 100% ✓ |
| (2,2) | 12 | 4 | 33% |
| (3,3) | 27 | 4 | 15% |

**Missing monomials might have:**
1. Different polynomial degree sensitivities
2. Sign patterns that favor κ* over κ
3. R-dependence that compensates for Case C

### 9.2 Derivative Term Subtraction

The I₁, I₃, I₄ derivative terms SUBTRACT from I₂:
- κ polynomials (degree 3) have larger derivatives
- κ* polynomials (degree 2) have smaller derivatives
- Net: κ reduced more than κ*, creating negative correlation

**But:** Our current implementation shows this effect is too small (Track 3 analysis in HANDOFF shows derivatives contribute ~40% of gap, not 100%).

### 9.3 (1-u) Power Weights

The (1-u)^(ℓ₁+ℓ₂) weights suppress higher pairs near u=1:
- (1,1): (1-u)²
- (2,2): (1-u)⁴
- (3,3): (1-u)⁶

κ has more weight in (2,2) and (3,3) pairs (larger P₂, P₃), which get suppressed more.

**Effect magnitude:** Needs quantitative analysis.

### 9.4 Case C + Ψ + Derivatives Combined

The ratio reversal likely requires ALL THREE effects:
1. **Case C** suppresses higher ω pairs
2. **Ψ structure** redistributes contributions across monomials
3. **Derivative terms** subtract more from κ than κ*

No single effect is sufficient.

---

## 10. Implications for Implementation

### 10.1 What We Know is Correct

From validation in mollifier_profiles.py and existing tests:
1. **Case B (ω=0):** P₁(u) evaluation is correct ✓
2. **Case C kernel K_ω(u; R):** Implemented and validated against simple test cases ✓
3. **Case C derivative K'_ω(u; R):** Formula derived correctly ✓
4. **R-dependence:** Kernels have correct R-scaling behavior ✓

### 10.2 What Remains Uncertain

1. **Integration stage:** When/how to apply Case C in the full DSL
2. **Variable structure:** Does (2,2) use summed arguments x₁+x₂ or something else?
3. **Ψ monomials:** How to evaluate the missing (a,b,c,d) terms for ℓ > 1
4. **Constant extraction:** How PRZZ separates const from t-integral

### 10.3 Path Forward: The Section 7 Oracle

The most direct path is to implement PRZZ's Section 7 formulas exactly:

1. **Use F_d factors** (TeX 2370-2384) with Case A/B/C built in
2. **Apply Euler-Maclaurin** (TeX 2391+) for n-sum → u-integral conversion
3. **Include S(z) zeta ratios** for proper pole structure
4. **Combine mirror terms analytically** before constant extraction

This bypasses the DSL entirely and computes c directly from PRZZ's formulas.

**Status:** Partial implementation exists in `src/przz_section7_oracle.py` but incomplete.

---

## 11. Key PRZZ TeX Line References

| Lines | Content | Implementation Status |
|-------|---------|----------------------|
| 2301-2310 | ω definition, variable rescaling | ✓ Understood |
| 2305-2323 | Case A formula | ⚠️ Not used (no ℓ=0) |
| 2324-2335 | Case B formula | ✓ Implemented |
| 2336-2362 | Case C derivation | ✓ Understood |
| 2370-2375 | F_d for Case C (α side) | ✓ Implemented in kernel |
| 2379-2383 | F_d for Case C (β side) | ✓ Symmetric to α |
| 2391+ | Euler-Maclaurin lemma | ⚠️ Needs integration |
| 1502-1511 | Mirror combination | ⚠️ Check analytic combination |

---

## 12. Summary: Case C's Role in κ Computation

### What Case C Does

1. **Introduces R-dependence** in P₂ and P₃ pieces through exp(Rθua) factor
2. **Suppresses higher ω polynomials** via (1-a) argument compression
3. **Modifies derivative structure** by adding Rθa·P term in K'_ω
4. **Couples u and a integrals** creating correlation structure

### What Case C Does Not Do

1. **Close the 9.6% gap** — it makes c 71% smaller, not larger
2. **Fix R-sensitivity** — it makes I₂ sensitivity worse (12.9% → 14.8%)
3. **Reverse the ratio** — I₂ ratio remains κ > κ* (2.15 instead of target 0.94)

### The Central Mystery

PRZZ's formula produces:
- const_κ = 2.98, const_κ* = 3.17 → ratio 0.94 (κ < κ*)

Naive integrals give:
- sum_κ = 3.38, sum_κ* = 1.97 → ratio 1.71 (κ > κ*)

Case C I₂ gives:
- I₂_κ = 0.71, I₂_κ* = 0.33 → ratio 2.15 (still κ > κ*)

**The ratio is backwards in all our approaches.**

### Recommended Next Action

**Build the complete PRZZ Section 7 oracle** to compute c directly from first principles, bypassing our DSL assumptions entirely. This is the only way to definitively determine:
1. What const actually equals in PRZZ's formula
2. Why the ratio reverses
3. What structural pieces we're missing

---

## 13. Files and Test Coverage

### Implementation Files

| File | Purpose | Status |
|------|---------|--------|
| `src/case_c_kernel.py` | K_ω(u; R) and K'_ω(u; R) | ✓ Complete |
| `src/mollifier_profiles.py` | Profile generators with Case B/C | ✓ Complete |
| `src/test_case_c_derivatives.py` | Full term computation with Case C | ✓ Complete |

### Documentation Files

| File | Purpose |
|------|---------|
| `docs/CASE_C_FINDINGS.md` | Original findings (pre-synthesis) |
| `docs/OMEGA_CASE_MAPPING.md` | Piece ↔ ω mapping |
| `docs/CASE_C_ANALYSIS.md` | This document (comprehensive) |

### Test Results

All kernel validation tests pass:
- Constant polynomial tests ✓
- Vectorization tests ✓
- R-dependence tests ✓
- Derivative formula tests ✓

But full integration tests show Case C makes gap worse.

---

## 14. Conclusions

1. **Case C is mathematically correct** as implemented in `case_c_kernel.py`
2. **Case C cannot fix our gap** because it pushes c in the wrong direction
3. **The ratio reversal remains unexplained** by Case C alone
4. **Multiple structural differences** likely compound to create the observed gap
5. **Section 7 oracle is the path forward** to resolve these mysteries

Case C is a necessary component of PRZZ's formula, but it is not sufficient to explain the difference between our computation and PRZZ's targets. The full formula requires Case C + correct Ψ structure + proper derivative handling + correct variable structure.
