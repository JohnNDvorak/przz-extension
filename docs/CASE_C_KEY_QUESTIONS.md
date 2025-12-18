# Case C Key Questions: Direct Answers

**Date:** 2025-12-17
**Purpose:** Answer the specific questions about Case C kernel structure

---

## Question 1: What is K_ω(u;R)?

### Definition (PRZZ TeX 2370-2375)

```
K_ω(u; R) = [u^ω / (ω-1)!] × ∫₀¹ P((1-a)u) × a^(ω-1) × exp(Rθua) da
```

where:
- **u** = normalized log argument = log(N/n)/log N ∈ [0,1]
- **ω** = ℓ - 1 (piece index minus 1)
- **R** = shift parameter (1.3036 for κ benchmark)
- **θ** = 4/7 (PRZZ standard value)
- **a** = auxiliary integration variable ∈ [0,1]
- **P** = polynomial (P₂ or P₃ for K=3)

### Specific Forms

**For ω=1 (P₂ kernel):**
```
K₁(u; R) = u × ∫₀¹ P₂((1-a)u) × exp(Rθua) da
```

**For ω=2 (P₃ kernel):**
```
K₂(u; R) = u² × ∫₀¹ P₃((1-a)u) × a × exp(Rθua) da
```

### How It Depends on R

The R-dependence enters through the **exponential factor exp(Rθua)** inside the integral:

1. **Larger R** → stronger exponential growth with a
2. **Shifts weight** toward larger a values
3. **Competes with** the (1-a) argument compression

**Numerical effect:**
- K/P ratio at R=1.3036: P₂ ~0.47, P₃ ~0.24
- K/P ratio at R=1.1167: P₂ ~0.46, P₃ ~0.23
- Higher R gives slightly larger kernel (but still < P)

---

## Question 2: When Does Case C Apply?

### Classification Rule

**For d=1:** ω = ℓ - 1

| Piece | ℓ | ω | Case | Formula |
|-------|---|---|------|---------|
| P₁ | 1 | 0 | **B** | P₁(u) directly |
| P₂ | 2 | 1 | **C** | K₁(u; R) kernel |
| P₃ | 3 | 2 | **C** | K₂(u; R) kernel |

### Cross-Term Pairs

| Pair | Cases | Which pieces use kernels? |
|------|-------|---------------------------|
| (1,1) | B×B | Neither (both P₁) |
| (1,2) | B×C | Right side only (P₂) |
| (1,3) | B×C | Right side only (P₃) |
| (2,2) | C×C | Both sides (P₂ × P₂) |
| (2,3) | C×C | Both sides (P₂ × P₃) |
| (3,3) | C×C | Both sides (P₃ × P₃) |

### Does (1,1) use Case B and (2,2) use Case C?

**Yes:**
- **(1,1) is pure Case B:** Both sides use P₁, which has ω=0
- **(2,2) is pure Case C:** Both sides use P₂, which has ω=1

**This is correct and matches PRZZ.**

### Where Case C Applies in Computation

Case C applies when evaluating the **F_d factors** (PRZZ TeX 2370-2384) that encode how the polynomial P enters the n-sum.

For ω > 0, instead of:
```
F_d ∝ P(u)
```

PRZZ has:
```
F_d ∝ ∫₀¹ P((1-a)u) × a^(ω-1) × exp(-αa·log(N/n)) da
```

This is the **kernel structure** built into the formula from the start, not a post-hoc correction.

---

## Question 3: How Does Case C Affect the Ratio?

### The Ratio Reversal Problem

**Target (from PRZZ):**
- const_κ / const_κ* = 2.98 / 3.17 = **0.942** (κ < κ*)

**Naive ∫P²du gives:**
- sum_κ / sum_κ* = 3.38 / 1.97 = **1.71** (κ > κ*)

**Case C I₂ gives:**
- I₂_κ / I₂_κ* = 0.71 / 0.33 = **2.15** (still κ > κ*)

### Case C Alone Is Not Enough

**Tested:** Case C I₂ ratio is 2.17 (not the target 0.94)

**Why it fails:**
1. Case C **suppresses** both κ and κ* I₂ values
2. κ suppressed **more** due to larger polynomials (degree 3 vs 2)
3. But the ratio remains **in wrong direction** (κ > κ* instead of κ < κ*)

### Combined with Ψ Structure?

The full answer requires analyzing how Case C interacts with:

1. **Ψ combinatorial structure** — Many monomials with varying signs
2. **Derivative terms (I₁, I₃, I₄)** — Subtract from I₂, reducing κ more than κ*
3. **(1-u) power weights** — Suppress higher pairs differently

**Key insight from testing:**

| Component | Effect on ratio | Magnitude |
|-----------|----------------|-----------|
| Case C kernel | Reduces both, wrong direction | Ratio 2.15 |
| Derivative subtraction | Reduces κ more (correct direction) | ~40% of gap |
| Ψ missing monomials | Unknown | Could be large |
| Combined | **Unknown** | Needs full oracle |

**None of our partial implementations achieve the target ratio of 1.10** for full c, or 0.94 for the const component.

### What Would Make Ratio Reverse?

To get κ < κ*, need:
- κ contribution suppressed MORE than κ*
- Despite κ having larger ||P||

**Candidate mechanisms:**
1. Higher-order derivative terms (P''' for degree 3, P''' for degree 2)
2. Ψ negative coefficients that favor κ*
3. (1-u)^(ℓ₁+ℓ₂) weights suppressing high-ℓ pairs more
4. Case C + all of the above combined

**Current status:** No single effect is sufficient. Likely need all mechanisms working together.

---

## Question 4: The a-Integral Structure

### PRZZ Lines 2369-2384

**Full F_d formula for Case C (ω > 0):**

```
F_d(l, α, n) = W(d,l) × [(-1)^(1-ω) / (ω-1)!] × (log N)^ω × [log(N/n)/log N]^ω
              × ∫₀¹ P((1-a)·log(N/n)/log N) × a^(ω-1) × (N/n)^(-αa) da
```

### Structure Breakdown

1. **Prefactor:** W(d,l) × (-1)^(1-ω) / (ω-1)! × (log N)^ω
2. **Argument power:** [log(N/n)/log N]^ω = u^ω
3. **a-integral:**
   - **Polynomial argument:** P((1-a)u) — evaluated at compressed u
   - **a-weight:** a^(ω-1) — shifts weight distribution
   - **Exponential:** (N/n)^(-αa) ≈ exp(-Rθua) in critical strip

### How It Integrates with Ψ Formula

The Ψ formula (HANDOFF Section 14) gives:

```
Ψ_{ℓ,ℓ̄}(A,B,C,D) = Σ_p C(ℓ,p)C(ℓ̄,p)p! × (D-C²)^p × (A-C)^(ℓ-p) × (B-C)^(ℓ̄-p)
```

where A, B, C, D are integrals of different derivative structures.

**Connection:**
- For ω > 0, the **polynomial factor inside A, B, C, D** should use the Case C kernel, not raw P
- The a-integral affects **how P enters each monomial term**
- Each (a,b,c,d) monomial gets its own F_d factor encoding

**Example for (2,2) pair:**
- Ψ_{2,2} has 12 monomials
- 4 involve D (mixed derivatives)
- 8 involve various A, B, C powers
- Each monomial's integrand should have P₂ replaced by K₁

### Integration Stage

**Critical insight:** The a-integral is **part of defining what F_d means**, not a post-hoc correction.

The full structure should be:
```
I_{ℓ₁,ℓ₂} = ∫∫ Ψ_{ℓ₁,ℓ₂}(A[F_d1], B[F_d2], C[F_d], D[F_d]) × Q² × exp(2Rt) du dt
```

where F_d already contains the Case C kernel for ω > 0.

**Our current implementation:**
```
I_{ℓ₁,ℓ₂} = ∫∫ Ψ_{ℓ₁,ℓ₂}(A[P], B[P], C[P], D[P]) × Q² × exp(2Rt) du dt
```

uses raw P everywhere, missing the Case C structure.

### Variable Rescaling (TeX 2309)

PRZZ uses x → x log N, which means:
- Derivatives w.r.t. x become derivatives w.r.t. (x log N)
- Changes the relative scaling of derivative terms
- May affect how u and formal variables interact

**Status:** We haven't verified this rescaling is correctly implemented.

---

## Success Criterion: Does Case C Contribute to Ratio Reversal?

### Answer: Partially, But Not Enough

**What Case C contributes:**
1. ✓ Reduces larger polynomials more (P₃ > P₂ > P₁)
2. ✓ Introduces R-dependence in ω > 0 pieces
3. ✓ Changes derivative structure via extra Rθa·P term

**What Case C fails to do:**
1. ✗ Ratio still wrong direction (2.15 vs target 0.94)
2. ✗ Makes c magnitude worse (1.95 → 0.57, need 1.95 → 2.14)
3. ✗ Worsens R-sensitivity (12.9% → 14.8%, target 10.3%)

### The Central Mystery

Even with Case C fully implemented:
- We still get const_κ > const_κ* (ratio 1.8-2.2 depending on approach)
- PRZZ needs const_κ < const_κ* (ratio 0.94)

**Possible explanations:**
1. **Missing Ψ monomials** for ℓ > 1 create different polynomial sensitivities
2. **Derivative subtraction** compounds with Case C in non-obvious ways
3. **Variable structure** may be wrong (our summed x₁+x₂ vs PRZZ's actual structure)
4. **Constant extraction** may have different normalization than we assume

### Recommended Resolution

**Build the complete PRZZ Section 7 oracle** that:
1. Uses exact F_d formulas (Case A/B/C built in)
2. Applies Euler-Maclaurin for n-sum → integral
3. Includes full Ψ structure with all monomials
4. Extracts constant following PRZZ's exact procedure

This is the only way to definitively answer whether Case C + Ψ + derivatives together achieve the ratio reversal.

**Status:** Partial implementation in `src/przz_section7_oracle.py` (incomplete).

---

## Summary Table

| Question | Answer | Implementation Status |
|----------|--------|----------------------|
| What is K_ω? | u^ω × ∫P((1-a)u)a^(ω-1)exp(Rθua)da | ✓ Implemented, tested |
| When does Case C apply? | ω > 0 (P₂, P₃ for K=3) | ✓ Correctly classified |
| Does (1,1)=B, (2,2)=C? | Yes | ✓ Correct |
| Does Case C fix ratio? | No — ratio 2.15 vs target 0.94 | ✗ Wrong direction |
| Case C + Ψ combined? | Unknown — needs full oracle | ⚠️ Partial impl. |
| How a-integral fits Ψ? | F_d replaces P in each monomial | ⚠️ Not integrated |

---

## Files Created

1. **`docs/CASE_C_ANALYSIS.md`** — Comprehensive 14-section analysis
2. **`docs/CASE_C_KEY_QUESTIONS.md`** — This document (direct answers)

Both documents synthesize findings from:
- `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/src/case_c_kernel.py`
- `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/docs/CASE_C_FINDINGS.md`
- `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/docs/HANDOFF_SUMMARY.md`
- PRZZ TeX lines 2301-2384
