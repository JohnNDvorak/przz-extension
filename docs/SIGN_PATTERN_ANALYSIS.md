# Ψ Sign Pattern Analysis: Investigating the Ratio Reversal

**Date**: 2025-12-17
**Context**: Investigating why our formula gives const ratio 1.71 when PRZZ needs 0.94

---

## Executive Summary

The PRZZ κ computation has a critical **ratio reversal problem**:

- **Required**: const_κ / const_κ* ≈ 0.94 (κ has SMALLER const than κ*)
- **Our naive formula**: ratio ≈ 1.71 (κ has LARGER const than κ*)
- **Direction is OPPOSITE**: We get κ > κ* when we need κ < κ*

This analysis investigates whether **Ψ sign patterns** could explain the reversal.

---

## Background: The Decomposition

From handoff investigation, c decomposes as:
```
c = const × ∫Q²e^{2Rt}dt
```

Where:
- **t-integral ratio** (κ/κ*): 1.171 — R-dependent, captures exp(2Rt) scaling
- **const ratio** (κ/κ*): 0.942 — should be nearly R-independent
- **Combined**: 1.171 × 0.942 = 1.103 ✓ matches PRZZ c_κ/c_κ* target!

The t-integral is understood and validated. **The mystery is the const ratio.**

---

## The Naive Formula Problem

Our simplest formula (ignoring derivatives) is:
```
const = (1/θ) × Σ_{i,j} ∫₀¹ P_i(u) · P_j(u) du
```

With κ polynomials:
- Sum: 3.38

With κ* polynomials:
- Sum: 1.97

**Ratio**: 3.38 / 1.97 = **1.71** (WRONG DIRECTION!)

---

## Why This Matters

The κ polynomials have:
- P₂, P₃ with degree 3
- Higher derivative magnitudes
- Larger L² norms

The κ* polynomials have:
- P₂, P₃ with degree 2 only
- Simpler structure
- Smaller L² norms

**Naive expectation**: Bigger polynomials → bigger integrals → κ should be LARGER.

**PRZZ reality**: Bigger polynomials → SMALLER const → κ is actually SMALLER.

This suggests PRZZ has a **negative feedback mechanism** where larger polynomials produce more **subtractive corrections**.

---

## Hypothesis: Sign Pattern Interaction

### The Ψ Combinatorial Formula

For pair (ℓ, ℓ̄):
```
Ψ_{ℓ,ℓ̄}(A,B,C,D) = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × (D-C²)^p × (A-C)^{ℓ-p} × (B-C)^{ℓ̄-p}
```

This expands to monomials with **both positive and negative coefficients**.

### Sign Pattern by Pair

| Pair | Total Monomials | Positive | Negative | Pos Fraction |
|------|-----------------|----------|----------|--------------|
| (1,1) | 4 | 2 | 2 | 50% |
| (2,2) | 12 | ? | ? | ? |
| (3,3) | 27 | ? | ? | ? |

For (1,1):
```
Ψ_{1,1} = AB - AC - BC + D
```
- Positive: AB (+1), D (+1)
- Negative: AC (-1), BC (-1)
- **Perfectly balanced**: 2 positive, 2 negative

### Key Questions

1. **What fraction of monomials are negative for (2,2) and (3,3)?**
   - If higher pairs have MORE negative terms, this could reduce const for κ

2. **Do negative monomials contribute MORE for κ than κ*?**
   - κ uses P₂³, P₃³ (higher derivatives)
   - κ* uses P₂², P₃² (lower derivatives)
   - If negative terms involve higher-order structures, they could scale differently

3. **What is the magnitude balance?**
   - For (2,2): Sum of |positive coeffs| vs |negative coeffs|
   - This determines potential for differential cancellation

---

## Cross-Integral Evidence

From handoff findings:
- **(1,3) and (2,3) cross-integrals have NEGATIVE ∫P_i·P_j values**
- **P₃ changes sign on [0,1]**

This means:
- Naive ∫P² is always positive
- But ∫P₁·P₃ and ∫P₂·P₃ can be NEGATIVE

For κ polynomials:
- More pairs involve P₃ (degree 3)
- More opportunity for negative cross-integrals

For κ* polynomials:
- P₃ is degree 2, simpler structure
- Less sign variation

**Could this asymmetry explain the ratio reversal?**

---

## Detailed (2,2) Analysis

The Ψ_{2,2} expansion has 12 monomials:

### Expected Structure
```
Ψ_{2,2} = (A-C)²(B-C)² + 4(D-C²)(A-C)(B-C) + 2(D-C²)²
```

Expanding:
- **p=0 term**: (A-C)²(B-C)² contributes 9 monomials
- **p=1 term**: 4×(D-C²)(A-C)(B-C) contributes to 8 monomials
- **p=2 term**: 2×(D-C²)² contributes 3 monomials

Many combine, giving 12 total with various signs.

### Critical Coefficient Analysis

For (2,2), we need to compute:
- Sum of all positive coefficients
- Sum of absolute values of negative coefficients
- Net balance

If negative coefficients dominate in magnitude, and if these negative terms involve higher-derivative structures that scale with polynomial degree, we could get:
- κ (higher degree): Large negative contribution → reduces const
- κ* (lower degree): Smaller negative contribution → less reduction

This would flip the ratio!

---

## Proposed Investigation Steps

### 1. Sign Pattern Census
Run `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/src/analyze_psi_signs.py` to compute:
- For each pair (ℓ, ℓ̄): fraction of negative monomials
- Sum of |positive coeffs| vs |negative coeffs|
- Identify which monomials have largest negative coefficients

### 2. Monomial-by-Monomial Evaluation
For (2,2), evaluate each of the 12 monomials separately:
- Compute ∫∫ mono(A,B,C,D) × F₀ du dt for each monomial
- Identify which monomials contribute positively vs negatively
- Check if negative monomials scale differently with polynomial degree

### 3. Derivative Order Correlation
Test hypothesis: Negative monomials involve higher A/B powers
- Count: How many negative monomials have high A or B exponents?
- Since A = ζ'/ζ(1+α+s) involves x-derivatives, high A power → high derivative order
- Higher derivatives scale with polynomial degree more strongly

### 4. Cross-Pair Asymmetry
Analyze (1,3) and (2,3):
- These pairs have negative ∫P_i·P_j
- Check if Ψ sign patterns amplify this for κ vs κ*

---

## Potential Mechanisms

### Mechanism 1: Derivative Term Subtraction
- I₁ involves d²/dxdy (mixed derivatives)
- I₃, I₄ have prefactor -1/θ (subtractive)
- If higher-degree polynomials produce larger I₃, I₄ magnitudes, they reduce const more

### Mechanism 2: (1-u) Weight Distribution
The weights are:
- I₁: (1-u)^{ℓ₁+ℓ₂}
- I₃: (1-u)^{ℓ₁}
- I₄: (1-u)^{ℓ₂}

For κ pairs:
- (3,3) has I₁ weight (1-u)⁶ — strongly suppresses near u=1
- Could shift contribution toward regions where P derivatives are larger

### Mechanism 3: Case C Kernel Modification
For ω > 0 pieces (Case C), PRZZ uses a-integral kernel:
```
∫₀¹ P((1-a)u) × a^{ω-1} × exp(...) da
```

This could dampen high-degree polynomial contributions more than low-degree.

### Mechanism 4: Ψ Sign Pattern + Polynomial Degree
If negative Ψ monomials involve terms like:
- A²B² (high derivatives in both x and y)
- ACD or BCD (mixed derivative + C subtraction)

These could scale as:
- κ (deg 3): P₂''·P₃'' → large magnitude
- κ* (deg 2): P₂''·P₃'' → P₃''=0 (quadratic has no x³!)

This would create asymmetric cancellation.

---

## Expected Findings

If sign patterns explain the ratio reversal, we should see:

1. **(2,2) and (3,3) have more negative monomials than (1,1)**
   - Target: >50% negative for higher pairs

2. **Negative monomials have larger coefficient magnitudes**
   - Sum |negative| > Sum |positive|

3. **Negative monomials involve high A, B powers**
   - These scale with P^(k)(u) for k > 1

4. **Cross-pairs (1,3), (2,3) have strongly negative Ψ balance**
   - Amplifies the P₃ sign-change effect

If these patterns hold, and if they interact with polynomial degree as hypothesized, the sign structure could naturally produce:
```
const_κ < const_κ*  (ratio 0.94)
```

even though:
```
||P_κ|| > ||P_κ*||  (naive expectation reversed)
```

---

## Validation Path

To confirm this hypothesis:

1. **Compute sign pattern statistics** (using analyze_psi_signs.py)
2. **Evaluate (2,2) monomials individually** to see which contribute negatively
3. **Test with modified polynomials**: Use κ degree structure with κ* magnitudes
   - If sign patterns are the cause, ratio should depend on degree, not magnitude
4. **Compare with PRZZ Section 7 formulas** to see if F_d factors encode this structure

---

## Files to Create/Modify

- `src/analyze_psi_signs.py` — Sign pattern census (CREATED)
- `src/psi_22_monomial_evaluator.py` — Individual monomial evaluation
- `docs/SIGN_PATTERN_FINDINGS.md` — Results document

---

## Success Criteria

This hypothesis is **confirmed** if:
- Negative Ψ monomials dominate for high pairs
- Their contribution scales with polynomial degree
- This produces const_κ / const_κ* ≈ 0.94

This hypothesis is **rejected** if:
- Sign patterns are balanced or random
- No clear correlation with polynomial degree
- Other mechanism (Case C, weights, etc.) must be the cause

---

## References

- **Handoff Summary**: `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/docs/HANDOFF_SUMMARY.md`
- **Ψ Combinatorial**: `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/src/psi_combinatorial.py`
- **Ψ Investigation**: `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/docs/PSI_INVESTIGATION_FINDINGS.md`

---

## Next Steps for Investigation

**Immediate**:
1. Run `python src/analyze_psi_signs.py` to get sign pattern census
2. Compute sum of |positive| vs |negative| coefficients for (2,2)
3. Identify which monomials have highest negative coefficients

**Follow-up**:
1. Evaluate each (2,2) monomial with κ polynomials
2. Evaluate same monomials with κ* polynomials
3. Check if negative monomials scale differently

**Deep dive**:
1. Trace PRZZ Section 7 to see if F_d formulas encode sign structure
2. Test polynomial degree vs magnitude separation
3. Compare with Feng's original code if found
