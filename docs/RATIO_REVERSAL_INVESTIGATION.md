# Ratio Reversal Investigation — Sign Patterns in Ψ

**Date**: 2025-12-17
**Investigator**: Analysis of PRZZ κ computation ratio problem
**Status**: Investigation in progress

---

## Problem Statement

The PRZZ κ computation requires:
```
c = const × ∫Q²e^{2Rt}dt
```

With the constraint:
```
const_κ / const_κ* ≈ 0.94  (κ has SMALLER const than κ*)
```

But our naive formula gives:
```
const_κ / const_κ* ≈ 1.71  (κ has LARGER const than κ*)
```

**The ratios are in OPPOSITE directions!**

---

## The Full Picture

| Component | κ (R=1.3036) | κ* (R=1.1167) | Ratio |
|-----------|--------------|---------------|-------|
| t-integral | 0.7163 | 0.6117 | 1.171 |
| const (target) | 2.984 | 3.168 | 0.942 |
| Combined c | 2.137 | 1.938 | 1.103 ✓ |

The t-integral ratio (1.171) is understood and validated. The mystery is **why const has ratio 0.94**.

---

## Why This Is Surprising

The κ polynomials have:
- P₂(x) = 1.048x + 1.320x² - 0.940x³  (degree 3)
- P₃(x) = 0.523x - 0.687x² - 0.050x³  (degree 3)
- Higher L² norms
- Larger derivative magnitudes

The κ* polynomials have:
- P₂(x) = 1.217x - 0.217x²  (degree 2 only!)
- P₃(x) = -0.007x + 0.007x²  (degree 2 only!)
- Smaller L² norms
- Lower derivative magnitudes

**Naive expectation**: Bigger polynomials → bigger ∫P² → larger const → κ should be LARGER.

**PRZZ reality**: Bigger polynomials → SMALLER const → κ is actually SMALLER.

This implies PRZZ has a **negative feedback mechanism** where larger polynomials produce more **subtractive corrections**.

---

## Hypothesis: Ψ Sign Patterns

### The Combinatorial Structure

For pair (ℓ, ℓ̄):
```
Ψ_{ℓ,ℓ̄}(A,B,C,D) = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × (D-C²)^p × (A-C)^{ℓ-p} × (B-C)^{ℓ̄-p}
```

Where:
- A = ζ'/ζ(1+α+s) — "x-derivative block"
- B = ζ'/ζ(1+β+u) — "y-derivative block"
- C = ζ'/ζ(1+s+u) — "base block"
- D = (ζ'/ζ)'(1+s+u) — "mixed derivative block"

This expands to monomials A^a × B^b × C^c × D^d with **both positive and negative integer coefficients**.

### Example: (1,1) Pair

```
Ψ_{1,1} = AB - AC - BC + D
```

- **Positive**: AB (+1), D (+1)
- **Negative**: AC (-1), BC (-1)
- Net balance: 0 (perfectly balanced)

For (1,1), 50% of monomials are positive, 50% are negative.

---

## Key Questions

### Question 1: Fraction of Negative Monomials

Do higher pairs have MORE negative monomials?

| Pair | Total Monomials | Expected Negative Fraction |
|------|-----------------|---------------------------|
| (1,1) | 4 | 50% (validated) |
| (2,2) | 12 | ? |
| (3,3) | 27 | ? |

**Prediction**: If negative fraction increases with pair degree, higher pairs contribute more subtraction.

### Question 2: Coefficient Magnitude Balance

For (2,2), what is the sum of coefficients?

- Sum of all positive coefficients: ?
- Sum of absolute values of negative coefficients: ?
- Ratio |negative| / |positive|: ?

**Prediction**: If |negative| > |positive|, net contribution is reduced by cancellation.

### Question 3: Derivative Order Correlation

Do negative monomials involve HIGH A, B exponents?

For example, in (2,2):
- A²B² has high derivative order (a+b=4)
- AC has medium derivative order (a+b=1)
- C⁴ has low derivative order (a+b=0)

**Prediction**: If negative monomials have high a, b exponents, they involve higher derivatives P^(k)(u) which scale with polynomial degree.

### Question 4: Polynomial Degree Scaling

How do negative monomial contributions scale with polynomial degree?

For a monomial with exponents (a, b):
- Contributes a factor P₁^(a)(u) × P₂^(b)(u)
- For κ: P₂'''(u) = derivative of x³ term (non-zero)
- For κ*: P₂'''(u) = derivative of x² term (ZERO!)

**Prediction**: Negative monomials with high a, b become LARGER for κ (higher degree) but SMALLER or ZERO for κ*.

This would flip the ratio!

---

## Supporting Evidence

### Cross-Integral Negativity

From handoff findings:
- Cross-integrals (1,3) and (2,3) have **NEGATIVE** ∫P_i·P_j values
- P₃ changes sign on [0,1]

For κ polynomials:
- P₃ is degree 3, more complex sign structure
- More pairs involve P₃

For κ* polynomials:
- P₃ is degree 2, simpler structure
- Less sign variation

This asymmetry could amplify the negative contribution difference.

### I₃, I₄ Prefactor

The I₃ and I₄ terms have prefactor **-1/θ** (subtractive).

These involve:
- I₃: ∫ dF/dx × (1-u)^{ℓ₁}
- I₄: ∫ dF/dy × (1-u)^{ℓ₂}

For higher ℓ values:
- Weights (1-u)^ℓ become stronger
- Derivative magnitudes increase with polynomial degree
- Net subtractive contribution grows

Could this be related to Ψ negative monomials?

---

## Mechanism Sketch

If the hypothesis is correct:

1. **Ψ expansion produces monomials with negative coefficients**
   - For (2,2): some of the 12 monomials are negative
   - For (3,3): some of the 27 monomials are negative

2. **Negative monomials involve high A, B exponents**
   - These correspond to high-order derivatives
   - Example: -2A²BC involves P₁''(u)² × P₂'(u) × C

3. **High-order derivatives scale with polynomial degree**
   - κ: P₂'''(u) ≠ 0 (degree 3 has x³)
   - κ*: P₂'''(u) = 0 (degree 2 has no x³)
   - Similarly P₃^(3)(u) ≠ 0 for κ, = 0 for κ*

4. **Negative contributions are larger for κ than κ***
   - κ: Large negative terms from high derivatives
   - κ*: Small or zero negative terms (derivatives vanish)
   - Net effect: const_κ reduced more than const_κ*

5. **Result: const_κ < const_κ***
   - Ratio ≈ 0.94 instead of naive 1.71
   - Direction reverses!

---

## Investigation Tools

### Files Created

1. **`src/analyze_psi_signs.py`**
   - Prints sign pattern analysis for all K=3 pairs
   - Shows top positive and negative monomials
   - Computes coefficient balance

2. **`src/compute_sign_statistics.py`**
   - Detailed statistics: fraction negative, sum magnitudes
   - Groups by derivative order (high a+b vs low)
   - Identifies which monomials dominate

3. **`docs/SIGN_PATTERN_ANALYSIS.md`**
   - Complete analysis framework
   - Hypothesis statement
   - Validation criteria

### Running the Analysis

To investigate, run:
```bash
cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
python src/compute_sign_statistics.py > results/sign_statistics.txt
```

This will output:
- Per-pair sign fractions
- Coefficient magnitude sums
- Derivative order breakdown
- Detailed (2,2) monomial list

---

## Expected Findings

If sign patterns explain the ratio reversal:

### Pattern 1: Increasing Negative Fraction
```
(1,1): 50% negative
(2,2): >50% negative
(3,3): >60% negative
```

### Pattern 2: Magnitude Dominance
```
(2,2): |negative| / |positive| > 1.0
(3,3): |negative| / |positive| > 1.2
```

### Pattern 3: High-Derivative Correlation
```
Negative monomials with a+b ≥ 2: dominant contribution
Positive monomials with a+b < 2: smaller contribution
```

### Pattern 4: Cross-Pair Balance
```
(1,3): Net balance negative
(2,3): Net balance negative
```

---

## Validation Tests

### Test 1: Modified Polynomials
Use κ polynomial **degrees** with κ* coefficient **magnitudes**:
- If ratio depends on degree, this should give ratio ≈ 1.71
- If ratio depends on magnitudes, this should give ratio ≈ 0.94

### Test 2: Individual Monomial Evaluation
For (2,2), evaluate each of the 12 monomials:
```python
for each monomial (k1, k2, l1, m1):
    value = evaluate_monomial_integral(k1, k2, l1, m1, polynomials)
```

Check if negative monomials scale with degree.

### Test 3: Degree Truncation
Truncate P₂, P₃ to degree 2 for κ benchmark:
- If hypothesis is correct, const should INCREASE (toward κ* value)
- Ratio should move toward 1.0

---

## Alternative Explanations

If sign patterns do NOT explain it, other mechanisms:

### Alternative 1: Case C Kernel
PRZZ uses a-integral kernel for ω > 0:
```
∫₀¹ P((1-a)u) × a^{ω-1} × exp(...) da
```

This could dampen high-degree contributions.

### Alternative 2: (1-u) Weight Distribution
Higher pairs have stronger (1-u)^{ℓ₁+ℓ₂} weights, which suppress near u=1. If polynomials have different behavior near u=1 vs u=0, this could create asymmetry.

### Alternative 3: logN Normalization
PRZZ formulas have (logN)^ω factors. If normalization is polynomial-degree-dependent, this could flip the ratio.

### Alternative 4: Missing Euler-Maclaurin Terms
PRZZ uses Euler-Maclaurin to convert sums → integrals. Missing correction terms could be degree-dependent.

---

## Success Criteria

### Hypothesis CONFIRMED if:
1. Negative monomials dominate for (2,2), (3,3)
2. Sum |negative| > Sum |positive| for high pairs
3. Negative monomials have high a, b exponents
4. Individual evaluation shows scaling with degree
5. Modified polynomial test supports degree-dependence

### Hypothesis REJECTED if:
1. Sign patterns are balanced (≈50% negative)
2. Coefficient magnitudes don't correlate with derivative order
3. Individual monomials don't show degree-scaling
4. Another mechanism (Case C, weights, etc.) is the cause

---

## Next Steps

### Immediate (Today)
1. Run `compute_sign_statistics.py` to get census
2. Analyze (2,2) coefficient balance
3. Check if high-derivative monomials are negative

### Short-term (This Week)
1. Implement individual monomial evaluation for (2,2)
2. Test with modified polynomials (degree vs magnitude)
3. Compare with κ and κ* polynomials

### Medium-term (If Confirmed)
1. Extend to (3,3) and cross-pairs
2. Derive theoretical formula for ratio
3. Validate against PRZZ Section 7 formulas

### Medium-term (If Rejected)
1. Investigate Case C kernel hypothesis
2. Study (1-u) weight effects with mpmath
3. Check for missing Euler-Maclaurin terms

---

## References

- **Handoff Summary**: Key finding of c = const × t-integral decomposition
  `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/docs/HANDOFF_SUMMARY.md`

- **Ψ Combinatorial Formula**: Implementation and validation
  `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/src/psi_combinatorial.py`

- **Ψ Investigation**: Previous findings about monomial structure
  `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/docs/PSI_INVESTIGATION_FINDINGS.md`

- **PRZZ TeX**: Original paper formulas
  Lines 1551-1628 (I₁-I₅ formulas)
  Lines 2567-2598 (κ and κ* polynomials)

---

## File Inventory

### Analysis Tools
- `src/analyze_psi_signs.py` — Sign pattern printer
- `src/compute_sign_statistics.py` — Detailed statistics

### Combinatorial Framework
- `src/psi_combinatorial.py` — Ψ monomial expansion
- `src/psi_block_configs.py` — p-sum representation
- `src/psi_monomial_expansion.py` — p-config to monomials

### Documentation
- `docs/SIGN_PATTERN_ANALYSIS.md` — Framework document
- `docs/RATIO_REVERSAL_INVESTIGATION.md` — This document
- `docs/HANDOFF_SUMMARY.md` — Context and background

---

## Summary

The ratio reversal problem is the **key unsolved mystery** in PRZZ reproduction:
- We need const_κ < const_κ* (ratio 0.94)
- We get const_κ > const_κ* (ratio 1.71)

**Sign patterns in Ψ monomials are a leading candidate** for explaining this:
1. Negative coefficients create subtractive terms
2. High A, B exponents → high derivatives → scale with polynomial degree
3. κ (higher degree) → larger negative contributions → smaller const
4. κ* (lower degree) → smaller negative contributions → larger const

If confirmed, this would explain the reversal and unlock the correct formula structure.

The investigation tools are ready. **Running the analysis is the next critical step.**
