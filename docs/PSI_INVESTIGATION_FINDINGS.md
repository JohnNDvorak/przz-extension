# Ψ Combinatorial Structure Investigation Findings

**Date**: 2025-12-17
**Status**: Partial validation, open questions remain

---

## Summary

Investigation of the Ψ combinatorial structure from PRZZ revealed:

1. **Validated**: (1,1) monomial expansion matches oracle perfectly
2. **Discovery**: PRZZ uses SUMMED P arguments, not SEPARATE
3. **Issue**: DSL I-term values don't match oracle for (2,2)
4. **Gap**: Computed c = 1.95 vs target c = 2.137 (91.3%)

---

## Ψ Combinatorial Formula

The main term contribution for pair (ℓ, ℓ̄) is:

```
c_{ℓ,ℓ̄} = ∫∫ Ψ_{ℓ,ℓ̄}(A,B,C,D) × F₀ du dt
```

where:
```
Ψ_{ℓ,ℓ̄} = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × (D-C²)^p × (A-C)^{ℓ-p} × (B-C)^{ℓ̄-p}
```

And:
- A = ζ'/ζ(1+α+s) → "singleton z-block" (x-derivative structure)
- B = ζ'/ζ(1+β+u) → "singleton w-block" (y-derivative structure)
- C = ζ'/ζ(1+s+u) → "base block" (log-integrand at 0)
- D = (ζ'/ζ)'(1+s+u) → "paired block" (mixed xy structure)

---

## Validated Results

### (1,1) Pair: Perfect Match ✓

The Ψ_{1,1} expansion gives 4 monomials:
```
Ψ_{1,1} = AB + D - AC - BC
```

**Validation Result**:
- Monomial sum: 0.359159
- Oracle total: 0.359159
- Difference: 5.55e-17 ✓

The mapping to I-terms:
| Monomial | Coefficient | I-term | Value |
|----------|-------------|--------|-------|
| AB | +1 | I₁ | +0.426 |
| D | +1 | I₂ | +0.385 |
| AC | -1 | |I₃| | -0.226 |
| BC | -1 | |I₄| | -0.226 |

---

## (2,2) Investigation: Unresolved Issues

### Ψ_{2,2} Expansion

The formula gives 12 monomials:
```
Ψ_{2,2} = X²Y² + 4XYZ + 2Z²
```
where X = A-C, Y = B-C, Z = D-C².

Expanding gives:
```
+1 × A²B²
-2 × A²BC - 2 × AB²C
+1 × A²C² + 4 × ABC² + 1 × B²C²
-2 × AC³ - 2 × BC³ + 1 × C⁴
+4 × ABD - 4 × ACD - 4 × BCD
+2 × D²
```

### Key Structural Question: SUMMED vs SEPARATE

**PRZZ uses SUMMED P arguments**:
- P(u + x₁ + x₂) for ℓ=2, not P(u+x₁)×P(u+x₂)
- d²/dx₁dx₂[P(u+X)] = P''(u), not (P')²

This means:
- A² does NOT require 4 variables with d⁴/dx₁dx₂dy₁dy₂
- The derivative order is d²/dxdy for ALL pairs
- The ℓ₁, ℓ₂ indices determine WHICH polynomial, not how many derivatives

### Observed Discrepancies

**Oracle vs DSL for (2,2)**:
| Term | Oracle | DSL V2 | Match? |
|------|--------|--------|--------|
| I₁ | 1.17 | 3.88 | ✗ 3.3× |
| I₂ | 0.91 | 0.91 | ✓ |
| I₃ | -0.54 | +0.13 | ✗ Wrong sign! |
| I₄ | -0.54 | +0.13 | ✗ Wrong sign! |
| Total | 0.99 | 5.04 raw | ✗ |

The DSL applies a 1/4 normalization, giving _c22_norm = 1.26.

**Full c Breakdown**:
```
DSL c = 1.950 (target: 2.137, 91.3%)
```

---

## Computed Values

### Base Integrals (no derivatives)

| Structure | Value | Description |
|-----------|-------|-------------|
| (1,1) I₂ with P₁² | 0.385 | Validated |
| (2,2) I₂ with P₂² | 0.909 | OLD oracle |
| (2,2) with P₂⁴ (no weight) | 1.195 | SEPARATE structure |
| (2,2) with P₂⁴ + (1-u)⁴ | 0.009 | Too small |

### Separate P Factor Computation

Using d⁴/dx₁dx₂dy₁dy₂ with P(x₁+u)×P(x₂+u)×P(y₁+u)×P(y₂+u):
- A²B² term: 1.524
- This is 1.30× the old oracle's I₁

---

## Key Files Created

| File | Purpose |
|------|---------|
| `src/psi_combinatorial.py` | Validate monomial counts |
| `src/psi_block_configs.py` | p-sum representation |
| `src/psi_monomial_expansion.py` | Expand p-configs to monomials |
| `src/psi_monomial_evaluator.py` | Map monomials to I-terms |
| `src/psi_22_oracle.py` | Log-derivative approach |
| `src/psi_22_monomial_oracle.py` | Direct monomial evaluation |
| `src/psi_22_full_oracle.py` | Full multi-variable oracle |

---

## Open Questions

1. **Why doesn't DSL I₁ match oracle for (2,2)?**
   - DSL gives 3.88, oracle gives 1.17
   - Both use d²/dxdy structure
   - The difference may be in prefactor handling

2. **Why are DSL I₃/I₄ positive when oracle gives negative?**
   - This is a sign convention issue
   - May indicate structural bug in V2 DSL

3. **What is the correct interpretation of A² in Ψ?**
   - SUMMED case: A² = (d/dx)² evaluated once, not d²/dx²
   - But the combinatorial expansion produces A² terms
   - Need to trace back to PRZZ derivation

4. **Why does the full c gap persist (91.3%)?**
   - Could be structural formula issue
   - Could be polynomial transcription error
   - Could be missing normalization

---

## Recommendations

1. **Re-examine V2 DSL prefactor handling**
   - The I₃/I₄ sign issues suggest a bug
   - Compare chain rule expansion term-by-term with oracle

2. **Trace PRZZ derivation for (2,2) specifically**
   - Verify the d²/dxdy (not d⁴) interpretation
   - Check how ℓ₁+ℓ₂ affects the formula

3. **Cross-validate with Feng's original code**
   - PRZZ TeX mentions "matched Feng's code"
   - Finding this could resolve all ambiguities

4. **Consider polynomial optimization test**
   - If formula is correct but norm wrong, optimization should still find good polynomials
   - If formula is fundamentally wrong, optimization will fail

---

## Conclusion

The (1,1) validation proves the Ψ combinatorial framework is sound.
However, the extension to (2,2) reveals unresolved structural questions about:
- SUMMED vs SEPARATE P factor interpretation
- DSL vs oracle discrepancies
- The correct mapping of 12 monomials to I-terms

Further investigation needed to resolve the 9% c gap before proceeding with optimization.

---

## Block Evaluator Attempt (2025-12-17 Update)

### Goal

Implement GPT's guidance: treat A, B, C, D as scalar values at each (u,t) point, compute X=A-C, Y=B-C, Z=D-C², and sum via p-configs.

### Implementation

Created `src/psi_block_evaluator.py` with:
- `BlockValues` dataclass for A, B, C, D scalars
- `PsiBlockEvaluator` class computing blocks from log-derivatives of integrand
- p-config integration with Ψ_{ℓ,ℓ̄} = Σ_p C(ℓ,p)C(ℓ̄,p)p! × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}

### Results: FAILED

| Test | Expected | Computed | Ratio |
|------|----------|----------|-------|
| (1,1) XY term | -0.026 | +0.421 | Wrong sign! |
| (1,1) Z term | 0.385 | 0.002 | 192× too small |
| (1,1) Total | 0.359 | 0.423 | 1.18 |
| (2,2) Total | 0.99 | 10273 | 10391× |

### Root Causes Identified

1. **Wrong C definition**: Tried C = log(ξ₀) or C = common Q/exp part. But C should be a specific ζ'/ζ block from PRZZ's asymptotic analysis, not a log-derivative of the polynomial integrand.

2. **Weight structure varies by I-term**: The oracle uses different (1-u)^k weights:
   - I₁ (d²/dxdy): (1-u)² weight
   - I₂ (base): NO weight
   - I₃ (d/dx): (1-u)¹ weight
   - I₄ (d/dy): (1-u)¹ weight

   This means Ψ with uniform weight can't reproduce the I-term decomposition.

3. **I₁ mixes multiple contributions**: The oracle's I₁ = dF/dx + dF/dy + (1/θ)d²F/dxdy, not just the "AB" term. The prefactor chain rule creates terms that don't map cleanly to single Ψ monomials.

### I₁ Decomposition for (1,1)

```
I₁ = ∫dF/dx × (1-u)² + ∫dF/dy × (1-u)² + (1/θ)×∫d²F/dxdy × (1-u)²
   = 0.045 + 0.045 + 0.336
   = 0.426
```

The dF/dx and dF/dy terms (from prefactor derivative) contribute 21% of I₁!

### Key Insight

The blocks A, B, C, D in PRZZ are **zeta-function quantities** from asymptotic analysis:
- A = ζ'/ζ(1+α+s)
- B = ζ'/ζ(1+β+u)
- C = ζ'/ζ(1+s+u)
- D = (ζ'/ζ)'(1+s+u)

These are NOT simple log-derivatives of the polynomial integrand. The Ψ formula expresses a deep identity about zeta-function structure that our polynomial approximation captures implicitly through the I₁-I₄ derivative extraction, but not through explicit block computation.

### Two-Benchmark Status

| Benchmark | Computed c | Target c | Factor |
|-----------|-----------|----------|--------|
| κ (R=1.3036) | 1.950 | 2.137 | 1.096 |
| κ* (R=1.1167) | 0.823 | 1.939 | 2.356 |

**c ratio (κ/κ*)**: 2.37 computed vs 1.10 target

The factor ratio is 0.47, confirming a **structural issue** (not global normalization).

### Conclusion

The block-based Ψ approach **cannot be implemented** without:
1. The exact PRZZ definitions of A, B, C, D in terms of our polynomial structure
2. Understanding how the weight structure (1-u)^k varies across monomials
3. Resolving the prefactor mixing in I₁

**The current I₁-I₄ oracle IS the correct computation** for our polynomial approximation. The issue is that this computation gives different c ratios for different polynomial sets (κ vs κ*), suggesting either:
- Missing polynomial-dependent normalization in PRZZ
- Transcription error in κ* polynomial coefficients
- A fundamental limitation of the polynomial approximation

---

## V1 vs V2 DSL Structure Discovery (2025-12-17 Update)

### Key Finding: Multi-Variable Structure Required for Cross-Pairs

**V1 (Multi-variable) DSL:**
- Uses (ℓ₁ + ℓ₂) variables with derivatives d^{ℓ₁}/dx₁...dx_{ℓ₁} × d^{ℓ₂}/dy₁...dy_{ℓ₂}
- P arguments: P_ℓ(u + x₁ + ... + x_ℓ) with SUMMED variables
- For (1,2): P₁(u+x₁) × P₂(u+y₁+y₂) with d/dx₁ d/dy₁ d/dy₂
- Extracts P_{ℓ₁}^{(ℓ₁)}(u) × P_{ℓ₂}^{(ℓ₂)}(u) (ℓ-th derivatives)
- Total c = 1.950 ← Close to target!

**V2 (Single-variable) DSL:**
- Uses 2 variables (x, y) for ALL pairs
- P arguments: P_ℓ(u+x) and P_ℓ̄(u+y) with single variables
- For (1,2): P₁(u+x) × P₂(u+y) with d/dx d/dy
- Extracts P_{ℓ₁}'(u) × P_{ℓ₂}'(u) (first derivatives only)
- Total c = -1.646 ← WRONG (negative!)

### Derivative Order Analysis

For pair (ℓ₁, ℓ₂), the derivative orders are:
| Pair | V1 Derivative Order | V2 Derivative Order |
|------|---------------------|---------------------|
| (1,1) | 2 (d/dx₁ d/dy₁) | 2 (d/dx d/dy) |
| (2,2) | 4 (d²/dx₁dx₂ d²/dy₁dy₂) | 2 (d/dx d/dy) |
| (3,3) | 6 (d³/dx₁dx₂dx₃ d³/dy₁dy₂dy₃) | 2 (d/dx d/dy) |
| (1,2) | 3 (d/dx₁ d²/dy₁dy₂) | 2 (d/dx d/dy) |
| (1,3) | 4 (d/dx₁ d³/dy₁dy₂dy₃) | 2 (d/dx d/dy) |
| (2,3) | 5 (d²/dx₁dx₂ d³/dy₁dy₂dy₃) | 2 (d/dx d/dy) |

For diagonal pairs (ℓ,ℓ), V1 and V2 can give similar results because:
- V1: SUMMED variables P_ℓ(u + x₁ + ... + x_ℓ)
- d^ℓ/dx₁...dx_ℓ [P_ℓ(u+X)] = P_ℓ^{(ℓ)}(u) at X=0
- V2: Single variable P_ℓ(u+x) with d/dx
- But V2 only extracts P_ℓ'(u), not P_ℓ^{(ℓ)}(u)

The V2 approach was validated only for (2,2), where it matches the oracle.
For cross-pairs (ℓ₁ ≠ ℓ₂), V2 gives completely wrong results.

### Polynomial Derivative Magnitudes for (1,2)

Simplified u-integral comparison (P₁' × P₂^{(k)} × (1-u)):
| Structure | Integral Value |
|-----------|----------------|
| V1 (P₁' × P₂'') | 0.149 |
| V2 (P₁' × P₂') | 0.686 |
| Ratio V1/V2 | 0.22 |

V1-style gives ~4.6× smaller magnitude due to using P₂'' instead of P₂'.

### Per-Pair Comparison

| Pair | V1 Raw | V1 Norm | V2 Raw | V2 Norm |
|------|--------|---------|--------|---------|
| (1,1) | 0.442 | 0.442 | 0.347 | 0.347 |
| (2,2) | 5.044 | 1.261 | 0.955 | 0.239 |
| (3,3) | 2.871 | 0.080 | 0.035 | 0.001 |
| (1,2) | -0.201 | -0.201 | -2.202 | -2.202 |
| (1,3) | -0.654 | -0.218 | -0.003 | -0.001 |
| (2,3) | 3.516 | 0.586 | -0.175 | -0.029 |
| **Total** | - | **1.950** | - | **-1.646** |

### Conclusion

1. **V1 multi-variable structure is correct** for cross-pairs
2. **V2 single-variable structure fails** for cross-pairs
3. The V2 approach from GPT's guidance was only valid for diagonal pairs
4. The 9% gap (1.950 vs 2.137) remains unexplained
5. The two-benchmark test shows κ needs factor 1.09, κ* needs factor 2.07

### Remaining Questions

1. Why does the gap differ between κ and κ* (1.09 vs 2.07)?
2. Is there a polynomial-degree-dependent normalization in PRZZ?
3. Could there be transcription errors in the κ* polynomial coefficients?

### Files Created

| File | Purpose |
|------|---------|
| `src/przz_generalized_iterm_evaluator.py` | Generalized I-term evaluator (validates pair-dependent weights) |

---

## Deep Dive Analysis: V1 vs V2 vs Oracle (2025-12-17 Continued)

### Key Structural Comparison

**For (2,2) pair specifically:**
| Source | I₁ | I₂ | I₃ | I₄ | Total |
|--------|-----|-----|-----|-----|-------|
| Oracle | +1.169 | +0.909 | -0.544 | -0.544 | +0.989 |
| V2 DSL | +1.135 | +0.909 | -0.544 | -0.544 | +0.955 |
| V1 DSL | +3.884 | +0.909 | +0.126 | +0.126 | +5.044 |

**Observations:**
- Oracle ≈ V2 for (2,2) - both use 2-variable d²/dxdy structure
- V1 is WRONG for (2,2) by 5× (I₁ too large, I₃/I₄ wrong sign)
- I₂ matches exactly (no derivatives involved)

### The Paradox

Despite V1 being wrong for (2,2), the OVERALL c values are:
- V1: c = 1.950 (91% of target 2.137) ← Closer to target!
- V2: c = -1.646 (negative, completely wrong)

**Why V2 fails overall:**
- Cross-pairs like (1,2) give -2.202 (huge negative)
- V1 (1,2) gives -0.201 (reasonable negative)
- The V2 cross-pair contributions overwhelm the correct diagonal terms

### Root Cause Analysis

**V1 structure (per pair ℓ₁, ℓ₂):**
- Variables: ℓ₁ + ℓ₂ total (x₁..x_{ℓ₁}, y₁..y_{ℓ₂})
- P arguments: SUMMED (P_ℓ(u + x₁ + ... + x_ℓ))
- Derivative: d^{ℓ₁}/dx₁...dx_{ℓ₁} × d^{ℓ₂}/dy₁...dy_{ℓ₂}
- Extracts: P_{ℓ₁}^{(ℓ₁)}(u) × P_{ℓ₂}^{(ℓ₂)}(u)
- Weight: (1-u)^{ℓ₁+ℓ₂}

**V2 structure (all pairs):**
- Variables: 2 (x, y)
- P arguments: SINGLE (P_ℓ(u + x), P_ℓ(u + y))
- Derivative: d²/dxdy
- Extracts: P_{ℓ₁}'(u) × P_{ℓ₂}'(u)
- Weight: (1-u)^{(ℓ₁-1)+(ℓ₂-1)}

**Neither is fully correct:**
- V2 matches oracle for (2,2) but fails for cross-pairs
- V1 fails for (2,2) but gives better overall c
- This suggests compensating errors in V1

### Two-Benchmark Deep Analysis

**κ benchmark (R=1.3036):**
- Target c = 2.137
- Computed c = 1.950
- Factor needed: 1.096

**κ* benchmark (R=1.1167):**
- Target c = 1.938
- Computed c = 0.823
- Factor needed: 2.355

**Polynomial structure differences:**
| Polynomial | κ degree | κ* degree |
|------------|----------|-----------|
| P₂ | 3 | 2 |
| P₃ | 3 | 2 |
| Q | 5 (odd powers) | 1 (linear) |

The κ* polynomials have simpler structure, leading to:
- P₂^(2)(u) = non-zero for κ (has x² term)
- P₂^(2)(u) = constant for κ* (only up to x²)
- P₃^(3)(u) = non-zero for κ (has x³ term)
- P₃^(3)(u) = 0 for κ* (only up to x², no x³!)

This explains why V1 structure fails more for κ*: extracting P^(3) from a degree-2 polynomial gives zero.

### Conclusions

1. **The 9% gap in V1 (κ benchmark) is not due to a simple missing factor**
2. **The V1 structure is fundamentally inconsistent with the oracle**
3. **Neither V1 nor V2 correctly implements PRZZ for all pairs**
4. **A correct implementation likely requires:**
   - Understanding PRZZ Section 7 F_d factor structure
   - Different formula structure for diagonal vs cross-pairs
   - Polynomial-degree-dependent handling

### Recommended Next Steps

1. **Study PRZZ Section 7 more carefully** - the F_d factors use Case A/B/C classification
2. **Implement a hybrid approach** - use V2 for diagonal, investigate correct cross-pair formula
3. **Contact PRZZ authors** - ask for clarification on the c computation formula
4. **Check Feng's original code** - PRZZ TeX mentions matching Feng's numerics

---

## Section 7 F_d Mapping Implementation (2025-12-17 Continued)

### Goal

Map each Ψ monomial (a,b,c,d) to its (k₁, l₁, m₁) triple for F_d wiring, then evaluate using PRZZ Section 7 Case A/B/C structure.

### Mapping Formulas (VALIDATED ✓)

From PRZZ Section 7, the mapping from (a,b,c,d) exponents to F_d indices is:
```
l₁ = a + d    (left derivative count: A's plus D's)
m₁ = b + d    (right derivative count: B's plus D's)
k₁ = c        (convolution index: C's)
ω_left = l₁ - 1
ω_right = m₁ - 1
```

Case classification (d=1):
- **Case A**: ω = -1 (l₁ = 0) → derivative form with 1/logN
- **Case B**: ω = 0 (l₁ = 1) → direct polynomial evaluation
- **Case C**: ω > 0 (l₁ > 1) → kernel integral with (logN)^ω factor

### K=3 Mapping Summary

| Pair | Monomials | Unique Triples | Case Pairs Distribution |
|------|-----------|----------------|-------------------------|
| (1,1) | 4 | 3 | A,B:1 \| B,A:1 \| B,B:2 |
| (2,2) | 12 | 8 | A,A:1 \| A,B:1 \| A,C:1 \| B,A:1 \| B,C:2 \| C,A:1 \| C,B:2 \| C,C:3 |
| (3,3) | 27 | 15 | A,A:1 \| A,B:1 \| A,C:2 \| B,A:1 \| B,B:2 \| B,C:4 \| C,A:2 \| C,B:4 \| C,C:10 |
| (1,2) | 7 | 5 | A,A:1 \| A,C:1 \| B,A:1 \| B,B:2 \| B,C:2 |
| (1,3) | 10 | 7 | A,A:1 \| A,B:1 \| A,C:1 \| B,A:1 \| B,B:2 \| B,C:4 |
| (2,3) | 18 | 11 | A,A:1 \| A,B:1 \| A,C:2 \| B,A:1 \| B,B:2 \| B,C:2 \| C,A:1 \| C,B:2 \| C,C:6 |

### (1,1) I-term Correspondence (VALIDATED ✓)

For (1,1), the 4 monomials map to 3 unique triples:
```
(0,1,1) Case B,B: +1×AB + +1×D  → I₁ + I₂ structure
(1,0,1) Case A,B: -1×BC          → I₄ structure
(1,1,0) Case B,A: -1×AC          → I₃ structure
```

This exactly matches the I-term oracle decomposition!

### Section 7 F_d Evaluator Implementation

Created `src/section7_fd_evaluator.py` that:
1. Uses `psi_fd_mapping.py` to get (k₁, l₁, m₁) triples
2. Evaluates F_d^{left} and F_d^{right} based on Case A/B/C
3. Combines with Ψ coefficients

### CRITICAL ISSUE: logN^ω Explosion in Case C

The PRZZ Case C formula:
```
F_d = W × (-1)^{1-ω}/(ω-1)! × (logN)^ω × u^ω × ∫₀¹ P((1-a)u) × a^{ω-1} × exp(...) da
```

With logT = 100 and logN = θ × logT ≈ 57:
- ω = 1: logN^1 ≈ 57
- ω = 2: logN^2 ≈ 3265

This causes Case C contributions to explode.

**Results with Section 7 F_d evaluator:**
| Pair | Computed | Expected |
|------|----------|----------|
| (1,1) | 0.289 | 0.359 |
| (2,2) | 991.5 | 0.99 |
| (3,3) | 35579.6 | 0.05 |

The logN^ω factors are not properly normalized in our finite implementation.

### Root Cause Analysis

The PRZZ formulas are **asymptotic** where logT → ∞. The (logN)^ω factors are balanced by:
1. 1/logN factors from Case A derivatives
2. 1/(logT)^k factors in the overall sum-to-integral conversion
3. Other asymptotic cancellations

The I-term oracle **avoids this issue** by working directly with finite integrals that implicitly absorb the normalization. It uses:
```python
darg_alpha_dx = theta * t  # No logN/logT explicitly
```

The F_d approach requires understanding how PRZZ normalizes across cases.

### Key Discovery: (k,l,m) Triple ≠ Direct Integral Mapping

**Critical insight**: Monomials with the SAME (k₁,l₁,m₁) triple can require DIFFERENT integral structures!

For (1,1):
- AB (a=1,b=1,c=0,d=0) → (0,1,1) Case B,B → I₁ structure (d²/dxdy)
- D (a=0,b=0,c=0,d=1) → (0,1,1) Case B,B → I₂ structure (base integral)

Both map to the SAME triple but have DIFFERENT integral formulas because:
- AB: derivative in x AND y
- D: no explicit derivative (the (ζ'/ζ)' structure is different)

The (k,l,m) mapping tells us the CASE classification, but the (a,b,c,d) structure determines the specific integral form.

### Files Created

| File | Purpose |
|------|---------|
| `src/psi_fd_mapping.py` | Maps Ψ (a,b,c,d) monomials to (k₁,l₁,m₁) triples with Case A/B/C |
| `src/section7_fd_evaluator.py` | Section 7 F_d evaluator (has logN^ω issue) |
| `src/generalized_monomial_evaluator.py` | Generalized monomial evaluator (heuristic, not validated) |

### Conclusions

1. **The mapping (a,b,c,d) → (k₁,l₁,m₁) is CORRECT and VALIDATED**
2. **The Case A/B/C classification matches PRZZ Section 7**
3. **The F_d evaluation has normalization issues** due to logN^ω factors
4. **The I-term oracle implicitly handles the normalization** by working with finite integrals
5. **A correct Section 7 implementation requires understanding PRZZ's asymptotic normalization**

### Path Forward

Two viable approaches:

**Option A: Hybrid approach**
- Use I-term oracle for diagonal pairs (validated)
- Investigate correct cross-pair formula separately
- Accept that this doesn't directly implement Section 7

**Option B: PRZZ Section 7 deep dive**
- Study how PRZZ normalizes F_d factors across cases
- Find the balancing factors that make c = O(1)
- Implement the complete asymptotic structure

Given the complexity, **Option A is recommended** for making progress on the κ optimization goal.

---

## Final Investigation Conclusions (2025-12-17)

### Two-Benchmark Gate Results

All implementation approaches were tested against the mandatory two-benchmark gate:

| Approach | κ Factor (R=1.3036) | κ* Factor (R=1.1167) | Factor Ratio | Pass? |
|----------|---------------------|----------------------|--------------|-------|
| V1 DSL | 1.0959 | 2.3558 | 0.465 | ✗ |
| V2 DSL | ~1.09 | ~2.07 | 0.53 | ✗ |
| I-term Oracle + Geom Mean | ~1.1 | ~2.2 | 0.51 | ✗ |
| Calibrated p-config | ~1.3 | ~3.7 | 0.35 | ✗ |

**Target**: Factor ratio ≈ 1.0 (both benchmarks need similar correction)

**Result**: All approaches have factor ratio ~0.35-0.53, failing the two-benchmark gate.

### Root Cause: Polynomial-Degree-Dependent Formula Behavior

The root cause is **STRUCTURAL**, not a coding bug:

1. **κ polynomials** (R=1.3036): P₂ and P₃ have degree 3
2. **κ* polynomials** (R=1.1167): P₂ and P₃ have degree 2 only

When extracting P^{(ℓ)}(u) in V1 structure:
- For κ: P₃^{(3)}(u) is non-zero (cubic has x³ term)
- For κ*: P₃^{(3)}(u) = 0 (quadratic has no x³ term!)

This explains why:
- κ is closer to target (9% gap)
- κ* is far from target (107% gap)
- The formula extracts different-order derivatives that vanish for lower-degree polynomials

### What Was Validated ✓

1. **Ψ → (k,l,m) mapping is CORRECT**
   - All K=3 pairs correctly mapped to F_d triples
   - Case A/B/C classification matches PRZZ Section 7
   - Created and validated `src/psi_fd_mapping.py`

2. **(1,1) pair is perfectly calibrated**
   - Constant block calibration: Ψ = 0.359159 = oracle.total (exact match)
   - I-term monomial evaluator: Difference < 1e-15

3. **Block algebra is correct for (1,1)**
   - X = A-C, Y = B-C, Z = D-C² expansion verified
   - p-config formula Ψ = XY + Z gives correct result

### What Remains Unresolved ✗

1. **Two-benchmark gate failure**
   - Neither approach improves both κ and κ* simultaneously
   - This is a MANDATORY requirement per project guidelines

2. **logN^ω normalization in Case C**
   - PRZZ asymptotic formulas have (logN)^ω factors
   - Our finite implementation cannot reproduce the balancing

3. **Cross-pair formula for higher degrees**
   - Constant blocks work for (1,1) only
   - (2,2)+ requires understanding how blocks vary with u,t

4. **Polynomial-degree-dependent normalization**
   - PRZZ may have normalization factors we're missing
   - This would explain the different factor requirements

### Files Created This Investigation

| File | Status | Purpose |
|------|--------|---------|
| `src/psi_fd_mapping.py` | ✓ Validated | Ψ → (k,l,m) mapping |
| `src/section7_fd_evaluator.py` | ✗ Has logN issue | F_d evaluation attempt |
| `src/section7_pconfig_engine.py` | ✗ Failed | p-config based evaluation |
| `src/calibrated_pconfig_engine.py` | Partial | Constant block calibration |
| `src/generalized_monomial_evaluator.py` | Heuristic | General monomial evaluation |

### Recommendations for Future Work

1. **Do NOT continue patching** - The two-benchmark failure is structural
2. **Study PRZZ Section 7 normalization** - Look for degree-dependent factors
3. **Contact PRZZ authors** - Ask for clarification on c formula and normalization
4. **Search for Feng's original code** - PRZZ TeX line 2566 mentions "matched Feng's code"
5. **Consider polynomial optimization directly** - Optimize polynomials to minimize c, bypassing formula

### Investigation Conclusion

**The Ψ combinatorial framework is mathematically correct, but our implementation cannot reproduce PRZZ's c values for different polynomial sets.**

The two-benchmark gate failure indicates either:
1. Missing polynomial-degree-dependent normalization in PRZZ
2. Incorrect transcription of κ* polynomial coefficients
3. Fundamental limitation of our finite-integral approach vs PRZZ's asymptotic formulas

**Further progress requires external information** (PRZZ authors, Feng's code, or additional PRZZ documentation) to resolve the formula interpretation ambiguity.

---

## Critical Polynomial Discrepancy Discovery (2025-12-17)

### User-Provided PRZZ Polynomials vs Original Codebase Polynomials

The user provided exact polynomial coefficients from PRZZ TeX lines 2567-2598:

**User's PRZZ κ polynomials (R=1.3036):**
```
P₁(x) = x + 0.138173 x(1-x) - 0.445606 x(1-x)² - 4.039834 x(1-x)³
P₂(x) = -0.101269 x + 3.571698 x² - 1.807283 x³
P₃(x) = 1.334025 x - 3.018815 x² + 1.133072 x³
Q(x) = 0.490068 + 0.509932(1-2x) = 1 - 1.019864 x  ← LINEAR!
```

**Original codebase κ polynomials (source unknown):**
```
P₁: tilde_coeffs = [0.261076, -1.071007, -0.236840, 0.260233]
P₂: coeffs = [0, 1.048274, 1.319912, -0.940058]
P₃: coeffs = [0, 0.522811, -0.686510, -0.049923]
Q: basis = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}  ← DEGREE 5!
```

### Critical Structural Difference: Q Polynomial Degree

| Source | Q Degree | Q Structure |
|--------|----------|-------------|
| User's PRZZ | 1 (linear) | Q(x) = 1 - 1.019864 x |
| Original codebase | 5 | Q(x) has (1-2x)¹, (1-2x)³, (1-2x)⁵ terms |

### Evaluation Results

| Polynomial Set | c Computed | Target c | Factor | κ Computed |
|----------------|------------|----------|--------|------------|
| User's PRZZ | 7.879 | 2.137 | 0.27 | -0.584 |
| Original codebase | 1.950 | 2.137 | 1.10 | 0.488 |

**The original (unknown source) polynomials give results 4× closer to target!**

### Per-Term Comparison

| Term | User's PRZZ | Original | Ratio |
|------|-------------|----------|-------|
| I1_22 | 10.55 | 3.88 | 2.7× |
| I1_33 | 16.41 | 2.86 | 5.7× |

The user's PRZZ polynomials have much larger derivative magnitudes.

### Possible Explanations

1. **Different PRZZ versions/sections**: The user's polynomials may be from a different section (e.g., theoretical analysis) vs numerical optimization results.

2. **Transcription source error**: The TeX lines 2567-2598 may not be the final optimized polynomials.

3. **Formula interpretation mismatch**: Our formula might be correct for the original polynomials but wrong for the user's PRZZ polynomials.

4. **Q polynomial plays critical role**: The degree-5 Q vs linear Q may account for most of the difference.

### Key Observation: Factorial Normalization Test

Created `src/test_factorial_normalization.py` which verified:
- Multi-variable extraction [x1x2y1y2] = 4 × [Z²W²] coefficient (for Z=x1+x2, W=y1+y2)
- The series engine correctly handles nilpotent variable expansions
- Factorial normalization 1/(ℓ!×ℓ̄!) appears to be applied correctly

**The factorial normalization is NOT the source of the discrepancy.**

### Recommendation

**URGENT**: Verify the polynomial source in PRZZ TeX:
1. Are lines 2567-2598 the final optimized polynomials for κ=0.417?
2. Is there another section with different polynomials?
3. What is the source of the original codebase polynomials?

The original codebase polynomials may be from Feng's original code or a different PRZZ run. Their provenance needs to be established.

### Files Created/Modified

| File | Purpose |
|------|---------|
| `data/przz_parameters.json` | Updated with user's PRZZ polynomials |
| `src/verify_przz_polynomials.py` | Polynomial verification script |
| `src/test_factorial_normalization.py` | Factorial normalization test |
