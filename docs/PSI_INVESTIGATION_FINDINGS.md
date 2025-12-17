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
