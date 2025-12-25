# Handoff Document: Session 5 - December 17, 2025

## Executive Summary

**TWO-C Ψ expansion successfully implemented.** The `psi_separated_c.py` module correctly generates:
- (1,1): exactly 4 monomials with C_α×C_β cancellation ✓
- (2,2): 12 monomials (vs 4 in old DSL) ✓
- All K=3 pairs have correct monomial counts ✓

**However**, the PRE-MIRROR → POST-MIRROR integration has a fundamental gap. The clean-path F_d evaluation doesn't correctly map to the I₁-I₄ structure.

## Key Accomplishments

### 1. TWO-C Ψ Expansion (`psi_separated_c.py`)

Implemented corrected block structure:
```python
X = A - C_β   (NOT A - C)
Y = B - C_α   (NOT B - C)
Z = D - C_α × C_β   (NOT D - C²)
```

**Monomial counts (all verified):**
| Pair | Monomials | (1-u) Exponents |
|------|-----------|-----------------|
| (1,1) | 4 | [2, 0] |
| (2,2) | 12 | [4, 2, 0] |
| (3,3) | 27 | [6, 4, 2, 0] |
| (1,2) | 7 | [3, 1] |
| (1,3) | 10 | [4, 2] |
| (2,3) | 18 | [5, 3, 1] |

### 2. (1,1) Monomial Verification

```
Ψ_{1,1} = AB - A×C_α - B×C_β + D
```

4 monomials with coefficients:
- D (+1): l1=1, m1=1 → Case (B,B)
- AB (+1): l1=1, m1=1 → Case (B,B)
- -A×C_α (-1): l1=1, m1=0 → Case (B,A)
- -B×C_β (-1): l1=0, m1=1 → Case (A,B)

**Key insight**: C_α×C_β terms CANCEL, giving exactly 4 terms matching I₁-I₄ structure.

### 3. F_d Cases Implementation

Implemented in `section7_clean_evaluator.py`:
- **Case A** (ω=-1, l=0): `F = α × P(u)`
- **Case B** (ω=0, l=1): `F = P(u)`
- **Case C** (ω≥1, l≥2): Kernel integral with correct asymptotic form
  ```
  K_ω(u;R) = ∫₀¹ P((1-a)u) × a^{ω-1} × exp(R×θ×u×a) da
  ```

## Critical Findings

### 1. I₂ Ratio Analysis

For (1,1) pair:
| Component | κ | κ* | Ratio |
|-----------|---|---|-------|
| ∫P₁²du | 0.307 | 0.300 | 1.024 |
| ∫Q²exp(2Rt)dt | 0.716 | 0.612 | 1.171 |
| **I₂** | **0.385** | **0.321** | **1.199** |

**I₂ ratio = 1.20**, but target c ratio = 1.10.

The derivative terms (I₁, I₃, I₄) should bring this DOWN to 1.10.

### 2. PRE-MIRROR vs POST-MIRROR Gap

**POST-MIRROR oracle (correct):**
```
I₂ = (1/θ) × ∫P₁²du × ∫Q²exp(2Rt)dt = 0.385
```

**PRE-MIRROR clean-path (current implementation):**
```
I_{1,d}(0,0) = 0.614  (at α=β=0)
```

**Problem**: Clean-path gives 2× ∫P₁²du, not (1/θ)×∫P₁²du×∫Q²exp(2Rt)dt

The D and AB monomials both evaluate to ∫P₁²du in the current implementation, but they represent DIFFERENT derivative structures in the Taylor expansion!

### 3. Mirror Term Issues

With exp(2R) ≈ 13.56 as the mirror factor:
- At α=β=0: I_d = (1+exp(2R))×I_{1,d} ≈ 9 (WAY too big)
- At α=β=-R: I_d becomes NEGATIVE (mirror dominates)

This suggests the mirror application is incorrect or the PRE-MIRROR integrals are wrong.

## Root Cause Analysis

The fundamental issue is that the clean-path F_d×F_d evaluation doesn't correctly map to PRZZ's I₁-I₄ structure:

1. **D monomial** should represent: mixed second derivative ∂²/∂x∂y contribution
2. **AB monomial** should represent: product of first derivatives (∂/∂x)(∂/∂y) contribution

In the Taylor expansion of P(x+u)P(y+u):
```
∂²(PP)/∂x∂y = P×(∂²P/∂x∂y) + (∂P/∂x)(∂P/∂y) + (∂P/∂y)(∂P/∂x) + (∂²P/∂x∂y)×P
            = 0 + P'×P' + P'×P' + 0  (at x=y=0)
            = 2×(P')²
```

But the current F_d implementation evaluates both D and AB to P(u)×P(u), missing the derivative structure!

## What Was Validated

1. **TWO-C Ψ expansion** - Correct structure with proper C_α/C_β separation
2. **Monomial counting** - All K=3 pairs have expected counts
3. **Case classification** - Correct mapping to F_d Cases A/B/C
4. **Case C kernel integral** - Bounded values with exp(Rθua) formula
5. **I₂ oracle** - Correctly computes 0.385 for (1,1)

## Files Created This Session

| File | Purpose |
|------|---------|
| `src/psi_separated_c.py` | TWO-C Ψ expansion implementation |
| `src/section7_clean_evaluator.py` | Clean-path F_d evaluation (WIP) |
| `src/test_clean_evaluator.py` | Clean evaluator tests |
| `src/test_two_benchmark_gate.py` | Two-benchmark validation tests |
| `src/debug_mirror_term.py` | Mirror term debugging |
| `src/debug_structure.py` | I₂ structure analysis |

## Recommended Next Steps

### Priority 1: Reconcile D and AB Monomials

The current implementation maps both D and AB to Case (B,B) which gives ∫P²du.

Need to understand how PRZZ distinguishes:
- D → (ζ'/ζ)' mixed derivative → I₂ base term?
- AB → (ζ'/ζ)×(ζ'/ζ) product → I₁ derivative term?

### Priority 2: Re-examine PRZZ Section 7 I-term Definitions

The I₁, I₂, I₃, I₄ terms have specific definitions in PRZZ TeX:
- I₁: mixed second derivative ∂²/∂x∂y
- I₂: base term (no derivatives)
- I₃: α-pole contribution
- I₄: β-pole contribution

How do these map to the Ψ monomials?

### Priority 3: Consider POST-MIRROR Direct Implementation

Instead of PRE-MIRROR → MIRROR → Q, consider directly implementing POST-MIRROR:
```python
c = sum over pairs of:
    (1/θ) × I-terms × ∫Q²exp(2Rt)dt
```

where I-terms are computed with their proper derivative structure.

## Key Equations

**PRZZ I₂ (POST-MIRROR):**
```
I₂ = (1/θ) × ∫₀¹ P_ℓ(u)P_ℓ̄(u) du × ∫₀¹ Q(t)² e^{2Rt} dt
```

**TWO-C Ψ formula:**
```
Ψ_{ℓ,ℓ̄} = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)×C(ℓ̄,p)×p! × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}
```

**Euler-Maclaurin weight:**
```
Weight = (1-u)^{ℓ+ℓ̄-2p}  (from Lemma 7.2)
```

## Test Commands

```bash
# Run TWO-C expansion tests
cd przz-extension && PYTHONPATH=. python3 src/psi_separated_c.py

# Run clean evaluator tests
cd przz-extension && PYTHONPATH=. python3 src/test_clean_evaluator.py

# Run two-benchmark gate
cd przz-extension && PYTHONPATH=. python3 src/test_two_benchmark_gate.py

# Run all pytest tests
cd przz-extension && python -m pytest tests/ -v
```

## Conclusion

The TWO-C Ψ expansion is correctly implemented and produces the expected monomial structure. However, the mapping from Ψ monomials to actual I-term integrals is not yet correct.

The key insight is that:
1. D and AB monomials represent DIFFERENT derivative contributions
2. The current F_d evaluation maps both to Case (B,B) → ∫P²du
3. This loses the derivative structure that distinguishes I₁ (derivatives) from I₂ (base)

Next session should focus on understanding how PRZZ's I₁-I₄ terms relate to the Ψ monomial structure, and implementing the correct derivative handling.
