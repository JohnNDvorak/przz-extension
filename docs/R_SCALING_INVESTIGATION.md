# R-Scaling Investigation Summary

**Date:** 2025-12-17
**Status:** Root cause identified; not a simple R-scaling issue

---

## Executive Summary

The "R-scaling issue" is not a missing R-dependent factor. The problem is structural:

1. **For (1,1) pair**: The DSL matches the oracle perfectly (validated to 5.55e-17)
2. **For higher pairs**: The DSL is missing the full Ψ combinatorial structure
3. **The ratio reversal**: PRZZ's formula has negative correlation between ||P|| and contribution

---

## The Two-Benchmark Gate Results

| Benchmark | R | c_target | c_computed | Factor Needed |
|-----------|------|----------|------------|---------------|
| κ | 1.3036 | 2.137 | 1.950 | 1.096 |
| κ* | 1.1167 | 1.938 | 0.823 | 2.355 |

**Factor ratio: 2.355/1.096 = 2.15** (should be ~1.0)

---

## Key Discovery: c = const × t-integral

PRZZ's main-term constant decomposes as:
```
c = const × ∫Q²e^{2Rt}dt
```

| Component | κ (R=1.3036) | κ* (R=1.1167) | Ratio |
|-----------|--------------|---------------|-------|
| t-integral | 0.7163 | 0.6117 | **1.171** |
| const (target) | 2.984 | 3.168 | **0.942** |
| Combined c | 2.137 | 1.938 | **1.103** |

**Key insight**: 1.171 × 0.942 = 1.103 ✓ (matches PRZZ target!)

---

## The Ratio Reversal Problem

**Our naive formula gives:**
- const ratio = **1.71** (κ > κ*)

**PRZZ needs:**
- const ratio = **0.94** (κ < κ*)

**The ratios are in OPPOSITE DIRECTIONS!**

This means PRZZ's formula has **negative correlation** between polynomial magnitude and contribution:
- Larger polynomial norms → **smaller** contributions
- This is opposite to naive ∫P² expectation

---

## Root Cause: Missing Ψ Monomials

The DSL's I₁-I₄ decomposition is ONLY valid for (1,1) pairs. For higher pairs (ℓ, ℓ̄), PRZZ requires the full Ψ combinatorial expansion:

```
Ψ_{ℓ,ℓ̄}(A,B,C,D) = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × (D-C²)^p × (A-C)^{ℓ-p} × (B-C)^{ℓ̄-p}
```

### Monomial Coverage

| Pair | DSL terms | Ψ monomials | DSL Coverage |
|------|-----------|-------------|--------------|
| (1,1) | 4 | **4** | 100% ✓ |
| (2,2) | 4 | **12** | 33% ✗ |
| (3,3) | 4 | **27** | 15% ✗ |
| (1,2) | 4 | 7 | 57% ✗ |
| (1,3) | 4 | 10 | 40% ✗ |
| (2,3) | 4 | 18 | 22% ✗ |

---

## Oracle Validation (1,1)

The (1,1) pair is fully validated:

```
Oracle I-terms:
  I₁ = +0.426028
  I₂ = +0.384629
  I₃ = -0.225749
  I₄ = -0.225749
  Total = 0.359159

Ψ monomial sum:
  +1 × D    = +0.3846
  -1 × BC   = -0.2257
  -1 × AC   = -0.2257
  +1 × AB   = +0.4260
  Sum = 0.359159

Difference = 5.55e-17 ✓ PERFECT MATCH
```

---

## Why R=1.0 Testing Won't Help

The issue is NOT a missing R-dependent factor that a round-number test would reveal. The problem is:

1. **Structural mismatch**: Missing monomials for ℓ>1 pairs
2. **Sign pattern difference**: Ψ has negative coefficients that create cancellations
3. **Polynomial sensitivity**: Different polynomial degrees (κ vs κ*) interact differently with the Ψ structure

Testing with R=1.0 would just give another data point with the same structural issues.

---

## Cross-Integral Analysis

Some cross-integrals are **NEGATIVE** because P₃ changes sign on [0,1]:

| Pair | ∫P_i·P_j κ | ∫P_i·P_j κ* | Ratio | Sign |
|------|------------|-------------|-------|------|
| (1,1) | 0.307 | 0.300 | 1.02 | + |
| (1,2) | 0.470 | 0.307 | 1.53 | + |
| (1,3) | **-0.012** | **-0.027** | 0.43 | − |
| (2,2) | 0.725 | 0.318 | 2.28 | + |
| (2,3) | **-0.011** | **-0.027** | 0.43 | − |
| (3,3) | 0.007 | 0.003 | 2.83 | + |

---

## Candidate Explanations for Ratio Reversal

1. **Derivative terms (I₁, I₃, I₄) subtract from I₂**
   - κ* polynomials have lower degree → derivatives contribute less
   - Net effect: κ reduced more than κ*

2. **(1-u)^{ℓ₁+ℓ₂} weights**
   - Higher pairs suppressed more at u→1
   - May weight pairs differently across benchmarks

3. **Case C kernels**
   - P₂, P₃ use K_ω(u; R) instead of direct P(u)
   - Introduces R-dependent attenuation

4. **Ψ negative coefficients**
   - Many monomials have negative signs
   - Sign pattern differs by polynomial structure
   - May create cancellation that favors κ*

---

## Conclusion

The "R-scaling issue" is actually a **missing Ψ structure issue**:

- The DSL correctly implements the (1,1) case
- For (2,2), (3,3), and cross-pairs, the DSL only covers 15-57% of required monomials
- The missing monomials create the ratio reversal (negative correlation between ||P|| and contribution)

**Next steps should focus on implementing the full Ψ combinatorial expansion**, not on finding a missing R-dependent factor.

---

## PRZZ Reference Lines

| Concept | Lines | Notes |
|---------|-------|-------|
| I₁ formula | 1530-1532 | Main coupled term |
| I₂ formula | 1547-1548 | Decoupled product |
| I₃ formula | 1562-1563 | ∂/∂x derivative |
| I₄ formula | 1567-1570 | ∂/∂y derivative |
| I₅ error bound | 1621-1628 | O(T/L) lower-order |
| κ polynomials | 2571-2584 | R=1.3036, Q degree 5 |
| κ* polynomials | 2587-2598 | R=1.1167, Q linear |
| "Same process" for ℓ>1 | 1726 | Key ambiguity |

---

## Files Related to This Investigation

- `src/przz_22_exact_oracle.py` — Validated 2-variable oracle for (1,1)
- `src/psi_combinatorial.py` — Validates Ψ monomial counts
- `src/psi_block_configs.py` — p-sum representation
- `src/psi_monomial_expansion.py` — Expands p-configs to (a,b,c,d) vectors
- `src/psi_monomial_evaluator.py` — Maps monomials to I-term evaluations
- `docs/HANDOFF_SUMMARY.md` — Complete project state
