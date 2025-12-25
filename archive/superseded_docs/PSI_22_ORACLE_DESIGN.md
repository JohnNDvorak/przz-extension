# Ψ_{2,2} Complete Oracle Design

## Overview

This document describes the implementation of the complete (2,2) oracle using the full Ψ expansion with 12 monomials, as opposed to the existing 4-term I₁-I₄ structure.

## Background

The PRZZ formula for pair (ℓ₁, ℓ₂) can be expressed using either:

1. **I-term structure** (existing): For (1,1), this gives I₁ + I₂ + I₃ + I₄
2. **Ψ monomial expansion** (new): Express as sum over monomials A^a B^b C^c D^d

For (2,2), the Ψ expansion has 12 monomials vs. 4 I-terms.

## The 12 Monomials for (2,2)

From the p-config expansion:
```
Ψ_{2,2} = 1×X²Y² + 4×ZXY + 2×Z²
```

Where X = (A-C), Y = (B-C), Z = (D-C²).

This expands to 12 monomials grouped as:

### D-terms (4 monomials):
```
  +4 × C⁰D¹A¹B¹  (coefficient = +4)
  +2 × C⁰D²A⁰B⁰  (coefficient = +2)
  -4 × C¹D¹A⁰B¹  (coefficient = -4)
  -4 × C¹D¹A¹B⁰  (coefficient = -4)
```

### Mixed A×B terms (3 monomials):
```
  +1 × C⁰D⁰A²B²  (coefficient = +1)
  -2 × C¹D⁰A¹B²  (coefficient = -2)
  -2 × C¹D⁰A²B¹  (coefficient = -2)
```

### A-only terms (2 monomials):
```
  +1 × C²D⁰A²B⁰  (coefficient = +1)
  +2 × C³D⁰A¹B⁰  (coefficient = +2)
```

### B-only terms (2 monomials):
```
  +1 × C²D⁰A⁰B²  (coefficient = +1)
  +2 × C³D⁰A⁰B¹  (coefficient = +2)
```

### Pure C term (1 monomial):
```
  -1 × C⁴D⁰A⁰B⁰  (coefficient = -1)
```

## Mapping to PRZZ Section 7 Integrals

The key insight from PRZZ Section 7 is the mapping:

- **A = ζ'/ζ(1+α+s)** → contributes z-derivative integral structure (like I₃)
- **B = ζ'/ζ(1+β+u)** → contributes w-derivative integral structure (like I₄)
- **C = ζ'/ζ(1+s+u)** → contributes base integral with no derivatives
- **D = (ζ'/ζ)'(1+s+u)** → contributes mixed derivative structure (like I₁ relates to I₂)

### Relationship to (1,1) I-terms

For (1,1), the validated mapping is:
```
Ψ_{1,1} = AB - AC - BC + D
        = I₁ + I₂ + I₃ + I₄
```

Where:
- AB → I₁ (mixed derivative, positive)
- D  → I₂ (base integral, positive)
- -AC → I₃ (z-derivative, negative)
- -BC → I₄ (w-derivative, negative)

## Implementation Strategy

### Phase 1: Baseline Implementation (CURRENT)

File: `src/psi_22_complete_oracle.py`

Current approach uses approximations:
1. Base integral: `_eval_base_integral()` - computes the no-derivative case
2. D² term: Uses scaled base integral
3. A²B² term: Uses (P'/P)^a × (P'/P)^b structure
4. A-only, B-only: Similar derivative scaling
5. C terms: Placeholder scaling

**Status**: This gives a working oracle but with approximate scalings. The ratios will NOT match 1.10 target yet.

### Phase 2: Proper PRZZ Section 7 Integration (NEEDED)

To get accurate results, each monomial needs proper evaluation using PRZZ Section 7 machinery:

1. **Understand Case A/B/C logic** from PRZZ Section 7
   - Case A: both variables appear in ζ arguments
   - Case B: variables appear separately
   - Case C: special handling for certain structures

2. **Implement proper derivative extraction**
   - Use the approach from `przz_22_exact_oracle.py` as reference
   - Each monomial type needs appropriate derivative formula

3. **Handle C factors correctly**
   - C represents "base" or "disconnected" contribution
   - Not just a scaling factor - needs proper integral representation

4. **Handle D factors correctly**
   - D involves the (ζ'/ζ)' structure
   - Related to second derivatives in Q

### Phase 3: Validation

Tests in `tests/test_psi_22_complete.py`:

1. **Monomial structure test**: Verify 12 monomials with correct coefficients
2. **κ consistency**: Compare against I-term oracle for R=1.3036
3. **κ* consistency**: Compare against I-term oracle for R=1.1167
4. **Two-benchmark ratio**: Verify κ/κ* ≈ 1.10
5. **Quadrature convergence**: Ensure results stabilize with more points
6. **Per-monomial validation**: Each monomial gives finite, reasonable values

## Key Files

- `src/psi_22_complete_oracle.py` - Main oracle implementation
- `src/psi_monomial_expansion.py` - Generates the 12 monomials from p-configs
- `src/psi_block_configs.py` - p-config generator (X, Y, Z blocks)
- `src/przz_22_exact_oracle.py` - Reference I-term oracle for (2,2)
- `tests/test_psi_22_complete.py` - Validation test suite

## Current Limitations

1. **Approximate scalings**: The current implementation uses placeholder scalings for C and D factors
2. **Missing Section 7 machinery**: Full PRZZ case logic not yet implemented
3. **No per-monomial validation**: Haven't verified individual monomials match expected structure

## Next Steps

### Immediate (to make oracle accurate):

1. **Study przz_22_exact_oracle.py in detail**
   - Understand how it computes I₁ (mixed derivative)
   - Understand how it computes I₂ (base integral)
   - Understand how it computes I₃, I₄ (single derivatives)

2. **Map each monomial type to appropriate integral**
   - D² → similar to I₂ structure
   - A²B² → similar to I₁ structure but higher order
   - DAB → mixed structure
   - C^k → powers of base integral

3. **Implement proper evaluation functions**
   - Replace placeholder scalings with actual integrals
   - Use derivative formulas from Section 7

4. **Validate against (1,1)**
   - Use (1,1) as sanity check: the 4 monomials should give I₁+I₂+I₃+I₄
   - This validates the monomial→integral mapping

### Long-term (for full PRZZ implementation):

1. Extend to (3,3) - 27 monomials
2. Extend to mixed pairs (1,2), (1,3), (2,3)
3. Build generic monomial evaluator for any (ℓ₁, ℓ₂)
4. Integration with optimization framework

## Testing Strategy

### Unit Tests:
- Each `_eval_*` method should have a test
- Test with simple polynomials where we can compute analytically
- Test quadrature convergence

### Integration Tests:
- Compare full (2,2) oracle against I-term oracle
- Verify two-benchmark ratio
- Cross-validate with (1,1) mapping

### Regression Tests:
- Once we get correct values, lock them in as golden outputs
- Detect any changes in future refactorings

## References

- PRZZ paper Section 7 (integral formulas)
- `src/przz_22_exact_oracle.py` (working I-term implementation)
- `HANDOFF_SUMMARY.md` (project status and V2 DSL findings)
- `CLAUDE.md` (development rules and validation gates)

## Questions for Future Sessions

1. What is the exact relationship between C and the base integral?
2. Can we derive D² directly from the (ζ'/ζ)' definition?
3. Should A² be computed as (d/dx₁)(d/dx₂) or d²/dx²?
4. How do the Case A/B/C distinctions affect monomial evaluation?
5. Is there a simpler mapping that bypasses full Section 7 machinery?
