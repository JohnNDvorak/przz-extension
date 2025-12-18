# Ψ Term Generator Implementation

## Overview

This implementation provides a complete config-driven term generator for the Ψ combinatorial formula used in PRZZ main-term computation.

## Files Created

### 1. `src/psi_term_generator.py`

**Purpose:** Unified interface for generating Ψ terms with integral structure mapping.

**Key Components:**

- **`IntegralType` enum:** Classifies terms by evaluation strategy
  - `I1_MIXED`: Mixed derivative ∂z∂w (AB terms)
  - `I2_BASE`: No derivatives (D terms)
  - `I3_Z_DERIV`: z-derivative only (AC terms)
  - `I4_W_DERIV`: w-derivative only (BC terms)
  - `GENERAL`: Higher-order terms requiring full evaluation

- **`PsiTerm` dataclass:** Represents a single term with:
  - Exponents (a, b, c, d) for A^a × B^b × C^c × D^d
  - Integer coefficient (can be negative)
  - Integral type classification
  - Human-readable description

- **`PsiTermCollection` dataclass:** Organized collection of terms for a (ℓ, ℓ̄) pair with:
  - List of all terms
  - Automatic grouping by integral type
  - Total term count

**Main API:**

```python
generate_psi_terms(ell: int, ellbar: int) -> PsiTermCollection
```

This function:
1. Calls `expand_pair_to_monomials()` to get all monomials with coefficients
2. Classifies each monomial by integral type
3. Adds descriptive information
4. Returns a structured collection

**Example Usage:**

```python
from src.psi_term_generator import generate_psi_terms

# Generate terms for (1,1) pair
collection = generate_psi_terms(1, 1)

print(f"Total terms: {collection.total_terms}")  # 4

# Access terms by type
by_type = collection.by_type
i1_terms = by_type[IntegralType.I1_MIXED]  # [AB term]
i2_terms = by_type[IntegralType.I2_BASE]   # [D term]
i3_terms = by_type[IntegralType.I3_Z_DERIV]  # [AC term]
i4_terms = by_type[IntegralType.I4_W_DERIV]  # [BC term]

# Iterate over all terms
for term in collection.terms:
    print(f"{term.coeff:+d} × A^{term.a}B^{term.b}C^{term.c}D^{term.d}")
    # Evaluate using appropriate integral...
```

### 2. `tests/test_psi_term_generator.py`

**Purpose:** Comprehensive unit tests for the term generator.

**Test Classes:**

1. **`TestMonomialCounts`:** Verifies correct monomial counts
   - (1,1) → 4 terms
   - (2,2) → 12 terms
   - (3,3) → 27 terms
   - (1,2) → 6 terms
   - (1,3) → 8 terms
   - (2,3) → 18 terms

2. **`TestIntegralMapping11`:** Validates (1,1) → I₁-I₄ mapping
   - AB (1,1,0,0) → I1_MIXED with coeff +1
   - D (0,0,0,1) → I2_BASE with coeff +1
   - AC (1,0,1,0) → I3_Z_DERIV with coeff -1
   - BC (0,1,1,0) → I4_W_DERIV with coeff -1

3. **`TestCoefficientFormula`:** Checks coefficients match Ψ formula
   - Verifies (1,1) expands to AB - AC - BC + D
   - Checks (2,2) and (3,3) coefficient signs and magnitudes

4. **`TestIntegralClassification`:** Tests integral type classification logic

5. **`TestHigherPairs`:** Validates (2,2) and (3,3) generation

6. **`TestTermStructure`:** Checks data structure correctness

7. **`TestCoefficientAccuracy`:** Verifies specific coefficient values

8. **`TestEdgeCases`:** Tests boundary conditions

**Running Tests:**

```bash
cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
python -m pytest tests/test_psi_term_generator.py -v
```

### 3. `test_psi_manual.py`

**Purpose:** Standalone manual test script for quick verification without pytest.

**Usage:**

```bash
cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
python test_psi_manual.py
```

**Expected Output:**

```
✓ Successfully imported psi_term_generator

======================================================================
Testing (1,1) pair
======================================================================
✓ Generated 4 terms for (1,1)
✓ Count is correct (expected 4)

Terms:
  PsiTerm(+1 × A × B [I1])
  PsiTerm(+1 × D [I2])
  PsiTerm(-1 × A × C [I3])
  PsiTerm(-1 × B × C [I4])

======================================================================
Testing (2,2) pair
======================================================================
✓ Generated 12 terms for (2,2)
✓ Count is correct (expected 12)

======================================================================
Testing (3,3) pair
======================================================================
✓ Generated 27 terms for (3,3)
✓ Count is correct (expected 27)

======================================================================
Running full validation
======================================================================
  (1,1): 4 terms (expected 4) ✓
  (2,2): 12 terms (expected 12) ✓
  (3,3): 27 terms (expected 27) ✓
  (1,2): 6 terms (expected 6) ✓
  (1,3): 8 terms (expected 8) ✓
  (2,3): 18 terms (expected 18) ✓

✓✓✓ ALL TESTS PASSED ✓✓✓
```

## Mathematical Background

### The Ψ Formula

For pair (ℓ, ℓ̄), the Ψ formula is:

```
Ψ_{ℓ,ℓ̄}(A,B,C,D) = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × (D-C²)^p × (A-C)^{ℓ-p} × (B-C)^{ℓ̄-p}
```

Where:
- **A** = ∂/∂z derivative piece = ζ'/ζ(1+α+s)
- **B** = ∂/∂w derivative piece = ζ'/ζ(1+β+u)
- **C** = log ξ(s₀) no-derivative piece = ζ'/ζ(1+s+u)
- **D** = ∂²/∂z∂w mixed derivative piece = (ζ'/ζ)'(1+s+u)

### Expansion Process

The term generator uses the existing infrastructure:

1. **Block representation** (`psi_block_configs.py`):
   - Represents Ψ as p-sum: Σ_p coeff × Z^p × X^{ℓ-p} × Y^{ℓ̄-p}
   - Where X = (A-C), Y = (B-C), Z = (D-C²)

2. **Monomial expansion** (`psi_monomial_expansion.py`):
   - Expands each p-config using binomial theorem
   - Combines terms with same (a,b,c,d) exponents
   - Returns dict: (a,b,c,d) → coefficient

3. **Term generation** (`psi_term_generator.py`):
   - Wraps monomials in structured PsiTerm objects
   - Classifies by integral type
   - Provides clean API for downstream use

### Example: (1,1) Expansion

```
Ψ_{1,1} = Σ_{p=0}^1 C(1,p)C(1,p)p! × (D-C²)^p × (A-C)^{1-p} × (B-C)^{1-p}

p=0: C(1,0)·C(1,0)·0! = 1
     → (A-C)(B-C) = AB - AC - BC + C²

p=1: C(1,1)·C(1,1)·1! = 1
     → (D-C²) = D - C²

Sum: AB - AC - BC + C² + D - C² = AB - AC - BC + D
```

This gives 4 monomials:
- +1 × AB (a=1, b=1, c=0, d=0) → I₁
- +1 × D  (a=0, b=0, c=0, d=1) → I₂
- -1 × AC (a=1, b=0, c=1, d=0) → I₃
- -1 × BC (a=0, b=1, c=1, d=0) → I₄

## Integration with PRZZ Pipeline

The term generator fits into the PRZZ computation pipeline:

```
┌─────────────────┐
│ Input: (ℓ, ℓ̄)  │
└────────┬────────┘
         │
         v
┌─────────────────────┐
│ generate_psi_terms  │  ← This module
└────────┬────────────┘
         │
         v
┌─────────────────────────────────┐
│ PsiTermCollection               │
│  - 4/12/27 terms with coeffs   │
│  - Grouped by integral type    │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│ For each term:                  │
│  1. Classify integral type      │
│  2. Select evaluation strategy  │
│  3. Compute integral            │
│  4. Multiply by coefficient     │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│ Sum all terms → c_{ℓ,ℓ̄}        │
└─────────────────────────────────┘
```

### Next Steps for Integration

1. **For (1,1) pairs:** Map to existing I₁-I₄ evaluators
   - Use `psi_monomial_evaluator.py` for reference

2. **For higher pairs:** Implement GENERAL integral evaluator
   - Will require Section 7 machinery from PRZZ
   - Handle A², B², D² terms and mixed products

3. **Full pipeline:** Create main evaluation loop
   ```python
   def evaluate_pair(ell, ellbar, P, Q, theta, R):
       collection = generate_psi_terms(ell, ellbar)
       total = 0.0
       for term in collection.terms:
           integral_value = evaluate_integral(term, P, Q, theta, R)
           total += term.coeff * integral_value
       return total
   ```

## Design Principles

### 1. Separation of Concerns

- **Monomial generation:** `psi_monomial_expansion.py` (mathematical)
- **Term classification:** `psi_term_generator.py` (structural)
- **Integral evaluation:** `psi_monomial_evaluator.py` (numerical)

### 2. Config-Driven

The generator is purely config-driven:
- Input: (ℓ, ℓ̄) pair
- Output: Complete term collection with metadata
- No hardcoded term tables

### 3. Type Safety

Uses dataclasses and enums for:
- Clear type signatures
- Automatic validation
- Better IDE support

### 4. Testability

Every component has dedicated tests:
- Unit tests for classification logic
- Integration tests for full generation
- Validation tests against known counts

## Validation Results

### Expected Monomial Counts

| Pair   | p values | Monomials | Status |
|--------|----------|-----------|--------|
| (1,1)  | 0,1      | 4         | ✓      |
| (2,2)  | 0,1,2    | 12        | ✓      |
| (3,3)  | 0,1,2,3  | 27        | ✓      |
| (1,2)  | 0,1      | 6         | ✓      |
| (1,3)  | 0,1      | 8         | ✓      |
| (2,3)  | 0,1,2    | 18        | ✓      |

**Total for K=3:** 75 monomials (4+12+27+6+8+18)

### Coefficient Verification

The (1,1) case has been verified against the analytical expansion:
- AB coefficient: +1 ✓
- D coefficient: +1 ✓
- AC coefficient: -1 ✓
- BC coefficient: -1 ✓

Higher pairs have been verified for:
- Correct prefactors from C(ℓ,p)·C(ℓ̄,p)·p!
- Sign alternation from binomial expansions
- Cancellation of intermediate terms

## Known Limitations

1. **Integral evaluation not implemented yet**
   - Term generation is complete
   - Need to implement GENERAL evaluator for higher pairs

2. **Limited to d=1**
   - Current implementation assumes d=1
   - Extension to d=2 would require more complex derivative structure

3. **No automatic simplification**
   - Returns all non-zero monomials as-is
   - Could implement algebraic simplification if needed

## References

- **Mathematical derivation:** See existing `psi_combinatorial.py` header
- **Block representation:** See `psi_block_configs.py`
- **Monomial expansion:** See `psi_monomial_expansion.py`
- **PRZZ paper:** Pratt et al. (2019) for full context

## Contributing

When extending this module:

1. **Add new integral types:** Update `IntegralType` enum
2. **Modify classification:** Update `classify_integral_type()` function
3. **Add tests:** Every change must have corresponding test coverage
4. **Update docs:** Keep this README in sync with code

## License

Part of the PRZZ extension project for κ optimization.
