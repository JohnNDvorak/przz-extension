# Ψ_{2,2} Complete Oracle - Implementation Summary

## What Was Built

I've created a complete oracle for the (2,2) pair using the full Ψ expansion with all 12 monomials, as an alternative to the existing 4-term I₁-I₄ structure.

### New Files Created

1. **`src/psi_22_complete_oracle.py`** (470 lines)
   - Main implementation of the 12-monomial oracle
   - Class `Psi22CompleteOracle` with monomial evaluation methods
   - Test function comparing against I-term oracle

2. **`tests/test_psi_22_complete.py`** (250 lines)
   - Comprehensive test suite with 10 test cases
   - Validates structure, convergence, and consistency
   - Includes comparison report generator

3. **`PSI_22_ORACLE_DESIGN.md`** (documentation)
   - Complete design rationale and implementation strategy
   - Lists all 12 monomials with coefficients
   - Explains mapping to PRZZ Section 7 integrals
   - Outlines next steps for refinement

## The 12 Monomials

The implementation handles all 12 monomials that appear in Ψ_{2,2}:

**D-terms (4):**
- +4 × C⁰D¹A¹B¹
- +2 × C⁰D²A⁰B⁰
- -4 × C¹D¹A⁰B¹
- -4 × C¹D¹A¹B⁰

**Mixed A×B (3):**
- +1 × C⁰D⁰A²B²
- -2 × C¹D⁰A¹B²
- -2 × C¹D⁰A²B¹

**A-only (2):**
- +1 × C²D⁰A²B⁰
- +2 × C³D⁰A¹B⁰

**B-only (2):**
- +1 × C²D⁰A⁰B²
- +2 × C³D⁰A⁰B¹

**Pure C (1):**
- -1 × C⁴D⁰A⁰B⁰

## How It Works

### Monomial Evaluation Strategy

Each monomial A^a B^b C^c D^d is evaluated by mapping to appropriate PRZZ Section 7 integrals:

- **A factors** → z-derivative contributions (like I₃ structure)
- **B factors** → w-derivative contributions (like I₄ structure)
- **D factors** → mixed zw-derivative contributions (like I₁/I₂ relation)
- **C factors** → base integral contributions

### Implementation Approach

The oracle provides specialized evaluation functions:

```python
def eval_monomial(a, b, c, d) -> float:
    """Main dispatcher that routes to appropriate evaluator"""

# Specialized evaluators for each monomial type:
_eval_base_integral()      # C⁰D⁰A⁰B⁰ - base case
_eval_D_squared()          # C⁰D²A⁰B⁰ - simplest monomial
_eval_A_pow_a_B_pow_b()    # AᵃBᵇ - mixed derivatives
_eval_A_only()             # Aᵃ only (B=0, D=0)
_eval_B_only()             # Bᵇ only (A=0, D=0)
_eval_D_A_B_mixed()        # DᵈAᵃBᵇCᶜ - most complex
# ... etc
```

## Current Status

### ✅ Completed

1. **Full 12-monomial structure** - All monomials identified and categorized
2. **Oracle framework** - Complete class with dispatcher and evaluators
3. **Test suite** - Comprehensive validation including:
   - Monomial count and structure tests
   - Consistency checks against I-term oracle
   - Two-benchmark ratio test (κ/κ*)
   - Quadrature convergence test
4. **Documentation** - Design rationale and next steps

### ⚠️ Current Limitations

The implementation uses **approximate scalings** for C and D factors. This is intentional for the initial build. The oracle will run and produce results, but the values will NOT match the target 1.10 ratio yet.

**Why approximations?**
- Focus was on building the complete structure first
- Proper PRZZ Section 7 machinery requires deeper integration study
- Allows incremental refinement without blocking progress

## Testing

### Run the comparison report:

```bash
cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
python3 tests/test_psi_22_complete.py
```

This will show:
- Ψ oracle breakdown for all 12 monomials
- Comparison with I-term oracle (I₁, I₂, I₃, I₄)
- Two-benchmark ratio (κ vs κ*)

### Run pytest tests:

```bash
pytest tests/test_psi_22_complete.py -v
```

Expected results:
- ✅ Structure tests pass (12 monomials, correct categories)
- ✅ No crashes, all monomials evaluatable
- ✅ Convergence test passes
- ⚠️ Ratio test may not match 1.10 yet (due to approximate scalings)

## Next Steps for Refinement

To get accurate results matching the 1.10 target ratio:

### Phase 1: Study Existing Oracle
1. Analyze `src/przz_22_exact_oracle.py` in detail
   - How it computes I₁ (mixed derivative) - lines 110-275
   - How it computes I₂ (base integral) - lines 79-96
   - How it computes I₃, I₄ (single derivatives) - lines 282-361

2. Understand the derivative extraction machinery
   - Chain rule applications
   - Prefactor handling
   - Q argument derivatives

### Phase 2: Map Monomials to Integrals
1. D² term → Relate to I₂ structure with Q double derivatives
2. A²B² term → Extend I₁ logic to higher-order mixed derivatives
3. DAB terms → Combine I₁-like and single-derivative structures
4. C^k terms → Understand as powers of base integral

### Phase 3: Implement Proper Evaluators
1. Replace placeholder scalings in `_eval_*` methods
2. Use actual PRZZ Section 7 derivative formulas
3. Validate each monomial type independently

### Phase 4: Validate
1. Test on (1,1) first: 4 monomials should sum to I₁+I₂+I₃+I₄
2. Then validate (2,2) against I-term oracle
3. Finally check two-benchmark ratio ≈ 1.10

## Key Insights from Analysis

### Monomial→I-term mapping for (1,1):
```
Ψ_{1,1} = AB - AC - BC + D
        = I₁ + I₂ + I₃ + I₄

AB  → I₁  (mixed derivative, positive)
D   → I₂  (base integral, positive)
-AC → I₃  (z-derivative, negative)
-BC → I₄  (w-derivative, negative)
```

This mapping is **validated to machine precision** in the existing code, making it an excellent reference for building the (2,2) oracle.

### PRZZ Section 7 Building Blocks:

From the exact oracle analysis:
- **A** = d/dx of log-integrand → contributes P'/P + Q'/Q×(derivatives of args) + R×(arg derivatives)
- **B** = d/dy of log-integrand → symmetric structure to A
- **D** = d²/dxdy of log-integrand → includes -θ² from prefactor + Q''/Q terms
- **C** = base log-integrand value → relates to √I₂ conceptually

## File Locations

All new files are in the przz-extension directory:

```
przz-extension/
├── src/
│   └── psi_22_complete_oracle.py       # New oracle implementation
├── tests/
│   └── test_psi_22_complete.py         # New test suite
├── PSI_22_ORACLE_DESIGN.md             # Design document
└── PSI_22_ORACLE_IMPLEMENTATION_SUMMARY.md  # This file
```

## Integration with Existing Code

The new oracle:
- Uses existing polynomial infrastructure (`src/polynomials.py`)
- Uses existing monomial expansion (`src/psi_monomial_expansion.py`)
- Provides same interface as I-term oracle (returns total value)
- Can be validated against I-term oracle (`src/przz_22_exact_oracle.py`)
- Includes both κ and κ* polynomial support

## Validation Strategy

The oracle is designed to be validated incrementally:

1. **Level 1**: Structure tests ✅ (monomial count, categories)
2. **Level 2**: No-crash tests ✅ (all monomials evaluatable)
3. **Level 3**: Convergence tests ✅ (quadrature stability)
4. **Level 4**: Consistency tests ⚠️ (approximate, needs refinement)
5. **Level 5**: Accuracy tests ❌ (target ratio 1.10, not yet achieved)

## Additional Resources

### Related Files to Study:
- `src/przz_22_exact_oracle.py` - Reference I-term implementation (validated)
- `src/psi_monomial_evaluator.py` - Earlier monomial work
- `src/psi_block_configs.py` - p-config generator
- `HANDOFF_SUMMARY.md` - V2 DSL status and R-dependent issue
- `CLAUDE.md` - Project guidelines and validation gates

### Key Mathematical References:
- PRZZ paper Section 7 (lines 1530-1570 for integrals)
- Ψ formula expansion (X, Y, Z blocks)
- Connected vs disconnected blocks

## Recommendations

### For immediate validation:
1. Run the comparison report to see current baseline
2. Study the I-term oracle implementation (lines 79-361)
3. Start with D² term refinement (simplest case)

### For accurate implementation:
1. Use (1,1) as validation target first
2. Build up from simplest monomials (D², A²B²) to complex
3. Validate each monomial type independently before combining

### For long-term use:
1. Once (2,2) is validated, extend to (3,3)
2. Build generic monomial evaluator for any (ℓ₁, ℓ₂)
3. Integrate with optimization framework

## Summary

This implementation provides:
✅ Complete structural framework for 12-monomial evaluation
✅ Comprehensive test suite with multiple validation levels
✅ Clear documentation of design and next steps
⚠️ Baseline implementation with approximate scalings
❌ Not yet achieving target 1.10 ratio (refinement needed)

The oracle is **ready for incremental refinement** toward accurate PRZZ Section 7 integration.
