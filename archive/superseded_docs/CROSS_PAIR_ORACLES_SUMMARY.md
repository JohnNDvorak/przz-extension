# Cross-Pair Oracles: (1,3) and (2,3) Implementation

## Summary

Built Ψ-expansion oracles for the remaining cross-pairs:
- **(1,3)**: μ × μ⋆Λ⋆Λ (P₁ × P₃) - 10 monomials
- **(2,3)**: μ⋆Λ × μ⋆Λ⋆Λ (P₂ × P₃) - 18 monomials

These oracles follow the same pattern as the existing (1,2) oracle and use the full Ψ combinatorial expansion to avoid the catastrophic cancellation observed in the old DSL implementation.

## Files Created

### Oracle Implementations
1. **`src/psi_13_oracle.py`** - (1,3) oracle with 10 monomials
2. **`src/psi_23_oracle.py`** - (2,3) oracle with 18 monomials

### Test Files
3. **`tests/test_psi_13_oracle.py`** - Comprehensive tests for (1,3)
4. **`tests/test_psi_23_oracle.py`** - Comprehensive tests for (2,3)

## Mathematical Background

### (1,3) Pair: μ × μ⋆Λ⋆Λ

**Formula:**
```
Ψ_{1,3}(A,B,C,D) = Σ_{p=0}^{1} C(1,p)C(3,p)p! × (D-C²)^p × (A-C)^{1-p} × (B-C)^{3-p}
```

**Expansion:**
- **p=0**: C(1,0)C(3,0)×0! = 1, giving: `(A-C)(B-C)³`
- **p=1**: C(1,1)C(3,1)×1! = 3, giving: `3(D-C²)(B-C)²`

**Monomials (10 total):**
```
AB³ - 3AB²C + 3ABC² - AC³ - B³C + 3BC³ - 2C⁴ + 3DB² - 6DBC + 3DC²
```

Note: B²C² has coefficient (3-3)=0 and cancels out.

### (2,3) Pair: μ⋆Λ × μ⋆Λ⋆Λ

**Formula:**
```
Ψ_{2,3}(A,B,C,D) = Σ_{p=0}^{2} C(2,p)C(3,p)p! × (D-C²)^p × (A-C)^{2-p} × (B-C)^{3-p}
```

**Expansion:**
- **p=0**: C(2,0)C(3,0)×0! = 1, giving: `(A-C)²(B-C)³`
- **p=1**: C(2,1)C(3,1)×1! = 6, giving: `6(D-C²)(A-C)(B-C)²`
- **p=2**: C(2,2)C(3,2)×2! = 6, giving: `6(D-C²)²(B-C)`

**Monomials (18 total):**
The 18 monomials come from expanding the three p-configurations above.

## Building Block Definitions

Following the (1,2) oracle pattern, the fundamental building blocks are:

```python
A = (1/θ) × ∫ P_ℓ(u) du × ∫ Q(t)² exp(2Rt) dt
B = (1/θ) × ∫ P_ℓ̄(u) du × ∫ Q(t)² exp(2Rt) dt
C = (1/θ) × ∫ 1 du × ∫ Q(t)² exp(2Rt) dt
D = (1/θ) × ∫ P_ℓ(u)P_ℓ̄(u) du × ∫ Q(t)² exp(2Rt) dt
```

For (1,3): P_ℓ = P₁, P_ℓ̄ = P₃
For (2,3): P_ℓ = P₂, P_ℓ̄ = P₃

## Important: P₃ Sign Changes

**Critical Note**: P₃ changes sign on [0,1], so the cross-integrals can be negative:
- ∫P₁(u)P₃(u) du can be negative
- ∫P₂(u)P₃(u) du can be negative

This is mathematically correct and expected. The sign changes are handled properly in the oracle implementations.

## Performance Expectations

### From HANDOFF_SUMMARY

The old DSL showed poor κ/κ* ratios for these cross-pairs:
- **(1,3)**: ratio = 5.73× (target: ~1.1×)
- **(2,3)**: ratio = 9.04× (target: ~1.1×)

The Ψ-expansion oracles should show **dramatically better** ratios, similar to how the (1,2) oracle improved from 129× to a more reasonable value.

### Test Criteria

The tests verify:
1. **Oracle completes without error** for both κ and κ* polynomials
2. **All monomials are finite**
3. **Quadrature convergence** (stable results at n=60, 80, 100)
4. **Ratio improvement**: κ/κ* ratio should be much better than DSL
   - (1,3): ratio should be < 4.0 (vs. DSL 5.73)
   - (2,3): ratio should be < 6.0 (vs. DSL 9.04)
5. **No catastrophic cancellation** in monomial sums

## Usage Example

```python
from src.psi_13_oracle import psi_oracle_13
from src.psi_23_oracle import psi_oracle_23
from src.polynomials import load_przz_polynomials

theta = 4/7
R_kappa = 1.3036
P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

# (1,3) oracle
result_13 = psi_oracle_13(P1, P3, Q, theta, R_kappa, n_quad=80, debug=True)
print(f"(1,3) total: {result_13.total}")

# (2,3) oracle
result_23 = psi_oracle_23(P2, P3, Q, theta, R_kappa, n_quad=80, debug=True)
print(f"(2,3) total: {result_23.total}")
```

## Test Execution

Run tests with:
```bash
pytest tests/test_psi_13_oracle.py -v
pytest tests/test_psi_23_oracle.py -v
```

Or test a specific function:
```bash
pytest tests/test_psi_13_oracle.py::TestPsi13Oracle::test_ratio_better_than_dsl -v
```

## Next Steps

1. **Verify monomial counts** by running the expansion module
2. **Run full test suite** to ensure no regressions
3. **Check actual ratios** against expected improvements
4. **Compare with DSL** to validate Ψ-expansion approach
5. **Integrate** into full c-value computation if ratios are good

## References

- **Template**: `src/psi_12_oracle.py` - (1,2) oracle pattern
- **Monomial Expansion**: `src/psi_monomial_expansion.py` - Generates monomial lists
- **Background**: `HANDOFF_SUMMARY.md` Section 2 - DSL ratio issues
- **Theory**: PRZZ Section 7 - Building block definitions (A, B, C, D)
