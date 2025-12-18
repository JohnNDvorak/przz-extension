# Ψ_{1,2} Oracle Implementation

**Date:** 2025-12-17
**Purpose:** Build an oracle for the (1,2) cross-pair using the full Ψ expansion (7 monomials)

---

## Background

### The Problem

The (1,2) pair (μ × μ⋆Λ) exhibits **catastrophic cancellation** in the current DSL:

| Metric | κ polynomials | κ* polynomials | Ratio |
|--------|---------------|----------------|-------|
| c₁₂ value | -0.2009 | -0.0016 | **129×** |

With κ* polynomials, the DSL shows near-perfect cancellation:
- Sum of positives: 0.380
- Sum of negatives: -0.382
- Net: -0.0016

This **129× ratio** is a red flag indicating the DSL is computing a structurally different object than PRZZ.

### The Hypothesis

The full Ψ combinatorial expansion should **NOT** have this artificial cancellation. By implementing all 7 monomials explicitly, we should see a ratio much closer to the PRZZ target of **~1.1×**.

---

## Ψ Formula for (1,2)

The general Ψ formula for pair (ℓ, ℓ̄):
```
Ψ_{ℓ,ℓ̄}(A,B,C,D) = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × (D-C²)^p × (A-C)^{ℓ-p} × (B-C)^{ℓ̄-p}
```

For (1,2):
```
Ψ_{1,2} = Σ_{p=0}^{1} C(1,p)C(2,p)p! × (D-C²)^p × (A-C)^{1-p} × (B-C)^{2-p}
```

### Expansion

**p=0 term:**
```
C(1,0)C(2,0)×0! × (D-C²)^0 × (A-C)^1 × (B-C)^2
= 1×1×1 × 1 × (A-C) × (B-C)²
= (A-C)(B² - 2BC + C²)
= AB² - 2ABC + AC² - B²C + 2BC² - C³
```

**p=1 term:**
```
C(1,1)C(2,1)×1! × (D-C²)^1 × (A-C)^0 × (B-C)^1
= 1×2×1 × (D-C²) × 1 × (B-C)
= 2(D-C²)(B-C)
= 2DB - 2DC - 2BC² + 2C³
```

**Combined (7 monomials after simplification):**
```
Ψ_{1,2} = AB² - 2ABC + AC² - B²C + C³ + 2DB - 2DC
```

Note: The BC² terms cancel (p=0: +2, p=1: -2).

---

## Implementation Approach

### Version 1: Simplified I₂-style Baseline

We start with the simplest possible interpretation:

**Building blocks:**
```
A = (1/θ) × ∫ P₁(u) du × ∫ Q(t)² exp(2Rt) dt
B = (1/θ) × ∫ P₂(u) du × ∫ Q(t)² exp(2Rt) dt
C = (1/θ) × ∫ 1 du × ∫ Q(t)² exp(2Rt) dt
D = (1/θ) × ∫ P₁(u)P₂(u) du × ∫ Q(t)² exp(2Rt) dt
```

Where:
- `D` is the I₂-type integral for the (1,2) pair
- `A`, `B`, `C` are simpler single-polynomial factors
- `(1/θ)` is the standard PRZZ prefactor

**Rationale:**
1. Start simple and verify the monomial structure works
2. Compare with DSL to understand what's missing
3. Refine based on testing and numerical results

**Expected refinements:**
- May need to include (1-u) weight factors
- May need to separate u and t dependencies differently
- May need to add derivative-weighted terms

---

## File Structure

### Main Files

**`src/psi_12_oracle.py`**
- Implements the 7-monomial Ψ expansion
- Returns `OracleResult12` with individual monomial contributions
- Includes debug mode for detailed output

**`tests/test_psi_12_oracle.py`**
- Tests oracle with both κ and κ* polynomials
- Verifies ratio is better than DSL's 129×
- Checks for catastrophic cancellation patterns
- Validates quadrature convergence

**`run_psi_12_oracle.py`**
- Quick runner script for manual testing
- Prints detailed breakdown and ratio analysis

---

## Testing Strategy

### Test 1: Basic Functionality
- Oracle runs without error
- All monomials are finite
- Total is reasonable magnitude

### Test 2: Ratio Analysis
**Key test:** Does Ψ expansion fix the ratio problem?
```python
ratio = result_κ.total / result_κ*.total
```
- DSL ratio: 129× (catastrophic)
- Target ratio: ~1.1×
- Acceptance: ratio < 50× (much better than DSL)
- Ideal: 0.5× < ratio < 5× (close to target)

### Test 3: Cancellation Analysis
With κ* polynomials, check:
```
|neg|/|pos| ratio
```
- DSL: 1.004 (near-perfect cancellation)
- Healthy: significantly different from 1.0
- Threshold: |ratio - 1| > 0.1

### Test 4: Quadrature Convergence
Results should be stable across n=40, 60, 80 to ~1% relative error.

---

## Expected Results

### Success Criteria

1. **Ratio improvement:** κ/κ* ratio < 50× (vs DSL's 129×)
2. **No catastrophic cancellation:** |neg|/|pos| not near 1.0 for κ*
3. **Stable convergence:** Results stable under quadrature refinement
4. **All monomials contribute:** No identically-zero terms

### If Ratio is Still Large

This would indicate that Version 1's simplified A,B,C,D definitions are missing key structure. Possible refinements:

1. **Add derivative weights:** Include P'(u) terms in A, B
2. **Add (1-u) factors:** Weight by (1-u)^ℓ powers
3. **Separate u/t structure:** Don't factor as product of u-int × t-int
4. **Study (1,1) mapping:** Use the validated (1,1) → I₁,I₂,I₃,I₄ mapping as a template

### If Ratio is Good

This validates the Ψ expansion approach! Next steps:

1. **Extend to (2,2):** Implement full 12-monomial expansion
2. **Extend to (2,3), (3,3):** Complete the Ψ framework
3. **Replace DSL:** Use Ψ oracles for all pairs
4. **Optimize:** Run polynomial optimization with correct formulas

---

## Key Insights from HANDOFF_SUMMARY

From Section 2 (Per-Pair Breakdown):
```
| Pair | κ (R=1.30) | κ* (R=1.12) | Ratio |
|------|------------|-------------|-------|
| c₁₂  | -0.2009    | -0.0016     | 129   |
```

From Section 2 (Root Cause):
> Analysis of the (1,2) pair with κ* polynomials:
> - Sum of positives: 0.380
> - Sum of negatives: -0.382
> - Net: -0.0016 (near-perfect cancellation!)
>
> This cancellation is an artifact of our DSL structure, not a feature of PRZZ's formula.

---

## Running the Oracle

### Quick Test
```bash
cd /path/to/przz-extension
python3 run_psi_12_oracle.py
```

### Run Tests
```bash
pytest tests/test_psi_12_oracle.py -v
```

### Use in Code
```python
from src.psi_12_oracle import psi_oracle_12
from src.polynomials import load_przz_polynomials

P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
theta = 4/7
R = 1.3036

result = psi_oracle_12(P1, P2, Q, theta, R, n_quad=80, debug=True)

print(f"Total Ψ_{1,2}: {result.total:.6f}")
print(f"Individual monomials: AB²={result.AB2:.6f}, ABC={result.ABC:.6f}, ...")
```

---

## Future Work

### Immediate Next Steps
1. Run the oracle and analyze results
2. Compare ratio with DSL's 129×
3. Refine A,B,C,D definitions if needed

### Medium Term
1. Extend to (2,2), (1,3), (2,3), (3,3)
2. Build general Ψ evaluator
3. Replace DSL entirely with Ψ framework

### Long Term
1. Understand the const/t-integral decomposition for Ψ
2. Solve the ratio reversal problem (const_κ/const_κ* ≈ 0.94)
3. Achieve PRZZ target: c_κ/c_κ* ≈ 1.10

---

## References

- **HANDOFF_SUMMARY.md Section 2:** Per-pair breakdown and (1,2) ratio problem
- **HANDOFF_SUMMARY.md Section 14:** Ψ formula and (1,1) validation
- **src/przz_22_exact_oracle.py:** Single-variable oracle structure template
- **PRZZ paper:** Original Ψ combinatorial formula (Section 7)
