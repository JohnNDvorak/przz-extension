# Quick Start: Ψ_{2,2} Complete Oracle

## TL;DR

New 12-monomial oracle for (2,2) pair is built and ready for refinement.

**Status**: Structural framework ✅ | Accurate integration ⚠️ (needs PRZZ Section 7 work)

## Files Created

1. `src/psi_22_complete_oracle.py` - Main oracle
2. `tests/test_psi_22_complete.py` - Test suite
3. `PSI_22_ORACLE_DESIGN.md` - Design doc
4. `PSI_22_ORACLE_IMPLEMENTATION_SUMMARY.md` - Full summary

## Quick Test

```bash
cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension

# Run comparison report
python3 tests/test_psi_22_complete.py

# Run pytest
pytest tests/test_psi_22_complete.py -v
```

## Basic Usage

```python
from src.psi_22_complete_oracle import Psi22CompleteOracle
from src.polynomials import load_przz_polynomials

# Load κ polynomials
P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
theta = 4.0 / 7.0
R = 1.3036

# Create oracle
oracle = Psi22CompleteOracle(P2, Q, theta, R, n_quad=60)

# Compute all 12 monomials
total, results = oracle.compute_all_monomials(verbose=True)

# Access individual monomial values
for (a, b, c, d), mv in results.items():
    print(f"{mv.coefficient:+d} × C{c}D{d}A{a}B{b} = {mv.contribution:+.4f}")

print(f"Total: {total:.6f}")
```

## The 12 Monomials (with coefficients)

```
D-terms (4):
  +4 × C⁰D¹A¹B¹
  +2 × C⁰D²A⁰B⁰
  -4 × C¹D¹A⁰B¹
  -4 × C¹D¹A¹B⁰

Mixed A×B (3):
  +1 × C⁰D⁰A²B²
  -2 × C¹D⁰A¹B²
  -2 × C¹D⁰A²B¹

A-only (2):
  +1 × C²D⁰A²B⁰
  +2 × C³D⁰A¹B⁰

B-only (2):
  +1 × C²D⁰A⁰B²
  +2 × C³D⁰A⁰B¹

Pure C (1):
  -1 × C⁴D⁰A⁰B⁰
```

## What Works Now

✅ All 12 monomials identified and structured
✅ Oracle framework complete with dispatcher
✅ Test suite with 10+ validation tests
✅ Comparison with I-term oracle (I₁-I₄)
✅ No crashes, handles both κ and κ* polynomials

## What Needs Work

⚠️ **Approximate scalings** for C and D factors
- Oracle runs but doesn't match target 1.10 ratio yet
- Needs proper PRZZ Section 7 integration

## Next Steps to Improve Accuracy

### 1. Study the validated I-term oracle
```python
# See this file:
src/przz_22_exact_oracle.py

# Key functions:
# Lines 79-96:   I₂ (base integral)
# Lines 110-275: I₁ (mixed derivative)
# Lines 282-342: I₃ (z-derivative)
# Lines 344-361: I₄ (w-derivative)
```

### 2. Map monomials to I-term structures
- D² → similar to I₂
- A²B² → extend I₁ to higher order
- DAB → combination of I₁ and I₃/I₄
- C^k → powers of base integral

### 3. Replace approximate scalings

In `psi_22_complete_oracle.py`, refine these methods:
```python
def _eval_D_squared(self):      # Currently uses I₂ × 0.5
def _eval_A_pow_a_B_pow_b():    # Has derivative structure, needs refinement
def _eval_C_only():             # Placeholder 0.5^c scaling
```

### 4. Validate incrementally
1. Test on (1,1) first: 4 monomials should sum to I₁+I₂+I₃+I₄
2. Test D² alone on (2,2)
3. Add other monomials one by one
4. Check two-benchmark ratio → 1.10

## Expected Output Format

```
======================================================================
Ψ_{2,2} Complete Oracle: 12 Monomials
======================================================================
  +1 × C0D0A2B2    =  +1 ×   1.1234 =  +1.1234
  +4 × C0D1A1B1    =  +4 ×   0.2345 =  +0.9380
  +2 × C0D2A0B0    =  +2 ×   0.4567 =  +0.9134
  -2 × C1D0A1B2    =  -2 ×   0.1234 =  -0.2468
  ... (8 more)

  Total Ψ_{2,2} = 2.3456
```

## Validation Gates

Before claiming accuracy:

1. ✅ Structure: 12 monomials with correct coefficients
2. ✅ No crashes: All monomials evaluatable
3. ✅ Convergence: Results stable with n_quad increase
4. ⚠️ Consistency: Within 10x of I-term oracle
5. ❌ Accuracy: Two-benchmark ratio ≈ 1.10

Current status: **3/5 gates passed**

## Key Insight

The (1,1) mapping is **validated to machine precision**:
```
Ψ_{1,1} = AB - AC - BC + D = I₁ + I₂ + I₃ + I₄
```

Use this as ground truth to validate the monomial→integral mapping before extending to (2,2).

## Questions?

See full documentation:
- `PSI_22_ORACLE_DESIGN.md` - Design rationale
- `PSI_22_ORACLE_IMPLEMENTATION_SUMMARY.md` - Complete summary
- `HANDOFF_SUMMARY.md` - Project status
