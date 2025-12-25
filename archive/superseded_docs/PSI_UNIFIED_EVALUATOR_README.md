# Ψ Unified Evaluator - Integration Guide

## Overview

The unified evaluator integrates all Ψ oracles into a single pipeline for computing c and κ in the PRZZ framework.

## Files Created

### 1. `src/psi_unified_evaluator.py`
Main evaluator module that computes c and κ using all 6 pairs.

**Key Functions:**
- `evaluate_c_psi(theta, R, n_quad, polynomials)` - Main evaluation function
- `print_evaluation_report(result, polynomial_set)` - Pretty-print results

**Features:**
- Integrates all available Ψ oracles: (1,1), (2,2), (3,3), (1,2)
- Provides I₂-type stubs for (1,3) and (2,3) pairs
- Applies factorial normalization: 1/(ℓ₁! × ℓ₂!)
- Applies symmetry factor 2 for off-diagonal pairs
- Returns detailed per-pair breakdown
- Computes κ = 1 - log(c)/R

### 2. `tests/test_psi_unified.py`
Comprehensive test suite for the unified evaluator.

**Test Classes:**
- `TestPsiUnifiedBasic` - Smoke tests and basic functionality
- `TestPsiUnifiedTwoBenchmarks` - Two-benchmark comparison (κ and κ*)
- `TestPsiUnifiedPerPairBreakdown` - Per-pair contribution validation
- `TestPsiUnifiedConvergence` - Quadrature convergence tests
- `TestPsiUnifiedReporting` - Report generation tests

## Usage

### Basic Example

```python
from src.psi_unified_evaluator import evaluate_c_psi, print_evaluation_report
from src.polynomials import load_przz_polynomials

# Load κ polynomials (R=1.3036)
P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

# Evaluate c and κ
theta = 4.0 / 7.0
R = 1.3036
n_quad = 60

result = evaluate_c_psi(theta, R, n_quad, polys)

# Print report
print_evaluation_report(result, polynomial_set="κ")

# Access results
print(f"c = {result.c_total}")
print(f"κ = {result.kappa}")
```

### Two-Benchmark Comparison

```python
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

# κ benchmark (R=1.3036)
P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}
result_k = evaluate_c_psi(4/7, 1.3036, 60, polys_k)

# κ* benchmark (R=1.1167)
P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
polys_ks = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}
result_ks = evaluate_c_psi(4/7, 1.1167, 60, polys_ks)

# Compare ratio
ratio = result_k.c_total / result_ks.c_total
print(f"c(κ) / c(κ*) = {ratio:.4f}  (target: 1.10)")
```

## Oracle Mapping

The unified evaluator uses the following oracle implementations:

| Pair   | Oracle Module                  | Status      | Notes                          |
|--------|--------------------------------|-------------|--------------------------------|
| (1,1)  | `przz_22_exact_oracle`         | ✓ Validated | Reference implementation       |
| (2,2)  | `psi_22_complete_oracle`       | ✓ Complete  | 12 monomials via Ψ expansion   |
| (3,3)  | `psi_33_oracle`                | ✓ Complete  | 27 monomials via Ψ expansion   |
| (1,2)  | `psi_12_oracle`                | ✓ Complete  | 7 monomials via Ψ expansion    |
| (1,3)  | `_estimate_13_pair` (stub)     | ⚠ Stub      | Simple I₂-type approximation   |
| (2,3)  | `_estimate_23_pair` (stub)     | ⚠ Stub      | Simple I₂-type approximation   |

**Note:** The (1,3) and (2,3) pairs currently use I₂-type stubs. Full Ψ oracle implementations for these pairs are needed for accurate results.

## Normalization Factors

The unified evaluator applies two layers of normalization:

### 1. Factorial Normalization
Each pair (ℓ₁, ℓ₂) is multiplied by **1/(ℓ₁! × ℓ₂!)**:

- (1,1): ×1 (= 1/(1!×1!))
- (2,2): ×1/4 (= 1/(2!×2!))
- (3,3): ×1/36 (= 1/(3!×3!))
- (1,2): ×1/2 (= 1/(1!×2!))
- (1,3): ×1/6 (= 1/(1!×3!))
- (2,3): ×1/12 (= 1/(2!×3!))

### 2. Symmetry Factor
Off-diagonal pairs (ℓ₁ < ℓ₂) are multiplied by **2**:

- (1,2), (1,3), (2,3): ×2

### Combined Multipliers

Final multiplier = (symmetry) × (factorial):

- (1,1): ×1
- (2,2): ×1/4
- (3,3): ×1/36
- (1,2): ×2/2 = ×1
- (1,3): ×2/6 = ×1/3
- (2,3): ×2/12 = ×1/6

## Testing

### Run Tests

```bash
# Run all unified evaluator tests
cd /path/to/przz-extension
python -m pytest tests/test_psi_unified.py -v

# Run with output
python -m pytest tests/test_psi_unified.py -v -s

# Run specific test class
python -m pytest tests/test_psi_unified.py::TestPsiUnifiedTwoBenchmarks -v
```

### Run Demo

```bash
# Run standalone demo
python src/psi_unified_evaluator.py

# Run test demo
python tests/test_psi_unified.py
```

## Expected Results (with current stubs)

### κ Benchmark (R=1.3036)

With the current implementation (including I₂-type stubs for (1,3) and (2,3)):

- **c**: Approximately 1.5-2.5 (exact value depends on stubs)
- **κ**: Approximately 0.3-0.5
- **Dominant pairs**: (1,1) and (2,2)

### κ* Benchmark (R=1.1167)

- **c**: Approximately 1.0-2.0
- **κ**: Approximately 0.2-0.4

### Ratio c(κ) / c(κ*)

**Current (with stubs):** May vary significantly from 1.10

**Expected (after full oracles):** Should converge to ~1.10

## Next Steps for Full Implementation

### 1. Implement Full (1,3) Oracle

Create `src/psi_13_oracle.py` following the pattern from `psi_12_oracle.py`:

```python
# Ψ_{1,3} has 8 monomials
from src.psi_term_generator import generate_psi_terms

def psi_oracle_13(P1, P3, Q, theta, R, n_quad, debug=False):
    """Compute (1,3) contribution using full Ψ expansion."""
    terms = generate_psi_terms(1, 3)
    # Implement evaluation for each monomial
    # ...
    return OracleResult13(...)
```

### 2. Implement Full (2,3) Oracle

Create `src/psi_23_oracle.py`:

```python
# Ψ_{2,3} has 18 monomials
def psi_oracle_23(P2, P3, Q, theta, R, n_quad, debug=False):
    """Compute (2,3) contribution using full Ψ expansion."""
    terms = generate_psi_terms(2, 3)
    # Implement evaluation for each monomial
    # ...
    return OracleResult23(...)
```

### 3. Update Unified Evaluator

Replace stubs in `psi_unified_evaluator.py`:

```python
# Replace _estimate_13_pair with:
from src.psi_13_oracle import psi_oracle_13
result_13 = psi_oracle_13(P1, P3, Q, theta, R, n_quad, debug=False)
c13_raw = result_13.total

# Replace _estimate_23_pair with:
from src.psi_23_oracle import psi_oracle_23
result_23 = psi_oracle_23(P2, P3, Q, theta, R, n_quad, debug=False)
c23_raw = result_23.total
```

### 4. Update Tests

In `test_psi_unified.py`, update the ratio test:

```python
def test_ratio_comparison(self, kappa_result, kappa_star_result):
    """Test the ratio c(κ) / c(κ*) matches 1.10 target."""
    ratio = kappa_result.c_total / kappa_star_result.c_total

    # With full oracles, this should pass:
    assert ratio == pytest.approx(1.10, rel=0.01), \
        f"Ratio {ratio:.4f} deviates from target 1.10"
```

## Architecture

### Data Flow

```
Input Polynomials (P₁, P₂, P₃, Q)
        ↓
Dedicated Oracles for Each Pair
        ↓
Raw Values (c₁₁, c₂₂, c₃₃, c₁₂, c₁₃, c₂₃)
        ↓
Apply Normalization (factorial × symmetry)
        ↓
Sum Normalized Values
        ↓
Compute κ = 1 - log(c)/R
        ↓
Return UnifiedResult
```

### Integration Points

All oracles use:
- **Polynomial loading:** `src/polynomials.py`
- **Quadrature:** `src/quadrature.py`
- **Monomial generation:** `src/psi_term_generator.py`

## Debugging

### Enable Verbose Output

```python
# For individual oracles
from src.przz_22_exact_oracle import przz_oracle_22
result = przz_oracle_22(P1, Q, theta, R, n_quad, debug=True)

# For complete (2,2) oracle
from src.psi_22_complete_oracle import Psi22CompleteOracle
oracle = Psi22CompleteOracle(P2, Q, theta, R, n_quad)
total, results = oracle.compute_all_monomials(verbose=True)
```

### Check Per-Pair Contributions

```python
result = evaluate_c_psi(theta, R, n_quad, polys)

print("Raw values:")
print(f"  c₁₁: {result.c11_raw:.6f}")
print(f"  c₂₂: {result.c22_raw:.6f}")
print(f"  c₃₃: {result.c33_raw:.6f}")
print(f"  c₁₂: {result.c12_raw:.6f}")
print(f"  c₁₃: {result.c13_raw:.6f}")
print(f"  c₂₃: {result.c23_raw:.6f}")

print("\nNormalized values:")
print(f"  c₁₁: {result.c11_norm:.6f}")
print(f"  c₂₂: {result.c22_norm:.6f}")
print(f"  c₃₃: {result.c33_norm:.6f}")
print(f"  c₁₂: {result.c12_norm:.6f}")
print(f"  c₁₃: {result.c13_norm:.6f}")
print(f"  c₂₃: {result.c23_norm:.6f}")
```

## Known Issues

### 1. Stub Implementations
(1,3) and (2,3) pairs use I₂-type stubs that do not capture the full Ψ structure. This affects:
- Absolute c values
- Ratio c(κ)/c(κ*)
- Overall κ accuracy

**Resolution:** Implement full Ψ oracles for these pairs.

### 2. Quadrature Convergence
Some pairs (especially (3,3)) may require higher n_quad for convergence due to P₃ sign changes.

**Recommendation:** Test with n_quad ∈ [60, 80, 100] and verify stability.

## References

- **PRZZ Paper:** arXiv:1802.10521 (Section 7 for Ψ formulas)
- **Existing Oracles:**
  - `src/przz_22_exact_oracle.py` - Validated (1,1) reference
  - `src/psi_22_complete_oracle.py` - Complete (2,2) implementation
  - `src/psi_33_oracle.py` - Complete (3,3) implementation
  - `src/psi_12_oracle.py` - Complete (1,2) implementation
- **Term Generation:** `src/psi_term_generator.py`
- **Monomial Expansion:** `src/psi_monomial_expansion.py`

## Contact

For questions or issues, refer to the main PRZZ project documentation or CLAUDE.md.
