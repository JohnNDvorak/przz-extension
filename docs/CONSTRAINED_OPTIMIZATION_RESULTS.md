# Constrained Optimization Results

**Date:** 2025-12-29
**Status:** COMPLETED - Results confirm coefficient scale concern

---

## Executive Summary

Constrained optimization testing reveals that the κ = 0.521 result depends critically on polynomial coefficient magnitudes that exceed those used in prior work:

| Coefficient Cap | Best κ | Max |Coeff| | Improvement over PRZZ |
|-----------------|--------|-------------|----------------------|
| 1.0 (PRZZ scale) | **0.4211** | 0.995 | **+0.9%** |
| 2.0 (moderate) | 0.4605 | 1.78 | +10% |
| Unconstrained | 0.5211 | 2.41 | +25% |

**Key Finding:** With coefficient bounds matching Conrey/PRZZ scale (||P||_∞ ≤ 1.0), optimization finds κ ≈ 0.42, essentially matching PRZZ's published result. The large improvement to κ = 0.52 requires coefficient magnitudes 3.5× larger than PRZZ used.

---

## The Coefficient Scale Problem

| Paper | Year | κ Result | Max |Coefficient| | Improvement |
|-------|------|----------|-------------------|-------------|
| Conrey-Bui-Young | 2011 | 0.4105 | **0.077** | +0.17% |
| PRZZ | 2019 | 0.4173 | **0.687** | +0.68% |
| **Our cap=1.0** | 2025 | 0.4211 | **0.995** | +0.9% |
| **Our cap=2.0** | 2025 | 0.4605 | **1.78** | +10% |
| **Our unconstrained** | 2025 | 0.5211 | **2.41** | +25% |

The pattern is clear: larger coefficients enable larger κ improvements.

---

## Constrained Optimization Results

### Cap = 1.0 (Conrey/PRZZ Scale)

```
Best: κ = 0.4211, c = 2.1269

P1 = [0.156, -0.995, -0.297, 0.352]
P2 = [0.945, 0.822, -0.707]
P3 = [0.527, -0.375, -0.056]

Max |coefficient| = 0.995
```

**Analysis:** With PRZZ-scale coefficient bounds, the best κ we can achieve is essentially PRZZ's result (0.42 vs 0.417). This suggests PRZZ's optimization was near-optimal within their implicit coefficient regime.

### Cap = 2.0 (Moderate Relaxation)

```
Best: κ = 0.4605, c = 2.0204

P1 = [0.177, -0.937, -0.301, 0.239]
P2 = [0.807, 1.777, -0.475]
P3 = [0.431, -0.410, -0.052]

Max |coefficient| = 1.777
```

**Analysis:** Relaxing the bound to 2.0 allows P2 coefficients up to 1.78, enabling a 10% improvement over PRZZ. This intermediate result shows how κ improvement scales with coefficient magnitude.

---

## Interpretation

### What This Means for the κ = 0.521 Claim

1. **The main-term computation is correct.** Given the polynomial coefficients used, c = 1.87 and κ = 0.521 are accurately computed.

2. **The error bounds are unvalidated.** PRZZ's O(T/L) error analysis assumes bounded coefficients. With coefficients 3.5× larger than PRZZ's, the asymptotic error behavior is not established.

3. **Historical context matters.** Conrey et al. (2011) used coefficients an order of magnitude smaller (max 0.077). PRZZ used coefficients up to 0.69. Neither paper describes their optimization methodology, but both worked within implicit coefficient bounds.

### Honest Scientific Statement

```
Main-term analysis: κ_main = 0.52  (with ||P||_∞ ≤ 2.41)
Constrained result: κ_const = 0.42 (with ||P||_∞ ≤ 1.0)

Open question: Can PRZZ's error analysis be extended to larger coefficients?
```

---

## Technical Details

### Optimization Method

- **Algorithm:** Nearly Orthogonal Latin Hypercube (NOLH) sampling
- **Samples:** 49 design points per configuration
- **Quadrature:** n=40 Gauss-Legendre points
- **Parameters:** P1 (4 coeffs) + P2 (3 coeffs) + P3 (3 coeffs) = 10 free parameters
- **Q polynomial:** Fixed at PRZZ values

### Bound Implementation

```python
def get_parameter_bounds(..., max_coeff_magnitude=None):
    # Clip all P1/P2/P3 coefficient bounds to [-max, +max]
    if max_coeff_magnitude is not None:
        for i in range(10):  # P1(4) + P2(3) + P3(3)
            lo, hi = bounds[i]
            lo = max(lo, -max_coeff_magnitude)
            hi = min(hi, max_coeff_magnitude)
            bounds[i] = (lo, hi)
```

### Parameter Bounds for Each Configuration

**Cap = 1.0:**
- p1_1: [-1.0, -0.54] (constrained from [-1.61, -0.54])
- p2_0: [0.52, 1.0] (constrained from [0.52, 1.57])
- p2_1: [0.66, 1.0] (constrained from [0.66, 1.98])
- p2_2: [-1.0, -0.47] (constrained from [-1.41, -0.47])
- p3_1: [-1.0, -0.34] (constrained from [-1.03, -0.34])

**Cap = 2.0:**
- All parameters within natural bounds (PRZZ ±50% variation)
- No active constraints

---

## Recommendations

### For Paper Presentation

**Option 1: Conservative (Recommended)**
Present the constrained result as the main claim:
> We optimize the PRZZ mollifier polynomials within the coefficient regime matching prior work, achieving κ ≥ 0.42.

Mention unconstrained as an observation:
> Main-term analysis with relaxed coefficient bounds suggests κ = 0.52 may be achievable if error bounds can be extended.

**Option 2: Exploratory**
Present both results with clear caveats:
> Constrained optimization (||P||_∞ ≤ 1): κ = 0.42 (rigorous)
> Unconstrained optimization: κ = 0.52 (main-term only, error analysis TBD)

### For Future Work

1. **Extend PRZZ error analysis** to larger coefficients (if possible)
2. **Gradient-based refinement** of constrained optimum
3. **Higher quadrature** validation of constrained results

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/nolh_optimization/design.py` | Added `max_coeff_magnitude` parameter |
| `scripts/run_constrained_sweep.py` | Script for running constrained sweeps |
| `results/constrained_sweeps/results_cap1.0.json` | Cap=1.0 results |
| `results/constrained_sweeps/results_cap2.0.json` | Cap=2.0 results |
| `docs/CONSTRAINED_OPTIMIZATION_RESULTS.md` | This documentation |
