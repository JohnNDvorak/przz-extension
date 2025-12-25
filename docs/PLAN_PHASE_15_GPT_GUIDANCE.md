# Phase 15 Plan: Root Out the 5% Gap — Full G-Product Investigation

**Date:** 2025-12-24
**Status:** PLANNING
**Predecessor:** Phase 14H-14K (Semantic Anchoring)

---

## Executive Summary

GPT's guidance identifies a critical flaw: **neither RAW_LOGDERIV nor POLE_CANCELLED is the correct J12 factor**. We've been choosing between two simplified modes when we should be computing the **actual G-product at finite α=-R**.

### The Three Interpretations

| Mode | Formula | Value at R=1.3036 | Basis |
|------|---------|-------------------|-------|
| RAW_LOGDERIV | `(ζ'/ζ)(1-R)² = (1/R + γ)²` | ~1.41 | Ignores 1/ζ factor |
| POLE_CANCELLED | `+1` | 1.00 | Limit as α→0 |
| **FULL G-PRODUCT** | `[(1/ζ)(ζ'/ζ)(1-R)]²` | **TBD** | Correct finite evaluation |

### Why Phase 14H Was Incomplete

Phase 14H proved that RAW_LOGDERIV matches the **bracket₂ formula as written**:
```
bracket₂(s,u) = A(0,0;β,α) × Σ 1/n^{1+s+u} × (ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)
```

But the **full autocorrelation ratio** (TECHNICAL_ANALYSIS.md Section 9.2) has:
```
Denominator: ζ(1+α+s)² × ζ(1+β+u)³ × ...
```

These 1/ζ factors are **not shown in bracket₂** but are part of the full J₁ structure!

### GPT's Core Insight

> "Do not 'pick RAW_LOGDERIV vs POLE_CANCELLED' as a mode toggle. Instead:
> Compute the J12 channel using the **actual product** implied by PRZZ:
> `(1/ζ)(ζ'/ζ)(1+α+s) × (1/ζ)(ζ'/ζ)(1+β+u)`"

---

## Phase 15A: Compute G(-R) Numerically

### Objective
Compute the **actual G-product** at the PRZZ evaluation point α=β=-R.

### Mathematical Setup

Define:
```
G(ε) = (1/ζ)(ζ'/ζ)(1+ε) = ζ'(1+ε) / ζ(1+ε)²
```

At ε→0: G(0) = -1 (pole cancellation)

At ε=-R (finite):
```
G(-R) = ζ'(1-R) / ζ(1-R)²
```

For R=1.3036: 1-R = -0.3036, so we evaluate zeta functions at s = -0.3036.

### Implementation

**New file:** `src/ratios/g_product_full.py`

```python
def compute_G_at_shift(alpha: float) -> float:
    """
    Compute G(alpha) = (1/ζ)(ζ'/ζ)(1+alpha) at finite alpha.

    This is NOT the pole limit. This is the actual evaluation.
    """
    s = 1 + alpha  # evaluation point
    zeta_val = mpmath.zeta(s)
    zeta_deriv = mpmath.zeta(s, derivative=1)

    # G(alpha) = zeta'(1+alpha) / zeta(1+alpha)^2
    return float(zeta_deriv / (zeta_val ** 2))

def compute_j12_full_G_product(R: float) -> float:
    """
    Compute J12 constant term using FULL G-product at α=β=-R.

    Returns G(-R)² = [(1/ζ)(ζ'/ζ)(1-R)]²
    """
    G_minus_R = compute_G_at_shift(-R)
    return G_minus_R ** 2
```

**Test file:** `tests/test_g_product_full.py`

### Acceptance Criteria

1. Compute G(-R) for R ∈ {1.3036, 1.1167} (both benchmarks)
2. Compare G(-R)² against:
   - RAW_LOGDERIV value: (1/R + γ)²
   - POLE_CANCELLED value: 1.0
3. If G(-R)² differs significantly, this explains the gap

---

## Phase 15B: Test G-Product Against +5 Gate

### Objective
Replace the Laurent factor in j12_as_integral() with the full G-product and rerun the +5 gate.

### Implementation

**Modify:** `src/ratios/j1_euler_maclaurin.py`

Add new mode:
```python
class LaurentMode(str, Enum):
    RAW_LOGDERIV = "raw_logderiv"
    POLE_CANCELLED = "pole_cancelled"
    FULL_G_PRODUCT = "full_g_product"  # NEW
```

In j12_as_integral():
```python
if laurent_mode == LaurentMode.FULL_G_PRODUCT:
    from src.ratios.g_product_full import compute_j12_full_G_product
    laurent_factor = compute_j12_full_G_product(R)
```

### Test Matrix

| Benchmark | R | RAW_LOGDERIV δ | POLE_CANCELLED δ | FULL_G_PRODUCT δ |
|-----------|------|----------------|------------------|------------------|
| κ | 1.3036 | +5.05% | ~+35% (worse) | **TBD** |
| κ* | 1.1167 | +1.58% | ~+30% (worse) | **TBD** |

### Success Criteria

If FULL_G_PRODUCT reduces δ significantly (to <1% for both benchmarks), this confirms GPT's diagnosis.

---

## Phase 15C: Trace 1/ζ Through the Full Structure

### Objective
Understand exactly WHERE the 1/ζ factors appear in the PRZZ derivation and why bracket₂ doesn't show them.

### Investigation Points

1. **TECHNICAL_ANALYSIS.md Section 9.2**: The denominator has ζ(1+α+s)² — where does this come from?

2. **psi_separated_c.py**: This file explicitly mentions:
   ```
   C_alpha = contribution from 1/zeta(1+alpha+s) pole
   C_beta = contribution from 1/zeta(1+beta+u) pole
   ```
   How does this relate to J12?

3. **Residue extraction**: When we extract residues at s=u=0, do the 1/ζ factors contribute?

4. **The F = exp(G) formalism**: Section 9.4 uses F = exp(G) where G = log(ratio). The ratio includes 1/ζ factors. Does F(0) include them?

### Key Question

Is bracket₂ = `(ζ'/ζ) × (ζ'/ζ)` correct **after** residue extraction, or should it be `G × G` = `[(1/ζ)(ζ'/ζ)] × [(1/ζ)(ζ'/ζ)]`?

---

## Phase 15D: Series-Order Stability Test

### Objective
Per GPT's guidance, verify Laurent coefficient extraction is converged.

### Implementation

**New file:** `tests/test_j12_series_stability.py`

```python
def test_j12_series_order_stability():
    """Verify B/A is stable across Laurent series orders."""
    results = {}
    for order in [2, 4, 6, 8]:
        result = compute_j12_with_order(R=1.3036, order=order)
        results[order] = result['B_over_A']

    # B/A should stabilize by order 6
    assert abs(results[6] - results[8]) < 1e-6
```

### Acceptance Criteria

- B/A varies < 1e-6 between order=6 and order=8
- If not stable → convergence problem that could blow up at k=4

---

## Phase 15E: K=4 Microcase Early Warning

### Objective
Create +7 gate (analog of +5 for K=4) to detect amplification.

### Implementation

**New file:** `src/ratios/microcase_plus7_signature_k4.py`

```python
def compute_plus7_signature(R: float, ...) -> Dict:
    """
    K=4 analog of +5 gate.
    Target B/A ≈ 7 (= 2K-1 for K=4)
    """
    # Same structure as plus5, but with K=4 formula
```

### Test Matrix

Run with all three Laurent modes:

| Mode | K=3 Gap | K=4 Gap | Amplification? |
|------|---------|---------|----------------|
| RAW_LOGDERIV | +5.05% | TBD | |
| POLE_CANCELLED | +35% | TBD | |
| FULL_G_PRODUCT | TBD | TBD | |

### Critical Check

If K=4 gap >> K=3 gap for a given mode, that mode will fail at higher K.

---

## Implementation Order

| Phase | Task | Files | Depends On |
|-------|------|-------|------------|
| 15A | Compute G(-R) numerically | g_product_full.py | None |
| 15A | Test G(-R) values | test_g_product_full.py | 15A impl |
| 15B | Add FULL_G_PRODUCT mode | j1_euler_maclaurin.py | 15A |
| 15B | Run +5 gate with new mode | test_plus5_gate.py | 15B impl |
| 15C | Trace 1/ζ through structure | Investigation | 15A,15B results |
| 15D | Series stability test | test_j12_series_stability.py | 15B |
| 15E | K=4 microcase gate | microcase_plus7_signature_k4.py | 15B |

---

## Predicted Outcomes

### If FULL_G_PRODUCT Works (Most Likely)

1. G(-R)² will differ from (1/R + γ)² by ~5-10%
2. Using FULL_G_PRODUCT will reduce δ to <1%
3. Phase 14H's "semantic proof" was correct for bracket₂, but bracket₂ is incomplete
4. The 1/ζ factors from the autocorrelation ratio were being dropped

### If FULL_G_PRODUCT Doesn't Work

1. The 1/ζ factors might appear elsewhere (not in J12)
2. Need to investigate other J1 pieces (J13, J14, J15)
3. The 5% gap might be structural in a different way

### If K=4 Amplifies the Gap

1. Current approach is fundamentally flawed for higher K
2. Need to derive formula from first principles before proceeding
3. Cannot rely on empirical calibration

---

## Questions Resolved

This plan directly addresses GPT's guidance:

1. ✅ "Compute the J12 channel using the actual product" → Phase 15A, 15B
2. ✅ "Add a J12-only diagnostic curve" → Phase 15C
3. ✅ "Add a series-order stability test" → Phase 15D
4. ✅ "Add a k=4 microcase gate now" → Phase 15E
5. ✅ "Redo L-sweep after the fix" → After 15B if successful

---

## Risk: What If G(-R) Is Already What RAW_LOGDERIV Computes?

The RAW_LOGDERIV formula is:
```
(ζ'/ζ)(1-R)² = (-1/(-R) + γ)² = (1/R + γ)²
```

This uses the **Laurent expansion** of ζ'/ζ, not direct evaluation.

The FULL G-product is:
```
G(-R)² = [ζ'(1-R) / ζ(1-R)²]²
```

These are mathematically **different objects**:
- Laurent: Uses `-1/ε + γ` approximation for `ζ'/ζ(1+ε)`
- Direct: Evaluates `ζ'(s)/ζ(s)²` at s = 1-R

At the PRZZ point (1-R = -0.3036), the difference could be significant because:
- ζ(-0.3) ≈ -0.08 (small, amplifies 1/ζ² term)
- ζ'(-0.3) ≈ some finite value
- The actual G(-R) might differ significantly from the Laurent approximation

This is exactly what needs to be computed in Phase 15A.

---

## Conclusion

The 5% gap is NOT acceptable. GPT's diagnosis is credible: we've been using a simplified Laurent approximation when we should be computing the full G-product. Phase 15A will immediately reveal whether this is the cause.

If the full G-product fixes the gap, we can proceed to K=4 with confidence. If not, we'll have ruled out one hypothesis and can investigate further.

**Do not proceed to Phase 1 optimization until this is resolved.**
