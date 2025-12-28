# Derivation Status Summary

**Date:** 2025-12-27
**Purpose:** Honest accounting of what is derived vs calibrated in the PRZZ κ formula

---

## The Formula

```
c = I₁(+R) + g₁×base×I₁(-R) + I₂(+R) + g₂×base×I₂(-R) + S₃₄

where:
  base = exp(R) + (2K-1)
  g₁, g₂ = component-specific corrections
  κ = 1 - log(c)/R
```

---

## Component Status

| Component | Status | Source | Accuracy |
|-----------|--------|--------|----------|
| **exp(R)** | **DERIVED** | Difference quotient T^{-(α+β)} at α=β=-R/L | Exact |
| | | PRZZ Lines 1502-1511 | |
| **(2K-1)** | **DERIVED** | Unified bracket B/A ratio | Exact |
| | | Phase 32 polynomial ladder (17 tests) | |
| **g_baseline = 1 + θ/(2K(2K+1))** | **DERIVED** | Beta moment from log factor cross-terms | Exact |
| | | Phase 34C, PRZZ Lines 1530, 2391-2409 | |
| **Mirror structure** | **DERIVED** | PRZZ Section 10 | Exact |
| | | c = S12(+R) + m×S12(-R) + S34 | |
| **g_I1 = 1.00091428** | **CALIBRATED** | Solved from 2-benchmark system | N/A |
| | | Uses c_target as input | |
| **g_I2 = 1.01945154** | **CALIBRATED** | Solved from 2-benchmark system | N/A |
| | | Uses c_target as input | |

---

## The Two Formulas

### Phase 36: Production Formula (Genuinely Derived)

```
m = [1 + θ/(2K(2K+1))] × [exp(R) + (2K-1)]
c = S12(+R) + m × S12(-R) + S34
```

**Accuracy:** ±0.15% on both κ and κ* benchmarks
**Residual source:** Q polynomial interaction (Phase 35 analysis)
**Status:** This represents genuine understanding

### Phase 45: I1/I2 Decomposition (With Calibrated Parameters)

```
g_total = f_I1 × g_I1 + (1 - f_I1) × g_I2
c = I1(+R) + g_I1×base×I1(-R) + I2(+R) + g_I2×base×I2(-R) + S34
```

**Accuracy:** ~0% gap (tautological - 2 params fit to 2 targets)
**Status:** Shows WHERE the ±0.15% residual distributes, not WHY

---

## The Honest Picture

```
┌─────────────────────────────────────────────────────────┐
│                    PRZZ κ FORMULA                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │           GENUINELY DERIVED (~98%)              │   │
│  │                                                 │   │
│  │  • exp(R) from difference quotient             │   │
│  │  • (2K-1) from bracket ratio                   │   │
│  │  • θ/(2K(2K+1)) from Beta moment               │   │
│  │  • Mirror structure from PRZZ Section 10       │   │
│  │                                                 │   │
│  │  → Production formula: ±0.15% accuracy         │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          CALIBRATED (~2%)                       │   │
│  │                                                 │   │
│  │  • g_I1 = 1.00091428                           │   │
│  │  • g_I2 = 1.01945154                           │   │
│  │                                                 │   │
│  │  → Fit to match targets exactly                │   │
│  │  → Explains WHERE residual goes, not WHY       │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Open Research Question

To truly complete the derivation, we need to understand:

1. **Why g_I1 ≈ 1.0?**
   - I1 has log factor cross-terms that may "self-correct"
   - Needs analysis of I1 integrand structure

2. **Why g_I2 ≈ 1.02?**
   - I2 lacks log factor, needs full Beta moment correction
   - But 1.02 is ~1.4× the baseline correction, not 1.0×

3. **What determines the split?**
   - Q polynomial interaction (Phase 35) suggests answer is in Q structure
   - May need Euler-Maclaurin expansion analysis

---

## Key Files

| File | Description |
|------|-------------|
| `scripts/run_production_kappa.py` | Production evaluator (±0.15%, derived) |
| `src/evaluator/g_first_principles.py` | Phase 45 I1/I2 decomposition (calibrated) |
| `docs/PHASE_45_DERIVATION.md` | Phase 45 documentation (updated for honesty) |

---

## Terminology Clarification

- **DERIVED:** Obtained from mathematical analysis without using target values
- **CALIBRATED:** Obtained by solving equations with target values as inputs
- **Production formula:** Phase 36 with ±0.15% gap (genuinely derived)
- **Decomposition formula:** Phase 45 with ~0% gap (uses calibrated parameters)
