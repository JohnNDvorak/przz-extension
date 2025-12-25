# GPT Run 5 Handoff (2025-12-20)

## Overview

This document classifies the codebase into **proven** vs **aspirational** components,
following GPT Run 5 guidance to prevent calibration fixes from silently becoming "truth".

---

## Classification: Proven vs Aspirational

### PROVEN (Hard-gated, trust these)

These components have been validated with structural tests and should be treated as ground truth.

| Component | Evidence | Test File |
|-----------|----------|-----------|
| **Ordered pairs required** | S34 asymmetry test shows Δ_S34 = 0.54 ≠ 0 | `test_operator_hard_gates.py::TestS34AsymmetryGate` |
| **S12 symmetry holds** | Δ_S12 = 0.000000 (exact) | `test_operator_hard_gates.py::TestS12SymmetryGate` |
| **Operator implied weights stable** | m_implied ≈ 1.04, span < 2% | `test_operator_hard_gates.py` |
| **I2 separability** | u and t integrals factor exactly | `run_gpt_run5_direct_i2.py` |
| **A₂ - A₁ = K - 1** | Structural invariant, exact | `test_tex_amplitudes.py::TestExpComponentModes::test_a_diff_invariant_across_modes` |
| **Q=1 case validation** | E[exp(2Rt)] = (exp(2R)-1)/(2R) | `test_tex_amplitudes.py::TestQEqualsOneCase` |

### ASPIRATIONAL (Calibration, use with care)

These components achieve good benchmark accuracy but are NOT proven from first principles.

| Component | Classification | Warning |
|-----------|----------------|---------|
| **tex_amplitudes()** | TeX-motivated surrogate | Formula `A = exp(...) + K-1 + ε` is NOT derived from TeX |
| **exp_R_ref mode** | **CALIBRATION ONLY** | Matches benchmarks by freezing R at 1.3036; no TeX justification |
| **compute_c_paper_tex_mirror()** | Aspirational evaluator | Assembly uses calibrated amplitudes |
| **Shape × Amplitude factorization** | Empirical model | `m_full = m_implied × A` works but isn't proven |

### WARNING MARKERS IN CODE

The following docstrings now contain explicit warnings:

1. **`tex_amplitudes()` (src/evaluate.py:4914)**
   > Warning: The exp_R_ref mode is a CALIBRATION fix, not TeX-derived.

2. **`compute_c_paper_tex_mirror()` (src/evaluate.py:5129)**
   > Classification: This is an ASPIRATIONAL evaluator, not proven PRZZ reproduction.

### TEST MARKERS

Tests are now classified with pytest markers:

- `@pytest.mark.calibration`: Tests for benchmark accuracy under heuristic modes
- No marker: Structural tests (invariants, architecture)

To run only structural tests:
```bash
pytest -m "not calibration"
```

---

## GPT Run 5 Findings

### 1. exp_R_ref Preserves R-Optimum

The R-sweep diagnostic (`run_gpt_run5_r_optimum_sweep.py`) found:

- Both `exp_R` and `exp_R_ref` modes find R_opt = 1.6 (edge of sweep)
- exp_R_ref does NOT distort the optimum relative to exp_R
- The issue is that our computed R_opt differs from PRZZ's stated values

| Benchmark | Expected R | Mode | R_opt | Deviation |
|-----------|------------|------|-------|-----------|
| κ | 1.3036 | exp_R | 1.60 | +0.30 |
| κ | 1.3036 | exp_R_ref | 1.60 | +0.30 |
| κ* | 1.1167 | exp_R | 1.60 | +0.48 |
| κ* | 1.1167 | exp_R_ref | 1.60 | +0.48 |

### 2. Direct TeX I2 Evaluation

The diagnostic (`run_gpt_run5_direct_i2.py`) reveals:

- **Direct I2 values differ from model by 2-2.5x**
  - κ: Direct I2_plus = 1.77 vs Model I2_plus = 0.71
  - κ*: Direct I2_plus = 1.03 vs Model I2_plus = 0.49

- **The simple mirror formula I2(+R) + exp(2R)×I2(-R) doesn't match model**
  - The exp(2Rt) is INSIDE the integral, not a standalone multiplier
  - Q² concentration near t=0 dramatically reduces the effective exp factor

- **Amplitude model compensates for multiple effects**
  - exp(2Rt) inside integral
  - Q² weighting effect
  - Polynomial normalization factors

### 3. Channel Attribution (Run 4)

Already documented in DECISION_TABLE_GPT_RUN4.md:
- I1 contributes 33% of the κ* deficit
- I2 contributes 67% of the κ* deficit

---

## Path Forward

### Immediate Actions

1. **Do not claim PRZZ reproduction from first principles**
   - The exp_R_ref mode is calibration, not derivation
   - Use for benchmark reproduction only

2. **For R-sweep/optimization, use exp_R mode**
   - exp_R_ref is calibrated at R=1.3036 only
   - It gives worse results at other R values

3. **Keep calibration tests separate from structural tests**
   - Use pytest markers to distinguish
   - Don't let calibration accuracy mask structural bugs

### Future Work

To achieve direct TeX evaluation (removing amplitude model):

1. **Understand the mirror recombination formula**
   - TeX lines 1502-1548 show T^{-α-β} factor
   - At α=β=-R/L, this gives exp(2R), but...
   - The exp factor is INSIDE the integral, affecting shape

2. **Factor out what amplitude model captures**
   - It's a surrogate for: integral normalization, Q² weighting, factorial norms
   - Direct TeX evaluation should compute these explicitly

3. **Start with I2 (separable)**
   - I2 = [∫ P₁P₂ du] × [(1/θ) ∫ Q² exp(2Rt) dt]
   - Compare direct evaluation with model prediction
   - Identify what's missing in the model

---

## Files Created/Modified in Run 5

| File | Changes |
|------|---------|
| `src/evaluate.py` | Added warning docstrings |
| `tests/test_tex_amplitudes.py` | Added `@pytest.mark.calibration` marker |
| `pytest.ini` | Registered `calibration` and `slow` markers |
| `run_gpt_run5_r_optimum_sweep.py` | R-sweep diagnostic |
| `run_gpt_run5_direct_i2.py` | Direct TeX I2 diagnostic |
| `docs/HANDOFF_GPT_RUN5.md` | This file |

---

## Summary

**GPT's Warning (verbatim):**

> Treat `exp_R_ref` as **a calibrated stopgap** (useful and worth keeping), but do **not** declare victory on "reproduced PRZZ from first principles" until you can point to a specific TeX step that forces that choice (or until you remove the need for it by direct TeX evaluation of the mirror piece).

**Current Status:**
- 21 tests pass in test_tex_amplitudes.py
- exp_R_ref achieves <1% error on both benchmarks
- But exp_R_ref is CALIBRATION, not TeX-derived

**Next GPT Run should focus on:**
- Understanding why R_opt = 1.6 instead of 1.3036
- Direct TeX I2 evaluation to replace amplitude model
- Finding the TeX prescription for mirror recombination
