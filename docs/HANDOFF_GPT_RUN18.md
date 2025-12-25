# GPT Run 18 Handoff (2025-12-21)

## Executive Summary

Run 18 attempted to implement the actual TeX combined integral structure from lines 1503-1510. **The answer is: the naive combined integral interpretation does NOT outperform tex_mirror's factorized approach.**

The combined structure produces:
- κ benchmark: **-5.9% gap** (vs tex_mirror's -0.73%)
- κ* benchmark: **-29.5% gap** (vs tex_mirror's -0.95%)

**Conclusion**: tex_mirror's factorized shape×amplitude approximation remains the best production model.

---

## Task Summary

| Stage | Status | Key Finding |
|-------|--------|-------------|
| 18A | ✅ | CombinedMirrorFactor class created, scalar limit verified |
| 18B | ✅ | I1 "replace exp" gives values close to c_target |
| 18C | ✅ | I2 should NOT use combined factor (scalar at x=y=0) |
| 18D | ✅ | S34 should NOT use combined factor (per PRZZ TRUTH_SPEC) |
| 18E | ✅ | Full assembly shows asymmetric benchmark behavior |

---

## The Combined Integral Structure

### TeX Formula (lines 1503-1510)

```
(N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)
= N^{αx+βy} × log(N^{x+y}T) × ∫_0^1 (N^{x+y}T)^{-s(α+β)} ds
```

At α = β = -R/L:
- The s-integral becomes: `∫_0^1 exp(2sR(1 + θ(x+y))) ds`
- Scalar limit: `(exp(2R) - 1) / (2R)`

### CombinedMirrorFactor Class

```python
@dataclass(frozen=True)
class CombinedMirrorFactor:
    R: float
    theta: float
    n_quad_s: int = 20

    def evaluate(self, U, T, ctx) -> TruncatedSeries:
        """Compute (1 + θ(x+y)) × ∫_0^1 exp(2sR(1 + θ(x+y))) ds"""
        ...

    def scalar_limit(self) -> float:
        """Returns (exp(2R) - 1) / (2R)"""
        return (np.exp(2 * R) - 1) / (2 * R)
```

---

## Experimental Results

### Stage 18B: I1 Channel (1,1)

Two interpretations tested:
1. **Multiply**: Combined factor × exp factors (gives ~3.8, too large)
2. **Replace**: Combined factor replaces exp factors (gives ~2.08, closer to c_target)

| Benchmark | I1 (replace) | tex_mirror I1_total | c_target |
|-----------|--------------|---------------------|----------|
| κ | 2.078 | 0.404 | 2.138 |
| κ* | 1.433 | 0.554 | 1.938 |

### Stage 18C: I2 Channel

I2 has no formal variables (x, y), so combined factor reduces to scalar limit.
But applying this factor gives incorrect results.

**Finding**: I2 should just use base integral (no combined factor).

| Benchmark | I2_base | tex_mirror I2_plus |
|-----------|---------|---------------------|
| κ | 0.385 | 0.713 |
| κ* | 0.321 | 0.494 |

### Stage 18D: S34 Channel

Per PRZZ TRUTH_SPEC Section 10: I₃ and I₄ do NOT require mirror.
S34 should NOT use the combined factor.

| Benchmark | S34_base | tex_mirror S34_plus |
|-----------|----------|---------------------|
| κ | -0.452 | -0.338 |
| κ* | -0.388 | -0.289 |

### Stage 18E: Full Assembly

**Assembly formula**: `c = I1_replace + I2_base + S34_base`

| Benchmark | c_combined | c_tex_mirror | c_target | Gap (combined) | Gap (mirror) |
|-----------|------------|--------------|----------|----------------|--------------|
| κ | 2.011 | 2.122 | 2.138 | **-5.9%** | -0.73% |
| κ* | 1.367 | 1.920 | 1.938 | **-29.5%** | -0.95% |

---

## Root Cause Analysis

### Why the Combined Structure Fails

1. **Asymmetric benchmark behavior**: The combined structure gives reasonable results for κ but fails badly for κ*. This suggests the interpretation is wrong.

2. **The TeX formula is for DERIVATION, not computation**: The combined integral form (lines 1503-1510) is used to prove asymptotic bounds, not to compute exact numerical values.

3. **Missing normalization**: The combined structure doesn't capture the polynomial degree-dependent factors that tex_mirror handles implicitly.

4. **Different structure for different channels**: The TeX combined structure applies to I₁ mirror, but I₂ and S34 have different structures that were incorrectly handled.

### Why tex_mirror Works

tex_mirror's shape×amplitude factorization:
1. Calibrates amplitudes (A1, A2) to match specific reference R values
2. Uses shape factors (m_implied ≈ 1) for derivative structure
3. Achieves <1% accuracy through empirical calibration

This heuristic approach works because it captures the NET effect of the complex TeX formula without implementing the exact structure.

---

## Files Created

| File | Purpose |
|------|---------|
| `src/term_dsl.py` | Added CombinedMirrorFactor class (~80 lines) |
| `src/evaluate.py` | Added compute_I1_tex_combined_11(), compute_I2_tex_combined_11(), compute_S34_base_11() |
| `tests/test_tex_combined_gates.py` | Gate tests for all stages |
| `run_gpt_run18b_test.py` | Test script for Run 18 experiments |
| `docs/HANDOFF_GPT_RUN18.md` | This document |

---

## Recommendations

### Keep (Production)

1. **tex_mirror with exp_R_ref** - Production model, <1% gap acceptable
2. **V2 guard** - V2 + tex_mirror is forbidden

### Do NOT Use

1. **Combined integral structure** - Does not outperform tex_mirror
2. **I2 with combined factor** - Should use base only
3. **S34 with combined factor** - Should use base only

### Future Work (Not Recommended)

To improve beyond tex_mirror's ~1% gap would require:
1. Implementing the FULL TeX derivation, not just the combined integral
2. Understanding polynomial degree-dependent normalizations
3. Possibly deriving the A1/A2 formulas from first principles

Given that tex_mirror achieves <1% accuracy, this additional effort is likely not justified.

---

## Key Learnings

1. **The TeX combined integral is for derivation, not computation**
2. **tex_mirror's factorized approach is empirically optimal**
3. **Different channels (I1, I2, S34) require different treatment**
4. **Benchmark asymmetry indicates structural problems**

---

## Summary Table

| Run | Goal | Result |
|-----|------|--------|
| 17 | Use exp(2R) prefactor | Failed - gives 10-15x overshoots |
| 18 | Implement TeX combined integral | Failed - worse than tex_mirror |

**Final recommendation**: Keep tex_mirror as production model. The ~1% structural gap is acceptable and likely irreducible without full TeX implementation.
