# Derivation of Zeta Log-Derivative Factor

**Date:** 2025-12-24
**Phase:** 18.3
**Status:** DOCUMENTED

---

## 1. Problem Statement

The J12 bracket term in PRZZ involves the product of zeta log-derivatives:

```
(ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)
```

At the evaluation point s=u=0 with α=β=-R, this becomes `(ζ'/ζ)(1-R)²`.

The question: should we use the **Laurent approximation** or the **actual numerical value**?

---

## 2. Laurent Expansion Origin

The Laurent expansion of ζ'/ζ near s=1 is:

```
(ζ'/ζ)(1+ε) = -1/ε + γ_E + O(ε)
```

where γ_E ≈ 0.5772 is the Euler-Mascheroni constant.

At ε = -R (i.e., s = 1-R):

```
(ζ'/ζ)(1-R) ≈ -1/(-R) + γ_E = 1/R + γ_E
```

For J12, we need the squared value:

```
(ζ'/ζ)²(1-R) ≈ (1/R + γ_E)²
```

---

## 3. Laurent Approximation Error at PRZZ R Values

The Laurent expansion is an asymptotic expansion valid for |ε| << 1.
At the PRZZ evaluation points, ε = -R where R ≈ 1.1-1.3, which is NOT small.

**Phase 15A Investigation Results (from g_product_full.py):**

| Benchmark | R | s = 1-R | Laurent (1/R+γ) | Actual (ζ'/ζ) | Error |
|-----------|------|---------|-----------------|---------------|-------|
| κ | 1.3036 | -0.3036 | 1.344 | 1.732 | **29%** |
| κ* | 1.1167 | -0.1167 | 1.473 | 1.778 | **21%** |

**Squared values (what J12 uses):**

| Benchmark | Laurent² | Actual² | Ratio | % Increase |
|-----------|----------|---------|-------|------------|
| κ | 1.81 | 3.00 | 1.66 | **+66%** |
| κ* | 2.17 | 3.16 | 1.46 | **+46%** |

---

## 4. Why This Matters for the +5 Signature

The mirror assembly formula is:

```
c = I₁₂(+R) + m × I₁₂(-R) + I₃₄(+R)
```

where m = exp(R) + 5 for K=3.

The "+5 signature" comes from the ratio B/A where:
- A = I₁₂(-R) [the exp(R) coefficient]
- B = I₁₂(+R) + I₃₄(+R) + 5×I₁₂(-R)

The J12 factor appears in both I₁₂(+R) and I₁₂(-R). If the factor is wrong by 66%,
this directly affects both the numerator (D = I₁₂(+R) + I₃₄(+R)) and denominator (A).

**Effect on delta = D/A:**

Since both D and A contain J12 contributions, the error partially cancels.
However, J13/J14 (which are part of I₃₄) use only a SINGLE (ζ'/ζ) factor,
creating asymmetry between κ and κ*.

---

## 5. J13/J14 Single Factor (Phase 16 Discovery)

J13 and J14 use a SINGLE (ζ'/ζ) factor, not squared:

| Benchmark | Laurent single | Actual single | Error |
|-----------|----------------|---------------|-------|
| κ | 1.344 | 1.732 | 29% |
| κ* | 1.473 | 1.778 | 21% |

The 29% vs 21% error difference explains the κ/κ* asymmetry:
- κ gets a larger ACTUAL correction
- κ* gets a smaller correction
- Result: asymmetry ratio of 1.89 in J13/J14 pieces

---

## 6. Semantic Decision (Decision 8)

**ACTUAL_LOGDERIV is the correct mode for production because:**

1. The Laurent expansion is only valid for |ε| << 1
2. At R=1.3, ε = -R ≈ -1.3 which is NOT small
3. The Laurent series diverges in this regime
4. Numerical evaluation via mpmath gives the true value

**Two modes are now documented (Decision 8):**

1. **Semantic Mode (RAW_LOGDERIV)**
   - Uses (1/R + γ)² Laurent approximation
   - Matches what the TeX formula structure literally says
   - Use for: theoretical validation, asymptotic analysis

2. **Numeric Mode (ACTUAL_LOGDERIV)**
   - Uses actual numerical (ζ'/ζ)(1-R)² via mpmath
   - Best accuracy at finite R
   - Use for: production κ computation, benchmark matching

---

## 7. Implementation Details

**File:** `src/ratios/g_product_full.py`

```python
def compute_zeta_logderiv_actual(R: float, precision: int = 50) -> float:
    """Single (ζ'/ζ)(1-R) for J13/J14."""

def compute_j12_actual_logderiv_squared(R: float, precision: int = 50) -> float:
    """Squared (ζ'/ζ)²(1-R) for J12."""
```

**Usage in j1_euler_maclaurin.py:**

```python
if laurent_mode == LaurentMode.ACTUAL_LOGDERIV:
    factor = compute_j12_actual_logderiv_squared(abs(R))
elif laurent_mode == LaurentMode.RAW_LOGDERIV:
    factor = (1/abs(R) + EULER_MASCHERONI) ** 2
```

---

## 8. Remaining ~1.35% Gap

Even with ACTUAL_LOGDERIV, a ~1.35% gap remains:

| Benchmark | c gap | κ gap |
|-----------|-------|-------|
| κ | -1.35% | +2.50% |
| κ* | -1.21% | +2.67% |

**This remaining gap is NOT from the Laurent approximation** (that's been fixed).

Possible remaining sources:
1. m1 formula not fully derived (empirical exp(R) + 5)
2. Polynomial transcription errors
3. Missing structural factor in the formula

---

## 9. Open Question: What is the "Correct" Finite-R Evaluation?

The paper uses asymptotic expansions valid for large L = log(T).
At finite R, which evaluation is truly correct?

**Hypothesis 1:** ACTUAL_LOGDERIV is correct because it evaluates the same
mathematical object (ζ'/ζ) at the same point, just more accurately.

**Hypothesis 2:** The Laurent approximation captures the "right" scaling
and the numerical difference is absorbed into other factors.

Current evidence supports Hypothesis 1: ACTUAL_LOGDERIV reduces the gap
from ~5% to ~1.35% for both benchmarks.

---

## 10. Verification Tests

See `tests/test_j12_c00_semantics.py`:

```python
def test_raw_logderiv_matches_literal_structure():
    """RAW mode matches the TeX structure (1/R + γ)²."""

def test_actual_logderiv_via_mpmath():
    """ACTUAL mode matches mpmath computation."""
```

---

## References

- **PRZZ TeX lines 2391-2409:** J12 bracket structure with ζ'/ζ product
- **Phase 15A:** Discovery of 22-29% Laurent error
- **Phase 16:** Extension to J13/J14 single factor
- **Decision 7:** RAW_LOGDERIV is semantically correct for structure
- **Decision 8:** ACTUAL_LOGDERIV for production accuracy
