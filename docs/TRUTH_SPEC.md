# TRUTH_SPEC.md — Single Source of Truth for PRZZ c/κ Computation

**Purpose:** Pin the exact mathematical object we're computing, with explicit PRZZ TeX line references.

**Document Status:** AUTHORITATIVE — No code changes without matching this spec.

**TeX Source:** All line numbers refer to `RMS_PRZZ.tex` (1-indexed)

**Verification command:**
```bash
nl -ba RMS_PRZZ.tex | sed -n 'START,ENDp'
```

---

## 1. κ Definition

**PRZZ TeX Lines 286–289** (core definition: σ₀, c, and κ):
```latex
κ ≥ 1 - (1/R) log((1/T) ∫₁ᵀ |Vψ(σ₀+it)|² dt) + o(1)
```

**Implication:**
- `c` is the **asymptotic main term constant** from the mean square integral
- `c` does NOT include lower-order terms like I₅ (unless PRZZ explicitly keeps them)
- κ = 1 - log(c)/R

---

## 2. Published Target Values

```
θ = 4/7 ≈ 0.5714285714285714
R = 1.3036
c_target = 2.13745440613217263636
κ_target = 0.417293962
```

**Verification:**
```python
import math
kappa = 1 - math.log(2.13745440613217263636) / 1.3036
# kappa ≈ 0.4172939615...  ✓
```

---

## 3. Mollifier Definition

**PRZZ TeX Lines 542-545:**
```latex
ψ_{d=1}(s) = Σ_{n≤y₁} μ(n)n^{σ₀-1/2}/n^s Σ_{k=2}^K Σ_{p₁...p_k|n}
             (log p₁...log p_k)/(log^k y₁) P_{1,k}(log(y₁/n)/log(y₁))
```

**PRZZ TeX Line 548:**
> "Feng has set the convention of starting at K=2."

### Critical Indexing

| Our Poly | Our Index | PRZZ k | Λ convolutions | Description |
|----------|-----------|--------|----------------|-------------|
| P₁ | 1 | 2 | 1 | μ⋆Λ piece |
| P₂ | 2 | 3 | 2 | μ⋆Λ⋆Λ piece |
| P₃ | 3 | 4 | 3 | μ⋆Λ⋆Λ⋆Λ piece |

**Our code:** k starts at 1 (P₁, P₂, P₃)
**PRZZ paper:** k starts at 2 (k=2, 3, 4)
**Mapping:** Our index i corresponds to PRZZ k = i+1

---

## 4. I₅ is an Error Term

### 4a. Primary Citation: I₅ ≪ T/L

**PRZZ TeX Lines 1621–1628:**
```latex
The term (I_5) is of smaller order in (log N).
...
I₅ ≪ T/L.
Hence the term associated to A_{α,β}^{(1,1)}(0,0;β,α) is an error term.
```

This is the **canonical citation** for "I₅ is error term".

### 4b. Earlier Statement (same theme)

**PRZZ TeX Lines 1490–1499:**
```latex
I_{1,5} … prime sum …
This is of smaller order in log N.
```

### 4c. General Architecture Rule

**PRZZ TeX Lines 1714–1727:**

Any nonzero derivatives of A contribute only to the error term O(T/L).

Tightest slice for "derivatives → error" is **Lines 1722–1727**.

### Implication
- I₅ is **O(T/L)** — lower order than main term **O(T)**
- I₅ should NOT be used to calibrate/match the published κ
- If we need I₅ to hit κ, we're computing the **wrong main term**
- All A-derivative contributions are absorbed into error terms

### Enforcement
- `mode="main"`: I₅ forbidden, not even imported
- `mode="with_error_terms"`: Diagnostic only, prints warning

---

## 5. Mirror Combination Identity

### 5a. Headline Identity

**PRZZ TeX Lines 1502–1504:**

The main object is written as base term plus mirror term with T^{-α-β}.

### 5b. Difference Quotient → Integral Representation

**PRZZ TeX Lines 1502–1511:**
```latex
(N^{αx+βy} - T^{-α-β}N^{-βx-αy})/(α+β)
= N^{αx+βy} log(N^{x+y}T) ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

This removes the singularity at α+β=0 analytically.

### 5c. Q(...) Operator Structure

**PRZZ TeX Lines 1514–1517:**

The `Q(...)Q(...)` arguments appearing in the I_{1,1}(α,β) line.

### 5d. Evaluated at α=β=-R/L

**PRZZ TeX Lines 1521–1523:**
```latex
= log(T^{θ(x+y)+1}) ∫₀¹ Q(...)Q(...) e^{R[...]} e^{R[...]} dt
```

### Implication
- PRZZ does analytic combination **BEFORE** extracting constants
- The difference quotient 1/(α+β) becomes an integral representation
- This removes the singularity at α+β=0
- Expanding separately and adding later → **wrong constant term**

---

## 6. I₃/I₄ Prefactor

### 6a. I₃ Formula

**PRZZ TeX Lines 1551–1564:**
```latex
I₃ = -TΦ̂(0) × (1+θx)/θ × d/dx[∫∫ (1-u)P₁(x+u)P₂(u) e^{R[...]} Q(...)Q(...) dtdu]|_{x=0}
```

At x=0: (1+θx)/θ = 1/θ

**Prefactor:** -1/θ (NOT -1)

### 6b. I₄ Formula

**PRZZ TeX Lines 1566–1569:**
```latex
I₄ = -TΦ̂(0) × (1+θy)/θ × d/dy[∫∫ (1-u)P₁(u)P₂(y+u) e^{R[...]} Q(...)Q(...) dtdu]|_{y=0}
```

Same prefactor structure by symmetry.

### Variable Stage Note
**PRZZ TeX Line 2309:**
> "by the change of variable x → x log N"

The variable x in the I₃/I₄ formula may be at a different scaling stage.
FD oracle must operate at the same stage as the paper formula.

---

## 7. ω-Case Classification

### 7a. ω Definition and Case A + Variable Rescaling

**PRZZ TeX Lines 2301–2310:**
```latex
ω(d,l) = 1×l₁ + 2×l₂ + ... + d×l_d - 1
```

For d=1: ω = l₁ - 1

**Critical:** Line 2309 contains "by the change of variable x → x log N"

### 7b. Case B and C Definitions

**PRZZ TeX Lines 2320–2344:**

Contains the U/V/W indicator definitions and case split structure.

### Cases

| Case | Condition | Structure |
|------|-----------|-----------|
| A | ω = -1 | Simple derivative |
| B | ω = 0 | Direct polynomial evaluation |
| C | ω > 0 | Auxiliary a-integral required |

### Mapping to Our Polynomials

| Our Poly | PRZZ k | ω = k-2 | Case |
|----------|--------|---------|------|
| P₁ | k=2 | 0 | B |
| P₂ | k=3 | 1 | C |
| P₃ | k=4 | 2 | C |

**Warning:** P₂ and P₃ pairs are Case C and may need auxiliary a-integral!

---

## 8. Case C Auxiliary Integral

**PRZZ TeX Lines 2370–2375** (g_d(k,α,n) Case C formula):
```latex
∫₀¹ (1-a)^i a^{ω-1} (N/n)^{-αa} da
```

**Also relevant:** Lines 2378–2383 show the analogous F_d(k,α,n) structure.

This is a Beta-type integral that appears in Case C (ω > 0).

For P₂-involving pairs: ω = 1, so integral is ∫₀¹ (1-a)^i (N/n)^{-αa} da
For P₃-involving pairs: ω = 2, so integral is ∫₀¹ (1-a)^i a (N/n)^{-αa} da

### Implication
If our I₁-I₄ formulas are missing this auxiliary integral for Case C pieces,
we'll have systematic missing contributions from P₂/P₃-involving pairs.

---

## 8b. Prime Sum / S(0) Value

**PRZZ TeX Lines 1377–1389:**

Contains the prime-sum expression (over p) coming from A_{α,β}^{(1,1)} evaluated at zero.

Numeric approximation: **S(0) ≈ 1.385603705**

This is what we call `S_AT_ZERO` in the code (`src/arithmetic_constants.py`).

For wider context including the definition, use **Lines 1372–1389**.

---

## 8c. N = T^θ / y_d Definition

**PRZZ TeX Lines 624–635:**

Contains the statement where `y_d = N = T^{θ_d}` appears and the "θ = 4/7 − ε" context.

This defines the normalization relationship between T, N, and θ.

---

## 9. Normalizations

### Φ̂(0) (Test Function)
Assumed normalized: Φ̂(0) = 1

### Factorial Normalization
Pairs have factor 1/(ℓ₁! × ℓ₂!) from bracket combinatorics.

| Pair | Factor |
|------|--------|
| (1,1) | 1/1 = 1 |
| (2,2) | 1/4 |
| (3,3) | 1/36 |
| (1,2) | 1/2 |
| (1,3) | 1/6 |
| (2,3) | 1/12 |

Off-diagonal pairs also get symmetry factor 2.

### Q(0) Constraint
Q is a polynomial with Q(0) = 1 (endpoint normalization).

---

## 10. Target Formula Identification

**The exact PRZZ formula producing published c:**

From Section 6.2.1, after combining:
- I₁(α,β) + T^{-α-β}I₁(-β,-α) (mirror terms)
- I₂(α,β) + T^{-α-β}I₂(-β,-α)
- I₃(α,β) and I₄(α,β)

Then evaluating at α=β=-R/L and extracting:
- d²/dxdy at x=y=0 for I₁
- No derivatives for I₂
- d/dx at x=0 for I₃
- d/dy at y=0 for I₄

**Main constant is:** c = Σ pairs (with normalizations)

**"Main" mode means:** This sum, without I₅.

---

## 11. Verification Checklist

Before claiming "matched PRZZ":

1. [ ] FD oracle confirms DSL prefactor = -1/θ
2. [ ] Gap attribution shows which pairs contribute deficit
3. [ ] Piece ↔ ω mapping proven from mollifier definition
4. [ ] Case C auxiliary integral included if needed
5. [ ] Mirror combination done correctly
6. [ ] mode="main" produces c ≈ 2.137 WITHOUT I₅
7. [ ] Every factor has PRZZ equation reference

---

## References

All line numbers refer to: `RMS_PRZZ.tex` (1-indexed)

### Must-Quote Shortlist (Highest Load-Bearing Citations)

| Line(s) | Claim |
|---------|-------|
| 286–289 | κ / c definition |
| 1621–1628 | I₅ is error term (primary) |
| 1722–1726 | A-derivatives are error terms O(T/L) |
| 1502–1504 | Mirror term structure |
| 1551–1564 | I₃ displayed formula |
| 1566–1569 | I₄ displayed formula |
| 2301–2310 | ω-case split + x → x log N |

### Complete Reference Table

| Line(s) | Content |
|---------|---------|
| 286–289 | κ definition (core: σ₀, c, and κ ≥ 1 − (1/R) log c) |
| 542–545 | Mollifier definition |
| 548 | "Feng convention starting at K=2" |
| 624–635 | N = T^θ / y_d definition |
| 1377–1389 | Prime sum S(0) ≈ 1.385603705 |
| 1490–1499 | I_{1,5} smaller order statement |
| 1502–1504 | Mirror term headline identity |
| 1502–1511 | Difference quotient → integral representation |
| 1514–1517 | Q(...)Q(...) operator structure |
| 1521–1523 | Mirror combination at α=β=-R/L |
| 1551–1564 | I₃ formula with (1+θx)/θ prefactor |
| 1566–1569 | I₄ formula with (1+θy)/θ prefactor |
| 1621–1628 | I₅ ≪ T/L (explicit, canonical citation) |
| 1714–1727 | A-derivatives contribute only to O(T/L) error |
| 1722–1726 | Tightest slice for "derivatives → error" |
| 2301–2310 | ω definition + Case A + "x → x log N" |
| 2320–2344 | Case B + Case C definitions (U/V/W indicators) |
| 2370–2375, 2378–2383 | Case C auxiliary a-integral structure |
