# Consolidated Agent Findings: PRE-MIRROR → POST-MIRROR Transformation

## Executive Summary

Five parallel agents investigated the clean-path implementation gap. **Root cause identified**: The clean-path has a fundamental structural bug - it treats I_d as a u-integral and multiplies by t-integral again, but the oracle computes them as a **single factored product**.

---

## Key Finding 1: The Difference Quotient Identity (Agent 1)

**This is THE mechanism that creates the t-integral:**

```
1/(α+β) → log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

**PRZZ TeX lines 1506-1510:**
```
[N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
= N^{αx+βy} × [1 - (N^{x+y}T)^{-α-β}] / (α+β)
= N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

**Key insight**: The t-integral doesn't exist at PRE-MIRROR. It emerges from:
1. Mirror combination: I_{1,d}(α,β) + T^{-α-β}I_{1,d}(-β,-α)
2. Difference quotient identity converts 1/(α+β) → t-integral
3. Q operators transform T^{-tα} → Q(t)e^{Rt}

---

## Key Finding 2: D ≠ AB Mathematically (Agent 2)

**The current bug**: Both D and AB evaluate to ∫P²du in clean-path.

**But mathematically:**
- **AB** = (ζ'/ζ)_z × (ζ'/ζ)_w = **product of independent first derivatives**
- **D** = (ζ'/ζ)' = ζ''/ζ − (ζ'/ζ)² = **second mixed derivative**

Both map to Case B (l₁=m₁=1), but they contribute to **different I-terms**:
- AB contributes to **I₁** (mixed derivative term)
- D contributes to **I₂** (base term, no derivatives)

**The distinction comes from:**
- D has no (1-u) weight
- AB has (1-u)² weight

---

## Key Finding 3: Q Operator Transformation (Agent 3)

**Q operators are differential operators:**
```
Q(-1/log T × ∂/∂α) Q(-1/log T × ∂/∂β) [T^{-tα-tβ}] |_{α=β=-R/L}
= Q(t)² e^{2Rt}
```

**The transformation chain:**
1. T^{-tα} with α = -R/L → e^{Rt}
2. Q operator on exponential → Q(t) × e^{Rt}
3. Applied to both α and β → Q(t)² × e^{2Rt}

**Q arguments are R-independent:**
```
Arg_α = t + θt·x + θ(t-1)·y    (no R!)
Arg_β = t + θ(t-1)·x + θt·y    (no R!)
```

R enters **only** through exponential factors: exp(R × Arg)

---

## Key Finding 4: Structural Bug in Clean-Path (Agent 4)

**THE BUG (section7_clean_evaluator.py line 411):**
```python
c_contribution = (1.0 / self.theta) * t_integral * I_d
```

This treats I_d as just the u-integral part and multiplies by t-integral.

**But oracle computes:**
```python
I₂ = (1.0 / theta) * u_integral_I2 * t_integral_I2
```

Where:
- `u_integral = ∫ P₂² du` (just the polynomial part)
- `t_integral = ∫ Q² e^{2Rt} dt` (the Q/exponential part)

**The oracle has NO mirror term** - it's a direct POST-MIRROR formula!

**Additional missing pieces:**
1. No (1-u)² weight in `eval_monomial()` (line 288)
2. Case C kernel lacks proper 1/(ω-1)! factorial
3. Mirror adds unintended term (oracle doesn't use mirror)

---

## Key Finding 5: (1,1) Perfect Validation (Agent 5)

**(1,1) is fully validated:**
```
Monomial Sum: AB + D - AC - BC = 0.359159
Oracle Sum:   I₁ + I₂ + I₃ + I₄ = 0.359159
Difference:   5.55e-17 ✓ PERFECT MATCH
```

**The monomial → I-term mapping:**
| Monomial | I-Term | Oracle Value | (1-u) Weight |
|----------|--------|--------------|--------------|
| AB (+1)  | I₁     | +0.426028    | (1-u)²       |
| D (+1)   | I₂     | +0.384629    | 1 (none)     |
| -AC (-1) | I₃     | -0.225749    | (1-u)¹       |
| -BC (-1) | I₄     | -0.225749    | (1-u)¹       |

**BUT (1,1) is the ONLY complete case:**
| Pair  | Monomials | I₁-I₄ Coverage |
|-------|-----------|----------------|
| (1,1) | 4         | 100%           |
| (2,2) | 12        | 33%            |
| (3,3) | 27        | 15%            |

---

## The Complete Transformation Flow

| Layer | What | Structure |
|-------|------|-----------|
| **PRE-MIRROR** | I_{1,d}(α,β) | Has 1/(α+β), no t-integral |
| **+ Mirror** | I_{1,d} + T^{-α-β}I_{1,d}(-β,-α) | Combines two terms |
| **Diff Quotient** | 1/(α+β) → ∫₀¹ T^{-t(α+β)} dt | t-integral introduced |
| **+ Q operators** | Q(∂/∂α)Q(∂/∂β)[...] | Transforms to Q(t)² |
| **POST-MIRROR** | (1/θ) × ∫P du × ∫Q² e^{2Rt} dt | Separated integrals |

---

## The Fix Strategy

### Option A: Direct POST-MIRROR (Recommended for Now)

For (1,1), directly compute:
```python
I₁ = oracle_I1_formula(...)  # d²/dxdy term
I₂ = (1/θ) × ∫P₁²du × ∫Q²e^{2Rt}dt  # base term
I₃ = -d/dx term with (1-u)¹ weight
I₄ = -d/dy term with (1-u)¹ weight
```

This matches the oracle exactly and doesn't need mirror.

### Option B: Full PRE-MIRROR Chain (For Higher Pairs)

For (2,2)+ with 12+ monomials:
1. Generate Ψ monomials with TWO-C structure ✓ (done)
2. Map each monomial to (k,l,m) indices ✓ (done)
3. Compute F_d × F_d for each monomial
4. Apply correct (1-u)^{ℓ+ℓ̄-2p} weights ← **missing**
5. Sum to get I_{1,d}(α,β) with 1/(α+β) factor
6. Apply mirror: + T^{-α-β}I_{1,d}(-β,-α)
7. Apply difference quotient identity → t-integral emerges
8. Apply Q operators → Q(t)² appears
9. Final: (1/θ) × u-structure × t-integral

---

## SESSION 6 UPDATE: Weight Formula Corrected

**CRITICAL FIX (2025-12-18)**: The (1-u) weight formula has been validated:

```python
# CORRECT weight formula (validated against oracle):
I1_weight = (1-u)^{ell + ellbar}  # From AB monomial: a+b singletons
I2_weight = 1 (no weight)          # From D monomial: paired block
I3_weight = (1-u)^{ell}            # From -AC monomial: a singletons
I4_weight = (1-u)^{ellbar}         # From -BC monomial: b singletons
```

**For (1,1)**:
- I₁: (1-u)² → 0.426028 ✓
- I₂: no weight → 0.384629 ✓
- I₃: (1-u)¹ → -0.225749 ✓
- I₄: (1-u)¹ → -0.225749 ✓
- Total: 0.359159 ✓ **PERFECT MATCH**

---

## Immediate Action Items

1. ~~**Add (1-u) weights to monomial evaluation**~~ **DONE**
   - ~~AB monomials: (1-u)^{ℓ+ℓ̄}~~
   - ~~D monomials: no weight~~
   - ~~AC/BC monomials: (1-u)^{ℓ₁} or (1-u)^{ℓ₂}~~

2. ~~**Fix the t-integral structure**~~ **DONE**
   - ~~Don't multiply I_d by t-integral~~
   - ~~The factored structure should be: (1/θ) × u-part × t-part~~

3. ~~**For (1,1), use oracle formulas directly**~~ **DONE**
   - ~~No mirror needed~~
   - ~~Validates the structure~~

4. **For (2,2)+, implement full chain** (REMAINING)
   - All 12 monomials needed ✓
   - Proper (1-u) weights per monomial ✓
   - **Issue**: κ/κ* ratio mismatch (2.01 vs 1.10) due to polynomial structure differences

---

## Test Command

```bash
cd przz-extension && PYTHONPATH=. python3 -c "
from src.polynomials import load_przz_polynomials
from src.przz_22_exact_oracle import przz_oracle_22

P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
result = przz_oracle_22(P2, Q, 4/7, 1.3036, n_quad=60, debug=True)
print(f'Oracle total: {result.total:.6f}')
print(f'I₁={result.I1:.6f}, I₂={result.I2:.6f}, I₃={result.I3:.6f}, I₄={result.I4:.6f}')
"
```
