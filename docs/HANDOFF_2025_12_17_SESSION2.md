# Handoff Document: Session 2 - December 17, 2025

## Executive Summary

Multi-agent investigation deployed to resolve the **ratio reversal mystery**: PRZZ needs larger polynomials to produce smaller contributions (ratio κ/κ* = 0.94), but naive ∫P² gives opposite (ratio 1.71).

## Critical Findings from This Session

### 1. MISSING FACTORIAL NORMALIZATIONS (Agent: PRZZ TeX Analysis)

**Location**: `PRZZ_2008_Proof.tex` lines 570-619

The PRZZ paper uses **different normalizations for Case C pieces (ω > 0)**:

```
Case C (ω > 0): Contains factor 1/(ω-1)! in the b coefficients
Case C (ω = 0): No factorial factor
```

From the TeX:
- `b_{\kappa}(\ell)` for ω=0: Direct coefficient
- `b_{\kappa}(\ell)` for ω>0: Includes `\frac{(-1)^{\omega-1} ... (\omega-1)!}{(\omega-1)!}` normalization

**Key insight**: The "const" factor may NOT be just ∫Q(u)²e^{2Rt}dt but includes piece-dependent factorial normalizations that differ between κ and κ*.

### 2. DERIVATIVE TERMS MAKE RATIO WORSE (Agent: Derivative Analysis)

**DISPROVED HYPOTHESIS**: Adding derivative terms does NOT fix the ratio.

Numerical analysis with model `const_new = base - k * derivative`:
- At k≈0.43: Achieves ratio 0.94 BUT both const values become **negative** (-0.30, -0.32)
- Physical constraint: const must be positive (it's a squared integral contribution)
- Derivative subtraction can only reduce the ratio gap, never achieve correct ratio with valid signs

**Raw values**:
```
κ:  base_integral = 0.3626, derivative_term = 1.539
κ*: base_integral = 0.2119, derivative_term = 0.800
Base ratio: 1.71 (wrong direction)
Derivative ratio: 1.92 (even worse!)
```

### 3. POLYNOMIAL STRUCTURE DIFFERENCES

**κ benchmark** (R=1.3036):
- P₂, P₃: degree 3 polynomials
- Q: degree 5 polynomial
- Higher complexity

**κ* benchmark** (R=1.0788):
- P₂, P₃: degree 2 polynomials
- Q: degree 1 (LINEAR!) polynomial
- Much simpler structure

### 4. BASE POLYNOMIAL INTEGRALS

Computed ∫₀¹ Pᵢ(u)Pⱼ(u)du for both benchmarks:

**κ benchmark**:
```
∫P₁² = 0.0556, ∫P₂² = 0.0278, ∫P₃² = 0.0159
∫P₁P₂ = -0.0139, ∫P₁P₃ = 0.0079, ∫P₂P₃ = -0.0040
```

**κ* benchmark**:
```
∫P₁² = 0.0556, ∫P₂² = 0.0611, ∫P₃² = 0.0762
∫P₁P₂ = 0.0028, ∫P₁P₃ = 0.0079, ∫P₂P₃ = 0.0317
```

**Critical observation**: κ* has LARGER base integrals than κ for P₂,P₃ terms, yet needs SMALLER contribution. The ratio reversal must come from the exponential weighting or piece-dependent factors.

### 5. Ψ VALIDATION STILL HOLDS

The (1,1) case validation remains solid:
```
Ψ_sum = oracle_value = 0.359159 ✓
```

This confirms individual Ψ computations are correct. The issue is in how const factors combine.

## Working Hypotheses (Ordered by Likelihood)

### H1: Piece-Dependent Normalization (HIGH)
Each (ω₁,ω₂) piece has its own normalization factor involving factorials. The ratio 0.94 emerges from the **sum of normalized pieces**, not from a single ∫Q² integral.

### H2: Cancellation Structure (MEDIUM)
Cross-terms with negative Ψ values create cancellations that depend on polynomial degree structure. Higher-degree polynomials (κ) may have MORE cancellation.

### H3: Different Integral Bounds (LOW)
The t-integral may have polynomial-dependent bounds we haven't identified.

## Recommended Next Steps

1. **Extract full piece decomposition from PRZZ TeX**
   - Map each (ω₁,ω₂) combination to its normalization factor
   - Compute c as sum over pieces with correct normalizations

2. **Implement piece-wise c computation**
   - Create function that computes c_κ and c_κ* using piece-dependent factors
   - Verify ratio matches 0.94

3. **Trace the full derivation path**
   - Follow PRZZ paper from Ψ definition through to final c expression
   - Identify where factorial normalizations enter

## Code Locations

| File | Purpose |
|------|---------|
| `src/przz_22_exact_oracle.py` | Oracle c values and (2,2) case Ψ computation |
| `src/psi_monomial_evaluator.py` | Ψ₍ₘ,ₙ₎(ω₁,ω₂) evaluator |
| `src/t_integral_decomposition.py` | T-integral structure (I₁-I₄) |
| `data/przz_parameters.json` | κ benchmark: P₁,P₂,P₃,Q coefficients, R=1.3036 |
| `data/przz_parameters_kappa_star.json` | κ* benchmark: simpler polynomials, R=1.0788 |

## Key Equations

**PRZZ structure** (from previous session):
```
c = const × ∫ Q(u)² e^{2Rt} dt
```

**T-integral decomposition**:
```
I_total = I₁ + I₂ + I₃ + I₄
I₁ = base integral (∫∫ P_i P_j (1-u)^{ω₁+ω₂} du dt)
I₂ = Q-dependent term
I₃, I₄ = derivative terms
```

**The mystery ratio**:
```
κ/κ* target: 1.171 (from t-integrals)
const ratio needed: 0.942 (to match oracle c values)
Our naive const ratio: 1.71 (WRONG DIRECTION)
```

## Numerical Values

| Quantity | κ | κ* | Ratio |
|----------|---|----|----|
| Oracle c | 0.359159 | 0.261248 | 1.375 |
| R | 1.3036 | 1.0788 | 1.208 |
| t-integral (naive) | 0.3626 | 0.2119 | 1.71 |

## Files Created This Session

- `docs/HANDOFF_2025_12_17_SESSION2.md` (this file)

## Session Statistics

- 6 parallel agents deployed
- 2 agents completed with findings (TeX normalization, derivative analysis)
- 4 agents still running when context limit reached
- Key insight: Factorial normalizations in Case C pieces likely explain ratio reversal
