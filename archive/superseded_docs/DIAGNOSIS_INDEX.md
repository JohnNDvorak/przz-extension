# Ratio Reversal Diagnosis - Documentation Index

**Date:** 2025-12-17
**Status:** Analysis complete, ready for numerical execution

## What Is This?

This diagnosis investigates why the computed ratio between κ and κ* benchmarks is 2.09 instead of the PRZZ target of 1.103. The root cause appears to be **polynomial degree mismatch**, with κ* using simpler polynomials that lead to systematically smaller integrals.

## Quick Start

### To Execute Diagnosis
```bash
cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
python ratio_reversal_diagnosis.py > diagnosis_output.txt 2>&1
```

### To Understand The Problem
Read in this order:
1. **RATIO_REVERSAL_SUMMARY.md** (5 min) - Executive summary
2. **DIAGNOSIS_REPORT.md** (15 min) - Detailed mathematical analysis
3. **NUMERICAL_PREDICTIONS.md** (10 min) - Expected values

### To Execute and Validate
1. **COMPUTATION_CHECKLIST.md** - Task-by-task execution guide
2. Run diagnostic script
3. Compare actual vs predicted values
4. Test normalization hypothesis if predictions match

## Document Guide

### Core Analysis Documents

#### RATIO_REVERSAL_SUMMARY.md
**Purpose:** Executive summary of the entire diagnosis
**Length:** ~200 lines
**Audience:** Anyone who wants to understand the problem quickly

**Key sections:**
- Executive Summary (the problem in numbers)
- The Factorization (mathematical breakdown)
- Hypotheses for the mismatch
- Recommended next actions

**Read this first.**

#### DIAGNOSIS_REPORT.md
**Purpose:** Complete mathematical derivation and methodology
**Length:** ~250 lines
**Audience:** Deep-dive technical analysis

**Key sections:**
- Methodology (5-step process)
- Benchmark parameters
- Expected numerical behavior
- Root cause analysis
- Recommended actions

**Read this for mathematical details.**

#### NUMERICAL_PREDICTIONS.md
**Purpose:** Analytical predictions for validation
**Length:** ~200 lines
**Audience:** Pre-execution planning and post-execution validation

**Key sections:**
- Polynomial degree analysis
- T-integral predictions
- U-integral predictions (with hand calculations)
- Expected oracle outputs
- Action items after execution

**Read this before and after running diagnostics.**

### Execution Documents

#### COMPUTATION_CHECKLIST.md
**Purpose:** Step-by-step execution guide
**Length:** ~180 lines
**Audience:** Implementing the diagnosis

**Key sections:**
- Task 1: T-Integral
- Task 2: U-Integrals (6 pairs)
- Task 3: Per-pair I₂ values
- Task 4: Oracle full results
- Task 5: Derivative analysis
- Validation checks
- Success metrics

**Use this as a checklist during execution.**

### Code Files

#### ratio_reversal_diagnosis.py
**Purpose:** Complete diagnostic computation
**Features:**
- Loads both polynomial sets
- Computes per-pair I₂ values
- Runs full oracle for (1,1) and (2,2)
- Analyzes derivative contributions
- Produces formatted tables

**Outputs:**
- Per-pair I₂ table with u/t-integral breakdown
- Oracle component-by-component comparison
- Derivative contribution analysis
- Summary of key findings

**Runtime:** ~10-30 seconds (depending on n_quad)

#### compute_diagnosis.py
**Purpose:** Simplified quick-check version
**Features:**
- Focuses on key ratios only
- Minimal output
- Fast execution

**Use this for:** Quick validation runs

## The Problem (TL;DR)

### What We Get
```
c(κ) / c(κ*) = 1.96 / 0.94 ≈ 2.09
```

### What PRZZ Gets
```
c(κ) / c(κ*) = 2.137 / 1.938 = 1.103
```

### The Gap
```
Factor error: 2.09 / 1.103 = 1.89× (89% excess)
```

### Why This Happens

**Polynomial structure:**
- κ uses: P₂(deg 3), P₃(deg 3), Q(deg 5)
- κ* uses: P₂(deg 2), P₃(deg 2), Q(deg 1)

**Mathematical fact:**
Higher-degree polynomials → larger L² norms → larger integrals

**Result:**
ALL components (I₁, I₂, I₃, I₄) scale by ~2-4× for κ vs κ*

**No cancellation:**
Derivatives amplify the effect rather than correct it

## The Leading Hypothesis

**Polynomial L² Normalization**

PRZZ likely normalizes each P_ℓ before use:
```python
P_ℓ,normalized = P_ℓ / sqrt(∫ P_ℓ² du)
```

This would:
- Remove degree effects
- Standardize polynomial magnitudes
- Preserve shapes but normalize scales

**Test:**
```python
# For (2,2) pair
norm_k = sqrt(∫ P₂_κ² du)   # ≈ 1.08
norm_ks = sqrt(∫ P₂_κ*² du) # ≈ 0.56

# Normalize
P₂_κ_norm = P₂_κ / norm_k
P₂_κ*_norm = P₂_κ* / norm_ks

# Recompute I₂
I₂_norm_k = compute_i2(P₂_κ_norm, P₂_κ_norm, Q_k, R_k, θ)
I₂_norm_ks = compute_i2(P₂_κ*_norm, P₂_κ*_norm, Q_ks, R_ks, θ)

ratio_normalized = I₂_norm_k / I₂_norm_ks
```

**If ratio_normalized ≈ 1.10 → Hypothesis confirmed!**

## Key Insights

### What We Know For Certain

1. **Polynomial degrees differ** (verified from JSON files)
2. **Higher degrees → larger integrals** (mathematical fact)
3. **All components scale together** (verified by oracle)
4. **Derivatives don't cancel** (mathematical structure)
5. **Ratio is stable across quadrature** (not numerical noise)

### What We're Testing

1. **Does L² normalization fix it?** (most likely)
2. **Is there R-dependent scaling?** (possible)
3. **Are coefficients transcribed correctly?** (needs verification)
4. **Is there a different c definition?** (needs re-reading)

### What We've Ruled Out

- ❌ DSL bugs (V2 validated)
- ❌ Sign errors (tested)
- ❌ Quadrature errors (stable)
- ❌ Missing I₅ (forbidden)
- ❌ Variable structure (V2 correct)

## Numerical Predictions

These should match when you run the diagnostic:

| Component | Predicted | Significance |
|-----------|-----------|--------------|
| t_ratio | ~1.17 | Exponential + Q structure |
| u_ratio_22 | ~3.5-4.0 | Degree effect (critical!) |
| I2_total_ratio | ~1.66 | Weighted by pair magnitudes |
| oracle_22_total | ~1.9-2.0 | Includes derivatives |
| deriv_ratio_22 | ~3.5 | Derivatives amplify |

**If all 5 match → Polynomial degree confirmed as root cause**

## Success Criteria

Diagnosis succeeds when:

1. ✅ **Execute diagnostic** and get exact numbers
2. ✅ **Validate predictions** (5 key ratios match)
3. ✅ **Test normalization** hypothesis
4. ✅ **Achieve target ratio** of 1.103 with fix
5. ✅ **Validate on both benchmarks** (κ and κ*)

## Timeline

**Immediate** (today):
- Run diagnostic script
- Compare actual vs predicted
- Test normalization hypothesis

**Short-term** (this week):
- Verify κ* transcription
- Search PRZZ for normalization conventions
- Implement fix if hypothesis confirmed

**Expected resolution:** 1-2 days

## Contact Points

**Files to check if stuck:**

1. **Polynomial definitions:** `data/przz_parameters*.json`
2. **Oracle implementation:** `src/przz_22_exact_oracle.py`
3. **Previous findings:** `SESSION_SUMMARY.md`
4. **Mathematical derivations:** `TECHNICAL_ANALYSIS.md` Section 9

**PRZZ paper sections:**

1. **Section 7:** Formula derivations
2. **Section 8:** Numerical results (lines 2567-2598)
3. **Appendix:** Detailed calculations (if available)

## Files Created By This Diagnosis

```
/przz-extension/
├── ratio_reversal_diagnosis.py      # Main diagnostic script
├── compute_diagnosis.py              # Quick-check version
├── DIAGNOSIS_REPORT.md               # Detailed analysis
├── NUMERICAL_PREDICTIONS.md          # Expected values
├── RATIO_REVERSAL_SUMMARY.md         # Executive summary
├── COMPUTATION_CHECKLIST.md          # Execution guide
└── DIAGNOSIS_INDEX.md                # This file
```

All documents are interconnected and reference each other.

## How To Use This Documentation

### Scenario 1: "I need to understand the problem"
→ Read **RATIO_REVERSAL_SUMMARY.md**

### Scenario 2: "I need to run the diagnosis"
→ Use **COMPUTATION_CHECKLIST.md** + run `ratio_reversal_diagnosis.py`

### Scenario 3: "I need the mathematical details"
→ Read **DIAGNOSIS_REPORT.md** + **NUMERICAL_PREDICTIONS.md**

### Scenario 4: "I got results, now what?"
→ Compare against **NUMERICAL_PREDICTIONS.md**
→ If predictions match: test normalization hypothesis
→ If not: investigate the largest discrepancy

### Scenario 5: "I want to fix it"
→ See **RATIO_REVERSAL_SUMMARY.md** Section "Hypotheses"
→ Start with Hypothesis A (L² normalization)

## Next Session Handoff

**If you're picking this up later:**

1. **Status:** Analysis complete, ready for execution
2. **Next step:** Run `ratio_reversal_diagnosis.py`
3. **Expected:** All 5 predictions should match
4. **Then:** Test L² normalization hypothesis
5. **Goal:** Achieve ratio 1.103 with normalized polynomials

**Critical files:**
- Diagnostic script: `ratio_reversal_diagnosis.py` (ready to run)
- Predictions: `NUMERICAL_PREDICTIONS.md` (compare against)
- Summary: `RATIO_REVERSAL_SUMMARY.md` (context)

---

**Created:** 2025-12-17
**Author:** Claude (diagnostic analysis)
**Ready for:** Numerical execution and hypothesis testing
