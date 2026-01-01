# CHANGELOG: Terminology Fix and Paper Revision

**Date:** 2026-01-01
**Status:** COMPLETE
**Purpose:** Fix double-count ambiguity and establish clear M_0/G/M notation

---

## Summary

This revision introduces clear notation separating:
- **M_0(R) = exp(R) + (2K-1)** -- the EXACT structural base
- **G = g_total** -- the DERIVED correction factor (~1.014)
- **M(R) = G * M_0(R)** -- the full mirror multiplier

The previous formulation showed `c = S_12(+R) + m * g_total * S_12(-R)` which implied double-counting since `m` already incorporated `g_total` in the code.

---

## Files Modified

### 1. src/kappa_engine.py

**Lines changed:** ~50 (header, docstrings, FORMULA_DOC)

| Location | Change |
|----------|--------|
| Lines 7-33 | Header FORMULAS block rewritten with M_0/G/M notation |
| Line 87 | IntegralComponents docstring: clarified M = G * M_0 |
| Lines 117-135 | CorrectionFactors docstring: added notation mapping |
| Lines 257-270 | compute_base: renamed references to M_0 |
| Lines 298-310 | compute_mirror_multiplier: added M/G/M_0 explanation |
| Lines 342-350 | compute_c_from_integrals: fixed sum vs product (S_12 not I_1*I_2) |
| Lines 768-850 | FORMULA_DOC: complete rewrite with notation table |

**Before (line 14):**
```python
c = S_12(+R) + m * g_total * S_12(-R) + S_34(+R)  # WRONG - double-count!
```

**After:**
```python
c = S_12(+R) + M * S_12(-R) + S_34(+R)
# where M = G * M_0, G = g_total, M_0 = exp(R) + (2K-1)
```

### 2. paper_output/tex/main_results.tex

**Lines changed:** ~80

| Location | Change |
|----------|--------|
| Line 89 | Abstract: changed m to M_0 |
| Lines 300-313 | Added M_0/G breakdown to mirror multiplier description |
| Line 306 | Assembly formula: changed m to M(R) |
| Line 976 | Assembly formula: changed m to M(R) |
| Lines 985-1010 | Split mirror theorem into three parts (Theorem + 2 Definitions) |
| Line 1012 | Proof title: changed to "Proof of structural base M_0" |
| Line 1047 | Align block: changed m to M_0 |
| Lines 1267-1269 | Derivation status table: added G and M rows |
| Line 1876 | Summary: updated to M_0 |
| Lines 1893-1895 | Summary claim: updated to M_0 = exp(R) + (2K-1) |
| Lines 1962-1976 | Derivation chain: added Steps I (G) and J (M) |

**Before (Theorem):**
```latex
\begin{theorem}[Mirror Multiplier --- Exact]
m = e^R + (2K-1)
\end{theorem}
```

**After:**
```latex
\begin{theorem}[Structural Mirror Base --- Exact]
M_0(R) = e^R + (2K-1)
\end{theorem}

\begin{definition}[Correction Factor --- Derived]
G = f_{I_1} \cdot g_{I_1} + (1-f_{I_1}) \cdot g_{I_2}
\end{definition}

\begin{definition}[Full Mirror Multiplier]
M(R) = G \cdot M_0(R)
\end{definition}
```

### 3. docs/DERIVATION_STATUS.md

**Lines changed:** ~40

| Location | Change |
|----------|--------|
| Lines 14-27 | Header table: added M_0/G/M rows with Status column |
| Line 33 | Section title: changed to "Structural Mirror Base: M_0" |
| Lines 43-47 | Algebraic proof: changed m to M_0 |
| Lines 90-112 | Complete Formula: fixed double-count, added code mapping |
| Lines 116-130 | Component Status Table: added G and M rows |
| Lines 137-176 | "Honest Picture" diagram: updated to M_0/G/M notation |
| Lines 186-200 | 1.8x Gap section: updated to M_0, added G and M items |
| Lines 252-260 | Numerical Verification: changed m to M_0 |
| Lines 280-294 | Paper-Ready Claims: updated to M_0, added M = G * M_0 |
| Lines 319-322 | Historical Note: updated phases to M_0 |

### 4. scripts/compute_ba_ratio_noncircular.py

**Lines changed:** 3

| Location | Change |
|----------|--------|
| Lines 245-247 | G-CORRECTION print statements: updated to M = G * M_0 |

---

## Backup Created

```
paper_output/tex/main_results_v1_2025-12-31.tex
```

This preserves the pre-revision paper for version tracking.

---

## Validation Results

| Check | Result |
|-------|--------|
| Python import | OK |
| kappa benchmark gap | 0.0005% (< 0.01%) |
| kappa_star benchmark gap | -0.0004% (< 0.01%) |
| M_0 in kappa_engine.py | 10+ occurrences |
| M_0 in main_results.tex | 10+ occurrences |
| M_0 in DERIVATION_STATUS.md | 28 occurrences |
| Double-count pattern in assembly | 0 occurrences |

---

## Key Formula Changes

### Assembly Formula

| Before | After |
|--------|-------|
| `c = S_12(+R) + m * g_total * S_12(-R) + S_34(+R)` | `c = S_12(+R) + M * S_12(-R) + S_34(+R)` |

### Notation Mapping

| Paper Symbol | Code Variable | Description | Status |
|--------------|---------------|-------------|--------|
| M_0 | `base` | Structural base exp(R) + (2K-1) | EXACT |
| G | `g_total` | Correction factor ~1.014 | DERIVED |
| M | `m` | Full multiplier G * M_0 | DERIVED |

### What Changed (Conceptual)

**Old claim:** "m = exp(R) + (2K-1) is EXACT"

**New claim:** "M_0 = exp(R) + (2K-1) is EXACT; the full multiplier M = G * M_0 includes a derived correction factor G ~ 1.014 with 0.09% residual"

This is more honest and precise. The structural base M_0 is still exact, but we now correctly distinguish it from the full multiplier M used in the assembly.

---

## Remaining Items

1. **Pre-existing test issue:** `tests/test_q_operator_collapse_gate.py` has an import error for `transform_term_q_factors` - this predates these changes
2. **LaTeX compilation:** pdflatex not available on system - manual compilation needed on Overleaf
3. **Unicode chars:** Some unicode (subscripts/superscripts) remain in markdown docs - these are acceptable for documentation readability

---

## Success Criteria Met

- [x] No double-count ambiguity in any formula
- [x] Clear M_0/G/M separation throughout
- [x] Code header matches implementation
- [x] Docstring sum/product typo fixed (S_12 = I_1 + I_2, not I_1*I_2)
- [x] Only M_0 labeled "EXACT"; G and M labeled "DERIVED"
- [x] Benchmarks validate to < 0.01%
- [x] Paper version archived before changes
