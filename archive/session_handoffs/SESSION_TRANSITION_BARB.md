# Session Transition: I₅ Bar B Derivation

**Date:** 2025-12-15
**Status:** Bar A complete, Bar B derivation REQUIRED (non-negotiable)

---

## Executive Summary

I₅ currently matches PRZZ numerically with calibrated parameters (g≈0.50, max_k=1). However, **Bar B is non-negotiable**: g must be DERIVED from the PRZZ paper with ZERO fitted constants.

**Critical resource added:** `RMS_PRZZ.tex` is in the parent folder at:
`/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/RMS_PRZZ.tex`

This is the PRZZ paper source — the key to deriving g.

---

## Current State

### What's Done (Bar A)

1. **I₅ implementation works numerically:**
   - `src/i5_diagonal.py` implements diagonal convolution
   - g≈0.50 and max_k=1 give Δc = +0.001% match to PRZZ
   - S_cache hoisted for efficiency
   - API accepts both dict and tuple polynomials

2. **Tests passing:**
   - 436 total tests pass
   - 14 falsifiability tests in `tests/test_i5_falsifiability.py`
   - PRZZ golden tests pass

3. **Baseline frozen:**
   - Git tag: `phase0_barA_calibrated_i5`
   - Commit: `f292a27`

### What's NOT Done (Bar B — REQUIRED)

1. **g is calibrated, not derived** — UNACCEPTABLE
2. **max_k=1 was determined by diagnostics** — must be derived from PRZZ
3. **No citation to PRZZ equations** — need equation numbers

---

## Bar B Definition of Done

I₅ is "properly derived" ONLY if:

1. We cite **explicit equations in PRZZ** (with equation/page numbers) that determine:
   - The arithmetic factor cross-term structure
   - The Mellin→formal variable mapping (z_i, w_j → x_i, y_j)
   - The exact multiplicative factor g

2. Implementation uses **only derived quantities** (no fitted g)

3. PRZZ reproduction still matches within numerical error

**If any constant is obtained by optimization, it is NOT Bar B.**

---

## The 5 Tasks for Bar B

### Task 0: Freeze Baseline ✓ COMPLETE
- Tag created: `phase0_barA_calibrated_i5`

### Task 1: Extract Exact PRZZ Arithmetic-Factor Statement
**Status:** IN PROGRESS — RMS_PRZZ.tex now available

**Action:** Read RMS_PRZZ.tex and extract (with equation numbers):
1. Exact definition of S(z) = Σ_p (log p / (p^{1+z} − 1))²
2. Exact statement of: ∂²/∂z_i∂w_j log A|₀ = -S(something)
3. What "something" is — is it α+β? 2Rt? Both?
4. Are there other nonzero second derivatives?

**Output:** Update TECHNICAL_ANALYSIS.md with paper-citable derivation.

### Task 2: Derive Mellin→Formal Variable Mapping (THIS GIVES g)
**Status:** PENDING — critical blocker

**Action:** In RMS_PRZZ.tex, find:
1. Where z_i, w_j are defined (Mellin/residue variables)
2. How they map to our formal variables x_i, y_j
3. The normalization (log T, θ factors)

**Deliverable:** Explicit mapping:
```
z_i = a(θ,t,u)·x_i + b(θ,t,u)·y_i
w_j = c(θ,t,u)·x_j + d(θ,t,u)·y_j
```

Then compute: Σ z_i w_j = g(θ,t,u) · X · Y

**This is the derived g.** It may be constant, θ-dependent, or (θ,t,u)-dependent.

### Task 3: Determine Linear vs Exponential from PRZZ
**Status:** PENDING

**Action:** Find in RMS_PRZZ.tex where arithmetic factor enters:
- As bracket substitution C → C - S (linear, k=1 only)
- As multiplicative exp(-S·Σz_iw_j) (all k)

**Key question:** At what stage does PRZZ move from Euler product to bracket expression?

### Task 4: Implement I₅ in Derived Mode
**Status:** PENDING

**Action:** Create two explicit modes:
```python
@dataclass(frozen=True)
class I5Model:
    mode: Literal["derived", "calibrated"]
    g_formula: Optional[Callable[[float,float,float], float]]  # g(θ,t,u)
    k_policy: Literal["linear", "exp"]
```

Derived mode has ZERO knobs. Default must be derived once Bar B is complete.

### Task 5: Add Bar B Validation Tests
**Status:** PENDING

Tests needed:
- `test_g_formula_matches_przz_mapping()` — algebraic verification
- `test_i5_mode_derived_has_no_free_parameters()`
- `test_przz_phase0_golden_derived_i5()` — must pass
- `test_wrong_g_breaks_golden()` — negative control, must FAIL

---

## Key Files

### Source Files
| File | Status | Notes |
|------|--------|-------|
| `../RMS_PRZZ.tex` | NEW | The PRZZ paper source — READ THIS |
| `src/i5_diagonal.py` | Bar A done | Has calibrated g, needs derived mode |
| `src/evaluate.py` | Working | Needs i5_model parameter |
| `TECHNICAL_ANALYSIS.md` | Needs update | Add PRZZ citations with eq numbers |

### Plan Files
| File | Status |
|------|--------|
| `~/.claude/plans/i5-barB-derivation.md` | Active plan |
| `~/.claude/plans/spicy-enchanting-horizon.md` | Bar A only |
| `~/.claude/plans/i5-g-derivation.md` | Superseded |

### Test Files
| File | Tests | Status |
|------|-------|--------|
| `tests/test_i5_falsifiability.py` | 14 | Passing (Bar A) |
| `tests/test_i5_validation.py` | 10 | Passing |
| `tests/test_i5_derived.py` | 0 | TO BE CREATED (Bar B) |

---

## The Core Mathematical Question

The diagonal convolution formula is:
```
ΔF_{ℓ₁,ℓ₂} = Σ_{k=1}^{max_k} [(-g·S)^k / k!] × F^{ratio}_{ℓ₁-k,ℓ₂-k}
```

Currently: g ≈ 0.50 (CALIBRATED)

**Bar B requires deriving g from PRZZ:**

If the Mellin variables map as z_i = a·x_i (no mixing), then:
```
Σ z_i w_j = a·d · Σx_i · Σy_j = a·d · ℓ₁·ℓ₂ · X·Y / (ℓ₁·ℓ₂) = a·d · X·Y
g = a·d (derivable)
```

If mixing exists (z_i = a·x_i + b·y_i), diagonal convolution may be approximate.

**The answer is in RMS_PRZZ.tex.** Find the variable definitions.

---

## Key Insight from User Guidance

> If PRZZ implies the arithmetic factor contributes via bracket replacement (C → C-S), then the correct operator is **structurally linear**, and any "exp convolution" is the wrong model class.

> Conversely, if PRZZ carries A=exp(...) into coefficient extraction, then linear-only is wrong.

**The fastest path to Bar B:**
1. Locate in RMS_PRZZ.tex where they move from Euler product to bracket expression
2. Read off whether S enters as shift (linear) or exponential (all k)
3. Derive g from the variable mapping
4. Implement with no knobs

---

## Commands to Resume

```bash
# Check current state
cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
git status
git log --oneline -5
git tag -l

# Run tests to verify nothing broke
python3 -m pytest tests/ -v --tb=short | tail -20

# The PRZZ paper source is at:
# ../RMS_PRZZ.tex (parent directory)

# The active plan is at:
# ~/.claude/plans/i5-barB-derivation.md
```

---

## Next Session Instructions

1. **Read ../RMS_PRZZ.tex** — specifically:
   - Section defining the arithmetic factor A
   - Section defining formal variables z_i, w_j
   - Equations showing how A-derivatives enter brackets

2. **Extract with equation numbers:**
   - S(z) definition
   - Cross-term ∂²/∂z_i∂w_j log A = -S(?)
   - Variable mapping z_i ↔ x_i

3. **Derive g** — no fitting allowed

4. **Implement derived mode** — zero knobs

5. **Add negative control tests** — verify wrong g breaks match

---

## Non-Negotiable Constraint

**Bar B is required.** The current g≈0.50 is a calibration knob. Even if it "works," it is NOT acceptable.

The PRZZ paper is now available. Derive g from first principles or document exactly why it cannot be derived (with specific PRZZ citations showing the gap).

No Phase 1 optimization until Bar B is complete.
