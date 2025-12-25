# DECISIONS.md — Architectural Decision Log

**Purpose:** Record key architectural decisions with dates, reasons, and test pointers.
This prevents repeated re-litigation of settled questions.

---

## Decision 1: I3/I4 Mirror is FORBIDDEN by Spec

**Date:** 2025-12-22
**Status:** SPEC-LOCKED
**Authority:** TRUTH_SPEC.md Section 10 (lines 370-388)

### Context

The PRZZ formula has different mirror structures for different integrals:
- I₁ and I₂ combine with their mirror terms: `I(α,β) + T^{-α-β}·I(-β,-α)`
- I₃ and I₄ do NOT have mirror terms

Early implementation attempts added mirror structure to I3/I4, causing a 350% overshoot.

### Decision

**I3/I4 mirror is architecturally forbidden.** The code raises `I34MirrorForbiddenError` if anyone attempts to apply mirror to I3/I4 integrals.

### Implementation

- **Guard function:** `_assert_i34_no_mirror()` in `src/evaluate.py`
- **Exception:** `I34MirrorForbiddenError` (subclass of ValueError)
- **Protected functions:**
  - `compute_S34_tex_combined_11()`
  - `compute_S34_base_11()`

### Rationale

1. TRUTH_SPEC.md Section 10 explicitly states I₃/I₄ have "NO MIRROR"
2. Empirical testing shows +350% overshoot when mirror is incorrectly applied
3. Making this a hard raise prevents accidental reintroduction

### Tests

- `tests/test_i34_structure_gate.py` — 14 tests covering:
  - `I34MirrorForbiddenError` is ValueError subclass
  - Guard function raises on `apply_mirror=True`
  - Default behavior is no mirror
  - Error messages reference TRUTH_SPEC.md

### What Would Falsify This

If a new interpretation of PRZZ shows I3/I4 DO require mirror, the guard would need to be removed. However, the empirical evidence (350% overshoot) strongly suggests plus-only is correct.

---

## Decision 2: m1 is Empirical — K>3 Requires Explicit Opt-In

**Date:** 2025-12-22
**Status:** HARD-LOCKED
**Authority:** Empirical calibration against PRZZ benchmarks

### Context

The mirror weight `m1` appears in the mirror term assembly formula:
```
c = I₁I₂(+R) + m1 × I₁I₂(-R) + I₃I₄(+R)
```

Multiple attempts to derive m1 from first principles failed:
- Naive formula `exp(2R)` is too large by 1.56× (κ) / 1.16× (κ*)
- Finite-L approach diverges: m1_eff(L) ≈ -9.15 × L
- Unified-t approach gives m1_eff ≈ 0.6, not ~8.7

### Decision

**m1 = exp(R) + (2K-1) is treated as empirical calibration.**

- For K=3: m1 = exp(R) + 5 (validated at both κ and κ* benchmarks)
- For K>3: Formula is EXTRAPOLATED and UNVALIDATED
- K>3 usage requires explicit opt-in via `allow_extrapolation=True`

### Implementation

- **Module:** `src/m1_policy.py`
- **Enum:** `M1Mode` with modes: `K3_EMPIRICAL`, `K_DEP_EMPIRICAL`, `PAPER_NAIVE`, `OVERRIDE`
- **Dataclass:** `M1Policy` with `mode`, `allow_extrapolation`, `override_value`
- **Function:** `m1_formula(K, R, policy)` with hard raises for K>3 without opt-in

### API Examples

```python
# Safe K=3 usage (default)
policy = M1Policy(mode=M1Mode.K3_EMPIRICAL)
m1 = m1_formula(K=3, R=1.3036, policy=policy)  # Works

# K=4 without opt-in (RAISES)
policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL)
m1 = m1_formula(K=4, R=1.3036, policy=policy)  # RAISES M1ExtrapolationError!

# K=4 with explicit opt-in (Works with warning)
policy = M1Policy(mode=M1Mode.K_DEP_EMPIRICAL, allow_extrapolation=True)
m1 = m1_formula(K=4, R=1.3036, policy=policy)  # Works, emits warning
```

### Rationale

1. m1 is NOT derived from first principles — it's calibration
2. The extrapolation to K>3 via (2K-1) pattern is CONJECTURED
3. Hard raises prevent accidental extrapolation that produces wrong results
4. Warnings for opted-in extrapolation ensure audit trail

### Tests

- `tests/test_m1_policy_gate.py` — 27 tests covering:
  - All M1Mode variants
  - K3_EMPIRICAL raises on K≠3
  - K_DEP_EMPIRICAL raises on K>3 without opt-in
  - K_DEP_EMPIRICAL works with opt-in (emits warning)
  - Benchmark validation (κ and κ*)
  - OVERRIDE mode

- `tests/test_m1_sensitivity.py` — 11 tests covering:
  - m1 monotonicity in R and K
  - m1 linearity in K (delta=2 per K)
  - Sensitivity range
  - K>3 extrapolation risk characterization

### What Would Falsify This

If a K=4 reference target appears (from PRZZ or other source), the calibration harness in `src/m1_calibration.py` can validate whether `exp(R) + 7` matches. If it doesn't, a new formula would be needed.

---

## Decision 3: Spec Authority Hierarchy

**Date:** 2025-12-22
**Status:** ESTABLISHED

### Decision

When sources disagree, resolve in this priority order:

1. **TRUTH_SPEC.md** (human-curated from TeX)
2. **RMS_PRZZ.tex excerpt + line ranges**
3. Code implementation
4. Tests
5. Empirical benchmark fit

### Rationale

- TeX is the mathematical ground truth
- TRUTH_SPEC.md is curated to resolve ambiguities in TeX
- Code should implement spec, not define it
- Tests verify code matches spec
- Empirical fits are last resort when spec is unclear

### Enforcement

**Rule:** If tests disagree with TRUTH_SPEC.md, tests must fail (and be fixed).

---

## Decision 4: Negative Control Tests for Forbidden Configurations

**Date:** 2025-12-22
**Status:** ESTABLISHED

### Context

Some configurations are architecturally forbidden (like I3/I4 mirror). Tests for these should verify that incorrect usage FAILS spectacularly.

### Decision

**Negative control tests verify that forbidden configurations raise or produce obvious failures.**

Examples:
- `test_raises_when_mirror_true()` — Verifies I3/I4 mirror raises
- `test_k4_raises_without_optin()` — Verifies K>3 extrapolation raises

### Rationale

1. Negative controls catch accidental reintroduction of bugs
2. They document "what NOT to do" in test form
3. They make the test suite self-documenting

---

## Decision 5: Calibration Harness for Future Validation

**Date:** 2025-12-22
**Status:** IMPLEMENTED

### Decision

A calibration harness exists to solve for implied m1 from reference c_target values:

```python
from src.m1_calibration import solve_m1_from_channels, validate_m1_formula_at_k3

# Validate K=3 formula
results = validate_m1_formula_at_k3(channels_kappa, channels_kappa_star)
# Returns CalibrationResult with m1_solved, m1_empirical, ratio
```

### Rationale

1. If a K=4 reference target appears, we can validate in minutes
2. For K=3, this confirms exp(R)+5 matches both benchmarks
3. Provides auditability for m1 formula

### Implementation

- **Module:** `src/m1_calibration.py`
- **Functions:**
  - `solve_m1_from_channels()` — Solve for m1 given c_target and channels
  - `validate_m1_formula_at_k3()` — Validate K=3 at both benchmarks
  - `estimate_m1_for_k4()` — Ready for K=4 validation when reference exists

---

## Decision 6: Q-Shift Investigation (Phase 6)

**Date:** 2025-12-22
**Status:** INVESTIGATED — m1 remains empirical
**Authority:** GPT guidance + numerical experiments

### Context

Phase 5 proved that the u-regularized and post-identity paths compute the same TeX I₁ object
to machine precision. This led to Phase 6: investigating whether the empirical m₁ could be
replaced by a derived formula via the operator shift identity.

The operator shift identity states:
```
Q(D_α)(T^{-s}F) = T^{-s} × Q(1 + D_α)F
```

The hypothesis was that m₁ could be derived from the Q polynomial shift Q → Q(1+·).

### Investigation

**New modules created:**
- `src/mirror_exact.py` — Implements I₁ computation with shifted Q
- `docs/TEX_MIRROR_OPERATOR_SHIFT.md` — Mathematical derivation

**Key function:** `compute_I1_with_shifted_Q(shift=1.0)` computes I₁ using Q(1+·) instead of Q(·).

**Numerical results (28 tests passing):**

| Pair | I₁_standard | I₁_shifted (Q→Q(1+·)) | Ratio |
|------|-------------|------------------------|-------|
| (1,1) | 0.413 | 46.5 | 112× |
| (1,2) | -0.568 | -72.0 | 127× |
| (2,2) | 0.161 | 16.6 | 103× |
| (3,3) | 0.000088 | 0.0075 | 85× |

**Comparison to empirical m₁:**
- κ benchmark (R=1.3036): Q-shift ratio = 112, m₁_empirical = 8.68
- κ* benchmark (R=1.1167): Q-shift ratio = 5.1, m₁_empirical = 8.05

### Conclusion

**The Q-shift ratio does NOT equal m₁.**

The operator shift Q → Q(1+·) dramatically changes I₁ values (by factors of 85-127× for κ),
which is much larger than the empirical m₁ ≈ 8.68.

**Why this happened:**
1. The combined identity already includes BOTH direct and mirror contributions
2. Applying shifted Q to the combined kernel is not the same as separating direct/mirror
3. The mirror contribution cannot be cleanly separated after the combined identity transformation

**What this means:**
- m₁ remains an empirical calibration (Decision 2 unchanged)
- The Q-shift identity is VALID but applies at a different level than I₁ computation
- Further derivation would require working BEFORE the combined identity, which reintroduces the 1/s pole

### Decision

**m₁ derivation via Q-shift is NOT viable with current infrastructure.**

The investigation proved the Q-shift mathematics is correct but doesn't directly yield m₁.
Decision 2 (m₁ is empirical) remains in force.

### Tests

- `tests/test_mirror_decomposition_gate.py` — 28 tests covering:
  - Q-shift mathematics (binomial lift)
  - Shifted I₁ computation
  - Comparison at both benchmarks
  - Series extraction with shifted Q

### What Would Change This

A derivation that works BEFORE the combined identity transformation, properly handling the
1/s pole while separating direct and mirror contributions, might yield a derived m₁.
This would require new mathematical techniques not currently implemented.

---

## Decision 7: RAW_LOGDERIV is Semantically Correct for J12

**Date:** 2025-12-24
**Status:** SEMANTIC-LOCKED
**Authority:** Direct series expansion of J12 bracket₂ structure

### Context

Phase 14G tested two Laurent factor modes for J12:
- `RAW_LOGDERIV`: Uses `(1/R + γ)²` - R-sensitive
- `POLE_CANCELLED`: Uses `+1` constant - R-invariant

Initial hypothesis: POLE_CANCELLED should be correct based on G-product pole cancellation.

Counterintuitive result: POLE_CANCELLED actually INCREASED delta by 30-200%.

### Investigation (Phase 14H)

Phase 14H asked: Which mode is **semantically correct** (not just "gives smaller delta")?

**Key insight from j12_c00_reference.py:**

The J12 bracket₂ zeta-factor is:
```
(ζ'/ζ)(1+α+s) × (ζ'/ζ)(1+β+u)
```

At s=u=0 with α=β=-R:
```
(ζ'/ζ)(1-R)² = (-1/(-R) + γ)² = (1/R + γ)²
```

This matches RAW_LOGDERIV **exactly**.

**Why POLE_CANCELLED is wrong:**

POLE_CANCELLED was based on the G-product G(α)G(β) where:
```
G(ε) = (1/ζ)(ζ'/ζ)(1+ε) = -1 + O(ε)
```

But J12 does **NOT** include the 1/ζ factors. It uses pure log-derivative:
```
(ζ'/ζ)(1+α) = -1/α + γ + O(α)
```

### Decision

**RAW_LOGDERIV is the semantically correct mode for J12.**

We choose by mathematical correctness, NOT by which gives smaller delta.

### Implementation

- **Module:** `src/ratios/j1_euler_maclaurin.py`
- **Constant:** `DEFAULT_LAURENT_MODE = LaurentMode.RAW_LOGDERIV`
- **All functions** use this constant as default

### Tests

- `tests/test_j12_c00_semantics.py` — 10 tests proving:
  - RAW_LOGDERIV matches literal log-deriv product exactly (1e-10 tolerance)
  - POLE_CANCELLED does NOT match literal (>0.5 difference)
  - G-product differs from log-deriv product (>5.0 difference)
  - Exactly one mode matches the literal structure

- `tests/test_delta_track_harness_invariants.py` — 36 tests including:
  - POLE_CANCELLED has larger delta than RAW_LOGDERIV
  - POLE_CANCELLED increase is significant (>10%)

### What Would Falsify This

If the J12 bracket₂ structure were shown to include 1/ζ factors (making it a G-product),
POLE_CANCELLED would become correct. However, the TeX reference (lines 2391-2409) clearly
shows log-derivative without 1/ζ factors.

---

---

## Decision 8: Semantic vs Numeric Computation Modes

**Date:** 2025-12-24
**Status:** POLICY-ESTABLISHED
**Authority:** Phase 17C analysis

### Context

Phase 15 found that the Laurent approximation `(1/R + γ)²` for `(ζ'/ζ)(1-R)²` has
significant error at the PRZZ benchmark R values (22% for κ, 17% for κ*).

Using the actual numerical value `ACTUAL_LOGDERIV` instead of the Laurent
approximation `RAW_LOGDERIV` dramatically improves accuracy.

However, Decision 7 established that `RAW_LOGDERIV` is the semantically correct
object (matches the TeX Laurent expansion structure).

This creates a tension: semantic correctness vs numerical accuracy.

### Decision

**Establish two distinct computation modes:**

1. **Semantic Mode** (`RAW_LOGDERIV`)
   - Uses Laurent expansion `(1/R + γ)²`
   - Matches what the TeX formula literally says
   - Use for: theoretical validation, asymptotic analysis

2. **Numeric Mode** (`ACTUAL_LOGDERIV`)
   - Uses actual numerical `(ζ'/ζ)(1-R)²` via mpmath
   - Best accuracy at finite R
   - Use for: production κ computation, benchmark matching

### Implementation

The `LaurentMode` enum provides both options. Functions should accept a mode
parameter and document which mode they default to.

**Production pipeline:** Should use `ACTUAL_LOGDERIV` for best accuracy.
**Diagnostic pipeline:** Should support both modes for comparison.

### Rationale

1. The TeX formula is the ground truth for what we're computing
2. But Laurent series diverges for |ε| > 1, and PRZZ uses ε = -R ≈ -1.3
3. Using the actual value is mathematically equivalent (same object, better evaluation)
4. Splitting modes prevents confusion about which interpretation is active

### Phase 17B Finding: Asymmetry Analysis

The per-piece delta decomposition (Phase 17B) showed that J13/J14 have the
highest κ/κ* asymmetry (ratio 1.89) when switching from RAW to ACTUAL mode.

This is because the Laurent approximation error is R-dependent:
- κ (R=1.3036): Laurent error 29%
- κ* (R=1.1167): Laurent error 21%

The larger error for κ means κ gets a larger correction, causing asymmetry.

### What Would Change This

If PRZZ published which evaluation method they used for their numerical results,
we would standardize on that method.

---

## Index of Decisions

| # | Topic | Status | Date | Tests |
|---|-------|--------|------|-------|
| 1 | I3/I4 Mirror FORBIDDEN | SPEC-LOCKED | 2025-12-22 | test_i34_structure_gate.py |
| 2 | m1 is Empirical | HARD-LOCKED | 2025-12-22 | test_m1_policy_gate.py, test_m1_sensitivity.py |
| 3 | Spec Authority Hierarchy | ESTABLISHED | 2025-12-22 | — |
| 4 | Negative Control Tests | ESTABLISHED | 2025-12-22 | Various |
| 5 | Calibration Harness | IMPLEMENTED | 2025-12-22 | — |
| 6 | Q-Shift Investigation | INVESTIGATED | 2025-12-22 | test_mirror_decomposition_gate.py |
| 7 | RAW_LOGDERIV Semantic | SEMANTIC-LOCKED | 2025-12-24 | test_j12_c00_semantics.py, test_delta_track_harness_invariants.py |
| 8 | Semantic vs Numeric Modes | POLICY-ESTABLISHED | 2025-12-24 | — |
