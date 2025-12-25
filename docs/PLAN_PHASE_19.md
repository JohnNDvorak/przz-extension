# Phase 19 Implementation Plan: Semantic-Numeric Separation

**Date:** 2025-12-24
**Status:** PLANNING
**Predecessor:** Phase 18 (Diagnostic-First m1 Investigation)

---

## Executive Summary

Phase 19 implements GPT's guidance to resolve the remaining ~1.35% gap by:
1. **Separating semantic (paper) vs numeric (production) evaluation modes**
2. **Preventing I₅/A-derivative contamination in main-term matching**
3. **Systematizing the delta-track decomposition**
4. **Attempting principled m₁ derivation without fitting**
5. **Preparing for K≥4 safety testing**
6. **Refactoring evaluate.py (6700 lines) safely**

### Critical Insight from GPT

> The remaining work splits cleanly into (A) "what is the mathematically correct object?" vs (B) "what is the numerically stable way to compute it?" — these have been getting mixed.

### Current State (from Phase 18)

| Metric | Value | Status |
|--------|-------|--------|
| B/A (κ, ACTUAL mode) | 4.95 | -0.93% from 5 |
| B/A (κ*, ACTUAL mode) | 4.87 | -2.66% from 5 |
| J1x implied m₁ | 15-22% of empirical | Structural mismatch |
| Empirical m₁ | exp(R) + 5 | Working formula |
| c gap | -1.35% | Target: 0% |

---

## Phase 19.0 — Lock Semantic Target (Prevent Calibration Creep)

### 19.0.1 Add/verify explicit "main-term only" guardrails

**Goal:** Ensure we never "fix κ" by accidentally leaning on I₅ / A-derivatives.

**Files to Create/Modify:**
- `src/evaluation_modes.py` (NEW) — Central mode switch
- `tests/test_no_I5_in_main_mode.py` (NEW)

**Implementation:**

```python
# src/evaluation_modes.py
from enum import Enum, auto
from typing import Optional
import warnings

class EvaluationMode(Enum):
    """Semantic evaluation modes per TRUTH_SPEC.md Section 4."""
    MAIN_TERM_ONLY = auto()      # I₅ forbidden, A-derivatives forbidden
    WITH_ERROR_TERMS = auto()    # I₅ allowed (diagnostic only)

_current_mode: EvaluationMode = EvaluationMode.MAIN_TERM_ONLY

def set_evaluation_mode(mode: EvaluationMode) -> None:
    global _current_mode
    _current_mode = mode
    if mode == EvaluationMode.WITH_ERROR_TERMS:
        warnings.warn(
            "WITH_ERROR_TERMS mode: I₅ contributions are error-order (≪ T/L). "
            "Do NOT use to calibrate main-term c.",
            UserWarning
        )

def get_evaluation_mode() -> EvaluationMode:
    return _current_mode

def assert_main_term_only(operation: str) -> None:
    """Raise if attempting error-term operation in MAIN_TERM_ONLY mode."""
    if _current_mode == EvaluationMode.MAIN_TERM_ONLY:
        raise ValueError(
            f"Operation '{operation}' forbidden in MAIN_TERM_ONLY mode. "
            "I₅ and A-derivatives are error-order per TRUTH_SPEC.md Lines 1621-1628."
        )
```

**Acceptance Criteria:**
- [ ] `pytest -k test_no_I5_in_main_mode` passes
- [ ] Attempting to use I₅ in MAIN_TERM_ONLY mode raises ValueError
- [ ] WITH_ERROR_TERMS mode prints warning but allows computation

---

## Phase 19.1 — Fix +5 Derivation Story (No I₅ Dependency)

### 19.1.1 Split +5 harness into main-only vs with-error-terms

**Goal:** Make it impossible to "pass +5" by using error terms.

**Files to Modify:**
- `src/ratios/microcase_plus5_signature_k3.py` — Add mode parameter
- `src/ratios/j1_k3_decomposition.py` — Expose J1,5 contribution separately
- `tests/test_plus5_gate.py` — Split assertions by mode

**Implementation:**

```python
# In microcase_plus5_signature_k3.py
from src.evaluation_modes import EvaluationMode, get_evaluation_mode

@dataclass
class Plus5Result:
    """Result with explicit mode tracking."""
    B_over_A: float
    B_over_A_main_only: float
    B_over_A_with_error: float
    delta: float
    delta_main_only: float
    j15_contribution: float  # The A^{(1,1)} piece
    mode: EvaluationMode

def compute_plus5_signature(
    benchmark: str,
    mode: EvaluationMode = EvaluationMode.MAIN_TERM_ONLY
) -> Plus5Result:
    """Compute B/A with explicit mode."""
    ...
```

**Acceptance Criteria:**
- [ ] Microcase prints both main-only and with-error values
- [ ] J1,5 (A^{(1,1)}) contribution is explicitly logged
- [ ] Tests assert ONLY on main-only quantity for paper semantics
- [ ] `mode="main"` gives same result as current ACTUAL_LOGDERIV without I₅

---

## Phase 19.2 — Reconcile Semantic vs Numeric ζ′/ζ Handling

### 19.2.1 Create single owner module for ζ evaluation + scaling

**Goal:** Remove duplicated logic and prevent "silent mode drift."

**Files:**
- `src/ratios/zeta_eval.py` (NEW) — Central ζ evaluation API
- `src/ratios/zeta_laurent.py` (EXISTS) — Keep as reference
- `tests/test_zeta_eval_modes.py` (NEW)

**Implementation:**

```python
# src/ratios/zeta_eval.py
from enum import Enum, auto
from mpmath import mp, zeta, euler
from typing import Tuple

class ZetaMode(Enum):
    """Evaluation modes for ζ′/ζ."""
    SEMANTIC_LAURENT_NEAR_1 = auto()   # Paper asymptotic expansion
    NUMERIC_FUNCTIONAL_EQ = auto()     # Analytic continuation (mpmath)

# Euler-Mascheroni constant
GAMMA_E = float(euler)  # ≈ 0.5772156649

def zeta_logderiv_scaled(
    alpha_over_L: float,
    L: float,
    mode: ZetaMode = ZetaMode.SEMANTIC_LAURENT_NEAR_1
) -> float:
    """
    Compute (ζ′/ζ) in the PRZZ asymptotic regime.

    PRZZ regime: α = -R/L with L = log(T), so α ≈ -1/log(T) is small.
    The Laurent expansion is valid: (ζ′/ζ)(1+ε) ≈ -1/ε + γ_E

    Parameters:
        alpha_over_L: The value α/L (dimensionless)
        L: log(T) scaling parameter
        mode: Which evaluation to use

    Returns:
        The scaled log-derivative value
    """
    if mode == ZetaMode.SEMANTIC_LAURENT_NEAR_1:
        # Paper regime: use Laurent at s = 1 + α = 1 - R/L
        # (ζ′/ζ)(1 - R/L) ≈ L/R + γ_E  (for small R/L)
        alpha = alpha_over_L * L
        if abs(alpha) < 1e-10:
            raise ValueError("α too small for Laurent expansion")
        return -1.0 / alpha + GAMMA_E
    else:
        # Numeric mode: evaluate at s = 1 + α
        alpha = alpha_over_L * L
        s = 1.0 + alpha
        return float(mp.diff(lambda x: mp.log(zeta(x)), s))

def zeta_logderiv_at_point(
    s: complex,
    mode: ZetaMode = ZetaMode.NUMERIC_FUNCTIONAL_EQ,
    precision: int = 50
) -> complex:
    """
    Compute (ζ′/ζ)(s) at an arbitrary point.

    For s = 1-R (far from 1), use NUMERIC mode.
    This is what the diagnostic pipelines use.
    """
    with mp.workdps(precision):
        if mode == ZetaMode.NUMERIC_FUNCTIONAL_EQ:
            return complex(mp.diff(lambda x: mp.log(zeta(x)), s))
        else:
            # Laurent only valid near s=1
            eps = s - 1
            if abs(eps) > 0.5:
                raise ValueError(
                    f"Laurent expansion invalid for s={s} (|s-1|={abs(eps)} > 0.5). "
                    "Use NUMERIC_FUNCTIONAL_EQ mode."
                )
            return -1.0 / eps + GAMMA_E
```

**Acceptance Criteria:**
- [ ] **Scaling sanity:** For fixed R and increasing L, semantic scaled result converges
- [ ] **Mode separation:** Semantic mode never evaluates ζ′/ζ at s = 1-R directly
- [ ] **Functional equation correctness:** Numeric mode matches mpmath to ~1e-10
- [ ] `tests/test_zeta_eval_modes.py` passes all three checks

---

## Phase 19.3 — Delta-Track Harness v2.0

### 19.3.1 Build single "delta report" runner

**Goal:** One command produces a stable delta decomposition report.

**Files:**
- `scripts/run_delta_report.py` (NEW) — Single entry point
- `src/ratios/delta_track.py` (MODIFY) — Add per-piece decomposition
- `tests/test_delta_invariants.py` (NEW)

**Output Specification:**

```json
{
  "benchmark": "kappa",
  "mode": "SEMANTIC_LAURENT_NEAR_1",
  "R": 1.3036,
  "A": 0.1234,
  "B": 0.6170,
  "D": 0.0050,
  "delta": 0.0405,
  "B_over_A": 5.0,
  "gap_percent": -0.93,
  "per_piece": {
    "I12_plus_R": 0.0025,
    "I12_minus_R": 0.1234,
    "I34_plus_R": 0.0025,
    "I34_minus_R": 0.0000
  },
  "invariants": {
    "delta_equals_D_over_A": true,
    "triangle_convention": true,
    "L_stability": "stable"
  }
}
```

**Implementation:**

```python
# scripts/run_delta_report.py
#!/usr/bin/env python3
"""Generate delta decomposition report for Phase 19."""

import argparse
import json
from src.ratios.delta_track import DeltaRecord, compute_delta_decomposition
from src.ratios.zeta_eval import ZetaMode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", nargs="+", default=["kappa", "kappa_star"])
    parser.add_argument("--mode", choices=["semantic", "numeric"], default="semantic")
    parser.add_argument("--output", default="delta_report.json")
    args = parser.parse_args()

    zeta_mode = (ZetaMode.SEMANTIC_LAURENT_NEAR_1 if args.mode == "semantic"
                 else ZetaMode.NUMERIC_FUNCTIONAL_EQ)

    results = {}
    for bench in args.bench:
        record = compute_delta_decomposition(bench, zeta_mode)
        results[bench] = record.to_dict()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    for bench, data in results.items():
        print(f"\n{bench}:")
        print(f"  B/A = {data['B_over_A']:.4f} (gap: {data['gap_percent']:+.2f}%)")
        print(f"  δ = {data['delta']:.6f}")

if __name__ == "__main__":
    main()
```

**Invariants to Assert:**
- [ ] δ == D/A within tolerance (1e-10)
- [ ] Triangle convention consistent with TRUTH_SPEC
- [ ] Under semantic mode, δ is stable as L increases

**Acceptance Criteria:**
- [ ] `pytest -k delta_invariants` passes
- [ ] `python scripts/run_delta_report.py --bench kappa kappa_star` produces JSON + readable report

---

## Phase 19.4 — Derive m₁ Without Fitting

### 19.4.1 Extract implied m₁ from production Term-DSL and bucket it

**Goal:** Determine whether the mismatch is in S₁₂(-R) only, or also in cross terms and Case C.

**Files:**
- `src/mirror/implied_m1.py` (NEW — extract from diagnostics)
- `scripts/run_implied_m1_breakdown.py` (NEW)
- `tests/test_implied_m1_stability.py` (NEW)

**Per-Bucket Breakdown:**

| Bucket | Pairs | Case Structure | Expected Behavior |
|--------|-------|----------------|-------------------|
| B×B | (1,1) | No aux integrals | J1x diagnostic applies |
| B×C | (1,2), (1,3) | One aux integral | Mixed behavior |
| C×C | (2,2), (2,3), (3,3) | Two aux integrals | Full production only |

**Implementation:**

```python
# src/mirror/implied_m1.py
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class ImpliedM1Breakdown:
    """Per-bucket implied m1 analysis."""
    c_target: float
    S12_plus: float
    S12_minus: float
    S34_plus: float
    implied_m1_total: float

    # Per-bucket breakdown
    bb_implied_m1: float   # B×B only: (1,1)
    bc_implied_m1: float   # B×C: (1,2), (1,3)
    cc_implied_m1: float   # C×C: (2,2), (2,3), (3,3)

    # Per-pair if available
    per_pair: Dict[Tuple[int,int], float]

def compute_implied_m1_breakdown(
    benchmark: str,
    quadrature_nodes: int = 100
) -> ImpliedM1Breakdown:
    """
    Compute implied m1 = (c_target - S12_plus - S34_plus) / S12_minus
    for each bucket separately.
    """
    ...
```

**Acceptance Criteria:**
- [ ] For each bucket, implied m₁ is numerically stable under tighter quadrature
- [ ] Breakdown shows which bucket drives the 15-22% ratio from Phase 18.1

### 19.4.2 Implement candidate exp-components and reject any requiring fitting

**Goal:** Find deterministic exp-component formula matching implied m₁ across both benchmarks.

**Files:**
- `src/mirror/m1_derived.py` (MODIFY — add candidates)
- `tests/test_m1_candidates.py` (NEW)
- `scripts/scan_m1_candidates.py` (NEW)

**Candidates to Implement:**

| Candidate | Formula | Source |
|-----------|---------|--------|
| `EXP_R` | exp(R) | Baseline |
| `E_EXP_2RT_UNDER_Q2` | E[exp(2Rt)] under Q² weight | PRZZ integral |
| `UNIFORM_AVG_EXP_2RT` | ∫₀¹ exp(2Rt) dt | Simple average |
| `PRZZ_LIMIT` | exp(2R/θ) | Paper asymptotic (known too large) |

**Implementation:**

```python
# Addition to src/mirror/m1_derived.py
from enum import Enum, auto

class M1ExpCandidate(Enum):
    EXP_R = auto()
    E_EXP_2RT_UNDER_Q2 = auto()
    UNIFORM_AVG_EXP_2RT = auto()
    PRZZ_LIMIT = auto()

def compute_m1_candidate(
    candidate: M1ExpCandidate,
    R: float,
    theta: float,
    Q_coeffs: list
) -> float:
    """Compute exp-component for m1 candidate."""
    if candidate == M1ExpCandidate.EXP_R:
        return math.exp(R)
    elif candidate == M1ExpCandidate.E_EXP_2RT_UNDER_Q2:
        # E[exp(2Rt)] where t ~ Q(t)² on [0,1]
        return _compute_weighted_exp_expectation(R, Q_coeffs)
    elif candidate == M1ExpCandidate.UNIFORM_AVG_EXP_2RT:
        # ∫₀¹ exp(2Rt) dt = (exp(2R) - 1) / (2R)
        return (math.exp(2*R) - 1) / (2*R)
    elif candidate == M1ExpCandidate.PRZZ_LIMIT:
        return math.exp(2*R/theta)
```

**Acceptance Criteria:**
- [ ] A candidate "wins" only if it matches implied m₁ across BOTH benchmarks within 5% WITHOUT tuning
- [ ] If no candidate wins, document and accept empirical exp(R)+5

---

## Phase 19.5 — K≥4 Safety

### 19.5.1 Add K-sweep harness

**Goal:** Verify residual doesn't scale badly with K.

**Files:**
- `scripts/k_sweep_report.py` (NEW)
- `tests/test_k_sweep_monotonic.py` (NEW)

**Metrics to Measure:**
- implied m₁ error vs K
- δ vs K
- κ error vs K (when K=4 polynomials available)

**Expected Behavior:**
- K=3: m₁ = exp(R) + 5 (2K-1 = 5)
- K=4: m₁ = exp(R) + 7 (2K-1 = 7)

**Acceptance Criteria:**
- [ ] No blow-up in δ or implied-m₁ residual as K increases
- [ ] Clear identification of which bucket scales with K

---

## Phase 19.6 — Refactor evaluate.py Safely

### 19.6.1 Extract public API modules with snapshot tests

**Goal:** Refactor 6700-line evaluate.py without behavior drift.

**Step 1: Add Snapshot Tests FIRST**

```python
# tests/test_snapshot_kappa.py
"""Snapshot tests for κ benchmark - run BEFORE any refactoring."""

import pytest
from src.evaluate import compute_c_paper_with_mirror

KAPPA_SNAPSHOT = {
    "c": 2.109,  # Within 1.35% of target
    "S12_plus": ...,
    "S12_minus": ...,
    "S34_plus": ...,
}

def test_kappa_snapshot():
    result = compute_c_paper_with_mirror("kappa")
    assert abs(result.c - KAPPA_SNAPSHOT["c"]) < 0.01
    # ... more assertions
```

**Step 2: Extract Modules in Order**

| Order | Module | Contents | LOC Est |
|-------|--------|----------|---------|
| 1 | `src/evaluator/mirror.py` | Mirror assembly functions | 500 |
| 2 | `src/evaluator/benchmarks.py` | Benchmark loading | 200 |
| 3 | `src/evaluator/channels.py` | S12, S34 assembly | 800 |
| 4 | `src/evaluator/compute_c.py` | Main c computation | 400 |
| 5 | `src/evaluate.py` | CLI entrypoint only | 200 |

**Acceptance Criteria:**
- [ ] Snapshot tests pass BEFORE refactor
- [ ] Snapshot tests pass AFTER each module extraction
- [ ] evaluate.py reduced to <500 lines
- [ ] All 445+ existing tests still pass

---

## Implementation Order

```
Phase 19.0.1 → 19.1.1 → 19.2.1 → 19.3.1 → 19.4.1 → 19.4.2 → 19.5.1 → 19.6.1
     ↓            ↓           ↓           ↓           ↓           ↓
  Guardrails   +5 split   ζ modes    Delta v2    m1 bucket   m1 cands   K-sweep   Refactor
```

**Dependencies:**
- 19.0.1 must complete before 19.1.1 (mode infrastructure)
- 19.2.1 must complete before 19.3.1 (zeta modes used in delta)
- 19.4.1 must complete before 19.4.2 (breakdown needed for candidates)
- 19.6.1 should be LAST (highest risk)

---

## Success Criteria (Phase 19 Complete)

| Criterion | Target | Verification |
|-----------|--------|--------------|
| Mode separation | Semantic/Numeric explicit | test_zeta_eval_modes |
| I₅ guardrail | Raises in main mode | test_no_I5_in_main_mode |
| Delta decomposition | JSON report | run_delta_report.py |
| m₁ breakdown | Per-bucket analysis | test_implied_m1_stability |
| K≥4 safety | No blow-up | test_k_sweep_monotonic |
| evaluate.py | <500 lines | Line count |
| All tests | Pass | pytest |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Refactor breaks production | Medium | High | Snapshot tests first |
| m₁ candidates all fail | High | Low | Accept empirical formula |
| K=4 blow-up | Low | Medium | Early detection via sweep |
| Semantic mode gives wrong answer | Low | High | Two-benchmark gate |

---

## Files Created/Modified Summary

**New Files (12):**
- `src/evaluation_modes.py`
- `src/ratios/zeta_eval.py`
- `src/mirror/implied_m1.py`
- `src/evaluator/mirror.py`
- `src/evaluator/benchmarks.py`
- `src/evaluator/channels.py`
- `src/evaluator/compute_c.py`
- `scripts/run_delta_report.py`
- `scripts/run_implied_m1_breakdown.py`
- `scripts/scan_m1_candidates.py`
- `scripts/k_sweep_report.py`
- `tests/test_no_I5_in_main_mode.py`
- `tests/test_zeta_eval_modes.py`
- `tests/test_delta_invariants.py`
- `tests/test_implied_m1_stability.py`
- `tests/test_m1_candidates.py`
- `tests/test_k_sweep_monotonic.py`
- `tests/test_snapshot_kappa.py`
- `tests/test_snapshot_kappa_star.py`

**Modified Files (5):**
- `src/ratios/microcase_plus5_signature_k3.py`
- `src/ratios/j1_k3_decomposition.py`
- `src/ratios/delta_track.py`
- `src/mirror/m1_derived.py`
- `tests/test_plus5_gate.py`

---

## References

- TRUTH_SPEC.md Section 4: I₅ is error term
- TRUTH_SPEC.md Lines 1621-1628: Canonical I₅ citation
- DERIVE_ZETA_LOGDERIV_FACTOR.md: Laurent vs ACTUAL modes
- PHASE_18_SUMMARY.md: Current state and findings
- GPT Phase 19 Guidance (2025-12-24): Task list source
