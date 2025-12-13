# CLAUDE.md — Project Guidelines for PRZZ Extension (κ Optimization)

## Project Purpose

Implement and extend the PRZZ (Pratt–Robles–Zaharescu–Zeindler, 2019) framework for computing
κ, the proportion of Riemann zeta zeros on the critical line, using Levinson/Feng-style mollifiers.

**Baseline target (Phase 0):**
- Reproduce PRZZ's reported numerical optimization: κ ≥ 0.417293962 (with K=3, d=1, θ=4/7, R=1.3036).

**Research goal:**
- Explore improved mollifier configurations (polynomial degrees, K=4, R sweep, possibly d=2)
  to plausibly push κ above 0.42.

**Key engineering insight:**
- For fixed R=1.3036, improving κ from 0.417293962 → 0.42 requires only ~0.35% reduction in c.

---

## Definitions and Conventions (DO NOT GUESS)

### κ bound
We use the Levinson-type bound:
```
κ ≥ 1 - (1/R)·log(c)
```

where:
- R is the shift in σ₀ = 1/2 − R/log T,
- c is the main-term constant in the asymptotic for the mollified mean square.

### Indexing of mollifier "pieces"
Be careful: PRZZ/Feng "piece number" and "Λ-convolution count" can be off-by-one in informal notes.
For this project we adopt:

- Piece index ℓ ∈ {1,…,K} corresponds to the polynomial P_ℓ.
- The ℓ-th piece uses coefficients proportional to (μ ⋆ Λ^{⋆(ℓ−1)})(n) / (log N)^{ℓ−1}.
  (So ℓ=1 corresponds to the μ piece.)

Cross terms are indexed by (ℓ₁,ℓ₂) with 1 ≤ ℓ₁ ≤ ℓ₂ ≤ K.

### Series variables
For pair (ℓ₁,ℓ₂) we must keep distinct formal variables:
- x₁,…,x_{ℓ₁} and y₁,…,y_{ℓ₂}
each with derivative order 1 (because the residues are from dz/z², dw/w²).

**Important:** even if no ζ(…+w₁+w₂) coupling exists, you still cannot "compress" w's into one variable.
Mixed derivatives require distinct y-variables.

---

## Non‑Negotiable Technical Rules

### Rule A: No finite differences for derivatives at 0
All x/y derivatives must be computed via a truncated multi-variable Taylor/jet engine.
Finite differences will be unstable and will sabotage optimization.

### Rule B: Multi-variable support is required
The (1,2) pilot establishes that we need:
```python
vars = ("x", "y1", "y2")
deriv_orders = {"x": 1, "y1": 1, "y2": 1}
```
even though there is no ζ(...+w₁+w₂) coupling.

Maximum variable counts for K=3, d=1:
- (3,3) uses 6 vars → at most 2^6 = 64 monomials (since each var is order ≤ 1).

For K=4, d=1:
- (4,4) uses 8 vars → at most 2^8 = 256 monomials.

### Rule C: Validation is continuous and term-by-term
We must validate:
- each module (polynomials, quadrature, series),
- then each pair (ℓ₁,ℓ₂),
- then the full c and κ.

If any checkpoint fails: STOP and report.

---

## "Memorize These" Regression Targets

Use these as regression tests for Phase 0 reproduction (from printed PRZZ digits):

```
R = 1.3036
θ = 4/7 ≈ 0.5714285714285714
κ_target = 0.417293962
c_target = exp(R(1-κ_target)) = 2.13745440613217263636...
```

Relationship:
```
κ = 1 - log(c)/R
c = exp(R(1-κ))
```

For the "κ=0.42" goal (fixed R=1.3036):
```
c(0.42) = exp(R·(1-0.42)) ≈ 2.12993
required relative drop ≈ 0.35%
```

**Tolerance guidance:**
- For Phase 0: match κ to 1e−6 and c to ~1e−6–1e−5 (depends on quadrature settings).
- Always include a quadrature convergence test (n=60/80/100).

---

## Project Phases and Milestones

### Phase 0 — Reproduce PRZZ κ exactly
**Deliverable:**
- End-to-end computation returns κ ≈ 0.417293962 (within tolerance).
- Per-pair breakdown c_{ℓ₁ℓ₂} logged and stable under quadrature refinement.

### Phase 1 — "Polish" within K=3, d=1
**Deliverables:**
- Increase polynomial degrees and re-optimize.
- Outer-loop scan over R (e.g. R ∈ [1.2, 1.4]).
- Multiple random restarts / alternating minimization.
- Document any κ improvement and the resulting polynomials.

### Phase 2 — K=4, d=1
**Deliverables:**
- Implement (ℓ₁,ℓ₂) pairs for K=4 (10 pair types).
- Start P₄ at low degree; re-optimize everything.
- Target: demonstrate whether κ can exceed 0.42.

### Phase 3 — d=2 (only if needed)
**Deliverables:**
- Expand infrastructure for d>1 (expect more bookkeeping).
- Proceed only if K=4 saturates.

---

## Development Rules (Claude Workflow)

### Rule 1: "No code without permission" (unless explicitly authorized)
- Present a concrete plan (files touched, tests added, acceptance criteria).
- Get explicit approval before writing implementation code or major refactors.

**Exceptions:**
- Small doc edits and bugfixes in docs/tests may proceed without permission.

### Rule 2: Tests before implementation
- Unit tests for each module.
- Integration tests for (1,1) first, then add pairs gradually.

### Rule 3: Always keep a "golden" regression run
- Store a reference JSON of parameters and expected c/κ.
- Any change that breaks regression must be justified (e.g. improved quadrature accuracy).

### Rule 4: Frequent commits with descriptive messages
- Commit after each successful validation checkpoint
- Never commit failing tests
- Use feature branches for new development

---

## Repository Structure

```
przz-extension/
├── CLAUDE.md                  # This file
├── TECHNICAL_ANALYSIS.md      # Complete mathematical derivations
├── ARCHITECTURE.md            # Implementation design
├── VALIDATION.md              # Test plan and checkpoints
├── TERM_DSL.md                # Term data model specification
├── OPTIMIZATION.md            # Strategy for improving κ
├── src/
│   ├── __init__.py
│   ├── polynomials.py         # Polynomial representation + constraints
│   ├── quadrature.py          # Gauss-Legendre integration
│   ├── series.py              # Multi-variable truncated series (bitset)
│   ├── term_dsl.py            # Term DSL definitions
│   ├── terms_k3_d1.py         # Manual term tables for K=3, d=1
│   ├── evaluate.py            # Compute c and κ
│   └── optimize.py            # Parameter optimization
├── tests/
│   ├── test_polynomials.py
│   ├── test_quadrature.py
│   ├── test_series.py
│   ├── test_term_dsl.py
│   ├── test_terms_k3_d1.py
│   ├── test_evaluate.py
│   └── test_regression_przz.py
└── data/
    ├── przz_parameters.json   # Published PRZZ values
    └── golden_outputs.json    # Expected outputs for regression
```

---

## Known Pitfalls / Warnings

1. **Do not assume "all integrals are [0,1]²."**
   - Many leading terms can be reduced to 2D, but some lower-order/arithmetic-correction terms
     may introduce extra integrals (e.g., Mellin variable or additional t's).
   - Architecture must support at least 3D quadrature and/or special-function evaluation.

2. **Be consistent about what c means.**
   - c is the main-term constant used in κ = 1 - log(c)/R.
   - PRZZ may drop lower-order corrections (I5-type); we must match what their numerical κ used.

3. **Indexing off-by-one is a common error.**
   - Confirm whether ℓ=1 corresponds to μ or μ⋆Λ.
   - Ensure P₁,P₂,P₃ align with the PRZZ numerical setup.

4. **Quadrature noise can masquerade as κ improvements.**
   - Always confirm improvements persist across n=60/80/100.
   - Consider a high-precision spot-check with mpmath for 2–3 random parameter points.

5. **Variable compression is FORBIDDEN.**
   - Even without ζ(...+w₁+w₂) coupling, you cannot set w₁=w₂=y.
   - This mixes mixed derivatives with pure second derivatives.

6. **Q(0) rounding and enforcement modes.**
   - PRZZ printed coefficients sum to Q(0)=0.999999, not exactly 1.0
   - Phase 0 reproduction should try **both** `enforce_Q0=False` (paper-literal) and `True` (constraint)
   - Log which mode matches downstream `c` best

---

## JSON Schema Conventions

**Canonical keys (prefer these when both exist):**
- P₁/P₂/P₃: use `tilde_coeffs`
- Q: use `coeffs_in_basis_terms`

**Schema version:** 1 (see `data/przz_parameters.json`)

---

## Success Criteria (Phase 0)

- [x] Polynomials module reproduces PRZZ polynomials exactly and respects constraints *(completed 2025-12-12)*
- [x] Quadrature module passes known integral tests *(completed 2025-12-13, 46 tests)*
- [x] Series engine passes symbolic coefficient tests; no finite differences used *(completed 2025-12-13, 57 tests)*
- [ ] (1,1) pair matches PRZZ sub-result (if available) and is stable under quadrature refinement
- [ ] Full K=3 assembly yields c ≈ 2.1374544061321726 and κ ≈ 0.417293962
- [ ] All tests passing; validated with quadrature convergence sweep
- [ ] Per-pair breakdown logged: c₁₁, c₂₂, c₃₃, c₁₂, c₁₃, c₂₃

---

## Implementation Order (STRICT)

1. ~~`polynomials.py` + tests → Validate PRZZ polynomials satisfy constraints~~ **DONE**
2. ~~`quadrature.py` + tests → Validate on known integrals (∫x^k dx = 1/(k+1))~~ **DONE**
3. ~~`series.py` + tests → Validate derivative extraction on symbolic examples~~ **DONE**
4. `term_dsl.py` + tests → Define Term structure, AffineExpr ← **NEXT**
5. `terms_k3_d1.py` - Start with (1,1) ONLY
6. `evaluate.py` + tests → Compute c for (1,1) only first, validate
7. Add remaining pairs one-by-one: (1,2), (1,3), (2,2), (2,3), (3,3)
8. Full integration test: c = 2.1374544, κ = 0.417293962
