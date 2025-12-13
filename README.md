# PRZZ Extension: Improving κ for Riemann Zeta Zeros

## Overview

This project implements and extends the PRZZ (Pratt-Robles-Zaharescu-Zeindler, 2019) 
framework for computing κ, the proportion of Riemann zeta function zeros that lie on 
the critical line Re(s) = 1/2.

**Current best result:** κ ≥ 0.417293962 (PRZZ, 2019)

**Goal:** Explore whether improved mollifier configurations can push κ > 0.42

## Quick Start

1. Read `CLAUDE.md` for project guidelines and development rules
2. Read `TECHNICAL_ANALYSIS.md` for mathematical background
3. Implementation starts with `src/polynomials.py`

## Key Insight

To achieve κ = 0.42 from the current 0.41729:
- Required improvement in c (mean value constant): **only 0.35%**
- This is achievable through optimization, not a moonshot

## Project Phases

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Reproduce κ = 0.417293962 | Not started |
| 1 | Polish K=3, d=1 optimization | - |
| 2 | Extend to K=4 mollifier pieces | - |
| 3 | (If needed) d=2 derivative depth | - |

## Documentation

- `CLAUDE.md` - Development guidelines and rules
- `TECHNICAL_ANALYSIS.md` - Complete mathematical derivations
- `ARCHITECTURE.md` - Implementation design
- `VALIDATION.md` - Test plan and checkpoints
- `TERM_DSL.md` - Term data model specification
- `OPTIMIZATION.md` - Strategy for improving κ

## Key Formula

```
κ ≥ 1 - (1/R)·log(c)

where:
- R = 1.3036 (shift parameter)
- c = 2.1374544... (mean value constant)
- κ = 0.417293962 (proportion on critical line)
```

## References

- PRZZ (2019): "More than five-twelfths of the zeros of ζ are on the critical line"
  arXiv:1802.10521
- Feng (2012): Multi-piece mollifier approach
- Bui (2014): Survey of Levinson-type methods
