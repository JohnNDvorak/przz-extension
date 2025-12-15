# Target Assumptions (Phase 1.3 Checkpoint)

## Source of Target Values

- **Source**: PRZZ (arXiv:1802.10521v3) Section 8 - Numerical Aspects
- **c_target**: 2.13745440613217263636
- **kappa_target**: 0.417293962

## 1.3.1 Arithmetic Factor Model

**Question**: Does the PRZZ target use full bracket (with arithmetic factor S(alpha+beta)) or simplified bracket (A=1)?

### Evidence from TECHNICAL_ANALYSIS.md

1. **Section 7.4** states:
   > "The A-factor corrections appear as 'I5-type' terms that are lower-order (suppressed by 1/(log N)^2)."
   > "Strategy: Start by implementing ratio-only brackets (A = 1). Add A-correction terms as optional 'audit' to **confirm they match what PRZZ included**."

2. **Section 9.5** states:
   > "Full = (A1*B^2 + 2BC) - 2B*S(alpha+beta)"
   > "The -2B*S term is the I5-type arithmetic correction."

3. **Section 10.3** mentions:
   > "I5 (arithmetic correction, lower order)"

### Conclusion

**The PRZZ target LIKELY includes the full bracket with arithmetic factor.**

Supporting evidence:
- The phrase "to confirm they match what PRZZ included" implies PRZZ's numerical optimization included I5 terms
- Our current computed value is 2% HIGH (c = 2.183 vs target 2.137)
- Missing a NEGATIVE correction term (I5 has a minus sign: -2B*S) would cause us to be too high
- This is consistent with I5 being the missing component

**Decision**: Implement I5 to close the gap. If discrepancy remains after I5, investigate bracket weights.

## 1.3.2 Q Normalization Convention

**Question**: Which `enforce_Q0` mode does the PRZZ target correspond to?

### Numerical Comparison

```
enforce_Q0=True:  Q(0) = 1.000000, c = 2.183162009, kappa = 0.401063
enforce_Q0=False: Q(0) = 0.999999, c = 2.183160155, kappa = 0.401064
Difference: Delta_c = 1.85e-6 (negligible vs 0.046 discrepancy)
```

### Conclusion

**The Q normalization mode has negligible impact (~2e-6 vs ~0.046 discrepancy).**

Either mode produces essentially identical results. The 2% discrepancy is NOT from Q normalization.

**Decision**: Use `enforce_Q0=True` (normalized Q(0)=1) for consistency and cleaner mathematics.

## Summary

| Assumption | Decision | Justification |
|------------|----------|---------------|
| Arithmetic factor model | Full bracket (I5 needed) | PRZZ included I5; we're 2% high |
| Q normalization | enforce_Q0=True | Difference is negligible |
| I5 expected effect | Reduce c by ~0.046 | Consistent with -2B*S sign |

## Next Steps

1. Proceed to Phase 2.0: Structural reducibility check
2. Build reference engine to validate normalization mapping
3. Implement I5 correction (Phase 3)

---
*Created: Phase 1.3 checkpoint completion*
*Last updated: 2025-12-14*
