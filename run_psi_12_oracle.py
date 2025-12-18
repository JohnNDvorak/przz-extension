#!/usr/bin/env python3
"""
Quick runner for Ψ_{1,2} oracle to test the fundamental hypothesis:

Does the Ψ expansion eliminate the catastrophic cancellation seen in the DSL?

DSL (1,2) ratio: 129× (catastrophic!)
Target ratio: ~1.1×
"""

from src.psi_12_oracle import psi_oracle_12
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

print("=" * 70)
print("Ψ_{1,2} Oracle Test: Full 7-Monomial Expansion")
print("=" * 70)

theta = 4/7

# Test with κ polynomials (R=1.3036)
print("\n" + "─" * 70)
print("BENCHMARK 1: κ polynomials (R=1.3036)")
print("─" * 70)
P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
R_kappa = 1.3036
result_k = psi_oracle_12(P1_k, P2_k, Q_k, theta, R_kappa, n_quad=80, debug=True)

# Test with κ* polynomials (R=1.1167)
print("\n" + "─" * 70)
print("BENCHMARK 2: κ* polynomials (R=1.1167)")
print("─" * 70)
P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
R_kappa_star = 1.1167
result_ks = psi_oracle_12(P1_ks, P2_ks, Q_ks, theta, R_kappa_star, n_quad=80, debug=True)

# Compare ratios
print("\n" + "=" * 70)
print("RATIO ANALYSIS")
print("=" * 70)

ratio = result_k.total / result_ks.total
print(f"\nΨ_{1,2} oracle ratio (κ/κ*): {ratio:.4f}")
print(f"DSL (1,2) ratio:              129.0 (CATASTROPHIC!)")
print(f"PRZZ target ratio:            ~1.1")
print()

if abs(ratio) < 10.0:
    print("✓ EXCELLENT: Ratio is much better than DSL!")
    print("  The Ψ expansion successfully eliminates the catastrophic cancellation.")
elif abs(ratio) < 50.0:
    print("✓ GOOD: Ratio is better than DSL, but not at target yet.")
    print("  May need to refine A,B,C,D definitions or add derivative terms.")
else:
    print("✗ PROBLEM: Ratio is still very large.")
    print("  The current implementation may not be capturing the full Ψ structure.")

# Cancellation analysis for κ*
print("\n" + "─" * 70)
print("κ* CANCELLATION ANALYSIS")
print("─" * 70)

contributions_ks = [
    ("AB²",  result_ks.AB2,      +1),
    ("ABC",  result_ks.ABC,      -2),
    ("AC²",  result_ks.AC2,      +1),
    ("B²C",  result_ks.B2C,      -1),
    ("C³",   result_ks.C3,       +1),
    ("DB",   result_ks.DB,       +2),
    ("DC",   result_ks.DC,       -2),
]

print("\nMonomial contributions (before coefficients):")
for name, val, coeff in contributions_ks:
    contrib = coeff * val
    sign = "+" if contrib >= 0 else ""
    print(f"  {name:4s}: {val:10.6f} × {coeff:+2d} = {sign}{contrib:10.6f}")

weighted_contribs = [coeff * val for _, val, coeff in contributions_ks]
sum_positive = sum(c for c in weighted_contribs if c > 0)
sum_negative = sum(c for c in weighted_contribs if c < 0)
net = sum_positive + sum_negative

print(f"\nSum of positives: {sum_positive:10.6f}")
print(f"Sum of negatives: {sum_negative:10.6f}")
print(f"Net (total):      {net:10.6f}")
print(f"|neg|/|pos| ratio: {abs(sum_negative/sum_positive):.6f}")

if abs(abs(sum_negative/sum_positive) - 1.0) < 0.05:
    print("\n⚠ WARNING: Near-perfect cancellation detected!")
    print("  This is similar to the DSL issue. May need different A,B,C,D definitions.")
else:
    print("\n✓ Healthy cancellation pattern (not catastrophic)")

print("\n" + "=" * 70)
