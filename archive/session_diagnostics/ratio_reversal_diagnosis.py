"""
Ratio Reversal Diagnosis - Precise Numerical Breakdown

This script computes exact numerical values to diagnose the ratio reversal issue.

Context:
- c = const × ∫Q²e^{2Rt}dt
- t-integral ratio (κ/κ*): 1.171
- PRZZ needs const ratio: 0.942
- Our naive gives: 1.71 (wrong!)

Steps:
1. Load both polynomial sets (κ and κ*)
2. Compute per-pair I₂ values (no derivatives) for BOTH benchmarks
3. Compute full I₁+I₂+I₃+I₄ for (1,1) and (2,2) using oracle
4. Calculate what derivative contribution ratio would achieve target
5. Check (2,2) oracle behavior and understand ratio scaling
"""

from __future__ import annotations
import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.przz_22_exact_oracle import przz_oracle_22, gauss_legendre_01
from math import exp

# Constants
THETA = 4/7
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167

# PRZZ targets
C_KAPPA_TARGET = 2.13745440613217263636
C_KAPPA_STAR_TARGET = 1.9379524124677437

print("="*80)
print("RATIO REVERSAL DIAGNOSIS - PRECISE NUMERICAL BREAKDOWN")
print("="*80)

# =============================================================================
# Step 1: Load polynomial sets
# =============================================================================
print("\nSTEP 1: Loading polynomial sets")
print("-"*80)

P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

print(f"κ benchmark (R={R_KAPPA}):")
print(f"  P1 degree: {P1_k.to_monomial().degree}")
print(f"  P2 degree: {P2_k.to_monomial().degree}")
print(f"  P3 degree: {P3_k.to_monomial().degree}")
print(f"  Q degree: {Q_k.to_monomial().degree}")
print(f"  Q(0) = {Q_k.Q_at_zero():.10f}")

print(f"\nκ* benchmark (R={R_KAPPA_STAR}):")
print(f"  P1 degree: {P1_ks.to_monomial().degree}")
print(f"  P2 degree: {P2_ks.to_monomial().degree}")
print(f"  P3 degree: {P3_ks.to_monomial().degree}")
print(f"  Q degree: {Q_ks.to_monomial().degree}")
print(f"  Q(0) = {Q_ks.Q_at_zero():.10f}")

# =============================================================================
# Step 2: Compute per-pair I₂ values (no derivatives)
# =============================================================================
print("\nSTEP 2: Computing I₂ (pure polynomial × exponential integral)")
print("-"*80)
print("Formula: I₂ = (1/θ) × [∫P_ℓ₁(u)P_ℓ₂(u)du] × [∫Q²e^{2Rt}dt]")

n_quad = 100  # High precision

def compute_i2_value(P_ell1, P_ell2, Q, R, theta):
    """Compute I₂ for a given pair."""
    u_nodes, u_weights = gauss_legendre_01(n_quad)
    t_nodes, t_weights = gauss_legendre_01(n_quad)

    # u-integral: ∫P_ℓ₁(u)P_ℓ₂(u) du
    P1_u = P_ell1.eval(u_nodes)
    P2_u = P_ell2.eval(u_nodes)
    u_integral = np.sum(u_weights * P1_u * P2_u)

    # t-integral: ∫Q²e^{2Rt} dt
    Q_t = Q.eval(t_nodes)
    exp_2Rt = np.exp(2 * R * t_nodes)
    t_integral = np.sum(t_weights * Q_t * Q_t * exp_2Rt)

    I2 = (1.0 / theta) * u_integral * t_integral

    return I2, u_integral, t_integral

# Define pairs
pairs = [
    ("(1,1)", P1_k, P1_k, P1_ks, P1_ks),
    ("(2,2)", P2_k, P2_k, P2_ks, P2_ks),
    ("(3,3)", P3_k, P3_k, P3_ks, P3_ks),
    ("(1,2)", P1_k, P2_k, P1_ks, P2_ks),
    ("(1,3)", P1_k, P3_k, P1_ks, P3_ks),
    ("(2,3)", P2_k, P3_k, P2_ks, P3_ks),
]

print("\nPer-pair I₂ breakdown:")
print(f"{'Pair':<8} {'I₂(κ)':<12} {'I₂(κ*)':<12} {'Ratio':<10} {'u-int(κ)':<12} {'u-int(κ*)':<12} {'t-int(κ)':<12} {'t-int(κ*)':<12}")
print("-"*120)

i2_results = {}

for pair_name, P1_k_pair, P2_k_pair, P1_ks_pair, P2_ks_pair in pairs:
    I2_k, u_int_k, t_int_k = compute_i2_value(P1_k_pair, P2_k_pair, Q_k, R_KAPPA, THETA)
    I2_ks, u_int_ks, t_int_ks = compute_i2_value(P1_ks_pair, P2_ks_pair, Q_ks, R_KAPPA_STAR, THETA)

    ratio = I2_k / I2_ks if I2_ks != 0 else float('inf')

    i2_results[pair_name] = {
        'I2_k': I2_k,
        'I2_ks': I2_ks,
        'ratio': ratio,
        'u_int_k': u_int_k,
        'u_int_ks': u_int_ks,
        't_int_k': t_int_k,
        't_int_ks': t_int_ks
    }

    print(f"{pair_name:<8} {I2_k:<12.6f} {I2_ks:<12.6f} {ratio:<10.4f} {u_int_k:<12.6f} {u_int_ks:<12.6f} {t_int_k:<12.6f} {t_int_ks:<12.6f}")

# Compute total I₂
total_I2_k = sum(r['I2_k'] for r in i2_results.values())
total_I2_ks = sum(r['I2_ks'] for r in i2_results.values())
total_I2_ratio = total_I2_k / total_I2_ks

print(f"{'TOTAL':<8} {total_I2_k:<12.6f} {total_I2_ks:<12.6f} {total_I2_ratio:<10.4f}")

# =============================================================================
# Step 3: Compute full I₁+I₂+I₃+I₄ for key pairs using oracle
# =============================================================================
print("\n\nSTEP 3: Full oracle computation (I₁+I₂+I₃+I₄)")
print("-"*80)

# (1,1) pair - uses P1
print("\n(1,1) pair:")
print(f"{'Component':<12} {'κ value':<15} {'κ* value':<15} {'Ratio':<10}")
print("-"*60)

result_11_k = przz_oracle_22(P1_k, Q_k, THETA, R_KAPPA, n_quad=n_quad)
result_11_ks = przz_oracle_22(P1_ks, Q_ks, THETA, R_KAPPA_STAR, n_quad=n_quad)

for component in ['I1', 'I2', 'I3', 'I4', 'total']:
    val_k = getattr(result_11_k, component)
    val_ks = getattr(result_11_ks, component)
    ratio = val_k / val_ks if val_ks != 0 else float('inf')
    print(f"{component:<12} {val_k:<15.6f} {val_ks:<15.6f} {ratio:<10.4f}")

# (2,2) pair - uses P2
print("\n(2,2) pair:")
print(f"{'Component':<12} {'κ value':<15} {'κ* value':<15} {'Ratio':<10}")
print("-"*60)

result_22_k = przz_oracle_22(P2_k, Q_k, THETA, R_KAPPA, n_quad=n_quad)
result_22_ks = przz_oracle_22(P2_ks, Q_ks, THETA, R_KAPPA_STAR, n_quad=n_quad)

for component in ['I1', 'I2', 'I3', 'I4', 'total']:
    val_k = getattr(result_22_k, component)
    val_ks = getattr(result_22_ks, component)
    ratio = val_k / val_ks if val_ks != 0 else float('inf')
    print(f"{component:<12} {val_k:<15.6f} {val_ks:<15.6f} {ratio:<10.4f}")

# =============================================================================
# Step 4: Derivative contribution analysis
# =============================================================================
print("\n\nSTEP 4: What derivative ratio would fix the total?")
print("-"*80)

# Target ratio from PRZZ
c_ratio_target = C_KAPPA_TARGET / C_KAPPA_STAR_TARGET
print(f"PRZZ target c ratio (κ/κ*): {c_ratio_target:.6f}")

# I₂-only ratio
print(f"I₂-only ratio: {total_I2_ratio:.6f}")

# For (2,2) specifically:
I2_22_ratio = result_22_k.I2 / result_22_ks.I2
deriv_22_k = result_22_k.I1 + result_22_k.I3 + result_22_k.I4
deriv_22_ks = result_22_ks.I1 + result_22_ks.I3 + result_22_ks.I4
deriv_22_ratio = deriv_22_k / deriv_22_ks if deriv_22_ks != 0 else float('inf')

print(f"\n(2,2) pair breakdown:")
print(f"  I₂ ratio: {I2_22_ratio:.6f}")
print(f"  I₁+I₃+I₄ (κ): {deriv_22_k:.6f}")
print(f"  I₁+I₃+I₄ (κ*): {deriv_22_ks:.6f}")
print(f"  Derivative ratio: {deriv_22_ratio:.6f}")
print(f"  Total ratio: {result_22_k.total / result_22_ks.total:.6f}")

# What fraction should derivatives contribute?
# If c = I₂ + derivatives, and we want c_ratio = target
# Then: (I₂_k + d_k) / (I₂_ks + d_ks) = target
# Need to solve for what d_k/d_ks should be given I₂_k/I₂_ks

print(f"\nRequired correction analysis:")
print(f"  If derivatives were zero: ratio = {total_I2_ratio:.6f}")
print(f"  Target ratio: {c_ratio_target:.6f}")
print(f"  Current (2,2) total ratio: {result_22_k.total / result_22_ks.total:.6f}")

# =============================================================================
# Step 5: Component-by-component ratio tracking
# =============================================================================
print("\n\nSTEP 5: Per-component ratio analysis for (2,2)")
print("-"*80)

print("\nκ benchmark (R=1.3036):")
print(f"  I₁ = {result_22_k.I1:.6f}")
print(f"  I₂ = {result_22_k.I2:.6f}")
print(f"  I₃ = {result_22_k.I3:.6f}")
print(f"  I₄ = {result_22_k.I4:.6f}")
print(f"  Total = {result_22_k.total:.6f}")

print("\nκ* benchmark (R=1.1167):")
print(f"  I₁ = {result_22_ks.I1:.6f}")
print(f"  I₂ = {result_22_ks.I2:.6f}")
print(f"  I₃ = {result_22_ks.I3:.6f}")
print(f"  I₄ = {result_22_ks.I4:.6f}")
print(f"  Total = {result_22_ks.total:.6f}")

print("\nRatios (κ/κ*):")
print(f"  I₁ ratio: {result_22_k.I1 / result_22_ks.I1:.6f}")
print(f"  I₂ ratio: {result_22_k.I2 / result_22_ks.I2:.6f}")
print(f"  I₃ ratio: {result_22_k.I3 / result_22_ks.I3:.6f}")
print(f"  I₄ ratio: {result_22_k.I4 / result_22_ks.I4:.6f}")
print(f"  Total ratio: {result_22_k.total / result_22_ks.total:.6f}")

# =============================================================================
# Summary and key findings
# =============================================================================
print("\n\n" + "="*80)
print("SUMMARY OF KEY FINDINGS")
print("="*80)

print(f"\n1. T-INTEGRAL (∫Q²e^{{2Rt}}dt) RATIO:")
print(f"   κ value: {i2_results['(1,1)']['t_int_k']:.6f}")
print(f"   κ* value: {i2_results['(1,1)']['t_int_ks']:.6f}")
print(f"   Ratio: {i2_results['(1,1)']['t_int_k'] / i2_results['(1,1)']['t_int_ks']:.6f}")

print(f"\n2. POLYNOMIAL INTEGRAL RATIOS (∫P_ℓ₁P_ℓ₂ du):")
for pair_name in ['(1,1)', '(2,2)', '(3,3)']:
    u_ratio = i2_results[pair_name]['u_int_k'] / i2_results[pair_name]['u_int_ks']
    print(f"   {pair_name}: {u_ratio:.6f}")

print(f"\n3. I₂-ONLY TOTAL RATIO:")
print(f"   {total_I2_ratio:.6f} (vs PRZZ target {c_ratio_target:.6f})")

print(f"\n4. (2,2) ORACLE VERIFICATION:")
print(f"   Total ratio: {result_22_k.total / result_22_ks.total:.6f}")
print(f"   I₂ fraction (κ): {result_22_k.I2 / result_22_k.total:.4f}")
print(f"   I₂ fraction (κ*): {result_22_ks.I2 / result_22_ks.total:.4f}")
print(f"   Derivative contribution (κ): {deriv_22_k:.6f}")
print(f"   Derivative contribution (κ*): {deriv_22_ks:.6f}")

print(f"\n5. DERIVATIVE CORRECTION NEEDED:")
# If I₂ gives ratio X and we need ratio Y, derivatives must compensate
X = total_I2_ratio
Y = c_ratio_target
print(f"   Base I₂ ratio: {X:.6f}")
print(f"   Target ratio: {Y:.6f}")
print(f"   Gap: {X - Y:.6f} ({100*(X-Y)/Y:.2f}% of target)")

print("\n" + "="*80)
