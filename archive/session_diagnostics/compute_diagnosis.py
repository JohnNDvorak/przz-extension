#!/usr/bin/env python3
"""
Simplified diagnosis computation - computing exact numerical values.
"""

import sys
sys.path.insert(0, '/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension')

import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.przz_22_exact_oracle import przz_oracle_22, gauss_legendre_01

# Constants
THETA = 4/7
R_K = 1.3036
R_KS = 1.1167

print("Loading polynomials...")
P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

def compute_i2(P1, P2, Q, R, theta, n=100):
    """Compute I₂ = (1/θ) × [∫P₁P₂du] × [∫Q²e^{2Rt}dt]"""
    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    P1_u = P1.eval(u_nodes)
    P2_u = P2.eval(u_nodes)
    u_int = np.sum(u_weights * P1_u * P2_u)

    Q_t = Q.eval(t_nodes)
    exp_2Rt = np.exp(2 * R * t_nodes)
    t_int = np.sum(t_weights * Q_t * Q_t * exp_2Rt)

    return (1/theta) * u_int * t_int, u_int, t_int

print("\n" + "="*80)
print("STEP 2: I₂ COMPUTATIONS (Pure polynomial × exponential)")
print("="*80)

pairs = [
    ("(1,1)", P1_k, P1_ks),
    ("(2,2)", P2_k, P2_ks),
    ("(3,3)", P3_k, P3_ks),
]

print(f"\n{'Pair':<8} {'I₂(κ)':<12} {'I₂(κ*)':<12} {'Ratio':<10}")
print("-"*50)

i2_total_k = 0
i2_total_ks = 0

for name, Pk, Pks in pairs:
    I2_k, u_k, t_k = compute_i2(Pk, Pk, Q_k, R_K, THETA)
    I2_ks, u_ks, t_ks = compute_i2(Pks, Pks, Q_ks, R_KS, THETA)

    i2_total_k += I2_k
    i2_total_ks += I2_ks

    ratio = I2_k / I2_ks
    print(f"{name:<8} {I2_k:<12.6f} {I2_ks:<12.6f} {ratio:<10.4f}")

print(f"{'TOTAL':<8} {i2_total_k:<12.6f} {i2_total_ks:<12.6f} {i2_total_k/i2_total_ks:<10.4f}")

print("\n" + "="*80)
print("STEP 3: FULL ORACLE (I₁+I₂+I₃+I₄)")
print("="*80)

print("\n(1,1) pair:")
r11_k = przz_oracle_22(P1_k, Q_k, THETA, R_K, n_quad=100)
r11_ks = przz_oracle_22(P1_ks, Q_ks, THETA, R_KS, n_quad=100)

print(f"{'Component':<10} {'κ value':<15} {'κ* value':<15} {'Ratio':<10}")
print("-"*55)
for comp in ['I1', 'I2', 'I3', 'I4', 'total']:
    vk = getattr(r11_k, comp)
    vks = getattr(r11_ks, comp)
    print(f"{comp:<10} {vk:<15.6f} {vks:<15.6f} {vk/vks:<10.4f}")

print("\n(2,2) pair:")
r22_k = przz_oracle_22(P2_k, Q_k, THETA, R_K, n_quad=100)
r22_ks = przz_oracle_22(P2_ks, Q_ks, THETA, R_KS, n_quad=100)

print(f"{'Component':<10} {'κ value':<15} {'κ* value':<15} {'Ratio':<10}")
print("-"*55)
for comp in ['I1', 'I2', 'I3', 'I4', 'total']:
    vk = getattr(r22_k, comp)
    vks = getattr(r22_ks, comp)
    print(f"{comp:<10} {vk:<15.6f} {vks:<15.6f} {vk/vks:<10.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nI₂-only total ratio: {i2_total_k/i2_total_ks:.6f}")
print(f"PRZZ target c ratio: {2.137/1.938:.6f}")
print(f"\n(2,2) breakdown:")
print(f"  I₂ ratio: {r22_k.I2/r22_ks.I2:.6f}")
print(f"  I₁+I₃+I₄ (κ): {r22_k.I1+r22_k.I3+r22_k.I4:.6f}")
print(f"  I₁+I₃+I₄ (κ*): {r22_ks.I1+r22_ks.I3+r22_ks.I4:.6f}")
print(f"  Deriv ratio: {(r22_k.I1+r22_k.I3+r22_k.I4)/(r22_ks.I1+r22_ks.I3+r22_ks.I4):.6f}")
print(f"  Total ratio: {r22_k.total/r22_ks.total:.6f}")
