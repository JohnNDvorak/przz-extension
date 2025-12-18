"""
Analyze derivative term contributions to understand the ratio reversal.

Context:
- Naive I₂-only formula gives κ/κ* ratio = 1.71 (wrong direction)
- Target ratio is 0.94
- Hypothesis: Derivative terms (I₁, I₃, I₄) subtract MORE from κ than κ*

Key insight:
- I₂ = (1/θ) × ∫∫ P²(u) Q²(t) e^{2Rt} du dt (NO derivatives)
- I₁, I₃, I₄ have derivatives of P and Q, evaluated at u values
- Higher degree polynomials → larger derivatives → more subtraction

This script computes:
1. I₂ (base integral, no derivatives)
2. I₁ (d²/dxdy derivatives)
3. I₃, I₄ (d/dx, d/dy derivatives)
4. Derivative contributions as fraction of I₂
5. Whether derivative subtraction can reverse ratio from 1.71 to 0.94
"""

from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from przz_22_exact_oracle import przz_oracle_22


def analyze_derivative_contributions(label: str, P1, P2, P3, Q, R: float, theta: float):
    """
    Analyze derivative contributions for a polynomial set.

    Returns:
        dict with I₁, I₂, I₃, I₄ values and analysis
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {label}")
    print(f"{'='*70}")

    # Compute (2,2) pair using oracle (exact PRZZ formulas)
    result = przz_oracle_22(P2, Q, theta, R, n_quad=80, debug=False)

    print(f"\nI-term values:")
    print(f"  I₁ = {result.I1:12.6f}  (d²/dxdy derivative)")
    print(f"  I₂ = {result.I2:12.6f}  (base integral, no derivatives)")
    print(f"  I₃ = {result.I3:12.6f}  (d/dx derivative)")
    print(f"  I₄ = {result.I4:12.6f}  (d/dy derivative)")
    print(f"  Total = {result.total:12.6f}")

    # Analyze derivative contributions
    deriv_total = result.I1 + result.I3 + result.I4
    deriv_fraction = deriv_total / result.I2 if result.I2 != 0 else 0

    print(f"\nDerivative analysis:")
    print(f"  Derivative sum (I₁+I₃+I₄) = {deriv_total:12.6f}")
    print(f"  Derivative / I₂ ratio = {deriv_fraction:12.6f}")
    print(f"  I₁ / I₂ = {result.I1/result.I2:12.6f}")
    print(f"  I₃ / I₂ = {result.I3/result.I2:12.6f}")
    print(f"  I₄ / I₂ = {result.I4/result.I2:12.6f}")

    # Polynomial degree info
    print(f"\nPolynomial degrees:")
    print(f"  P₂: degree {P2.to_monomial().degree}")
    print(f"  Q:  degree {Q.to_monomial().degree}")

    # Sample derivative magnitudes at key u values
    u_samples = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    P2_vals = P2.eval(u_samples)
    P2_prime = P2.eval_deriv(u_samples, 1)
    P2_double = P2.eval_deriv(u_samples, 2)

    print(f"\nP₂ derivatives at sample u values:")
    print(f"  u:         {' '.join(f'{u:8.2f}' for u in u_samples)}")
    print(f"  P₂(u):     {' '.join(f'{v:8.4f}' for v in P2_vals)}")
    print(f"  P₂'(u):    {' '.join(f'{v:8.4f}' for v in P2_prime)}")
    print(f"  P₂''(u):   {' '.join(f'{v:8.4f}' for v in P2_double)}")

    # Q derivatives at sample t values
    t_samples = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    Q_vals = Q.eval(t_samples)
    Q_prime = Q.eval_deriv(t_samples, 1)
    Q_double = Q.eval_deriv(t_samples, 2)

    print(f"\nQ derivatives at sample t values:")
    print(f"  t:         {' '.join(f'{t:8.2f}' for t in t_samples)}")
    print(f"  Q(t):      {' '.join(f'{v:8.4f}' for v in Q_vals)}")
    print(f"  Q'(t):     {' '.join(f'{v:8.4f}' for v in Q_prime)}")
    print(f"  Q''(t):    {' '.join(f'{v:8.4f}' for v in Q_double)}")

    return {
        'I1': result.I1,
        'I2': result.I2,
        'I3': result.I3,
        'I4': result.I4,
        'total': result.total,
        'deriv_sum': deriv_total,
        'deriv_fraction': deriv_fraction,
        'P2_degree': P2.to_monomial().degree,
        'Q_degree': Q.to_monomial().degree,
    }


def main():
    """Main analysis comparing κ and κ* polynomial sets."""

    theta = 4.0 / 7.0

    # Load κ polynomials (R=1.3036)
    print("Loading κ polynomials (Benchmark 1)...")
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    R_kappa = 1.3036

    # Load κ* polynomials (R=1.1167)
    print("Loading κ* polynomials (Benchmark 2)...")
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    R_kappa_star = 1.1167

    # Analyze both
    kappa_results = analyze_derivative_contributions(
        "κ Benchmark (R=1.3036)", P1_k, P2_k, P3_k, Q_k, R_kappa, theta
    )

    kappa_star_results = analyze_derivative_contributions(
        "κ* Benchmark (R=1.1167)", P1_ks, P2_ks, P3_ks, Q_ks, R_kappa_star, theta
    )

    # COMPARISON
    print(f"\n{'='*70}")
    print("RATIO ANALYSIS: κ / κ*")
    print(f"{'='*70}")

    print("\nI₂-only (naive formula) ratios:")
    I2_ratio = kappa_results['I2'] / kappa_star_results['I2']
    print(f"  I₂(κ) / I₂(κ*) = {I2_ratio:.4f}")
    print(f"  (This gives the wrong direction: 1.71)")

    print("\nDerivative contribution ratios:")
    deriv_ratio = kappa_results['deriv_sum'] / kappa_star_results['deriv_sum']
    print(f"  Deriv(κ) / Deriv(κ*) = {deriv_ratio:.4f}")

    print("\nDerivative as fraction of I₂:")
    print(f"  κ:  {kappa_results['deriv_fraction']:.4f}")
    print(f"  κ*: {kappa_star_results['deriv_fraction']:.4f}")
    print(f"  Ratio: {kappa_results['deriv_fraction'] / kappa_star_results['deriv_fraction']:.4f}")

    print("\nTotal c₂₂ ratios:")
    total_ratio = kappa_results['total'] / kappa_star_results['total']
    print(f"  Total(κ) / Total(κ*) = {total_ratio:.4f}")
    print(f"  (Target: 0.94)")

    print("\nPolynomial degree comparison:")
    print(f"  P₂: κ has degree {kappa_results['P2_degree']}, κ* has degree {kappa_star_results['P2_degree']}")
    print(f"  Q:  κ has degree {kappa_results['Q_degree']}, κ* has degree {kappa_star_results['Q_degree']}")

    # HYPOTHESIS TEST
    print(f"\n{'='*70}")
    print("HYPOTHESIS: Derivative subtraction reverses ratio?")
    print(f"{'='*70}")

    # If derivatives subtract MORE from κ than κ*, they could reverse the ratio
    # Total = I₂ + (deriv), where deriv < 0 for I₃, I₄
    # If κ has larger |deriv|, then κ total is smaller relative to I₂

    print("\nLogic:")
    print("  - I₂(κ) is 1.71× larger than I₂(κ*) [naive ratio]")
    print("  - Derivatives SUBTRACT from I₂")
    print("  - If derivatives subtract MORE from κ, this reduces the ratio")

    # Compute what derivative ratio would achieve target 0.94
    target_ratio = 0.94

    # Let's denote:
    # c_κ = I₂_κ + D_κ  (where D includes I₁, I₃, I₄)
    # c_κ* = I₂_κ* + D_κ*
    # We want: (I₂_κ + D_κ) / (I₂_κ* + D_κ*) = 0.94

    I2_k = kappa_results['I2']
    I2_ks = kappa_star_results['I2']
    D_k = kappa_results['deriv_sum']
    D_ks = kappa_star_results['deriv_sum']

    print(f"\nActual values:")
    print(f"  I₂(κ) = {I2_k:.4f}")
    print(f"  I₂(κ*) = {I2_ks:.4f}")
    print(f"  D(κ) = {D_k:.4f}")
    print(f"  D(κ*) = {D_ks:.4f}")

    actual_ratio = (I2_k + D_k) / (I2_ks + D_ks)
    print(f"\nActual ratio: {actual_ratio:.4f}")
    print(f"Target ratio: {target_ratio:.4f}")
    print(f"Gap: {actual_ratio - target_ratio:.4f}")

    # What D_k would we need to hit target 0.94?
    # (I₂_κ + D_κ) / (I₂_κ* + D_κ*) = 0.94
    # I₂_κ + D_κ = 0.94 * (I₂_κ* + D_κ*)
    # D_κ = 0.94 * (I₂_κ* + D_κ*) - I₂_κ

    D_k_needed = target_ratio * (I2_ks + D_ks) - I2_k

    print(f"\nTo achieve target ratio 0.94:")
    print(f"  D(κ) would need to be: {D_k_needed:.4f}")
    print(f"  Actual D(κ) is:        {D_k:.4f}")
    print(f"  Shortfall:             {D_k - D_k_needed:.4f}")
    print(f"  Shortfall as % of I₂:  {100*(D_k - D_k_needed)/I2_k:.2f}%")

    # CONCLUSION
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")

    if abs(actual_ratio - target_ratio) < 0.01:
        print("✓ Derivative terms successfully explain the ratio reversal!")
    else:
        print("✗ Derivative terms do NOT fully explain the ratio reversal.")
        print(f"  The (2,2) pair ratio is {actual_ratio:.4f}, not the target 0.94")
        print(f"  Additional factors must be involved (other pairs, I₅, etc.)")

    # Additional insight: degree effect
    print(f"\nDegree effect on derivatives:")
    if kappa_results['P2_degree'] > kappa_star_results['P2_degree']:
        print(f"  ✓ κ has HIGHER degree P₂ ({kappa_results['P2_degree']} vs {kappa_star_results['P2_degree']})")
        print(f"    → Larger derivatives → More subtraction from I₂")
    else:
        print(f"  ✗ Degree hypothesis does not hold")

    if kappa_results['Q_degree'] > kappa_star_results['Q_degree']:
        print(f"  ✓ κ has HIGHER degree Q ({kappa_results['Q_degree']} vs {kappa_star_results['Q_degree']})")
        print(f"    → Larger derivatives → More subtraction from I₂")
    else:
        print(f"  Note: κ has LOWER degree Q ({kappa_results['Q_degree']} vs {kappa_star_results['Q_degree']})")
        print(f"        → This works AGAINST the hypothesis")


if __name__ == "__main__":
    main()
