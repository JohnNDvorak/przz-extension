"""
src/pair_11_diagnostic.py
Diagnostic for (1,1) pair only - this is B×B and should be correct.

If (1,1) alone shows the two-benchmark gap, the issue is NOT Case C integrals.
If (1,1) is consistent, then Case C integrals are the culprit.
"""

import numpy as np
import math
from src.terms_k3_d1 import make_I1_11, make_I2_11, make_I3_11, make_I4_11
from src.evaluate import evaluate_term
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def analyze_11_pair():
    """Analyze (1,1) pair for both benchmarks."""

    theta = 4.0 / 7.0
    R_k = 1.3036
    R_ks = 1.1167
    n_quad = 60

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_k = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    polys_ks = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    print("\n" + "=" * 70)
    print("(1,1) PAIR DIAGNOSTIC - B×B (NO CASE C INTEGRALS)")
    print("=" * 70)

    # Build and evaluate (1,1) terms for κ benchmark
    I1_11_k = evaluate_term(make_I1_11(theta, R_k), polys_k, n_quad)
    I2_11_k = evaluate_term(make_I2_11(theta, R_k), polys_k, n_quad)
    I3_11_k = evaluate_term(make_I3_11(theta, R_k), polys_k, n_quad)
    I4_11_k = evaluate_term(make_I4_11(theta, R_k), polys_k, n_quad)

    c_11_k = I1_11_k.value + I2_11_k.value + I3_11_k.value + I4_11_k.value

    print(f"\n--- κ Benchmark (R={R_k}) ---")
    print(f"  I₁(1,1) = {I1_11_k.value:+.6f}")
    print(f"  I₂(1,1) = {I2_11_k.value:+.6f}")
    print(f"  I₃(1,1) = {I3_11_k.value:+.6f}")
    print(f"  I₄(1,1) = {I4_11_k.value:+.6f}")
    print(f"  Total c(1,1) = {c_11_k:.6f}")

    # Build and evaluate (1,1) terms for κ* benchmark
    I1_11_ks = evaluate_term(make_I1_11(theta, R_ks), polys_ks, n_quad)
    I2_11_ks = evaluate_term(make_I2_11(theta, R_ks), polys_ks, n_quad)
    I3_11_ks = evaluate_term(make_I3_11(theta, R_ks), polys_ks, n_quad)
    I4_11_ks = evaluate_term(make_I4_11(theta, R_ks), polys_ks, n_quad)

    c_11_ks = I1_11_ks.value + I2_11_ks.value + I3_11_ks.value + I4_11_ks.value

    print(f"\n--- κ* Benchmark (R={R_ks}) ---")
    print(f"  I₁(1,1) = {I1_11_ks.value:+.6f}")
    print(f"  I₂(1,1) = {I2_11_ks.value:+.6f}")
    print(f"  I₃(1,1) = {I3_11_ks.value:+.6f}")
    print(f"  I₄(1,1) = {I4_11_ks.value:+.6f}")
    print(f"  Total c(1,1) = {c_11_ks:.6f}")

    # Compare ratios
    print("\n--- (1,1) Pair Ratios (κ / κ*) ---")
    print(f"  I₁ ratio: {I1_11_k.value / I1_11_ks.value:.4f}" if abs(I1_11_ks.value) > 1e-10 else "  I₁ ratio: N/A")
    print(f"  I₂ ratio: {I2_11_k.value / I2_11_ks.value:.4f}" if abs(I2_11_ks.value) > 1e-10 else "  I₂ ratio: N/A")
    print(f"  I₃ ratio: {I3_11_k.value / I3_11_ks.value:.4f}" if abs(I3_11_ks.value) > 1e-10 else "  I₃ ratio: N/A")
    print(f"  I₄ ratio: {I4_11_k.value / I4_11_ks.value:.4f}" if abs(I4_11_ks.value) > 1e-10 else "  I₄ ratio: N/A")
    print(f"  Total ratio: {c_11_k / c_11_ks:.4f}")

    # If (1,1) alone were c, what κ would we get?
    # κ = 1 - log(c)/R
    kappa_from_11_k = 1 - math.log(c_11_k) / R_k if c_11_k > 0 else float('nan')
    kappa_from_11_ks = 1 - math.log(c_11_ks) / R_ks if c_11_ks > 0 else float('nan')

    print("\n--- If (1,1) pair alone were c ---")
    print(f"  κ from c(1,1):  κ = {kappa_from_11_k:.6f} (target: 0.417294)")
    print(f"  κ* from c(1,1): κ* = {kappa_from_11_ks:.6f} (target: 0.407511)")

    # Target c values
    c_target_k = 2.13745440613217263636
    c_target_ks = 1.9379524124677437

    # What fraction of target c is (1,1)?
    frac_k = c_11_k / c_target_k * 100
    frac_ks = c_11_ks / c_target_ks * 100

    print("\n--- Fraction of Target c ---")
    print(f"  κ:  c(1,1) = {frac_k:.1f}% of target c = 2.137")
    print(f"  κ*: c(1,1) = {frac_ks:.1f}% of target c = 1.938")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if abs(frac_k - frac_ks) < 10:  # Within 10 percentage points
        print("""
The (1,1) pair contributes similarly to both benchmarks (as % of target).
This suggests:
1. The B×B formula is consistent
2. The gap must come from P₂/P₃ pairs (which need Case C integrals)
""")
    else:
        print(f"""
The (1,1) pair contributions differ significantly!
  κ:  {frac_k:.1f}% of target
  κ*: {frac_ks:.1f}% of target

This suggests even the B×B (1,1) pair has issues beyond Case C.
Possible causes:
1. Different polynomial structures between benchmarks
2. Missing normalization factor
3. Incorrect formula for (1,1) that depends on Q structure
""")

    # Check the polynomial-specific parts
    print("\n--- Polynomial Values at Key Points ---")
    print("  At u=0.5:")
    print(f"    κ:  P₁(0.5) = {P1_k.eval(np.array([0.5]))[0]:.6f}, Q(0.5) = {Q_k.eval(np.array([0.5]))[0]:.6f}")
    print(f"    κ*: P₁(0.5) = {P1_ks.eval(np.array([0.5]))[0]:.6f}, Q(0.5) = {Q_ks.eval(np.array([0.5]))[0]:.6f}")

    print("  At boundaries:")
    print(f"    κ:  P₁(0) = {P1_k.eval(np.array([0.0]))[0]:.6f}, P₁(1) = {P1_k.eval(np.array([1.0]))[0]:.6f}")
    print(f"    κ*: P₁(0) = {P1_ks.eval(np.array([0.0]))[0]:.6f}, P₁(1) = {P1_ks.eval(np.array([1.0]))[0]:.6f}")
    print(f"    κ:  Q(0) = {Q_k.eval(np.array([0.0]))[0]:.6f}, Q(1) = {Q_k.eval(np.array([1.0]))[0]:.6f}")
    print(f"    κ*: Q(0) = {Q_ks.eval(np.array([0.0]))[0]:.6f}, Q(1) = {Q_ks.eval(np.array([1.0]))[0]:.6f}")


if __name__ == "__main__":
    analyze_11_pair()
