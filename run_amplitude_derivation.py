"""
run_amplitude_derivation.py
Derive A1, A2 amplitudes from TeX formula (Line 1548)

The goal is to derive A1 ≈ 5.94, A2 ≈ 7.98 from first principles.

TeX formula (line 1548):
    I₂ = T·Φ̂(0)/θ · ∫₀¹∫₀¹ Q(t)² e^{2Rt} P₁(u)P₂(u) dt du

The exp(2Rt) factor comes from the mirror term T^{-α-β} evaluated at α=β=-R/L.
"""

from __future__ import annotations

import numpy as np

from src.quadrature import gauss_legendre_01
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


THETA = 4.0 / 7.0
KAPPA_R = 1.3036
KAPPA_STAR_R = 1.1167

# Empirical residual amplitudes from Claude Task 1
A1_EMPIRICAL = 5.9408
A2_EMPIRICAL = 7.9794


def compute_tex_amplitude_integrals(
    Q,
    R: float,
    theta: float,
    n_quad: int = 200,
) -> dict:
    """
    Compute the TeX-motivated amplitude integrals.

    From line 1548: the integral involves Q(t)² exp(2Rt).

    Returns various candidate amplitude formulas.
    """
    nodes, weights = gauss_legendre_01(n_quad)

    # Evaluate Q(t) at nodes
    Q_vals = Q.eval(nodes)

    # Base integral: ∫₀¹ Q(t)² dt
    base_integral = np.sum(Q_vals**2 * weights)

    # Weighted integral: ∫₀¹ Q(t)² exp(2Rt) dt
    weighted_integral = np.sum(Q_vals**2 * np.exp(2 * R * nodes) * weights)

    # Ratio: weighted / base
    ratio = weighted_integral / base_integral if base_integral != 0 else float('inf')

    # Alternative weighting: ∫₀¹ Q(t)² exp(Rt) dt (single R, not 2R)
    weighted_R = np.sum(Q_vals**2 * np.exp(R * nodes) * weights)
    ratio_R = weighted_R / base_integral if base_integral != 0 else float('inf')

    # Mean of exp(2Rt) under Q(t)² weighting
    Z = np.sum(Q_vals**2 * weights)
    E_exp2Rt = np.sum(Q_vals**2 * np.exp(2 * R * nodes) * weights) / Z if Z != 0 else 0

    # Various candidate amplitude formulas
    candidates = {
        "exp(R)": np.exp(R),
        "exp(2R)": np.exp(2 * R),
        "(exp(2R)-1)/(2R)": (np.exp(2 * R) - 1) / (2 * R),
        "E[exp(2Rt)]": E_exp2Rt,
        "∫Q²e^{2Rt}/∫Q²": ratio,
        "∫Q²e^{Rt}/∫Q²": ratio_R,
        "exp(R)+5": np.exp(R) + 5,
        "exp(2R/θ)": np.exp(2 * R / theta),
        "1/θ·(exp(2R)-1)/(2R)": (np.exp(2 * R) - 1) / (2 * R * theta),
        "sqrt(exp(2R))=exp(R)": np.exp(R),
        "(1+exp(2R))/2": (1 + np.exp(2 * R)) / 2,
        "exp(R)·(exp(R)+1)/2": np.exp(R) * (np.exp(R) + 1) / 2,
    }

    return {
        "base_integral": base_integral,
        "weighted_integral": weighted_integral,
        "E_exp2Rt": E_exp2Rt,
        "candidates": candidates,
    }


def main():
    print("=" * 90)
    print("AMPLITUDE DERIVATION FROM TEX (Line 1548)")
    print("=" * 90)
    print()

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()

    print(f"Empirical residual amplitudes (from Claude Task 1):")
    print(f"  A1 = {A1_EMPIRICAL:.4f}")
    print(f"  A2 = {A2_EMPIRICAL:.4f}")
    print(f"  A_avg = {(A1_EMPIRICAL + A2_EMPIRICAL) / 2:.4f}")
    print()

    # Compute for κ benchmark
    print("=" * 90)
    print(f"κ BENCHMARK (R = {KAPPA_R})")
    print("=" * 90)
    print()

    result_k = compute_tex_amplitude_integrals(Q_k, KAPPA_R, THETA)

    print("TeX-MOTIVATED INTEGRALS:")
    print(f"  ∫₀¹ Q(t)² dt = {result_k['base_integral']:.6f}")
    print(f"  ∫₀¹ Q(t)² exp(2Rt) dt = {result_k['weighted_integral']:.6f}")
    print(f"  E[exp(2Rt)] under Q² = {result_k['E_exp2Rt']:.6f}")
    print()

    print("AMPLITUDE CANDIDATES vs EMPIRICAL:")
    print(f"{'Candidate':<30} {'Value':>12} {'vs A1':>12} {'vs A2':>12} {'vs A_avg':>12}")
    print("-" * 80)

    A_avg = (A1_EMPIRICAL + A2_EMPIRICAL) / 2

    for name, value in result_k["candidates"].items():
        diff_A1 = (value - A1_EMPIRICAL) / A1_EMPIRICAL * 100
        diff_A2 = (value - A2_EMPIRICAL) / A2_EMPIRICAL * 100
        diff_avg = (value - A_avg) / A_avg * 100
        print(f"{name:<30} {value:>12.4f} {diff_A1:>+11.1f}% {diff_A2:>+11.1f}% {diff_avg:>+11.1f}%")
    print()

    # Try combinations
    print("COMBINATION FORMULAS:")
    exp_R = np.exp(KAPPA_R)
    exp_2R = np.exp(2 * KAPPA_R)

    combos = {
        "exp(R) + 2": exp_R + 2,
        "exp(R) + 3": exp_R + 3,
        "exp(R) + 4": exp_R + 4,
        "exp(R) + 4.5": exp_R + 4.5,
        "exp(R) + 5": exp_R + 5,
        "2·exp(R)": 2 * exp_R,
        "1.5·exp(R) + 1": 1.5 * exp_R + 1,
        "exp(R)·(1+1/θ)": exp_R * (1 + 1/THETA),
        "(exp(2R)+1)/2": (exp_2R + 1) / 2,
        "sqrt(exp(2R)·(exp(2R)+1))": np.sqrt(exp_2R * (exp_2R + 1)),
        "exp(R)·sqrt(1+exp(-R))": exp_R * np.sqrt(1 + np.exp(-KAPPA_R)),
        "(exp(2R)-1)/(2R) + 1": (exp_2R - 1) / (2*KAPPA_R) + 1,
        "exp(R) + (exp(R)-1)/R": exp_R + (exp_R - 1) / KAPPA_R,
    }

    print(f"{'Formula':<35} {'Value':>12} {'vs A1':>12} {'vs A2':>12}")
    print("-" * 75)

    for name, value in combos.items():
        diff_A1 = (value - A1_EMPIRICAL) / A1_EMPIRICAL * 100
        diff_A2 = (value - A2_EMPIRICAL) / A2_EMPIRICAL * 100
        print(f"{name:<35} {value:>12.4f} {diff_A1:>+11.1f}% {diff_A2:>+11.1f}%")
    print()

    # Check if A1, A2 have a simple relationship
    print("=" * 90)
    print("A1/A2 RELATIONSHIP ANALYSIS")
    print("=" * 90)
    print()

    ratio_A = A1_EMPIRICAL / A2_EMPIRICAL
    print(f"A1/A2 = {ratio_A:.4f}")
    print(f"A2 - A1 = {A2_EMPIRICAL - A1_EMPIRICAL:.4f}")
    print()

    # Check theoretical ratios
    theo_ratios = {
        "1/θ": 1/THETA,
        "θ": THETA,
        "1 - θ/2": 1 - THETA/2,
        "1 + θ/4": 1 + THETA/4,
        "exp(-θR)": np.exp(-THETA * KAPPA_R),
        "1 - 1/(2θ)": 1 - 1/(2*THETA),
        "3/4": 0.75,
        "4/5": 0.8,
        "7/9": 7/9,
    }

    print(f"{'Theoretical ratio':<20} {'Value':>10} {'Diff from A1/A2':>15}")
    print("-" * 50)
    for name, value in theo_ratios.items():
        diff = abs(value - ratio_A)
        print(f"{name:<20} {value:>10.4f} {diff:>15.4f}")
    print()

    # R-sweep to check amplitude formula stability
    print("=" * 90)
    print("R-SWEEP: CHECK IF AMPLITUDE FORMULA IS STABLE")
    print("=" * 90)
    print()

    R_values = [1.0, 1.15, 1.3036, 1.4, 1.5]

    print(f"{'R':>8} {'exp(R)+2':>12} {'exp(R)+4.5':>12} {'E[exp(2Rt)]':>12} {'(exp(2R)-1)/(2R)':>18}")
    print("-" * 65)

    for R in R_values:
        exp_R = np.exp(R)
        exp_2R = np.exp(2 * R)
        result = compute_tex_amplitude_integrals(Q_k, R, THETA)

        print(f"{R:>8.4f} {exp_R + 2:>12.4f} {exp_R + 4.5:>12.4f} "
              f"{result['E_exp2Rt']:>12.4f} {(exp_2R - 1)/(2*R):>18.4f}")

    print()

    # Check κ* benchmark
    print("=" * 90)
    print(f"κ* BENCHMARK (R = {KAPPA_STAR_R})")
    print("=" * 90)
    print()

    result_ks = compute_tex_amplitude_integrals(Q_s, KAPPA_STAR_R, THETA)

    print("TeX-MOTIVATED INTEGRALS:")
    print(f"  ∫₀¹ Q(t)² dt = {result_ks['base_integral']:.6f}")
    print(f"  ∫₀¹ Q(t)² exp(2Rt) dt = {result_ks['weighted_integral']:.6f}")
    print(f"  E[exp(2Rt)] under Q² = {result_ks['E_exp2Rt']:.6f}")
    print()

    print("KEY AMPLITUDE CANDIDATES FOR κ* (R=1.1167):")
    exp_R_ks = np.exp(KAPPA_STAR_R)
    exp_2R_ks = np.exp(2 * KAPPA_STAR_R)

    ks_candidates = {
        "exp(R)+5": exp_R_ks + 5,
        "exp(R)+4.5": exp_R_ks + 4.5,
        "exp(R)+4": exp_R_ks + 4,
        "E[exp(2Rt)]": result_ks["E_exp2Rt"],
        "(exp(2R)-1)/(2R)": (exp_2R_ks - 1) / (2 * KAPPA_STAR_R),
    }

    for name, value in ks_candidates.items():
        print(f"  {name:<25} = {value:.4f}")
    print()

    # Find the best match
    print("=" * 90)
    print("BEST AMPLITUDE FORMULA SEARCH")
    print("=" * 90)
    print()

    # Try to find a formula that matches both A1 and A2
    # A1 corresponds to I₁ channel, A2 to I₂ channel
    # The difference might come from different Q-weighting structures

    print("Hypothesis: A1 and A2 differ because I₁ and I₂ have different t-structures.")
    print()

    # For I₂ (line 1548): Q(t)² exp(2Rt)
    # For I₁ (different structure): might have different weighting

    # Check if A2 ≈ exp(R) + 4.something
    for offset in np.arange(3.5, 5.5, 0.1):
        formula = np.exp(KAPPA_R) + offset
        if abs(formula - A2_EMPIRICAL) < 0.1:
            print(f"A2 ≈ exp(R) + {offset:.1f} = {formula:.4f} (exact: {A2_EMPIRICAL:.4f})")

    # Check if A1 ≈ exp(R) + something else
    for offset in np.arange(1.5, 3.5, 0.1):
        formula = np.exp(KAPPA_R) + offset
        if abs(formula - A1_EMPIRICAL) < 0.1:
            print(f"A1 ≈ exp(R) + {offset:.1f} = {formula:.4f} (exact: {A1_EMPIRICAL:.4f})")

    print()

    # Final summary
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print(f"A1 (I₁ channel) = {A1_EMPIRICAL:.4f}")
    print(f"A2 (I₂ channel) = {A2_EMPIRICAL:.4f}")
    print()
    print(f"At R = {KAPPA_R}:")
    print(f"  exp(R) = {np.exp(KAPPA_R):.4f}")
    print(f"  exp(R) + 2.2 ≈ {np.exp(KAPPA_R) + 2.2:.4f} (close to A1 = {A1_EMPIRICAL:.4f})")
    print(f"  exp(R) + 4.3 ≈ {np.exp(KAPPA_R) + 4.3:.4f} (close to A2 = {A2_EMPIRICAL:.4f})")
    print()
    print("The offset difference (4.3 - 2.2 = 2.1) may come from")
    print("the different I₁ vs I₂ integral structures in TeX.")
    print()


if __name__ == "__main__":
    main()
