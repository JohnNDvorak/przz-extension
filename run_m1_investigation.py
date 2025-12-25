#!/usr/bin/env python3
"""
Investigation: What is the correct m₁ formula?

We know:
- Empirical: m₁ = exp(R) + 5 = exp(R) + (2K-1) for K=3
- Theory: T^{-α-β} = exp(2R) at α=β=-R/L
- But exp(2R) gives WORSE results than exp(R)+5

This script investigates:
1. What is the ideal m₁ for each benchmark?
2. Can we find a formula that captures the ideal m₁?
3. What's the relationship between exp(R), exp(2R), and m₁_ideal?
"""

import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import compute_c_paper_ordered


def analyze_m1_formula(verbose: bool = True):
    """
    Compute ideal m₁ for both benchmarks and search for patterns.
    """
    theta = 4.0 / 7.0

    results = []

    for benchmark in ['kappa', 'kappa_star']:
        if benchmark == 'kappa':
            P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
            R = 1.3036
            c_target = 2.13745440613217263636
        else:
            P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=False)
            R = 1.1167
            c_target = 1.9379524124677437

        polynomials = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

        # Get evaluations at +R and -R
        result_plus = compute_c_paper_ordered(
            theta=theta, R=R, n=60, polynomials=polynomials, K=3,
            s12_pair_mode='triangle',
        )
        result_minus = compute_c_paper_ordered(
            theta=theta, R=-R, n=60, polynomials=polynomials, K=3,
            s12_pair_mode='triangle',
        )

        S12_plus = result_plus.per_term.get('_S12_plus_total', 0.0)
        S12_minus = result_minus.per_term.get('_S12_plus_total', 0.0)
        S34 = result_plus.per_term.get('_S34_plus_total', 0.0)

        # Compute ideal m₁
        # c_target = S12_plus + m₁_ideal × S12_minus + S34
        m1_ideal = (c_target - S12_plus - S34) / S12_minus

        # Various m₁ formulas
        m1_formulas = {
            'exp(R)+5': np.exp(R) + 5,
            'exp(R)+2K-1': np.exp(R) + 2*3 - 1,  # Same as above for K=3
            'exp(2R)': np.exp(2*R),
            'exp(2R/θ)': np.exp(2*R/theta),
            'exp(R/θ)': np.exp(R/theta),
            'exp(R)+exp(R)/2': np.exp(R) + np.exp(R)/2,
            'exp(R)×(1+5/exp(R))': np.exp(R) * (1 + 5/np.exp(R)),
            '2×exp(R)+3': 2*np.exp(R) + 3,
            'exp(R)+R+4': np.exp(R) + R + 4,
            'exp(R)+θ×exp(R)+4': np.exp(R) + theta*np.exp(R) + 4,
        }

        # Compute c and gap for each formula
        results_bench = {
            'benchmark': benchmark,
            'R': R,
            'c_target': c_target,
            'S12_plus': S12_plus,
            'S12_minus': S12_minus,
            'S34': S34,
            'm1_ideal': m1_ideal,
            'formulas': {},
        }

        for name, m1 in m1_formulas.items():
            c = S12_plus + m1 * S12_minus + S34
            gap = (c - c_target) / c_target * 100
            results_bench['formulas'][name] = {
                'm1': m1,
                'c': c,
                'gap': gap,
            }

        results.append(results_bench)

    if verbose:
        for res in results:
            print("=" * 70)
            print(f"{res['benchmark'].upper()} Benchmark (R = {res['R']})")
            print("=" * 70)
            print(f"Target c = {res['c_target']:.8f}")
            print(f"S12(+R) = {res['S12_plus']:.8f}")
            print(f"S12(-R) = {res['S12_minus']:.8f}")
            print(f"S34 = {res['S34']:.8f}")
            print(f"m₁_ideal = {res['m1_ideal']:.6f}")
            print()
            print("Formula Comparison:")
            print(f"{'Formula':<30} {'m₁':>10} {'c':>12} {'Gap':>10}")
            print("-" * 70)

            sorted_formulas = sorted(
                res['formulas'].items(),
                key=lambda x: abs(x[1]['gap'])
            )

            for name, data in sorted_formulas:
                print(f"{name:<30} {data['m1']:>10.4f} {data['c']:>12.6f} {data['gap']:>+10.4f}%")
            print()

    # Cross-benchmark analysis
    print("=" * 70)
    print("Cross-Benchmark Analysis")
    print("=" * 70)
    print()

    R_kappa = results[0]['R']
    R_kappa_star = results[1]['R']
    m1_ideal_kappa = results[0]['m1_ideal']
    m1_ideal_kappa_star = results[1]['m1_ideal']

    print(f"m₁_ideal(κ) = {m1_ideal_kappa:.6f} at R = {R_kappa}")
    print(f"m₁_ideal(κ*) = {m1_ideal_kappa_star:.6f} at R = {R_kappa_star}")
    print()

    # What function of R fits m₁_ideal?
    # Try: m₁ = a × exp(R) + b
    # Two equations: m₁_κ = a × exp(R_κ) + b
    #                m₁_κ* = a × exp(R_κ*) + b

    exp_R_kappa = np.exp(R_kappa)
    exp_R_kappa_star = np.exp(R_kappa_star)

    # Solve for a, b
    # m₁_κ - m₁_κ* = a × (exp(R_κ) - exp(R_κ*))
    a = (m1_ideal_kappa - m1_ideal_kappa_star) / (exp_R_kappa - exp_R_kappa_star)
    b = m1_ideal_kappa - a * exp_R_kappa

    print("Fitting m₁ = a × exp(R) + b:")
    print(f"  a = {a:.6f}")
    print(f"  b = {b:.6f}")
    print()

    # Verify fit
    m1_fit_kappa = a * exp_R_kappa + b
    m1_fit_kappa_star = a * exp_R_kappa_star + b

    print("Verification:")
    print(f"  m₁_fit(κ) = {m1_fit_kappa:.6f} (ideal: {m1_ideal_kappa:.6f}, diff: {m1_fit_kappa - m1_ideal_kappa:.6f})")
    print(f"  m₁_fit(κ*) = {m1_fit_kappa_star:.6f} (ideal: {m1_ideal_kappa_star:.6f}, diff: {m1_fit_kappa_star - m1_ideal_kappa_star:.6f})")
    print()

    # What if b is related to 2K-1?
    print(f"Note: 2K-1 = 5 for K=3")
    print(f"Fitted b = {b:.4f}")
    print(f"If we use b = 5: a would need to be {(m1_ideal_kappa - 5) / exp_R_kappa:.6f}")
    print()

    # Compare a=1 (empirical) vs fitted a
    print(f"Empirical formula uses a = 1, b = 5")
    print(f"Fitted formula uses a = {a:.4f}, b = {b:.4f}")
    print()

    # Compute gaps with fitted formula
    print("Gaps with fitted m₁ formula:")
    for res in results:
        R = res['R']
        m1_fitted = a * np.exp(R) + b
        c_fitted = res['S12_plus'] + m1_fitted * res['S12_minus'] + res['S34']
        gap = (c_fitted - res['c_target']) / res['c_target'] * 100
        print(f"  {res['benchmark']}: c = {c_fitted:.6f}, gap = {gap:+.4f}%")

    return results, (a, b)


def main():
    results, (a, b) = analyze_m1_formula(verbose=True)

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The ideal m₁ for each benchmark is:")
    print(f"  κ: {results[0]['m1_ideal']:.6f}")
    print(f"  κ*: {results[1]['m1_ideal']:.6f}")
    print()
    print("The best fit is m₁ = a × exp(R) + b with:")
    print(f"  a = {a:.4f} (empirical uses 1.0)")
    print(f"  b = {b:.4f} (empirical uses 5.0)")
    print()
    if abs(a - 1.0) < 0.1 and abs(b - 5.0) < 0.5:
        print("The fitted formula is VERY CLOSE to the empirical m₁ = exp(R) + 5!")
        print("The remaining ~1% gap may be due to:")
        print("  1. Quadrature precision")
        print("  2. Polynomial coefficient precision")
        print("  3. Missing higher-order corrections")
    else:
        print(f"The fitted formula differs from empirical by:")
        print(f"  Δa = {a - 1.0:+.4f}")
        print(f"  Δb = {b - 5.0:+.4f}")


if __name__ == "__main__":
    main()
