"""
trace_22_monomials.py
Trace (2,2) monomial contributions for both benchmarks

This script computes each monomial's contribution separately to understand
where the ratio error originates.
"""

import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.psi_expansion import expand_psi, MonomialTwoC, classify_monomial
from src.psi_series_evaluator import PsiSeriesEvaluator


def trace_22_monomials(P2, Q, R: float, theta: float = 4/7, n_quad: int = 60,
                       label: str = "Benchmark"):
    """
    Trace all 12 monomials in (2,2) pair.

    Returns:
        Dict mapping monomial key to contribution value
    """
    print(f"\n{'='*70}")
    print(f"TRACE (2,2) MONOMIALS - {label} (R={R})")
    print(f"{'='*70}")

    # Get monomials
    monomials = expand_psi(2, 2)
    print(f"\nTotal monomials: {len(monomials)}")

    # Create series evaluator
    evaluator = PsiSeriesEvaluator(P2, P2, Q, R, theta, max_order=2, n_quad=n_quad)
    integral_grid = evaluator.compute_integral_grid()

    # Group monomials by (a, b)
    ab_groups = {}
    for m in monomials:
        key = (m.a, m.b)
        if key not in ab_groups:
            ab_groups[key] = []
        ab_groups[key].append(m)

    print(f"\nMonomials grouped by (a,b):")
    for (a, b), group in sorted(ab_groups.items()):
        print(f"  (a={a}, b={b}): {len(group)} monomials")
        for m in group:
            print(f"    {m}")

    # Compute contributions
    print(f"\n{'Monomial':<30} {'Coeff':>6} {'(a,b,w)':<12} {'Integral':>12} {'Contrib':>12}")
    print("-" * 80)

    contributions = {}
    total = 0.0

    for m in sorted(monomials, key=lambda x: x.key()):
        weight_exp = m.weight_exponent
        key = (m.a, m.b, weight_exp)
        integral_val = integral_grid.get(key, 0.0)
        contrib = m.coeff * integral_val
        contributions[m.key()] = contrib
        total += contrib

        mono_str = str(m)[:28]
        print(f"{mono_str:<30} {m.coeff:>+6} {str(key):<12} {integral_val:>+12.6f} {contrib:>+12.6f}")

    print("-" * 80)
    print(f"{'TOTAL':<30} {'':<6} {'':<12} {'':<12} {total:>+12.6f}")

    return contributions, total, integral_grid


def compare_benchmarks():
    """Compare (2,2) monomials between kappa and kappa-star benchmarks."""
    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()

    theta = 4.0 / 7.0
    R_kappa = 1.3036
    R_kappa_star = 1.1167

    # Trace both benchmarks
    contrib_k, total_k, grid_k = trace_22_monomials(
        P2_k, Q_k, R_kappa, theta, label="κ"
    )
    contrib_s, total_s, grid_s = trace_22_monomials(
        P2_s, Q_s, R_kappa_star, theta, label="κ*"
    )

    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON: κ vs κ*")
    print(f"{'='*70}")

    print(f"\n{'Monomial Key':<25} {'κ contrib':>12} {'κ* contrib':>12} {'Ratio':>10}")
    print("-" * 65)

    monomials = expand_psi(2, 2)
    for m in sorted(monomials, key=lambda x: x.key()):
        k_val = contrib_k.get(m.key(), 0.0)
        s_val = contrib_s.get(m.key(), 0.0)
        ratio = k_val / s_val if abs(s_val) > 1e-10 else float('inf')
        print(f"{str(m.key()):<25} {k_val:>+12.6f} {s_val:>+12.6f} {ratio:>10.4f}")

    print("-" * 65)
    overall_ratio = total_k / total_s if abs(total_s) > 1e-10 else float('inf')
    print(f"{'TOTAL':<25} {total_k:>+12.6f} {total_s:>+12.6f} {overall_ratio:>10.4f}")

    # Compare integral grids
    print(f"\n{'='*70}")
    print("INTEGRAL GRID COMPARISON")
    print(f"{'='*70}")

    print(f"\n{'(a,b,w)':<15} {'κ integral':>12} {'κ* integral':>12} {'Ratio':>10}")
    print("-" * 55)

    all_keys = set(grid_k.keys()) | set(grid_s.keys())
    for key in sorted(all_keys):
        if key[0] <= 2 and key[1] <= 2 and key[2] <= 4:  # Only relevant keys
            k_val = grid_k.get(key, 0.0)
            s_val = grid_s.get(key, 0.0)
            if abs(k_val) > 1e-10 or abs(s_val) > 1e-10:
                ratio = k_val / s_val if abs(s_val) > 1e-10 else float('inf')
                print(f"{str(key):<15} {k_val:>+12.6f} {s_val:>+12.6f} {ratio:>10.4f}")

    # P2 polynomial comparison
    print(f"\n{'='*70}")
    print("P₂ POLYNOMIAL COMPARISON")
    print(f"{'='*70}")

    u_test = np.linspace(0, 1, 11)
    print(f"\n{'u':<8} {'κ P₂(u)':>12} {'κ* P₂(u)':>12} {'Ratio':>10}")
    print("-" * 50)
    for u in u_test:
        k_val = P2_k.eval(np.array([u]))[0]
        s_val = P2_s.eval(np.array([u]))[0]
        ratio = k_val / s_val if abs(s_val) > 1e-10 else float('inf')
        print(f"{u:<8.2f} {k_val:>+12.6f} {s_val:>+12.6f} {ratio:>10.4f}")


def analyze_ratio_sources():
    """Analyze what causes the ratio difference."""
    print(f"\n{'='*70}")
    print("RATIO SOURCE ANALYSIS")
    print(f"{'='*70}")

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_s, P2_s, P3_s, Q_s = load_przz_polynomials_kappa_star()

    # Show polynomial info
    print("\nPolynomial evaluation at u=0.5:")
    print(f"  κ P₂(0.5): {P2_k.eval(np.array([0.5]))[0]:.6f}")
    print(f"  κ* P₂(0.5): {P2_s.eval(np.array([0.5]))[0]:.6f}")

    # Compute L2 norms
    u_nodes, u_weights = np.polynomial.legendre.leggauss(60)
    u_nodes = 0.5 * (u_nodes + 1)
    u_weights = 0.5 * u_weights

    norm_k = np.sqrt(np.sum(u_weights * P2_k.eval(u_nodes)**2))
    norm_s = np.sqrt(np.sum(u_weights * P2_s.eval(u_nodes)**2))

    print(f"\n||P₂||_L²[0,1]:")
    print(f"  κ:  {norm_k:.6f}")
    print(f"  κ*: {norm_s:.6f}")
    print(f"  Ratio: {norm_k / norm_s:.4f}")

    # Compute weighted norms with (1-u)^4 (for (2,2) pair)
    weight = (1 - u_nodes)**4
    wnorm_k = np.sqrt(np.sum(u_weights * weight * P2_k.eval(u_nodes)**2))
    wnorm_s = np.sqrt(np.sum(u_weights * weight * P2_s.eval(u_nodes)**2))

    print(f"\n||P₂||_L²[0,1] with (1-u)⁴ weight:")
    print(f"  κ:  {wnorm_k:.6f}")
    print(f"  κ*: {wnorm_s:.6f}")
    print(f"  Ratio: {wnorm_k / wnorm_s:.4f}")

    # Compare Q integrals
    R_kappa = 1.3036
    R_kappa_star = 1.1167

    t_nodes, t_weights = np.polynomial.legendre.leggauss(60)
    t_nodes = 0.5 * (t_nodes + 1)
    t_weights = 0.5 * t_weights

    Q_int_k = np.sum(t_weights * Q_k.eval(t_nodes)**2 * np.exp(2*R_kappa*t_nodes))
    Q_int_s = np.sum(t_weights * Q_s.eval(t_nodes)**2 * np.exp(2*R_kappa_star*t_nodes))

    print(f"\n∫Q(t)²e^{{2Rt}}dt:")
    print(f"  κ (R=1.3036):  {Q_int_k:.6f}")
    print(f"  κ* (R=1.1167): {Q_int_s:.6f}")
    print(f"  Ratio: {Q_int_k / Q_int_s:.4f}")


if __name__ == "__main__":
    compare_benchmarks()
    analyze_ratio_sources()
