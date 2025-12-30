# -*- coding: utf-8 -*-
"""
Tail-Share Diagnostic for Quadrature Integrand

Checks whether the quadrature integrand is dominated by a few extreme points.
If the top 0.1% of quadrature points contribute > 10% of the total integral,
this indicates sensitivity to numerical outliers.

This addresses the "second-moment outlier sensitivity" concern: whether
rare large values could dominate the integral and inflate c.

References:
- Plan Phase 64: Tail-share diagnostic requirement
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

from src.quadrature import gauss_legendre_01


@dataclass
class TailShareResult:
    """Result of tail-share diagnostic."""
    total_integral: float
    n_points: int

    # Per-percentile results
    percentile_shares: Dict[float, Dict]

    # Flags
    is_acceptable: bool  # True if top 0.1% contributes < 10%
    max_point_share: float  # Share of largest single point

    def __repr__(self):
        return (f"TailShareResult(n={self.n_points}, "
                f"top_0.1%={self.percentile_shares[99.9]['share']:.1f}%, "
                f"max_point={self.max_point_share*100:.2f}%)")


def compute_tail_share_2d(
    integrand_func,
    n_quad: int = 80,
    percentile_thresholds: List[float] = [99.9, 99.0, 95.0, 90.0]
) -> TailShareResult:
    """
    Compute tail-share diagnostic for a 2D quadrature integrand.

    Args:
        integrand_func: Function (u, t) -> value that computes the integrand
        n_quad: Number of quadrature points in each dimension
        percentile_thresholds: Percentile thresholds to analyze

    Returns:
        TailShareResult with share analysis
    """
    # Get quadrature nodes and weights for [0,1]
    nodes, weights = gauss_legendre_01(n_quad)

    # Compute all weighted contributions
    contributions = []
    total = 0.0

    for i, (u, wu) in enumerate(zip(nodes, weights)):
        for j, (t, wt) in enumerate(zip(nodes, weights)):
            try:
                value = integrand_func(u, t)
                weighted = wu * wt * value
                contributions.append({
                    'u': u,
                    't': t,
                    'raw': value,
                    'weighted': weighted,
                    'abs_weighted': abs(weighted)
                })
                total += weighted
            except Exception as e:
                # Skip problematic points
                pass

    n_points = len(contributions)

    if n_points == 0:
        return TailShareResult(
            total_integral=0.0,
            n_points=0,
            percentile_shares={},
            is_acceptable=False,
            max_point_share=0.0
        )

    # Sort by absolute weighted contribution
    sorted_contribs = sorted(contributions,
                            key=lambda x: x['abs_weighted'],
                            reverse=True)

    # Compute max point share
    max_point_share = sorted_contribs[0]['abs_weighted'] / abs(total) if total != 0 else 0

    # Compute percentile shares
    percentile_shares = {}
    for pct in percentile_thresholds:
        n_top = max(1, int(n_points * (100 - pct) / 100))
        top_contribs = sorted_contribs[:n_top]
        top_sum = sum(c['abs_weighted'] for c in top_contribs)
        share = top_sum / abs(total) if total != 0 else 0

        percentile_shares[pct] = {
            'n_points': n_top,
            'fraction': (100 - pct) / 100,
            'share': share * 100,  # As percentage
            'top_values': [c['abs_weighted'] for c in top_contribs[:5]]  # Top 5 values
        }

    # Check acceptability: top 0.1% should contribute < 10%
    is_acceptable = percentile_shares[99.9]['share'] < 10.0

    return TailShareResult(
        total_integral=total,
        n_points=n_points,
        percentile_shares=percentile_shares,
        is_acceptable=is_acceptable,
        max_point_share=max_point_share
    )


def run_tail_share_diagnostic(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    theta: float = 4/7,
    R: float = 1.3036,
    n_quad: int = 80
) -> Dict:
    """
    Run tail-share diagnostic on the main integral integrand.

    This creates a simplified integrand that captures the essential
    structure of the I2 integration (the dominant component).

    Args:
        P1_coeffs, P2_coeffs, P3_coeffs: Tilde basis coefficients
        theta, R: PRZZ parameters
        n_quad: Quadrature points

    Returns:
        Dictionary with diagnostic results
    """
    P1 = np.array(P1_coeffs)
    P2 = np.array(P2_coeffs)
    P3 = np.array(P3_coeffs)

    def eval_poly(coeffs, x):
        """Evaluate polynomial in tilde basis: sum c_k * x^{k+1}"""
        result = 0.0
        for k, c in enumerate(coeffs):
            result += c * x**(k+1)
        return result

    def integrand_11(u, t):
        """P1(u)^2 * exp(-2Rt) component"""
        return eval_poly(P1, u)**2 * np.exp(-2*R*t)

    def integrand_22(u, t):
        """P2(u)^2 * exp(-2Rt) component"""
        return eval_poly(P2, u)**2 * np.exp(-2*R*t)

    def integrand_33(u, t):
        """P3(u)^2 * exp(-2Rt) component"""
        return eval_poly(P3, u)**2 * np.exp(-2*R*t)

    def integrand_total(u, t):
        """Simplified total integrand (sum of all pairs)"""
        p1 = eval_poly(P1, u)
        p2 = eval_poly(P2, u)
        p3 = eval_poly(P3, u)
        exp_factor = np.exp(-2*R*t)

        return (p1**2 + p2**2 + p3**2 +
                2*p1*p2 + 2*p1*p3 + 2*p2*p3) * exp_factor

    results = {
        'pair_11': compute_tail_share_2d(integrand_11, n_quad),
        'pair_22': compute_tail_share_2d(integrand_22, n_quad),
        'pair_33': compute_tail_share_2d(integrand_33, n_quad),
        'total': compute_tail_share_2d(integrand_total, n_quad),
    }

    return results


def format_tail_share_report(results: Dict) -> str:
    """Format tail-share diagnostic results as markdown."""
    lines = [
        "# Tail-Share Diagnostic Report",
        "",
        "## Summary",
        "",
        "| Component | Top 0.1% Share | Top 1% Share | Max Point | Status |",
        "|-----------|----------------|--------------|-----------|--------|",
    ]

    for name, result in results.items():
        share_01 = result.percentile_shares[99.9]['share']
        share_1 = result.percentile_shares[99.0]['share']
        max_pct = result.max_point_share * 100
        status = "OK" if result.is_acceptable else "CHECK"
        lines.append(
            f"| {name} | {share_01:.2f}% | {share_1:.2f}% | {max_pct:.3f}% | {status} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Top 0.1% share < 10%: No outlier dominance",
        "- Top 1% share < 20%: Good numerical stability",
        "- Max point < 1%: Well-distributed contributions",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    # PRZZ baseline
    P1_przz = [0.261076, -1.071007, -0.236840, 0.260233]
    P2_przz = [1.048274, 1.319912, -0.940058]
    P3_przz = [0.522811, -0.686510, -0.049923]

    # Optimal
    P1_opt = [0.163919, -0.786613, -0.216214, 0.327516]
    P2_opt = [1.006479, -0.229290, -0.193641]
    P3_opt = [-1.333122, -2.409307, -0.150797]

    print("=" * 70)
    print("TAIL-SHARE DIAGNOSTIC")
    print("=" * 70)

    print("\n--- PRZZ Baseline ---")
    results_przz = run_tail_share_diagnostic(P1_przz, P2_przz, P3_przz)
    print(format_tail_share_report(results_przz))

    print("\n--- Optimal ---")
    results_opt = run_tail_share_diagnostic(P1_opt, P2_opt, P3_opt)
    print(format_tail_share_report(results_opt))
