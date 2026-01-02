"""
LaTeX export functionality.

Generates paper-ready LaTeX tables for results.
"""

from typing import Dict, List, Optional
from datetime import datetime


def generate_coefficients_table(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    Q_coeffs: Dict[int, float],
) -> str:
    """
    Generate LaTeX table for polynomial coefficients.

    Args:
        P1_coeffs: P1 tilde coefficients
        P2_coeffs: P2 tilde coefficients
        P3_coeffs: P3 tilde coefficients
        Q_coeffs: Q coefficients

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mollifier Polynomial Coefficients}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Polynomial & $c_0$ & $c_1$ & $c_2$ & $c_3$ \\",
        r"\midrule",
    ]

    # P1 row
    p1_vals = " & ".join(f"{c:.6f}" for c in P1_coeffs)
    if len(P1_coeffs) < 4:
        p1_vals += " & --" * (4 - len(P1_coeffs))
    lines.append(f"$P_1$ (tilde) & {p1_vals} \\\\")

    # P2 row
    p2_vals = " & ".join(f"{c:.6f}" for c in P2_coeffs)
    if len(P2_coeffs) < 4:
        p2_vals += " & --" * (4 - len(P2_coeffs))
    lines.append(f"$P_2$ (tilde) & {p2_vals} \\\\")

    # P3 row
    p3_vals = " & ".join(f"{c:.6f}" for c in P3_coeffs)
    if len(P3_coeffs) < 4:
        p3_vals += " & --" * (4 - len(P3_coeffs))
    lines.append(f"$P_3$ (tilde) & {p3_vals} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:coefficients}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_decomposition_table(result: Dict) -> str:
    """
    Generate LaTeX table for decomposition breakdown.

    Args:
        result: Computation result dict

    Returns:
        LaTeX table string
    """
    # Safe access with defaults
    S12_plus = result.get('S12_plus', 0)
    S12_minus = result.get('S12_minus', 0)
    S34 = result.get('S34', 0)
    m = result.get('m', 0)
    c = result.get('c', 0)
    kappa = result.get('kappa', 0)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Main-Term Decomposition}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Component & Value \\",
        r"\midrule",
        f"$S_{{12}}(+R)$ & {S12_plus:.6f} \\\\",
        f"$S_{{12}}(-R)$ & {S12_minus:.6f} \\\\",
        f"$S_{{34}}(+R)$ & {S34:.6f} \\\\",
        f"$m$ & {m:.4f} \\\\",
        r"\midrule",
        f"$c$ & {c:.6f} \\\\",
        f"$\\kappa$ & {kappa:.6f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:decomposition}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def generate_error_table(error_bounds: Dict) -> str:
    """
    Generate LaTeX table for error bounds.

    Args:
        error_bounds: Error bounds dict

    Returns:
        LaTeX table string
    """
    if "error" in error_bounds:
        return f"% Error computing bounds: {error_bounds['error']}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Error Bound Sources}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Source & Value & Order \\",
        r"\midrule",
        f"$C_{{\\text{{contour}}}}$ & {error_bounds.get('C_contour', 0):.6e} & $O(T/L)$ \\\\",
        f"$C_{{\\text{{Taylor}}}}$ & {error_bounds.get('C_Taylor', 0):.6e} & $O(T/L)$ \\\\",
        f"$C_{{I_5}}$ & {error_bounds.get('C_I5', 0):.6e} & $O(T/L^2)$ \\\\",
        f"$C_{{\\text{{EM}}}}$ & {error_bounds.get('C_EM', 0):.6e} & $O(T/L)$ \\\\",
        r"\midrule",
        f"Total (per $L$) & {error_bounds.get('total_per_L', 0):.6e} & $O(T/L)$ \\\\",
        f"Total (per $L^2$) & {error_bounds.get('total_per_L2', 0):.6e} & $O(T/L^2)$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:errors}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def generate_full_report(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    Q_coeffs: Dict[int, float],
    result: Dict,
    constraint_mode: str,
) -> str:
    """
    Generate complete LaTeX report.

    Args:
        P1_coeffs: P1 tilde coefficients
        P2_coeffs: P2 tilde coefficients
        P3_coeffs: P3 tilde coefficients
        Q_coeffs: Q coefficients
        result: Computation result dict
        constraint_mode: Constraint mode used

    Returns:
        Complete LaTeX document string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Safe access to configuration values
    R = result.get('R', 0)
    theta = result.get('theta', 0)
    K = result.get('K', 3)

    lines = [
        r"% PRZZ Mollifier Explorer - Computation Report",
        f"% Generated: {timestamp}",
        f"% Constraint mode: {constraint_mode}",
        "",
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        "",
        r"\begin{document}",
        "",
        r"\section*{Mollifier Computation Report}",
        "",
        f"Configuration: $R = {R:.4f}$, $\\theta = {theta:.6f}$, $K = {K}$",
        "",
        generate_coefficients_table(P1_coeffs, P2_coeffs, P3_coeffs, Q_coeffs),
        "",
        generate_decomposition_table(result),
        "",
    ]

    if result.get("error_bounds"):
        lines.append(generate_error_table(result["error_bounds"]))
        lines.append("")

    # Assembly formula
    lines.extend([
        r"\section*{Assembly Formula}",
        r"\begin{equation}",
        r"c = S_{12}(+R) + m \times S_{12}(-R) + S_{34}(+R)",
        r"\end{equation}",
        "",
        r"\begin{equation}",
        r"\kappa = 1 - \frac{\log c}{R}",
        r"\end{equation}",
        "",
        r"\end{document}",
    ])

    return "\n".join(lines)
