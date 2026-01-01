"""
JSON export functionality.

Generates machine-readable JSON reports compatible with project schema.
"""

import json
from typing import Dict, List, Optional
from datetime import datetime


def generate_json_report(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    Q_coeffs: Dict[int, float],
    result: Dict,
    constraint_mode: str,
) -> Dict:
    """
    Generate JSON report for computation results.

    Args:
        P1_coeffs: P1 tilde coefficients
        P2_coeffs: P2 tilde coefficients
        P3_coeffs: P3 tilde coefficients
        Q_coeffs: Q coefficients
        result: Computation result dict
        constraint_mode: Constraint mode used

    Returns:
        Dict ready for JSON serialization
    """
    timestamp = datetime.now().isoformat()

    report = {
        "schema_version": 1,
        "metadata": {
            "source": "PRZZ Mollifier Explorer",
            "generated_at": timestamp,
            "constraint_mode": constraint_mode,
        },
        "configuration": {
            "K": result["K"],
            "d": 1,
            "theta": result["theta"],
            "theta_exact": "4/7",
            "R": result["R"],
        },
        "polynomials": {
            "P1": {
                "tilde_coeffs": list(P1_coeffs),
                "form": "constrained",
                "description": "P1(x) = x + x(1-x)*P_tilde(x)",
            },
            "P2": {
                "tilde_coeffs": list(P2_coeffs),
                "form": "monomial",
                "description": "P2(x) = x*P_tilde(x)",
            },
            "P3": {
                "tilde_coeffs": list(P3_coeffs),
                "form": "monomial",
                "description": "P3(x) = x*P_tilde(x)",
            },
            "Q": {
                "coeffs_in_basis": {str(k): v for k, v in Q_coeffs.items()},
                "form": "przz_basis",
                "description": "Q(x) in (1-2x)^k basis",
            },
        },
        "results": {
            "kappa": result["kappa"],
            "kappa_rigorous": result.get("kappa_rigorous"),
            "c": result["c"],
            "decomposition": {
                "S12_plus": result["S12_plus"],
                "S12_minus": result["S12_minus"],
                "S34": result["S34"],
                "m": result["m"],
            },
            "integrals": {
                "I1_plus": result["I1_plus"],
                "I1_minus": result["I1_minus"],
                "I2_plus": result["I2_plus"],
                "I2_minus": result["I2_minus"],
                "I3_plus": result["I3_plus"],
                "I4_plus": result["I4_plus"],
            },
            "corrections": {
                "g_I1": result["g_I1"],
                "g_I2": result["g_I2"],
                "g_total": result["g_total"],
                "base": result["base"],
            },
        },
        "formulas": {
            "kappa_from_c": "kappa = 1 - log(c)/R",
            "c_assembly": "c = S12(+R) + m * S12(-R) + S34(+R)",
            "m_formula": "m = g_total * (exp(R) + (2K-1))",
        },
    }

    # Add error bounds if available
    if result.get("error_bounds"):
        report["error_bounds"] = result["error_bounds"]

    return report


def export_to_json_string(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    Q_coeffs: Dict[int, float],
    result: Dict,
    constraint_mode: str,
    indent: int = 2,
) -> str:
    """
    Generate formatted JSON string.

    Args:
        P1_coeffs: P1 tilde coefficients
        P2_coeffs: P2 tilde coefficients
        P3_coeffs: P3 tilde coefficients
        Q_coeffs: Q coefficients
        result: Computation result dict
        constraint_mode: Constraint mode used
        indent: JSON indentation level

    Returns:
        Formatted JSON string
    """
    report = generate_json_report(
        P1_coeffs, P2_coeffs, P3_coeffs, Q_coeffs,
        result, constraint_mode
    )
    return json.dumps(report, indent=indent)


def generate_minimal_json(result: Dict) -> str:
    """
    Generate minimal JSON with just key results.

    Args:
        result: Computation result dict

    Returns:
        Minimal JSON string
    """
    minimal = {
        "kappa": result["kappa"],
        "c": result["c"],
        "R": result["R"],
        "theta": result["theta"],
        "K": result["K"],
        "S12_plus": result["S12_plus"],
        "S12_minus": result["S12_minus"],
        "S34": result["S34"],
        "m": result["m"],
    }
    return json.dumps(minimal, indent=2)
