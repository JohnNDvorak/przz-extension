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
            "K": result.get("K", 3),
            "d": 1,
            "theta": result.get("theta", 4/7),
            "theta_exact": "4/7",
            "R": result.get("R", 0),
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
            "kappa": result.get("kappa", 0),
            "kappa_rigorous": result.get("kappa_rigorous"),
            "c": result.get("c", 0),
            "decomposition": {
                "S12_plus": result.get("S12_plus", 0),
                "S12_minus": result.get("S12_minus", 0),
                "S34": result.get("S34", 0),
                "m": result.get("m", 0),
            },
            "integrals": {
                "I1_plus": result.get("I1_plus", 0),
                "I1_minus": result.get("I1_minus", 0),
                "I2_plus": result.get("I2_plus", 0),
                "I2_minus": result.get("I2_minus", 0),
                "I3_plus": result.get("I3_plus", 0),
                "I4_plus": result.get("I4_plus", 0),
            },
            "corrections": {
                "g_I1": result.get("g_I1", 1.0),
                "g_I2": result.get("g_I2", 1.0),
                "g_total": result.get("g_total", 1.0),
                "base": result.get("base", 0),
            },
        },
        "formulas": {
            "kappa_from_c": "kappa >= 1 - max(log(c), 0)/R",
            "c_assembly": "c = S12(+R) + m * S12(-R) + S34(+R)",
            "m_formula": "m = g_total * (exp(R) + (2K-1))",
        },
    }

    # Add error bounds if available
    eb = result.get("error_bounds")
    if eb and isinstance(eb, dict) and "error" not in eb:
        report["error_bounds"] = eb

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
        "kappa": result.get("kappa", 0),
        "c": result.get("c", 0),
        "R": result.get("R", 0),
        "theta": result.get("theta", 4/7),
        "K": result.get("K", 3),
        "S12_plus": result.get("S12_plus", 0),
        "S12_minus": result.get("S12_minus", 0),
        "S34": result.get("S34", 0),
        "m": result.get("m", 0),
    }
    return json.dumps(minimal, indent=2)
