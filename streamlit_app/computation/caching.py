"""
Streamlit caching utilities for expensive computations.

Uses @st.cache_data for coefficient-based caching.
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional
import json


def coefficients_to_hash_key(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    Q_coeffs: Dict[int, float],
    R: float,
    theta: float,
) -> str:
    """
    Convert coefficients to a hashable string key.

    Args:
        P1_coeffs: P1 tilde coefficients
        P2_coeffs: P2 tilde coefficients
        P3_coeffs: P3 tilde coefficients
        Q_coeffs: Q coefficients dict
        R: Shift parameter
        theta: Mollifier exponent

    Returns:
        JSON string representation for hashing
    """
    data = {
        "P1": [round(c, 10) for c in P1_coeffs],
        "P2": [round(c, 10) for c in P2_coeffs],
        "P3": [round(c, 10) for c in P3_coeffs],
        "Q": {str(k): round(v, 10) for k, v in Q_coeffs.items()},
        "R": round(R, 6),
        "theta": round(theta, 10),
    }
    return json.dumps(data, sort_keys=True)


@st.cache_data(ttl=3600)
def cached_quick_kappa(
    P1_tuple: Tuple[float, ...],
    P2_tuple: Tuple[float, ...],
    P3_tuple: Tuple[float, ...],
    Q_json: str,
    R: float,
    theta: float,
    K: int,
    n_quad: int = 40,
) -> Dict:
    """
    Cached quick κ computation.

    Args:
        P1_tuple: P1 tilde coefficients as tuple
        P2_tuple: P2 tilde coefficients as tuple
        P3_tuple: P3 tilde coefficients as tuple
        Q_json: Q coefficients as JSON string
        R: Shift parameter
        theta: Mollifier exponent
        K: Number of pieces
        n_quad: Quadrature points (default 40 for fast updates)

    Returns:
        Dict with kappa and c values
    """
    from .engine_wrapper import compute_quick_kappa
    import json

    Q_coeffs = {int(k): v for k, v in json.loads(Q_json).items()}

    result = compute_quick_kappa(
        P1_coeffs=list(P1_tuple),
        P2_coeffs=list(P2_tuple),
        P3_coeffs=list(P3_tuple),
        Q_coeffs=Q_coeffs,
        R=R,
        theta=theta,
        K=K,
        n_quad=n_quad,
    )

    return {
        "kappa": result.kappa,
        "c": result.c,
        "valid": result.valid,
        "message": result.message,
    }


@st.cache_data(ttl=3600)
def cached_full_kappa(
    P1_tuple: Tuple[float, ...],
    P2_tuple: Tuple[float, ...],
    P3_tuple: Tuple[float, ...],
    Q_json: str,
    R: float,
    theta: float,
    K: int,
    n_quad: int,
) -> Dict:
    """
    Cached full κ computation.

    Args:
        P1_tuple: P1 tilde coefficients as tuple
        P2_tuple: P2 tilde coefficients as tuple
        P3_tuple: P3 tilde coefficients as tuple
        Q_json: Q coefficients as JSON string
        R: Shift parameter
        theta: Mollifier exponent
        K: Number of pieces
        n_quad: Quadrature points

    Returns:
        Dict with complete result
    """
    from .engine_wrapper import compute_full_result
    from dataclasses import asdict
    import json

    Q_coeffs = {int(k): v for k, v in json.loads(Q_json).items()}

    result = compute_full_result(
        P1_coeffs=list(P1_tuple),
        P2_coeffs=list(P2_tuple),
        P3_coeffs=list(P3_tuple),
        Q_coeffs=Q_coeffs,
        R=R,
        theta=theta,
        K=K,
        n_quad=n_quad,
        compute_errors=True,
        compute_per_pair=True,
    )

    # Convert to dict for caching
    return {
        "kappa": result.kappa,
        "c": result.c,
        "R": result.R,
        "theta": result.theta,
        "K": result.K,
        "S12_plus": result.S12_plus,
        "S12_minus": result.S12_minus,
        "S34": result.S34,
        "m": result.m,
        "I1_plus": result.I1_plus,
        "I1_minus": result.I1_minus,
        "I2_plus": result.I2_plus,
        "I2_minus": result.I2_minus,
        "I3_plus": result.I3_plus,
        "I4_plus": result.I4_plus,
        "g_I1": result.g_I1,
        "g_I2": result.g_I2,
        "g_total": result.g_total,
        "base": result.base,
        "error_bounds": result.error_bounds,
        "kappa_rigorous": result.kappa_rigorous,
        "per_pair": result.per_pair,
    }


def invalidate_cache():
    """Clear all cached computations."""
    st.cache_data.clear()
