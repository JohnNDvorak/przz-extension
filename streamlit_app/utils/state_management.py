"""
Session state management for the Streamlit application.

Provides a typed interface to Streamlit's session state.
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .constants import (
    PRZZ_P1_TILDE, PRZZ_P2_TILDE, PRZZ_P3_TILDE, PRZZ_Q_COEFFS,
    OPTIMIZED_P1_TILDE, OPTIMIZED_P2_TILDE, OPTIMIZED_P3_TILDE,
    R_OPTIMIZED_KAPPA, R_MIN, R_MAX, THETA, K
)


@dataclass
class ComputationResult:
    """Result of a full kappa computation."""
    kappa: float
    c: float
    S12_plus: float
    S12_minus: float
    S34: float
    m: float
    integrals: Dict[str, float] = field(default_factory=dict)
    per_pair: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error_bounds: Dict[str, float] = field(default_factory=dict)
    kappa_rigorous: Optional[float] = None


def initialize_state():
    """Initialize session state with OPTIMIZED defaults (kappa ~ 1.0)."""
    defaults = {
        # Polynomial coefficients - OPTIMIZED for c=1, kappa=1
        "P1_tilde": OPTIMIZED_P1_TILDE.copy(),
        "P2_tilde": OPTIMIZED_P2_TILDE.copy(),
        "P3_tilde": OPTIMIZED_P3_TILDE.copy(),
        "Q_coeffs": PRZZ_Q_COEFFS.copy(),  # Keep PRZZ Q for kappa mode

        # Configuration
        "constraint_mode": "cap2",  # Default to cap=2
        "input_mode": "sliders",    # "sliders" or "text"
        "R_value": R_OPTIMIZED_KAPPA,  # Default to rounded R (paper: R* = 1.149760...)
        "theta": THETA,
        "K": K,

        # Computation state
        "last_result": None,
        "computation_in_progress": False,

        # UI state
        "active_tab": "Polynomials",
        "show_advanced": False,

        # Quick mode result (for live updates)
        "quick_kappa": None,
        "quick_c": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_coefficients() -> Dict[str, Any]:
    """Get current polynomial coefficients from session state."""
    return {
        "P1_tilde": list(st.session_state.P1_tilde),
        "P2_tilde": list(st.session_state.P2_tilde),
        "P3_tilde": list(st.session_state.P3_tilde),
        "Q_coeffs": dict(st.session_state.Q_coeffs),
    }


def set_coefficients(P1_tilde: List[float], P2_tilde: List[float],
                     P3_tilde: List[float], Q_coeffs: Optional[Dict] = None):
    """Set polynomial coefficients in session state."""
    st.session_state.P1_tilde = list(P1_tilde)
    st.session_state.P2_tilde = list(P2_tilde)
    st.session_state.P3_tilde = list(P3_tilde)
    if Q_coeffs is not None:
        st.session_state.Q_coeffs = dict(Q_coeffs)


def sync_widget_state_from_values() -> None:
    """Sync widget keys to current coefficients/R to avoid stale widget overrides."""
    min_val, max_val = get_constraint_bounds()

    def _clamp(value: float) -> float:
        return max(min(value, max_val), min_val)

    def _sync_poly(poly_name: str, values: List[float], label_prefix: str) -> None:
        for i, val in enumerate(values):
            st.session_state[f"{poly_name}_{label_prefix}{i}"] = float(_clamp(val))
        st.session_state[f"{poly_name}_text"] = ", ".join(f"{v:.6f}" for v in values)

    _sync_poly("P1", list(st.session_state.P1_tilde), "a")
    _sync_poly("P2", list(st.session_state.P2_tilde), "b")
    _sync_poly("P3", list(st.session_state.P3_tilde), "c")

    current_r = float(st.session_state.get("R_value", R_OPTIMIZED_KAPPA))
    st.session_state["r_text_input_widget"] = str(current_r)
    st.session_state["r_slider_widget"] = float(max(min(current_r, R_MAX), R_MIN))


def reset_to_przz():
    """Reset all coefficients to PRZZ defaults (mode-aware)."""
    from .constants import (
        PRZZ_KAPPA_STAR_P1_TILDE, PRZZ_KAPPA_STAR_P2_TILDE,
        PRZZ_KAPPA_STAR_P3_TILDE, PRZZ_KAPPA_STAR_Q_COEFFS,
        R_PRZZ_KAPPA, R_PRZZ_KAPPA_STAR
    )

    mode = st.session_state.get("mode", "kappa")

    if mode == "kappa_star":
        st.session_state.P1_tilde = PRZZ_KAPPA_STAR_P1_TILDE.copy()
        st.session_state.P2_tilde = PRZZ_KAPPA_STAR_P2_TILDE.copy()
        st.session_state.P3_tilde = PRZZ_KAPPA_STAR_P3_TILDE.copy()
        st.session_state.Q_coeffs = PRZZ_KAPPA_STAR_Q_COEFFS.copy()
        st.session_state.R_value = R_PRZZ_KAPPA_STAR
    else:
        st.session_state.P1_tilde = PRZZ_P1_TILDE.copy()
        st.session_state.P2_tilde = PRZZ_P2_TILDE.copy()
        st.session_state.P3_tilde = PRZZ_P3_TILDE.copy()
        st.session_state.Q_coeffs = PRZZ_Q_COEFFS.copy()
        st.session_state.R_value = R_PRZZ_KAPPA

    st.session_state.last_result = None
    st.session_state.quick_kappa = None
    st.session_state.quick_c = None
    sync_widget_state_from_values()


def get_R() -> float:
    """Get current R value."""
    return st.session_state.get("R_value", 1.15)


def get_constraint_bounds():
    """Get current constraint bounds based on mode."""
    from .constants import CONSTRAINT_BOUNDS
    return CONSTRAINT_BOUNDS[st.session_state.constraint_mode]


def set_result(result: ComputationResult):
    """Store computation result in session state."""
    st.session_state.last_result = result


def get_result() -> Optional[ComputationResult]:
    """Get the last computation result."""
    return st.session_state.last_result


def coefficients_as_tuple() -> tuple:
    """Return coefficients as a hashable tuple for caching."""
    return (
        tuple(st.session_state.P1_tilde),
        tuple(st.session_state.P2_tilde),
        tuple(st.session_state.P3_tilde),
        tuple(sorted(st.session_state.Q_coeffs.items())),
        get_R(),
        st.session_state.theta,
    )
