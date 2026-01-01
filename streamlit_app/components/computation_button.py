"""
Full calculation button component.

Triggers complete computation with progress display.
"""

import streamlit as st
from typing import Optional, Dict
import json

from ..computation.full_calculation import run_full_calculation, display_full_result
from ..utils.state_management import get_coefficients, get_R


def render_compute_button() -> Optional[Dict]:
    """
    Render the full computation button.

    Returns:
        Computation result dict if button clicked, None otherwise
    """
    if st.button(
        "Compute Full Result",
        type="primary",
        width='stretch',
        help="Run complete computation with error bounds",
        key="btn_compute_full"
    ):
        coeffs = get_coefficients()
        R = get_R()

        with st.spinner("Running full computation..."):
            result = run_full_calculation(
                P1_coeffs=coeffs["P1_tilde"],
                P2_coeffs=coeffs["P2_tilde"],
                P3_coeffs=coeffs["P3_tilde"],
                Q_coeffs=coeffs["Q_coeffs"],
                R=R,
                theta=st.session_state.theta,
                K=st.session_state.K,
                n_quad=60,
            )

        st.session_state.last_result = result
        return result

    return st.session_state.get("last_result")


def render_reset_button():
    """Render reset to PRZZ defaults button."""
    if st.button(
        "Reset to PRZZ",
        width='stretch',
        help="Reset all coefficients to PRZZ baseline",
        key="btn_reset_przz"
    ):
        from ..utils.state_management import reset_to_przz
        reset_to_przz()
        st.rerun()
