"""
Constraint mode selector component.

Allows selection between cap=1, cap=2, and unbounded coefficient constraints.
"""

import streamlit as st
from ..utils.constants import CONSTRAINT_BOUNDS


def render_constraint_mode():
    """
    Render constraint mode radio selector.

    Updates session state with selected mode.
    """
    st.markdown("**Coefficient Bounds**")

    options = {
        "cap1": "Cap = 1 (bounds [-1, 1])",
        "cap2": "Cap = 2 (bounds [-2, 2])",
        "unbounded": "Unbounded (no limits)",
    }

    current = st.session_state.get("constraint_mode", "cap2")

    # Find index of current mode
    mode_keys = list(options.keys())
    current_index = mode_keys.index(current) if current in mode_keys else 1

    selected = st.radio(
        "Constraint mode",
        options=mode_keys,
        format_func=lambda x: options[x],
        index=current_index,
        key="constraint_radio",
        label_visibility="collapsed",
    )

    # Update session state if changed
    if selected != current:
        st.session_state.constraint_mode = selected
        # Clear cached results when constraints change
        st.session_state.last_result = None
        st.session_state.quick_kappa = None
        st.session_state.quick_c = None


def render_constraint_info():
    """Display information about the current constraint mode."""
    mode = st.session_state.get("constraint_mode", "cap2")
    bounds = CONSTRAINT_BOUNDS[mode]

    if mode == "cap1":
        st.info(
            "**Cap = 1**: Conservative bounds matching theoretical safe region. "
            f"All coefficients must be in [{bounds[0]}, {bounds[1]}]."
        )
    elif mode == "cap2":
        st.info(
            "**Cap = 2**: Extended bounds allowing larger coefficients. "
            f"All coefficients must be in [{bounds[0]}, {bounds[1]}]. "
            "This is the constraint used in the paper's main result."
        )
    else:
        st.warning(
            "**Unbounded**: No coefficient limits. Larger coefficients can yield "
            "higher kappa but may compromise rigor. Use with caution."
        )


def get_current_bounds():
    """Get the current constraint bounds as (min, max) tuple."""
    mode = st.session_state.get("constraint_mode", "cap2")
    return CONSTRAINT_BOUNDS[mode]
