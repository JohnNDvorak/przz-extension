"""
R parameter control component.

Allows locking R to PRZZ value or free adjustment via slider or text.
"""

import streamlit as st
from ..utils.constants import (
    R_PRZZ_KAPPA, R_PRZZ_KAPPA_STAR,
    R_OPTIMIZED_KAPPA, R_OPTIMIZED_KAPPA_STAR,
    R_MIN, R_MAX, R_STEP
)


def render_r_parameter():
    """
    Render R parameter control with text input and slider.

    Updates session state with R value.
    """
    st.markdown("**R Parameter**")

    # Initialize R_value if not set
    if "R_value" not in st.session_state:
        st.session_state.R_value = R_OPTIMIZED_KAPPA  # Default to optimized κ value

    current_R = st.session_state.R_value

    # Sync the text input widget with R_value
    # This ensures button-set values appear in the text input
    if "r_text_input" not in st.session_state:
        st.session_state.r_text_input = str(current_R)
    elif abs(float(st.session_state.r_text_input or "0") - current_R) > 0.0001:
        # R_value was changed externally (by button), update text input
        st.session_state.r_text_input = str(current_R)

    # Text input for exact value (always visible)
    col1, col2 = st.columns([2, 1])
    with col1:
        new_R_str = st.text_input(
            "R (exact)",
            key="r_text_input",
        )
    with col2:
        st.caption("")  # Spacer
        st.caption(f"Range: {R_MIN}-{R_MAX}")

    # Parse text input and update R_value
    try:
        new_R = float(new_R_str) if new_R_str else current_R
        if R_MIN <= new_R <= R_MAX:
            if abs(new_R - current_R) > 0.0001:
                st.session_state.R_value = new_R
        else:
            st.warning(f"R must be between {R_MIN} and {R_MAX}")
    except ValueError:
        st.error("Invalid number")

    # Slider for quick adjustment
    slider_R = st.slider(
        "R (slider)",
        min_value=R_MIN,
        max_value=R_MAX,
        value=float(st.session_state.R_value),
        step=0.01,
        key="r_slider",
        format="%.2f",
    )
    # Update from slider if it changed
    if abs(slider_R - st.session_state.R_value) > 0.001:
        st.session_state.R_value = slider_R
        st.session_state.r_text_input = str(slider_R)
        st.rerun()

    # Mode-aware quick preset buttons
    mode = st.session_state.get("mode", "kappa")
    st.caption("Presets:")

    if mode == "kappa_star":
        # κ* mode presets
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("1.08 (Best)", key="r_best_ks", type="primary"):
                st.session_state.R_value = R_OPTIMIZED_KAPPA_STAR
                st.session_state.r_text_input = str(R_OPTIMIZED_KAPPA_STAR)
                st.rerun()
        with c2:
            if st.button("1.1167 (PRZZ)", key="r_przz_ks"):
                st.session_state.R_value = R_PRZZ_KAPPA_STAR
                st.session_state.r_text_input = str(R_PRZZ_KAPPA_STAR)
                st.rerun()
        with c3:
            if st.button("0.85", key="r_085"):
                st.session_state.R_value = 0.85
                st.session_state.r_text_input = "0.85"
                st.rerun()
    else:
        # κ mode presets
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("1.15 (Best)", key="r_best_k", type="primary"):
                st.session_state.R_value = R_OPTIMIZED_KAPPA
                st.session_state.r_text_input = str(R_OPTIMIZED_KAPPA)
                st.rerun()
        with c2:
            if st.button("1.3036 (PRZZ)", key="r_przz_k"):
                st.session_state.R_value = R_PRZZ_KAPPA
                st.session_state.r_text_input = str(R_PRZZ_KAPPA)
                st.rerun()
        with c3:
            if st.button("0.85", key="r_085_k"):
                st.session_state.R_value = 0.85
                st.session_state.r_text_input = "0.85"
                st.rerun()

    return st.session_state.R_value


def render_r_info():
    """Display information about R parameter."""
    R = st.session_state.get("R_value", R_PRZZ)

    import math
    c_przz = 2.137
    sensitivity = -1 / (R * c_przz)
    st.caption(
        f"At R = {R:.4f}, sensitivity = {sensitivity:.3f} "
        "(kappa change per unit c change)"
    )


def get_current_R():
    """Get the current R value."""
    return st.session_state.get("R_value", 1.15)
