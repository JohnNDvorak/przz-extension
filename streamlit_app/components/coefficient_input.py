"""
Coefficient input components with slider/text toggle.

Supports P1, P2, P3 polynomial coefficients with constraint enforcement.
"""

import streamlit as st
import numpy as np
from typing import List, Tuple
from ..utils.constants import (
    N_COEFFS_P1, N_COEFFS_P2, N_COEFFS_P3,
    PRZZ_P1_TILDE, PRZZ_P2_TILDE, PRZZ_P3_TILDE,
    SLIDER_STEP
)
from ..utils.state_management import get_constraint_bounds


def coefficient_sliders(
    poly_name: str,
    current_values: List[float],
    key_prefix: str
) -> List[float]:
    """
    Render coefficient sliders for a polynomial.

    Args:
        poly_name: "P1", "P2", or "P3"
        current_values: Current coefficient values
        key_prefix: Unique key prefix for Streamlit widgets

    Returns:
        Updated coefficient list
    """
    bounds = get_constraint_bounds()
    min_val, max_val = bounds

    new_values = []
    n_coeffs = len(current_values)

    # Coefficient labels
    if poly_name == "P1":
        labels = [f"a{i}" for i in range(n_coeffs)]
        help_texts = [
            "Coefficient of x(1-x)",
            "Coefficient of x(1-x)^2",
            "Coefficient of x(1-x)^3",
            "Coefficient of x(1-x)^4",
        ]
    elif poly_name == "P2":
        labels = [f"b{i}" for i in range(n_coeffs)]
        help_texts = [
            "Coefficient of x",
            "Coefficient of x^2",
            "Coefficient of x^3",
        ]
    else:  # P3
        labels = [f"c{i}" for i in range(n_coeffs)]
        help_texts = [
            "Coefficient of x",
            "Coefficient of x^2",
            "Coefficient of x^3",
        ]

    for i, (val, label) in enumerate(zip(current_values, labels)):
        new_val = st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=float(np.clip(val, min_val, max_val)),
            step=SLIDER_STEP,
            key=f"{key_prefix}_{label}",
            help=help_texts[i] if i < len(help_texts) else None,
        )
        new_values.append(new_val)

    return new_values


def extract_numbers_from_text(text: str) -> List[float]:
    """
    Extract coefficient numbers from text, ignoring JSON structure and labels.

    Handles formats like:
    - "0.123, -0.456, 0.789"
    - '["P1_tilde": [0.123, -0.456]]'
    - Pasted JSON with labels

    Args:
        text: Input text possibly containing JSON or other formatting

    Returns:
        List of extracted float values
    """
    import re

    # Strategy: Look for numbers that appear to be coefficient values
    # - Must have a decimal point OR be a negative integer OR follow certain patterns
    # - Exclude things like "p1_0" indices

    values = []

    # First, try to find array-like patterns: [num, num, num] or num, num, num
    # Match floats with decimals (most reliable for coefficients)
    float_pattern = r'(?<![a-zA-Z_])(-?\d+\.\d+(?:[eE][+-]?\d+)?)'
    float_matches = re.findall(float_pattern, text)

    for match in float_matches:
        try:
            val = float(match)
            values.append(val)
        except ValueError:
            continue

    # If we found decimal numbers, return those
    if values:
        return values

    # Fallback: try comma-separated integers/floats (simple format)
    simple_pattern = r'(-?\d+\.?\d*)'
    for part in text.split(','):
        part = part.strip()
        match = re.match(simple_pattern, part)
        if match:
            try:
                val = float(match.group(1))
                values.append(val)
            except ValueError:
                continue

    return values


def coefficient_text_input(
    poly_name: str,
    current_values: List[float],
    key_prefix: str
) -> Tuple[List[float], bool]:
    """
    Render text input for coefficient entry.

    Accepts pasted JSON or comma-separated values - extracts numbers automatically.

    Args:
        poly_name: "P1", "P2", or "P3"
        current_values: Current coefficient values
        key_prefix: Unique key prefix for Streamlit widgets

    Returns:
        Tuple of (updated coefficient list, is_valid)
    """
    bounds = get_constraint_bounds()
    min_val, max_val = bounds
    constraint_mode = st.session_state.constraint_mode

    # Format current values as comma-separated string
    current_str = ", ".join(f"{v:.6f}" for v in current_values)

    # Help text
    if poly_name == "P1":
        help_text = "Paste JSON or enter 4 coefficients - numbers are extracted automatically"
    else:
        help_text = f"Paste JSON or enter 3 coefficients - numbers are extracted automatically"

    # Text area
    input_str = st.text_area(
        f"{poly_name} coefficients",
        value=current_str,
        key=f"{key_prefix}_text",
        help=help_text,
        height=68,
    )

    # Parse and validate - extract numbers from any format
    try:
        values = extract_numbers_from_text(input_str)
        expected_len = len(current_values)

        if len(values) < expected_len:
            st.error(f"Expected {expected_len} coefficients, found {len(values)}")
            return current_values, False

        # Take only the expected number of coefficients
        if len(values) > expected_len:
            values = values[:expected_len]
            st.info(f"Using first {expected_len} numbers: {', '.join(f'{v:.6f}' for v in values)}")

        # Check bounds (only warn, don't reject for unbounded)
        if constraint_mode != "unbounded":
            violations = []
            for i, v in enumerate(values):
                if v < min_val or v > max_val:
                    violations.append(f"coeff[{i}]={v:.4f}")
            if violations:
                st.warning(f"Bounds [{min_val}, {max_val}] exceeded: {', '.join(violations)}")

        return values, True

    except Exception as e:
        st.error(f"Error parsing input: {e}")
        return current_values, False


def render_polynomial_input(poly_name: str, key_prefix: str = None) -> List[float]:
    """
    Render complete coefficient input panel for a polynomial.

    Handles both slider and text input modes based on session state.
    Supports variable coefficient counts for different modes (κ vs κ*).

    Args:
        poly_name: "P1", "P2", or "P3"
        key_prefix: Optional key prefix (defaults to poly_name)

    Returns:
        Updated coefficient list
    """
    from ..utils.constants import (
        PRZZ_KAPPA_STAR_P1_TILDE, PRZZ_KAPPA_STAR_P2_TILDE, PRZZ_KAPPA_STAR_P3_TILDE
    )

    if key_prefix is None:
        key_prefix = poly_name

    # Get current values from session state
    state_key = f"{poly_name}_tilde"
    current_values = list(st.session_state.get(state_key, []))

    # Get mode-appropriate defaults if values are empty
    mode = st.session_state.get("mode", "kappa")

    if len(current_values) == 0:
        # No values set, use mode-appropriate defaults
        if mode == "kappa_star":
            if poly_name == "P1":
                current_values = PRZZ_KAPPA_STAR_P1_TILDE.copy()
            elif poly_name == "P2":
                current_values = PRZZ_KAPPA_STAR_P2_TILDE.copy()
            else:
                current_values = PRZZ_KAPPA_STAR_P3_TILDE.copy()
        else:
            if poly_name == "P1":
                current_values = PRZZ_P1_TILDE.copy()
            elif poly_name == "P2":
                current_values = PRZZ_P2_TILDE.copy()
            else:
                current_values = PRZZ_P3_TILDE.copy()

    # Note: We don't enforce a fixed length - κ* mode has 2 coefficients for P2/P3,
    # while κ mode has 3. The loaded configuration determines the length.

    # Render based on input mode
    input_mode = st.session_state.get("input_mode", "sliders")

    if input_mode == "sliders":
        new_values = coefficient_sliders(poly_name, current_values, key_prefix)
    else:
        new_values, _ = coefficient_text_input(poly_name, current_values, key_prefix)

    # Update session state
    st.session_state[state_key] = new_values

    return new_values


def render_input_mode_toggle():
    """Render the toggle between slider and text input modes."""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sliders", width='stretch',
                     type="primary" if st.session_state.input_mode == "sliders" else "secondary",
                     key="btn_sliders"):
            st.session_state.input_mode = "sliders"
            st.rerun()
    with col2:
        if st.button("Text Entry", width='stretch',
                     type="primary" if st.session_state.input_mode == "text" else "secondary",
                     key="btn_text_entry"):
            st.session_state.input_mode = "text"
            st.rerun()


def render_all_coefficients():
    """Render coefficient inputs for all polynomials."""
    # Input mode toggle
    st.markdown("**Input Mode**")
    render_input_mode_toggle()
    st.divider()

    # P1 coefficients
    st.markdown("**P1 Coefficients** (tilde basis)")
    st.caption("P1(x) = x + x(1-x) * P_tilde(1-x)")
    render_polynomial_input("P1")

    st.divider()

    # P2 coefficients
    st.markdown("**P2 Coefficients**")
    st.caption("P2(x) = b0*x + b1*x^2 + b2*x^3")
    render_polynomial_input("P2")

    st.divider()

    # P3 coefficients
    st.markdown("**P3 Coefficients**")
    st.caption("P3(x) = c0*x + c1*x^2 + c2*x^3")
    render_polynomial_input("P3")
