"""
Coefficient amplitude visualization.

Compares coefficients against cap bounds across modes.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional

from ..utils.constants import (
    CONSTRAINT_BOUNDS, PRZZ_P1_TILDE, PRZZ_P2_TILDE, PRZZ_P3_TILDE, COLORS
)


def create_amplitude_chart(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    constraint_mode: str,
) -> go.Figure:
    """
    Create bar chart showing coefficient amplitudes vs bounds.

    Args:
        P1_coeffs: P1 tilde coefficients
        P2_coeffs: P2 tilde coefficients
        P3_coeffs: P3 tilde coefficients
        constraint_mode: Current constraint mode

    Returns:
        Plotly Figure object
    """
    bounds = CONSTRAINT_BOUNDS[constraint_mode]
    min_bound, max_bound = bounds

    # Build coefficient data
    labels = []
    values = []
    colors = []

    # P1 coefficients
    for i, v in enumerate(P1_coeffs):
        labels.append(f"P1.a{i}")
        values.append(v)
        if constraint_mode == "unbounded" or (min_bound <= v <= max_bound):
            colors.append(COLORS["P1"])
        else:
            colors.append(COLORS["exceeds_cap"])

    # P2 coefficients
    for i, v in enumerate(P2_coeffs):
        labels.append(f"P2.b{i}")
        values.append(v)
        if constraint_mode == "unbounded" or (min_bound <= v <= max_bound):
            colors.append(COLORS["P2"])
        else:
            colors.append(COLORS["exceeds_cap"])

    # P3 coefficients
    for i, v in enumerate(P3_coeffs):
        labels.append(f"P3.c{i}")
        values.append(v)
        if constraint_mode == "unbounded" or (min_bound <= v <= max_bound):
            colors.append(COLORS["P3"])
        else:
            colors.append(COLORS["exceeds_cap"])

    fig = go.Figure()

    # Coefficient bars
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        name="Current coefficients",
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))

    # Add bound lines (if not unbounded)
    if constraint_mode != "unbounded":
        fig.add_hline(
            y=max_bound, line_dash="dash", line_color="red",
            annotation_text=f"Upper bound ({max_bound})",
            annotation_position="top right"
        )
        fig.add_hline(
            y=min_bound, line_dash="dash", line_color="red",
            annotation_text=f"Lower bound ({min_bound})",
            annotation_position="bottom right"
        )

    # Zero line
    fig.add_hline(y=0, line_color="gray", line_width=1)

    fig.update_layout(
        title=f"Coefficient Amplitudes ({constraint_mode})",
        xaxis_title="Coefficient",
        yaxis_title="Value",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    return fig


def create_mode_comparison_chart(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
) -> go.Figure:
    """
    Create chart comparing current coefficients with PRZZ baseline.

    Args:
        P1_coeffs: Current P1 coefficients
        P2_coeffs: Current P2 coefficients
        P3_coeffs: Current P3 coefficients

    Returns:
        Plotly Figure object
    """
    labels = []
    current_vals = []
    przz_vals = []

    # P1 coefficients
    for i, (curr, przz) in enumerate(zip(P1_coeffs, PRZZ_P1_TILDE)):
        labels.append(f"P1.a{i}")
        current_vals.append(curr)
        przz_vals.append(przz)

    # P2 coefficients
    for i, (curr, przz) in enumerate(zip(P2_coeffs, PRZZ_P2_TILDE)):
        labels.append(f"P2.b{i}")
        current_vals.append(curr)
        przz_vals.append(przz)

    # P3 coefficients
    for i, (curr, przz) in enumerate(zip(P3_coeffs, PRZZ_P3_TILDE)):
        labels.append(f"P3.c{i}")
        current_vals.append(curr)
        przz_vals.append(przz)

    fig = go.Figure()

    # Current values
    fig.add_trace(go.Bar(
        x=labels,
        y=current_vals,
        name="Current",
        marker_color="#1f77b4",
    ))

    # PRZZ baseline
    fig.add_trace(go.Bar(
        x=labels,
        y=przz_vals,
        name="PRZZ baseline",
        marker_color="#aec7e8",
    ))

    fig.update_layout(
        title="Coefficient Comparison: Current vs PRZZ",
        xaxis_title="Coefficient",
        yaxis_title="Value",
        template="plotly_white",
        height=400,
        barmode="group",
    )

    return fig


def render_coefficient_amplitude(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    constraint_mode: str,
):
    """
    Render coefficient amplitude visualization in Streamlit.

    Args:
        P1_coeffs: P1 tilde coefficients
        P2_coeffs: P2 tilde coefficients
        P3_coeffs: P3 tilde coefficients
        constraint_mode: Current constraint mode
    """
    # Amplitude chart
    fig_amp = create_amplitude_chart(P1_coeffs, P2_coeffs, P3_coeffs, constraint_mode)
    st.plotly_chart(fig_amp, use_container_width=True)

    # Comparison with PRZZ
    st.markdown("**Comparison with PRZZ Baseline:**")
    fig_compare = create_mode_comparison_chart(P1_coeffs, P2_coeffs, P3_coeffs)
    st.plotly_chart(fig_compare, use_container_width=True)

    # Statistics
    st.markdown("**Coefficient Statistics:**")
    all_coeffs = P1_coeffs + P2_coeffs + P3_coeffs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Max", f"{max(all_coeffs):.4f}")
    with col2:
        st.metric("Min", f"{min(all_coeffs):.4f}")
    with col3:
        st.metric("Range", f"{max(all_coeffs) - min(all_coeffs):.4f}")
    with col4:
        st.metric("Max |coeff|", f"{max(abs(c) for c in all_coeffs):.4f}")
