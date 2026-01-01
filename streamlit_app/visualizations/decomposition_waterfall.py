"""
Decomposition waterfall chart visualization.

Shows the assembly: S12(+R) + m*S12(-R) + S34 = c -> kappa
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Optional
import math


def create_decomposition_waterfall(result: Dict) -> go.Figure:
    """
    Create a waterfall chart showing c assembly.

    Args:
        result: Dict with S12_plus, S12_minus, S34, m, c, kappa

    Returns:
        Plotly Figure object
    """
    S12_plus = result["S12_plus"]
    S12_minus = result["S12_minus"]
    S34 = result["S34"]
    m = result["m"]
    c = result["c"]
    kappa = result["kappa"]

    # Mirror contribution
    mirror_contrib = m * S12_minus

    # Create waterfall data
    labels = [
        "S12(+R)",
        "m × S12(-R)",
        "S34(+R)",
        "c (total)"
    ]

    values = [
        S12_plus,
        mirror_contrib,
        S34,
        0  # Total computed automatically
    ]

    measures = ["relative", "relative", "relative", "total"]

    # Colors
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    fig = go.Figure(go.Waterfall(
        name="c assembly",
        orientation="v",
        measure=measures,
        x=labels,
        textposition="outside",
        text=[f"{v:.4f}" for v in [S12_plus, mirror_contrib, S34, c]],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ca02c"}},
        decreasing={"marker": {"color": "#d62728"}},
        totals={"marker": {"color": "#9467bd"}},
    ))

    # Add annotations
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=f"m = exp(R) + (2K-1) = {m:.4f}",
        showarrow=False,
        font=dict(size=12),
    )

    fig.update_layout(
        title=f"c Assembly (kappa = {kappa:.6f})",
        showlegend=False,
        template="plotly_white",
        height=400,
        yaxis_title="Value",
    )

    return fig


def create_integral_breakdown(result: Dict) -> go.Figure:
    """
    Create bar chart showing integral components.

    Args:
        result: Dict with I1_plus, I2_plus, etc.

    Returns:
        Plotly Figure object
    """
    labels = ["I1(+R)", "I1(-R)", "I2(+R)", "I2(-R)", "I3(+R)", "I4(+R)"]
    values = [
        result["I1_plus"],
        result["I1_minus"],
        result["I2_plus"],
        result["I2_minus"],
        result["I3_plus"],
        result["I4_plus"],
    ]

    colors = ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a", "#d62728", "#ff9896"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Individual Integral Components",
        xaxis_title="Integral Term",
        yaxis_title="Value",
        template="plotly_white",
        height=350,
    )

    return fig


def render_decomposition(result: Optional[Dict]):
    """
    Render decomposition visualization in Streamlit.

    Args:
        result: Dict from full computation or None
    """
    if result is None:
        st.info("Click 'Compute Full Result' to see decomposition")
        return

    # Waterfall chart
    fig_waterfall = create_decomposition_waterfall(result)
    st.plotly_chart(fig_waterfall, width='stretch')

    # Integral breakdown
    st.markdown("**Integral Breakdown:**")
    fig_integrals = create_integral_breakdown(result)
    st.plotly_chart(fig_integrals, width='stretch')

    # Formula display
    st.markdown("**Assembly Formula:**")
    st.latex(r"c = S_{12}(+R) + m \times S_{12}(-R) + S_{34}(+R)")
    st.latex(r"\kappa = 1 - \frac{\log c}{R}")

    # Correction factors
    st.markdown("**Correction Factors:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("g_I1", f"{result['g_I1']:.6f}")
    with col2:
        st.metric("g_I2", f"{result['g_I2']:.6f}")
    with col3:
        st.metric("g_total", f"{result['g_total']:.6f}")
    with col4:
        st.metric("base", f"{result['base']:.4f}")
