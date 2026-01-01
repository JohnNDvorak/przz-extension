"""
Per-pair breakdown visualization.

Shows detailed I-term values for each of the 6 pairs (K=3):
(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional
import numpy as np


def create_per_pair_table(per_pair: Dict) -> pd.DataFrame:
    """
    Create a DataFrame for per-pair integral breakdown.

    Args:
        per_pair: Dict from compute_per_pair_breakdown

    Returns:
        DataFrame with per-pair integral values
    """
    if "error" in per_pair:
        return pd.DataFrame({"Error": [per_pair["error"]]})

    rows = []
    pair_order = ["11", "12", "13", "22", "23", "33"]

    for pair_key in pair_order:
        if pair_key not in per_pair:
            continue
        data = per_pair[pair_key]
        rows.append({
            "Pair": data["label"],
            "I1(+R)": data["I1_plus"],
            "I1(-R)": data["I1_minus"],
            "I2(+R)": data["I2_plus"],
            "I2(-R)": data["I2_minus"],
            "I3(+R)": data["I3"],
            "I4(+R)": data["I4"],
            "S12(+R)": data["S12_plus"],
            "S12(-R)": data["S12_minus"],
            "S34(+R)": data["S34"],
        })

    return pd.DataFrame(rows)


def create_i2_matrix(per_pair: Dict) -> pd.DataFrame:
    """
    Create I2 matrix visualization like in main_results.tex.

    Shows I2(+R) values in a symmetric matrix format.
    """
    if "error" in per_pair:
        return pd.DataFrame()

    # Create a 3x3 matrix for pairs
    matrix = np.zeros((3, 3))
    pair_order = ["11", "12", "13", "22", "23", "33"]

    for pair_key in pair_order:
        if pair_key not in per_pair:
            continue
        i = int(pair_key[0]) - 1
        j = int(pair_key[1]) - 1
        val = per_pair[pair_key]["I2_plus"]
        matrix[i, j] = val
        if i != j:
            matrix[j, i] = val  # Symmetric

    return pd.DataFrame(
        matrix,
        index=["P1", "P2", "P3"],
        columns=["P1", "P2", "P3"],
    )


def create_contribution_chart(per_pair: Dict) -> go.Figure:
    """
    Create a stacked bar chart showing contributions from each pair.
    """
    if "error" in per_pair:
        return go.Figure()

    pair_order = ["11", "12", "13", "22", "23", "33"]
    labels = []
    s12_plus = []
    s12_minus = []
    s34 = []

    for pair_key in pair_order:
        if pair_key not in per_pair:
            continue
        data = per_pair[pair_key]
        labels.append(data["label"])
        s12_plus.append(data["S12_plus"])
        s12_minus.append(data["S12_minus"])
        s34.append(data["S34"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="S12(+R)",
        x=labels,
        y=s12_plus,
        marker_color="steelblue",
    ))

    fig.add_trace(go.Bar(
        name="S12(-R)",
        x=labels,
        y=s12_minus,
        marker_color="lightblue",
    ))

    fig.add_trace(go.Bar(
        name="S34(+R)",
        x=labels,
        y=s34,
        marker_color="coral",
    ))

    fig.update_layout(
        barmode="group",
        title="Contributions by Pair",
        xaxis_title="Pair (ℓ₁, ℓ₂)",
        yaxis_title="Value",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_destructive_interference_chart(per_pair: Dict, m: float) -> go.Figure:
    """
    Visualize constructive vs destructive interference between pairs.

    Some pairs contribute positively, others negatively to the final c.
    """
    if "error" in per_pair:
        return go.Figure()

    pair_order = ["11", "12", "13", "22", "23", "33"]
    labels = []
    contributions = []

    for pair_key in pair_order:
        if pair_key not in per_pair:
            continue
        data = per_pair[pair_key]
        # Total contribution: S12(+R) + m * S12(-R) + S34(+R)
        total = data["S12_plus"] + m * data["S12_minus"] + data["S34"]
        labels.append(data["label"])
        contributions.append(total)

    # Color based on sign
    colors = ["green" if c > 0 else "red" for c in contributions]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=contributions,
        marker_color=colors,
        text=[f"{c:.4f}" for c in contributions],
        textposition="outside",
    ))

    fig.update_layout(
        title="Per-Pair Contribution to c (with mirror multiplier)",
        xaxis_title="Pair (ℓ₁, ℓ₂)",
        yaxis_title="Contribution",
        height=350,
    )

    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig


def create_i2_heatmap(per_pair: Dict) -> go.Figure:
    """
    Create a heatmap of I2(+R) values in matrix form.
    """
    if "error" in per_pair:
        return go.Figure()

    matrix = create_i2_matrix(per_pair)
    if matrix.empty:
        return go.Figure()

    fig = px.imshow(
        matrix,
        labels=dict(x="P index", y="P index", color="I2(+R)"),
        x=["P1", "P2", "P3"],
        y=["P1", "P2", "P3"],
        color_continuous_scale="Blues",
        text_auto=".4f",
    )

    fig.update_layout(
        title="I2(+R) Pair Matrix",
        height=350,
    )

    return fig


def render_per_pair_breakdown(result: Optional[Dict]):
    """
    Render the complete per-pair breakdown visualization.

    Args:
        result: Dict from full computation with per_pair key
    """
    if result is None:
        st.info("Click 'Compute Full Result' to see per-pair breakdown")
        return

    per_pair = result.get("per_pair")
    if per_pair is None or (isinstance(per_pair, dict) and len(per_pair) == 0):
        st.warning("Per-pair breakdown not available")
        return

    if "error" in per_pair:
        st.error(f"Error computing per-pair breakdown: {per_pair['error']}")
        return

    m = result.get("m", 8.68)  # Default mirror multiplier

    # Display as expandable sections
    st.markdown("### Per-Pair Integral Breakdown")
    st.markdown("""
    The PRZZ framework decomposes the main-term constant c into contributions from
    6 pairs for K=3: (1,1), (1,2), (1,3), (2,2), (2,3), (3,3).
    """)

    # Main table
    st.markdown("#### Integral Values by Pair")
    df = create_per_pair_table(per_pair)
    st.dataframe(
        df.style.format({
            col: "{:.6f}" for col in df.columns if col != "Pair"
        }),
        width='stretch',
        hide_index=True,
    )

    # Two-column layout for charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Contribution by Pair")
        fig1 = create_contribution_chart(per_pair)
        st.plotly_chart(fig1, width='stretch')

    with col2:
        st.markdown("#### I2 Matrix (Pair Correlations)")
        fig2 = create_i2_heatmap(per_pair)
        st.plotly_chart(fig2, width='stretch')

    # Destructive interference
    st.markdown("#### Destructive Interference Analysis")
    st.markdown(f"""
    With mirror multiplier m = {m:.4f}, some pairs contribute constructively (green)
    while others contribute destructively (red) to the final c value.
    """)
    fig3 = create_destructive_interference_chart(per_pair, m)
    st.plotly_chart(fig3, width='stretch')

    # Summary statistics
    st.markdown("#### Summary")
    total_s12_plus = sum(per_pair[k]["S12_plus"] for k in per_pair if k != "error")
    total_s12_minus = sum(per_pair[k]["S12_minus"] for k in per_pair if k != "error")
    total_s34 = sum(per_pair[k]["S34"] for k in per_pair if k != "error")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total S12(+R)", f"{total_s12_plus:.6f}")
    with col2:
        st.metric("Total S12(-R)", f"{total_s12_minus:.6f}")
    with col3:
        st.metric("Total S34(+R)", f"{total_s34:.6f}")

    # Formula reminder
    st.markdown("---")
    st.markdown("""
    **Assembly Formula:**
    ```
    c = S12(+R) + m × S12(-R) + S34(+R)
    κ = 1 - log(c) / R
    ```
    where m = exp(R) + (2K-1) for K pieces.
    """)
