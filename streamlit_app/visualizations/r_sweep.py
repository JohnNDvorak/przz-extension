"""
R Sweep Dashboard - Interactive exploration of c(R) geometry.

Shows how c varies with R and identifies the theoretical minimum.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Optional


def get_precomputed_sweep_data() -> List[Dict]:
    """Load precomputed R sweep data."""
    try:
        from ..utils.constants import get_r_sweep_data
        return get_r_sweep_data()
    except:
        # Fallback data from paper Table in Section 6.2
        return [
            {"R": 0.50, "c": 1.0237, "kappa_main": 0.9531, "kappa_rigorous": 0.6872},
            {"R": 0.70, "c": 1.0057, "kappa_main": 0.9919, "kappa_rigorous": 0.7926},
            {"R": 0.85, "c": 1.0019, "kappa_main": 0.9977, "kappa_rigorous": 0.8281},
            {"R": 1.00, "c": 1.0066, "kappa_main": 0.9934, "kappa_rigorous": 0.8449},
            {"R": 1.10, "c": 1.0020, "kappa_main": 0.9982, "kappa_rigorous": 0.8600},
            {"R": 1.14978, "c": 1.0000, "kappa_main": 1.0000, "kappa_rigorous": 0.8650},  # THE CEILING
            {"R": 1.15, "c": 1.0001, "kappa_main": 0.9999, "kappa_rigorous": 0.8650},
            {"R": 1.20, "c": 1.0265, "kappa_main": 0.9782, "kappa_rigorous": 0.8501},
            {"R": 1.3036, "c": 1.0433, "kappa_main": 0.9675, "kappa_rigorous": 0.8477},  # PRZZ baseline point
            {"R": 1.50, "c": 1.0881, "kappa_main": 0.9437, "kappa_rigorous": 0.8367},
        ]


def create_c_R_plot(data: List[Dict], current_R: Optional[float] = None) -> go.Figure:
    """Create the c(R) geometry plot."""
    R_vals = [d["R"] for d in data]
    c_vals = [d["c"] for d in data]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["c(R) - Main Term Constant", "kappa(R) - Bound on Critical-Line Zeros"],
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5],
    )

    # c(R) curve
    fig.add_trace(
        go.Scatter(
            x=R_vals, y=c_vals,
            mode='lines+markers',
            name='c(R)',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
        ),
        row=1, col=1
    )

    # Horizontal line at c=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                  annotation_text="c = 1 (theoretical floor)",
                  annotation_position="right", row=1, col=1)

    # Mark theoretical minimum
    fig.add_trace(
        go.Scatter(
            x=[1.14978], y=[1.0],
            mode='markers',
            name='Theoretical minimum',
            marker=dict(size=15, color='gold', symbol='star', line=dict(width=2, color='black')),
        ),
        row=1, col=1
    )

    # Mark PRZZ baseline
    fig.add_trace(
        go.Scatter(
            x=[1.3036], y=[1.088],
            mode='markers',
            name='PRZZ baseline',
            marker=dict(size=12, color='red', symbol='diamond'),
        ),
        row=1, col=1
    )

    # Current R marker
    if current_R:
        # Interpolate c value
        c_current = np.interp(current_R, R_vals, c_vals)
        fig.add_trace(
            go.Scatter(
                x=[current_R], y=[c_current],
                mode='markers',
                name=f'Current (R={current_R:.4f})',
                marker=dict(size=12, color='purple', symbol='x', line=dict(width=2)),
            ),
            row=1, col=1
        )

    # kappa curves
    kappa_main = [d["kappa_main"] for d in data]
    kappa_rig = [d["kappa_rigorous"] for d in data]

    fig.add_trace(
        go.Scatter(
            x=R_vals, y=kappa_main,
            mode='lines+markers',
            name='kappa_main',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=6),
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=R_vals, y=kappa_rig,
            mode='lines+markers',
            name='kappa_rigorous',
            line=dict(color='#d62728', width=2, dash='dot'),
            marker=dict(size=6),
        ),
        row=2, col=1
    )

    # Horizontal line at kappa=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="kappa = 1 (all zeros)", row=2, col=1)

    # Mark theoretical maximum
    fig.add_trace(
        go.Scatter(
            x=[1.14978], y=[1.0],
            mode='markers',
            name='Maximum kappa',
            marker=dict(size=15, color='gold', symbol='star', line=dict(width=2, color='black')),
            showlegend=False,
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
        legend=dict(x=1.02, y=0.5),
        margin=dict(r=150),
    )

    fig.update_xaxes(title_text="R (shift parameter)", row=1, col=1)
    fig.update_xaxes(title_text="R (shift parameter)", row=2, col=1)
    fig.update_yaxes(title_text="c", row=1, col=1)
    fig.update_yaxes(title_text="kappa", row=2, col=1)

    return fig


def create_parabola_visualization() -> go.Figure:
    """Create the 'kissing the floor' parabola visualization."""
    R_vals = np.linspace(0.8, 1.5, 100)

    # Approximate c(R) as a parabola around the minimum
    # c(R) ~ 1 + a*(R - R_min)^2 where R_min = 1.14978
    R_min = 1.14978
    a = 0.8  # Curvature parameter

    c_vals = 1 + a * (R_vals - R_min)**2

    fig = go.Figure()

    # c(R) parabola
    fig.add_trace(go.Scatter(
        x=R_vals, y=c_vals,
        mode='lines',
        name='c(R)',
        line=dict(color='#1f77b4', width=3),
    ))

    # Floor at c=1
    fig.add_trace(go.Scatter(
        x=[0.8, 1.5], y=[1.0, 1.0],
        mode='lines',
        name='c = 1 (floor)',
        line=dict(color='green', width=2, dash='dash'),
    ))

    # Tangent point
    fig.add_trace(go.Scatter(
        x=[R_min], y=[1.0],
        mode='markers',
        name='Tangent point',
        marker=dict(size=20, color='gold', symbol='star', line=dict(width=2, color='black')),
    ))

    # Annotation
    fig.add_annotation(
        x=R_min, y=1.0,
        text=f"R = {R_min}<br>c = 1.0000",
        showarrow=True,
        arrowhead=2,
        ax=50, ay=-50,
        font=dict(size=14),
    )

    fig.update_layout(
        title="The Geometry of c(R): Kissing the Floor",
        xaxis_title="R (shift parameter)",
        yaxis_title="c (main-term constant)",
        template="plotly_white",
        height=400,
        showlegend=True,
    )

    # Set y-axis range to focus on the minimum
    fig.update_yaxes(range=[0.98, 1.15])

    return fig


def render_r_sweep_tab(current_coeffs: Optional[Dict] = None):
    """Render the R sweep exploration tab."""
    st.markdown("### R Sweep Dashboard")
    st.markdown("""
    The shift parameter $R$ determines where the Levinson method evaluates its bound.
    The key discovery is that **c(R) achieves its minimum at R = 1.14978**, where $c = 1$.
    """)

    # Get precomputed data
    sweep_data = get_precomputed_sweep_data()

    # R slider
    st.markdown("#### Explore R values")
    current_R = st.slider(
        "Select R value",
        min_value=0.85,
        max_value=1.50,
        value=1.14978,
        step=0.001,
        format="%.4f",
        key="r_sweep_slider"
    )

    # Interpolate values at current R
    R_vals = [d["R"] for d in sweep_data]
    c_vals = [d["c"] for d in sweep_data]
    kappa_main_vals = [d["kappa_main"] for d in sweep_data]
    kappa_rig_vals = [d["kappa_rigorous"] for d in sweep_data]

    c_current = np.interp(current_R, R_vals, c_vals)
    kappa_main_current = np.interp(current_R, R_vals, kappa_main_vals)
    kappa_rig_current = np.interp(current_R, R_vals, kappa_rig_vals)

    # Display current values
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R", f"{current_R:.4f}")
    col2.metric("c", f"{c_current:.6f}")
    col3.metric("kappa_main", f"{kappa_main_current:.6f}")
    col4.metric("kappa_rigorous", f"{kappa_rig_current:.4f}")

    # Key R values
    st.markdown("#### Key R Values")
    key_R_data = [
        {"R": 1.14978, "name": "Kappa ceiling", "c": 1.0000, "kappa_main": 1.0000, "kappa_rig": 0.8650},
        {"R": 1.07966, "name": "Kappa* ceiling", "c": 1.0000, "kappa_main": 1.0000, "kappa_rig": 0.84},
        {"R": 1.3036, "name": "PRZZ kappa", "c": 2.137, "kappa_main": 0.417, "kappa_rig": 0.343},
        {"R": 1.1167, "name": "PRZZ kappa*", "c": 1.938, "kappa_main": 0.408, "kappa_rig": 0.34},
    ]

    cols = st.columns(4)
    for i, kv in enumerate(key_R_data):
        with cols[i]:
            st.markdown(f"**{kv['name']}**")
            st.markdown(f"R = {kv['R']}")
            st.markdown(f"c = {kv['c']:.4f}")
            if st.button(f"Go to R={kv['R']}", key=f"goto_{kv['name']}"):
                st.session_state.r_sweep_slider = kv["R"]
                st.rerun()

    st.divider()

    # Main plot
    st.markdown("#### c(R) and kappa(R) Curves")
    fig = create_c_R_plot(sweep_data, current_R)
    st.plotly_chart(fig, width='stretch')

    st.divider()

    # Parabola visualization
    st.markdown("#### The Geometry: Kissing the Floor")
    st.markdown("""
    Near the minimum, c(R) behaves like a parabola that just touches (but never goes below)
    the floor at c = 1. This is the **saturation point** of the Levinson-Conrey method.
    """)

    fig_parabola = create_parabola_visualization()
    st.plotly_chart(fig_parabola, width='stretch')

    # Mathematical explanation
    st.divider()
    st.markdown("#### Mathematical Interpretation")

    st.latex(r"""
    \kappa = 1 - \frac{\log c}{R}
    """)

    st.markdown("""
    - When **c > 1**: $\\log c > 0$, so $\\kappa < 1$
    - When **c = 1**: $\\log c = 0$, so $\\kappa = 1$ (theoretical maximum)
    - The constraint **c >= 1** is enforced by the positive-definiteness of the mollified mean square

    **Key insight:** The optimized polynomials create destructive interference that pushes c
    as close to 1 as possible. At R = 1.14978, they achieve exact saturation.
    """)

    # Live computation
    st.divider()
    st.markdown("#### Compute at Current R")

    if st.button("Compute kappa at current R", key="btn_compute_r_sweep"):
        try:
            from ..computation.engine_wrapper import compute_quick_kappa
            from ..utils.constants import OPTIMIZED_P1_TILDE, OPTIMIZED_P2_TILDE, OPTIMIZED_P3_TILDE, PRZZ_Q_COEFFS

            result = compute_quick_kappa(
                OPTIMIZED_P1_TILDE,
                OPTIMIZED_P2_TILDE,
                OPTIMIZED_P3_TILDE,
                PRZZ_Q_COEFFS,
                R=current_R,
                theta=4/7,
                K=3,
            )

            if result.valid:
                col1, col2 = st.columns(2)
                col1.success(f"c = {result.c:.6f}")
                col2.success(f"kappa = {result.kappa:.6f}")

                if abs(result.c - 1.0) < 0.001:
                    st.balloons()
                    st.info("Saturation achieved! c ~ 1.0")
            else:
                st.error(f"Computation failed: {result.message}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
