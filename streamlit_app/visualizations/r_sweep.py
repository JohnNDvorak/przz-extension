"""
R Sweep Dashboard - Interactive exploration of c(R) geometry.

Shows how c varies with R and identifies the saturation threshold.
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
        # Fallback data from v13 table (optimized polynomials)
        return [
            {"R": 0.80, "c": 0.9432, "kappa_main": 1.0000, "kappa_rigorous": None, "error_percent": None, "error_scale": "Vacuous"},
            {"R": 1.00, "c": 0.9863, "kappa_main": 1.0000, "kappa_rigorous": None, "error_percent": None, "error_scale": "Vacuous"},
            {"R": 1.14976, "c": 1.0000, "kappa_main": 1.0000, "kappa_rigorous": 0.8650, "error_percent": 13.50, "error_scale": "Saturation"},
            {"R": 1.20, "c": 1.0066, "kappa_main": 0.9943, "kappa_rigorous": 0.8593, "error_percent": 13.60, "error_scale": "Non-trivial"},
            {"R": 1.3036, "c": 1.0433, "kappa_main": 0.9675, "kappa_rigorous": 0.8477, "error_percent": 12.39, "error_scale": "PRZZ point"},
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
                  annotation_text="c = 1 (saturation threshold)",
                  annotation_position="right", row=1, col=1)

    # Mark saturation point (rounded for display)
    fig.add_trace(
        go.Scatter(
            x=[1.14976], y=[1.0],
            mode='markers',
            name='Saturation point',
            marker=dict(size=15, color='gold', symbol='star', line=dict(width=2, color='black')),
        ),
        row=1, col=1
    )

    # Mark PRZZ R reference (optimized polynomials)
    przz_c = np.interp(1.3036, R_vals, c_vals)
    fig.add_trace(
        go.Scatter(
            x=[1.3036], y=[przz_c],
            mode='markers',
            name='PRZZ R reference',
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

    # Plot rigorous values only where available
    rig_points = [(r, k) for r, k in zip(R_vals, kappa_rig) if k is not None]
    if rig_points:
        rig_R, rig_vals = zip(*rig_points)
        fig.add_trace(
            go.Scatter(
                x=list(rig_R), y=list(rig_vals),
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

    # Mark saturation point (rounded for display)
    fig.add_trace(
        go.Scatter(
            x=[1.14976], y=[1.0],
            mode='markers',
            name='Saturation point',
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
    """Create the saturation-threshold parabola visualization."""
    R_vals = np.linspace(0.8, 1.5, 100)

    # Approximate c(R) as a parabola around the saturation point
    # c(R) ~ 1 + a*(R - R_min)^2 where R_min ≈ 1.1497602315
    R_min = 1.1497602315
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

    # Saturation threshold at c=1
    fig.add_trace(go.Scatter(
        x=[0.8, 1.5], y=[1.0, 1.0],
        mode='lines',
        name='c = 1 (saturation)',
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
        text=f"R_opt = {R_min:.10f}<br>c = 1.0000",
        showarrow=True,
        arrowhead=2,
        ax=50, ay=-50,
        font=dict(size=14),
    )

    fig.update_layout(
        title="The Geometry of c(R): Saturation Threshold",
        xaxis_title="R (shift parameter)",
        yaxis_title="c (main-term constant)",
        template="plotly_white",
        height=400,
        showlegend=True,
    )

    # Set y-axis range to focus on saturation region
    fig.update_yaxes(range=[0.98, 1.15])

    return fig


def render_r_sweep_tab(current_coeffs: Optional[Dict] = None):
    """Render the R sweep exploration tab."""
    st.markdown("### R Sweep Dashboard")
    st.markdown("""
    The shift parameter $R$ determines where the Levinson method evaluates its bound.
    The key discovery is a unique **saturation threshold** at
    **$R_{\\mathrm{opt}} = 1.149760231531068\\ldots$**, where $c(R_{\\mathrm{opt}}) = 1$.
    """)
    st.caption(
        "Paper values use adaptive quadrature (n=100, stable to n=200). "
        "This module uses fixed quadrature (live n=40, full n=60) and rounded R values for interactivity."
    )
    st.caption("At R = 1.14978, the paper reports c = 1.0000024; the deviation vanishes as R → R_opt.")

    # Get precomputed data
    sweep_data = get_precomputed_sweep_data()

    R_vals = [d["R"] for d in sweep_data]
    c_vals = [d["c"] for d in sweep_data]
    kappa_main_vals = [d["kappa_main"] for d in sweep_data]
    kappa_rig_vals = [d["kappa_rigorous"] for d in sweep_data]

    # R slider
    st.markdown("#### Explore R values")
    min_R = min(R_vals)
    max_R = max(R_vals)
    current_R = st.slider(
        "Select R value",
        min_value=float(min_R),
        max_value=float(max_R),
        value=1.14976,
        step=0.001,
        format="%.4f",
        key="r_sweep_slider"
    )

    c_current = np.interp(current_R, R_vals, c_vals)
    kappa_main_current = np.interp(current_R, R_vals, kappa_main_vals)

    rig_points = [(r, k) for r, k in zip(R_vals, kappa_rig_vals) if k is not None]
    if rig_points:
        rig_R, rig_vals = zip(*rig_points)
        kappa_rig_current = np.interp(current_R, rig_R, rig_vals) if current_R >= min(rig_R) and current_R <= max(rig_R) else None
    else:
        kappa_rig_current = None

    # Display current values
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R", f"{current_R:.4f}")
    col2.metric("c", f"{c_current:.6f}")
    col3.metric("kappa_main", f"{min(kappa_main_current, 1.0):.6f}")
    if c_current < 1.0 or kappa_rig_current is None:
        col4.metric("kappa_rigorous", "N/A")
    else:
        col4.metric("kappa_rigorous", f"{kappa_rig_current:.4f}")

    # Key R values
    st.markdown("#### Key R Values")
    key_R_data = [
        {"R": 1.1497602315, "name": "Kappa saturation (R_opt)", "c": 1.0000, "kappa_main": 1.0000, "kappa_rig": 0.8650},
        {"R": 1.0796557513, "name": "Kappa* saturation (R*_opt)", "c": 1.0000, "kappa_main": 1.0000, "kappa_rig": 0.84},
        {"R": 1.3036, "name": "PRZZ kappa (baseline R)", "c": 2.137449, "kappa_main": 0.417293962, "kappa_rig": 0.343},
        {"R": 1.1167, "name": "PRZZ kappa* (baseline R)", "c": 1.9380, "kappa_main": 0.407511457, "kappa_rig": 0.34},
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

    st.markdown("#### Complete R Sweep Results (paper)")
    sweep_table = []
    for row in sweep_data:
        R_val = row.get("R")
        c_val = row.get("c")
        kappa_main_val = row.get("kappa_main")
        kappa_rig_val = row.get("kappa_rigorous")
        error_pct = row.get("error_percent")
        error_scale = row.get("error_scale")

        kappa_main_display = "—"
        if isinstance(kappa_main_val, (int, float)):
            kappa_main_display = f"{min(kappa_main_val, 1.0):.4f}"
            if isinstance(c_val, (int, float)) and c_val < 1.0:
                kappa_main_display += " (cap)"
        kappa_rig_display = "—"
        if isinstance(kappa_rig_val, (int, float)):
            kappa_rig_display = f"{kappa_rig_val:.4f}"

        sweep_table.append({
            "R": f"{R_val:.5f}" if isinstance(R_val, (int, float)) else str(R_val),
            "c": f"{c_val:.4f}" if isinstance(c_val, (int, float)) else "—",
            "kappa_main": kappa_main_display,
            "kappa_rigorous": kappa_rig_display,
            "Error %": f"{error_pct:.2f}%" if isinstance(error_pct, (int, float)) else "—",
            "Error Scale": error_scale or "—",
        })
    st.table(sweep_table)
    st.caption("Error % and scale are from the explicit error model at L = 40 (relative to R=1.3036 baseline).")

    st.markdown("#### Convergence to Saturation (paper)")
    convergence_rows = [
        {"R": "1.14978", "c(R)": "1.000002380", "|c-1|": "2.38e-6"},
        {"R": "1.14977", "c(R)": "1.000001176", "|c-1|": "1.18e-6"},
        {"R": "1.149765", "c(R)": "1.000000089", "|c-1|": "8.9e-8"},
        {"R": "1.149760231...", "c(R)": "1.000000000", "|c-1|": "<5e-16"},
    ]
    st.table(convergence_rows)

    st.divider()

    # Main plot
    st.markdown("#### c(R) and kappa(R) Curves")
    fig = create_c_R_plot(sweep_data, current_R)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Parabola visualization
    st.markdown("#### The Geometry: Saturation Threshold")
    st.markdown("""
    Near the saturation threshold, c(R) behaves like a parabola that crosses $c=1$.
    The crossing point marks the **saturation threshold** of the Levinson-Conrey method.
    """)

    fig_parabola = create_parabola_visualization()
    st.plotly_chart(fig_parabola, use_container_width=True)

    # Mathematical explanation
    st.divider()
    st.markdown("#### Mathematical Interpretation")

    st.latex(r"""
    \kappa = 1 - \frac{\log c}{R}
    """)

    st.markdown("""
    - When **c > 1**: $\\log c > 0$, so $\\kappa_{\\text{main}} < 1$ (non-trivial bound)
    - When **c = 1**: $\\log c = 0$, so $\\kappa_{\\text{main}} = 1$ (saturated)
    - When **c < 1**: $\\log c < 0$, so $\\kappa_{\\text{main}} > 1$ (vacuous, capped at 1)
    - The rigorous gap scales roughly like $1/R$, so smaller $R$ increases error even when $\\kappa_{\\text{main}}$ rises

    **Key insight:** The optimized polynomials create destructive interference that drives $c$
    to the saturation threshold. At $R_{\\mathrm{opt}} \\approx 1.14976$, we have $c(R_{\\mathrm{opt}})=1$.

    **Flat profile:** The paper notes $c < 1.03$ for $R \\in [0.85, 1.2]$, highlighting
    robustness near saturation.
    """)

    st.markdown("""
    **Why the optimized polynomials are special:**
    - Optimized: $c \\in [1.002, 1.088]$ for $R \\in [0.5, 1.5]$
    - PRZZ baseline: $c \\in [2.04, 2.24]$ for $R \\in [0.5, 1.5]$
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
