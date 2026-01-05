"""
Asymptotic Explorer - Visualizing kappa_explicit -> 1 as L -> infinity.

Shows how the error term vanishes and the explicit bound approaches the main term.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict
import math


def get_asymptotic_data() -> List[Dict]:
    """Get precomputed asymptotic data."""
    try:
        from ..utils.constants import get_asymptotic_data
        data = get_asymptotic_data()
        if data:
            return data
    except:
        pass

    # Fallback data (from paper Table in Section 9.2)
    return [
        {"L": 9.2, "T_approx": "10^4", "error_percent": 57.6, "kappa_rigorous": 0.424},
        {"L": 13.8, "T_approx": "10^6", "error_percent": 38.4, "kappa_rigorous": 0.616},
        {"L": 23.0, "T_approx": "10^10", "error_percent": 23.1, "kappa_rigorous": 0.769},
        {"L": 39.1, "T_approx": "10^17", "error_percent": 13.6, "kappa_rigorous": 0.865},
        {"L": 46.1, "T_approx": "10^20", "error_percent": 11.5, "kappa_rigorous": 0.885},
        {"L": 115.1, "T_approx": "10^50", "error_percent": 4.6, "kappa_rigorous": 0.954},
        {"L": 230.3, "T_approx": "10^100", "error_percent": 2.3, "kappa_rigorous": 0.977},
        {"L": 2302.6, "T_approx": "10^1000", "error_percent": 0.23, "kappa_rigorous": 0.9977},
        {"L": 23025.9, "T_approx": "10^10000", "error_percent": 0.023, "kappa_rigorous": 0.99977},
    ]


def compute_error_at_L(L: float, kappa_main: float = 1.0, C_error: float = 5.4) -> Dict:
    """Compute error metrics at a given L value."""
    # Error scales as C/L
    error_percent = C_error / L * 100
    error_fraction = C_error / L
    kappa_rigorous = kappa_main - error_fraction

    # T approximation (T ~ e^L for large L)
    if L < 100:
        T_approx = math.exp(L)
        T_str = f"{T_approx:.2e}"
    elif L < 500:
        # Express as 10^x
        log10_T = L / math.log(10)
        T_str = f"10^{log10_T:.0f}"
    else:
        log10_T = L / math.log(10)
        T_str = f"10^{log10_T:.0f}"

    return {
        "L": L,
        "T_approx": T_str,
        "error_percent": error_percent,
        "error_fraction": error_fraction,
        "kappa_rigorous": kappa_rigorous,
    }


def create_asymptotic_plot(data: List[Dict], current_L: float = 40) -> go.Figure:
    """Create the asymptotic behavior plot."""
    L_vals = [d["L"] for d in data if isinstance(d["L"], (int, float))]
    kappa_vals = [d["kappa_rigorous"] for d in data if isinstance(d["L"], (int, float))]
    error_vals = [d["error_percent"] for d in data if isinstance(d["L"], (int, float))]

    fig = go.Figure()

    # Kappa explicit curve
    fig.add_trace(go.Scatter(
        x=L_vals, y=kappa_vals,
        mode='lines+markers',
        name='kappa_explicit(L)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10),
    ))

    # Limit line at kappa = 1
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="green",
        annotation_text="kappa = 1 (liminf as L -> infinity)",
        annotation_position="right"
    )

    # Current L marker
    if current_L in L_vals:
        idx = L_vals.index(current_L)
        kappa_current = kappa_vals[idx]
    else:
        kappa_current = np.interp(current_L, L_vals, kappa_vals)

    fig.add_trace(go.Scatter(
        x=[current_L], y=[kappa_current],
        mode='markers',
        name=f'Current L={current_L}',
        marker=dict(size=15, color='red', symbol='star'),
    ))

    fig.update_layout(
        title="Asymptotic Convergence: kappa_explicit -> 1 as L -> infinity",
        xaxis_title="L = log(T)",
        yaxis_title="kappa_explicit",
        template="plotly_white",
        height=400,
        xaxis=dict(type="log"),
    )

    return fig


def create_error_decay_plot(data: List[Dict]) -> go.Figure:
    """Create the error decay plot."""
    L_vals = [d["L"] for d in data if isinstance(d["L"], (int, float))]
    error_vals = [d["error_percent"] for d in data if isinstance(d["L"], (int, float))]

    fig = go.Figure()

    # Error decay curve
    fig.add_trace(go.Scatter(
        x=L_vals, y=error_vals,
        mode='lines+markers',
        name='Error (%)',
        line=dict(color='#d62728', width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.2)',
    ))

    # 1/L reference curve
    L_ref = np.linspace(40, 1000, 100)
    error_ref = 5.4 / L_ref * 100
    fig.add_trace(go.Scatter(
        x=L_ref, y=error_ref,
        mode='lines',
        name='O(1/L) reference',
        line=dict(color='gray', width=2, dash='dot'),
    ))

    fig.update_layout(
        title="Error Decay: O(1/log T) Vanishing",
        xaxis_title="L = log(T)",
        yaxis_title="Error (%)",
        template="plotly_white",
        height=350,
        xaxis=dict(type="log"),
        yaxis=dict(type="log"),
    )

    return fig


def render_asymptotic_tab():
    """Render the asymptotic explorer tab."""
    st.markdown("### Asymptotic Behavior")
    st.markdown("""
    As $T \\to \\infty$ (equivalently, $L = \\log T \\to \\infty$), the error term vanishes
    and $\\kappa_{\\text{explicit}} \\to 1$.

    This supports **Theorem 1.3**: $\\kappa := \\liminf_{T \\to \\infty} N_0(T)/N(T) = 1$.
    The explicit finite-height bound is valid for $T \\gtrsim 10^{17}$ (i.e., $L \\approx 40$).
    The app stores this as kappa_rigorous; the paper denotes it $\\kappa_{\\mathrm{explicit}}(T)$.
    """)
    st.caption(
        "Rows below the validity threshold $T_0 \\approx 10^{17}$ are illustrative; "
        "the explicit error model is not proven for those heights."
    )

    # Get data
    data = get_asymptotic_data()

    # L selector
    st.markdown("#### Explore L = log(T)")
    numeric_L = [d["L"] for d in data if isinstance(d["L"], (int, float))]
    if not numeric_L:
        st.warning("No asymptotic data available.")
        return

    L_options = sorted(numeric_L)
    default_L = min(L_options, key=lambda x: abs(x - 39.1))
    current_L = st.select_slider(
        "Select L value",
        options=L_options,
        value=default_L,
        key="asymptotic_L_slider",
        format_func=lambda x: f"{x:.1f}",
    )

    # Use table values when available for exact display
    row = next((d for d in data if d.get("L") == current_L), None)
    if row:
        metrics = {
            "L": current_L,
            "T_approx": row.get("T_approx", "—"),
            "error_percent": row.get("error_percent", 0.0),
            "kappa_rigorous": row.get("kappa_rigorous", 0.0),
        }
    else:
        metrics = compute_error_at_L(current_L)

    # Display current values
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("L = log(T)", f"{current_L:.1f}")
    col2.metric("T approximation", metrics["T_approx"])
    col3.metric("Error", f"{metrics['error_percent']:.2f}%")
    col4.metric("kappa_explicit", f"{metrics['kappa_rigorous']:.4f}")

    st.divider()

    # Asymptotic plot
    st.markdown("#### Convergence to kappa = 1")
    fig = create_asymptotic_plot(data, current_L)
    st.plotly_chart(fig, use_container_width=True)

    # Error decay plot
    st.markdown("#### Error Decay Rate")
    fig_error = create_error_decay_plot(data)
    st.plotly_chart(fig_error, use_container_width=True)

    st.divider()

    # Data table
    st.markdown("#### Asymptotic Data Table")

    # Prepare table data - use strings for all values to avoid Arrow conversion issues
    table_data = []
    for d in data:
        if isinstance(d["L"], (int, float)):
            L_val = d["L"]
            L_str = f"{L_val:.1f}"
            table_data.append({
                "L": L_str,
                "T (approx)": str(d["T_approx"]),
                "Error (%)": f"{d['error_percent']:.2f}",
                "kappa_explicit": f"{d['kappa_rigorous']:.4f}",
                "kappa_gap from 1": f"{1 - d['kappa_rigorous']:.4f}",
            })

    # Add infinity row
    table_data.append({
        "L": "infinity",
        "T (approx)": "infinity",
        "Error (%)": "0.00",
        "kappa_explicit": "1.0000",
        "kappa_gap from 1": "0.0000",
    })

    import pandas as pd
    df = pd.DataFrame(table_data)
    st.dataframe(df, hide_index=True)

    st.markdown("#### Height Milestones")
    milestone_rows = [
        {"Target κ": "≥ 50%", "Required T": "10^5"},
        {"Target κ": "≥ 86.5%", "Required T": "10^17"},
        {"Target κ": "≥ 90%", "Required T": "10^23"},
        {"Target κ": "≥ 99%", "Required T": "10^230"},
        {"Target κ": "≥ 99.99%", "Required T": "10^23026"},
    ]
    st.table(milestone_rows)
    st.caption("From the paper's explicit error constants and $\\kappa_{\\text{explicit}}$ table.")

    st.divider()

    # Mathematical explanation
    st.markdown("#### Mathematical Framework")

    st.latex(r"""
    \kappa_{\text{explicit}}(T) = \kappa_{\text{main}} - O\left(\frac{1}{\log T}\right)
    """)

    st.markdown("""
    **The error structure:**
    - The Levinson method produces error terms of order $O(T/L)$ and $O(T/L^2)$
    - These contribute to $\\kappa$ as $O(1/L) = O(1/\\log T)$
    - At our saturation point: $\\kappa_{\\text{main}} = 1.0$

    **Taking the limit:**
    """)

    st.latex(r"""
    \lim_{T \to \infty} \kappa_{\text{explicit}}(T) = 1 - \lim_{T \to \infty} O\left(\frac{1}{\log T}\right) = 1
    """)

    st.markdown("""
    **Consequence (Theorem 1.3):**
    """)

    st.latex(r"""
    \liminf_{T \to \infty} \frac{N_0(T)}{N(T)} = 1
    """)

    st.info("""
    **Key interpretation:** Almost all zeros of the Riemann zeta function lie on the critical line
    in the limit $T \\to \\infty$. Any zeros off the critical line have density zero.

    **This does NOT prove the Riemann Hypothesis** --- RH requires *every* zero to be on the
    critical line, while our result permits a sparse (measure-zero) set of exceptions.
    """)

    # Interactive exploration
    st.divider()
    st.markdown("#### What Height Do We Need?")

    target_kappa = st.slider(
        "Target kappa_explicit",
        min_value=0.90,
        max_value=0.9999,
        value=0.99,
        step=0.001,
        format="%.4f",
        key="target_kappa_slider"
    )

    # Compute required L
    # kappa_rig = 1 - C/L => L = C / (1 - kappa_rig)
    C_error = 5.4  # Error constant
    required_L = C_error / (1 - target_kappa)

    # Compute T
    required_log10_T = required_L / math.log(10)

    col1, col2, col3 = st.columns(3)
    col1.metric("Required L", f"{required_L:.1f}")
    col2.metric("Required log10(T)", f"{required_log10_T:.0f}")

    if required_log10_T < 100:
        col3.metric("T approximation", f"10^{required_log10_T:.0f}")
    else:
        col3.metric("T approximation", f"10^{required_log10_T:.0f}")

    if target_kappa > 0.999:
        st.warning(f"To achieve kappa > {target_kappa:.4f}, we need heights T ~ 10^{required_log10_T:.0f}, which is astronomically large but mathematically valid.")
    else:
        st.success(f"Achievable at heights T ~ 10^{required_log10_T:.0f}")
