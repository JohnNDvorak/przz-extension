"""
Derivation Chain - Interactive flowchart showing the derivation path.

Shows: PRZZ Axioms -> Derived Constants -> Final kappa
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Optional
import math


# Derivation chain data
AXIOMS = {
    "axiom1": {
        "name": "Theta Permitted",
        "short": "theta = 4/7",
        "description": r"The mollifier exponent $\theta = 4/7$ is permitted by PRZZ Theorem 1.1",
        "value": 4/7,
        "source": "PRZZ Section 1",
    },
    "axiom2": {
        "name": "Mirror Identity",
        "short": "Mirror formula",
        "description": r"$I(\alpha,\beta) + T^{-\alpha-\beta} \cdot I(-\beta,-\alpha)$ for $S_{12}$",
        "value": None,
        "source": "PRZZ Section 10",
    },
    "axiom3": {
        "name": "Euler-Maclaurin Weight",
        "short": "(1-u)^{2K-1}",
        "description": r"The Euler-Maclaurin kernel weight $(1-u)^{2K-1}$ for $I_2$",
        "value": None,
        "source": "PRZZ Lemma 5.1",
    },
    "axiom4": {
        "name": "Log Factor",
        "short": "1/theta + x + y",
        "description": r"The $I_1$ integrand includes $\log(1/\theta + x + y)$",
        "value": None,
        "source": "PRZZ Section 7",
    },
}

DERIVED_CONSTANTS = {
    "beta": {
        "name": "Beta Function",
        "formula": r"$\text{Beta}(2, 2K) = \frac{1}{2K(2K+1)}$",
        "value": 1/(2*3*(2*3+1)),  # K=3
        "value_display": "1/42",
        "depends_on": ["axiom3"],
        "source": "Lemma 5.1",
    },
    "g_I2": {
        "name": "g_{I2} Factor",
        "formula": r"$g_{I_2} = 1 + \frac{\theta(2-\theta)}{2K(2K+1)}$",
        "value": 1 + (4/7)*(10/7)/(6*7),
        "value_display": "1.01944",
        "depends_on": ["axiom1", "axiom3"],
        "source": "Theorem 5.1",
    },
    "g_I1": {
        "name": "g_{I1} Factor",
        "formula": r"$g_{I_1} = 1 + \frac{\theta(1-\theta)(2(K-1)+\theta)}{8K(2K+1)^2}$",
        "value": 1 + 16/16807,
        "value_display": "1.00095",
        "depends_on": ["axiom1", "axiom4"],
        "source": "Theorem 5.2",
    },
    "enhancement": {
        "name": "Enhancement Factor",
        "formula": r"$1 + \frac{1}{K(K+1)(2K+1) + 2K\theta}$",
        "value": 1 + 7/612,
        "value_display": "1.01144",
        "depends_on": ["axiom1"],
        "source": "Theorem 5.3",
    },
    "M0": {
        "name": "Mirror Base M0",
        "formula": r"$M_0 = e^R + (2K-1)$",
        "value": None,  # Depends on R
        "value_display": "e^R + 5",
        "depends_on": ["axiom2"],
        "source": "Theorem 4.2",
    },
    "G_total": {
        "name": "Total G Factor",
        "formula": r"$G = f_{I_1} \cdot g_{I_1} + (1 - f_{I_1}) \cdot g_{I_2}$",
        "value": 1.014,  # Approximately
        "value_display": "1.014",
        "depends_on": ["g_I1", "g_I2"],
        "source": "Definition 4.1",
    },
    "M": {
        "name": "Full Mirror M",
        "formula": r"$M = G \times M_0$",
        "value": None,  # Depends on R
        "value_display": "G * (e^R + 5)",
        "depends_on": ["G_total", "M0"],
        "source": "Derived",
    },
}

FINAL_RESULT = {
    "c": {
        "name": "Main Constant c",
        "formula": r"$c = S_{12}(+R) + M \cdot S_{12}(-R) + S_{34}(+R)$",
        "depends_on": ["M"],
    },
    "kappa": {
        "name": "Kappa Bound",
        "formula": r"$\kappa = 1 - \frac{\log(c)}{R}$",
        "depends_on": ["c"],
    },
}


def compute_derived_values(R: float, theta: float = 4/7, K: int = 3) -> Dict:
    """Compute all derived constants at a given R value."""
    values = {}

    # Beta function
    values["beta"] = 1 / (2 * K * (2*K + 1))

    # G-factors
    values["g_I2"] = 1 + theta * (2 - theta) / (2 * K * (2*K + 1))
    values["g_I1"] = 1 + theta * (1 - theta) * (2*(K-1) + theta) / (8 * K * (2*K + 1)**2)
    values["enhancement"] = 1 + 1 / (K * (K+1) * (2*K+1) + 2*K*theta)

    # G total
    values["G_total"] = values["g_I1"] * values["g_I2"] * values["enhancement"]

    # Mirror base
    values["M0"] = math.exp(R) + (2*K - 1)
    values["M"] = values["G_total"] * values["M0"]

    return values


def create_flowchart() -> go.Figure:
    """Create a visual flowchart using Plotly."""

    # Node positions (x, y)
    positions = {
        # Axioms (left column)
        "axiom1": (0, 3),
        "axiom2": (0, 2),
        "axiom3": (0, 1),
        "axiom4": (0, 0),
        # Derived (middle column)
        "beta": (1, 2.5),
        "g_I2": (1, 1.5),
        "g_I1": (1, 0.5),
        "enhancement": (2, 2),
        "M0": (2, 1),
        "G_total": (2, 0),
        "M": (3, 0.5),
        # Final (right column)
        "c": (4, 1),
        "kappa": (5, 1),
    }

    # Edges
    edges = [
        ("axiom3", "beta"),
        ("axiom1", "g_I2"), ("axiom3", "g_I2"),
        ("axiom1", "g_I1"), ("axiom4", "g_I1"),
        ("axiom1", "enhancement"),
        ("axiom2", "M0"),
        ("g_I1", "G_total"), ("g_I2", "G_total"), ("enhancement", "G_total"),
        ("G_total", "M"), ("M0", "M"),
        ("M", "c"),
        ("c", "kappa"),
    ]

    fig = go.Figure()

    # Draw edges
    for start, end in edges:
        x0, y0 = positions[start]
        x1, y1 = positions[end]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color='#888', width=1),
            hoverinfo='none',
        ))

    # Draw nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node_id, (x, y) in positions.items():
        node_x.append(x)
        node_y.append(y)

        if node_id.startswith("axiom"):
            text = AXIOMS[node_id]["short"]
            color = "#1f77b4"  # Blue for axioms
        elif node_id in DERIVED_CONSTANTS:
            text = DERIVED_CONSTANTS[node_id]["name"].replace("_", "<sub>") + "</sub>" if "_" in DERIVED_CONSTANTS[node_id]["name"] else DERIVED_CONSTANTS[node_id]["name"]
            color = "#2ca02c"  # Green for derived
        else:
            text = FINAL_RESULT[node_id]["name"]
            color = "#d62728"  # Red for final

        node_text.append(text)
        node_colors.append(color)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=30, color=node_colors, line=dict(width=2, color='white')),
        text=node_text,
        textposition="bottom center",
        hoverinfo='text',
        hovertext=[f"{t}" for t in node_text],
    ))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        title="Derivation Chain: Axioms -> Constants -> Kappa",
    )

    return fig


def render_derivation_node(node_id: str, values: Dict):
    """Render details for a selected derivation node."""
    if node_id in AXIOMS:
        ax = AXIOMS[node_id]
        st.markdown(f"### Axiom: {ax['name']}")
        st.markdown(ax['description'])
        if ax['value'] is not None:
            st.metric("Value", f"{ax['value']:.6f}")
        st.caption(f"Source: {ax['source']}")

    elif node_id in DERIVED_CONSTANTS:
        dc = DERIVED_CONSTANTS[node_id]
        st.markdown(f"### {dc['name']}")
        st.latex(dc['formula'].replace('$', ''))

        if node_id in values:
            col1, col2 = st.columns(2)
            col1.metric("Computed", f"{values[node_id]:.6f}")
            col2.metric("Symbolic", dc['value_display'])

        st.caption(f"Depends on: {', '.join(dc['depends_on'])}")
        st.caption(f"Source: {dc['source']}")

    elif node_id in FINAL_RESULT:
        fr = FINAL_RESULT[node_id]
        st.markdown(f"### {fr['name']}")
        st.latex(fr['formula'].replace('$', ''))
        st.caption(f"Depends on: {', '.join(fr['depends_on'])}")


def render_derivation_chain(R: float = 1.14978, theta: float = 4/7, K: int = 3):
    """Render the full derivation chain tab."""
    st.markdown("### Derivation Chain")
    st.markdown("""
    This diagram shows how the final $\kappa$ bound is derived from PRZZ axioms
    through a sequence of derived constants.
    """)

    # Compute values
    values = compute_derived_values(R, theta, K)

    # Show flowchart
    fig = create_flowchart()
    st.plotly_chart(fig, width='stretch')

    # Legend
    col1, col2, col3 = st.columns(3)
    col1.markdown(":blue_circle: **Axioms** (PRZZ foundations)")
    col2.markdown(":green_circle: **Derived** (computed constants)")
    col3.markdown(":red_circle: **Final** (results)")

    st.divider()

    # Interactive node explorer
    st.markdown("### Explore Derivation Steps")

    all_nodes = list(AXIOMS.keys()) + list(DERIVED_CONSTANTS.keys()) + list(FINAL_RESULT.keys())
    node_names = {
        **{k: f"Axiom: {v['name']}" for k, v in AXIOMS.items()},
        **{k: f"Derived: {v['name']}" for k, v in DERIVED_CONSTANTS.items()},
        **{k: f"Result: {v['name']}" for k, v in FINAL_RESULT.items()},
    }

    selected = st.selectbox(
        "Select a node to see details",
        all_nodes,
        format_func=lambda x: node_names[x],
        key="derivation_node"
    )

    render_derivation_node(selected, values)

    # Show all computed values
    st.divider()
    st.markdown("### Computed Values at Current R")
    st.caption(f"R = {R}, theta = {theta:.6f}, K = {K}")

    cols = st.columns(4)
    cols[0].metric("g_I1", f"{values['g_I1']:.6f}")
    cols[1].metric("g_I2", f"{values['g_I2']:.6f}")
    cols[2].metric("Enhancement", f"{values['enhancement']:.6f}")
    cols[3].metric("G_total", f"{values['G_total']:.6f}")

    cols = st.columns(3)
    cols[0].metric("M0", f"{values['M0']:.4f}")
    cols[1].metric("M", f"{values['M']:.4f}")
    cols[2].metric("Beta(2,2K)", f"{values['beta']:.6f}")

    # Verification section
    st.divider()
    st.markdown("### Verify Derivation")

    if st.button("Compute Full Result", key="btn_verify_chain"):
        try:
            from ..computation.engine_wrapper import compute_quick_kappa
            from ..utils.constants import OPTIMIZED_P1_TILDE, OPTIMIZED_P2_TILDE, OPTIMIZED_P3_TILDE, PRZZ_Q_COEFFS

            result = compute_quick_kappa(
                OPTIMIZED_P1_TILDE,
                OPTIMIZED_P2_TILDE,
                OPTIMIZED_P3_TILDE,
                PRZZ_Q_COEFFS,
                R=R,
                theta=theta,
                K=K,
            )

            if result.valid:
                st.success(f"c = {result.c:.6f}")
                st.success(f"kappa = {result.kappa:.6f}")

                if abs(result.c - 1.0) < 0.01:
                    st.info("The method is SATURATED (c ~ 1.0)")
            else:
                st.error(f"Computation failed: {result.message}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
