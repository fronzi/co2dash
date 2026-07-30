"""
Plotly figure builders for the co2dash GUI. Pure functions (no Streamlit) so they
can be unit-tested. Visual identity: a precise 'laboratory instrument' palette —
electrochemistry teal, a carbon/energy amber, and clear feasibility semantics.
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

# --- design tokens ------------------------------------------------------------
INK = "#14242E"; MUTED = "#5C6B73"; GRID = "#E3E8EC"
TEAL = "#0E7C86"; TEAL_BR = "#14A3B0"; AMBER = "#C77D17"
GREEN = "#1E8E5A"; RED = "#C0392B"
# cheap -> expensive (teal -> amber -> red)
MAC_SCALE = [[0.0, "#0E7C86"], [0.45, "#5BB6A6"], [0.7, AMBER], [1.0, "#9B2D20"]]


def _style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=56, r=20, t=34, b=44), height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_family="JetBrains Mono, monospace"))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def cost_waterfall(fixed: float, co2: float, elec: float, h2: float) -> go.Figure:
    """LCOP build-up as a waterfall ($/kg)."""
    labels = ["Capital + fixed", "CO₂ feed", "Electricity", "H₂", "LCOP"]
    vals = [fixed, co2, elec, h2, 0]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["relative", "relative", "relative", "relative", "total"],
        x=labels, y=vals, connector=dict(line=dict(color=GRID)),
        increasing=dict(marker=dict(color=TEAL)),
        totals=dict(marker=dict(color=INK)),
        text=[f"{v:.2f}" for v in [fixed, co2, elec, h2, fixed+co2+elec+h2]],
        textposition="outside"))
    fig.update_yaxes(title_text="$/kg product")
    return _style(fig)


def mac_distribution(mac: np.ndarray, carbon_price: float,
                     p05: float, median: float, p95: float) -> go.Figure:
    finite = np.isfinite(mac)
    data = mac[finite]
    if data.size:
        hi = np.percentile(data, 99)
        data = np.clip(data, None, hi)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=50, marker=dict(color=TEAL_BR, opacity=0.85),
                               name="MAC draws", hovertemplate="MAC %{x:.2f} $/kg<extra></extra>"))
    fig.add_vrect(x0=float(np.min(data)) if data.size else 0, x1=carbon_price,
                  fillcolor=GREEN, opacity=0.08, line_width=0)
    fig.add_vline(x=carbon_price, line=dict(color=GREEN, dash="dash", width=2),
                  annotation_text="carbon price", annotation_font_color=GREEN)
    fig.add_vline(x=median, line=dict(color=INK, width=2),
                  annotation_text="median", annotation_position="top")
    fig.update_xaxes(title_text="Marginal abatement cost ($/kg CO₂)")
    fig.update_yaxes(title_text="count")
    return _style(fig)


def envelope_heatmap(X, Y, mac, feasible, xlabel, ylabel) -> go.Figure:
    Z = mac.copy()
    fin = np.isfinite(Z)
    if fin.any():
        Z = np.where(fin, Z, np.nan)
        cap = np.nanpercentile(Z[fin], 95)
        Z = np.clip(Z, None, cap)
    x = X[0, :]; y = Y[:, 0]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Z, x=x, y=y, colorscale=MAC_SCALE,
                             colorbar=dict(title="MAC<br>$/kg CO₂"),
                             hovertemplate=f"{xlabel} %{{x:.3g}}<br>{ylabel} %{{y:.3g}}"
                                           "<br>MAC %{z:.2f}<extra></extra>"))
    fig.add_trace(go.Contour(z=feasible.astype(float), x=x, y=y,
                             showscale=False, contours=dict(start=0.5, end=0.5, size=1,
                             coloring="lines"), line=dict(color="white", width=3),
                             hoverinfo="skip", name="feasibility boundary"))
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    return _style(fig, height=420)


def sobol_tornado(names, s1, st) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=st, orientation="h", name="ST (total)",
                         marker=dict(color=AMBER)))
    fig.add_trace(go.Bar(y=names, x=s1, orientation="h", name="S1 (first-order)",
                         marker=dict(color=TEAL)))
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
    fig.update_xaxes(title_text="Sobol index (share of MAC variance)")
    return _style(fig, height=300)


def reliability_diagram(levels, before, after) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=MUTED, dash="dot"), name="ideal"))
    fig.add_trace(go.Scatter(x=levels, y=before, mode="lines+markers",
                             line=dict(color=RED), name="before"))
    fig.add_trace(go.Scatter(x=levels, y=after, mode="lines+markers",
                             line=dict(color=GREEN), name="after"))
    fig.update_xaxes(title_text="nominal coverage", range=[0, 1])
    fig.update_yaxes(title_text="empirical coverage", range=[0, 1])
    return _style(fig, height=320)
