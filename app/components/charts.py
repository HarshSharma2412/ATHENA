from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_CRORE = 10_000_000.0

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "athena_chart",
        "height": 720,
        "width": 1280,
        "scale": 2,
    },
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
}


def _apply_terminal_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        hovermode="x unified",
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.38)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#e2e8f0"},
        xaxis={"showgrid": False, "rangeslider": {"visible": False}},
        yaxis={"gridcolor": "rgba(148, 163, 184, 0.16)"},
    )
    return fig


def _render(fig: go.Figure, title: str) -> None:
    st.plotly_chart(
        _apply_terminal_layout(fig, title),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


def render_price_history(history: pd.DataFrame) -> None:
    """Render an interactive Plotly price-history line chart."""
    if history.empty or "close" not in history.columns:
        st.info("No price history available.")
        return

    chart_data = history.copy()
    if "date" in chart_data.columns:
        chart_data["date"] = pd.to_datetime(chart_data["date"], errors="coerce")
        chart_data = chart_data.dropna(subset=["date"]).sort_values("date")
    chart_data = chart_data.dropna(subset=["close"])
    if chart_data.empty:
        st.info("No price history available.")
        return

    fig = px.line(
        chart_data,
        x="date",
        y="close",
        title="Price History",
        color_discrete_sequence=["#38bdf8"],
    )
    fig.update_traces(
        line={"width": 2.6},
        hovertemplate="Date: %{x|%d %b %Y}<br>Close: \u20b9%{y:,.2f}<extra></extra>",
    )
    _render(fig, "Price History")


def render_yearly_trend(frame: pd.DataFrame, title: str, color: str = "#22c55e") -> None:
    """Render a yearly financial metric trend (values shown in ₹ crore)."""
    if frame.empty or not {"year", "value"}.issubset(frame.columns):
        st.info(f"No {title} data available.")
        return

    chart_data = frame.copy()
    chart_data["year"] = pd.to_numeric(chart_data["year"], errors="coerce")
    chart_data["value"] = pd.to_numeric(chart_data["value"], errors="coerce")
    chart_data = chart_data.dropna(subset=["year", "value"]).sort_values("year")
    if chart_data.empty:
        st.info(f"No {title} data available.")
        return

    chart_data["crore"] = chart_data["value"] / _CRORE

    fig = px.line(
        chart_data,
        x="year",
        y="crore",
        markers=True,
        title=title,
        color_discrete_sequence=[color],
    )
    fig.update_traces(
        line={"width": 2.6},
        marker={"size": 7},
        hovertemplate="Year: %{x:.0f}<br>%{fullData.name}: \u20b9%{y:,.0f} Cr<extra></extra>",
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title_text="\u20b9 Crore")
    _render(fig, title)


__all__ = ["PLOTLY_CONFIG", "render_price_history", "render_yearly_trend"]
