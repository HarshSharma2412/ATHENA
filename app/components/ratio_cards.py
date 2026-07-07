from __future__ import annotations

from math import isfinite
from typing import Any

import streamlit as st


def _format_ratio(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not isfinite(number):
        return "-"
    return f"{number:,.2f}"


def render_ratio_cards(ratios: dict[str, Any]) -> None:
    """Render ratio-focused health cards."""
    cards = [
        ("ROE", ratios.get("roe")),
        ("ROCE", ratios.get("roce")),
        ("ROA", ratios.get("roa")),
        ("Debt/Equity", ratios.get("debt_equity")),
        ("Current Ratio", ratios.get("current_ratio")),
        ("Quick Ratio", ratios.get("quick_ratio")),
        ("Financial Health Score", ratios.get("financial_health_score")),
    ]

    cols = st.columns(4)
    for index, (label, value) in enumerate(cards):
        with cols[index % 4]:
            st.metric(label=label, value=_format_ratio(value))
