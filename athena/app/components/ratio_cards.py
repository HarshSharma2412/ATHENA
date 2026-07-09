from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from athena.utils.formatting import format_ratio, format_ratio_percent, format_score

# (label, value key, formatter) - profitability ratios are shown as percentages,
# leverage/liquidity ratios as plain multiples and the health score out of 100.
_RATIO_CARDS: tuple[tuple[str, str, Callable[[Any], str]], ...] = (
    ("ROE", "roe", format_ratio_percent),
    ("ROCE", "roce", format_ratio_percent),
    ("ROA", "roa", format_ratio_percent),
    ("Debt/Equity", "debt_equity", format_ratio),
    ("Current Ratio", "current_ratio", format_ratio),
    ("Quick Ratio", "quick_ratio", format_ratio),
    ("Financial Health Score", "financial_health_score", format_score),
)


def render_ratio_cards(ratios: dict[str, Any]) -> None:
    """Render ratio-focused health cards, never displaying ``nan``."""
    cols = st.columns(4)
    for index, (label, key, formatter) in enumerate(_RATIO_CARDS):
        with cols[index % 4]:
            st.metric(label=label, value=formatter(ratios.get(key)))


__all__ = ["render_ratio_cards"]
