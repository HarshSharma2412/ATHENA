from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from athena.utils.formatting import (
    NOT_AVAILABLE,
    format_market_cap_crores,
    format_percent,
    format_price,
    format_ratio,
    to_float,
)


def _price_no_decimals(value: Any) -> str:
    return format_price(value, decimals=0)


def _tone_for_value(label: str, raw_value: Any) -> str:
    normalized_label = label.casefold()
    number = to_float(raw_value)
    if number is None:
        return "neutral"
    if normalized_label == "52 week low" or number < 0:
        return "negative"
    if number > 0:
        return "positive"
    return "neutral"


def _text(value: Any) -> str:
    return str(value) if value not in (None, "") else NOT_AVAILABLE


def render_summary_cards(summary: dict[str, Any]) -> None:
    """Render high-level company summary cards with professional formatting."""
    cards: list[tuple[str, str, Any]] = [
        ("Company", _text(summary.get("company_name") or summary.get("ticker")), summary.get("company_name")),
        ("Ticker", _text(summary.get("ticker")), summary.get("ticker")),
        ("Current Price", format_price(summary.get("current_price")), summary.get("current_price")),
        ("Market Cap", format_market_cap_crores(summary.get("market_cap")), summary.get("market_cap")),
        ("PE", format_ratio(summary.get("pe")), summary.get("pe")),
        ("PB", format_ratio(summary.get("pb")), summary.get("pb")),
        ("Dividend Yield", format_percent(summary.get("dividend_yield")), summary.get("dividend_yield")),
        ("52 Week High", _price_no_decimals(summary.get("fifty_two_week_high")), summary.get("fifty_two_week_high")),
        ("52 Week Low", _price_no_decimals(summary.get("fifty_two_week_low")), summary.get("fifty_two_week_low")),
    ]

    st.markdown(_CARD_STYLE, unsafe_allow_html=True)

    cols = st.columns(3)
    for index, (label, value, raw_value) in enumerate(cards):
        tone = _tone_for_value(label, raw_value)
        with cols[index % 3]:
            st.markdown(
                f"""
                <div class="athena-card">
                    <div class="athena-card-label">
                        <span class="athena-indicator athena-{tone}"></span>
                        {escape(label)}
                    </div>
                    <div class="athena-card-value">{escape(str(value))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


_CARD_STYLE = """
    <style>
    .athena-card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 16px 18px;
        min-height: 104px;
        background: rgba(15, 23, 42, 0.78);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.16);
    }
    .athena-card-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }
    .athena-card-value {
        color: #f8fafc;
        font-size: 1.24rem;
        font-weight: 760;
        line-height: 1.25;
        margin-top: 8px;
        overflow-wrap: anywhere;
    }
    .athena-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .athena-positive { background: #16a34a; }
    .athena-negative { background: #dc2626; }
    .athena-neutral { background: #64748b; }
    </style>
"""

__all__ = ["render_summary_cards"]
