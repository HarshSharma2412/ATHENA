from __future__ import annotations

from html import escape
from math import isfinite
from typing import Any

import streamlit as st


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(number):
        return None
    return number


def _format_number(value: Any, suffix: str = "") -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    if abs(number) >= 100:
        formatted = f"{number:,.0f}"
    elif abs(number) >= 10:
        formatted = f"{number:,.1f}"
    else:
        formatted = f"{number:,.2f}"
    return f"{formatted}{suffix}"


def _format_price(value: Any) -> str:
    return _format_number(value)


def _format_crores(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    return f"{number / 10_000_000:,.0f} Cr"


def _format_percent(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    if abs(number) <= 1:
        number *= 100
    return f"{number:,.2f}%"


def _tone_for_value(label: str, raw_value: Any) -> str:
    normalized_label = label.casefold()
    number = _to_float(raw_value)
    if number is None:
        return "neutral"
    if normalized_label == "52 week low" or number < 0:
        return "negative"
    if number > 0:
        return "positive"
    return "neutral"


def render_summary_cards(summary: dict[str, Any]) -> None:
    """Render high-level company summary cards."""
    cards = [
        (
            "Company",
            summary.get("company_name") or summary.get("ticker") or "-",
            summary.get("company_name"),
        ),
        ("Ticker", summary.get("ticker") or "-", summary.get("ticker")),
        (
            "Current Price",
            _format_price(summary.get("current_price")),
            summary.get("current_price"),
        ),
        (
            "Market Cap",
            _format_crores(summary.get("market_cap")),
            summary.get("market_cap"),
        ),
        ("PE", _format_number(summary.get("pe")), summary.get("pe")),
        ("PB", _format_number(summary.get("pb")), summary.get("pb")),
        (
            "Dividend Yield",
            _format_percent(summary.get("dividend_yield")),
            summary.get("dividend_yield"),
        ),
        (
            "52 Week High",
            _format_price(summary.get("fifty_two_week_high")),
            summary.get("fifty_two_week_high"),
        ),
        (
            "52 Week Low",
            _format_price(summary.get("fifty_two_week_low")),
            summary.get("fifty_two_week_low"),
        ),
    ]

    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )

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
