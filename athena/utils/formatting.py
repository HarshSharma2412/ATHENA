"""Reusable, professional number and currency formatting helpers.

All helpers gracefully degrade to ``N/A`` for missing, non-numeric or
non-finite values so the dashboard never renders ``nan``.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

NOT_AVAILABLE = "N/A"
RUPEE = "\u20b9"

_CRORE = 10_000_000.0


def to_float(value: Any) -> float | None:
    """Coerce a value to a finite float, returning ``None`` when impossible."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def format_indian_grouping(number: float, decimals: int = 0) -> str:
    """Format a number using the Indian digit grouping system (lakh/crore)."""
    negative = number < 0
    quantized = f"{abs(number):.{decimals}f}"
    integer_part, _, fraction_part = quantized.partition(".")

    if len(integer_part) > 3:
        head, tail = integer_part[:-3], integer_part[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        integer_part = ",".join(groups) + "," + tail

    grouped = integer_part if not fraction_part else f"{integer_part}.{fraction_part}"
    return f"-{grouped}" if negative else grouped


def format_price(value: Any, decimals: int = 2) -> str:
    """Format a share price as an INR currency string (e.g. ``₹4,235.50``)."""
    number = to_float(value)
    if number is None:
        return NOT_AVAILABLE
    return f"{RUPEE}{format_indian_grouping(number, decimals)}"


def format_market_cap_crores(value: Any) -> str:
    """Format a raw market cap (in rupees) as crores (e.g. ``₹15,67,842 Cr``)."""
    number = to_float(value)
    if number is None:
        return NOT_AVAILABLE
    return f"{RUPEE}{format_indian_grouping(number / _CRORE, 0)} Cr"


def format_ratio(value: Any, decimals: int = 2) -> str:
    """Format a bare ratio (e.g. PE, PB, Debt/Equity) with fixed decimals."""
    number = to_float(value)
    if number is None:
        return NOT_AVAILABLE
    return f"{number:,.{decimals}f}"


def format_percent(value: Any, decimals: int = 2) -> str:
    """Format a percentage, scaling fractional inputs (<= 1) to percents."""
    number = to_float(value)
    if number is None:
        return NOT_AVAILABLE
    if abs(number) <= 1:
        number *= 100
    return f"{number:,.{decimals}f}%"


def format_ratio_percent(value: Any, decimals: int = 2) -> str:
    """Format a ratio expressed as a fraction into a percentage string."""
    number = to_float(value)
    if number is None:
        return NOT_AVAILABLE
    return f"{number * 100:,.{decimals}f}%"


def format_score(value: Any, decimals: int = 1, out_of: int = 100) -> str:
    """Format a bounded score such as the financial health score."""
    number = to_float(value)
    if number is None:
        return NOT_AVAILABLE
    return f"{number:,.{decimals}f} / {out_of}"


__all__ = [
    "NOT_AVAILABLE",
    "RUPEE",
    "format_indian_grouping",
    "format_market_cap_crores",
    "format_percent",
    "format_price",
    "format_ratio",
    "format_ratio_percent",
    "format_score",
    "to_float",
]
