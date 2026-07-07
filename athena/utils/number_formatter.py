"""Reusable, production-grade number and currency formatters for ATHENA.

Every helper degrades gracefully to ``N/A`` for missing, non-numeric or
non-finite input so the UI never renders ``None``, ``NaN`` or ``nan``.

The helpers build on the shared primitives in :mod:`athena.utils.formatting`
(``to_float`` and ``format_indian_grouping``) to avoid duplicated logic.
"""

from __future__ import annotations

import logging
from typing import Any

from athena.utils.formatting import (
    NOT_AVAILABLE,
    RUPEE,
    format_indian_grouping,
    to_float,
)

logger = logging.getLogger(__name__)

_CRORE = 10_000_000.0
# Below this many crores we keep two decimals (e.g. ₹1.50 Cr); at or above it
# the fractional part is noise, so we drop it (e.g. ₹2,182 Cr).
_CRORE_DECIMAL_THRESHOLD = 100.0


def format_currency(value: Any, decimals: int = 2) -> str:
    """Format a rupee amount with Indian grouping, e.g. ``321.9`` -> ``₹321.90``."""
    number = to_float(value)
    if number is None:
        return NOT_AVAILABLE
    return f"{RUPEE}{format_indian_grouping(number, decimals)}"


def format_crore(value: Any) -> str:
    """Convert a raw rupee amount into crores, e.g. ``21820000000`` -> ``₹2,182 Cr``."""
    number = to_float(value)
    if number is None:
        return NOT_AVAILABLE
    crore = number / _CRORE
    decimals = 2 if abs(crore) < _CRORE_DECIMAL_THRESHOLD else 0
    return f"{RUPEE}{format_indian_grouping(crore, decimals)} Cr"


def format_market_cap(value: Any) -> str:
    """Format a market capitalisation (raw rupees) in crores, e.g. ``₹2,182 Cr``."""
    return format_crore(value)


def format_percentage(value: Any, decimals: int = 2) -> str:
    """Format a fractional ratio as a percentage, e.g. ``0.3473`` -> ``34.73%``."""
    number = to_float(value)
    if number is None:
        return NOT_AVAILABLE
    return f"{number * 100:,.{decimals}f}%"


def format_ratio(value: Any, decimals: int = 2) -> str:
    """Format a bare ratio with fixed decimals, e.g. ``4.1823`` -> ``4.18``."""
    number = to_float(value)
    if number is None:
        return NOT_AVAILABLE
    return f"{number:,.{decimals}f}"


__all__ = [
    "NOT_AVAILABLE",
    "format_currency",
    "format_crore",
    "format_market_cap",
    "format_percentage",
    "format_ratio",
]
