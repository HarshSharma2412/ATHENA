"""Pure, reusable financial-mathematics primitives for ATHENA's valuation stack.

Every helper is side-effect free, fully typed and defensive: non-finite or
missing inputs collapse to ``None`` (or are skipped) rather than raising, so the
valuation engine never crashes on sparse real-world data.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


def to_float(value: object) -> Optional[float]:
    """Coerce a value to a finite float, returning ``None`` when impossible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clean_series(values: Iterable[object]) -> list[float]:
    """Return the finite floats from an iterable, preserving order."""
    cleaned: list[float] = []
    for value in values:
        number = to_float(value)
        if number is not None:
            cleaned.append(number)
    return cleaned


def safe_divide(numerator: object, denominator: object) -> Optional[float]:
    """Divide two values, returning ``None`` on missing or zero denominator."""
    num = to_float(numerator)
    den = to_float(denominator)
    if num is None or den is None or den == 0.0:
        return None
    return num / den


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp ``value`` into the inclusive ``[lower, upper]`` range."""
    return max(lower, min(upper, value))


def mean(values: Sequence[float]) -> Optional[float]:
    """Arithmetic mean of finite values, or ``None`` when empty."""
    cleaned = clean_series(values)
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def weighted_average(
    values: Sequence[Optional[float]], weights: Sequence[float]
) -> Optional[float]:
    """Weighted average skipping ``None`` values and renormalising weights."""
    if len(values) != len(weights):
        raise ValueError("values and weights must be the same length")
    total_weight = 0.0
    accumulated = 0.0
    for value, weight in zip(values, weights):
        number = to_float(value)
        if number is None or weight <= 0:
            continue
        accumulated += number * weight
        total_weight += weight
    if total_weight == 0.0:
        return None
    return accumulated / total_weight


def stdev(values: Sequence[float]) -> Optional[float]:
    """Population standard deviation of finite values."""
    cleaned = clean_series(values)
    if len(cleaned) < 2:
        return None
    average = sum(cleaned) / len(cleaned)
    variance = sum((value - average) ** 2 for value in cleaned) / len(cleaned)
    return math.sqrt(variance)


def coefficient_of_variation(values: Sequence[float]) -> Optional[float]:
    """Relative dispersion (``stdev / |mean|``); lower means more stable."""
    average = mean(values)
    deviation = stdev(values)
    if average is None or deviation is None or average == 0.0:
        return None
    return deviation / abs(average)


def cagr(begin: object, end: object, years: int) -> Optional[float]:
    """Compound annual growth rate between two positive values over ``years``."""
    start = to_float(begin)
    finish = to_float(end)
    if start is None or finish is None or years <= 0:
        return None
    if start <= 0.0 or finish <= 0.0:
        return None
    return (finish / start) ** (1.0 / years) - 1.0


def series_cagr(values: Sequence[float], years: int) -> Optional[float]:
    """CAGR over the last ``years`` of an oldest-to-newest series."""
    cleaned = clean_series(values)
    if len(cleaned) < 2:
        return None
    window = cleaned[-(years + 1):] if years + 1 <= len(cleaned) else cleaned
    span = len(window) - 1
    if span <= 0:
        return None
    return cagr(window[0], window[-1], span)


def discount_factor(rate: float, period: int) -> float:
    """Present-value discount factor ``1 / (1 + rate) ** period``."""
    return 1.0 / ((1.0 + rate) ** period)


def present_value(cash_flow: float, rate: float, period: int) -> float:
    """Discount a single future cash flow back to today."""
    return cash_flow * discount_factor(rate, period)


def present_value_of_series(cash_flows: Sequence[float], rate: float) -> float:
    """Present value of a series of cash flows starting one period from now."""
    return sum(
        present_value(cash_flow, rate, period)
        for period, cash_flow in enumerate(cash_flows, start=1)
    )


def gordon_terminal_value(
    final_cash_flow: float, discount_rate: float, terminal_growth: float
) -> Optional[float]:
    """Gordon-growth terminal value; ``None`` when growth >= discount rate."""
    spread = discount_rate - terminal_growth
    if spread <= 0.0:
        return None
    return final_cash_flow * (1.0 + terminal_growth) / spread


__all__ = [
    "cagr",
    "clamp",
    "clean_series",
    "coefficient_of_variation",
    "discount_factor",
    "gordon_terminal_value",
    "mean",
    "present_value",
    "present_value_of_series",
    "safe_divide",
    "series_cagr",
    "stdev",
    "to_float",
    "weighted_average",
]
