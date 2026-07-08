"""Immutable data models for ATHENA's automatic DCF valuation engine.

These dataclasses describe every stage of the valuation pipeline: the
historical analysis, the automatically estimated assumptions, the year-by-year
forecast, scenario/sensitivity output and the final :class:`DCFResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Recommendation(str, Enum):
    """Investment recommendation buckets returned by the engine."""

    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    REDUCE = "Reduce"
    SELL = "Sell"


class MarketCapCategory(str, Enum):
    """Indian-market capitalisation buckets used to seed the discount rate."""

    LARGE = "Large Cap"
    MID = "Mid Cap"
    SMALL = "Small Cap"


@dataclass(frozen=True)
class HistoricalAnalysis:
    """Historical figures extracted from the financial statements."""

    years: int
    revenue: list[float] = field(default_factory=list)
    ebit: list[float] = field(default_factory=list)
    operating_margin: list[float] = field(default_factory=list)
    tax_rate: list[float] = field(default_factory=list)
    capex_to_revenue: list[float] = field(default_factory=list)
    working_capital_to_revenue: list[float] = field(default_factory=list)
    depreciation_to_revenue: list[float] = field(default_factory=list)
    free_cash_flow: list[float] = field(default_factory=list)
    fcf_margin: list[float] = field(default_factory=list)
    latest_revenue: Optional[float] = None
    latest_working_capital: Optional[float] = None
    total_debt: Optional[float] = None
    cash: Optional[float] = None


@dataclass(frozen=True)
class ForecastAssumptions:
    """Automatically estimated forward-looking assumptions."""

    revenue_growth: float
    operating_margin: float
    tax_rate: float
    capex_to_revenue: float
    working_capital_to_revenue: float
    depreciation_to_revenue: float
    forecast_years: int
    discount_rate: float
    terminal_growth: float


@dataclass(frozen=True)
class ForecastYear:
    """A single projected year of the explicit forecast horizon."""

    year: int
    revenue: float
    ebit: float
    nopat: float
    capex: float
    working_capital: float
    change_in_working_capital: float
    free_cash_flow: float
    discount_factor: float
    present_value: float


@dataclass(frozen=True)
class ScenarioResult:
    """Intrinsic value for a single scenario (bear/base/bull)."""

    name: str
    intrinsic_value: Optional[float]
    revenue_growth: float
    discount_rate: float
    terminal_growth: float


@dataclass(frozen=True)
class DCFResult:
    """Full output of the automatic DCF valuation."""

    ticker: Optional[str]
    current_price: Optional[float]
    intrinsic_value: Optional[float]
    enterprise_value: Optional[float]
    equity_value: Optional[float]
    margin_of_safety: Optional[float]
    expected_cagr: Optional[float]
    discount_rate: float
    terminal_growth: float
    revenue_growth: float
    confidence_score: float
    bear_value: Optional[float]
    base_value: Optional[float]
    bull_value: Optional[float]
    sensitivity_table: dict[str, dict[str, Optional[float]]]
    recommendation: Recommendation
    assumptions: ForecastAssumptions
    forecast: list[ForecastYear] = field(default_factory=list)
    scenarios: list[ScenarioResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


__all__ = [
    "DCFResult",
    "ForecastAssumptions",
    "ForecastYear",
    "HistoricalAnalysis",
    "MarketCapCategory",
    "Recommendation",
    "ScenarioResult",
]
