"""Automatic assumption estimation and cash-flow forecasting for ATHENA.

The :class:`ForecastEngine` turns raw financial statements into a
:class:`~athena.models.valuation_models.HistoricalAnalysis`, derives every
forward-looking assumption from history (no user input), and projects an
explicit free-cash-flow forecast. It is deliberately independent of the DCF
engine so it can be reused and unit-tested on its own.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from athena.models.valuation_models import (
    ForecastAssumptions,
    ForecastYear,
    HistoricalAnalysis,
)
from athena.utils import finance_math as fm

logger = logging.getLogger(__name__)

# Weighted blend for revenue growth: recent years dominate.
_GROWTH_WEIGHTS: dict[int, float] = {10: 0.20, 5: 0.30, 3: 0.50}

# Guardrails so a single explosive historical year cannot distort the forecast.
_MIN_GROWTH = -0.05
_MAX_GROWTH = 0.30
_DEFAULT_GROWTH = 0.08
_DEFAULT_TAX_RATE = 0.25
_MIN_TAX_RATE = 0.0
_MAX_TAX_RATE = 0.50
_AVERAGING_WINDOW = 5

_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("totalrevenue", "revenue", "operatingrevenue"),
    "ebit": ("ebit",),
    "operating_income": ("operatingincome",),
    "pretax_income": ("pretaxincome", "incomebeforetax", "ebt"),
    "tax_provision": ("taxprovision", "incometaxexpensebenefit", "incometaxexpense"),
    "current_assets": ("currentassets", "totalcurrentassets"),
    "current_liabilities": ("currentliabilities", "totalcurrentliabilities"),
    "capex": ("capitalexpenditure", "capex", "purchaseofppe"),
    "free_cash_flow": ("freecashflow",),
    "depreciation": (
        "depreciationandamortization",
        "depreciationamortizationdepletion",
        "depreciation",
    ),
    "total_debt": ("totaldebt", "longtermdebt"),
    "cash": ("cashandcashequivalents", "cashcashequivalentsandshortterminvestments"),
}


def _normalize(label: object) -> str:
    return "".join(ch for ch in str(label).casefold() if ch.isalnum())


def _timeseries_frame(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Reshape a statement to metrics-as-index, columns ordered oldest to newest.

    Accepts both the service long-form frame (with a ``metric`` period column)
    and an already engine-shaped frame (metrics as index).
    """
    if frame is None or frame.empty:
        return pd.DataFrame()

    if "metric" in frame.columns:
        working = frame.copy()
        periods = pd.to_datetime(working["metric"], errors="coerce")
        line_items = working.drop(
            columns=[column for column in ("metric", "ticker") if column in working.columns]
        )
        transposed = line_items.T
        transposed.columns = periods.to_numpy()
        transposed = transposed.loc[:, transposed.columns.notna()]
        return transposed.reindex(sorted(transposed.columns), axis=1)

    working = frame.copy()
    parsed = pd.to_datetime(pd.Series(list(working.columns)), errors="coerce")
    if parsed.notna().all():
        ordered = [column for _, column in sorted(zip(parsed, working.columns))]
        working = working.reindex(ordered, axis=1)
    return working


def _row(frame: pd.DataFrame, key: str) -> list[float]:
    """Return an oldest-to-newest numeric series for the first matching row."""
    if frame.empty:
        return []
    aliases = {_normalize(alias) for alias in _ALIASES[key]}
    for label in frame.index:
        if _normalize(label) in aliases:
            return [fm.to_float(value) or 0.0 for value in frame.loc[label].tolist()]
    for label in frame.index:
        normalized = _normalize(label)
        if any(alias in normalized for alias in aliases):
            return [fm.to_float(value) or 0.0 for value in frame.loc[label].tolist()]
    return []


def _ratio_series(numerator: list[float], denominator: list[float]) -> list[float]:
    ratios: list[float] = []
    for num, den in zip(numerator, denominator):
        value = fm.safe_divide(num, den)
        if value is not None:
            ratios.append(value)
    return ratios


class ForecastEngine:
    """Estimate assumptions and project free cash flows from history alone."""

    def __init__(self, forecast_years: int = 10) -> None:
        self.forecast_years = forecast_years

    def build_history(
        self,
        income_statement: Optional[pd.DataFrame],
        balance_sheet: Optional[pd.DataFrame],
        cash_flow: Optional[pd.DataFrame],
    ) -> HistoricalAnalysis:
        """Extract the historical figures needed to seed the forecast."""
        income = _timeseries_frame(income_statement)
        balance = _timeseries_frame(balance_sheet)
        cash = _timeseries_frame(cash_flow)

        revenue = _row(income, "revenue")
        ebit = _row(income, "ebit") or _row(income, "operating_income")
        pretax = _row(income, "pretax_income")
        tax = _row(income, "tax_provision")
        current_assets = _row(balance, "current_assets")
        current_liabilities = _row(balance, "current_liabilities")
        capex = [abs(value) for value in _row(cash, "capex")]
        depreciation = [abs(value) for value in _row(cash, "depreciation")]
        free_cash_flow = _row(cash, "free_cash_flow")

        working_capital = [
            assets - liabilities
            for assets, liabilities in zip(current_assets, current_liabilities)
        ]

        operating_margin = _ratio_series(ebit, revenue)
        tax_rate = [
            fm.clamp(rate, _MIN_TAX_RATE, _MAX_TAX_RATE)
            for rate in _ratio_series(tax, pretax)
        ]
        capex_to_revenue = _ratio_series(capex, revenue)
        working_capital_to_revenue = _ratio_series(working_capital, revenue)
        depreciation_to_revenue = _ratio_series(depreciation, revenue)
        fcf_margin = _ratio_series(free_cash_flow, revenue)

        debt = _row(balance, "total_debt")
        cash_balance = _row(balance, "cash")

        return HistoricalAnalysis(
            years=len(revenue),
            revenue=revenue,
            ebit=ebit,
            operating_margin=operating_margin,
            tax_rate=tax_rate,
            capex_to_revenue=capex_to_revenue,
            working_capital_to_revenue=working_capital_to_revenue,
            depreciation_to_revenue=depreciation_to_revenue,
            free_cash_flow=free_cash_flow,
            fcf_margin=fcf_margin,
            latest_revenue=revenue[-1] if revenue else None,
            latest_working_capital=working_capital[-1] if working_capital else None,
            total_debt=debt[-1] if debt else None,
            cash=cash_balance[-1] if cash_balance else None,
        )

    def estimate_revenue_growth(self, history: HistoricalAnalysis) -> float:
        """Weighted blend of 3/5/10-year revenue CAGR (recent years weighted)."""
        periods = sorted(_GROWTH_WEIGHTS)
        cagrs = [fm.series_cagr(history.revenue, period) for period in periods]
        weights = [_GROWTH_WEIGHTS[period] for period in periods]
        blended = fm.weighted_average(cagrs, weights)
        if blended is None:
            return _DEFAULT_GROWTH
        return fm.clamp(blended, _MIN_GROWTH, _MAX_GROWTH)

    @staticmethod
    def _recent_mean(values: list[float], default: float) -> float:
        window = values[-_AVERAGING_WINDOW:]
        average = fm.mean(window)
        return average if average is not None else default

    def estimate_operating_assumptions(
        self, history: HistoricalAnalysis
    ) -> dict[str, float]:
        """Estimate margins/ratios as recent historical averages."""
        return {
            "operating_margin": self._recent_mean(history.operating_margin, 0.15),
            "tax_rate": self._recent_mean(history.tax_rate, _DEFAULT_TAX_RATE),
            "capex_to_revenue": self._recent_mean(history.capex_to_revenue, 0.04),
            "working_capital_to_revenue": self._recent_mean(
                history.working_capital_to_revenue, 0.0
            ),
            "depreciation_to_revenue": self._recent_mean(
                history.depreciation_to_revenue, 0.0
            ),
        }

    def project(
        self, history: HistoricalAnalysis, assumptions: ForecastAssumptions
    ) -> list[ForecastYear]:
        """Project revenue, EBIT, NOPAT, CapEx, WC and FCF over the horizon."""
        base_revenue = history.latest_revenue
        if base_revenue is None or base_revenue <= 0.0:
            return []

        prior_working_capital = (
            history.latest_working_capital
            if history.latest_working_capital is not None
            else base_revenue * assumptions.working_capital_to_revenue
        )

        forecast: list[ForecastYear] = []
        revenue = base_revenue
        for period in range(1, assumptions.forecast_years + 1):
            revenue = revenue * (1.0 + assumptions.revenue_growth)
            ebit = revenue * assumptions.operating_margin
            nopat = ebit * (1.0 - assumptions.tax_rate)
            capex = revenue * assumptions.capex_to_revenue
            depreciation = revenue * assumptions.depreciation_to_revenue
            working_capital = revenue * assumptions.working_capital_to_revenue
            change_in_working_capital = working_capital - prior_working_capital
            prior_working_capital = working_capital

            free_cash_flow = nopat + depreciation - capex - change_in_working_capital
            factor = fm.discount_factor(assumptions.discount_rate, period)

            forecast.append(
                ForecastYear(
                    year=period,
                    revenue=revenue,
                    ebit=ebit,
                    nopat=nopat,
                    capex=capex,
                    working_capital=working_capital,
                    change_in_working_capital=change_in_working_capital,
                    free_cash_flow=free_cash_flow,
                    discount_factor=factor,
                    present_value=free_cash_flow * factor,
                )
            )
        return forecast


__all__ = ["ForecastEngine"]
