"""ATHENA's automatic discounted-cash-flow valuation engine.

The engine estimates every assumption from historical statements - the user
never supplies growth, discount rate, terminal growth or any other input. Given
a company's financials it returns a fully populated
:class:`~athena.models.dcf_models.DCFResult` including scenario and sensitivity
analysis and an investment recommendation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from athena.engines.ratio_engine import RatioResult
from athena.models.dcf_models import (
    DCFResult,
    ForecastAssumptions,
    ForecastYear,
    HistoricalAnalysis,
    MarketCapCategory,
    Recommendation,
    ScenarioResult,
)
from athena.utils import finance_math as fm
from athena.utils.forecast_engine import ForecastEngine

logger = logging.getLogger(__name__)

# Indian market-cap thresholds (in rupees): >= ₹20,000 Cr large, >= ₹5,000 Cr mid.
_LARGE_CAP_FLOOR = 20_000e7
_MID_CAP_FLOOR = 5_000e7

_BASE_DISCOUNT_RATE: dict[MarketCapCategory, float] = {
    MarketCapCategory.LARGE: 0.10,
    MarketCapCategory.MID: 0.11,
    MarketCapCategory.SMALL: 0.12,
}
_MIN_DISCOUNT_RATE = 0.08
_MAX_DISCOUNT_RATE = 0.15

# Terminal growth is bounded by a nominal GDP assumption.
_NOMINAL_GDP_CEILING = 0.05
_TERMINAL_SLOW = 0.03
_TERMINAL_STABLE = 0.04
_TERMINAL_FAST = 0.05
_FAST_GROWTH_THRESHOLD = 0.15
_SLOW_GROWTH_THRESHOLD = 0.05

_HIGH_DEBT_EQUITY = 1.0
_STRONG_RETURN_THRESHOLD = 0.20

_SCENARIO_GROWTH_DELTA = 0.03
_SCENARIO_RATE_DELTA = 0.01
_SCENARIO_TERMINAL_DELTA = 0.01

_EXPECTED_CAGR_HORIZON = 5

_RECOMMENDATION_ORDER: tuple[Recommendation, ...] = (
    Recommendation.STRONG_BUY,
    Recommendation.BUY,
    Recommendation.HOLD,
    Recommendation.REDUCE,
    Recommendation.SELL,
)


@dataclass(frozen=True)
class DCFInput:
    """All data the DCF engine needs; every field is sourced automatically."""

    income_statement: Optional[pd.DataFrame] = None
    balance_sheet: Optional[pd.DataFrame] = None
    cash_flow: Optional[pd.DataFrame] = None
    fast_info: Optional[dict[str, Any]] = None
    ratios: Optional[RatioResult] = None
    current_price: Optional[float] = None
    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None
    ticker: Optional[str] = None


@dataclass(frozen=True)
class _Valuation:
    """Intermediate valuation output shared by the base case and scenarios."""

    enterprise_value: Optional[float]
    equity_value: Optional[float]
    intrinsic_value: Optional[float]
    forecast: list[ForecastYear]


class DCFEngine:
    """Compute an automatic intrinsic valuation from financial statements."""

    def __init__(self, forecast_years: int = 10) -> None:
        self.forecast_years = forecast_years
        self._forecast_engine = ForecastEngine(forecast_years=forecast_years)

    # ------------------------------------------------------------------ public
    def value(self, data: DCFInput) -> DCFResult:
        """Run the full automatic DCF pipeline and return a :class:`DCFResult`."""
        history = self._forecast_engine.build_history(
            data.income_statement, data.balance_sheet, data.cash_flow
        )
        revenue_growth = self._forecast_engine.estimate_revenue_growth(history)
        operating = self._forecast_engine.estimate_operating_assumptions(history)

        market_cap = self._resolve_market_cap(data)
        discount_rate = self._estimate_discount_rate(data.ratios, market_cap)
        terminal_growth = self._estimate_terminal_growth(revenue_growth, discount_rate)

        assumptions = ForecastAssumptions(
            revenue_growth=revenue_growth,
            operating_margin=operating["operating_margin"],
            tax_rate=operating["tax_rate"],
            capex_to_revenue=operating["capex_to_revenue"],
            working_capital_to_revenue=operating["working_capital_to_revenue"],
            depreciation_to_revenue=operating["depreciation_to_revenue"],
            forecast_years=self.forecast_years,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
        )

        shares = fm.to_float(data.shares_outstanding) or self._shares_from_fast_info(data)
        debt = history.total_debt or 0.0
        cash = history.cash or 0.0

        base = self._value_core(history, assumptions, shares, debt, cash)
        notes: list[str] = []
        if base.intrinsic_value is None:
            notes.append("Insufficient historical data to compute an intrinsic value.")

        scenarios = self._build_scenarios(history, assumptions, shares, debt, cash)
        scenario_by_name = {scenario.name: scenario.intrinsic_value for scenario in scenarios}

        margin_of_safety = fm.safe_divide(
            (base.intrinsic_value - data.current_price)
            if base.intrinsic_value is not None and data.current_price is not None
            else None,
            data.current_price,
        )
        expected_cagr = self._expected_cagr(data.current_price, base.intrinsic_value)
        confidence = self._confidence_score(history, data.ratios)
        sensitivity = self._sensitivity_table(history, assumptions, shares, debt, cash)
        recommendation = self._recommend(
            margin_of_safety=margin_of_safety,
            ratios=data.ratios,
            history=history,
        )

        return DCFResult(
            ticker=data.ticker,
            current_price=fm.to_float(data.current_price),
            intrinsic_value=base.intrinsic_value,
            enterprise_value=base.enterprise_value,
            equity_value=base.equity_value,
            margin_of_safety=margin_of_safety,
            expected_cagr=expected_cagr,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            revenue_growth=revenue_growth,
            confidence_score=confidence,
            bear_value=scenario_by_name.get("Bear"),
            base_value=scenario_by_name.get("Base"),
            bull_value=scenario_by_name.get("Bull"),
            sensitivity_table=sensitivity,
            recommendation=recommendation,
            assumptions=assumptions,
            forecast=base.forecast,
            scenarios=scenarios,
            notes=notes,
        )

    # -------------------------------------------------------------- valuation
    def _value_core(
        self,
        history: HistoricalAnalysis,
        assumptions: ForecastAssumptions,
        shares: Optional[float],
        debt: float,
        cash: float,
    ) -> _Valuation:
        forecast = self._forecast_engine.project(history, assumptions)
        if not forecast:
            return _Valuation(None, None, None, [])

        pv_explicit = sum(year.present_value for year in forecast)
        terminal_value = fm.gordon_terminal_value(
            forecast[-1].free_cash_flow,
            assumptions.discount_rate,
            assumptions.terminal_growth,
        )
        pv_terminal = (
            fm.present_value(terminal_value, assumptions.discount_rate, assumptions.forecast_years)
            if terminal_value is not None
            else 0.0
        )

        enterprise_value = pv_explicit + pv_terminal
        equity_value = enterprise_value + cash - debt
        intrinsic_value = fm.safe_divide(equity_value, shares)
        return _Valuation(enterprise_value, equity_value, intrinsic_value, forecast)

    def _build_scenarios(
        self,
        history: HistoricalAnalysis,
        assumptions: ForecastAssumptions,
        shares: Optional[float],
        debt: float,
        cash: float,
    ) -> list[ScenarioResult]:
        definitions = {
            "Bear": (
                assumptions.revenue_growth - _SCENARIO_GROWTH_DELTA,
                assumptions.discount_rate + _SCENARIO_RATE_DELTA,
                assumptions.terminal_growth - _SCENARIO_TERMINAL_DELTA,
            ),
            "Base": (
                assumptions.revenue_growth,
                assumptions.discount_rate,
                assumptions.terminal_growth,
            ),
            "Bull": (
                assumptions.revenue_growth + _SCENARIO_GROWTH_DELTA,
                assumptions.discount_rate - _SCENARIO_RATE_DELTA,
                assumptions.terminal_growth + _SCENARIO_TERMINAL_DELTA,
            ),
        }

        scenarios: list[ScenarioResult] = []
        for name, (growth, rate, terminal) in definitions.items():
            rate = fm.clamp(rate, _MIN_DISCOUNT_RATE, _MAX_DISCOUNT_RATE)
            terminal = self._bound_terminal_growth(terminal, rate)
            scenario_assumptions = self._with_overrides(assumptions, growth, rate, terminal)
            valuation = self._value_core(history, scenario_assumptions, shares, debt, cash)
            scenarios.append(
                ScenarioResult(
                    name=name,
                    intrinsic_value=valuation.intrinsic_value,
                    revenue_growth=growth,
                    discount_rate=rate,
                    terminal_growth=terminal,
                )
            )
        return scenarios

    def _sensitivity_table(
        self,
        history: HistoricalAnalysis,
        assumptions: ForecastAssumptions,
        shares: Optional[float],
        debt: float,
        cash: float,
    ) -> dict[str, dict[str, Optional[float]]]:
        rate_offsets = (-0.01, -0.005, 0.0, 0.005, 0.01)
        terminal_offsets = (-0.01, 0.0, 0.01)

        table: dict[str, dict[str, Optional[float]]] = {}
        for rate_offset in rate_offsets:
            rate = fm.clamp(
                assumptions.discount_rate + rate_offset, _MIN_DISCOUNT_RATE, _MAX_DISCOUNT_RATE
            )
            row_key = f"{rate * 100:.1f}%"
            row: dict[str, Optional[float]] = {}
            for terminal_offset in terminal_offsets:
                terminal = self._bound_terminal_growth(
                    assumptions.terminal_growth + terminal_offset, rate
                )
                column_key = f"{terminal * 100:.1f}%"
                scenario_assumptions = self._with_overrides(
                    assumptions, assumptions.revenue_growth, rate, terminal
                )
                valuation = self._value_core(history, scenario_assumptions, shares, debt, cash)
                row[column_key] = valuation.intrinsic_value
            table[row_key] = row
        return table

    @staticmethod
    def _with_overrides(
        assumptions: ForecastAssumptions,
        revenue_growth: float,
        discount_rate: float,
        terminal_growth: float,
    ) -> ForecastAssumptions:
        return ForecastAssumptions(
            revenue_growth=revenue_growth,
            operating_margin=assumptions.operating_margin,
            tax_rate=assumptions.tax_rate,
            capex_to_revenue=assumptions.capex_to_revenue,
            working_capital_to_revenue=assumptions.working_capital_to_revenue,
            depreciation_to_revenue=assumptions.depreciation_to_revenue,
            forecast_years=assumptions.forecast_years,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
        )

    # ------------------------------------------------------------ assumptions
    def _estimate_discount_rate(
        self, ratios: Optional[RatioResult], market_cap: Optional[float]
    ) -> float:
        category = self._market_cap_category(market_cap)
        rate = _BASE_DISCOUNT_RATE[category]

        if ratios is not None:
            if ratios.debt_equity is not None and ratios.debt_equity > _HIGH_DEBT_EQUITY:
                rate += 0.01
            if ratios.roe is not None and ratios.roe > _STRONG_RETURN_THRESHOLD:
                rate -= 0.005
            if ratios.roce is not None and ratios.roce > _STRONG_RETURN_THRESHOLD:
                rate -= 0.005

        return fm.clamp(rate, _MIN_DISCOUNT_RATE, _MAX_DISCOUNT_RATE)

    @staticmethod
    def _market_cap_category(market_cap: Optional[float]) -> MarketCapCategory:
        value = fm.to_float(market_cap)
        if value is None:
            return MarketCapCategory.MID
        if value >= _LARGE_CAP_FLOOR:
            return MarketCapCategory.LARGE
        if value >= _MID_CAP_FLOOR:
            return MarketCapCategory.MID
        return MarketCapCategory.SMALL

    def _estimate_terminal_growth(self, revenue_growth: float, discount_rate: float) -> float:
        if revenue_growth > _FAST_GROWTH_THRESHOLD:
            terminal = _TERMINAL_FAST
        elif revenue_growth < _SLOW_GROWTH_THRESHOLD:
            terminal = _TERMINAL_SLOW
        else:
            terminal = _TERMINAL_STABLE
        return self._bound_terminal_growth(terminal, discount_rate)

    @staticmethod
    def _bound_terminal_growth(terminal_growth: float, discount_rate: float) -> float:
        capped = min(terminal_growth, _NOMINAL_GDP_CEILING)
        # Terminal growth must stay safely below the discount rate.
        return max(0.0, min(capped, discount_rate - 0.01))

    # ---------------------------------------------------------------- scoring
    def _confidence_score(
        self, history: HistoricalAnalysis, ratios: Optional[RatioResult]
    ) -> float:
        health = (
            ratios.financial_health_score
            if ratios is not None and ratios.financial_health_score is not None
            else 50.0
        )
        debt_component = self._debt_confidence(ratios)
        cash_flow_component = self._stability_confidence(history.free_cash_flow)
        revenue_component = self._stability_confidence(history.revenue)
        margin_component = self._stability_confidence(history.operating_margin)

        components = [
            (health, 0.30),
            (debt_component, 0.20),
            (cash_flow_component, 0.20),
            (revenue_component, 0.15),
            (margin_component, 0.15),
        ]
        total = sum(score * weight for score, weight in components)
        return round(fm.clamp(total, 0.0, 100.0), 2)

    @staticmethod
    def _debt_confidence(ratios: Optional[RatioResult]) -> float:
        if ratios is None or ratios.debt_equity is None:
            return 50.0
        # 0 debt -> 100, debt/equity >= 2 -> 0.
        return fm.clamp(100.0 * (1.0 - ratios.debt_equity / 2.0), 0.0, 100.0)

    @staticmethod
    def _stability_confidence(values: list[float]) -> float:
        cv = fm.coefficient_of_variation(values)
        if cv is None:
            return 50.0
        return fm.clamp(100.0 * (1.0 - cv), 0.0, 100.0)

    def _expected_cagr(
        self, current_price: Optional[float], intrinsic_value: Optional[float]
    ) -> Optional[float]:
        return fm.cagr(current_price, intrinsic_value, _EXPECTED_CAGR_HORIZON)

    # -------------------------------------------------------- recommendation
    def _recommend(
        self,
        *,
        margin_of_safety: Optional[float],
        ratios: Optional[RatioResult],
        history: HistoricalAnalysis,
    ) -> Recommendation:
        if margin_of_safety is None:
            return Recommendation.HOLD

        if margin_of_safety >= 0.35:
            base = Recommendation.STRONG_BUY
        elif margin_of_safety >= 0.15:
            base = Recommendation.BUY
        elif margin_of_safety >= -0.10:
            base = Recommendation.HOLD
        elif margin_of_safety >= -0.25:
            base = Recommendation.REDUCE
        else:
            base = Recommendation.SELL

        downgrades = 0
        if ratios is not None:
            if ratios.financial_health_score is not None and ratios.financial_health_score < 40.0:
                downgrades += 1
            if ratios.debt_equity is not None and ratios.debt_equity > 1.5:
                downgrades += 1
        if history.free_cash_flow and history.free_cash_flow[-1] < 0.0:
            downgrades += 1

        return self._apply_downgrades(base, downgrades)

    @staticmethod
    def _apply_downgrades(base: Recommendation, downgrades: int) -> Recommendation:
        index = _RECOMMENDATION_ORDER.index(base)
        new_index = min(index + downgrades, len(_RECOMMENDATION_ORDER) - 1)
        return _RECOMMENDATION_ORDER[new_index]

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _resolve_market_cap(data: DCFInput) -> Optional[float]:
        if data.market_cap is not None:
            return fm.to_float(data.market_cap)
        if isinstance(data.fast_info, dict):
            return fm.to_float(data.fast_info.get("market_cap"))
        return None

    @staticmethod
    def _shares_from_fast_info(data: DCFInput) -> Optional[float]:
        if isinstance(data.fast_info, dict):
            return fm.to_float(data.fast_info.get("shares_outstanding"))
        return None


__all__ = ["DCFEngine", "DCFInput"]
