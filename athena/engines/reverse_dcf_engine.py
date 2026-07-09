"""ATHENA's Reverse DCF Engine.

A forward DCF estimates intrinsic value from assumptions. This engine inverts
that process: holding the operating profile fixed, it solves for the revenue
growth (and, independently, the operating margin) that a discounted-cash-flow
model would need in order to reproduce the *current market price*. Those implied
figures are then compared with what the business has actually delivered, yielding
a gap analysis, an expectation score and a plain-language verdict.

The engine reuses ATHENA's existing valuation machinery
(:class:`~athena.engines.forecast_engine.ForecastEngine` and
:mod:`athena.utils.finance_math`) and consumes objects produced elsewhere
(:class:`~athena.engines.ratio_engine.FinancialData` /
:class:`~athena.engines.ratio_engine.RatioResult`,
:class:`~athena.models.valuation_models.DCFResult` and
:class:`~athena.models.business_quality_models.BusinessQualityResult`). It never
mutates its inputs and contains no Streamlit or presentation code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable, Optional

from athena.engines.forecast_engine import ForecastEngine
from athena.engines.ratio_engine import FinancialData, RatioResult
from athena.models.business_quality_models import BusinessQualityResult
from athena.models.reverse_dcf_models import (
    ExpectationLevel,
    GapAnalysis,
    HistoricalReality,
    ImpliedExpectations,
    ReverseDCFResult,
    ValuationRisk,
)
from athena.models.valuation_models import DCFResult, ForecastAssumptions, HistoricalAnalysis
from athena.utils import finance_math as fm

logger = logging.getLogger(__name__)

# Search bounds and tolerance for the root-finding solvers.
_MIN_SOLVE_GROWTH = -0.20
_MAX_SOLVE_GROWTH = 0.60
_MIN_SOLVE_MARGIN = 0.005
_MAX_SOLVE_MARGIN = 0.80
_MAX_ITERATIONS = 100
_RELATIVE_TOLERANCE = 1e-4

_NEUTRAL_SCORE = 50.0

# Fallback assumptions used only when no DCFResult is supplied.
_DEFAULT_DISCOUNT_RATE = 0.11
_DEFAULT_TERMINAL_GROWTH = 0.04
_DEFAULT_FORECAST_YEARS = 10

# Expectation-score calibration: score = base + gap * slope (per unit of gap).
_SCORE_BASE = 25.0
_GROWTH_GAP_SLOPE = 550.0
_MARGIN_GAP_SLOPE = 200.0

# Expectation-level cut-offs on the 0-100 expectation score.
_LEVEL_BANDS: tuple[tuple[float, ExpectationLevel], ...] = (
    (80.0, ExpectationLevel.EXTREME),
    (55.0, ExpectationLevel.AGGRESSIVE),
    (30.0, ExpectationLevel.MODERATE),
)

# Valuation-risk cut-offs on the 0-100 expectation score.
_RISK_BANDS: tuple[tuple[float, ValuationRisk], ...] = (
    (70.0, ValuationRisk.HIGH),
    (40.0, ValuationRisk.MEDIUM),
)


@dataclass(frozen=True)
class ReverseDCFInput:
    """Everything the engine needs; all fields come from existing ATHENA layers."""

    current_price: Optional[float]
    shares_outstanding: Optional[float]
    cash: Optional[float]
    debt: Optional[float]
    financial_data: FinancialData
    ratios: RatioResult
    business_quality: Optional[BusinessQualityResult] = None
    dcf: Optional[DCFResult] = None
    ticker: Optional[str] = None


class ReverseDCFEngine:
    """Solve for the market-implied assumptions behind the current price."""

    def __init__(self, forecast_engine: Optional[ForecastEngine] = None) -> None:
        self._forecast_engine = forecast_engine or ForecastEngine()

    # ------------------------------------------------------------------ public
    def analyze(self, data: ReverseDCFInput) -> ReverseDCFResult:
        """Run the reverse-DCF analysis and return a :class:`ReverseDCFResult`."""
        if not isinstance(data, ReverseDCFInput):
            raise TypeError("data must be an instance of ReverseDCFInput")

        notes: list[str] = []
        history = self._forecast_engine.build_history(
            data.financial_data.income_statement,
            data.financial_data.balance_sheet,
            data.financial_data.cash_flow,
        )
        assumptions = self._baseline_assumptions(data, history, notes)

        price = fm.to_float(data.current_price)
        shares = fm.to_float(data.shares_outstanding) or self._shares_from_dcf(data)
        debt = fm.to_float(data.debt)
        cash = fm.to_float(data.cash)
        debt = debt if debt is not None else (history.total_debt or 0.0)
        cash = cash if cash is not None else (history.cash or 0.0)

        historical = self._historical_reality(data.ratios, history)

        if price is None or shares is None or shares <= 0.0 or not history.revenue:
            notes.append("Insufficient price/share/statement data to invert the DCF.")
            return self._empty_result(assumptions, historical, notes)

        required_growth, growth_capped = self._solve_growth(
            history, assumptions, shares, debt, cash, price
        )
        required_margin = self._solve_margin(
            history, assumptions, shares, debt, cash, price, historical.historical_growth
        )
        required_fcf_margin = self._implied_fcf_margin(
            history, assumptions, required_growth, shares, debt, cash
        )
        required_roic = self._implied_roic(required_margin, assumptions.tax_rate, data.ratios)

        implied = ImpliedExpectations(
            required_growth=required_growth,
            required_ebit_margin=required_margin,
            required_fcf_margin=required_fcf_margin,
            required_roic=required_roic,
            required_terminal_growth=assumptions.terminal_growth,
            required_discount_rate=assumptions.discount_rate,
        )

        gap = self._gap_analysis(implied, historical)
        expectation_score = self._expectation_score(gap, growth_capped)
        expectation_level = self._expectation_level(expectation_score)
        valuation_risk = self._valuation_risk(expectation_score, data.business_quality)
        ai_summary = self._ai_summary(
            price=price,
            implied=implied,
            historical=historical,
            gap=gap,
            level=expectation_level,
            risk=valuation_risk,
            growth_capped=growth_capped,
            business_quality=data.business_quality,
        )

        return ReverseDCFResult(
            required_growth=required_growth,
            required_margin=required_margin,
            required_roic=required_roic,
            expectation_score=round(expectation_score, 2),
            expectation_level=expectation_level,
            valuation_risk=valuation_risk,
            historical_growth=historical.historical_growth,
            historical_margin=historical.historical_margin,
            difference=gap.growth_gap,
            ai_summary=ai_summary,
            implied=implied,
            historical=historical,
            gap=gap,
            growth_capped=growth_capped,
            notes=notes,
        )

    # ------------------------------------------------------------- assumptions
    def _baseline_assumptions(
        self, data: ReverseDCFInput, history: HistoricalAnalysis, notes: list[str]
    ) -> ForecastAssumptions:
        """Fix the operating profile (margins/rates) that the solver holds constant."""
        if data.dcf is not None:
            return data.dcf.assumptions

        notes.append("No DCFResult supplied; using estimated operating assumptions.")
        operating = self._forecast_engine.estimate_operating_assumptions(history)
        return ForecastAssumptions(
            revenue_growth=self._forecast_engine.estimate_revenue_growth(history),
            operating_margin=operating["operating_margin"],
            tax_rate=operating["tax_rate"],
            capex_to_revenue=operating["capex_to_revenue"],
            working_capital_to_revenue=operating["working_capital_to_revenue"],
            depreciation_to_revenue=operating["depreciation_to_revenue"],
            forecast_years=_DEFAULT_FORECAST_YEARS,
            discount_rate=_DEFAULT_DISCOUNT_RATE,
            terminal_growth=_DEFAULT_TERMINAL_GROWTH,
        )

    # -------------------------------------------------------------- valuation
    def _intrinsic_per_share(
        self,
        history: HistoricalAnalysis,
        assumptions: ForecastAssumptions,
        shares: float,
        debt: float,
        cash: float,
    ) -> Optional[float]:
        """Discounted-cash-flow intrinsic value per share for given assumptions."""
        forecast = self._forecast_engine.project(history, assumptions)
        if not forecast:
            return None
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
        equity_value = pv_explicit + pv_terminal + cash - debt
        return fm.safe_divide(equity_value, shares)

    # ---------------------------------------------------------------- solvers
    def _solve_growth(
        self,
        history: HistoricalAnalysis,
        assumptions: ForecastAssumptions,
        shares: float,
        debt: float,
        cash: float,
        target_price: float,
    ) -> tuple[Optional[float], bool]:
        """Solve for the revenue CAGR implied by ``target_price``.

        Returns the implied growth and a flag indicating the price is so
        demanding that even the maximum searched growth cannot justify it.
        """

        def value_at(growth: float) -> Optional[float]:
            return self._intrinsic_per_share(
                history, replace(assumptions, revenue_growth=growth), shares, debt, cash
            )

        low_value = value_at(_MIN_SOLVE_GROWTH)
        high_value = value_at(_MAX_SOLVE_GROWTH)
        if low_value is None or high_value is None:
            return None, False
        if target_price <= low_value:
            return _MIN_SOLVE_GROWTH, False
        if target_price >= high_value:
            return _MAX_SOLVE_GROWTH, True

        solved = self._bisect(value_at, _MIN_SOLVE_GROWTH, _MAX_SOLVE_GROWTH, target_price)
        return solved, False

    def _solve_margin(
        self,
        history: HistoricalAnalysis,
        assumptions: ForecastAssumptions,
        shares: float,
        debt: float,
        cash: float,
        target_price: float,
        historical_growth: Optional[float],
    ) -> Optional[float]:
        """Solve for the operating margin implied by ``target_price``.

        Growth is pinned to the historical CAGR so the margin lever alone must
        reconcile the price - this isolates the profitability the market expects.
        """
        growth = historical_growth if historical_growth is not None else assumptions.revenue_growth
        pinned = replace(assumptions, revenue_growth=fm.clamp(growth, _MIN_SOLVE_GROWTH, _MAX_SOLVE_GROWTH))

        def value_at(margin: float) -> Optional[float]:
            return self._intrinsic_per_share(
                history, replace(pinned, operating_margin=margin), shares, debt, cash
            )

        low_value = value_at(_MIN_SOLVE_MARGIN)
        high_value = value_at(_MAX_SOLVE_MARGIN)
        if low_value is None or high_value is None:
            return None
        if target_price <= low_value:
            return _MIN_SOLVE_MARGIN
        if target_price >= high_value:
            return _MAX_SOLVE_MARGIN
        return self._bisect(value_at, _MIN_SOLVE_MARGIN, _MAX_SOLVE_MARGIN, target_price)

    @staticmethod
    def _bisect(
        value_at: Callable[[float], Optional[float]],
        low: float,
        high: float,
        target: float,
    ) -> Optional[float]:
        """Bisection root-find for a value function that increases with its input."""
        for _ in range(_MAX_ITERATIONS):
            midpoint = (low + high) / 2.0
            value = value_at(midpoint)
            if value is None:
                return None
            if abs(value - target) <= _RELATIVE_TOLERANCE * max(1.0, abs(target)):
                return midpoint
            if value < target:
                low = midpoint
            else:
                high = midpoint
        return (low + high) / 2.0

    def _implied_fcf_margin(
        self,
        history: HistoricalAnalysis,
        assumptions: ForecastAssumptions,
        required_growth: Optional[float],
        shares: float,
        debt: float,
        cash: float,
    ) -> Optional[float]:
        """First-year free-cash-flow margin under the implied-growth forecast."""
        if required_growth is None:
            return None
        forecast = self._forecast_engine.project(
            history, replace(assumptions, revenue_growth=required_growth)
        )
        if not forecast:
            return None
        first = forecast[0]
        return fm.safe_divide(first.free_cash_flow, first.revenue)

    @staticmethod
    def _implied_roic(
        required_margin: Optional[float],
        tax_rate: float,
        ratios: RatioResult,
    ) -> Optional[float]:
        """Implied ROIC = NOPAT margin x capital turnover (asset turnover proxy)."""
        if required_margin is None or ratios.asset_turnover is None:
            return None
        nopat_margin = required_margin * (1.0 - tax_rate)
        return nopat_margin * ratios.asset_turnover

    # -------------------------------------------------------------- historical
    @staticmethod
    def _historical_reality(ratios: RatioResult, history: HistoricalAnalysis) -> HistoricalReality:
        historical_growth = ratios.revenue_cagr
        if historical_growth is None:
            historical_growth = fm.series_cagr(history.revenue, len(history.revenue))
        historical_fcf_margin = history.fcf_margin[-1] if history.fcf_margin else None
        return HistoricalReality(
            historical_growth=historical_growth,
            historical_margin=ratios.operating_margin,
            historical_fcf_margin=historical_fcf_margin,
            historical_roic=ratios.roce,  # ROCE is ATHENA's practical ROIC proxy.
            historical_roe=ratios.roe,
            historical_roce=ratios.roce,
        )

    # ------------------------------------------------------------- gap / score
    @staticmethod
    def _gap_analysis(implied: ImpliedExpectations, historical: HistoricalReality) -> GapAnalysis:
        return GapAnalysis(
            growth_gap=_difference(implied.required_growth, historical.historical_growth),
            margin_gap=_difference(implied.required_ebit_margin, historical.historical_margin),
            roic_gap=_difference(implied.required_roic, historical.historical_roic),
        )

    @staticmethod
    def _expectation_score(gap: GapAnalysis, growth_capped: bool) -> float:
        if growth_capped:
            return 100.0
        if gap.growth_gap is None:
            return _NEUTRAL_SCORE
        score = _SCORE_BASE + gap.growth_gap * _GROWTH_GAP_SLOPE
        if gap.margin_gap is not None:
            score += gap.margin_gap * _MARGIN_GAP_SLOPE
        return fm.clamp(score, 0.0, 100.0)

    @staticmethod
    def _expectation_level(score: float) -> ExpectationLevel:
        for threshold, level in _LEVEL_BANDS:
            if score >= threshold:
                return level
        return ExpectationLevel.LOW

    @staticmethod
    def _valuation_risk(score: float, business_quality: Optional[BusinessQualityResult]) -> ValuationRisk:
        risk = ValuationRisk.LOW
        for threshold, band in _RISK_BANDS:
            if score >= threshold:
                risk = band
                break
        # A demanding price is riskier when the underlying business is weak.
        if business_quality is not None and business_quality.overall_score < 50.0 and risk is ValuationRisk.MEDIUM:
            risk = ValuationRisk.HIGH
        return risk

    # ---------------------------------------------------------------- narrative
    @staticmethod
    def _ai_summary(
        *,
        price: float,
        implied: ImpliedExpectations,
        historical: HistoricalReality,
        gap: GapAnalysis,
        level: ExpectationLevel,
        risk: ValuationRisk,
        growth_capped: bool,
        business_quality: Optional[BusinessQualityResult],
    ) -> str:
        parts: list[str] = []
        if implied.required_growth is not None:
            growth_text = f"{implied.required_growth * 100:.1f}%"
            if growth_capped:
                growth_text = f"more than {implied.required_growth * 100:.0f}%"
            parts.append(f"At {price:,.0f}, the market implies revenue growth of {growth_text}.")
        if historical.historical_growth is not None:
            parts.append(f"The business has historically grown {historical.historical_growth * 100:.1f}%.")
        if gap.growth_gap is not None:
            if gap.growth_gap > 0.0:
                parts.append(
                    f"That is {gap.growth_gap * 100:.1f} percentage points above its track record - "
                    f"a {level.value.lower()} expectation."
                )
            else:
                parts.append(
                    f"That is at or below its track record ({abs(gap.growth_gap) * 100:.1f} pts lower) - "
                    f"a {level.value.lower()} expectation."
                )
        if level in (ExpectationLevel.AGGRESSIVE, ExpectationLevel.EXTREME):
            parts.append("The current valuation requires performance well above historical reality.")
        elif level is ExpectationLevel.LOW:
            parts.append("The current valuation appears undemanding relative to history.")
        if business_quality is not None:
            parts.append(
                f"Business quality is {business_quality.grade.value} "
                f"({business_quality.overall_score:.0f}/100)."
            )
        parts.append(f"Valuation risk: {risk.value}.")
        return " ".join(parts)

    # ------------------------------------------------------------------ helpers
    def _empty_result(
        self,
        assumptions: ForecastAssumptions,
        historical: HistoricalReality,
        notes: list[str],
    ) -> ReverseDCFResult:
        implied = ImpliedExpectations(
            required_growth=None,
            required_ebit_margin=None,
            required_fcf_margin=None,
            required_roic=None,
            required_terminal_growth=assumptions.terminal_growth,
            required_discount_rate=assumptions.discount_rate,
        )
        gap = GapAnalysis(growth_gap=None, margin_gap=None, roic_gap=None)
        return ReverseDCFResult(
            required_growth=None,
            required_margin=None,
            required_roic=None,
            expectation_score=_NEUTRAL_SCORE,
            expectation_level=ExpectationLevel.MODERATE,
            valuation_risk=ValuationRisk.MEDIUM,
            historical_growth=historical.historical_growth,
            historical_margin=historical.historical_margin,
            difference=None,
            ai_summary="Insufficient data to infer market-implied expectations.",
            implied=implied,
            historical=historical,
            gap=gap,
            growth_capped=False,
            notes=notes,
        )

    @staticmethod
    def _shares_from_dcf(data: ReverseDCFInput) -> Optional[float]:
        if data.dcf is not None and data.dcf.equity_value is not None and data.dcf.intrinsic_value:
            return fm.safe_divide(data.dcf.equity_value, data.dcf.intrinsic_value)
        return None


def _difference(implied: Optional[float], historical: Optional[float]) -> Optional[float]:
    if implied is None or historical is None:
        return None
    return implied - historical


__all__ = ["ReverseDCFEngine", "ReverseDCFInput"]
