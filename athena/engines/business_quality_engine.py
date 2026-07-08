"""ATHENA's Business Quality Engine.

This engine answers a single question: *is this a fundamentally high-quality
business?* It is deliberately independent of valuation - a wonderful business can
be expensive and a poor business can be cheap. Quality is scored across seven
dimensions (growth, profitability, financial strength, cash-flow quality,
capital allocation, consistency and competitive strength), blended into an
overall 0-100 score with a letter grade, a star rating and a plain-language
explanation of strengths, weaknesses and risks.

The engine consumes the objects produced elsewhere in ATHENA
(:class:`~athena.engines.ratio_engine.FinancialData`,
:class:`~athena.engines.ratio_engine.RatioResult` and, optionally,
:class:`~athena.models.valuation_models.DCFResult`) and never mutates them, so it
is fully compatible with ``FinancialService``, ``FinancialParser``,
``RatioEngine`` and ``DCFEngine``. It contains no Streamlit or presentation code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

from athena.engines.forecast_engine import ForecastEngine
from athena.engines.ratio_engine import FinancialData, RatioResult
from athena.models.business_quality_models import (
    BusinessQualityResult,
    Grade,
    QualityDimension,
    Rating,
)
from athena.models.valuation_models import DCFResult
from athena.utils import finance_math as fm
from athena.utils import statements as st

logger = logging.getLogger(__name__)

# Overall-score weights (must sum to 1.0).
_WEIGHTS: dict[str, float] = {
    "growth": 0.20,
    "profitability": 0.20,
    "financial_strength": 0.20,
    "cashflow": 0.15,
    "capital_allocation": 0.10,
    "consistency": 0.10,
    "competitive_strength": 0.05,
}

_NEUTRAL_SCORE = 50.0

# Bucketed scores shared by every threshold table (Excellent -> Poor).
_EXCELLENT, _GOOD, _AVERAGE, _WEAK, _POOR = 100.0, 80.0, 60.0, 40.0, 20.0

# Growth: annual CAGR thresholds (fractions, e.g. 0.15 == 15%).
_GROWTH_BANDS: tuple[tuple[float, float], ...] = (
    (0.15, _EXCELLENT),
    (0.10, _GOOD),
    (0.05, _AVERAGE),
    (0.0, _WEAK),
)

# "Higher is better" profitability tables (fractions).
_ROE_BANDS = ((0.20, _EXCELLENT), (0.15, _GOOD), (0.10, _AVERAGE), (0.05, _WEAK))
_ROCE_BANDS = _ROE_BANDS
_ROA_BANDS = ((0.10, _EXCELLENT), (0.07, _GOOD), (0.04, _AVERAGE), (0.02, _WEAK))
_GROSS_MARGIN_BANDS = ((0.40, _EXCELLENT), (0.30, _GOOD), (0.20, _AVERAGE), (0.10, _WEAK))
_OPERATING_MARGIN_BANDS = ((0.20, _EXCELLENT), (0.15, _GOOD), (0.10, _AVERAGE), (0.05, _WEAK))
_NET_MARGIN_BANDS = ((0.15, _EXCELLENT), (0.10, _GOOD), (0.07, _AVERAGE), (0.03, _WEAK))

# Financial strength: "higher is better" tables.
_CURRENT_RATIO_BANDS = ((2.0, _EXCELLENT), (1.5, _GOOD), (1.0, _AVERAGE), (0.75, _WEAK))
_QUICK_RATIO_BANDS = ((1.5, _EXCELLENT), (1.0, _GOOD), (0.75, _AVERAGE), (0.5, _WEAK))
_INTEREST_COVERAGE_BANDS = ((10.0, _EXCELLENT), (5.0, _GOOD), (3.0, _AVERAGE), (1.5, _WEAK))
_CASH_RATIO_BANDS = ((0.5, _EXCELLENT), (0.3, _GOOD), (0.15, _AVERAGE), (0.05, _WEAK))

# Financial strength: "lower is better" leverage table (debt / equity).
_DEBT_EQUITY_BANDS = ((0.10, _EXCELLENT), (0.50, _GOOD), (1.0, _AVERAGE), (2.0, _WEAK))

# Cash-flow quality.
_FCF_MARGIN_BANDS = ((0.15, _EXCELLENT), (0.10, _GOOD), (0.05, _AVERAGE), (0.0, _WEAK))
_CASH_CONVERSION_BANDS = ((1.20, _EXCELLENT), (1.0, _GOOD), (0.8, _AVERAGE), (0.5, _WEAK))

# Capital allocation.
_ROIC_BANDS = _ROCE_BANDS
_CAPEX_INTENSITY_BANDS = ((0.03, _EXCELLENT), (0.06, _GOOD), (0.10, _AVERAGE), (0.15, _WEAK))

# Grade cut-offs (overall score >= threshold).
_GRADE_BANDS: tuple[tuple[float, Grade], ...] = (
    (95.0, Grade.A_PLUS),
    (90.0, Grade.A),
    (80.0, Grade.B_PLUS),
    (70.0, Grade.B),
    (60.0, Grade.C),
)

# Star cut-offs (overall score >= threshold -> N filled stars).
_STAR_BANDS: tuple[tuple[float, int], ...] = (
    (80.0, 5),
    (60.0, 4),
    (40.0, 3),
    (20.0, 2),
)

_FILLED_STAR = "\u2605"  # ★
_EMPTY_STAR = "\u2606"  # ☆
_MAX_STARS = 5


@dataclass(frozen=True)
class BusinessQualityInput:
    """Everything the engine needs; all fields come from existing ATHENA layers."""

    financial_data: FinancialData
    ratios: RatioResult
    dcf: Optional[DCFResult] = None
    ticker: Optional[str] = None


@dataclass(frozen=True)
class _SeriesBundle:
    """Historical, oldest-to-newest series used for growth and consistency."""

    revenue: list[float] = field(default_factory=list)
    net_income: list[float] = field(default_factory=list)
    operating_margin: list[float] = field(default_factory=list)
    net_margin: list[float] = field(default_factory=list)
    free_cash_flow: list[float] = field(default_factory=list)
    fcf_margin: list[float] = field(default_factory=list)
    operating_cash_flow: list[float] = field(default_factory=list)
    working_capital_to_revenue: list[float] = field(default_factory=list)
    capex_to_revenue: list[float] = field(default_factory=list)


def _rating_for(score: float) -> Rating:
    """Map a 0-100 score to a qualitative rating band."""
    if score >= _EXCELLENT - 5.0:
        return Rating.EXCELLENT
    if score >= _GOOD - 5.0:
        return Rating.GOOD
    if score >= _AVERAGE - 5.0:
        return Rating.AVERAGE
    if score >= _WEAK - 5.0:
        return Rating.WEAK
    return Rating.POOR


def _band_score(value: Optional[float], bands: Sequence[tuple[float, float]], *, lower_is_better: bool = False) -> Optional[float]:
    """Score a value against descending/ascending threshold bands.

    ``bands`` is ordered from the best threshold to the worst. For
    ``lower_is_better`` tables the thresholds are ascending ceilings; otherwise
    they are descending floors. A value that beats no band scores ``_POOR``.
    """
    number = fm.to_float(value)
    if number is None:
        return None
    if lower_is_better:
        for ceiling, score in bands:
            if number <= ceiling:
                return score
        return _POOR
    for floor, score in bands:
        if number >= floor:
            return score
    return _POOR


def _average(scores: Sequence[Optional[float]]) -> Optional[float]:
    """Mean of the non-``None`` scores, or ``None`` when all are missing."""
    present = [score for score in scores if score is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _stability_score(values: Sequence[float]) -> Optional[float]:
    """Convert a series' coefficient of variation into a 0-100 stability score."""
    cv = fm.coefficient_of_variation(values)
    if cv is None:
        return None
    return fm.clamp(100.0 * (1.0 - cv), 0.0, 100.0)


class BusinessQualityEngine:
    """Score the fundamental quality of a business across seven dimensions."""

    def __init__(self, history_engine: Optional[ForecastEngine] = None) -> None:
        self._history_engine = history_engine or ForecastEngine()

    # ------------------------------------------------------------------ public
    def evaluate(self, data: BusinessQualityInput) -> BusinessQualityResult:
        """Run the full quality assessment and return a :class:`BusinessQualityResult`."""
        if not isinstance(data, BusinessQualityInput):
            raise TypeError("data must be an instance of BusinessQualityInput")

        ratios = data.ratios
        series = self._build_series(data.financial_data)
        notes: list[str] = []

        growth = self._score_growth(ratios, series)
        profitability = self._score_profitability(ratios)
        financial_strength = self._score_financial_strength(ratios)
        cashflow = self._score_cashflow(ratios, series)
        capital_allocation = self._score_capital_allocation(ratios, series)
        consistency = self._score_consistency(series)
        competitive = self._score_competitive_strength(ratios, series)

        dimensions = [
            growth,
            profitability,
            financial_strength,
            cashflow,
            capital_allocation,
            consistency,
            competitive,
        ]

        overall = self._overall_score(
            growth=growth.score,
            profitability=profitability.score,
            financial_strength=financial_strength.score,
            cashflow=cashflow.score,
            capital_allocation=capital_allocation.score,
            consistency=consistency.score,
            competitive_strength=competitive.score,
        )

        strengths, weaknesses = self._explain(dimensions, ratios)
        risk_flags = self._risk_flags(ratios, series)
        reasons = self._reasons(overall, dimensions, risk_flags)

        if not series.revenue:
            notes.append("Historical statements were sparse; scores rely on latest ratios only.")

        return BusinessQualityResult(
            overall_score=round(overall, 2),
            growth_score=round(growth.score, 2),
            profitability_score=round(profitability.score, 2),
            financial_strength_score=round(financial_strength.score, 2),
            cashflow_score=round(cashflow.score, 2),
            capital_allocation_score=round(capital_allocation.score, 2),
            consistency_score=round(consistency.score, 2),
            competitive_strength_score=round(competitive.score, 2),
            grade=self._grade(overall),
            stars=self._stars(overall),
            strengths=strengths,
            weaknesses=weaknesses,
            risk_flags=risk_flags,
            reasons=reasons,
            dimensions=dimensions,
            notes=notes,
        )

    # ----------------------------------------------------------- section 1: growth
    def _score_growth(self, ratios: RatioResult, series: _SeriesBundle) -> QualityDimension:
        revenue_cagr = ratios.revenue_cagr if ratios.revenue_cagr is not None else fm.series_cagr(series.revenue, len(series.revenue))
        pat_cagr = ratios.pat_cagr if ratios.pat_cagr is not None else fm.series_cagr(series.net_income, len(series.net_income))
        fcf_cagr = ratios.fcf_cagr if ratios.fcf_cagr is not None else fm.series_cagr(series.free_cash_flow, len(series.free_cash_flow))
        eps_cagr = ratios.eps_cagr

        components = {
            "revenue_cagr": self._score_cagr(revenue_cagr),
            "pat_cagr": self._score_cagr(pat_cagr),
            "eps_cagr": self._score_cagr(eps_cagr),
            "fcf_cagr": self._score_cagr(fcf_cagr),
        }
        score = _average(list(components.values()))
        score = score if score is not None else _NEUTRAL_SCORE
        return QualityDimension(
            name="Growth",
            score=score,
            rating=_rating_for(score),
            metrics={
                "revenue_cagr": revenue_cagr,
                "pat_cagr": pat_cagr,
                "eps_cagr": eps_cagr,
                "fcf_cagr": fcf_cagr,
            },
        )

    @staticmethod
    def _score_cagr(cagr: Optional[float]) -> Optional[float]:
        number = fm.to_float(cagr)
        if number is None:
            return None
        if number < 0.0:
            return 0.0  # declining business
        return _band_score(number, _GROWTH_BANDS)

    # --------------------------------------------------- section 2: profitability
    def _score_profitability(self, ratios: RatioResult) -> QualityDimension:
        components = {
            "roe": _band_score(ratios.roe, _ROE_BANDS),
            "roce": _band_score(ratios.roce, _ROCE_BANDS),
            "roa": _band_score(ratios.roa, _ROA_BANDS),
            "gross_margin": _band_score(ratios.gross_margin, _GROSS_MARGIN_BANDS),
            "operating_margin": _band_score(ratios.operating_margin, _OPERATING_MARGIN_BANDS),
            "net_margin": _band_score(ratios.net_margin, _NET_MARGIN_BANDS),
        }
        score = _average(list(components.values()))
        score = score if score is not None else _NEUTRAL_SCORE
        return QualityDimension(
            name="Profitability",
            score=score,
            rating=_rating_for(score),
            metrics={
                "roe": ratios.roe,
                "roce": ratios.roce,
                "roa": ratios.roa,
                "gross_margin": ratios.gross_margin,
                "operating_margin": ratios.operating_margin,
                "net_margin": ratios.net_margin,
            },
        )

    # ---------------------------------------------- section 3: financial strength
    def _score_financial_strength(self, ratios: RatioResult) -> QualityDimension:
        components = {
            "debt_equity": _band_score(ratios.debt_equity, _DEBT_EQUITY_BANDS, lower_is_better=True),
            "current_ratio": _band_score(ratios.current_ratio, _CURRENT_RATIO_BANDS),
            "quick_ratio": _band_score(ratios.quick_ratio, _QUICK_RATIO_BANDS),
            "interest_coverage": _band_score(ratios.interest_coverage, _INTEREST_COVERAGE_BANDS),
            "cash_ratio": _band_score(ratios.cash_ratio, _CASH_RATIO_BANDS),
        }
        score = _average(list(components.values()))
        score = score if score is not None else _NEUTRAL_SCORE
        return QualityDimension(
            name="Financial Strength",
            score=score,
            rating=_rating_for(score),
            metrics={
                "debt_equity": ratios.debt_equity,
                "current_ratio": ratios.current_ratio,
                "quick_ratio": ratios.quick_ratio,
                "interest_coverage": ratios.interest_coverage,
                "cash_ratio": ratios.cash_ratio,
            },
        )

    # ------------------------------------------------- section 4: cash-flow quality
    def _score_cashflow(self, ratios: RatioResult, series: _SeriesBundle) -> QualityDimension:
        latest_fcf_margin = series.fcf_margin[-1] if series.fcf_margin else None
        cash_conversion = self._cash_conversion(series)
        ocf_positive = self._positive_share(series.operating_cash_flow)
        fcf_positive = self._positive_share(series.free_cash_flow)
        stability = _stability_score(series.free_cash_flow)

        components = {
            "fcf_margin": _band_score(latest_fcf_margin, _FCF_MARGIN_BANDS),
            "cash_conversion": _band_score(cash_conversion, _CASH_CONVERSION_BANDS),
            "ocf_positive_share": None if ocf_positive is None else ocf_positive * 100.0,
            "fcf_positive_share": None if fcf_positive is None else fcf_positive * 100.0,
            "fcf_stability": stability,
        }
        score = _average(list(components.values()))
        score = score if score is not None else _NEUTRAL_SCORE
        return QualityDimension(
            name="Cash Flow Quality",
            score=score,
            rating=_rating_for(score),
            metrics={
                "fcf_margin": latest_fcf_margin,
                "cash_conversion": cash_conversion,
                "ocf_positive_share": ocf_positive,
                "fcf_positive_share": fcf_positive,
                "fcf_stability": stability,
            },
        )

    def _cash_conversion(self, series: _SeriesBundle) -> Optional[float]:
        """Latest operating cash flow relative to reported profit (OCF / PAT)."""
        if not series.operating_cash_flow or not series.net_income:
            return None
        return fm.safe_divide(series.operating_cash_flow[-1], series.net_income[-1])

    @staticmethod
    def _positive_share(values: Sequence[float]) -> Optional[float]:
        cleaned = fm.clean_series(values)
        if not cleaned:
            return None
        positive = sum(1 for value in cleaned if value > 0.0)
        return positive / len(cleaned)

    # ---------------------------------------------- section 5: capital allocation
    def _score_capital_allocation(self, ratios: RatioResult, series: _SeriesBundle) -> QualityDimension:
        roic = ratios.roce  # ROCE is used as the practical ROIC proxy in ATHENA.
        latest_capex_intensity = series.capex_to_revenue[-1] if series.capex_to_revenue else None
        debt_reduction = self._debt_reduction_score(ratios)
        self_funding = self._positive_share(series.free_cash_flow)

        components = {
            "roic": _band_score(roic, _ROIC_BANDS),
            "capex_intensity": _band_score(latest_capex_intensity, _CAPEX_INTENSITY_BANDS, lower_is_better=True),
            "debt_reduction": debt_reduction,
            "self_funding": None if self_funding is None else self_funding * 100.0,
        }
        score = _average(list(components.values()))
        score = score if score is not None else _NEUTRAL_SCORE
        return QualityDimension(
            name="Capital Allocation",
            score=score,
            rating=_rating_for(score),
            metrics={
                "roic": roic,
                "capex_intensity": latest_capex_intensity,
                "owner_earnings": ratios.owner_earnings,
            },
        )

    @staticmethod
    def _debt_reduction_score(ratios: RatioResult) -> Optional[float]:
        """Reward a conservative balance sheet as a proxy for prudent allocation."""
        if ratios.debt_equity is None:
            return None
        return _band_score(ratios.debt_equity, _DEBT_EQUITY_BANDS, lower_is_better=True)

    # --------------------------------------------------- section 6: consistency
    def _score_consistency(self, series: _SeriesBundle) -> QualityDimension:
        components = {
            "revenue_stability": _stability_score(series.revenue),
            "operating_margin_stability": _stability_score(series.operating_margin),
            "net_margin_stability": _stability_score(series.net_margin),
            "fcf_stability": _stability_score(series.free_cash_flow),
        }
        score = _average(list(components.values()))
        score = score if score is not None else _NEUTRAL_SCORE
        return QualityDimension(
            name="Consistency",
            score=score,
            rating=_rating_for(score),
            metrics={
                "revenue_cv": fm.coefficient_of_variation(series.revenue),
                "operating_margin_cv": fm.coefficient_of_variation(series.operating_margin),
                "fcf_cv": fm.coefficient_of_variation(series.free_cash_flow),
            },
        )

    # -------------------------------------------- section 7: competitive strength
    def _score_competitive_strength(self, ratios: RatioResult, series: _SeriesBundle) -> QualityDimension:
        roce_score = _band_score(ratios.roce, _ROCE_BANDS)
        margin_score = _band_score(ratios.operating_margin, _OPERATING_MARGIN_BANDS)
        growth_stability = _stability_score(series.revenue)
        low_debt_score = _band_score(ratios.debt_equity, _DEBT_EQUITY_BANDS, lower_is_better=True)

        components = [roce_score, margin_score, growth_stability, low_debt_score]
        score = _average(components)
        score = score if score is not None else _NEUTRAL_SCORE
        return QualityDimension(
            name="Competitive Strength",
            score=score,
            rating=_rating_for(score),
            metrics={
                "roce": ratios.roce,
                "operating_margin": ratios.operating_margin,
                "debt_equity": ratios.debt_equity,
            },
        )

    # -------------------------------------------------- sections 9-11: aggregate
    @staticmethod
    def _overall_score(
        *,
        growth: float,
        profitability: float,
        financial_strength: float,
        cashflow: float,
        capital_allocation: float,
        consistency: float,
        competitive_strength: float,
    ) -> float:
        total = (
            growth * _WEIGHTS["growth"]
            + profitability * _WEIGHTS["profitability"]
            + financial_strength * _WEIGHTS["financial_strength"]
            + cashflow * _WEIGHTS["cashflow"]
            + capital_allocation * _WEIGHTS["capital_allocation"]
            + consistency * _WEIGHTS["consistency"]
            + competitive_strength * _WEIGHTS["competitive_strength"]
        )
        return fm.clamp(total, 0.0, 100.0)

    @staticmethod
    def _grade(overall: float) -> Grade:
        for threshold, grade in _GRADE_BANDS:
            if overall >= threshold:
                return grade
        return Grade.D

    @staticmethod
    def _stars(overall: float) -> str:
        filled = 1
        for threshold, count in _STAR_BANDS:
            if overall >= threshold:
                filled = count
                break
        return _FILLED_STAR * filled + _EMPTY_STAR * (_MAX_STARS - filled)

    # ------------------------------------------------ section 8 & 12: narrative
    @staticmethod
    def _explain(dimensions: Sequence[QualityDimension], ratios: RatioResult) -> tuple[list[str], list[str]]:
        strengths: list[str] = []
        weaknesses: list[str] = []

        if ratios.revenue_cagr is not None and ratios.revenue_cagr >= 0.15:
            strengths.append(f"Revenue CAGR {ratios.revenue_cagr * 100:.0f}%")
        if ratios.roce is not None and ratios.roce >= 0.20:
            strengths.append(f"ROCE {ratios.roce * 100:.0f}%")
        if ratios.roe is not None and ratios.roe >= 0.18:
            strengths.append(f"ROE {ratios.roe * 100:.0f}%")
        if ratios.debt_equity is not None and ratios.debt_equity <= 0.25:
            strengths.append("Debt low")
        if ratios.operating_margin is not None and ratios.operating_margin >= 0.20:
            strengths.append(f"Operating margin {ratios.operating_margin * 100:.0f}%")

        for dimension in dimensions:
            if dimension.rating in (Rating.EXCELLENT, Rating.GOOD):
                strengths.append(f"{dimension.name} strong ({dimension.rating.value})")
            elif dimension.rating in (Rating.WEAK, Rating.POOR):
                weaknesses.append(f"{dimension.name} weak ({dimension.rating.value})")

        if ratios.debt_equity is not None and ratios.debt_equity > 1.0:
            weaknesses.append(f"High leverage (D/E {ratios.debt_equity:.2f})")
        if ratios.net_margin is not None and ratios.net_margin < 0.03:
            weaknesses.append("Thin net margin")

        return _dedupe(strengths), _dedupe(weaknesses)

    @staticmethod
    def _risk_flags(ratios: RatioResult, series: _SeriesBundle) -> list[str]:
        flags: list[str] = []

        if _is_declining(series.revenue) or (ratios.revenue_cagr is not None and ratios.revenue_cagr < 0.0):
            flags.append("Declining revenue")
        if series.free_cash_flow and series.free_cash_flow[-1] < 0.0:
            flags.append("Negative free cash flow")
        if ratios.debt_equity is not None and ratios.debt_equity > 1.0:
            flags.append("High debt")
        if _is_compressing(series.operating_margin):
            flags.append("Margin compression")
        if ratios.roe is not None and ratios.roe < 0.08:
            flags.append("Low / falling ROE")
        if _is_deteriorating(series.working_capital_to_revenue):
            flags.append("Working capital deterioration")

        return _dedupe(flags)

    @staticmethod
    def _reasons(overall: float, dimensions: Sequence[QualityDimension], risk_flags: Sequence[str]) -> list[str]:
        reasons: list[str] = [f"Overall business-quality score {overall:.0f}/100."]
        ranked = sorted(dimensions, key=lambda dimension: dimension.score, reverse=True)
        if ranked:
            best = ranked[0]
            worst = ranked[-1]
            reasons.append(f"Strongest area: {best.name} ({best.score:.0f}).")
            reasons.append(f"Weakest area: {worst.name} ({worst.score:.0f}).")
        if risk_flags:
            reasons.append("Risk flags: " + ", ".join(risk_flags) + ".")
        return reasons

    # ---------------------------------------------------------------- extraction
    def _build_series(self, financial_data: FinancialData) -> _SeriesBundle:
        """Extract oldest-to-newest historical series from the statements."""
        history = self._history_engine.build_history(
            financial_data.income_statement,
            financial_data.balance_sheet,
            financial_data.cash_flow,
        )

        net_income = self._statement_series(financial_data.income_statement, ("NetIncome", "netincome"))
        operating_cash_flow = self._statement_series(
            financial_data.cash_flow, ("OperatingCashFlow", "operatingcashflow")
        )
        revenue = history.revenue
        net_margin = _ratio_pairs(net_income, revenue)

        return _SeriesBundle(
            revenue=revenue,
            net_income=net_income,
            operating_margin=history.operating_margin,
            net_margin=net_margin,
            free_cash_flow=history.free_cash_flow,
            fcf_margin=history.fcf_margin,
            operating_cash_flow=operating_cash_flow,
            working_capital_to_revenue=history.working_capital_to_revenue,
            capex_to_revenue=history.capex_to_revenue,
        )

    @staticmethod
    def _statement_series(frame: Optional[pd.DataFrame], aliases: tuple[str, ...]) -> list[float]:
        """Return an oldest-to-newest numeric series for a statement line item.

        Handles both the service long-form frame (with a ``metric`` period
        column) and an already engine-shaped frame (metrics as index).
        """
        if frame is None or frame.empty:
            return []

        if "metric" in frame.columns:
            engine_frame = st.to_engine_frame(frame)
        else:
            engine_frame = _order_columns_by_date(frame)

        if engine_frame.empty:
            return []

        normalized = {st.normalize_label(alias) for alias in aliases}
        for label in engine_frame.index:
            if st.normalize_label(label) in normalized:
                return [fm.to_float(value) or 0.0 for value in engine_frame.loc[label].tolist()]
        for label in engine_frame.index:
            if any(alias in st.normalize_label(label) for alias in normalized):
                return [fm.to_float(value) or 0.0 for value in engine_frame.loc[label].tolist()]
        return []


def _order_columns_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    parsed = pd.to_datetime(pd.Series(list(working.columns)), errors="coerce")
    if parsed.notna().all():
        ordered = [column for _, column in sorted(zip(parsed, working.columns))]
        working = working.reindex(ordered, axis=1)
    return working


def _ratio_pairs(numerator: Sequence[float], denominator: Sequence[float]) -> list[float]:
    ratios: list[float] = []
    for num, den in zip(numerator, denominator):
        value = fm.safe_divide(num, den)
        if value is not None:
            ratios.append(value)
    return ratios


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _is_declining(values: Sequence[float]) -> bool:
    cleaned = fm.clean_series(values)
    return len(cleaned) >= 2 and cleaned[-1] < cleaned[-2]


def _is_compressing(values: Sequence[float]) -> bool:
    cleaned = fm.clean_series(values)
    if len(cleaned) < 2:
        return False
    earlier = fm.mean(cleaned[:-1])
    return earlier is not None and cleaned[-1] < earlier * 0.9


def _is_deteriorating(values: Sequence[float]) -> bool:
    cleaned = fm.clean_series(values)
    if len(cleaned) < 2:
        return False
    return cleaned[-1] > cleaned[0] * 1.1 and cleaned[0] > 0.0


__all__ = ["BusinessQualityEngine", "BusinessQualityInput"]
