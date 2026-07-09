"""ATHENA's Risk Engine.

A dedicated engine that scores a company across ten independent risk dimensions
(financial, business, valuation, growth, liquidity, leverage, cash flow,
cyclicality, execution and governance), rolls them into a single overall risk
score, assigns a risk grade, ranks the top risks with probability / impact /
mitigation, and writes a plain-language summary.

Every score is on a 0-100 scale where **lower is better**. The engine reads
already-computed ATHENA objects (``FinancialData``, ``RatioResult``,
``DCFResult``, ``BusinessQualityResult``, ``ReverseDCFResult``) plus the raw
statements, never mutates its inputs, and contains no Streamlit code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from athena.engines.forecast_engine import ForecastEngine
from athena.engines.ratio_engine import FinancialData, RatioResult
from athena.models.business_quality_models import BusinessQualityResult
from athena.models.reverse_dcf_models import ReverseDCFResult, ValuationRisk
from athena.models.risk_models import (
    CyclicalityLevel,
    Impact,
    Likelihood,
    RankedRisk,
    RiskGrade,
    RiskResult,
)
from athena.models.valuation_models import DCFResult, HistoricalAnalysis
from athena.utils import finance_math as fm
from athena.utils import statements as st

logger = logging.getLogger(__name__)

_NEUTRAL_RISK = 50.0
_MIN_SERIES = 2

# Overall-risk weights (must sum to 1.0).
_WEIGHTS: dict[str, float] = {
    "financial": 0.18,
    "business": 0.12,
    "valuation": 0.15,
    "growth": 0.10,
    "liquidity": 0.10,
    "leverage": 0.12,
    "cashflow": 0.10,
    "cyclicality": 0.05,
    "execution": 0.05,
    "governance": 0.03,
}

# Risk grade cut-offs on the overall score (lower is safer).
_GRADE_BANDS: tuple[tuple[float, RiskGrade], ...] = (
    (80.0, RiskGrade.VERY_HIGH),
    (60.0, RiskGrade.HIGH),
    (40.0, RiskGrade.MODERATE),
    (20.0, RiskGrade.LOW),
)

# Cyclicality level -> risk contribution.
_CYCLICALITY_RISK: dict[CyclicalityLevel, float] = {
    CyclicalityLevel.STABLE: 15.0,
    CyclicalityLevel.MODERATE: 45.0,
    CyclicalityLevel.HIGHLY_CYCLICAL: 80.0,
}

# Sector keywords that tend to be cyclical / defensive.
_CYCLICAL_SECTORS: tuple[str, ...] = (
    "auto", "automobile", "metal", "steel", "cement", "commodity", "commodities",
    "oil", "gas", "energy", "mining", "real estate", "realty", "construction",
    "capital goods", "shipping", "airline", "hotel", "chemical", "semiconductor",
)
_DEFENSIVE_SECTORS: tuple[str, ...] = (
    "fmcg", "consumer staple", "pharma", "pharmaceutical", "healthcare",
    "utility", "utilities", "software", "it services", "telecom", "insurance",
)

# Bands are (threshold, risk_score). See ``_band_risk`` for interpretation.
_DEBT_EQUITY_BANDS = ((0.25, 5.0), (0.5, 20.0), (1.0, 45.0), (2.0, 70.0))
_DEBT_EBITDA_BANDS = ((1.0, 5.0), (2.0, 20.0), (3.0, 45.0), (4.0, 70.0))
_INTEREST_COVER_BANDS = ((8.0, 5.0), (4.0, 20.0), (2.0, 45.0), (1.0, 70.0))
_CURRENT_RATIO_BANDS = ((2.0, 10.0), (1.5, 25.0), (1.0, 50.0), (0.75, 75.0))
_QUICK_RATIO_BANDS = ((1.5, 10.0), (1.0, 25.0), (0.75, 50.0), (0.5, 75.0))
_CASH_RATIO_BANDS = ((0.75, 10.0), (0.5, 25.0), (0.25, 50.0), (0.1, 75.0))
_MARGIN_OF_SAFETY_BANDS = ((0.30, 5.0), (0.10, 25.0), (-0.10, 50.0), (-0.25, 75.0))
_DEBT_GROWTH_BANDS = ((0.0, 10.0), (0.05, 30.0), (0.15, 60.0))
_WORST_RISK = 95.0


@dataclass(frozen=True)
class RiskInput:
    """Everything the Risk Engine needs to assess a company."""

    financial_data: Optional[FinancialData] = None
    ratios: Optional[RatioResult] = None
    dcf: Optional[DCFResult] = None
    business_quality: Optional[BusinessQualityResult] = None
    reverse_dcf: Optional[ReverseDCFResult] = None
    sector: Optional[str] = None
    ticker: Optional[str] = None


@dataclass(frozen=True)
class _RiskSeries:
    """Historical series used across several risk dimensions."""

    revenue: list[float] = field(default_factory=list)
    net_income: list[float] = field(default_factory=list)
    operating_margin: list[float] = field(default_factory=list)
    free_cash_flow: list[float] = field(default_factory=list)
    fcf_margin: list[float] = field(default_factory=list)
    operating_cash_flow: list[float] = field(default_factory=list)
    debt: list[float] = field(default_factory=list)
    ebitda: list[float] = field(default_factory=list)
    working_capital_to_revenue: list[float] = field(default_factory=list)
    capex_to_revenue: list[float] = field(default_factory=list)


def _band_risk(
    value: Optional[float], bands: tuple[tuple[float, float], ...], *, higher_is_safer: bool
) -> Optional[float]:
    """Map a metric to a 0-100 risk score using ordered threshold bands.

    ``higher_is_safer`` metrics (e.g. interest coverage) use descending
    thresholds; otherwise (e.g. debt/equity) ascending thresholds are used. A
    value beyond the last band earns the worst risk score.
    """
    if value is None:
        return None
    for threshold, risk in bands:
        if higher_is_safer and value >= threshold:
            return risk
        if not higher_is_safer and value <= threshold:
            return risk
    return _WORST_RISK


def _average(scores: Sequence[Optional[float]]) -> Optional[float]:
    present = [score for score in scores if score is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _instability_risk(values: Sequence[float]) -> Optional[float]:
    """Convert a series' coefficient of variation into a 0-100 risk score."""
    cleaned = fm.clean_series(values)
    if len(cleaned) < _MIN_SERIES:
        return None
    cv = fm.coefficient_of_variation(cleaned)
    if cv is None:
        return None
    return fm.clamp(cv * 200.0, 0.0, 100.0)


class RiskEngine:
    """Assess and rank the risks embedded in a company's fundamentals."""

    def __init__(self, history_engine: Optional[ForecastEngine] = None) -> None:
        self._history_engine = history_engine or ForecastEngine()

    def assess(self, data: RiskInput) -> RiskResult:
        """Run the full multi-dimensional risk assessment."""
        if not isinstance(data, RiskInput):
            raise TypeError("data must be an instance of RiskInput")

        notes: list[str] = []
        series = self._build_series(data.financial_data)
        ratios = data.ratios

        cyclicality_level = self._cyclicality_level(data.sector, series)

        financial = self._financial_risk(ratios, series)
        business = self._business_risk(series, notes)
        valuation = self._valuation_risk(data.dcf, data.reverse_dcf)
        growth = self._growth_risk(series)
        liquidity = self._liquidity_risk(ratios)
        leverage = self._leverage_risk(ratios, series)
        cashflow = self._cashflow_risk(series, ratios)
        cyclicality = _CYCLICALITY_RISK[cyclicality_level]
        execution = self._execution_risk(data.business_quality, notes)
        governance = self._governance_risk(notes)

        components = {
            "financial": financial,
            "business": business,
            "valuation": valuation,
            "growth": growth,
            "liquidity": liquidity,
            "leverage": leverage,
            "cashflow": cashflow,
            "cyclicality": cyclicality,
            "execution": execution,
            "governance": governance,
        }
        overall = sum(components[name] * weight for name, weight in _WEIGHTS.items())
        overall = fm.clamp(overall, 0.0, 100.0)

        risk_flags = self._risk_flags(ratios, series, data.dcf, data.reverse_dcf, cyclicality_level)
        top_risks = self._rank_risks(components, cyclicality_level)
        ai_summary = self._ai_summary(overall, self._grade(overall), top_risks, risk_flags)

        return RiskResult(
            overall_risk_score=round(overall, 2),
            financial_risk=round(financial, 2),
            business_risk=round(business, 2),
            valuation_risk=round(valuation, 2),
            growth_risk=round(growth, 2),
            liquidity_risk=round(liquidity, 2),
            leverage_risk=round(leverage, 2),
            cashflow_risk=round(cashflow, 2),
            cyclicality_risk=round(cyclicality, 2),
            execution_risk=round(execution, 2),
            governance_risk=round(governance, 2),
            risk_grade=self._grade(overall),
            cyclicality_level=cyclicality_level,
            risk_flags=risk_flags,
            top_risks=top_risks,
            ai_summary=ai_summary,
            notes=notes,
        )

    # ----------------------------------------------------------- dimension: 1
    def _financial_risk(self, ratios: Optional[RatioResult], series: _RiskSeries) -> float:
        if ratios is None:
            return _NEUTRAL_RISK
        debt_ebitda = self._debt_to_ebitda(ratios, series)
        scores = [
            _band_risk(ratios.debt_equity, _DEBT_EQUITY_BANDS, higher_is_safer=False),
            _band_risk(ratios.interest_coverage, _INTEREST_COVER_BANDS, higher_is_safer=True),
            _band_risk(debt_ebitda, _DEBT_EBITDA_BANDS, higher_is_safer=False),
            _band_risk(ratios.current_ratio, _CURRENT_RATIO_BANDS, higher_is_safer=True),
            _band_risk(ratios.quick_ratio, _QUICK_RATIO_BANDS, higher_is_safer=True),
        ]
        return _average(scores) or _NEUTRAL_RISK

    # ----------------------------------------------------------- dimension: 2
    def _business_risk(self, series: _RiskSeries, notes: list[str]) -> float:
        scores = [
            _instability_risk(series.revenue),
            _instability_risk(series.operating_margin),
        ]
        notes.append("Customer/supplier concentration not assessed (no disclosure data).")
        return _average(scores) or _NEUTRAL_RISK

    # ----------------------------------------------------------- dimension: 3
    def _valuation_risk(
        self, dcf: Optional[DCFResult], reverse_dcf: Optional[ReverseDCFResult]
    ) -> float:
        scores: list[Optional[float]] = []
        if dcf is not None:
            scores.append(_band_risk(dcf.margin_of_safety, _MARGIN_OF_SAFETY_BANDS, higher_is_safer=True))
        if reverse_dcf is not None:
            scores.append(fm.clamp(reverse_dcf.expectation_score, 0.0, 100.0))
            scores.append(self._reverse_risk_score(reverse_dcf.valuation_risk))
        return _average(scores) or _NEUTRAL_RISK

    @staticmethod
    def _reverse_risk_score(risk: ValuationRisk) -> float:
        return {ValuationRisk.LOW: 20.0, ValuationRisk.MEDIUM: 55.0, ValuationRisk.HIGH: 85.0}[risk]

    # ----------------------------------------------------------- dimension: 4
    def _growth_risk(self, series: _RiskSeries) -> float:
        scores = [
            _instability_risk(series.revenue),
            _instability_risk(series.net_income),
            _instability_risk(series.free_cash_flow),
        ]
        risk = _average(scores)
        if risk is None:
            return _NEUTRAL_RISK
        if _is_declining(series.revenue):
            risk = fm.clamp(risk + 15.0, 0.0, 100.0)
        return risk

    # ----------------------------------------------------------- dimension: 5
    def _liquidity_risk(self, ratios: Optional[RatioResult]) -> float:
        if ratios is None:
            return _NEUTRAL_RISK
        scores = [
            _band_risk(ratios.current_ratio, _CURRENT_RATIO_BANDS, higher_is_safer=True),
            _band_risk(ratios.quick_ratio, _QUICK_RATIO_BANDS, higher_is_safer=True),
            _band_risk(ratios.cash_ratio, _CASH_RATIO_BANDS, higher_is_safer=True),
        ]
        return _average(scores) or _NEUTRAL_RISK

    # ----------------------------------------------------------- dimension: 6
    def _leverage_risk(self, ratios: Optional[RatioResult], series: _RiskSeries) -> float:
        debt_growth = fm.series_cagr(series.debt, len(fm.clean_series(series.debt)) - 1) if series.debt else None
        scores = [
            _band_risk(debt_growth, _DEBT_GROWTH_BANDS, higher_is_safer=False),
        ]
        if ratios is not None:
            scores.append(_band_risk(ratios.debt_equity, _DEBT_EQUITY_BANDS, higher_is_safer=False))
            scores.append(_band_risk(ratios.interest_coverage, _INTEREST_COVER_BANDS, higher_is_safer=True))
        return _average(scores) or _NEUTRAL_RISK

    # ----------------------------------------------------------- dimension: 7
    def _cashflow_risk(self, series: _RiskSeries, ratios: Optional[RatioResult]) -> float:
        scores: list[Optional[float]] = []
        negative_fcf = self._negative_share(series.free_cash_flow)
        if negative_fcf is not None:
            scores.append(fm.clamp(negative_fcf * 100.0, 0.0, 100.0))
        scores.append(_instability_risk(series.free_cash_flow))
        conversion = self._cash_conversion(series)
        if conversion is not None:
            scores.append(_band_risk(conversion, ((1.0, 10.0), (0.8, 30.0), (0.5, 55.0), (0.25, 80.0)), higher_is_safer=True))
        return _average(scores) or _NEUTRAL_RISK

    # ----------------------------------------------------------- dimension: 8
    def _cyclicality_level(self, sector: Optional[str], series: _RiskSeries) -> CyclicalityLevel:
        if sector:
            lowered = sector.lower()
            if any(keyword in lowered for keyword in _CYCLICAL_SECTORS):
                return CyclicalityLevel.HIGHLY_CYCLICAL
            if any(keyword in lowered for keyword in _DEFENSIVE_SECTORS):
                return CyclicalityLevel.STABLE
        revenue_risk = _instability_risk(series.revenue)
        if revenue_risk is None:
            return CyclicalityLevel.MODERATE
        if revenue_risk >= 60.0:
            return CyclicalityLevel.HIGHLY_CYCLICAL
        if revenue_risk <= 25.0:
            return CyclicalityLevel.STABLE
        return CyclicalityLevel.MODERATE

    # ----------------------------------------------------------- dimension: 9
    @staticmethod
    def _execution_risk(business_quality: Optional[BusinessQualityResult], notes: list[str]) -> float:
        if business_quality is None:
            notes.append("Execution risk uses a neutral prior (no BusinessQualityResult supplied).")
            return _NEUTRAL_RISK
        base = 100.0 - business_quality.overall_score
        consistency_gap = 100.0 - business_quality.consistency_score
        return fm.clamp(0.7 * base + 0.3 * consistency_gap, 0.0, 100.0)

    # ---------------------------------------------------------- dimension: 10
    @staticmethod
    def _governance_risk(notes: list[str]) -> float:
        notes.append("Governance risk is a neutral placeholder (future annual-report AI integration).")
        return _NEUTRAL_RISK

    # -------------------------------------------------------------- flags
    def _risk_flags(
        self,
        ratios: Optional[RatioResult],
        series: _RiskSeries,
        dcf: Optional[DCFResult],
        reverse_dcf: Optional[ReverseDCFResult],
        cyclicality_level: CyclicalityLevel,
    ) -> list[str]:
        flags: list[str] = []
        if _is_declining(series.revenue):
            flags.append("Revenue slowing / declining")
        if self._negative_share(series.free_cash_flow):
            flags.append("Negative free cash flow")
        if ratios is not None:
            if ratios.debt_equity is not None and ratios.debt_equity > 1.5:
                flags.append("High leverage (D/E > 1.5)")
            if ratios.interest_coverage is not None and ratios.interest_coverage < 2.0:
                flags.append("Weak interest coverage")
            if ratios.current_ratio is not None and ratios.current_ratio < 1.0:
                flags.append("Current ratio below 1")
        if _is_deteriorating(series.working_capital_to_revenue):
            flags.append("Working capital increasing")
        if _is_compressing(series.operating_margin):
            flags.append("Operating margin compression")
        if dcf is not None and dcf.margin_of_safety is not None and dcf.margin_of_safety < -0.1:
            flags.append("Trading above intrinsic value")
        if reverse_dcf is not None and reverse_dcf.valuation_risk is ValuationRisk.HIGH:
            flags.append("Market expectations demanding vs history")
        if cyclicality_level is CyclicalityLevel.HIGHLY_CYCLICAL:
            flags.append("Commodity / cyclical price sensitivity")
        return _dedupe(flags)

    # -------------------------------------------------------------- ranking
    def _rank_risks(
        self, components: dict[str, float], cyclicality_level: CyclicalityLevel
    ) -> list[RankedRisk]:
        catalogue: dict[str, tuple[str, str]] = {
            "financial": ("Balance-sheet / solvency risk", "Reduce debt, lengthen maturities, build interest cover"),
            "business": ("Unstable revenue & margin profile", "Diversify customers/products, defend pricing"),
            "valuation": ("Rich valuation vs fundamentals", "Wait for margin of safety; size position for downside"),
            "growth": ("Volatile or slowing growth", "Track order book & demand; avoid extrapolating peaks"),
            "liquidity": ("Tight liquidity position", "Hold more cash, refinance short-term dues"),
            "leverage": ("Rising leverage / servicing risk", "Prioritise deleveraging, cap incremental debt"),
            "cashflow": ("Weak / erratic cash generation", "Improve working-capital cycle and cash conversion"),
            "cyclicality": ("Cyclical / commodity exposure", "Stress-test through-cycle; watch input prices"),
            "execution": ("Execution & capital-allocation risk", "Monitor project delivery and ROIC discipline"),
            "governance": ("Governance not yet assessed", "Review promoter conduct, audits, related-party deals"),
        }
        ranked = [
            RankedRisk(
                title=catalogue[name][0],
                category=name,
                severity=round(score, 2),
                probability=self._likelihood(score),
                impact=self._impact(name, score, cyclicality_level),
                mitigation=catalogue[name][1],
            )
            for name, score in components.items()
        ]
        ranked.sort(key=lambda risk: risk.severity, reverse=True)
        return ranked[:10]

    @staticmethod
    def _likelihood(score: float) -> Likelihood:
        if score >= 66.0:
            return Likelihood.HIGH
        if score >= 40.0:
            return Likelihood.MEDIUM
        return Likelihood.LOW

    @staticmethod
    def _impact(name: str, score: float, cyclicality_level: CyclicalityLevel) -> Impact:
        high_impact = {"financial", "leverage", "liquidity", "valuation", "cashflow"}
        if name in high_impact and score >= 50.0:
            return Impact.HIGH
        if score >= 66.0:
            return Impact.HIGH
        if score >= 35.0:
            return Impact.MEDIUM
        return Impact.LOW

    # -------------------------------------------------------------- summary
    @staticmethod
    def _ai_summary(
        overall: float,
        grade: RiskGrade,
        top_risks: Sequence[RankedRisk],
        risk_flags: Sequence[str],
    ) -> str:
        parts = [f"Overall risk {overall:.0f}/100 ({grade.value}); lower is safer."]
        material = [risk for risk in top_risks if risk.severity >= 40.0][:5]
        if material:
            bullets = "; ".join(f"{risk.title} ({risk.severity:.0f})" for risk in material)
            parts.append("Major risks: " + bullets + ".")
        else:
            parts.append("No individual dimension registers material risk.")
        if risk_flags:
            parts.append("Flags: " + ", ".join(risk_flags) + ".")
        return " ".join(parts)

    # -------------------------------------------------------------- helpers
    def _build_series(self, financial_data: Optional[FinancialData]) -> _RiskSeries:
        if financial_data is None:
            return _RiskSeries()
        history = self._history_engine.build_history(
            financial_data.income_statement,
            financial_data.balance_sheet,
            financial_data.cash_flow,
        )
        net_income = st.metric_series(financial_data.income_statement, ("NetIncome", "netincome"))
        operating_cash_flow = st.metric_series(
            financial_data.cash_flow, ("OperatingCashFlow", "operatingcashflow")
        )
        debt = st.metric_series(financial_data.balance_sheet, ("TotalDebt", "totaldebt", "longtermdebt"))
        ebitda = st.metric_series(financial_data.income_statement, ("EBITDA", "ebitda"))
        if not ebitda:
            ebitda = self._ebitda_from_history(history)
        return _RiskSeries(
            revenue=history.revenue,
            net_income=net_income,
            operating_margin=history.operating_margin,
            free_cash_flow=history.free_cash_flow,
            fcf_margin=history.fcf_margin,
            operating_cash_flow=operating_cash_flow,
            debt=debt,
            ebitda=ebitda,
            working_capital_to_revenue=history.working_capital_to_revenue,
            capex_to_revenue=history.capex_to_revenue,
        )

    @staticmethod
    def _ebitda_from_history(history: HistoricalAnalysis) -> list[float]:
        depreciation = [
            revenue * ratio
            for revenue, ratio in zip(history.revenue, history.depreciation_to_revenue)
        ]
        return [ebit + dep for ebit, dep in zip(history.ebit, depreciation)]

    @staticmethod
    def _debt_to_ebitda(ratios: RatioResult, series: _RiskSeries) -> Optional[float]:
        ebitda = fm.clean_series(series.ebitda)
        debt = fm.clean_series(series.debt)
        if not ebitda or not debt or ebitda[-1] <= 0.0:
            return None
        return fm.safe_divide(debt[-1], ebitda[-1])

    @staticmethod
    def _negative_share(values: Sequence[float]) -> Optional[float]:
        cleaned = fm.clean_series(values)
        if not cleaned:
            return None
        negatives = sum(1 for value in cleaned if value < 0.0)
        return negatives / len(cleaned)

    @staticmethod
    def _cash_conversion(series: _RiskSeries) -> Optional[float]:
        ocf = fm.clean_series(series.operating_cash_flow)
        net_income = fm.clean_series(series.net_income)
        if not ocf or not net_income or net_income[-1] <= 0.0:
            return None
        return fm.safe_divide(ocf[-1], net_income[-1])

    @staticmethod
    def _grade(overall: float) -> RiskGrade:
        for threshold, grade in _GRADE_BANDS:
            if overall >= threshold:
                return grade
        return RiskGrade.VERY_LOW


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


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


__all__ = ["RiskEngine", "RiskInput"]
