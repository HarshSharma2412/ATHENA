"""ATHENA's Investment Thesis Engine.

The capstone engine: it does not compute new fundamentals, it *synthesises* the
outputs of every upstream engine - valuation (:class:`DCFResult`), market
expectations (:class:`ReverseDCFResult`), business quality
(:class:`BusinessQualityResult`) and risk (:class:`RiskResult`) - into a single
investor-facing one-page thesis: a rating, a bull and bear case, ranked
strengths / weaknesses / risks, section summaries, an expected-return outlook
and a suggested holding horizon.

The engine reads immutable inputs, never mutates them, and contains no Streamlit
or presentation code. Narrative is generated from the actual numbers (no generic
filler).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from athena.engines.ratio_engine import FinancialData, RatioResult
from athena.models.business_quality_models import BusinessQualityResult
from athena.models.investment_thesis_models import (
    Impact,
    InvestmentHorizon,
    InvestmentRating,
    InvestmentThesisResult,
    Likelihood,
    MajorRisk,
)
from athena.models.reverse_dcf_models import ExpectationLevel, ReverseDCFResult
from athena.models.risk_models import RankedRisk, RiskGrade, RiskResult
from athena.models.valuation_models import DCFResult
from athena.utils import finance_math as fm

logger = logging.getLogger(__name__)

_NEUTRAL = 50.0
_MAX_LIST = 5

# Composite-score weights (must sum to 1.0).
_WEIGHTS: dict[str, float] = {"quality": 0.35, "valuation": 0.35, "risk": 0.30}

# Rating cut-offs on the overall score.
_RATING_BANDS: tuple[tuple[float, InvestmentRating], ...] = (
    (80.0, InvestmentRating.STRONG_BUY),
    (70.0, InvestmentRating.BUY),
    (60.0, InvestmentRating.ACCUMULATE),
    (45.0, InvestmentRating.HOLD),
    (30.0, InvestmentRating.REDUCE),
)

# Valuation attractiveness from margin of safety (higher MoS => more attractive).
_MOS_BANDS: tuple[tuple[float, float], ...] = (
    (0.30, 92.0),
    (0.15, 78.0),
    (0.0, 62.0),
    (-0.15, 42.0),
    (-0.30, 25.0),
)

# Rating ordering for guardrail capping (best -> worst).
_RATING_ORDER: tuple[InvestmentRating, ...] = (
    InvestmentRating.STRONG_BUY,
    InvestmentRating.BUY,
    InvestmentRating.ACCUMULATE,
    InvestmentRating.HOLD,
    InvestmentRating.REDUCE,
    InvestmentRating.SELL,
)


@dataclass(frozen=True)
class InvestmentThesisInput:
    """Every upstream result the thesis synthesises."""

    financial_data: Optional[FinancialData] = None
    ratios: Optional[RatioResult] = None
    dcf: Optional[DCFResult] = None
    reverse_dcf: Optional[ReverseDCFResult] = None
    business_quality: Optional[BusinessQualityResult] = None
    risk: Optional[RiskResult] = None
    ticker: Optional[str] = None
    current_price: Optional[float] = None


def _percent(value: Optional[float]) -> str:
    return f"{value * 100:.0f}%" if value is not None else "n/a"


class InvestmentThesisEngine:
    """Combine ATHENA's engine outputs into a single actionable thesis."""

    def build(self, data: InvestmentThesisInput) -> InvestmentThesisResult:
        """Produce the one-page investment thesis."""
        if not isinstance(data, InvestmentThesisInput):
            raise TypeError("data must be an instance of InvestmentThesisInput")

        notes: list[str] = []
        price = data.current_price if data.current_price is not None else self._price(data.dcf)

        margin_of_safety = self._margin_of_safety(data.dcf, price)

        quality_score = self._quality_score(data.business_quality, notes)
        valuation_score = self._valuation_score(margin_of_safety, data.reverse_dcf, notes)
        risk_score = self._risk_score(data.risk, notes)

        overall = fm.clamp(
            quality_score * _WEIGHTS["quality"]
            + valuation_score * _WEIGHTS["valuation"]
            + risk_score * _WEIGHTS["risk"],
            0.0,
            100.0,
        )

        rating = self._rating(overall, data.risk, margin_of_safety)
        confidence = self._confidence(data)

        expected_3y, expected_5y = self._expected_returns(data.dcf, margin_of_safety)
        downside = self._expected_downside(data.dcf, price)
        horizon = self._horizon(data.business_quality, data.risk)

        strengths = self._strengths(data, margin_of_safety)
        weaknesses = self._weaknesses(data, margin_of_safety)
        major_risks = self._major_risks(data.risk)

        valuation_summary = self._valuation_summary(data.dcf, data.reverse_dcf, price)
        business_summary = self._business_summary(data.business_quality)
        financial_summary = self._financial_summary(data.ratios)
        risk_summary = self._risk_summary(data.risk)

        bull_case = self._bull_case(data, margin_of_safety, price, expected_3y)
        bear_case = self._bear_case(data, downside)
        summary = self._summary(
            ticker=data.ticker,
            rating=rating,
            overall=overall,
            price=price,
            margin_of_safety=margin_of_safety,
            expected_3y=expected_3y,
            horizon=horizon,
        )

        return InvestmentThesisResult(
            investment_rating=rating,
            overall_score=round(overall, 2),
            confidence_score=round(confidence, 2),
            summary=summary,
            bull_case=bull_case,
            bear_case=bear_case,
            key_strengths=strengths,
            key_weaknesses=weaknesses,
            major_risks=major_risks,
            valuation_summary=valuation_summary,
            business_summary=business_summary,
            financial_summary=financial_summary,
            risk_summary=risk_summary,
            expected_return_3y=expected_3y,
            expected_return_5y=expected_5y,
            expected_downside=downside,
            margin_of_safety=margin_of_safety,
            investment_horizon=horizon,
            notes=notes,
        )

    # ------------------------------------------------------------- components
    @staticmethod
    def _quality_score(business_quality: Optional[BusinessQualityResult], notes: list[str]) -> float:
        if business_quality is None:
            notes.append("No BusinessQualityResult supplied; quality uses a neutral prior.")
            return _NEUTRAL
        return fm.clamp(business_quality.overall_score, 0.0, 100.0)

    @staticmethod
    def _margin_of_safety(dcf: Optional[DCFResult], price: Optional[float]) -> Optional[float]:
        if dcf is None:
            return None
        if dcf.margin_of_safety is not None:
            return dcf.margin_of_safety
        if dcf.intrinsic_value is not None and price is not None and price > 0.0:
            return fm.safe_divide(dcf.intrinsic_value - price, price)
        return None

    def _valuation_score(
        self,
        margin_of_safety: Optional[float],
        reverse_dcf: Optional[ReverseDCFResult],
        notes: list[str],
    ) -> float:
        scores: list[float] = []
        if margin_of_safety is not None:
            scores.append(self._mos_score(margin_of_safety))
        if reverse_dcf is not None:
            # High market expectations reduce the valuation appeal.
            scores.append(fm.clamp(100.0 - reverse_dcf.expectation_score, 0.0, 100.0))
        if not scores:
            notes.append("No valuation inputs supplied; valuation uses a neutral prior.")
            return _NEUTRAL
        return fm.clamp(sum(scores) / len(scores), 0.0, 100.0)

    @staticmethod
    def _mos_score(margin_of_safety: float) -> float:
        for threshold, score in _MOS_BANDS:
            if margin_of_safety >= threshold:
                return score
        return 10.0

    @staticmethod
    def _risk_score(risk: Optional[RiskResult], notes: list[str]) -> float:
        if risk is None:
            notes.append("No RiskResult supplied; risk uses a neutral prior.")
            return _NEUTRAL
        return fm.clamp(100.0 - risk.overall_risk_score, 0.0, 100.0)

    # ---------------------------------------------------------------- rating
    def _rating(
        self,
        overall: float,
        risk: Optional[RiskResult],
        margin_of_safety: Optional[float],
    ) -> InvestmentRating:
        rating = InvestmentRating.SELL
        for threshold, candidate in _RATING_BANDS:
            if overall >= threshold:
                rating = candidate
                break

        # Guardrails: a great score cannot override severe risk or gross overvaluation.
        if risk is not None and risk.risk_grade is RiskGrade.VERY_HIGH:
            rating = self._cap(rating, InvestmentRating.HOLD)
        if margin_of_safety is not None and margin_of_safety <= -0.30:
            rating = self._cap(rating, InvestmentRating.REDUCE)
        return rating

    @staticmethod
    def _cap(rating: InvestmentRating, ceiling: InvestmentRating) -> InvestmentRating:
        order = _RATING_ORDER
        return rating if order.index(rating) >= order.index(ceiling) else ceiling

    def _confidence(self, data: InvestmentThesisInput) -> float:
        provided = [
            data.business_quality is not None,
            data.dcf is not None,
            data.reverse_dcf is not None,
            data.risk is not None,
            data.ratios is not None,
        ]
        completeness = sum(provided) / len(provided) * 100.0
        dcf_confidence = data.dcf.confidence_score if data.dcf is not None else _NEUTRAL
        return fm.clamp(0.5 * completeness + 0.5 * dcf_confidence, 0.0, 100.0)

    # -------------------------------------------------------------- returns
    def _expected_returns(
        self, dcf: Optional[DCFResult], margin_of_safety: Optional[float]
    ) -> tuple[Optional[float], Optional[float]]:
        if dcf is None:
            return None, None
        fundamental = dcf.expected_cagr if dcf.expected_cagr is not None else dcf.revenue_growth
        reprice_3y = self._annualised_reprice(margin_of_safety, 3)
        reprice_5y = self._annualised_reprice(margin_of_safety, 5)
        three_year = self._combine_return(fundamental, reprice_3y)
        five_year = self._combine_return(fundamental, reprice_5y)
        return three_year, five_year

    @staticmethod
    def _annualised_reprice(margin_of_safety: Optional[float], years: int) -> Optional[float]:
        if margin_of_safety is None or years <= 0:
            return None
        return (1.0 + margin_of_safety) ** (1.0 / years) - 1.0

    @staticmethod
    def _combine_return(fundamental: Optional[float], reprice: Optional[float]) -> Optional[float]:
        if fundamental is None and reprice is None:
            return None
        base = fundamental or 0.0
        adjust = reprice or 0.0
        return (1.0 + base) * (1.0 + adjust) - 1.0

    @staticmethod
    def _expected_downside(dcf: Optional[DCFResult], price: Optional[float]) -> Optional[float]:
        if dcf is None or dcf.bear_value is None or price is None or price <= 0.0:
            return None
        return fm.safe_divide(dcf.bear_value - price, price)

    @staticmethod
    def _horizon(
        business_quality: Optional[BusinessQualityResult], risk: Optional[RiskResult]
    ) -> InvestmentHorizon:
        quality = business_quality.overall_score if business_quality is not None else _NEUTRAL
        overall_risk = risk.overall_risk_score if risk is not None else _NEUTRAL
        if quality >= 70.0 and overall_risk < 55.0:
            return InvestmentHorizon.LONG_TERM
        if overall_risk >= 70.0:
            return InvestmentHorizon.SHORT_TERM
        return InvestmentHorizon.MEDIUM_TERM

    # ------------------------------------------------------ strengths/weakness
    def _strengths(self, data: InvestmentThesisInput, margin_of_safety: Optional[float]) -> list[str]:
        strengths: list[str] = []
        if data.business_quality is not None:
            strengths.extend(data.business_quality.strengths)
        if margin_of_safety is not None and margin_of_safety > 0.15:
            strengths.append(f"Trades {_percent(margin_of_safety)} below intrinsic value")
        if data.risk is not None and data.risk.overall_risk_score < 40.0:
            strengths.append(f"Low overall risk ({data.risk.overall_risk_score:.0f}/100)")
        if data.ratios is not None and data.ratios.roce is not None and data.ratios.roce >= 0.20:
            strengths.append(f"High ROCE {_percent(data.ratios.roce)}")
        return _dedupe(strengths)[:_MAX_LIST]

    def _weaknesses(self, data: InvestmentThesisInput, margin_of_safety: Optional[float]) -> list[str]:
        weaknesses: list[str] = []
        if data.business_quality is not None:
            weaknesses.extend(data.business_quality.weaknesses)
        if margin_of_safety is not None and margin_of_safety < -0.10:
            weaknesses.append(f"Trades {_percent(abs(margin_of_safety))} above intrinsic value")
        if (
            data.reverse_dcf is not None
            and data.reverse_dcf.expectation_level in (ExpectationLevel.AGGRESSIVE, ExpectationLevel.EXTREME)
        ):
            weaknesses.append("Market already prices in aggressive growth")
        if data.risk is not None and data.risk.overall_risk_score >= 60.0:
            weaknesses.append(f"Elevated overall risk ({data.risk.overall_risk_score:.0f}/100)")
        return _dedupe(weaknesses)[:_MAX_LIST]

    @staticmethod
    def _major_risks(risk: Optional[RiskResult]) -> list[MajorRisk]:
        if risk is None:
            return []
        monitors: dict[str, str] = {
            "financial": "Debt/EBITDA and interest-coverage trend each quarter",
            "business": "Quarterly revenue growth and gross-margin stability",
            "valuation": "Price vs intrinsic value and implied growth",
            "growth": "YoY revenue and order-book / volume growth",
            "liquidity": "Current ratio and cash balance vs short-term dues",
            "leverage": "Absolute net debt and refinancing schedule",
            "cashflow": "Operating cash flow vs PAT and FCF conversion",
            "cyclicality": "Input/commodity prices and demand cycle",
            "execution": "Project milestones, capex payback and ROIC",
            "governance": "Auditor notes, related-party deals, promoter pledging",
        }
        return [
            MajorRisk(
                title=ranked.title,
                probability=_to_likelihood(ranked.probability.value),
                impact=_to_impact(ranked.impact.value),
                monitoring_indicator=monitors.get(ranked.category, "Track relevant quarterly disclosures"),
            )
            for ranked in _top(risk.top_risks, _MAX_LIST)
        ]

    # --------------------------------------------------------------- summaries
    def _valuation_summary(
        self, dcf: Optional[DCFResult], reverse_dcf: Optional[ReverseDCFResult], price: Optional[float]
    ) -> str:
        if dcf is None and reverse_dcf is None:
            return "Valuation not assessed - no DCF or reverse-DCF inputs supplied."
        parts: list[str] = []
        if dcf is not None:
            if dcf.intrinsic_value is not None and price is not None:
                verdict = "undervalued" if (dcf.margin_of_safety or 0.0) > 0 else "overvalued"
                parts.append(
                    f"DCF intrinsic value is {dcf.intrinsic_value:.0f} versus a price of {price:.0f}, "
                    f"a margin of safety of {_percent(dcf.margin_of_safety)} ({verdict})."
                )
            parts.append(f"DCF assumes {_percent(dcf.revenue_growth)} revenue growth at a "
                         f"{_percent(dcf.discount_rate)} discount rate.")
        if reverse_dcf is not None:
            parts.append(
                f"The current price implies {_percent(reverse_dcf.required_growth)} growth versus a "
                f"{_percent(reverse_dcf.historical_growth)} history "
                f"({reverse_dcf.expectation_level.value.lower()} expectations)."
            )
        return " ".join(parts)

    @staticmethod
    def _business_summary(business_quality: Optional[BusinessQualityResult]) -> str:
        if business_quality is None:
            return "Business quality not assessed - no BusinessQualityResult supplied."
        return (
            f"Business quality scores {business_quality.overall_score:.0f}/100 "
            f"(grade {business_quality.grade.value}, {business_quality.stars}). "
            f"Growth {business_quality.growth_score:.0f}, profitability "
            f"{business_quality.profitability_score:.0f}, financial strength "
            f"{business_quality.financial_strength_score:.0f}, consistency "
            f"{business_quality.consistency_score:.0f}."
        )

    @staticmethod
    def _financial_summary(ratios: Optional[RatioResult]) -> str:
        if ratios is None:
            return "Financials not assessed - no RatioResult supplied."
        return (
            f"ROE {_percent(ratios.roe)}, ROCE {_percent(ratios.roce)}, operating margin "
            f"{_percent(ratios.operating_margin)}, net margin {_percent(ratios.net_margin)}; "
            f"D/E {ratios.debt_equity if ratios.debt_equity is not None else 'n/a'}, "
            f"current ratio {ratios.current_ratio if ratios.current_ratio is not None else 'n/a'}, "
            f"revenue CAGR {_percent(ratios.revenue_cagr)}."
        )

    @staticmethod
    def _risk_summary(risk: Optional[RiskResult]) -> str:
        if risk is None:
            return "Risk not assessed - no RiskResult supplied."
        flags = ", ".join(risk.risk_flags) if risk.risk_flags else "no acute flags"
        return (
            f"Overall risk {risk.overall_risk_score:.0f}/100 ({risk.risk_grade.value}); "
            f"{risk.cyclicality_level.value.lower()} cyclicality. Key flags: {flags}."
        )

    # --------------------------------------------------------------- narrative
    def _bull_case(
        self,
        data: InvestmentThesisInput,
        margin_of_safety: Optional[float],
        price: Optional[float],
        expected_3y: Optional[float],
    ) -> str:
        parts: list[str] = []
        bq = data.business_quality
        if bq is not None and bq.overall_score >= 60.0:
            parts.append(
                f"A grade-{bq.grade.value} business ({bq.overall_score:.0f}/100) with durable "
                f"returns can keep compounding earnings."
            )
        if margin_of_safety is not None and margin_of_safety > 0.10:
            parts.append(
                f"Shares sit {_percent(margin_of_safety)} below DCF value, so re-rating to fair "
                f"value adds to returns."
            )
        if data.dcf is not None and data.dcf.bull_value is not None and price:
            upside = fm.safe_divide(data.dcf.bull_value - price, price)
            if upside is not None:
                parts.append(f"The bull scenario implies about {_percent(upside)} upside.")
        if expected_3y is not None:
            parts.append(f"Base-case 3-year return is roughly {_percent(expected_3y)} annualised.")
        if not parts:
            parts.append("Upside depends on sustained execution; supplied data is too thin for a strong bull case.")
        return " ".join(parts)

    def _bear_case(self, data: InvestmentThesisInput, downside: Optional[float]) -> str:
        parts: list[str] = []
        if (
            data.reverse_dcf is not None
            and data.reverse_dcf.expectation_level in (ExpectationLevel.AGGRESSIVE, ExpectationLevel.EXTREME)
        ):
            parts.append(
                f"The price already demands {_percent(data.reverse_dcf.required_growth)} growth versus "
                f"{_percent(data.reverse_dcf.historical_growth)} delivered, leaving little room for error."
            )
        if data.risk is not None and data.risk.risk_flags:
            parts.append("Risk flags include " + ", ".join(data.risk.risk_flags[:3]) + ".")
        if downside is not None:
            parts.append(f"The bear scenario implies about {_percent(downside)} downside.")
        if data.business_quality is not None and data.business_quality.weaknesses:
            parts.append("Weaknesses: " + ", ".join(data.business_quality.weaknesses[:3]) + ".")
        if not parts:
            parts.append("Main danger is paying up for a business whose growth or margins disappoint.")
        return " ".join(parts)

    @staticmethod
    def _summary(
        *,
        ticker: Optional[str],
        rating: InvestmentRating,
        overall: float,
        price: Optional[float],
        margin_of_safety: Optional[float],
        expected_3y: Optional[float],
        horizon: InvestmentHorizon,
    ) -> str:
        name = ticker or "The company"
        lead = f"{name}: {rating.value} (thesis score {overall:.0f}/100)."
        valuation = (
            f" At {price:.0f} the margin of safety is {_percent(margin_of_safety)}."
            if price is not None and margin_of_safety is not None
            else ""
        )
        outlook = (
            f" Base-case 3-year return ~{_percent(expected_3y)}." if expected_3y is not None else ""
        )
        return f"{lead}{valuation}{outlook} Suggested horizon: {horizon.value.lower()}."

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _price(dcf: Optional[DCFResult]) -> Optional[float]:
        return dcf.current_price if dcf is not None else None


def _top(risks: Sequence[RankedRisk], limit: int) -> list[RankedRisk]:
    return sorted(risks, key=lambda risk: risk.severity, reverse=True)[:limit]


def _to_likelihood(value: str) -> Likelihood:
    return Likelihood(value)


def _to_impact(value: str) -> Impact:
    return Impact(value)


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


__all__ = ["InvestmentThesisEngine", "InvestmentThesisInput"]
