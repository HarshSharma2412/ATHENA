"""ATHENA's Management Quality Engine.

This engine judges the *people* running a business rather than the business
itself. It reads historical management communications (earnings-call
transcripts, annual reports and investor presentations), extracts the concrete
forward-looking commitments management made (revenue/margin guidance, CapEx,
debt-reduction, expansion plans), compares each promise against the actual
result, and grades management on execution, guidance accuracy, transparency,
capital allocation and communication - rolling those into an overall management
score and trust score with a plain-language verdict.

The engine reuses :mod:`athena.utils.finance_math` for statistics and consumes
objects produced elsewhere in ATHENA
(:class:`~athena.models.business_quality_models.BusinessQualityResult`,
:class:`~athena.models.valuation_models.DCFResult`,
:class:`~athena.models.reverse_dcf_models.ReverseDCFResult`). It never mutates
its inputs and contains no Streamlit or presentation code.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from athena.models.business_quality_models import BusinessQualityResult
from athena.models.management_models import (
    Commitment,
    CommitmentOutcome,
    CommitmentStatus,
    CommitmentType,
    DocumentType,
    ManagementDocument,
    ManagementGrade,
    ManagementQualityResult,
)
from athena.models.reverse_dcf_models import ExpectationLevel, ReverseDCFResult
from athena.models.valuation_models import DCFResult
from athena.utils import finance_math as fm

logger = logging.getLogger(__name__)

_NEUTRAL_SCORE = 50.0

# Status scores used to turn commitment outcomes into a 0-100 accuracy figure.
_STATUS_SCORE: dict[CommitmentStatus, float] = {
    CommitmentStatus.DELIVERED: 100.0,
    CommitmentStatus.DELAYED: 60.0,
    CommitmentStatus.MISSED: 0.0,
}

# Delivery bands on the achieved/promised ratio (higher-is-better metrics).
_DELIVERED_RATIO = 0.95
_DELAYED_RATIO = 0.85
_SHORTFALL_RATIO = 0.70  # below this is a material shortfall / red flag

# Overall-score weights (must sum to 1.0).
_WEIGHTS: dict[str, float] = {
    "execution": 0.25,
    "guidance_accuracy": 0.25,
    "transparency": 0.15,
    "capital_allocation": 0.20,
    "communication": 0.15,
}

# Grade cut-offs on the overall score.
_GRADE_BANDS: tuple[tuple[float, ManagementGrade], ...] = (
    (90.0, ManagementGrade.A_PLUS),
    (80.0, ManagementGrade.A),
    (65.0, ManagementGrade.B),
    (50.0, ManagementGrade.C),
)

# Metric each commitment type is verified against in the ``actuals`` mapping.
_METRIC_KEY: dict[CommitmentType, str] = {
    CommitmentType.REVENUE_GUIDANCE: "revenue_growth",
    CommitmentType.MARGIN_GUIDANCE: "operating_margin",
    CommitmentType.CAPEX: "capex",
    CommitmentType.DEBT_REDUCTION: "net_debt",
    CommitmentType.EXPANSION: "capacity",
    CommitmentType.GENERAL_TARGET: "revenue_growth",
}

# Commitment types where a *lower* actual than promised counts as success.
_LOWER_IS_BETTER: frozenset[CommitmentType] = frozenset({CommitmentType.DEBT_REDUCTION})

# Operational (execution-oriented) commitment types.
_OPERATIONAL: frozenset[CommitmentType] = frozenset(
    {CommitmentType.CAPEX, CommitmentType.DEBT_REDUCTION, CommitmentType.EXPANSION}
)

# Cue words that signal a forward-looking commitment.
_CUE_WORDS: tuple[str, ...] = (
    "expect", "guidance", "guide", "target", "aim", "plan", "intend",
    "commit", "anticipate", "project", "will", "aspire", "outlook",
    "forecast", "estimate", "on track",
)

# Ordered so the most specific category wins when several keywords appear.
_TYPE_KEYWORDS: tuple[tuple[CommitmentType, tuple[str, ...]], ...] = (
    (CommitmentType.DEBT_REDUCTION, ("debt", "deleverage", "net debt", "borrowing", "leverage")),
    (CommitmentType.CAPEX, ("capex", "capital expenditure", "capital outlay", "capital spend")),
    (CommitmentType.MARGIN_GUIDANCE, ("margin", "profitability", "ebitda margin", "operating margin")),
    (CommitmentType.EXPANSION, (
        "capacity", "expansion", "expand", "new plant", "new store", "greenfield",
        "brownfield", "facility", "footprint", "stores",
    )),
    (CommitmentType.REVENUE_GUIDANCE, ("revenue", "topline", "top line", "sales", "turnover")),
)

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)", re.IGNORECASE)
_AMOUNT_RE = re.compile(
    r"(?:\u20b9|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)\s*(cr|crore|crores|bn|billion|mn|million)",
    re.IGNORECASE,
)
_PERIOD_RE = re.compile(r"fy\s*'?\s*(\d{2,4})", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Amount unit -> multiplier to normalise everything to crore.
_UNIT_TO_CRORE: dict[str, float] = {
    "cr": 1.0, "crore": 1.0, "crores": 1.0,
    "bn": 100.0, "billion": 100.0,
    "mn": 0.1, "million": 0.1,
}


@dataclass(frozen=True)
class ManagementQualityInput:
    """Everything the engine needs to assess management quality."""

    documents: Sequence[ManagementDocument] = field(default_factory=tuple)
    actuals: dict[str, dict[str, float]] = field(default_factory=dict)
    business_quality: Optional[BusinessQualityResult] = None
    dcf: Optional[DCFResult] = None
    reverse_dcf: Optional[ReverseDCFResult] = None
    ticker: Optional[str] = None


def _normalize_period(period: str) -> str:
    """Normalise a fiscal-period token to canonical ``FYNN`` form."""
    match = _PERIOD_RE.search(period or "")
    if match is None:
        return (period or "").strip().upper()
    digits = match.group(1)
    if len(digits) == 4:
        digits = digits[-2:]
    return f"FY{digits.zfill(2)}"


class ManagementQualityEngine:
    """Extract management commitments and grade how well they were delivered."""

    def evaluate(self, data: ManagementQualityInput) -> ManagementQualityResult:
        """Run the full management-quality assessment."""
        if not isinstance(data, ManagementQualityInput):
            raise TypeError("data must be an instance of ManagementQualityInput")

        notes: list[str] = []
        actuals = {_normalize_period(period): metrics for period, metrics in data.actuals.items()}

        commitments = self._extract_commitments(data.documents)
        outcomes = [self._verify(commitment, actuals) for commitment in commitments]

        delivered = sum(1 for outcome in outcomes if outcome.status is CommitmentStatus.DELIVERED)
        delayed = sum(1 for outcome in outcomes if outcome.status is CommitmentStatus.DELAYED)
        missed = sum(1 for outcome in outcomes if outcome.status is CommitmentStatus.MISSED)

        if not data.documents:
            notes.append("No management documents supplied; scores fall back to neutral.")
        if not commitments:
            notes.append("No measurable commitments detected in the supplied documents.")

        guidance_accuracy = self._guidance_accuracy(outcomes)
        execution = self._execution_score(outcomes, guidance_accuracy, data.business_quality)
        transparency = self._transparency_score(data.documents, commitments)
        capital_allocation = self._capital_allocation_score(outcomes, data.business_quality)
        communication = self._communication_score(data.documents, commitments)
        trust = self._trust_score(guidance_accuracy, transparency, capital_allocation)

        overall = self._overall_score(
            execution=execution,
            guidance_accuracy=guidance_accuracy,
            transparency=transparency,
            capital_allocation=capital_allocation,
            communication=communication,
        )

        strengths, weaknesses = self._explain(
            execution=execution,
            guidance_accuracy=guidance_accuracy,
            transparency=transparency,
            capital_allocation=capital_allocation,
            communication=communication,
            delivered=delivered,
            missed=missed,
        )
        red_flags = self._red_flags(outcomes, transparency, data.reverse_dcf, data.business_quality)
        ai_summary = self._ai_summary(
            overall=overall,
            outcomes=outcomes,
            delivered=delivered,
            delayed=delayed,
            missed=missed,
            red_flags=red_flags,
            reverse_dcf=data.reverse_dcf,
        )

        return ManagementQualityResult(
            overall_score=round(overall, 2),
            management_trust_score=round(trust, 2),
            execution_score=round(execution, 2),
            guidance_accuracy=round(guidance_accuracy, 2),
            transparency=round(transparency, 2),
            capital_allocation=round(capital_allocation, 2),
            communication=round(communication, 2),
            grade=self._grade(overall),
            delivered=delivered,
            delayed=delayed,
            missed=missed,
            strengths=strengths,
            weaknesses=weaknesses,
            red_flags=red_flags,
            ai_summary=ai_summary,
            outcomes=outcomes,
            notes=notes,
        )

    # -------------------------------------------------------------- detection
    def _extract_commitments(self, documents: Sequence[ManagementDocument]) -> list[Commitment]:
        commitments: list[Commitment] = []
        for document in documents:
            source_period = _normalize_period(document.fiscal_period)
            for sentence in self._sentences(document.text):
                commitment = self._commitment_from_sentence(sentence, document, source_period)
                if commitment is not None:
                    commitments.append(commitment)
        return commitments

    @staticmethod
    def _sentences(text: str) -> list[str]:
        if not text:
            return []
        return [sentence.strip() for sentence in _SENTENCE_RE.split(text) if sentence.strip()]

    def _commitment_from_sentence(
        self, sentence: str, document: ManagementDocument, source_period: str
    ) -> Optional[Commitment]:
        lowered = sentence.lower()
        if not any(cue in lowered for cue in _CUE_WORDS):
            return None

        commitment_type = self._classify(lowered)
        if commitment_type is None:
            return None

        promised_value, unit = self._extract_value(sentence, commitment_type)
        target_period = self._target_period(sentence, source_period)
        return Commitment(
            commitment_type=commitment_type,
            source_period=source_period,
            target_period=target_period,
            sentence=sentence,
            promised_value=promised_value,
            unit=unit,
            document_type=document.document_type,
        )

    @staticmethod
    def _classify(lowered: str) -> Optional[CommitmentType]:
        for commitment_type, keywords in _TYPE_KEYWORDS:
            if any(keyword in lowered for keyword in keywords):
                return commitment_type
        if "target" in lowered or "guidance" in lowered:
            return CommitmentType.GENERAL_TARGET
        return None

    @staticmethod
    def _extract_value(sentence: str, commitment_type: CommitmentType) -> tuple[Optional[float], Optional[str]]:
        if commitment_type in (CommitmentType.CAPEX, CommitmentType.DEBT_REDUCTION):
            amount = _AMOUNT_RE.search(sentence)
            if amount is not None:
                value = fm.to_float(amount.group(1).replace(",", ""))
                unit = amount.group(2).lower()
                if value is not None:
                    return value * _UNIT_TO_CRORE.get(unit, 1.0), "cr"
            percent = _PERCENT_RE.search(sentence)
            if percent is not None:
                parsed = fm.to_float(percent.group(1))
                return (parsed / 100.0 if parsed is not None else None), "%"
            return None, None

        percent = _PERCENT_RE.search(sentence)
        if percent is not None:
            parsed = fm.to_float(percent.group(1))
            return (parsed / 100.0 if parsed is not None else None), "%"
        amount = _AMOUNT_RE.search(sentence)
        if amount is not None:
            value = fm.to_float(amount.group(1).replace(",", ""))
            unit = amount.group(2).lower()
            if value is not None:
                return value * _UNIT_TO_CRORE.get(unit, 1.0), "cr"
        return None, None

    @staticmethod
    def _target_period(sentence: str, source_period: str) -> str:
        match = _PERIOD_RE.search(sentence)
        if match is not None:
            return _normalize_period(match.group(0))
        return source_period

    # ------------------------------------------------------------ verification
    def _verify(self, commitment: Commitment, actuals: dict[str, dict[str, float]]) -> CommitmentOutcome:
        metric_key = _METRIC_KEY[commitment.commitment_type]
        period_actuals = actuals.get(commitment.target_period, {})
        actual = fm.to_float(period_actuals.get(metric_key))

        if commitment.promised_value is None or actual is None:
            status = CommitmentStatus.UNVERIFIED if commitment.promised_value is None else CommitmentStatus.PENDING
            return CommitmentOutcome(commitment=commitment, actual_value=actual, variance=None, status=status)

        variance = actual - commitment.promised_value
        status = self._status_from_ratio(commitment, actual)
        return CommitmentOutcome(
            commitment=commitment, actual_value=actual, variance=variance, status=status
        )

    @staticmethod
    def _status_from_ratio(commitment: Commitment, actual: float) -> CommitmentStatus:
        promised = commitment.promised_value
        if promised is None or promised == 0.0:
            return CommitmentStatus.UNVERIFIED
        ratio = actual / promised
        if commitment.commitment_type in _LOWER_IS_BETTER:
            # For "reduce to X" targets, a lower actual is better.
            ratio = promised / actual if actual != 0.0 else 0.0
        if ratio >= _DELIVERED_RATIO:
            return CommitmentStatus.DELIVERED
        if ratio >= _DELAYED_RATIO:
            return CommitmentStatus.DELAYED
        return CommitmentStatus.MISSED

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def _verified(outcomes: Sequence[CommitmentOutcome]) -> list[CommitmentOutcome]:
        return [outcome for outcome in outcomes if outcome.status in _STATUS_SCORE]

    def _guidance_accuracy(self, outcomes: Sequence[CommitmentOutcome]) -> float:
        verified = self._verified(outcomes)
        if not verified:
            return _NEUTRAL_SCORE
        scores = [_STATUS_SCORE[outcome.status] for outcome in verified]
        return fm.clamp(sum(scores) / len(scores), 0.0, 100.0)

    def _execution_score(
        self,
        outcomes: Sequence[CommitmentOutcome],
        guidance_accuracy: float,
        business_quality: Optional[BusinessQualityResult],
    ) -> float:
        operational = [
            outcome
            for outcome in self._verified(outcomes)
            if outcome.commitment.commitment_type in _OPERATIONAL
        ]
        if operational:
            operational_score = sum(_STATUS_SCORE[outcome.status] for outcome in operational) / len(operational)
        else:
            operational_score = guidance_accuracy

        blended = 0.5 * guidance_accuracy + 0.5 * operational_score
        if business_quality is not None:
            blended = 0.8 * blended + 0.2 * business_quality.consistency_score
        return fm.clamp(blended, 0.0, 100.0)

    @staticmethod
    def _transparency_score(
        documents: Sequence[ManagementDocument], commitments: Sequence[Commitment]
    ) -> float:
        if not documents:
            return _NEUTRAL_SCORE
        quantified = sum(1 for commitment in commitments if commitment.promised_value is not None)
        specificity = quantified / len(commitments) if commitments else 0.0
        volume = fm.clamp(len(commitments) / (2.0 * len(documents)), 0.0, 1.0)
        diversity = len({document.document_type for document in documents}) / len(DocumentType)
        score = 100.0 * (0.5 * specificity + 0.3 * volume + 0.2 * diversity)
        return fm.clamp(score, 0.0, 100.0)

    def _capital_allocation_score(
        self,
        outcomes: Sequence[CommitmentOutcome],
        business_quality: Optional[BusinessQualityResult],
    ) -> float:
        allocation = [
            outcome
            for outcome in self._verified(outcomes)
            if outcome.commitment.commitment_type in (CommitmentType.CAPEX, CommitmentType.DEBT_REDUCTION)
        ]
        delivery = (
            sum(_STATUS_SCORE[outcome.status] for outcome in allocation) / len(allocation)
            if allocation
            else None
        )
        if business_quality is None:
            return delivery if delivery is not None else _NEUTRAL_SCORE
        if delivery is None:
            return business_quality.capital_allocation_score
        return fm.clamp(0.6 * business_quality.capital_allocation_score + 0.4 * delivery, 0.0, 100.0)

    @staticmethod
    def _communication_score(
        documents: Sequence[ManagementDocument], commitments: Sequence[Commitment]
    ) -> float:
        if not documents:
            return _NEUTRAL_SCORE
        frequency = fm.clamp(len(documents) / 4.0, 0.0, 1.0)
        clarity = (
            sum(1 for commitment in commitments if commitment.promised_value is not None) / len(commitments)
            if commitments
            else 0.0
        )
        revenue_targets = [
            commitment.promised_value
            for commitment in commitments
            if commitment.commitment_type is CommitmentType.REVENUE_GUIDANCE
            and commitment.promised_value is not None
        ]
        cv = fm.coefficient_of_variation(revenue_targets)
        consistency = fm.clamp(1.0 - cv, 0.0, 1.0) if cv is not None else 0.5
        score = 100.0 * (0.4 * frequency + 0.3 * clarity + 0.3 * consistency)
        return fm.clamp(score, 0.0, 100.0)

    @staticmethod
    def _trust_score(guidance_accuracy: float, transparency: float, capital_allocation: float) -> float:
        return fm.clamp(
            0.5 * guidance_accuracy + 0.25 * transparency + 0.25 * capital_allocation, 0.0, 100.0
        )

    @staticmethod
    def _overall_score(
        *,
        execution: float,
        guidance_accuracy: float,
        transparency: float,
        capital_allocation: float,
        communication: float,
    ) -> float:
        total = (
            execution * _WEIGHTS["execution"]
            + guidance_accuracy * _WEIGHTS["guidance_accuracy"]
            + transparency * _WEIGHTS["transparency"]
            + capital_allocation * _WEIGHTS["capital_allocation"]
            + communication * _WEIGHTS["communication"]
        )
        return fm.clamp(total, 0.0, 100.0)

    @staticmethod
    def _grade(overall: float) -> ManagementGrade:
        for threshold, grade in _GRADE_BANDS:
            if overall >= threshold:
                return grade
        return ManagementGrade.D

    # ---------------------------------------------------------------- narrative
    @staticmethod
    def _explain(
        *,
        execution: float,
        guidance_accuracy: float,
        transparency: float,
        capital_allocation: float,
        communication: float,
        delivered: int,
        missed: int,
    ) -> tuple[list[str], list[str]]:
        labelled = {
            "Execution": execution,
            "Guidance accuracy": guidance_accuracy,
            "Transparency": transparency,
            "Capital allocation": capital_allocation,
            "Communication": communication,
        }
        strengths = [f"{name} strong ({score:.0f})" for name, score in labelled.items() if score >= 75.0]
        weaknesses = [f"{name} weak ({score:.0f})" for name, score in labelled.items() if score < 45.0]
        if delivered and delivered >= missed:
            strengths.append(f"Delivered {delivered} of its stated commitments")
        if missed > delivered:
            weaknesses.append(f"Missed {missed} commitments")
        return strengths, weaknesses

    def _red_flags(
        self,
        outcomes: Sequence[CommitmentOutcome],
        transparency: float,
        reverse_dcf: Optional[ReverseDCFResult],
        business_quality: Optional[BusinessQualityResult],
    ) -> list[str]:
        flags: list[str] = []
        verified = self._verified(outcomes)
        missed = [outcome for outcome in verified if outcome.status is CommitmentStatus.MISSED]

        if len(missed) >= 2 or (verified and len(missed) / len(verified) > 0.4):
            flags.append("Repeatedly missed guidance")
        for outcome in outcomes:
            promised = outcome.commitment.promised_value
            if outcome.actual_value is None or promised is None or promised == 0.0:
                continue
            ratio = outcome.actual_value / promised
            if outcome.commitment.commitment_type not in _LOWER_IS_BETTER and ratio < _SHORTFALL_RATIO:
                flags.append(
                    f"Material shortfall on {outcome.commitment.commitment_type.value} "
                    f"({outcome.commitment.target_period})"
                )
        if transparency < 40.0:
            flags.append("Low disclosure / few measurable targets")
        if (
            reverse_dcf is not None
            and reverse_dcf.expectation_level in (ExpectationLevel.AGGRESSIVE, ExpectationLevel.EXTREME)
            and missed
        ):
            flags.append("Aggressive market expectations against a weak delivery record")
        return _dedupe(flags)

    @staticmethod
    def _ai_summary(
        *,
        overall: float,
        outcomes: Sequence[CommitmentOutcome],
        delivered: int,
        delayed: int,
        missed: int,
        red_flags: Sequence[str],
        reverse_dcf: Optional[ReverseDCFResult],
    ) -> str:
        parts = [f"Overall management score {overall:.0f}/100."]
        verified_total = delivered + delayed + missed
        if verified_total:
            parts.append(
                f"Of {verified_total} verifiable commitments, {delivered} delivered, "
                f"{delayed} delayed and {missed} missed."
            )
        else:
            parts.append("No commitments could be verified against actual results.")

        example = _example_miss(outcomes)
        if example is not None:
            parts.append(example)
        if red_flags:
            parts.append("Red flags: " + ", ".join(red_flags) + ".")
        if reverse_dcf is not None and reverse_dcf.required_growth is not None:
            parts.append(
                f"The market implies {reverse_dcf.required_growth * 100:.0f}% growth, "
                f"so execution against guidance is critical."
            )
        return " ".join(parts)


def _example_miss(outcomes: Sequence[CommitmentOutcome]) -> Optional[str]:
    for outcome in outcomes:
        if (
            outcome.status is CommitmentStatus.MISSED
            and outcome.commitment.promised_value is not None
            and outcome.actual_value is not None
            and outcome.commitment.unit == "%"
        ):
            return (
                f"Example: {outcome.commitment.target_period} "
                f"{outcome.commitment.commitment_type.value} "
                f"{outcome.commitment.promised_value * 100:.0f}% vs actual "
                f"{outcome.actual_value * 100:.0f}% (Missed)."
            )
    return None


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


__all__ = ["ManagementQualityEngine", "ManagementQualityInput"]
