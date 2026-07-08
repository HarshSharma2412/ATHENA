"""Immutable data models for ATHENA's Investment Thesis Engine.

These dataclasses describe the final, investor-facing one-page thesis that
synthesises every upstream engine (valuation, reverse DCF, business quality and
risk) into a single rating, narrative and return outlook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class InvestmentRating(str, Enum):
    """Final actionable call on the stock."""

    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    ACCUMULATE = "Accumulate"
    HOLD = "Hold"
    REDUCE = "Reduce"
    SELL = "Sell"


class InvestmentHorizon(str, Enum):
    """Suggested holding period for the thesis to play out."""

    SHORT_TERM = "Short Term"
    MEDIUM_TERM = "Medium Term"
    LONG_TERM = "Long Term"


class Likelihood(str, Enum):
    """Qualitative probability band for a monitored risk."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Impact(str, Enum):
    """Qualitative impact band for a monitored risk."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass(frozen=True)
class MajorRisk:
    """A headline risk with how likely it is, how much it hurts, and what to watch."""

    title: str
    probability: Likelihood
    impact: Impact
    monitoring_indicator: str


@dataclass(frozen=True)
class InvestmentThesisResult:
    """Full output of the Investment Thesis Engine - a one-page analyst report."""

    investment_rating: InvestmentRating
    overall_score: float
    confidence_score: float
    summary: str
    bull_case: str
    bear_case: str
    key_strengths: list[str] = field(default_factory=list)
    key_weaknesses: list[str] = field(default_factory=list)
    major_risks: list[MajorRisk] = field(default_factory=list)
    valuation_summary: str = ""
    business_summary: str = ""
    financial_summary: str = ""
    risk_summary: str = ""
    expected_return_3y: Optional[float] = None
    expected_return_5y: Optional[float] = None
    expected_downside: Optional[float] = None
    margin_of_safety: Optional[float] = None
    investment_horizon: InvestmentHorizon = InvestmentHorizon.LONG_TERM
    notes: list[str] = field(default_factory=list)


__all__ = [
    "Impact",
    "InvestmentHorizon",
    "InvestmentRating",
    "InvestmentThesisResult",
    "Likelihood",
    "MajorRisk",
]
