"""Immutable data models for ATHENA's Reverse DCF Engine.

A forward DCF asks *what is the business worth?* A reverse DCF flips the
question: *given today's market price, what future must the business deliver to
justify it?* These dataclasses capture the implied market expectations, the
historical reality, the gap between them and a plain-language verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExpectationLevel(str, Enum):
    """How demanding the market's implied assumptions are versus history."""

    LOW = "Low"
    MODERATE = "Moderate"
    AGGRESSIVE = "Aggressive"
    EXTREME = "Extreme"


class ValuationRisk(str, Enum):
    """Risk that the price cannot be justified by achievable fundamentals."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass(frozen=True)
class ImpliedExpectations:
    """Assumptions the current market price implies the business must deliver."""

    required_growth: Optional[float]
    required_ebit_margin: Optional[float]
    required_fcf_margin: Optional[float]
    required_roic: Optional[float]
    required_terminal_growth: float
    required_discount_rate: float


@dataclass(frozen=True)
class HistoricalReality:
    """What the business has actually delivered historically."""

    historical_growth: Optional[float]
    historical_margin: Optional[float]
    historical_fcf_margin: Optional[float]
    historical_roic: Optional[float]
    historical_roe: Optional[float]
    historical_roce: Optional[float]


@dataclass(frozen=True)
class GapAnalysis:
    """Signed gaps (implied minus historical); positive means market demands more."""

    growth_gap: Optional[float]
    margin_gap: Optional[float]
    roic_gap: Optional[float]


@dataclass(frozen=True)
class ReverseDCFResult:
    """Full output of the reverse DCF assessment."""

    required_growth: Optional[float]
    required_margin: Optional[float]
    required_roic: Optional[float]
    expectation_score: float
    expectation_level: ExpectationLevel
    valuation_risk: ValuationRisk
    historical_growth: Optional[float]
    historical_margin: Optional[float]
    difference: Optional[float]
    ai_summary: str
    implied: ImpliedExpectations
    historical: HistoricalReality
    gap: GapAnalysis
    growth_capped: bool = False
    notes: list[str] = field(default_factory=list)


__all__ = [
    "ExpectationLevel",
    "GapAnalysis",
    "HistoricalReality",
    "ImpliedExpectations",
    "ReverseDCFResult",
    "ValuationRisk",
]
