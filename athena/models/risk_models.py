"""Immutable data models for ATHENA's Risk Engine.

All risk scores are expressed on a 0-100 scale where **lower is better** (0 = no
meaningful risk, 100 = severe risk). This is the inverse of the quality/score
engines and is deliberate: a risk report should read like a threat register.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskGrade(str, Enum):
    """Overall risk band for a company."""

    VERY_LOW = "Very Low"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"


class CyclicalityLevel(str, Enum):
    """How sensitive the business is to the economic cycle."""

    STABLE = "Stable"
    MODERATE = "Moderate"
    HIGHLY_CYCLICAL = "Highly Cyclical"


class Likelihood(str, Enum):
    """Qualitative probability band for a ranked risk."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Impact(str, Enum):
    """Qualitative impact band for a ranked risk."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass(frozen=True)
class RankedRisk:
    """A single prioritised risk with probability, impact and mitigation."""

    title: str
    category: str
    severity: float
    probability: Likelihood
    impact: Impact
    mitigation: str


@dataclass(frozen=True)
class RiskResult:
    """Full output of the Risk Engine (all sub-scores: lower is better)."""

    overall_risk_score: float
    financial_risk: float
    business_risk: float
    valuation_risk: float
    growth_risk: float
    liquidity_risk: float
    leverage_risk: float
    cashflow_risk: float
    cyclicality_risk: float
    execution_risk: float
    governance_risk: float
    risk_grade: RiskGrade
    cyclicality_level: CyclicalityLevel
    risk_flags: list[str] = field(default_factory=list)
    top_risks: list[RankedRisk] = field(default_factory=list)
    ai_summary: str = ""
    notes: list[str] = field(default_factory=list)


__all__ = [
    "CyclicalityLevel",
    "Impact",
    "Likelihood",
    "RankedRisk",
    "RiskGrade",
    "RiskResult",
]
