"""Immutable data models for ATHENA's Business Quality Engine.

These dataclasses describe the output of the
:class:`~athena.engines.business_quality_engine.BusinessQualityEngine`, which
judges whether a company is a fundamentally high-quality business (distinct from
whether it is cheaply valued). Every score is expressed on a 0-100 scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Grade(str, Enum):
    """Letter grade summarising the overall business-quality score."""

    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C = "C"
    D = "D"


class Rating(str, Enum):
    """Qualitative rating for an individual quality dimension."""

    EXCELLENT = "Excellent"
    GOOD = "Good"
    AVERAGE = "Average"
    WEAK = "Weak"
    POOR = "Poor"


@dataclass(frozen=True)
class QualityDimension:
    """Score and supporting metrics for a single business-quality dimension."""

    name: str
    score: float
    rating: Rating
    metrics: dict[str, Optional[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class BusinessQualityResult:
    """Full output of the Business Quality Engine."""

    overall_score: float
    growth_score: float
    profitability_score: float
    financial_strength_score: float
    cashflow_score: float
    capital_allocation_score: float
    consistency_score: float
    competitive_strength_score: float
    grade: Grade
    stars: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    dimensions: list[QualityDimension] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


__all__ = [
    "BusinessQualityResult",
    "Grade",
    "QualityDimension",
    "Rating",
]
