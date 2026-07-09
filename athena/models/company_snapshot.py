"""Immutable data models for ATHENA's Company Memory.

A :class:`CompanySnapshot` is a point-in-time record of everything ATHENA
concluded about a company on a given date: prices, the headline scores from each
engine, and JSON-serialisable summaries of the DCF, reverse-DCF and investment
thesis. The trend and comparison models describe how those records evolve over
time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class CompanySnapshot:
    """A single stored analysis for one company at one point in time."""

    ticker: str
    date: datetime
    current_price: Optional[float] = None
    intrinsic_value: Optional[float] = None
    business_score: Optional[float] = None
    risk_score: Optional[float] = None
    management_score: Optional[float] = None
    recommendation: Optional[str] = None
    dcf: dict[str, Any] = field(default_factory=dict)
    reverse_dcf: dict[str, Any] = field(default_factory=dict)
    investment_thesis: dict[str, Any] = field(default_factory=dict)
    snapshot_id: Optional[int] = None

    @property
    def quarter(self) -> tuple[int, int]:
        """The ``(year, quarter)`` this snapshot belongs to."""
        return self.date.year, (self.date.month - 1) // 3 + 1


@dataclass(frozen=True)
class MetricTrend:
    """The trajectory of one metric across a company's stored snapshots."""

    metric: str
    points: list[tuple[datetime, float]] = field(default_factory=list)
    first: Optional[float] = None
    last: Optional[float] = None
    change: Optional[float] = None
    direction: str = "Unknown"


@dataclass(frozen=True)
class TrendReport:
    """Trends for every headline metric ATHENA tracks over time."""

    ticker: str
    business_score: MetricTrend
    intrinsic_value: MetricTrend
    risk_score: MetricTrend
    management_score: MetricTrend
    current_price: MetricTrend


@dataclass(frozen=True)
class SnapshotComparison:
    """Current-quarter snapshot measured against the previous quarter."""

    ticker: str
    current: CompanySnapshot
    previous: Optional[CompanySnapshot]
    deltas: dict[str, Optional[float]] = field(default_factory=dict)
    summary: str = ""


__all__ = [
    "CompanySnapshot",
    "MetricTrend",
    "SnapshotComparison",
    "TrendReport",
]
