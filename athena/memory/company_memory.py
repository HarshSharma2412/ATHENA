"""ATHENA's Company Memory engine.

Every analysis ATHENA produces can be persisted as a :class:`CompanySnapshot`
so the platform *remembers* how its view of a company has evolved. Storage is
plain ``sqlite3`` (stdlib, zero extra dependencies) - a file path for durable
use or ``":memory:"`` for tests.

Capabilities:
* ``save_snapshot`` - persist one analysis.
* ``load_latest`` / ``load_history`` - read the most recent or the full series.
* ``compare`` - current quarter versus the previous quarter.
* ``trend`` - trajectory of business score, DCF intrinsic value, risk,
  management and price across every snapshot.

No Streamlit / presentation code.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Optional

from athena.engines.athena_engine import AthenaAnalysis
from athena.models.company_snapshot import (
    CompanySnapshot,
    MetricTrend,
    SnapshotComparison,
    TrendReport,
)

logger = logging.getLogger(__name__)

_JSON_COLUMNS = ("dcf", "reverse_dcf", "investment_thesis")

# Metric name -> (snapshot attribute, higher_is_better) used by trend/compare.
_METRICS: dict[str, tuple[str, bool]] = {
    "business_score": ("business_score", True),
    "intrinsic_value": ("intrinsic_value", True),
    "risk_score": ("risk_score", False),
    "management_score": ("management_score", True),
    "current_price": ("current_price", True),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS company_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    current_price REAL,
    intrinsic_value REAL,
    business_score REAL,
    risk_score REAL,
    management_score REAL,
    recommendation TEXT,
    dcf TEXT NOT NULL DEFAULT '{}',
    reverse_dcf TEXT NOT NULL DEFAULT '{}',
    investment_thesis TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""

_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_date "
    "ON company_snapshots (ticker, snapshot_date)"
)


def snapshot_from_analysis(
    analysis: AthenaAnalysis,
    *,
    management_score: Optional[float] = None,
    date: Optional[datetime] = None,
) -> CompanySnapshot:
    """Build a :class:`CompanySnapshot` from a full :class:`AthenaAnalysis`."""
    dcf = analysis.dcf
    reverse = analysis.reverse_dcf
    thesis = analysis.investment_thesis
    recommendation = thesis.investment_rating.value if thesis is not None else None
    return CompanySnapshot(
        ticker=analysis.ticker,
        date=date or analysis.generated_at,
        current_price=analysis.current_price,
        intrinsic_value=dcf.intrinsic_value if dcf is not None else None,
        business_score=(
            analysis.business_quality.overall_score if analysis.business_quality is not None else None
        ),
        risk_score=analysis.risk.overall_risk_score if analysis.risk is not None else None,
        management_score=management_score,
        recommendation=recommendation,
        dcf=_dcf_summary(dcf),
        reverse_dcf=_reverse_summary(reverse),
        investment_thesis=_thesis_summary(thesis),
    )


class CompanyMemory:
    """SQLite-backed store of every historical company analysis."""

    def __init__(
        self,
        database_path: str = "athena_memory.db",
        *,
        connection: Optional[sqlite3.Connection] = None,
    ) -> None:
        if connection is not None:
            self._conn = connection
        else:
            self._conn = sqlite3.connect(database_path)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    # -------------------------------------------------------------- public API
    def save_snapshot(self, snapshot: CompanySnapshot) -> CompanySnapshot:
        """Persist a snapshot and return it with its assigned ``snapshot_id``."""
        if not isinstance(snapshot, CompanySnapshot):
            raise TypeError("snapshot must be a CompanySnapshot")
        ticker = snapshot.ticker.strip().upper()
        if not ticker:
            raise ValueError("snapshot.ticker must not be empty")

        cursor = self._conn.execute(
            """
            INSERT INTO company_snapshots (
                ticker, snapshot_date, current_price, intrinsic_value,
                business_score, risk_score, management_score, recommendation,
                dcf, reverse_dcf, investment_thesis, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                snapshot.date.isoformat(),
                snapshot.current_price,
                snapshot.intrinsic_value,
                snapshot.business_score,
                snapshot.risk_score,
                snapshot.management_score,
                snapshot.recommendation,
                json.dumps(snapshot.dcf),
                json.dumps(snapshot.reverse_dcf),
                json.dumps(snapshot.investment_thesis),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        logger.info("Saved snapshot for %s (id=%s)", ticker, cursor.lastrowid)
        return replace(snapshot, ticker=ticker, snapshot_id=cursor.lastrowid)

    def load_latest(self, ticker: str) -> Optional[CompanySnapshot]:
        """Return the most recent snapshot for ``ticker`` (or ``None``)."""
        row = self._conn.execute(
            """
            SELECT * FROM company_snapshots
            WHERE ticker = ?
            ORDER BY snapshot_date DESC, id DESC
            LIMIT 1
            """,
            (self._normalize(ticker),),
        ).fetchone()
        return self._to_snapshot(row) if row is not None else None

    def load_history(self, ticker: str, *, limit: Optional[int] = None) -> list[CompanySnapshot]:
        """Return every snapshot for ``ticker``, oldest first."""
        query = (
            "SELECT * FROM company_snapshots WHERE ticker = ? "
            "ORDER BY snapshot_date ASC, id ASC"
        )
        params: tuple[Any, ...] = (self._normalize(ticker),)
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._to_snapshot(row) for row in rows]

    def compare(self, ticker: str) -> Optional[SnapshotComparison]:
        """Compare the latest snapshot against the most recent earlier quarter."""
        history = self.load_history(ticker)
        if not history:
            return None
        current = history[-1]
        previous = self._previous_quarter_snapshot(history, current)
        deltas = self._deltas(current, previous)
        return SnapshotComparison(
            ticker=current.ticker,
            current=current,
            previous=previous,
            deltas=deltas,
            summary=self._comparison_summary(current, previous, deltas),
        )

    def trend(self, ticker: str) -> TrendReport:
        """Return per-metric trends across every stored snapshot."""
        history = self.load_history(ticker)
        normalized = self._normalize(ticker)
        return TrendReport(
            ticker=normalized,
            business_score=self._metric_trend(history, "business_score"),
            intrinsic_value=self._metric_trend(history, "intrinsic_value"),
            risk_score=self._metric_trend(history, "risk_score"),
            management_score=self._metric_trend(history, "management_score"),
            current_price=self._metric_trend(history, "current_price"),
        )

    # ------------------------------------------------------------- internals
    def _create_schema(self) -> None:
        self._conn.execute(_SCHEMA)
        self._conn.execute(_INDEX)
        self._conn.commit()

    @staticmethod
    def _normalize(ticker: str) -> str:
        if not isinstance(ticker, str):
            raise TypeError("ticker must be a string")
        return ticker.strip().upper()

    @staticmethod
    def _to_snapshot(row: sqlite3.Row) -> CompanySnapshot:
        payload = {column: json.loads(row[column]) for column in _JSON_COLUMNS}
        return CompanySnapshot(
            ticker=row["ticker"],
            date=datetime.fromisoformat(row["snapshot_date"]),
            current_price=row["current_price"],
            intrinsic_value=row["intrinsic_value"],
            business_score=row["business_score"],
            risk_score=row["risk_score"],
            management_score=row["management_score"],
            recommendation=row["recommendation"],
            dcf=payload["dcf"],
            reverse_dcf=payload["reverse_dcf"],
            investment_thesis=payload["investment_thesis"],
            snapshot_id=row["id"],
        )

    @staticmethod
    def _previous_quarter_snapshot(
        history: list[CompanySnapshot], current: CompanySnapshot
    ) -> Optional[CompanySnapshot]:
        for snapshot in reversed(history[:-1]):
            if snapshot.quarter != current.quarter:
                return snapshot
        return history[-2] if len(history) >= 2 else None

    @staticmethod
    def _deltas(
        current: CompanySnapshot, previous: Optional[CompanySnapshot]
    ) -> dict[str, Optional[float]]:
        deltas: dict[str, Optional[float]] = {}
        for metric, (attribute, _) in _METRICS.items():
            current_value = _metric_value(current, attribute)
            previous_value = _metric_value(previous, attribute) if previous is not None else None
            if current_value is None or previous_value is None:
                deltas[metric] = None
            else:
                deltas[metric] = current_value - previous_value
        return deltas

    def _comparison_summary(
        self,
        current: CompanySnapshot,
        previous: Optional[CompanySnapshot],
        deltas: dict[str, Optional[float]],
    ) -> str:
        if previous is None:
            return f"{current.ticker}: first recorded snapshot; no prior quarter to compare."
        parts: list[str] = []
        for metric, (_, higher_is_better) in _METRICS.items():
            change = deltas.get(metric)
            if change is None or change == 0.0:
                continue
            improved = (change > 0) == higher_is_better
            verb = "improved" if improved else "worsened"
            parts.append(f"{metric.replace('_', ' ')} {verb} by {abs(change):.2f}")
        body = "; ".join(parts) if parts else "no material change"
        return f"{current.ticker} vs previous quarter: {body}."

    def _metric_trend(self, history: list[CompanySnapshot], metric: str) -> MetricTrend:
        attribute, higher_is_better = _METRICS[metric]
        points: list[tuple[datetime, float]] = []
        for snapshot in history:
            value = _metric_value(snapshot, attribute)
            if value is not None:
                points.append((snapshot.date, float(value)))
        if not points:
            return MetricTrend(metric=metric)
        first = points[0][1]
        last = points[-1][1]
        change = last - first
        return MetricTrend(
            metric=metric,
            points=points,
            first=first,
            last=last,
            change=change,
            direction=self._direction(change, higher_is_better),
        )

    @staticmethod
    def _direction(change: float, higher_is_better: bool) -> str:
        if abs(change) < 1e-9:
            return "Stable"
        improved = (change > 0) == higher_is_better
        return "Improving" if improved else "Deteriorating"


def _metric_value(snapshot: CompanySnapshot, attribute: str) -> Optional[float]:
    """Read one of the numeric snapshot metrics by name via an explicit mapping."""
    mapping: dict[str, Optional[float]] = {
        "current_price": snapshot.current_price,
        "intrinsic_value": snapshot.intrinsic_value,
        "business_score": snapshot.business_score,
        "risk_score": snapshot.risk_score,
        "management_score": snapshot.management_score,
    }
    return mapping[attribute]


def _dcf_summary(dcf: Any) -> dict[str, Any]:
    if dcf is None:
        return {}
    return {
        "intrinsic_value": dcf.intrinsic_value,
        "margin_of_safety": dcf.margin_of_safety,
        "expected_cagr": dcf.expected_cagr,
        "recommendation": dcf.recommendation.value,
        "confidence_score": dcf.confidence_score,
    }


def _reverse_summary(reverse: Any) -> dict[str, Any]:
    if reverse is None:
        return {}
    return {
        "required_growth": reverse.required_growth,
        "historical_growth": reverse.historical_growth,
        "expectation_score": reverse.expectation_score,
        "expectation_level": reverse.expectation_level.value,
        "valuation_risk": reverse.valuation_risk.value,
    }


def _thesis_summary(thesis: Any) -> dict[str, Any]:
    if thesis is None:
        return {}
    return {
        "investment_rating": thesis.investment_rating.value,
        "overall_score": thesis.overall_score,
        "confidence_score": thesis.confidence_score,
        "expected_return_3y": thesis.expected_return_3y,
        "expected_return_5y": thesis.expected_return_5y,
        "margin_of_safety": thesis.margin_of_safety,
    }


__all__ = ["CompanyMemory", "snapshot_from_analysis"]
