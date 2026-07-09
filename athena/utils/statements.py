"""Helpers for extracting metrics from yfinance-style financial statements.

The :class:`~athena.services.financial_service.FinancialService` returns each
statement as a long-form frame with a ``metric`` column of period dates and one
column per line item. These helpers locate line items by fuzzy label matching,
build yearly trends for charts, and reshape a statement into the canonical,
engine-friendly frame expected by :class:`~athena.engines.ratio_engine.RatioEngine`.
"""

from __future__ import annotations

import pandas as pd

# Canonical engine labels keyed by the normalized aliases seen in yfinance data.
_ENGINE_ALIASES: dict[str, str] = {
    # Income statement
    "totalrevenue": "Revenue",
    "revenue": "Revenue",
    "operatingrevenue": "Revenue",
    "grossprofit": "GrossProfit",
    "operatingincome": "OperatingIncome",
    "netincome": "NetIncome",
    "netincomecommonstockholders": "NetIncome",
    "netincomecontinuousoperations": "NetIncome",
    "ebit": "EBIT",
    "ebitda": "EBITDA",
    "normalizedebitda": "EBITDA",
    "interestexpense": "InterestExpense",
    # Balance sheet
    "totalassets": "TotalAssets",
    "stockholdersequity": "StockholdersEquity",
    "totalequitygrossminorityinterest": "StockholdersEquity",
    "totalliabilitiesnetminorityinterest": "TotalLiabilities",
    "totalliabilities": "TotalLiabilities",
    "currentassets": "CurrentAssets",
    "totalcurrentassets": "CurrentAssets",
    "currentliabilities": "CurrentLiabilities",
    "totalcurrentliabilities": "CurrentLiabilities",
    "inventory": "Inventory",
    "receivables": "Receivables",
    "accountsreceivable": "Receivables",
    "totaldebt": "TotalDebt",
    "cashandcashequivalents": "CashAndCashEquivalents",
    "cashcashequivalentsandshortterminvestments": "CashAndCashEquivalents",
    # Cash flow
    "operatingcashflow": "OperatingCashFlow",
    "cashflowfromcontinuingoperatingactivities": "OperatingCashFlow",
    "freecashflow": "FreeCashFlow",
    "capitalexpenditure": "CapitalExpenditure",
}


def normalize_label(label: object) -> str:
    """Lower-case a label and strip every non-alphanumeric character."""
    return "".join(character for character in str(label).casefold() if character.isalnum())


def matching_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    """Return the first column matching any alias (exact, then substring)."""
    normalized_aliases = {normalize_label(alias) for alias in aliases}
    for column in frame.columns:
        if normalize_label(column) in normalized_aliases:
            return str(column)
    for column in frame.columns:
        normalized_column = normalize_label(column)
        if any(alias in normalized_column for alias in normalized_aliases):
            return str(column)
    return None


def latest_metric(frame: pd.DataFrame, aliases: tuple[str, ...]) -> float | None:
    """Return the most recent numeric value for the first matching column."""
    if frame.empty:
        return None
    column = matching_column(frame, aliases)
    if column is None:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def build_yearly_trend(frame: pd.DataFrame, aliases: tuple[str, ...]) -> pd.DataFrame:
    """Build a ``year``/``value`` frame for the first matching line item."""
    empty = pd.DataFrame(columns=["year", "value"])
    if frame.empty or "metric" not in frame.columns:
        return empty
    column = matching_column(frame, aliases)
    if column is None:
        return empty

    trend = pd.DataFrame(
        {
            "year": pd.to_datetime(frame["metric"], errors="coerce").dt.year,
            "value": pd.to_numeric(frame[column], errors="coerce"),
        }
    )
    return trend.dropna(subset=["year", "value"]).sort_values("year")


def _order_columns_by_period(frame: pd.DataFrame) -> pd.DataFrame:
    """Order an engine-shaped frame's columns oldest to newest by period date."""
    working = frame.copy()
    parsed = pd.to_datetime(pd.Series(list(working.columns)), errors="coerce")
    if parsed.notna().all():
        ordered = [column for _, column in sorted(zip(parsed, working.columns))]
        working = working.reindex(ordered, axis=1)
    return working


def metric_series(frame: pd.DataFrame | None, aliases: tuple[str, ...]) -> list[float]:
    """Return an oldest-to-newest numeric series for the first matching line item.

    Accepts either the service long-form frame (with a ``metric`` period column)
    or an already engine-shaped frame (metrics as index). Non-numeric cells
    collapse to ``0.0`` so downstream statistics never choke on sparse data.
    """
    if frame is None or frame.empty:
        return []

    engine_frame = to_engine_frame(frame) if "metric" in frame.columns else _order_columns_by_period(frame)
    if engine_frame.empty:
        return []

    normalized_aliases = {normalize_label(alias) for alias in aliases}
    for label in engine_frame.index:
        if normalize_label(label) in normalized_aliases:
            values = pd.to_numeric(engine_frame.loc[label], errors="coerce").fillna(0.0)
            return [float(value) for value in values.tolist()]
    for label in engine_frame.index:
        if any(alias in normalize_label(label) for alias in normalized_aliases):
            values = pd.to_numeric(engine_frame.loc[label], errors="coerce").fillna(0.0)
            return [float(value) for value in values.tolist()]
    return []


def to_engine_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reshape a service statement into a canonical, engine-friendly frame.

    The result is indexed by canonical metric names (matching the labels the
    :class:`RatioEngine` looks for) with one column per period, ordered oldest to
    newest so the engine reads the latest reported value.
    """
    if frame.empty or "metric" not in frame.columns:
        return frame

    working = frame.copy()
    periods = pd.to_datetime(working["metric"], errors="coerce")
    line_items = working.drop(columns=[column for column in ("metric", "ticker") if column in working.columns])

    transposed = line_items.T
    transposed.columns = periods.to_numpy()
    transposed = transposed.loc[:, transposed.columns.notna()]
    transposed = transposed.reindex(sorted(transposed.columns), axis=1)

    rename_map: dict[object, str] = {}
    for label in transposed.index:
        canonical = _ENGINE_ALIASES.get(normalize_label(label))
        if canonical is not None and canonical not in rename_map.values():
            rename_map[label] = canonical

    canonical_frame = transposed.rename(index=rename_map)
    return canonical_frame.loc[canonical_frame.index.isin(rename_map.values())]


__all__ = [
    "build_yearly_trend",
    "latest_metric",
    "matching_column",
    "metric_series",
    "normalize_label",
    "to_engine_frame",
]
