from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from athena.utils.formatting import format_indian_grouping, NOT_AVAILABLE
from athena.utils.statements import matching_column

_CRORE = 10_000_000.0

# Metrics that are per-share values and must NOT be converted to crores.
_PER_SHARE_METRICS = {"EPS"}

IMPORTANT_METRICS: dict[str, tuple[str, ...]] = {
    "Income Statement": ("Revenue", "EBIT", "PAT", "EPS"),
    "Balance Sheet": ("Assets", "Liabilities", "Debt", "Equity"),
    "Cash Flow": ("Operating Cash Flow", "CapEx", "Free Cash Flow"),
}

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "Revenue": ("TotalRevenue", "Revenue"),
    "EBIT": ("EBIT", "OperatingIncome"),
    "PAT": ("NetIncome", "Net Income", "ProfitAfterTax", "PAT"),
    "EPS": ("DilutedEPS", "BasicEPS", "EPS"),
    "Assets": ("TotalAssets", "Assets"),
    "Liabilities": ("TotalLiabilitiesNetMinorityInterest", "TotalLiabilities"),
    "Debt": ("TotalDebt", "NetDebt", "LongTermDebt"),
    "Equity": ("StockholdersEquity", "TotalEquityGrossMinorityInterest"),
    "Operating Cash Flow": ("OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"),
    "CapEx": ("CapitalExpenditure", "CapEx"),
    "Free Cash Flow": ("FreeCashFlow",),
}


def _year_index(frame: pd.DataFrame) -> pd.Series:
    if "metric" in frame.columns:
        return pd.to_datetime(frame["metric"], errors="coerce").dt.year
    return pd.Series(range(len(frame)), index=frame.index)


def _build_important_metric_table(title: str, frame: pd.DataFrame) -> pd.DataFrame:
    years = _year_index(frame)
    columns: dict[str, pd.Series] = {}
    for display_name in IMPORTANT_METRICS.get(title, ()):
        column = matching_column(frame, METRIC_ALIASES[display_name])
        if column is not None:
            columns[display_name] = pd.to_numeric(frame[column], errors="coerce").to_numpy()

    if not columns:
        return pd.DataFrame()

    per_year = pd.DataFrame(columns, index=years)
    per_year = per_year[per_year.index.notna()]
    if per_year.empty:
        return pd.DataFrame()

    per_year = per_year.groupby(level=0).last()
    table = per_year.T
    table = table.reindex(sorted(table.columns, reverse=True), axis=1)
    table.columns = [str(int(year)) for year in table.columns]
    return table.dropna(how="all")


def render_financial_table(title: str, frame: pd.DataFrame) -> None:
    """Render an important-metrics financial statement table without ``nan``."""
    st.subheader(title)
    if frame.empty:
        st.info("No data available for this statement.")
        return

    display_frame = _build_important_metric_table(title, frame.copy())
    if display_frame.empty:
        st.info("No important metrics are available for this statement.")
        return

    # Convert raw rupee values to crores (skip per-share metrics like EPS)
    for metric in display_frame.index:
        if metric not in _PER_SHARE_METRICS:
            display_frame.loc[metric] = display_frame.loc[metric] / _CRORE

    def _fmt_cell(value: object, metric: str) -> str:
        """Format a single cell: crore values get ₹…Cr, EPS stays plain."""
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return NOT_AVAILABLE
        try:
            num = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return NOT_AVAILABLE
        if metric in _PER_SHARE_METRICS:
            return f"{num:,.2f}"
        return f"\u20b9{format_indian_grouping(num, 2)} Cr"

    styled = display_frame.copy().astype(object)
    for metric in styled.index:
        for col in styled.columns:
            styled.at[metric, col] = _fmt_cell(display_frame.at[metric, col], metric)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=False,
    )


__all__ = ["render_financial_table"]
