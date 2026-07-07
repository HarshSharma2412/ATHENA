from __future__ import annotations

import pandas as pd
import streamlit as st

from athena.utils.statements import matching_column

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

    st.dataframe(
        display_frame.style.format("{:,.2f}", na_rep="N/A"),
        use_container_width=True,
        hide_index=False,
    )


__all__ = ["render_financial_table"]
