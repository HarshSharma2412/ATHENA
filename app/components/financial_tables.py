from __future__ import annotations

import pandas as pd
import streamlit as st

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


def _normalize_label(label: str) -> str:
    return "".join(character for character in str(label).casefold() if character.isalnum())


def _matching_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    normalized_aliases = {_normalize_label(alias) for alias in aliases}
    for column in frame.columns:
        if _normalize_label(str(column)) in normalized_aliases:
            return str(column)
    for column in frame.columns:
        normalized_column = _normalize_label(str(column))
        if any(alias in normalized_column for alias in normalized_aliases):
            return str(column)
    return None


def _year_columns(frame: pd.DataFrame) -> pd.Series:
    if "metric" in frame.columns:
        return pd.to_datetime(frame["metric"], errors="coerce").dt.year
    return pd.Series(frame.index, index=frame.index).astype(str)


def _build_important_metric_table(title: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}
    years = _year_columns(frame)
    for display_name in IMPORTANT_METRICS.get(title, ()):
        column = _matching_column(frame, METRIC_ALIASES[display_name])
        if column is not None:
            values = pd.to_numeric(frame[column], errors="coerce")
            rows[display_name] = pd.Series(values.to_numpy(), index=years)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows).T
    table = table.loc[:, table.columns.notna()]
    table = table.groupby(level=0, axis=1).last()
    table = table.reindex(sorted(table.columns, reverse=True), axis=1)
    return table.dropna(how="all")


def render_financial_table(title: str, frame: pd.DataFrame) -> None:
    """Render an important-metrics financial statement table."""
    st.subheader(title)
    if frame.empty:
        st.info("No data available for this statement.")
        return

    display_frame = _build_important_metric_table(title, frame.copy())
    if display_frame.empty:
        st.info("No important metrics are available for this statement.")
        return

    st.dataframe(
        display_frame.style.format("{:,.2f}", na_rep="-"),
        use_container_width=True,
        hide_index=False,
    )
