from __future__ import annotations

import pandas as pd
import streamlit as st

from athena.utils.number_formatter import format_crore, format_ratio
from athena.utils.statements import matching_column

# Per-share values must never be converted to crores.
_PER_SHARE_METRICS = {"EPS"}

IMPORTANT_METRICS: dict[str, tuple[str, ...]] = {
    "Income Statement": (
        "Revenue",
        "Gross Profit",
        "EBITDA",
        "EBIT",
        "Operating Income",
        "PAT",
        "EPS",
    ),
    "Balance Sheet": (
        "Assets",
        "Liabilities",
        "Debt",
        "Equity",
        "Cash",
        "Working Capital",
    ),
    "Cash Flow": (
        "Operating Cash Flow",
        "Capital Expenditure",
        "Free Cash Flow",
        "Investing Cash Flow",
        "Financing Cash Flow",
    ),
}

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "Revenue": ("TotalRevenue", "Revenue", "OperatingRevenue"),
    "Gross Profit": ("GrossProfit",),
    "EBITDA": ("EBITDA", "NormalizedEBITDA"),
    "EBIT": ("EBIT",),
    "Operating Income": ("OperatingIncome",),
    "PAT": ("NetIncome", "NetIncomeCommonStockholders", "ProfitAfterTax", "PAT"),
    "EPS": ("DilutedEPS", "BasicEPS", "EPS"),
    "Assets": ("TotalAssets", "Assets"),
    "Liabilities": ("TotalLiabilitiesNetMinorityInterest", "TotalLiabilities"),
    "Debt": ("TotalDebt", "NetDebt", "LongTermDebt"),
    "Equity": ("StockholdersEquity", "TotalEquityGrossMinorityInterest"),
    "Cash": ("CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments"),
    "Working Capital": ("WorkingCapital",),
    "Operating Cash Flow": ("OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"),
    "Capital Expenditure": ("CapitalExpenditure", "CapEx"),
    "Free Cash Flow": ("FreeCashFlow",),
    "Investing Cash Flow": ("InvestingCashFlow", "CashFlowFromContinuingInvestingActivities"),
    "Financing Cash Flow": ("FinancingCashFlow", "CashFlowFromContinuingFinancingActivities"),
}


def _year_index(frame: pd.DataFrame) -> pd.Series:
    if "metric" in frame.columns:
        return pd.to_datetime(frame["metric"], errors="coerce").dt.year
    return pd.Series(range(len(frame)), index=frame.index)


def _build_important_metric_table(title: str, frame: pd.DataFrame) -> pd.DataFrame:
    years = _year_index(frame)
    columns: dict[str, "pd.Series[float]"] = {}
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


def _format_row(metric: str, values: "pd.Series[float]") -> list[str]:
    """Format one metric row for display without mutating source data."""
    formatter = format_ratio if metric in _PER_SHARE_METRICS else format_crore
    return [formatter(value) for value in values]


def render_financial_table(title: str, frame: pd.DataFrame) -> None:
    """Render an important-metrics financial statement table.

    Monetary metrics are shown in crores; per-share metrics (EPS) are left as
    plain ratios. Formatting happens only here and never mutates the source
    DataFrame. Missing values render as ``N/A``.
    """
    st.subheader(title)
    if frame.empty:
        st.info("No data available for this statement.")
        return

    numeric_table = _build_important_metric_table(title, frame.copy())
    if numeric_table.empty:
        st.info("No important metrics are available for this statement.")
        return

    display_table = numeric_table.astype(object)
    for metric in numeric_table.index:
        display_table.loc[metric] = _format_row(metric, numeric_table.loc[metric])

    st.dataframe(display_table, use_container_width=True, hide_index=False)


__all__ = ["render_financial_table"]
