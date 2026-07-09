from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from athena.engines.ratio_engine import FinancialData, RatioEngine
from athena.services.financial_service import FinancialService
from athena.utils.formatting import to_float
from athena.utils.statements import build_yearly_trend, latest_metric, to_engine_frame

from athena.app.components.charts import render_price_history, render_yearly_trend
from athena.app.components.financial_tables import render_financial_table
from athena.app.components.ratio_cards import render_ratio_cards
from athena.app.components.sidebar import render_sidebar
from athena.app.components.summary_cards import render_summary_cards

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompanyFinancialSnapshot:
    """Single dashboard data payload for a selected company."""

    ticker: str
    profile: dict[str, Any]
    fast_info: dict[str, Any]
    history: pd.DataFrame
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame


@st.cache_resource(show_spinner=False)
def _build_services() -> tuple[FinancialService, RatioEngine, Any | None]:
    repository: Any | None = None
    try:
        from athena.database.database import DatabaseManager
        from athena.database.repository import AthenaRepository

        database_manager = DatabaseManager("sqlite:///athena.db")
        repository = AthenaRepository(database_manager)
    except ImportError as exc:
        logger.warning("Database layer unavailable for dashboard cache: %s", exc)
    return FinancialService(), RatioEngine(), repository


@st.cache_data(show_spinner=False, ttl=900)
def _load_company_snapshot(ticker: str) -> CompanyFinancialSnapshot:
    """Download every statement for a ticker exactly once (cached per ticker)."""
    service, _, _ = _build_services()
    normalized_ticker = ticker.strip().upper()
    return CompanyFinancialSnapshot(
        ticker=normalized_ticker,
        profile=service.get_company_profile(normalized_ticker),
        fast_info=service.get_fast_info(normalized_ticker),
        history=service.get_price_history(normalized_ticker),
        income_statement=service.get_income_statement(normalized_ticker),
        balance_sheet=service.get_balance_sheet(normalized_ticker),
        cash_flow=service.get_cash_flow(normalized_ticker),
    )


def _safe_divide(numerator: Any, denominator: Any) -> float | None:
    numerator_value = to_float(numerator)
    denominator_value = to_float(denominator)
    if numerator_value is None or denominator_value is None or denominator_value == 0:
        return None
    return numerator_value / denominator_value


def _build_summary_payload(snapshot: CompanyFinancialSnapshot) -> dict[str, Any]:
    profile = snapshot.profile
    fast_info = snapshot.fast_info
    market_cap = profile.get("market_cap") or fast_info.get("market_cap")
    net_income = latest_metric(snapshot.income_statement, ("NetIncome", "Net Income"))
    equity = latest_metric(
        snapshot.balance_sheet,
        ("StockholdersEquity", "TotalEquityGrossMinorityInterest"),
    )
    return {
        "company_name": profile.get("company_name") or snapshot.ticker.upper(),
        "ticker": snapshot.ticker.upper(),
        "current_price": fast_info.get("current_price"),
        "market_cap": market_cap,
        "pe": _safe_divide(market_cap, net_income),
        "pb": _safe_divide(market_cap, equity),
        "dividend_yield": profile.get("dividend_yield") or fast_info.get("dividend_yield"),
        "fifty_two_week_high": fast_info.get("fifty_two_week_high"),
        "fifty_two_week_low": fast_info.get("fifty_two_week_low"),
    }


def _build_ratio_payload(snapshot: CompanyFinancialSnapshot) -> dict[str, Any]:
    _, ratio_engine, _ = _build_services()
    data = FinancialData(
        income_statement=to_engine_frame(snapshot.income_statement),
        balance_sheet=to_engine_frame(snapshot.balance_sheet),
        cash_flow=to_engine_frame(snapshot.cash_flow),
        fast_info={
            **snapshot.fast_info,
            "shares_outstanding": snapshot.profile.get("shares_outstanding"),
        },
    )
    result = ratio_engine.calculate(data)
    return {
        "roe": result.roe,
        "roce": result.roce,
        "roa": result.roa,
        "debt_equity": result.debt_equity,
        "current_ratio": result.current_ratio,
        "quick_ratio": result.quick_ratio,
        "financial_health_score": result.financial_health_score,
    }


def _build_chart_data(snapshot: CompanyFinancialSnapshot) -> dict[str, pd.DataFrame]:
    return {
        "history": snapshot.history,
        "revenue": build_yearly_trend(snapshot.income_statement, ("TotalRevenue", "Revenue")),
        "net_income": build_yearly_trend(
            snapshot.income_statement,
            ("NetIncome", "Net Income", "PAT", "ProfitAfterTax"),
        ),
        "free_cash_flow": build_yearly_trend(snapshot.cash_flow, ("FreeCashFlow",)),
    }


def _display_company_view(ticker: str) -> None:
    snapshot = _load_company_snapshot(ticker)

    st.subheader(f"Research Overview - {snapshot.ticker.upper()}")
    render_summary_cards(_build_summary_payload(snapshot))

    st.divider()
    render_ratio_cards(_build_ratio_payload(snapshot))

    st.divider()
    chart_data = _build_chart_data(snapshot)
    col1, col2 = st.columns(2)
    with col1:
        render_price_history(chart_data["history"])
    with col2:
        render_yearly_trend(chart_data["revenue"], "Revenue", "#38bdf8")

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        render_yearly_trend(chart_data["net_income"], "PAT", "#22c55e")
    with col4:
        render_yearly_trend(chart_data["free_cash_flow"], "Free Cash Flow", "#f59e0b")

    st.divider()
    render_financial_table("Income Statement", snapshot.income_statement)
    render_financial_table("Balance Sheet", snapshot.balance_sheet)
    render_financial_table("Cash Flow", snapshot.cash_flow)


def main() -> None:
    st.set_page_config(page_title="ATHENA", layout="wide")
    st.title("ATHENA AI Investment Research Platform")

    ticker, _ = render_sidebar(default_ticker="TCS")

    if ticker:
        try:
            with st.spinner("Loading Financial Data..."):
                _display_company_view(ticker)
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.exception("Dashboard failed for %s", ticker)
            st.error(f"Unable to render the dashboard: {exc}")
    else:
        st.info("Select a company to begin research.")


if __name__ == "__main__":
    main()
