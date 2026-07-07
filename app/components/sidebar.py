from __future__ import annotations

from typing import Tuple

import streamlit as st

from athena.services.company_search_engine import CompanySearchEngine


@st.cache_resource
def _get_company_search_engine() -> CompanySearchEngine:
    engine = CompanySearchEngine()
    engine.load_company_master()
    return engine


def _format_company_option(option: dict[str, str]) -> str:
    return f"{option['company_name']}      {option['ticker']}"


def _get_company_options() -> list[dict[str, str]]:
    return [
        company.as_dict()
        for company in _get_company_search_engine().load_company_master()
    ]


def _get_selected_index(options: list[dict[str, str]], default_ticker: str) -> int:
    selected_ticker = st.session_state.get("ticker", default_ticker)
    normalized = str(selected_ticker or default_ticker).replace(".NS", "").upper()
    for index, option in enumerate(options):
        if option["ticker"].upper() == normalized:
            return index
    return 0


def render_sidebar(default_ticker: str = "TCS") -> Tuple[str, bool]:
    """Render the dashboard sidebar with cached company autocomplete."""
    with st.sidebar:
        st.header("ATHENA")
        st.caption("AI investment research workspace")

        company_options = _get_company_options()
        selected_company = st.selectbox(
            "Company Search",
            options=company_options,
            index=_get_selected_index(company_options, default_ticker),
            format_func=_format_company_option,
            placeholder="Search by company or ticker",
        )

        ticker = selected_company["ticker"] if selected_company else default_ticker
        st.session_state["ticker"] = ticker.upper()

        if selected_company:
            st.caption(
                f"{selected_company.get('sector', 'Unknown sector')} - "
                f"{selected_company.get('industry', 'Unknown industry')}"
            )
        else:
            st.caption("Select a company to load research.")

        theme_enabled = st.checkbox(
            "Dark Mode",
            value=bool(st.session_state.get("theme_enabled", True)),
        )
        st.session_state["theme_enabled"] = theme_enabled

        st.divider()
        st.caption(
            "Dashboard uses the service layer for data access and "
            "the engine layer for ratios."
        )

    return st.session_state["ticker"], False
