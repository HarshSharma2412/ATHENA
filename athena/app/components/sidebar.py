from __future__ import annotations

from typing import Tuple

import streamlit as st

from athena.services.company_search_engine import CompanySearchEngine


@st.cache_resource(show_spinner=False)
def _get_company_search_engine() -> CompanySearchEngine:
    """Build the search engine once and keep it cached across reruns."""
    engine = CompanySearchEngine()
    engine.load_company_master()
    return engine


def _format_option(result: dict[str, str]) -> str:
    sector = result.get("sector") or "Unknown"
    return f"{result['ticker']}  \u2014  {result['company_name']}  \u00b7  {sector}"


def _resolve_selection(query: str, engine: CompanySearchEngine) -> dict[str, str] | None:
    results = engine.search(query)
    if not results:
        st.warning("No matching companies found.")
        return None

    choice_index = st.selectbox(
        "Matches",
        options=range(len(results)),
        format_func=lambda index: _format_option(results[index]),
        key="company_choice",
    )
    return results[choice_index]


def render_sidebar(default_ticker: str = "TCS") -> Tuple[str, bool]:
    """Render the sidebar autocomplete and return the selected ticker."""
    with st.sidebar:
        st.header("ATHENA")
        st.caption("AI investment research workspace")

        engine = _get_company_search_engine()
        query = st.text_input(
            "Search company or ticker",
            key="company_query",
            placeholder="Search across all NSE companies (e.g. HDFC, TCS, waaree)",
        )

        selected = _resolve_selection(query, engine) if query.strip() else None

        if selected:
            ticker = selected["ticker"].upper()
            st.session_state["ticker"] = ticker
            st.caption(
                f"**{selected['company_name']}**  \n"
                f"{selected.get('sector', 'Unknown')} \u00b7 {selected.get('industry', 'Unknown')}"
            )
        else:
            ticker = str(st.session_state.get("ticker", default_ticker)).upper()
            st.session_state["ticker"] = ticker
            st.caption("Start typing to search across every NSE listed company.")

        st.divider()
        st.caption(
            "Company data is served from a cached NSE master; financials load "
            "once per company via the service layer."
        )

    return st.session_state["ticker"], False


__all__ = ["render_sidebar"]
