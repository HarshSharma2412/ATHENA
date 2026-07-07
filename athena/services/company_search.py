from __future__ import annotations

from typing import Dict, List, Optional


class CompanySearchService:
    """Search service for resolving company names to NSE tickers."""

    _company_map: Dict[str, Dict[str, Optional[str]]] = {
        "TCS": {
            "name": "Tata Consultancy Services Limited",
            "ticker": "TCS.NS",
            "exchange": "NSE",
            "sector": "Information Technology",
        },
        "TATA CONSULTANCY SERVICES": {
            "name": "Tata Consultancy Services Limited",
            "ticker": "TCS.NS",
            "exchange": "NSE",
            "sector": "Information Technology",
        },
        "WAAREE": {
            "name": "Waaree Energies Limited",
            "ticker": "WAAREEENER.NS",
            "exchange": "NSE",
            "sector": "Energy",
        },
        "RELIANCE": {
            "name": "Reliance Industries Limited",
            "ticker": "RELIANCE.NS",
            "exchange": "NSE",
            "sector": "Energy",
        },
        "KPIT": {
            "name": "KPIT Technologies Limited",
            "ticker": "KPITTECH.NS",
            "exchange": "NSE",
            "sector": "Information Technology",
        },
        "HAPPIEST MINDS": {
            "name": "Happiest Minds Technologies Limited",
            "ticker": "HAPPSTMNDS.NS",
            "exchange": "NSE",
            "sector": "Information Technology",
        },
    }

    def __init__(self) -> None:
        self._cache: Dict[str, Optional[Dict[str, Optional[str]]]] = {}

    def _normalize_query(self, query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        return query.strip().upper()

    def _build_company_result(
        self, metadata: Dict[str, Optional[str]]
    ) -> Dict[str, Optional[str]]:
        return {
            "company_name": metadata.get("name"),
            "ticker": metadata.get("ticker"),
            "exchange": metadata.get("exchange"),
            "sector": metadata.get("sector"),
        }

    def _find_company_metadata(
        self, normalized_query: str
    ) -> Optional[Dict[str, Optional[str]]]:
        if normalized_query in self._company_map:
            return self._company_map[normalized_query]

        for key, metadata in self._company_map.items():
            if normalized_query in key:
                return metadata

            company_name = metadata.get("name") or ""
            if normalized_query in company_name.upper():
                return metadata

            ticker = metadata.get("ticker") or ""
            if normalized_query == ticker.replace(".NS", ""):
                return metadata

        return None

    def search_company(self, query: str) -> Dict[str, Optional[str]]:
        """Search for a company by name or ticker alias."""
        normalized_query = self._normalize_query(query)
        if normalized_query in self._cache:
            return self._cache[normalized_query] or {
                "company_name": None,
                "ticker": None,
                "exchange": None,
                "sector": None,
            }

        result = self._find_company_metadata(normalized_query)
        if result is None:
            empty_result = {
                "company_name": None,
                "ticker": None,
                "exchange": None,
                "sector": None,
            }
            self._cache[normalized_query] = None
            return empty_result

        company_result = self._build_company_result(result)
        self._cache[normalized_query] = company_result
        return company_result

    def resolve_ticker(self, query: str) -> Optional[str]:
        """Resolve a query into a normalized NSE ticker symbol."""
        result = self.search_company(query)
        return result.get("ticker")
