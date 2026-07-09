"""CSV-backed company autocomplete engine for the ATHENA dashboard.

The engine loads ``athena/data/company_master.csv`` once into memory and serves
fast, ranked autocomplete queries across every NSE listed company. Results are
ranked by exact symbol, prefix, substring and fuzzy matches, and ties are broken
by index membership and market-cap tier so that well-known large caps surface
first.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from urllib.error import URLError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompanySearchResult:
    """Company metadata returned by the autocomplete search engine."""

    ticker: str
    company_name: str
    sector: str
    industry: str
    isin: str = ""
    market_cap_category: str = "Unknown"
    nifty50: bool = False
    nifty_next50: bool = False
    listing_status: str = "Listed"

    @property
    def symbol(self) -> str:
        """Alias for :attr:`ticker` to match the master CSV schema."""
        return self.ticker

    def as_dict(self) -> dict[str, str]:
        return {
            "ticker": self.ticker,
            "symbol": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "industry": self.industry,
            "isin": self.isin,
            "market_cap_category": self.market_cap_category,
            "nifty50": "true" if self.nifty50 else "false",
            "nifty_next50": "true" if self.nifty_next50 else "false",
            "listing_status": self.listing_status,
        }


class CompanySearchEngine:
    """In-memory, ranked autocomplete over the NSE company master."""

    RESULT_LIMIT: ClassVar[int] = 15

    _CAP_ORDER: ClassVar[dict[str, int]] = {
        "Large Cap": 0,
        "Mid Cap": 1,
        "Small Cap": 2,
        "Micro Cap": 3,
        "Unknown": 4,
    }

    FALLBACK_COMPANIES: ClassVar[tuple[CompanySearchResult, ...]] = (
        CompanySearchResult(
            "HCLTECH", "HCL Technologies Ltd", "Information Technology",
            "IT Services", "INE860A01027", "Large Cap", True, False,
        ),
        CompanySearchResult(
            "HDFCBANK", "HDFC Bank Ltd", "Financial Services",
            "Private Sector Bank", "INE040A01034", "Large Cap", True, False,
        ),
        CompanySearchResult(
            "HCG", "Healthcare Global Enterprises Ltd", "Healthcare",
            "Hospital Services", "INE075I01017", "Micro Cap", False, False,
        ),
        CompanySearchResult(
            "WAAREEENER", "Waaree Energies Ltd", "Capital Goods",
            "Solar Equipment", "INE377N01017", "Mid Cap", False, False,
        ),
        CompanySearchResult(
            "TCS", "Tata Consultancy Services Ltd", "Information Technology",
            "IT Services", "INE467B01029", "Large Cap", True, False,
        ),
        CompanySearchResult(
            "RELIANCE", "Reliance Industries Ltd", "Oil Gas & Consumable Fuels",
            "Refineries", "INE002A01018", "Large Cap", True, False,
        ),
    )

    def __init__(self, company_master_path: Path | None = None) -> None:
        self._company_master_path = company_master_path or self._default_master_path()
        self._companies: list[CompanySearchResult] | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_company_master(self) -> list[CompanySearchResult]:
        """Load and cache the company master, returning the same list instance."""
        if self._companies is not None:
            return self._companies

        if not self._company_master_path.exists():
            logger.info("Company master missing at %s; using fallback set", self._company_master_path)
            self._companies = list(self.FALLBACK_COMPANIES)
            return self._companies

        logger.info("Loading company master from %s", self._company_master_path)
        with self._company_master_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self._companies = [
                company
                for company in (self._row_to_result(row) for row in reader)
                if company is not None
            ]
        if not self._companies:
            logger.warning("Company master at %s was empty; using fallback", self._company_master_path)
            self._companies = list(self.FALLBACK_COMPANIES)
        logger.info("Loaded %s company master records", len(self._companies))
        return self._companies

    def refresh(self) -> list[CompanySearchResult]:
        """Rebuild the master CSV from NSE sources and reload it into memory."""
        try:
            from athena.scripts.update_company_master import build_company_master

            build_company_master(master_path=self._company_master_path)
        except (OSError, URLError, TimeoutError, ValueError, ImportError) as exc:
            logger.warning("Unable to refresh company master: %s", exc)
        self._companies = None
        return self.load_company_master()

    # ------------------------------------------------------------------
    # Search API
    # ------------------------------------------------------------------
    def search(self, query: str) -> list[dict[str, str]]:
        """Return up to ``RESULT_LIMIT`` ranked matches for a ticker or name."""
        return self._ranked_search(query, match_symbol=True, match_name=True)

    def search_by_symbol(self, query: str) -> list[dict[str, str]]:
        """Return ranked matches restricted to the company symbol/ticker."""
        return self._ranked_search(query, match_symbol=True, match_name=False)

    def search_by_company_name(self, query: str) -> list[dict[str, str]]:
        """Return ranked matches restricted to the company name."""
        return self._ranked_search(query, match_symbol=False, match_name=True)

    def _ranked_search(
        self,
        query: str,
        *,
        match_symbol: bool,
        match_name: bool,
    ) -> list[dict[str, str]]:
        normalized_query = self._normalize(query)
        if not normalized_query:
            return []

        scored: list[tuple[int, tuple[int, int, int, str], CompanySearchResult]] = []
        for company in self.load_company_master():
            rank = self._match_rank(company, normalized_query, match_symbol, match_name)
            if rank is not None:
                scored.append((rank, self._popularity_key(company), company))

        scored.sort(key=lambda item: (item[0], item[1]))
        return [company.as_dict() for _, _, company in scored[: self.RESULT_LIMIT]]

    def _match_rank(
        self,
        company: CompanySearchResult,
        query: str,
        match_symbol: bool,
        match_name: bool,
    ) -> int | None:
        symbol = self._normalize(company.ticker) if match_symbol else ""
        name = self._normalize(company.company_name) if match_name else ""
        words = name.split()

        if match_symbol and symbol == query:
            return 0
        if match_name and name == query:
            return 1
        if (match_symbol and symbol.startswith(query)) or (
            match_name and name.startswith(query)
        ):
            return 2
        if match_name and any(word.startswith(query) for word in words):
            return 2
        if (match_symbol and query in symbol) or (match_name and query in name):
            return 3
        if match_symbol and self._is_subsequence(query, symbol.replace(" ", "")):
            return 3
        if match_name and self._acronym(words).startswith(query):
            return 4
        return None

    def _popularity_key(self, company: CompanySearchResult) -> tuple[int, int, int, str]:
        return (
            0 if company.nifty50 else 1,
            0 if company.nifty_next50 else 1,
            self._CAP_ORDER.get(company.market_cap_category, 4),
            company.ticker,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _row_to_result(self, row: dict[str, str]) -> CompanySearchResult | None:
        symbol = (row.get("symbol") or row.get("ticker") or "").strip().upper()
        name = (row.get("company_name") or "").strip()
        if not symbol or not name:
            return None
        return CompanySearchResult(
            ticker=symbol,
            company_name=name,
            sector=(row.get("sector") or "Unknown").strip() or "Unknown",
            industry=(row.get("industry") or "Unknown").strip() or "Unknown",
            isin=(row.get("isin") or "").strip(),
            market_cap_category=(row.get("market_cap_category") or "Unknown").strip() or "Unknown",
            nifty50=self._to_bool(row.get("nifty50")),
            nifty_next50=self._to_bool(row.get("nifty_next50")),
            listing_status=(row.get("listing_status") or "Listed").strip() or "Listed",
        )

    @staticmethod
    def _to_bool(value: str | None) -> bool:
        return str(value or "").strip().lower() in {"true", "1", "yes", "y"}

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(str(value).strip().casefold().split())

    @staticmethod
    def _acronym(words: list[str]) -> str:
        return "".join(word[0] for word in words if word)

    @staticmethod
    def _is_subsequence(query: str, value: str) -> bool:
        if not query:
            return False
        position = 0
        for character in value:
            if character == query[position]:
                position += 1
                if position == len(query):
                    return True
        return False

    @staticmethod
    def _default_master_path() -> Path:
        return Path(__file__).resolve().parents[1] / "data" / "company_master.csv"


__all__ = ["CompanySearchEngine", "CompanySearchResult"]
