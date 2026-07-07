from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompanySearchResult:
    """Company metadata returned by the autocomplete search engine."""

    ticker: str
    company_name: str
    sector: str
    industry: str

    def as_dict(self) -> dict[str, str]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "industry": self.industry,
        }


class CompanySearchEngine:
    """CSV-backed company autocomplete engine for fast in-memory search."""

    NSE_EQUITY_URL: ClassVar[str] = (
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    )
    REQUIRED_COLUMNS: ClassVar[set[str]] = {
        "ticker",
        "company_name",
        "sector",
        "industry",
    }
    CSV_COLUMNS: ClassVar[list[str]] = [
        "ticker",
        "company_name",
        "sector",
        "industry",
    ]
    FALLBACK_COMPANIES: ClassVar[tuple[CompanySearchResult, ...]] = (
        CompanySearchResult(
            "HCLTECH",
            "HCL Technologies Ltd",
            "Information Technology",
            "IT Services",
        ),
        CompanySearchResult(
            "HDFCBANK",
            "HDFC Bank Ltd",
            "Financial Services",
            "Private Sector Bank",
        ),
        CompanySearchResult(
            "HCG",
            "HealthCare Global Enterprises Ltd",
            "Healthcare",
            "Hospital Services",
        ),
        CompanySearchResult(
            "HCL-INSYS",
            "HCL Infosystems Ltd",
            "Information Technology",
            "Technology Hardware",
        ),
        CompanySearchResult(
            "WAAREEENER",
            "Waaree Energies Ltd",
            "Capital Goods",
            "Solar Equipment",
        ),
        CompanySearchResult(
            "TCS",
            "Tata Consultancy Services Ltd",
            "Information Technology",
            "IT Services",
        ),
        CompanySearchResult("INFY", "Infosys Ltd", "Information Technology", "IT Services"),
        CompanySearchResult(
            "RELIANCE",
            "Reliance Industries Ltd",
            "Energy",
            "Oil Gas and Consumable Fuels",
        ),
        CompanySearchResult(
            "KPITTECH",
            "KPIT Technologies Ltd",
            "Information Technology",
            "Software Services",
        ),
        CompanySearchResult(
            "HAPPSTMNDS",
            "Happiest Minds Technologies Ltd",
            "Information Technology",
            "Software Services",
        ),
    )

    def __init__(self, company_master_path: Path | None = None) -> None:
        self._company_master_path = company_master_path or self._default_master_path()
        self._companies: list[CompanySearchResult] | None = None

    def load_company_master(self) -> list[CompanySearchResult]:
        """Load company master data once and return cached records."""
        if self._companies is not None:
            return self._companies

        if not self._company_master_path.exists():
            self.build_company_master()

        logger.info("Loading company master from %s", self._company_master_path)
        with self._company_master_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            self._validate_columns(reader.fieldnames)
            self._companies = [
                CompanySearchResult(
                    ticker=(row.get("ticker") or "").strip().upper(),
                    company_name=(row.get("company_name") or "").strip(),
                    sector=(row.get("sector") or "").strip(),
                    industry=(row.get("industry") or "").strip(),
                )
                for row in reader
                if (row.get("ticker") or "").strip()
                and (row.get("company_name") or "").strip()
            ]
        logger.info("Loaded %s company master records", len(self._companies))
        return self._companies

    def build_company_master(self, force: bool = False) -> Path:
        """Build and cache company_master.csv from NSE equity symbols."""
        if self._company_master_path.exists() and not force:
            return self._company_master_path

        try:
            companies = self._download_nse_equity_master()
        except (OSError, URLError, TimeoutError, ValueError) as exc:
            logger.warning("Unable to build company master from NSE: %s", exc)
            companies = list(self.FALLBACK_COMPANIES)

        self._company_master_path.parent.mkdir(parents=True, exist_ok=True)
        with self._company_master_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            for company in companies:
                writer.writerow(company.as_dict())

        self._companies = companies
        logger.info("Cached %s companies at %s", len(companies), self._company_master_path)
        return self._company_master_path

    def search(self, query: str) -> list[dict[str, str]]:
        """Return the top 15 company matches for a ticker or company query."""
        normalized_query = self._normalize(query)
        if not normalized_query:
            return []

        ranked_matches: list[tuple[int, int, CompanySearchResult]] = []
        for index, company in enumerate(self.load_company_master()):
            rank = self._match_rank(company, normalized_query)
            if rank is not None:
                ranked_matches.append((rank, index, company))

        ranked_matches.sort(key=lambda item: (item[0], item[1]))
        return [company.as_dict() for _, _, company in ranked_matches[:15]]

    def _match_rank(
        self,
        company: CompanySearchResult,
        normalized_query: str,
    ) -> int | None:
        ticker = self._normalize(company.ticker)
        name = self._normalize(company.company_name)
        words = name.split()

        if normalized_query in {ticker, name}:
            return 0
        if ticker.startswith(normalized_query) or name.startswith(normalized_query):
            return 1
        if any(word.startswith(normalized_query) for word in words):
            return 1
        if self._is_subsequence(normalized_query, ticker):
            return 1
        if normalized_query in ticker or normalized_query in name:
            return 2
        return None

    def _validate_columns(self, fieldnames: list[str] | None) -> None:
        columns = set(fieldnames or [])
        missing_columns = self.REQUIRED_COLUMNS - columns
        if missing_columns:
            raise ValueError(
                "company_master.csv is missing required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

    def _download_nse_equity_master(self) -> list[CompanySearchResult]:
        request = Request(
            self.NSE_EQUITY_URL,
            headers={
                "User-Agent": "Mozilla/5.0 ATHENA Investment Research",
                "Accept": "text/csv,*/*",
            },
        )
        with urlopen(request, timeout=12) as response:
            payload = response.read().decode("utf-8-sig")

        reader = csv.DictReader(payload.splitlines())
        companies = [
            CompanySearchResult(
                ticker=(row.get("SYMBOL") or "").strip().upper(),
                company_name=(row.get("NAME OF COMPANY") or "").strip(),
                sector=(row.get("SECTOR") or row.get("MACRO") or "Unknown").strip(),
                industry=(row.get("INDUSTRY") or row.get("BASIC INDUSTRY") or "Unknown").strip(),
            )
            for row in reader
            if (row.get("SYMBOL") or "").strip()
            and (row.get("NAME OF COMPANY") or "").strip()
            and (row.get("SERIES") or "EQ").strip().upper() == "EQ"
        ]
        if not companies:
            raise ValueError("NSE equity master returned no EQ companies")
        return companies

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.strip().casefold().split())

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
        return Path(__file__).resolve().parents[2] / "company_master.csv"


__all__ = ["CompanySearchEngine", "CompanySearchResult"]
