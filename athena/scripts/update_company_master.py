"""Build and refresh the ATHENA NSE company master database.

Running this module downloads the official NSE equity list plus the public
index constituent files and regenerates ``athena/data/company_master.csv``.

Usage::

    python athena/scripts/update_company_master.py
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_MASTER_PATH = Path(__file__).resolve().parents[1] / "data" / "company_master.csv"

CSV_COLUMNS: tuple[str, ...] = (
    "symbol",
    "company_name",
    "sector",
    "industry",
    "isin",
    "market_cap_category",
    "nifty50",
    "nifty_next50",
    "listing_status",
)

_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (ATHENA Investment Research)",
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class IndexSource:
    """A downloadable NSE index constituent file used for enrichment."""

    name: str
    url: str
    market_cap_category: str | None = None
    is_nifty50: bool = False
    is_nifty_next50: bool = False


@dataclass
class CompanyRecord:
    """A single company row in the master database."""

    symbol: str
    company_name: str
    isin: str = ""
    sector: str = "Unknown"
    industry: str = "Unknown"
    market_cap_category: str = "Unknown"
    nifty50: bool = False
    nifty_next50: bool = False
    listing_status: str = "Listed"

    def as_row(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "sector": self.sector,
            "industry": self.industry,
            "isin": self.isin,
            "market_cap_category": self.market_cap_category,
            "nifty50": "true" if self.nifty50 else "false",
            "nifty_next50": "true" if self.nifty_next50 else "false",
            "listing_status": self.listing_status,
        }


class CompanyMasterBuilder:
    """Download NSE listings and build the company master database."""

    NSE_EQUITY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    # Ordered from broadest cap tier to narrowest so later (smaller) tiers do
    # not overwrite a large-cap classification that was set earlier.
    INDEX_SOURCES: tuple[IndexSource, ...] = (
        IndexSource(
            "Nifty 500",
            "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        ),
        IndexSource(
            "Nifty Total Market",
            "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
        ),
        IndexSource(
            "Nifty Microcap 250",
            "https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
            market_cap_category="Micro Cap",
        ),
        IndexSource(
            "Nifty Smallcap 250",
            "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
            market_cap_category="Small Cap",
        ),
        IndexSource(
            "Nifty Midcap 150",
            "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
            market_cap_category="Mid Cap",
        ),
        IndexSource(
            "Nifty Next 50",
            "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
            market_cap_category="Large Cap",
            is_nifty_next50=True,
        ),
        IndexSource(
            "Nifty 50",
            "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
            market_cap_category="Large Cap",
            is_nifty50=True,
        ),
    )

    def __init__(self, master_path: Path | None = None, timeout: int = 20) -> None:
        self._master_path = master_path or DEFAULT_MASTER_PATH
        self._timeout = timeout

    def build(self) -> Path:
        """Download, enrich and persist the company master, returning the path."""
        records = self._download_equity_master()
        logger.info("Downloaded %s base NSE equities", len(records))

        for source in self.INDEX_SOURCES:
            try:
                self._apply_index_source(records, source)
            except (OSError, URLError, TimeoutError, ValueError) as exc:
                logger.warning("Skipping index source %s: %s", source.name, exc)

        self._write(records.values())
        logger.info("Wrote %s companies to %s", len(records), self._master_path)
        return self._master_path

    def _download_equity_master(self) -> dict[str, CompanyRecord]:
        payload = self._fetch(self.NSE_EQUITY_URL)
        reader = csv.DictReader(StringIO(payload))
        records: dict[str, CompanyRecord] = {}
        for row in reader:
            normalized = {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            symbol = normalized.get("SYMBOL", "").upper()
            name = normalized.get("NAME OF COMPANY", "")
            series = normalized.get("SERIES", "EQ").upper()
            if not symbol or not name or series != "EQ":
                continue
            records[symbol] = CompanyRecord(
                symbol=symbol,
                company_name=self._normalize_name(name),
                isin=normalized.get("ISIN NUMBER", ""),
            )
        if not records:
            raise ValueError("NSE equity master returned no EQ companies")
        return dict(sorted(records.items()))

    def _apply_index_source(
        self,
        records: dict[str, CompanyRecord],
        source: IndexSource,
    ) -> None:
        payload = self._fetch(source.url)
        reader = csv.DictReader(StringIO(payload))
        applied = 0
        for row in reader:
            normalized = {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            symbol = normalized.get("Symbol", "").upper()
            if not symbol:
                continue
            record = records.get(symbol)
            if record is None:
                record = CompanyRecord(
                    symbol=symbol,
                    company_name=self._normalize_name(normalized.get("Company Name", symbol)),
                    isin=normalized.get("ISIN Code", ""),
                )
                records[symbol] = record

            industry = normalized.get("Industry", "")
            if industry:
                record.sector = industry
                if record.industry == "Unknown":
                    record.industry = industry
            if source.market_cap_category:
                record.market_cap_category = source.market_cap_category
            if source.is_nifty50:
                record.nifty50 = True
            if source.is_nifty_next50:
                record.nifty_next50 = True
            applied += 1
        logger.info("Applied %s rows from %s", applied, source.name)

    def _write(self, records: Iterable[CompanyRecord]) -> None:
        self._master_path.parent.mkdir(parents=True, exist_ok=True)
        with self._master_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for record in sorted(records, key=lambda item: item.symbol):
                writer.writerow(record.as_row())

    def _fetch(self, url: str) -> str:
        request = Request(url, headers=_REQUEST_HEADERS)
        with urlopen(request, timeout=self._timeout) as response:  # noqa: S310 - trusted NSE host
            return response.read().decode("utf-8-sig")

    @staticmethod
    def _normalize_name(name: str) -> str:
        cleaned = " ".join(name.strip().split())
        for suffix in (" Limited", " LIMITED", " limited"):
            if cleaned.endswith(suffix):
                return f"{cleaned[: -len(suffix)]} Ltd"
        return cleaned


def build_company_master(master_path: Path | None = None) -> Path:
    """Convenience wrapper used by callers and the CLI entry point."""
    return CompanyMasterBuilder(master_path=master_path).build()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the ATHENA NSE company master database.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MASTER_PATH,
        help="Destination CSV path (defaults to athena/data/company_master.csv).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = build_company_master(master_path=args.output)
    logger.info("Company master ready at %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
