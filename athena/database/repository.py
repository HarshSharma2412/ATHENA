from __future__ import annotations

from typing import Any, Optional

from athena.database.database import DatabaseManager
from athena.database.models import Company, PriceHistory, Ratios


class AthenaRepository:
    """Thin repository façade for common persistence operations."""

    def __init__(self, database_manager: DatabaseManager) -> None:
        self._database_manager = database_manager

    def save_company(self, *, ticker: str, company_name: Optional[str] = None, sector: Optional[str] = None, industry: Optional[str] = None, market_cap: Optional[float] = None, currency: Optional[str] = None, website: Optional[str] = None, country: Optional[str] = None, business_summary: Optional[str] = None) -> Company:
        return self._database_manager.save_company(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            industry=industry,
            market_cap=market_cap,
            currency=currency,
            website=website,
            country=country,
            business_summary=business_summary,
        )

    def load_company(self, ticker: str) -> Optional[Company]:
        return self._database_manager.load_company(ticker)

    def save_price_history(self, ticker: str, history: list[dict[str, Any]]) -> list[PriceHistory]:
        return self._database_manager.save_price_history(ticker=ticker, history=history)

    def load_price_history(self, ticker: str) -> list[PriceHistory]:
        return self._database_manager.load_price_history(ticker)

    def save_financials(self, *, ticker: str, statement_type: str, period: Optional[str], data: dict[str, Any]) -> Any:
        return self._database_manager.save_financials(ticker=ticker, statement_type=statement_type, period=period, data=data)

    def load_financials(self, ticker: str) -> list[dict[str, Any]]:
        return self._database_manager.load_financials(ticker)

    def save_ratios(self, ticker: str, ratios: dict[str, Any]) -> list[Ratios]:
        return self._database_manager.save_ratios(ticker=ticker, ratios=ratios)

    def load_ratios(self, ticker: str) -> list[Ratios]:
        return self._database_manager.load_ratios(ticker)


__all__ = ["AthenaRepository"]
