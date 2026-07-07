from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class FinancialService:
    """Production-ready financial service for investment research workflows."""

    def __init__(self, session: Optional[Any] = None, retries: int = 3) -> None:
        self._session = session or yf
        self._retries = retries
        self._cache: Dict[str, Any] = {}

    def normalize_ticker(self, symbol: str) -> str:
        """Normalize an NSE symbol by uppercasing and appending .NS if needed."""
        if not isinstance(symbol, str):
            raise TypeError("symbol must be a string")

        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")

        if not normalized.endswith(".NS"):
            normalized = f"{normalized}.NS"

        return normalized

    def _get_ticker(self, symbol: str) -> Any:
        return self._session.Ticker(self.normalize_ticker(symbol))

    def _request_with_retries(self, symbol: str, attribute: str, *args: Any, **kwargs: Any) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(self._retries):
            try:
                stock = self._get_ticker(symbol)
                value = getattr(stock, attribute, None)
                if callable(value):
                    return value(*args, **kwargs)
                if value is not None:
                    return value
                raise AttributeError(f"{attribute} is unavailable for {symbol}")
            except Exception as exc:  # pragma: no cover - defensive branch
                last_error = exc
                logger.warning(
                    "Retry %s/%s for %s failed: %s",
                    attempt + 1,
                    self._retries,
                    symbol,
                    exc,
                )
        raise RuntimeError(f"Unable to fetch {attribute} for {symbol}: {last_error}") from last_error

    def _get_cached_or_fetch(self, cache_key: str, fetcher: Any) -> Any:
        if cache_key in self._cache:
            return self._cache[cache_key]

        value = fetcher()
        self._cache[cache_key] = value
        return value

    def _safe_error_payload(self, symbol: str, exc: Exception) -> Dict[str, Any]:
        return {
            "ticker": self.normalize_ticker(symbol) if isinstance(symbol, str) and symbol.strip() else "INVALID",
            "error": str(exc),
        }

    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Return a company profile dictionary for the supplied symbol."""
        try:
            normalized = self.normalize_ticker(symbol)
            return self._get_cached_or_fetch(
                f"profile:{normalized}",
                lambda: self._get_company_profile_payload(normalized),
            )
        except Exception as exc:
            logger.exception("Failed to fetch company profile for %s", symbol)
            return self._safe_error_payload(symbol, exc)

    def _get_company_profile_payload(self, symbol: str) -> Dict[str, Any]:
        stock = self._get_ticker(symbol)
        info = getattr(stock, "info", {}) or {}
        return {
            "company_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "business_summary": info.get("longBusinessSummary"),
            "website": info.get("website"),
            "country": info.get("country"),
            "employees": info.get("fullTimeEmployees"),
            "ticker": symbol,
        }

    def get_fast_info(self, symbol: str) -> Dict[str, Any]:
        """Return a compact fast-info dictionary for the supplied symbol."""
        try:
            normalized = self.normalize_ticker(symbol)
            return self._get_cached_or_fetch(
                f"fast_info:{normalized}",
                lambda: self._get_fast_info_payload(normalized),
            )
        except Exception as exc:
            logger.exception("Failed to fetch fast info for %s", symbol)
            return self._safe_error_payload(symbol, exc)

    def _get_fast_info_payload(self, symbol: str) -> Dict[str, Any]:
        stock = self._get_ticker(symbol)
        fast_info = getattr(stock, "fast_info", None)
        if fast_info is None:
            return {"ticker": symbol, "error": "Fast info unavailable"}

        return {
            "current_price": getattr(fast_info, "last_price", None),
            "fifty_two_week_high": getattr(fast_info, "year_high", None),
            "fifty_two_week_low": getattr(fast_info, "year_low", None),
            "volume": getattr(fast_info, "day_volume", None),
            "average_volume": getattr(fast_info, "average_volume", None),
            "market_cap": getattr(fast_info, "market_cap", None),
            "currency": getattr(fast_info, "currency", None),
            "exchange": getattr(fast_info, "exchange", None),
            "ticker": symbol,
        }

    def get_price_history(self, symbol: str, period: str = "10y") -> pd.DataFrame:
        """Return historical price data as a DataFrame."""
        try:
            normalized = self.normalize_ticker(symbol)
            return self._get_cached_or_fetch(
                f"history:{normalized}:{period}",
                lambda: self._get_price_history_payload(normalized, period),
            )
        except Exception as exc:
            logger.exception("Failed to fetch price history for %s", symbol)
            return pd.DataFrame([self._safe_error_payload(symbol, exc)])

    def _get_price_history_payload(self, symbol: str, period: str) -> pd.DataFrame:
        history = self._request_with_retries(symbol, "history", period=period, auto_adjust=True)
        if history.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "ticker"])

        history = history.copy()
        history.reset_index(inplace=True)
        history.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }, inplace=True)
        history["ticker"] = symbol
        return history

    def get_income_statement(self, symbol: str) -> pd.DataFrame:
        """Return the income statement as a DataFrame."""
        try:
            normalized = self.normalize_ticker(symbol)
            return self._get_cached_or_fetch(
                f"income:{normalized}",
                lambda: self._get_statement_payload(normalized, "income_stmt"),
            )
        except Exception as exc:
            logger.exception("Failed to fetch income statement for %s", symbol)
            return pd.DataFrame([self._safe_error_payload(symbol, exc)])

    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """Return the balance sheet as a DataFrame."""
        try:
            normalized = self.normalize_ticker(symbol)
            return self._get_cached_or_fetch(
                f"balance:{normalized}",
                lambda: self._get_statement_payload(normalized, "balance_sheet"),
            )
        except Exception as exc:
            logger.exception("Failed to fetch balance sheet for %s", symbol)
            return pd.DataFrame([self._safe_error_payload(symbol, exc)])

    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        """Return the cash flow statement as a DataFrame."""
        try:
            normalized = self.normalize_ticker(symbol)
            return self._get_cached_or_fetch(
                f"cashflow:{normalized}",
                lambda: self._get_statement_payload(normalized, "cashflow"),
            )
        except Exception as exc:
            logger.exception("Failed to fetch cash flow for %s", symbol)
            return pd.DataFrame([self._safe_error_payload(symbol, exc)])

    def _get_statement_payload(self, symbol: str, attribute: str) -> pd.DataFrame:
        statement = self._request_with_retries(symbol, attribute)
        if statement is None:
            return pd.DataFrame(columns=["metric", "ticker"])
        return statement.T.reset_index().rename(columns={"index": "metric"})


__all__ = ["FinancialService"]
