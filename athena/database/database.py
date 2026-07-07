from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from athena.database.models import (
    AnnualReport,
    BalanceSheet,
    Base,
    CashFlow,
    Company,
    Concall,
    DCF,
    IncomeStatement,
    News,
    PriceHistory,
    Ratios,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Production-ready SQLite persistence layer for the ATHENA backend."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        raw_url = database_url or os.environ.get("ATHENA_DB_URL", "sqlite:///athena.db")
        self.database_url = raw_url if raw_url.startswith("sqlite") else f"sqlite:///{raw_url}"
        self.engine = create_engine(self.database_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.create_database()

    def close(self) -> None:
        self.engine.dispose()

    def create_database(self) -> None:
        Base.metadata.create_all(self.engine)
        self._run_migrations()
        logger.info("Database initialized at %s", self.database_url)

    def _run_migrations(self) -> None:
        with self._session() as session:
            session.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, version INTEGER NOT NULL)"))
            session.execute(text("INSERT OR IGNORE INTO schema_migrations (name, version) VALUES ('athena', 1)"))
            session.commit()

    def save_company(
        self,
        *,
        ticker: str,
        company_name: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        market_cap: Optional[float] = None,
        currency: Optional[str] = None,
        website: Optional[str] = None,
        country: Optional[str] = None,
        business_summary: Optional[str] = None,
    ) -> Company:
        with self._session() as session:
            company = session.query(Company).filter_by(ticker=ticker.upper()).one_or_none()
            if company is None:
                company = Company(ticker=ticker.upper())
                session.add(company)

            company.company_name = company_name
            company.sector = sector
            company.industry = industry
            company.market_cap = market_cap
            company.currency = currency
            company.website = website
            company.country = country
            company.business_summary = business_summary
            session.commit()
            session.refresh(company)
            return company

    def load_company(self, ticker: str) -> Optional[Company]:
        with self._session() as session:
            return session.query(Company).filter_by(ticker=ticker.upper()).one_or_none()

    def save_price_history(self, *, ticker: str, history: list[dict[str, Any]]) -> list[PriceHistory]:
        with self._session() as session:
            session.query(PriceHistory).filter_by(ticker=ticker.upper()).delete()
            rows: list[PriceHistory] = []
            for item in history:
                row = PriceHistory(
                    ticker=ticker.upper(),
                    date=str(item.get("date", "")),
                    open=self._coerce_float(item.get("open")),
                    high=self._coerce_float(item.get("high")),
                    low=self._coerce_float(item.get("low")),
                    close=self._coerce_float(item.get("close")),
                    volume=self._coerce_float(item.get("volume")),
                )
                session.add(row)
                rows.append(row)
            session.commit()
            return rows

    def load_price_history(self, ticker: str) -> list[PriceHistory]:
        with self._session() as session:
            return session.query(PriceHistory).filter_by(ticker=ticker.upper()).order_by(PriceHistory.date.asc()).all()

    def save_financials(self, *, ticker: str, statement_type: str, period: Optional[str], data: dict[str, Any]) -> Any:
        with self._session() as session:
            model_type = self._statement_model(statement_type)
            record = model_type(ticker=ticker.upper(), period=period, data=json.dumps(data))
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def load_financials(self, ticker: str) -> list[dict[str, Any]]:
        with self._session() as session:
            records: list[dict[str, Any]] = []
            for model in (IncomeStatement, BalanceSheet, CashFlow):
                rows = session.query(model).filter_by(ticker=ticker.upper()).order_by(model.id.asc()).all()
                for row in rows:
                    records.append({"table": model.__tablename__, "period": row.period, "data": json.loads(row.data or "{}")})
            return records

    def save_ratios(self, *, ticker: str, ratios: dict[str, Any]) -> list[Ratios]:
        with self._session() as session:
            session.query(Ratios).filter_by(ticker=ticker.upper()).delete()
            rows: list[Ratios] = []
            for metric, value in ratios.items():
                row = Ratios(ticker=ticker.upper(), metric=str(metric), value=self._coerce_float(value))
                session.add(row)
                rows.append(row)
            session.commit()
            return rows

    def load_ratios(self, ticker: str) -> list[Ratios]:
        with self._session() as session:
            return session.query(Ratios).filter_by(ticker=ticker.upper()).order_by(Ratios.metric.asc()).all()

    def save_dcf(self, *, ticker: str, terminal_value: Optional[float], enterprise_value: Optional[float], equity_value: Optional[float], assumptions: Optional[dict[str, Any]] = None) -> DCF:
        with self._session() as session:
            record = DCF(
                ticker=ticker.upper(),
                terminal_value=self._coerce_float(terminal_value),
                enterprise_value=self._coerce_float(enterprise_value),
                equity_value=self._coerce_float(equity_value),
                assumptions=json.dumps(assumptions) if assumptions is not None else None,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def save_annual_report(self, *, ticker: str, title: Optional[str], content: Optional[str]) -> AnnualReport:
        with self._session() as session:
            record = AnnualReport(ticker=ticker.upper(), title=title, content=content)
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def save_concall(self, *, ticker: str, title: Optional[str], transcript: Optional[str]) -> Concall:
        with self._session() as session:
            record = Concall(ticker=ticker.upper(), title=title, transcript=transcript)
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def save_news(self, *, ticker: str, title: Optional[str], content: Optional[str], published_at: Optional[str] = None) -> News:
        with self._session() as session:
            record = News(ticker=ticker.upper(), title=title, content=content, published_at=published_at)
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def _session(self) -> Session:
        return self.session_factory()

    def _statement_model(self, statement_type: str) -> type[Any]:
        mapping: dict[str, type[Any]] = {
            "income": IncomeStatement,
            "balance": BalanceSheet,
            "cashflow": CashFlow,
        }
        return mapping.get(statement_type.lower(), IncomeStatement)

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None


def create_database(database_url: Optional[str] = None) -> DatabaseManager:
    """Create a configured database manager and initialize the schema."""
    return DatabaseManager(database_url)


__all__ = ["DatabaseManager", "create_database"]
