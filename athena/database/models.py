from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import String, Text, Float, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


@dataclass
class Company(Base):
    __tablename__ = "Company"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    company_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    sector: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    market_cap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    website: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    business_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


@dataclass
class PriceHistory(Base):
    __tablename__ = "PriceHistory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    date: Mapped[str] = mapped_column(String(50), nullable=False)
    open: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


@dataclass
class IncomeStatement(Base):
    __tablename__ = "IncomeStatement"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    period: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


@dataclass
class BalanceSheet(Base):
    __tablename__ = "BalanceSheet"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    period: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


@dataclass
class CashFlow(Base):
    __tablename__ = "CashFlow"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    period: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


@dataclass
class Ratios(Base):
    __tablename__ = "Ratios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    metric: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


@dataclass
class DCF(Base):
    __tablename__ = "DCF"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    terminal_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    enterprise_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    equity_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    assumptions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


@dataclass
class AnnualReport(Base):
    __tablename__ = "AnnualReport"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


@dataclass
class Concall(Base):
    __tablename__ = "Concall"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


@dataclass
class News(Base):
    __tablename__ = "News"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    published_at: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


__all__ = [
    "Any",
    "Base",
    "Company",
    "PriceHistory",
    "IncomeStatement",
    "BalanceSheet",
    "CashFlow",
    "Ratios",
    "DCF",
    "AnnualReport",
    "Concall",
    "News",
]
