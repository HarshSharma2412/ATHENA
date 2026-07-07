from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IncomeStatement:
    """Standardized income statement payload for ATHENA."""

    year: Optional[str]
    revenue: Optional[float]
    cost_of_revenue: Optional[float]
    gross_profit: Optional[float]
    operating_income: Optional[float]
    ebit: Optional[float]
    ebitda: Optional[float]
    net_income: Optional[float]
    eps: Optional[float]


@dataclass
class BalanceSheet:
    """Standardized balance sheet payload for ATHENA."""

    year: Optional[str]
    cash: Optional[float]
    total_assets: Optional[float]
    total_liabilities: Optional[float]
    equity: Optional[float]
    debt: Optional[float]
    working_capital: Optional[float]


@dataclass
class CashFlow:
    """Standardized cash flow payload for ATHENA."""

    year: Optional[str]
    operating_cash_flow: Optional[float]
    capital_expenditure: Optional[float]
    free_cash_flow: Optional[float]
    investing_cash_flow: Optional[float]
    financing_cash_flow: Optional[float]


class FinancialParser:
    """Parse raw yfinance financial statement frames into ATHENA dataclasses."""

    def __init__(self) -> None:
        self._income_aliases = {
            "revenue": ["revenue", "totalrevenue", "sales", "salesrevenue", "salesrevenuenet"],
            "cost_of_revenue": ["costofrevenue", "costofsales", "cogs"],
            "gross_profit": ["grossprofit", "grossmargin", "grossincome"],
            "operating_income": ["operatingincome", "operatingprofit"],
            "ebit": ["ebit", "earningsbeforeinterestandtaxes", "operatingincomebeforedepreciation"],
            "ebitda": ["ebitda", "earningsbeforeinteresttaxesdepreciationamortization"],
            "net_income": ["netincome", "netearnings", "profitaftertax", "pat"],
            "eps": ["eps", "earningspershare", "earningpershare"],
        }
        self._balance_aliases = {
            "cash": ["cash", "cashandcashequivalents", "cashandshortterminvestments"],
            "total_assets": ["totalassets", "assets"],
            "total_liabilities": ["totalliabilities", "liabilities"],
            "equity": ["stockholdersequity", "shareholdersequity", "equity", "bookvalue"],
            "debt": ["totaldebt", "longtermdebt", "shorttermdebt", "debt"],
            "working_capital": ["workingcapital", "networkingcapital"],
        }
        self._cash_flow_aliases = {
            "operating_cash_flow": ["operatingcashflow", "cashfromoperations", "cashflowfromoperations"],
            "capital_expenditure": ["capitalexpenditure", "capex", "capitalexpenditures", "ppne"],
            "free_cash_flow": ["freecashflow", "fcf"],
            "investing_cash_flow": ["investingcashflow", "cashfrominvestingactivities"],
            "financing_cash_flow": ["financingcashflow", "cashfromfinancingactivities"],
        }

    def parse_income_statement(self, df: pd.DataFrame) -> IncomeStatement:
        """Parse a raw income statement DataFrame into an IncomeStatement dataclass."""
        frame = self._validate_frame(df, "income statement")
        values = self._extract_values(frame, self._income_aliases)
        return IncomeStatement(
            year=self._extract_period(frame),
            revenue=values.get("revenue"),
            cost_of_revenue=values.get("cost_of_revenue"),
            gross_profit=values.get("gross_profit"),
            operating_income=values.get("operating_income"),
            ebit=values.get("ebit"),
            ebitda=values.get("ebitda"),
            net_income=values.get("net_income"),
            eps=values.get("eps"),
        )

    def parse_balance_sheet(self, df: pd.DataFrame) -> BalanceSheet:
        """Parse a raw balance sheet DataFrame into a BalanceSheet dataclass."""
        frame = self._validate_frame(df, "balance sheet")
        values = self._extract_values(frame, self._balance_aliases)
        return BalanceSheet(
            year=self._extract_period(frame),
            cash=values.get("cash"),
            total_assets=values.get("total_assets"),
            total_liabilities=values.get("total_liabilities"),
            equity=values.get("equity"),
            debt=values.get("debt"),
            working_capital=values.get("working_capital"),
        )

    def parse_cash_flow(self, df: pd.DataFrame) -> CashFlow:
        """Parse a raw cash flow DataFrame into a CashFlow dataclass."""
        frame = self._validate_frame(df, "cash flow")
        values = self._extract_values(frame, self._cash_flow_aliases)
        return CashFlow(
            year=self._extract_period(frame),
            operating_cash_flow=values.get("operating_cash_flow"),
            capital_expenditure=values.get("capital_expenditure"),
            free_cash_flow=values.get("free_cash_flow"),
            investing_cash_flow=values.get("investing_cash_flow"),
            financing_cash_flow=values.get("financing_cash_flow"),
        )

    def _validate_frame(self, df: Any, label: str) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{label} data must be a pandas DataFrame")
        if df.empty:
            raise ValueError(f"{label} data cannot be empty")
        return df

    def _extract_values(self, frame: pd.DataFrame, aliases: dict[str, list[str]]) -> dict[str, Optional[float]]:
        values: dict[str, Optional[float]] = {}
        for field_name, field_aliases in aliases.items():
            try:
                values[field_name] = self._find_metric_value(frame, field_aliases)
            except Exception as exc:  # pragma: no cover - defensive branch
                logger.warning("Failed to parse %s from financial frame: %s", field_name, exc)
                values[field_name] = None
        return values

    def _find_metric_value(self, frame: pd.DataFrame, aliases: list[str]) -> Optional[float]:
        normalized_aliases = {self._normalize_label(alias) for alias in aliases}

        for label in frame.index:
            if self._normalize_label(str(label)) in normalized_aliases:
                return self._coerce_value(self._select_latest_value(frame.loc[label]))

        for label in frame.columns:
            if self._normalize_label(str(label)) in normalized_aliases:
                return self._coerce_value(self._select_latest_value(frame[label]))

        return None

    def _select_latest_value(self, series_or_scalar: Any) -> Any:
        if isinstance(series_or_scalar, pd.Series):
            values = [self._coerce_value(value) for value in series_or_scalar.tolist()]
            for value in reversed(values):
                if value is not None:
                    return value
            return None

        if isinstance(series_or_scalar, pd.DataFrame):
            if series_or_scalar.shape[1] == 1:
                return self._select_latest_value(series_or_scalar.iloc[:, 0])
            return None

        return series_or_scalar

    def _extract_period(self, frame: pd.DataFrame) -> Optional[str]:
        candidates: list[Any] = []
        for label in list(frame.columns) + list(frame.index):
            if self._is_period_label(label):
                candidates.append(label)
        if not candidates:
            return None

        numeric_candidates = [int(label) for label in candidates if isinstance(label, (int, float))]
        if numeric_candidates:
            return max(numeric_candidates)

        selected = candidates[-1]
        return str(selected)

    def _is_period_label(self, label: Any) -> bool:
        if label is None:
            return False
        if isinstance(label, (int, float)):
            return 1900 <= int(label) <= 2100
        if isinstance(label, str):
            return bool(re.fullmatch(r"\d{4}", label.strip()))
        return False

    def _coerce_value(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        if isinstance(value, pd.Timestamp):
            return None
        if pd.isna(value):
            return None
        return None

    def _normalize_label(self, label: str) -> str:
        if not isinstance(label, str):
            return str(label).strip().lower()
        return re.sub(r"[^a-z0-9]+", "", label.strip().lower())


__all__ = ["BalanceSheet", "CashFlow", "FinancialParser", "IncomeStatement"]
