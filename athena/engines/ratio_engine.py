from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FinancialData:
    """Container for financial statement input used by the ratio engine."""

    income_statement: Optional[pd.DataFrame] = None
    balance_sheet: Optional[pd.DataFrame] = None
    cash_flow: Optional[pd.DataFrame] = None
    fast_info: Optional[dict[str, Any]] = None


@dataclass
class RatioResult:
    """Container for calculated profitability, liquidity, leverage, efficiency, growth, valuation, and health metrics."""

    roe: Optional[float] = None
    roce: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    ebitda_margin: Optional[float] = None
    net_margin: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    debt_equity: Optional[float] = None
    debt_assets: Optional[float] = None
    interest_coverage: Optional[float] = None
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivable_turnover: Optional[float] = None
    cash_conversion_cycle: Optional[float] = None
    revenue_cagr: Optional[float] = None
    pat_cagr: Optional[float] = None
    eps_cagr: Optional[float] = None
    book_value_cagr: Optional[float] = None
    fcf_cagr: Optional[float] = None
    book_value_per_share: Optional[float] = None
    owner_earnings: Optional[float] = None
    enterprise_value: Optional[float] = None
    financial_health_score: Optional[float] = None


class RatioEngine:
    """Calculate a portfolio of financial ratios from financial statements and fast info."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, int, int], RatioResult] = {}

    def calculate(self, data: FinancialData) -> RatioResult:
        """Calculate the requested ratios from the provided financial data."""
        if not isinstance(data, FinancialData):
            raise TypeError("data must be an instance of FinancialData")

        cache_key = (
            id(data.income_statement),
            id(data.balance_sheet),
            id(data.cash_flow),
            id(data.fast_info),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        income_statement = data.income_statement
        balance_sheet = data.balance_sheet
        cash_flow = data.cash_flow
        fast_info = data.fast_info or {}

        metrics: dict[str, Optional[float]] = {}
        for metric_name in (
            "Revenue",
            "GrossProfit",
            "OperatingIncome",
            "NetIncome",
            "EBIT",
            "EBITDA",
            "InterestExpense",
            "TotalAssets",
            "StockholdersEquity",
            "TotalLiabilities",
            "CurrentAssets",
            "CurrentLiabilities",
            "Inventory",
            "Receivables",
            "TotalDebt",
            "CashAndCashEquivalents",
            "OperatingCashFlow",
            "FreeCashFlow",
            "CapitalExpenditure",
        ):
            if metric_name in {"Revenue", "GrossProfit", "OperatingIncome", "NetIncome", "EBIT", "EBITDA", "InterestExpense"}:
                metrics[metric_name] = self._get_metric(income_statement, metric_name)
            elif metric_name in {"TotalAssets", "StockholdersEquity", "TotalLiabilities", "CurrentAssets", "CurrentLiabilities", "Inventory", "Receivables", "TotalDebt", "CashAndCashEquivalents"}:
                metrics[metric_name] = self._get_metric(balance_sheet, metric_name)
            else:
                metrics[metric_name] = self._get_metric(cash_flow, metric_name)

        profitability = self._calculate_profitability(
            revenue=metrics["Revenue"],
            gross_profit=metrics["GrossProfit"],
            operating_income=metrics["OperatingIncome"],
            net_income=metrics["NetIncome"],
            ebit=metrics["EBIT"],
            ebitda=metrics["EBITDA"],
            equity=metrics["StockholdersEquity"],
            debt=metrics["TotalDebt"],
            assets=metrics["TotalAssets"],
        )
        liquidity = self._calculate_liquidity(
            current_assets=metrics["CurrentAssets"],
            current_liabilities=metrics["CurrentLiabilities"],
            inventory=metrics["Inventory"],
            cash=metrics["CashAndCashEquivalents"],
        )
        leverage = self._calculate_leverage(
            debt=metrics["TotalDebt"],
            equity=metrics["StockholdersEquity"],
            assets=metrics["TotalAssets"],
            ebit=metrics["EBIT"],
            interest_expense=metrics["InterestExpense"],
        )
        efficiency = self._calculate_efficiency(
            revenue=metrics["Revenue"],
            assets=metrics["TotalAssets"],
            inventory=metrics["Inventory"],
            receivables=metrics["Receivables"],
        )
        growth = self._calculate_growth(
            income_statement=income_statement,
            cash_flow=cash_flow,
            fast_info=fast_info,
        )
        valuation = self._calculate_valuation(
            equity=metrics["StockholdersEquity"],
            operating_cash_flow=metrics["OperatingCashFlow"],
            capex=metrics["CapitalExpenditure"],
            free_cash_flow=metrics["FreeCashFlow"],
            market_cap=fast_info.get("market_cap") if isinstance(fast_info, dict) else None,
            shares_outstanding=fast_info.get("shares_outstanding") if isinstance(fast_info, dict) else None,
            debt=metrics["TotalDebt"],
            cash=metrics["CashAndCashEquivalents"],
        )
        health = self._calculate_financial_health(
            roe=profitability.get("roe"),
            current_ratio=liquidity.get("current_ratio"),
            debt_equity=leverage.get("debt_equity"),
            interest_coverage=leverage.get("interest_coverage"),
            fcf_margin=profitability.get("fcf_margin"),
        )

        result = RatioResult(
            roe=profitability.get("roe"),
            roce=profitability.get("roce"),
            roa=profitability.get("roa"),
            gross_margin=profitability.get("gross_margin"),
            operating_margin=profitability.get("operating_margin"),
            ebitda_margin=profitability.get("ebitda_margin"),
            net_margin=profitability.get("net_margin"),
            current_ratio=liquidity.get("current_ratio"),
            quick_ratio=liquidity.get("quick_ratio"),
            cash_ratio=liquidity.get("cash_ratio"),
            debt_equity=leverage.get("debt_equity"),
            debt_assets=leverage.get("debt_assets"),
            interest_coverage=leverage.get("interest_coverage"),
            asset_turnover=efficiency.get("asset_turnover"),
            inventory_turnover=efficiency.get("inventory_turnover"),
            receivable_turnover=efficiency.get("receivable_turnover"),
            cash_conversion_cycle=efficiency.get("cash_conversion_cycle"),
            revenue_cagr=growth.get("revenue_cagr"),
            pat_cagr=growth.get("pat_cagr"),
            eps_cagr=growth.get("eps_cagr"),
            book_value_cagr=growth.get("book_value_cagr"),
            fcf_cagr=growth.get("fcf_cagr"),
            book_value_per_share=valuation.get("book_value_per_share"),
            owner_earnings=valuation.get("owner_earnings"),
            enterprise_value=valuation.get("enterprise_value"),
            financial_health_score=health,
        )
        self._cache[cache_key] = result
        return result

    def _get_metric(self, frame: Optional[pd.DataFrame], metric: str) -> Optional[float]:
        if frame is None or frame.empty:
            return None

        normalized_metric = self._normalize_label(metric)
        for label in frame.index:
            if self._normalize_label(str(label)) == normalized_metric:
                return self._get_latest_value(frame.loc[label])

        for label in frame.columns:
            if self._normalize_label(str(label)) == normalized_metric:
                return self._get_latest_value(frame[label])

        return None

    def _get_series(self, frame: Optional[pd.DataFrame], metric: str) -> list[Optional[float]]:
        if frame is None or frame.empty:
            return []

        normalized_metric = self._normalize_label(metric)
        for label in frame.index:
            if self._normalize_label(str(label)) == normalized_metric:
                values = frame.loc[label]
                if isinstance(values, pd.Series):
                    return [self._to_float(value) for value in values.tolist()]
                return [self._get_latest_value(frame.loc[label])]

        for label in frame.columns:
            if self._normalize_label(str(label)) == normalized_metric:
                values = frame[label]
                if isinstance(values, pd.Series):
                    return [self._to_float(value) for value in values.tolist()]
                return [self._to_float(values)]

        return []

    def _safe_divide(self, numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        if numerator is None or denominator in (None, 0):
            return None
        return self._to_float(self._safe_decimal(numerator) / self._safe_decimal(denominator))

    def _safe_decimal(self, value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float, np.integer, np.floating)):
            return Decimal(str(value))
        if isinstance(value, str):
            try:
                return Decimal(value.strip())
            except InvalidOperation:
                return None
        return None

    def _calculate_profitability(
        self,
        *,
        revenue: Optional[float],
        gross_profit: Optional[float],
        operating_income: Optional[float],
        net_income: Optional[float],
        ebit: Optional[float],
        ebitda: Optional[float],
        equity: Optional[float],
        debt: Optional[float],
        assets: Optional[float],
    ) -> dict[str, Optional[float]]:
        return {
            "roe": self._safe_divide(net_income, equity),
            "roce": self._safe_divide(ebit, equity + debt if equity is not None and debt is not None else None),
            "roa": self._safe_divide(net_income, assets),
            "gross_margin": self._safe_divide(gross_profit, revenue),
            "operating_margin": self._safe_divide(operating_income, revenue),
            "ebitda_margin": self._safe_divide(ebitda, revenue),
            "net_margin": self._safe_divide(net_income, revenue),
            "fcf_margin": self._safe_divide(None, revenue),
        }

    def _calculate_liquidity(
        self,
        *,
        current_assets: Optional[float],
        current_liabilities: Optional[float],
        inventory: Optional[float],
        cash: Optional[float],
    ) -> dict[str, Optional[float]]:
        quick_numerator = current_assets - inventory if current_assets is not None and inventory is not None else None
        return {
            "current_ratio": self._safe_divide(current_assets, current_liabilities),
            "quick_ratio": self._safe_divide(quick_numerator, current_liabilities),
            "cash_ratio": self._safe_divide(cash, current_liabilities),
        }

    def _calculate_leverage(
        self,
        *,
        debt: Optional[float],
        equity: Optional[float],
        assets: Optional[float],
        ebit: Optional[float],
        interest_expense: Optional[float],
    ) -> dict[str, Optional[float]]:
        return {
            "debt_equity": self._safe_divide(debt, equity),
            "debt_assets": self._safe_divide(debt, assets),
            "interest_coverage": self._safe_divide(ebit, interest_expense),
        }

    def _calculate_efficiency(
        self,
        *,
        revenue: Optional[float],
        assets: Optional[float],
        inventory: Optional[float],
        receivables: Optional[float],
    ) -> dict[str, Optional[float]]:
        return {
            "asset_turnover": self._safe_divide(revenue, assets),
            "inventory_turnover": self._safe_divide(revenue, inventory),
            "receivable_turnover": self._safe_divide(revenue, receivables),
            "cash_conversion_cycle": self._safe_divide((self._safe_divide(revenue, inventory) or 0.0) - (self._safe_divide(revenue, receivables) or 0.0), 1.0),
        }

    def _calculate_growth(
        self,
        *,
        income_statement: Optional[pd.DataFrame],
        cash_flow: Optional[pd.DataFrame],
        fast_info: Optional[dict[str, Any]],
    ) -> dict[str, Optional[float]]:
        return {
            "revenue_cagr": self._growth_rate(self._get_series(income_statement, "Revenue")),
            "pat_cagr": self._growth_rate(self._get_series(income_statement, "NetIncome")),
            "eps_cagr": self._growth_rate(self._series_from_fast_info(fast_info, "eps")),
            "book_value_cagr": self._growth_rate(self._series_from_fast_info(fast_info, "book_value")),
            "fcf_cagr": self._growth_rate(self._get_series(cash_flow, "FreeCashFlow")),
        }

    def _calculate_valuation(
        self,
        *,
        equity: Optional[float],
        operating_cash_flow: Optional[float],
        capex: Optional[float],
        free_cash_flow: Optional[float],
        market_cap: Optional[float],
        shares_outstanding: Optional[float],
        debt: Optional[float],
        cash: Optional[float],
    ) -> dict[str, Optional[float]]:
        return {
            "book_value_per_share": self._safe_divide(equity, shares_outstanding),
            "owner_earnings": None if operating_cash_flow is None or capex is None else operating_cash_flow - capex,
            "enterprise_value": None if market_cap is None else market_cap + (debt or 0.0) - (cash or 0.0),
        }

    def _calculate_financial_health(
        self,
        *,
        roe: Optional[float],
        current_ratio: Optional[float],
        debt_equity: Optional[float],
        interest_coverage: Optional[float],
        fcf_margin: Optional[float],
    ) -> Optional[float]:
        components = [
            (roe, 25.0, self._normalize_health_score(roe, scale=0.20, upper_bound=100.0)),
            (debt_equity, 25.0, self._normalize_health_score(debt_equity, scale=None, upper_bound=100.0, invert=True)),
            (interest_coverage, 20.0, self._normalize_health_score(interest_coverage, scale=10.0, upper_bound=100.0)),
            (current_ratio, 15.0, self._normalize_health_score(current_ratio, scale=2.0, upper_bound=100.0)),
            (fcf_margin, 15.0, self._normalize_health_score(fcf_margin, scale=1.0, upper_bound=100.0)),
        ]
        total_weight = sum(weight for _, weight, _ in components if _ is not None)
        if total_weight == 0:
            return None
        weighted_total = sum(score * weight for _, weight, score in components if score is not None)
        return round(weighted_total / total_weight, 2)

    def _normalize_health_score(self, value: Optional[float], *, scale: Optional[float], upper_bound: float, invert: bool = False) -> Optional[float]:
        if value is None:
            return None
        if scale is None:
            normalized = 100.0 / (1.0 + abs(value)) if value >= 0 else 0.0
        else:
            normalized = value / scale
        if invert:
            normalized = 100.0 / (1.0 + abs(value))
        return max(0.0, min(upper_bound, normalized * 100.0))

    def _growth_rate(self, values: list[Optional[float]]) -> Optional[float]:
        if len(values) < 2:
            return None
        valid_values = [value for value in values if value not in (None, 0)]
        if len(valid_values) < 2:
            return None
        start_value = valid_values[0]
        end_value = valid_values[-1]
        if start_value in (None, 0) or end_value in (None, 0):
            return None
        return float((self._safe_decimal(end_value) / self._safe_decimal(start_value)) ** Decimal(1 / (len(values) - 1)) - Decimal("1"))

    def _series_from_fast_info(self, fast_info: Optional[dict[str, Any]], key: str) -> list[Optional[float]]:
        if not fast_info:
            return []
        value = fast_info.get(key)
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return [self._to_float(item) for item in value]
        return [self._to_float(value)]

    def _get_latest_value(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, pd.Series):
            for item in reversed(value.tolist()):
                converted = self._to_float(item)
                if converted is not None:
                    return converted
            return None
        if isinstance(value, np.ndarray):
            return self._get_latest_value(pd.Series(value))
        return self._to_float(value)

    def _normalize_label(self, label: str) -> str:
        if not isinstance(label, str):
            return str(label).strip().lower()
        return "".join(ch for ch in label.lower() if ch.isalnum())

    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        if isinstance(value, (Decimal,)):
            return float(value)
        return None


__all__ = ["FinancialData", "RatioEngine", "RatioResult"]
