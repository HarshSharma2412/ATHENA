"""ATHENA's central orchestration engine.

``ATHENAEngine`` is the single entry point the dashboard (or any client) should
call. Given a ticker it runs the full research pipeline -

    FinancialService -> FinancialParser -> RatioEngine -> BusinessQualityEngine
    -> DCFEngine -> ReverseDCFEngine -> RiskEngine -> InvestmentThesisEngine

- and returns a consolidated :class:`AthenaAnalysis`.

Design goals:
* Dependency injection - every collaborator can be supplied for testing; sane
  production defaults are constructed otherwise.
* Graceful degradation - if one stage fails the rest still run and a *partial*
  analysis is returned with the failure recorded (never raises for a single
  engine fault).
* No duplicated business logic - each engine owns its own maths; this engine
  only wires inputs and outputs together.
* No Streamlit / presentation code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, TypeVar

import pandas as pd

from athena.engines.business_quality_engine import (
    BusinessQualityEngine,
    BusinessQualityInput,
)
from athena.engines.dcf_engine import DCFEngine, DCFInput
from athena.engines.investment_thesis_engine import (
    InvestmentThesisEngine,
    InvestmentThesisInput,
)
from athena.engines.ratio_engine import FinancialData, RatioEngine, RatioResult
from athena.engines.reverse_dcf_engine import ReverseDCFEngine, ReverseDCFInput
from athena.engines.risk_engine import RiskEngine, RiskInput
from athena.models.business_quality_models import BusinessQualityResult
from athena.models.investment_thesis_models import InvestmentThesisResult
from athena.models.reverse_dcf_models import ReverseDCFResult
from athena.models.risk_models import RiskResult
from athena.models.valuation_models import DCFResult
from athena.services.financial_parser import FinancialParser
from athena.services.financial_service import FinancialService
from athena.utils import statements as st
from athena.utils.formatting import to_float

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_DEBT_ALIASES = ("TotalDebt", "LongTermDebt", "totaldebt")
_CASH_ALIASES = ("CashAndCashEquivalents", "CashAndCashEquivalentsAndShortTermInvestments", "cash")


@dataclass(frozen=True)
class AthenaAnalysis:
    """Consolidated, investor-facing output of the full ATHENA pipeline."""

    ticker: str
    company: Optional[str] = None
    sector: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    ratios: Optional[RatioResult] = None
    business_quality: Optional[BusinessQualityResult] = None
    dcf: Optional[DCFResult] = None
    reverse_dcf: Optional[ReverseDCFResult] = None
    risk: Optional[RiskResult] = None
    investment_thesis: Optional[InvestmentThesisResult] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_partial(self) -> bool:
        """True when at least one pipeline stage failed."""
        return bool(self.errors)


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of :meth:`ATHENAEngine.validate` for a ticker."""

    ticker: str
    is_valid: bool
    issues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _RawStatements:
    """Engine-shaped statement frames plus market data for one company."""

    ticker: str
    company: Optional[str]
    sector: Optional[str]
    current_price: Optional[float]
    market_cap: Optional[float]
    shares_outstanding: Optional[float]
    cash: Optional[float]
    debt: Optional[float]
    financial_data: FinancialData


class ATHENAEngine:
    """Central orchestrator - the only engine the dashboard needs to call."""

    def __init__(
        self,
        *,
        service: Optional[FinancialService] = None,
        parser: Optional[FinancialParser] = None,
        ratio_engine: Optional[RatioEngine] = None,
        business_quality_engine: Optional[BusinessQualityEngine] = None,
        dcf_engine: Optional[DCFEngine] = None,
        reverse_dcf_engine: Optional[ReverseDCFEngine] = None,
        risk_engine: Optional[RiskEngine] = None,
        thesis_engine: Optional[InvestmentThesisEngine] = None,
    ) -> None:
        self._service = service or FinancialService()
        self._parser = parser or FinancialParser()
        self._ratio_engine = ratio_engine or RatioEngine()
        self._business_quality_engine = business_quality_engine or BusinessQualityEngine()
        self._dcf_engine = dcf_engine or DCFEngine()
        self._reverse_dcf_engine = reverse_dcf_engine or ReverseDCFEngine()
        self._risk_engine = risk_engine or RiskEngine()
        self._thesis_engine = thesis_engine or InvestmentThesisEngine()
        self._last_ticker: Optional[str] = None

    # ------------------------------------------------------------------ public
    def analyze(self, ticker: str) -> AthenaAnalysis:
        """Run the full research pipeline for ``ticker`` and return a result.

        A failure in any single stage is logged and recorded; the pipeline
        continues so the caller always receives a (possibly partial) analysis.
        """
        if not isinstance(ticker, str):
            raise TypeError("ticker must be a string")
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("ticker must not be empty")
        self._last_ticker = symbol

        errors: list[str] = []
        notes: list[str] = []

        raw = self._load_statements(symbol, errors)
        if raw is None:
            return AthenaAnalysis(ticker=symbol, errors=errors, notes=notes)

        ratios = self._safe("RatioEngine", errors, lambda: self._ratio_engine.calculate(raw.financial_data))

        business_quality = self._safe(
            "BusinessQualityEngine",
            errors,
            lambda: self._business_quality_engine.evaluate(
                BusinessQualityInput(
                    financial_data=raw.financial_data,
                    ratios=self._require(ratios, "ratios"),
                    ticker=symbol,
                )
            ),
        )

        dcf = self._safe(
            "DCFEngine",
            errors,
            lambda: self._dcf_engine.value(
                DCFInput(
                    income_statement=raw.financial_data.income_statement,
                    balance_sheet=raw.financial_data.balance_sheet,
                    cash_flow=raw.financial_data.cash_flow,
                    ratios=ratios,
                    current_price=raw.current_price,
                    shares_outstanding=raw.shares_outstanding,
                    market_cap=raw.market_cap,
                    ticker=symbol,
                )
            ),
        )

        reverse_dcf = self._safe(
            "ReverseDCFEngine",
            errors,
            lambda: self._reverse_dcf_engine.analyze(
                ReverseDCFInput(
                    current_price=raw.current_price,
                    shares_outstanding=raw.shares_outstanding,
                    cash=raw.cash,
                    debt=raw.debt,
                    financial_data=raw.financial_data,
                    ratios=self._require(ratios, "ratios"),
                    business_quality=business_quality,
                    dcf=dcf,
                    ticker=symbol,
                )
            ),
        )

        risk = self._safe(
            "RiskEngine",
            errors,
            lambda: self._risk_engine.assess(
                RiskInput(
                    financial_data=raw.financial_data,
                    ratios=ratios,
                    dcf=dcf,
                    business_quality=business_quality,
                    reverse_dcf=reverse_dcf,
                    sector=raw.sector,
                    ticker=symbol,
                )
            ),
        )

        thesis = self._safe(
            "InvestmentThesisEngine",
            errors,
            lambda: self._thesis_engine.build(
                InvestmentThesisInput(
                    financial_data=raw.financial_data,
                    ratios=ratios,
                    dcf=dcf,
                    reverse_dcf=reverse_dcf,
                    business_quality=business_quality,
                    risk=risk,
                    ticker=symbol,
                    current_price=raw.current_price,
                )
            ),
        )

        return AthenaAnalysis(
            ticker=symbol,
            company=raw.company,
            sector=raw.sector,
            current_price=raw.current_price,
            market_cap=raw.market_cap,
            ratios=ratios,
            business_quality=business_quality,
            dcf=dcf,
            reverse_dcf=reverse_dcf,
            risk=risk,
            investment_thesis=thesis,
            errors=errors,
            notes=notes,
        )

    def refresh(self, ticker: Optional[str] = None) -> AthenaAnalysis:
        """Drop cached market data and re-run the pipeline.

        Uses ``ticker`` when supplied, otherwise the most recently analysed one.
        """
        symbol = (ticker or self._last_ticker or "").strip().upper()
        if not symbol:
            raise ValueError("no ticker supplied and nothing analysed yet")
        self._clear_cache()
        return self.analyze(symbol)

    def validate(self, ticker: str) -> ValidationReport:
        """Check that ``ticker`` resolves to usable financial statements."""
        if not isinstance(ticker, str):
            raise TypeError("ticker must be a string")
        symbol = ticker.strip().upper()
        issues: list[str] = []
        if not symbol:
            return ValidationReport(ticker=ticker, is_valid=False, issues=["empty ticker"])

        try:
            income = self._service.get_income_statement(symbol)
            balance = self._service.get_balance_sheet(symbol)
            cash_flow = self._service.get_cash_flow(symbol)
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.exception("Validation fetch failed for %s", symbol)
            return ValidationReport(ticker=symbol, is_valid=False, issues=[str(exc)])

        for label, frame in (("income statement", income), ("balance sheet", balance), ("cash flow", cash_flow)):
            if self._is_unusable(frame):
                issues.append(f"missing or empty {label}")
        return ValidationReport(ticker=symbol, is_valid=not issues, issues=issues)

    # ------------------------------------------------------------- data loading
    def _load_statements(self, symbol: str, errors: list[str]) -> Optional[_RawStatements]:
        try:
            income = self._service.get_income_statement(symbol)
            balance = self._service.get_balance_sheet(symbol)
            cash_flow = self._service.get_cash_flow(symbol)
            profile = self._service.get_company_profile(symbol)
            fast_info = self._service.get_fast_info(symbol)
        except Exception as exc:
            logger.exception("Failed to load statements for %s", symbol)
            errors.append(f"FinancialService: {exc}")
            return None

        if self._is_unusable(income) and self._is_unusable(balance) and self._is_unusable(cash_flow):
            errors.append("FinancialService: no financial statements available")
            return None

        income_frame = st.to_engine_frame(income)
        balance_frame = st.to_engine_frame(balance)
        cash_frame = st.to_engine_frame(cash_flow)

        market_cap = to_float(profile.get("market_cap")) or to_float(fast_info.get("market_cap"))
        shares = to_float(profile.get("shares_outstanding"))
        financial_data = FinancialData(
            income_statement=income_frame,
            balance_sheet=balance_frame,
            cash_flow=cash_frame,
            fast_info={**fast_info, "shares_outstanding": profile.get("shares_outstanding")},
        )
        return _RawStatements(
            ticker=symbol,
            company=profile.get("company_name"),
            sector=profile.get("sector"),
            current_price=to_float(fast_info.get("current_price")),
            market_cap=market_cap,
            shares_outstanding=shares,
            cash=self._latest(balance_frame, _CASH_ALIASES),
            debt=self._latest(balance_frame, _DEBT_ALIASES),
            financial_data=financial_data,
        )

    # -------------------------------------------------------------- error guard
    def _safe(self, stage: str, errors: list[str], func: Callable[[], _T]) -> Optional[_T]:
        try:
            return func()
        except Exception as exc:
            logger.exception("%s failed", stage)
            errors.append(f"{stage}: {exc}")
            return None

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _require(value: Optional[_T], name: str) -> _T:
        if value is None:
            raise ValueError(f"required upstream result '{name}' is missing")
        return value

    @staticmethod
    def _latest(frame: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[float]:
        series = st.metric_series(frame, aliases)
        for value in reversed(series):
            if value:
                return float(value)
        return None

    @staticmethod
    def _is_unusable(frame: object) -> bool:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return True
        return "error" in {str(column).lower() for column in frame.columns}

    def _clear_cache(self) -> None:
        self._service._cache.clear()


__all__ = ["ATHENAEngine", "AthenaAnalysis", "ValidationReport"]
