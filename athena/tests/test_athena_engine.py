import unittest
from typing import Any

import pandas as pd

from athena.engines.athena_engine import ATHENAEngine, AthenaAnalysis, ValidationReport
from athena.engines.ratio_engine import FinancialData


def _years(n: int) -> list[str]:
    return [f"{2018 + i}" for i in range(n)]


def _financial(growth: float = 0.12) -> FinancialData:
    years = _years(6)
    revenue = [100.0 * ((1.0 + growth) ** i) for i in range(6)]
    income = pd.DataFrame(
        {
            "TotalRevenue": revenue,
            "EBIT": [value * 0.20 for value in revenue],
            "OperatingIncome": [value * 0.20 for value in revenue],
            "NetIncome": [value * 0.14 for value in revenue],
            "PretaxIncome": [value * 0.19 for value in revenue],
            "TaxProvision": [value * 0.0475 for value in revenue],
        },
        index=years,
    ).T
    balance = pd.DataFrame(
        {
            "CurrentAssets": [value * 0.6 for value in revenue],
            "CurrentLiabilities": [value * 0.25 for value in revenue],
            "TotalDebt": [20.0] * 6,
            "CashAndCashEquivalents": [40.0] * 6,
            "StockholdersEquity": [value * 0.9 for value in revenue],
        },
        index=years,
    ).T
    cash = pd.DataFrame(
        {
            "OperatingCashFlow": [value * 0.18 for value in revenue],
            "FreeCashFlow": [value * 0.13 for value in revenue],
            "CapitalExpenditure": [-value * 0.04 for value in revenue],
            "DepreciationAndAmortization": [value * 0.03 for value in revenue],
        },
        index=years,
    ).T
    return FinancialData(income_statement=income, balance_sheet=balance, cash_flow=cash)


class FakeService:
    """A FinancialService stand-in returning canned engine-shaped statements."""

    def __init__(self, financial: FinancialData, *, empty: bool = False) -> None:
        self._financial = financial
        self._empty = empty
        self._cache: dict[str, Any] = {}
        self.income_calls = 0

    def _frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame() if self._empty else frame

    def get_income_statement(self, symbol: str) -> pd.DataFrame:
        self.income_calls += 1
        return self._frame(self._financial.income_statement)

    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        return self._frame(self._financial.balance_sheet)

    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        return self._frame(self._financial.cash_flow)

    def get_company_profile(self, symbol: str) -> dict[str, Any]:
        return {
            "company_name": "Test Corp",
            "sector": "FMCG",
            "market_cap": 5000.0,
            "shares_outstanding": 10.0,
        }

    def get_fast_info(self, symbol: str) -> dict[str, Any]:
        return {"current_price": 120.0, "market_cap": 5000.0}


class _BoomRatioEngine:
    def calculate(self, data: FinancialData) -> Any:
        raise RuntimeError("ratio boom")


class ATHENAEngineTests(unittest.TestCase):
    def _engine(self, **kwargs: Any) -> ATHENAEngine:
        service = kwargs.pop("service", FakeService(_financial()))
        return ATHENAEngine(service=service, **kwargs)  # type: ignore[arg-type]

    def test_full_pipeline_populates_every_stage(self) -> None:
        result = self._engine().analyze("TEST")
        self.assertIsInstance(result, AthenaAnalysis)
        self.assertEqual(result.company, "Test Corp")
        self.assertEqual(result.sector, "FMCG")
        self.assertEqual(result.current_price, 120.0)
        self.assertIsNotNone(result.ratios)
        self.assertIsNotNone(result.business_quality)
        self.assertIsNotNone(result.dcf)
        self.assertIsNotNone(result.reverse_dcf)
        self.assertIsNotNone(result.risk)
        self.assertIsNotNone(result.investment_thesis)
        self.assertFalse(result.errors)
        self.assertFalse(result.is_partial)

    def test_single_engine_failure_is_isolated(self) -> None:
        result = self._engine(ratio_engine=_BoomRatioEngine()).analyze("TEST")
        self.assertIsNone(result.ratios)
        self.assertTrue(result.is_partial)
        self.assertTrue(any("RatioEngine" in error for error in result.errors))
        # The pipeline still produces a thesis despite the missing ratios.
        self.assertIsNotNone(result.investment_thesis)

    def test_no_statements_returns_partial_early(self) -> None:
        result = ATHENAEngine(service=FakeService(_financial(), empty=True)).analyze("TEST")  # type: ignore[arg-type]
        self.assertTrue(result.errors)
        self.assertIsNone(result.ratios)
        self.assertIsNone(result.investment_thesis)

    def test_validate_good_ticker(self) -> None:
        report = self._engine().validate("TEST")
        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(report.is_valid)
        self.assertFalse(report.issues)

    def test_validate_bad_ticker(self) -> None:
        report = ATHENAEngine(service=FakeService(_financial(), empty=True)).validate("TEST")  # type: ignore[arg-type]
        self.assertFalse(report.is_valid)
        self.assertTrue(report.issues)

    def test_validate_empty_ticker(self) -> None:
        report = self._engine().validate("   ")
        self.assertFalse(report.is_valid)

    def test_refresh_reuses_last_ticker_and_clears_cache(self) -> None:
        service = FakeService(_financial())
        engine = ATHENAEngine(service=service)  # type: ignore[arg-type]
        engine.analyze("TEST")
        service._cache["stale"] = "value"
        result = engine.refresh()
        self.assertEqual(result.ticker, "TEST")
        self.assertNotIn("stale", service._cache)
        self.assertGreaterEqual(service.income_calls, 2)

    def test_refresh_without_ticker_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._engine().refresh()

    def test_analyze_type_validation(self) -> None:
        with self.assertRaises(TypeError):
            self._engine().analyze(123)  # type: ignore[arg-type]

    def test_analyze_empty_ticker_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._engine().analyze("  ")


if __name__ == "__main__":
    unittest.main()
