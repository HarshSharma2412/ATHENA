import unittest

import pandas as pd

from athena.engines.dcf_engine import DCFEngine, DCFInput
from athena.engines.ratio_engine import FinancialData, RatioResult
from athena.engines.reverse_dcf_engine import ReverseDCFEngine, ReverseDCFInput
from athena.models.reverse_dcf_models import (
    ExpectationLevel,
    ReverseDCFResult,
    ValuationRisk,
)


def _years(n: int) -> list[str]:
    return [f"{2019 + i}" for i in range(n)]


def _statements(growth: float = 0.11) -> FinancialData:
    years = _years(6)
    revenue = [100.0 * ((1.0 + growth) ** i) for i in range(6)]
    ebit = [value * 0.20 for value in revenue]

    income = pd.DataFrame(
        {
            "TotalRevenue": revenue,
            "EBIT": ebit,
            "OperatingIncome": ebit,
            "NetIncome": [value * 0.14 for value in revenue],
            "PretaxIncome": [value * 0.19 for value in revenue],
            "TaxProvision": [value * 0.0475 for value in revenue],
        },
        index=years,
    ).T
    balance = pd.DataFrame(
        {
            "CurrentAssets": [value * 0.50 for value in revenue],
            "CurrentLiabilities": [value * 0.30 for value in revenue],
            "TotalDebt": [20.0] * 6,
            "CashAndCashEquivalents": [15.0] * 6,
            "StockholdersEquity": [value * 0.8 for value in revenue],
        },
        index=years,
    ).T
    cash = pd.DataFrame(
        {
            "CapitalExpenditure": [-value * 0.04 for value in revenue],
            "DepreciationAndAmortization": [value * 0.03 for value in revenue],
            "OperatingCashFlow": [value * 0.15 for value in revenue],
            "FreeCashFlow": [value * 0.12 for value in revenue],
        },
        index=years,
    ).T
    return FinancialData(income_statement=income, balance_sheet=balance, cash_flow=cash)


def _ratios() -> RatioResult:
    return RatioResult(
        roe=0.20,
        roce=0.22,
        roa=0.12,
        operating_margin=0.20,
        net_margin=0.14,
        debt_equity=0.3,
        asset_turnover=0.9,
        revenue_cagr=0.11,
    )


class ReverseDCFEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ReverseDCFEngine()
        self.financial = _statements()
        self.ratios = _ratios()
        # Produce a real DCFResult to supply the baseline operating profile.
        self.dcf = DCFEngine().value(
            DCFInput(
                income_statement=self.financial.income_statement,
                balance_sheet=self.financial.balance_sheet,
                cash_flow=self.financial.cash_flow,
                ratios=self.ratios,
                current_price=None,
                shares_outstanding=10.0,
                market_cap=None,
            )
        )

    def _input(self, price: float) -> ReverseDCFInput:
        return ReverseDCFInput(
            current_price=price,
            shares_outstanding=10.0,
            cash=15.0,
            debt=20.0,
            financial_data=self.financial,
            ratios=self.ratios,
            dcf=self.dcf,
        )

    def test_returns_result_type(self) -> None:
        result = self.engine.analyze(self._input(price=100.0))
        self.assertIsInstance(result, ReverseDCFResult)

    def test_price_at_intrinsic_implies_baseline_growth(self) -> None:
        # Pricing the stock at its own DCF intrinsic value should imply a growth
        # rate close to the DCF's own revenue-growth assumption.
        intrinsic = self.dcf.intrinsic_value
        assert intrinsic is not None
        result = self.engine.analyze(self._input(price=intrinsic))
        self.assertIsNotNone(result.required_growth)
        assert result.required_growth is not None
        self.assertAlmostEqual(result.required_growth, self.dcf.revenue_growth, places=2)

    def test_higher_price_implies_higher_growth(self) -> None:
        intrinsic = self.dcf.intrinsic_value
        assert intrinsic is not None
        low = self.engine.analyze(self._input(price=intrinsic * 0.8))
        high = self.engine.analyze(self._input(price=intrinsic * 1.5))
        assert low.required_growth is not None and high.required_growth is not None
        self.assertGreater(high.required_growth, low.required_growth)
        self.assertGreaterEqual(high.expectation_score, low.expectation_score)

    def test_expensive_price_is_aggressive_and_risky(self) -> None:
        intrinsic = self.dcf.intrinsic_value
        assert intrinsic is not None
        result = self.engine.analyze(self._input(price=intrinsic * 3.0))
        self.assertIn(
            result.expectation_level,
            {ExpectationLevel.AGGRESSIVE, ExpectationLevel.EXTREME},
        )
        self.assertIn(result.valuation_risk, {ValuationRisk.MEDIUM, ValuationRisk.HIGH})
        self.assertGreater(result.difference or 0.0, 0.0)
        self.assertTrue(result.ai_summary)

    def test_cheap_price_is_low_expectation(self) -> None:
        intrinsic = self.dcf.intrinsic_value
        assert intrinsic is not None
        result = self.engine.analyze(self._input(price=intrinsic * 0.4))
        self.assertIn(
            result.expectation_level,
            {ExpectationLevel.LOW, ExpectationLevel.MODERATE},
        )
        self.assertEqual(result.valuation_risk, ValuationRisk.LOW)

    def test_gap_analysis_present(self) -> None:
        result = self.engine.analyze(self._input(price=150.0))
        self.assertIsNotNone(result.gap.growth_gap)
        self.assertEqual(result.historical_growth, self.ratios.revenue_cagr)
        self.assertIsNotNone(result.implied.required_fcf_margin)

    def test_missing_price_returns_notes(self) -> None:
        result = self.engine.analyze(
            ReverseDCFInput(
                current_price=None,
                shares_outstanding=10.0,
                cash=15.0,
                debt=20.0,
                financial_data=self.financial,
                ratios=self.ratios,
                dcf=self.dcf,
            )
        )
        self.assertIsNone(result.required_growth)
        self.assertTrue(result.notes)

    def test_works_without_dcf(self) -> None:
        result = self.engine.analyze(
            ReverseDCFInput(
                current_price=150.0,
                shares_outstanding=10.0,
                cash=15.0,
                debt=20.0,
                financial_data=self.financial,
                ratios=self.ratios,
                dcf=None,
            )
        )
        self.assertIsInstance(result, ReverseDCFResult)
        self.assertTrue(any("DCFResult" in note for note in result.notes))

    def test_type_validation(self) -> None:
        with self.assertRaises(TypeError):
            self.engine.analyze(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
