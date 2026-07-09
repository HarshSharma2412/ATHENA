import unittest

import pandas as pd

from athena.engines.dcf_engine import DCFEngine, DCFInput
from athena.engines.ratio_engine import RatioResult
from athena.models.valuation_models import DCFResult, Recommendation
from athena.engines.forecast_engine import ForecastEngine


def _years(n: int) -> list[str]:
    return [f"{2019 + i}" for i in range(n)]


def _build_statements() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = _years(6)
    revenue = [100.0 * (1.10**i) for i in range(6)]
    ebit = [value * 0.20 for value in revenue]
    pretax = [value * 0.19 for value in revenue]
    tax = [value * 0.0475 for value in revenue]  # ~25% of pretax

    income = pd.DataFrame(
        {
            "TotalRevenue": revenue,
            "EBIT": ebit,
            "OperatingIncome": ebit,
            "PretaxIncome": pretax,
            "TaxProvision": tax,
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
            "FreeCashFlow": [value * 0.12 for value in revenue],
        },
        index=years,
    ).T
    return income, balance, cash


class ForecastEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.income, self.balance, self.cash = _build_statements()
        self.engine = ForecastEngine()

    def test_history_extraction(self) -> None:
        history = self.engine.build_history(self.income, self.balance, self.cash)
        self.assertEqual(history.years, 6)
        self.assertAlmostEqual(history.latest_revenue, 100.0 * (1.10**5))
        self.assertEqual(history.total_debt, 20.0)
        self.assertEqual(history.cash, 15.0)
        # Operating margin should be ~20% every year.
        self.assertAlmostEqual(history.operating_margin[-1], 0.20, places=6)

    def test_revenue_growth_matches_history(self) -> None:
        history = self.engine.build_history(self.income, self.balance, self.cash)
        self.assertAlmostEqual(self.engine.estimate_revenue_growth(history), 0.10, places=6)

    def test_operating_assumptions(self) -> None:
        history = self.engine.build_history(self.income, self.balance, self.cash)
        assumptions = self.engine.estimate_operating_assumptions(history)
        self.assertAlmostEqual(assumptions["operating_margin"], 0.20, places=6)
        self.assertAlmostEqual(assumptions["tax_rate"], 0.25, places=2)
        self.assertAlmostEqual(assumptions["capex_to_revenue"], 0.04, places=6)
        self.assertAlmostEqual(assumptions["working_capital_to_revenue"], 0.20, places=6)


class DCFEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.income, self.balance, self.cash = _build_statements()
        self.engine = DCFEngine()
        self.ratios = RatioResult(
            roe=0.25, roce=0.25, debt_equity=0.2, financial_health_score=80.0
        )
        self.data = DCFInput(
            income_statement=self.income,
            balance_sheet=self.balance,
            cash_flow=self.cash,
            fast_info={"shares_outstanding": 10.0, "market_cap": 25_000e7},
            ratios=self.ratios,
            current_price=20.0,
            shares_outstanding=10.0,
            market_cap=25_000e7,
            ticker="TEST",
        )

    def test_returns_populated_result(self) -> None:
        result = self.engine.value(self.data)
        self.assertIsInstance(result, DCFResult)
        self.assertIsNotNone(result.intrinsic_value)
        self.assertGreater(result.intrinsic_value or 0, 0.0)
        self.assertEqual(len(result.forecast), 10)
        self.assertEqual(result.assumptions.forecast_years, 10)

    def test_discount_rate_bounds_and_adjustments(self) -> None:
        # Large cap (0.10) with ROE and ROCE > 20% => 0.10 - 0.005 - 0.005 = 0.09.
        result = self.engine.value(self.data)
        self.assertAlmostEqual(result.discount_rate, 0.09, places=6)
        self.assertGreaterEqual(result.discount_rate, 0.08)
        self.assertLessEqual(result.discount_rate, 0.15)

    def test_terminal_growth_below_discount_rate(self) -> None:
        result = self.engine.value(self.data)
        self.assertLess(result.terminal_growth, result.discount_rate)
        self.assertLessEqual(result.terminal_growth, 0.05)

    def test_confidence_in_range(self) -> None:
        result = self.engine.value(self.data)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 100.0)

    def test_scenarios_and_sensitivity(self) -> None:
        result = self.engine.value(self.data)
        names = {scenario.name for scenario in result.scenarios}
        self.assertEqual(names, {"Bear", "Base", "Bull"})
        self.assertEqual(result.base_value, result.intrinsic_value)
        # Bull should value at least as high as bear.
        self.assertGreaterEqual(result.bull_value or 0, result.bear_value or 0)
        self.assertTrue(result.sensitivity_table)
        for row in result.sensitivity_table.values():
            self.assertEqual(len(row), 3)

    def test_recommendation_is_enum(self) -> None:
        result = self.engine.value(self.data)
        self.assertIsInstance(result.recommendation, Recommendation)

    def test_high_debt_increases_discount_rate(self) -> None:
        ratios = RatioResult(roe=0.1, roce=0.1, debt_equity=1.5, financial_health_score=50.0)
        data = DCFInput(
            income_statement=self.income,
            balance_sheet=self.balance,
            cash_flow=self.cash,
            fast_info={"shares_outstanding": 10.0},
            ratios=ratios,
            current_price=20.0,
            shares_outstanding=10.0,
            market_cap=25_000e7,
        )
        result = self.engine.value(data)
        # Large cap base 0.10 + high debt 0.01 = 0.11.
        self.assertAlmostEqual(result.discount_rate, 0.11, places=6)

    def test_empty_statements_do_not_crash(self) -> None:
        result = self.engine.value(DCFInput(current_price=100.0, shares_outstanding=10.0))
        self.assertIsInstance(result, DCFResult)
        self.assertIsNone(result.intrinsic_value)
        self.assertEqual(result.recommendation, Recommendation.HOLD)
        self.assertTrue(result.notes)


if __name__ == "__main__":
    unittest.main()
