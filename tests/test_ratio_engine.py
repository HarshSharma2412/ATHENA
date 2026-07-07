import unittest

import pandas as pd

from athena.engines.ratio_engine import FinancialData, RatioEngine, RatioResult


class RatioEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RatioEngine()
        self.income_statement = pd.DataFrame(
            {
                "Revenue": [1000.0, 900.0],
                "GrossProfit": [200.0, 180.0],
                "OperatingIncome": [150.0, 140.0],
                "NetIncome": [100.0, 90.0],
                "EBITDA": [200.0, 190.0],
                "EBIT": [120.0, 110.0],
                "InterestExpense": [10.0, 10.0],
            },
            index=["2023", "2022"],
        ).T
        self.balance_sheet = pd.DataFrame(
            {
                "TotalAssets": [1000.0, 900.0],
                "StockholdersEquity": [500.0, 450.0],
                "TotalLiabilities": [400.0, 360.0],
                "CurrentAssets": [100.0, 90.0],
                "CurrentLiabilities": [50.0, 45.0],
                "Inventory": [20.0, 18.0],
                "Receivables": [30.0, 27.0],
                "TotalDebt": [100.0, 90.0],
                "CashAndCashEquivalents": [30.0, 25.0],
            },
            index=["2023", "2022"],
        ).T
        self.cash_flow = pd.DataFrame(
            {
                "OperatingCashFlow": [150.0, 140.0],
                "FreeCashFlow": [120.0, 110.0],
                "CapitalExpenditure": [30.0, 30.0],
            },
            index=["2023", "2022"],
        ).T
        self.fast_info = {
            "eps": [1.0, 0.9],
            "book_value": [2.0, 1.8],
            "dividend": [0.5, 0.45],
            "shares_outstanding": 10.0,
            "current_price": 100.0,
            "market_cap": 1000.0,
        }

    def test_calculate_returns_ratio_result(self) -> None:
        data = FinancialData(
            income_statement=self.income_statement,
            balance_sheet=self.balance_sheet,
            cash_flow=self.cash_flow,
            fast_info=self.fast_info,
        )
        result = self.engine.calculate(data)

        self.assertIsInstance(result, RatioResult)
        self.assertIsNotNone(result.roe)
        self.assertIsNotNone(result.current_ratio)
        self.assertIsNotNone(result.net_margin)
        self.assertIsNotNone(result.revenue_cagr)
        self.assertIsNotNone(result.cash_ratio)
        self.assertIsNotNone(result.debt_assets)
        self.assertIsNotNone(result.cash_conversion_cycle)
        self.assertIsNotNone(result.book_value_per_share)
        self.assertIsNotNone(result.enterprise_value)
        self.assertIsNotNone(result.financial_health_score)

    def test_calculate_handles_empty_inputs(self) -> None:
        result = self.engine.calculate(FinancialData())
        self.assertIsInstance(result, RatioResult)
        self.assertIsNone(result.roe)
        self.assertIsNone(result.current_ratio)
        self.assertIsNone(result.cash_ratio)
        self.assertIsNone(result.enterprise_value)


if __name__ == "__main__":
    unittest.main()
