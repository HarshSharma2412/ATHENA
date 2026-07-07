import unittest

import pandas as pd

from athena.services.financial_parser import BalanceSheet, CashFlow, FinancialParser, IncomeStatement


class FinancialParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = FinancialParser()
        self.income_df = pd.DataFrame(
            [
                [900.0, 1000.0],
                [540.0, 600.0],
                [360.0, 400.0],
                [130.0, 150.0],
                [100.0, 120.0],
                [160.0, 180.0],
                [70.0, 80.0],
                [1.8, 2.0],
            ],
            index=["Revenue", "CostOfRevenue", "GrossProfit", "OperatingIncome", "EBIT", "EBITDA", "NetIncome", "EPS"],
            columns=[2023, 2024],
        )
        self.balance_df = pd.DataFrame(
            [
                [90.0, 100.0],
                [950.0, 1000.0],
                [550.0, 600.0],
                [400.0, 400.0],
                [45.0, 50.0],
                [90.0, 100.0],
            ],
            index=["CashAndCashEquivalents", "TotalAssets", "TotalLiabilities", "StockholdersEquity", "TotalDebt", "WorkingCapital"],
            columns=[2023, 2024],
        )
        self.cash_flow_df = pd.DataFrame(
            [
                [110.0, 120.0],
                [35.0, 40.0],
                [75.0, 80.0],
                [18.0, 20.0],
                [8.0, 10.0],
            ],
            index=["OperatingCashFlow", "CapitalExpenditure", "FreeCashFlow", "InvestingCashFlow", "FinancingCashFlow"],
            columns=[2023, 2024],
        )

    def test_parse_income_statement_returns_dataclass(self) -> None:
        result = self.parser.parse_income_statement(self.income_df)
        self.assertIsInstance(result, IncomeStatement)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.revenue, 1000.0)
        self.assertEqual(result.net_income, 80.0)

    def test_parse_balance_sheet_returns_dataclass(self) -> None:
        result = self.parser.parse_balance_sheet(self.balance_df)
        self.assertIsInstance(result, BalanceSheet)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.equity, 400.0)
        self.assertEqual(result.working_capital, 100.0)

    def test_parse_cash_flow_returns_dataclass(self) -> None:
        result = self.parser.parse_cash_flow(self.cash_flow_df)
        self.assertIsInstance(result, CashFlow)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.operating_cash_flow, 120.0)
        self.assertEqual(result.free_cash_flow, 80.0)

    def test_parser_handles_missing_values(self) -> None:
        frame = pd.DataFrame([[1.0, None]], index=["Revenue"], columns=[2023, 2024])
        result = self.parser.parse_income_statement(frame)
        self.assertIsNone(result.cost_of_revenue)
        self.assertEqual(result.revenue, 1.0)

    def test_parser_rejects_invalid_input(self) -> None:
        with self.assertRaises(TypeError):
            self.parser.parse_income_statement([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
