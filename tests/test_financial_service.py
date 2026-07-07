import unittest
from unittest.mock import patch

import pandas as pd

from athena.services.financial_service import FinancialService


class FakeTicker:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.info = {
            "longName": "Tata Motors",
            "sector": "Automobiles",
            "industry": "Auto Manufacturers",
            "marketCap": 1000000000,
            "sharesOutstanding": 1000000,
            "longBusinessSummary": "Electric vehicle manufacturer",
            "website": "https://tatamotors.com",
            "country": "India",
            "fullTimeEmployees": 70000,
        }
        self.fast_info = type(
            "FastInfo",
            (),
            {
                "last_price": 1000,
                "year_high": 1100,
                "year_low": 900,
                "day_volume": 2000000,
                "average_volume": 1500000,
                "market_cap": 1000000000,
                "currency": "INR",
                "exchange": "NSE",
            },
        )()
        self.history_data = pd.DataFrame(
            {
                "Close": [1000, 1005],
                "Open": [995, 1000],
                "High": [1010, 1015],
                "Low": [990, 1000],
                "Volume": [1000, 2000],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        self.income_stmt = pd.DataFrame({"2023": [10, 5]}, index=["Revenue", "NetIncome"])
        self.balance_sheet = pd.DataFrame({"2023": [20, 10]}, index=["Assets", "Liabilities"])
        self.cashflow = pd.DataFrame({"2023": [8, 3]}, index=["OperatingCashFlow", "CapEx"])

    def history(self, period: str = "10y", auto_adjust: bool = True) -> pd.DataFrame:
        return self.history_data


class FinancialServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FinancialService()

    def test_normalize_ticker_appends_ns(self) -> None:
        self.assertEqual(self.service.normalize_ticker("tcs"), "TCS.NS")

    def test_company_profile_returns_expected_fields(self) -> None:
        with patch.object(self.service, "_session") as mock_session:
            mock_session.Ticker.return_value = FakeTicker("TATAMOTORS")
            profile = self.service.get_company_profile("tatamotors")
            self.assertEqual(profile["ticker"], "TATAMOTORS.NS")
            self.assertEqual(profile["company_name"], "Tata Motors")
            self.assertEqual(profile["sector"], "Automobiles")

    def test_get_fast_info_returns_expected_fields(self) -> None:
        with patch.object(self.service, "_session") as mock_session:
            mock_session.Ticker.return_value = FakeTicker("TATAMOTORS")
            fast_info = self.service.get_fast_info("tatamotors")
            self.assertEqual(fast_info["current_price"], 1000)
            self.assertEqual(fast_info["exchange"], "NSE")

    def test_price_history_returns_dataframe(self) -> None:
        with patch.object(self.service, "_session") as mock_session:
            mock_session.Ticker.return_value = FakeTicker("TATAMOTORS")
            history = self.service.get_price_history("tatamotors")
            self.assertTrue(isinstance(history, pd.DataFrame))
            self.assertIn("close", history.columns)

    def test_financial_statement_methods_return_dataframes(self) -> None:
        with patch.object(self.service, "_session") as mock_session:
            mock_session.Ticker.return_value = FakeTicker("TATAMOTORS")
            income = self.service.get_income_statement("tatamotors")
            balance = self.service.get_balance_sheet("tatamotors")
            cashflow = self.service.get_cash_flow("tatamotors")
            self.assertTrue(isinstance(income, pd.DataFrame))
            self.assertTrue(isinstance(balance, pd.DataFrame))
            self.assertTrue(isinstance(cashflow, pd.DataFrame))

    def test_invalid_ticker_returns_error_payload(self) -> None:
        with patch.object(self.service, "_session") as mock_session:
            mock_session.Ticker.side_effect = ValueError("bad ticker")
            profile = self.service.get_company_profile("")
            self.assertIn("error", profile)


if __name__ == "__main__":
    unittest.main()
