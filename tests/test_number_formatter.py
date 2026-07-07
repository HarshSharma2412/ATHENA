import math
import unittest

from athena.services.financial_service import FinancialService
from athena.utils.number_formatter import (
    NOT_AVAILABLE,
    format_crore,
    format_currency,
    format_market_cap,
    format_percentage,
    format_ratio,
)


class NumberFormatterTests(unittest.TestCase):
    def test_format_currency(self) -> None:
        self.assertEqual(format_currency(321.9), "\u20b9321.90")
        self.assertEqual(format_currency(4585, decimals=0), "\u20b94,585")

    def test_format_crore(self) -> None:
        self.assertEqual(format_crore(15_000_000), "\u20b91.50 Cr")
        self.assertEqual(format_crore(21_820_000_000), "\u20b92,182 Cr")
        self.assertEqual(format_crore(2_670_210_000_000), "\u20b92,67,021 Cr")

    def test_format_market_cap_matches_crore(self) -> None:
        self.assertEqual(format_market_cap(21_820_000_000), "\u20b92,182 Cr")

    def test_format_percentage(self) -> None:
        self.assertEqual(format_percentage(0.3473), "34.73%")
        self.assertEqual(format_percentage(0.0123), "1.23%")

    def test_format_ratio(self) -> None:
        self.assertEqual(format_ratio(4.1823), "4.18")

    def test_missing_values_render_not_available(self) -> None:
        for func in (format_currency, format_crore, format_market_cap, format_percentage, format_ratio):
            self.assertEqual(func(None), NOT_AVAILABLE)
            self.assertEqual(func(math.nan), NOT_AVAILABLE)
            self.assertEqual(func(math.inf), NOT_AVAILABLE)
            self.assertEqual(func("not-a-number"), NOT_AVAILABLE)


class DividendYieldTests(unittest.TestCase):
    def test_percentage_style_yield_is_normalised(self) -> None:
        # yfinance often returns dividendYield as a percentage (e.g. 5.55).
        self.assertAlmostEqual(
            FinancialService._extract_dividend_yield({"dividendYield": 5.55}), 0.0555
        )

    def test_fraction_style_yield_is_kept(self) -> None:
        self.assertAlmostEqual(
            FinancialService._extract_dividend_yield({"dividendYield": 0.0123}), 0.0123
        )

    def test_falls_back_to_trailing_yield(self) -> None:
        self.assertAlmostEqual(
            FinancialService._extract_dividend_yield(
                {"dividendYield": None, "trailingAnnualDividendYield": 0.0503}
            ),
            0.0503,
        )

    def test_missing_yield_returns_none(self) -> None:
        self.assertIsNone(FinancialService._extract_dividend_yield({}))
        self.assertIsNone(FinancialService._extract_dividend_yield({"dividendYield": 0}))


if __name__ == "__main__":
    unittest.main()
