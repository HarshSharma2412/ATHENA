import math
import unittest

from athena.utils.formatting import (
    NOT_AVAILABLE,
    format_indian_grouping,
    format_market_cap_crores,
    format_percent,
    format_price,
    format_ratio,
    format_ratio_percent,
    format_score,
    to_float,
)


class FormattingTests(unittest.TestCase):
    def test_to_float_rejects_non_finite(self) -> None:
        self.assertIsNone(to_float(float("nan")))
        self.assertIsNone(to_float(float("inf")))
        self.assertIsNone(to_float("abc"))
        self.assertEqual(to_float("12.5"), 12.5)

    def test_indian_grouping(self) -> None:
        self.assertEqual(format_indian_grouping(1567842, decimals=0), "15,67,842")
        self.assertEqual(format_indian_grouping(4235.5, decimals=2), "4,235.50")
        self.assertEqual(format_indian_grouping(-1567842, decimals=0), "-15,67,842")

    def test_price_formatting(self) -> None:
        self.assertEqual(format_price(4235.5), "\u20b94,235.50")
        self.assertEqual(format_price(4585, decimals=0), "\u20b94,585")

    def test_market_cap_in_crores(self) -> None:
        self.assertEqual(format_market_cap_crores(15_67_842_00_00_000), "\u20b915,67,842 Cr")

    def test_ratio_and_percent(self) -> None:
        self.assertEqual(format_ratio(34.6215), "34.62")
        self.assertEqual(format_percent(1.23), "1.23%")
        self.assertEqual(format_ratio_percent(0.0123), "1.23%")
        self.assertEqual(format_score(75.5), "75.5 / 100")

    def test_missing_values_render_not_available(self) -> None:
        self.assertEqual(format_price(None), NOT_AVAILABLE)
        self.assertEqual(format_ratio(math.nan), NOT_AVAILABLE)
        self.assertEqual(format_percent(None), NOT_AVAILABLE)
        self.assertEqual(format_market_cap_crores("x"), NOT_AVAILABLE)
        self.assertEqual(format_score(float("inf")), NOT_AVAILABLE)


if __name__ == "__main__":
    unittest.main()
