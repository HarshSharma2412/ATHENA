import math
import unittest

from athena.utils import finance_math as fm


class FinanceMathTests(unittest.TestCase):
    def test_to_float_rejects_non_finite_and_bool(self) -> None:
        self.assertIsNone(fm.to_float(float("nan")))
        self.assertIsNone(fm.to_float(float("inf")))
        self.assertIsNone(fm.to_float(True))
        self.assertEqual(fm.to_float("12.5"), 12.5)

    def test_safe_divide(self) -> None:
        self.assertEqual(fm.safe_divide(10, 4), 2.5)
        self.assertIsNone(fm.safe_divide(10, 0))
        self.assertIsNone(fm.safe_divide(None, 4))

    def test_clamp(self) -> None:
        self.assertEqual(fm.clamp(5, 0, 10), 5)
        self.assertEqual(fm.clamp(-1, 0, 10), 0)
        self.assertEqual(fm.clamp(11, 0, 10), 10)

    def test_weighted_average_skips_none_and_renormalises(self) -> None:
        # (0.10*0.2 + 0.20*0.5) / (0.2 + 0.5) with the None weight dropped.
        self.assertAlmostEqual(
            fm.weighted_average([0.10, None, 0.20], [0.2, 0.3, 0.5]), 0.171428571, places=6
        )
        self.assertIsNone(fm.weighted_average([None, None], [0.5, 0.5]))

    def test_cagr(self) -> None:
        self.assertAlmostEqual(fm.cagr(100, 200, 1), 1.0)
        self.assertAlmostEqual(fm.cagr(100, 121, 2), 0.10)
        self.assertIsNone(fm.cagr(0, 100, 2))
        self.assertIsNone(fm.cagr(100, -10, 2))

    def test_series_cagr_uses_recent_window(self) -> None:
        series = [100, 110, 121, 133.1]  # +10% per year
        self.assertAlmostEqual(fm.series_cagr(series, 3), 0.10, places=6)

    def test_coefficient_of_variation(self) -> None:
        self.assertEqual(fm.coefficient_of_variation([10, 10, 10]), 0.0)
        cv = fm.coefficient_of_variation([1, 5, 10])
        self.assertIsNotNone(cv)
        self.assertGreater(cv, 0.0)

    def test_gordon_terminal_value_requires_spread(self) -> None:
        self.assertAlmostEqual(fm.gordon_terminal_value(100, 0.10, 0.04) or 0, 1733.333, places=2)
        self.assertIsNone(fm.gordon_terminal_value(100, 0.04, 0.05))

    def test_present_value(self) -> None:
        self.assertAlmostEqual(fm.present_value(110, 0.10, 1), 100.0)
        self.assertAlmostEqual(
            fm.present_value_of_series([110, 121], 0.10), 200.0, places=6
        )
        self.assertTrue(math.isclose(fm.discount_factor(0.10, 1), 1 / 1.1))


if __name__ == "__main__":
    unittest.main()
