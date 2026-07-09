import unittest

import pandas as pd

from athena.utils.statements import (
    build_yearly_trend,
    latest_metric,
    matching_column,
    to_engine_frame,
)


def _service_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": pd.to_datetime(["2024-03-31", "2023-03-31"]),
            "Total Revenue": [1000.0, 900.0],
            "Net Income": [150.0, 120.0],
            "ticker": ["TCS", "TCS"],
        }
    )


class StatementsTests(unittest.TestCase):
    def test_matching_column_is_fuzzy(self) -> None:
        frame = _service_frame()
        self.assertEqual(matching_column(frame, ("TotalRevenue",)), "Total Revenue")
        self.assertIsNone(matching_column(frame, ("Nonexistent",)))

    def test_latest_metric_uses_newest_period(self) -> None:
        frame = _service_frame()
        self.assertEqual(latest_metric(frame, ("NetIncome", "Net Income")), 150.0)

    def test_build_yearly_trend_is_sorted(self) -> None:
        trend = build_yearly_trend(_service_frame(), ("TotalRevenue", "Revenue"))
        self.assertEqual(list(trend["year"]), [2023, 2024])
        self.assertEqual(list(trend["value"]), [900.0, 1000.0])

    def test_to_engine_frame_canonicalises_and_orders(self) -> None:
        engine_frame = to_engine_frame(_service_frame())
        self.assertIn("Revenue", engine_frame.index)
        self.assertIn("NetIncome", engine_frame.index)
        # Columns ordered oldest -> newest, so the last column is the latest year.
        self.assertEqual(engine_frame.loc["NetIncome"].iloc[-1], 150.0)

    def test_empty_frame_is_safe(self) -> None:
        empty = pd.DataFrame()
        self.assertIsNone(latest_metric(empty, ("Revenue",)))
        self.assertTrue(build_yearly_trend(empty, ("Revenue",)).empty)


if __name__ == "__main__":
    unittest.main()
