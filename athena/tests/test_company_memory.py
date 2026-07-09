import sqlite3
import unittest
from datetime import datetime, timezone

from athena.memory.company_memory import CompanyMemory, snapshot_from_analysis
from athena.models.company_snapshot import CompanySnapshot, SnapshotComparison


def _snapshot(
    *,
    date: datetime,
    price: float,
    intrinsic: float,
    business: float,
    risk: float,
    management: float,
    recommendation: str = "Buy",
) -> CompanySnapshot:
    return CompanySnapshot(
        ticker="test",
        date=date,
        current_price=price,
        intrinsic_value=intrinsic,
        business_score=business,
        risk_score=risk,
        management_score=management,
        recommendation=recommendation,
        dcf={"intrinsic_value": intrinsic},
        reverse_dcf={"required_growth": 0.2},
        investment_thesis={"investment_rating": recommendation},
    )


class CompanyMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self.memory = CompanyMemory(connection=self._conn)

    def tearDown(self) -> None:
        self.memory.close()

    def test_save_assigns_id_and_normalizes_ticker(self) -> None:
        saved = self.memory.save_snapshot(
            _snapshot(
                date=datetime(2024, 3, 31, tzinfo=timezone.utc),
                price=100.0,
                intrinsic=120.0,
                business=70.0,
                risk=30.0,
                management=65.0,
            )
        )
        self.assertIsNotNone(saved.snapshot_id)
        self.assertEqual(saved.ticker, "TEST")

    def test_load_latest_returns_most_recent(self) -> None:
        self.memory.save_snapshot(
            _snapshot(
                date=datetime(2024, 3, 31, tzinfo=timezone.utc),
                price=100.0,
                intrinsic=120.0,
                business=70.0,
                risk=30.0,
                management=65.0,
            )
        )
        self.memory.save_snapshot(
            _snapshot(
                date=datetime(2024, 6, 30, tzinfo=timezone.utc),
                price=110.0,
                intrinsic=130.0,
                business=75.0,
                risk=28.0,
                management=68.0,
            )
        )
        latest = self.memory.load_latest("TEST")
        assert latest is not None
        self.assertEqual(latest.current_price, 110.0)
        self.assertEqual(latest.business_score, 75.0)
        # JSON blobs round-trip.
        self.assertEqual(latest.dcf["intrinsic_value"], 130.0)

    def test_load_latest_missing_ticker(self) -> None:
        self.assertIsNone(self.memory.load_latest("NOPE"))

    def test_load_history_ordered_oldest_first(self) -> None:
        for month, price in ((3, 100.0), (6, 110.0), (9, 120.0)):
            self.memory.save_snapshot(
                _snapshot(
                    date=datetime(2024, month, 28, tzinfo=timezone.utc),
                    price=price,
                    intrinsic=price + 20,
                    business=70.0,
                    risk=30.0,
                    management=65.0,
                )
            )
        history = self.memory.load_history("TEST")
        self.assertEqual([snap.current_price for snap in history], [100.0, 110.0, 120.0])
        limited = self.memory.load_history("TEST", limit=2)
        self.assertEqual(len(limited), 2)

    def test_compare_current_vs_previous_quarter(self) -> None:
        self.memory.save_snapshot(
            _snapshot(
                date=datetime(2024, 3, 31, tzinfo=timezone.utc),
                price=100.0,
                intrinsic=120.0,
                business=70.0,
                risk=35.0,
                management=60.0,
            )
        )
        self.memory.save_snapshot(
            _snapshot(
                date=datetime(2024, 6, 30, tzinfo=timezone.utc),
                price=110.0,
                intrinsic=140.0,
                business=78.0,
                risk=30.0,
                management=66.0,
            )
        )
        comparison = self.memory.compare("TEST")
        assert isinstance(comparison, SnapshotComparison)
        assert comparison.previous is not None
        self.assertEqual(comparison.current.current_price, 110.0)
        self.assertEqual(comparison.previous.current_price, 100.0)
        self.assertAlmostEqual(comparison.deltas["business_score"] or 0.0, 8.0)
        self.assertAlmostEqual(comparison.deltas["risk_score"] or 0.0, -5.0)
        self.assertIn("improved", comparison.summary)

    def test_compare_single_snapshot_has_no_previous(self) -> None:
        self.memory.save_snapshot(
            _snapshot(
                date=datetime(2024, 3, 31, tzinfo=timezone.utc),
                price=100.0,
                intrinsic=120.0,
                business=70.0,
                risk=30.0,
                management=65.0,
            )
        )
        comparison = self.memory.compare("TEST")
        assert comparison is not None
        self.assertIsNone(comparison.previous)
        self.assertIn("first recorded", comparison.summary)

    def test_compare_missing_ticker_is_none(self) -> None:
        self.assertIsNone(self.memory.compare("NOPE"))

    def test_trend_direction_respects_metric_polarity(self) -> None:
        self.memory.save_snapshot(
            _snapshot(
                date=datetime(2023, 12, 31, tzinfo=timezone.utc),
                price=90.0,
                intrinsic=100.0,
                business=60.0,
                risk=40.0,
                management=55.0,
            )
        )
        self.memory.save_snapshot(
            _snapshot(
                date=datetime(2024, 12, 31, tzinfo=timezone.utc),
                price=120.0,
                intrinsic=150.0,
                business=75.0,
                risk=25.0,
                management=70.0,
            )
        )
        trend = self.memory.trend("TEST")
        self.assertEqual(trend.business_score.direction, "Improving")
        self.assertEqual(trend.business_score.change, 15.0)
        # Risk fell, and lower risk is better -> improving.
        self.assertEqual(trend.risk_score.direction, "Improving")
        self.assertEqual(trend.intrinsic_value.direction, "Improving")
        self.assertEqual(len(trend.current_price.points), 2)

    def test_trend_empty_history(self) -> None:
        trend = self.memory.trend("NOPE")
        self.assertEqual(trend.business_score.direction, "Unknown")
        self.assertEqual(trend.business_score.points, [])

    def test_save_type_validation(self) -> None:
        with self.assertRaises(TypeError):
            self.memory.save_snapshot(object())  # type: ignore[arg-type]

    def test_persists_across_instances(self) -> None:
        self.memory.save_snapshot(
            _snapshot(
                date=datetime(2024, 3, 31, tzinfo=timezone.utc),
                price=100.0,
                intrinsic=120.0,
                business=70.0,
                risk=30.0,
                management=65.0,
            )
        )
        # A second CompanyMemory over the same connection sees the data.
        other = CompanyMemory(connection=self._conn)
        self.assertIsNotNone(other.load_latest("TEST"))


class SnapshotFromAnalysisTests(unittest.TestCase):
    def test_from_analysis_maps_headline_fields(self) -> None:
        from athena.engines.athena_engine import AthenaAnalysis

        analysis = AthenaAnalysis(
            ticker="TEST",
            company="Test Corp",
            current_price=100.0,
        )
        snapshot = snapshot_from_analysis(analysis, management_score=72.0)
        self.assertEqual(snapshot.ticker, "TEST")
        self.assertEqual(snapshot.current_price, 100.0)
        self.assertEqual(snapshot.management_score, 72.0)
        self.assertIsNone(snapshot.intrinsic_value)
        self.assertEqual(snapshot.dcf, {})


if __name__ == "__main__":
    unittest.main()
