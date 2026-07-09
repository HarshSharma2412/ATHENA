import unittest

from athena.engines.management_quality_engine import (
    ManagementQualityEngine,
    ManagementQualityInput,
)
from athena.models.management_models import (
    CommitmentStatus,
    CommitmentType,
    DocumentType,
    ManagementDocument,
    ManagementGrade,
    ManagementQualityResult,
)


def _docs() -> list[ManagementDocument]:
    return [
        ManagementDocument(
            document_type=DocumentType.EARNINGS_CALL,
            fiscal_period="FY24",
            text=(
                "We expect revenue to grow 20% in FY25. "
                "We are targeting an operating margin of 22% in FY25. "
                "We plan capex of 500 crore in FY25. "
                "We aim to reduce net debt to 300 crore in FY25."
            ),
        ),
        ManagementDocument(
            document_type=DocumentType.ANNUAL_REPORT,
            fiscal_period="FY24",
            text="Management aims to expand capacity by 15% in FY25 through a new plant.",
        ),
        ManagementDocument(
            document_type=DocumentType.INVESTOR_PRESENTATION,
            fiscal_period="FY23",
            text="We guide for revenue growth of 18% in FY24.",
        ),
    ]


class ManagementQualityEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ManagementQualityEngine()

    def test_returns_result_type(self) -> None:
        result = self.engine.evaluate(ManagementQualityInput(documents=_docs()))
        self.assertIsInstance(result, ManagementQualityResult)

    def test_detects_commitments_by_type(self) -> None:
        result = self.engine.evaluate(ManagementQualityInput(documents=_docs()))
        types = {outcome.commitment.commitment_type for outcome in result.outcomes}
        self.assertIn(CommitmentType.REVENUE_GUIDANCE, types)
        self.assertIn(CommitmentType.MARGIN_GUIDANCE, types)
        self.assertIn(CommitmentType.CAPEX, types)
        self.assertIn(CommitmentType.DEBT_REDUCTION, types)
        self.assertIn(CommitmentType.EXPANSION, types)

    def test_parses_promised_values(self) -> None:
        result = self.engine.evaluate(ManagementQualityInput(documents=_docs()))
        revenue = next(
            outcome
            for outcome in result.outcomes
            if outcome.commitment.commitment_type is CommitmentType.REVENUE_GUIDANCE
            and outcome.commitment.target_period == "FY25"
        )
        self.assertAlmostEqual(revenue.commitment.promised_value, 0.20, places=6)
        capex = next(
            outcome
            for outcome in result.outcomes
            if outcome.commitment.commitment_type is CommitmentType.CAPEX
        )
        self.assertAlmostEqual(capex.commitment.promised_value, 500.0, places=6)

    def test_delivered_when_actual_meets_promise(self) -> None:
        actuals = {"FY25": {"revenue_growth": 0.21, "operating_margin": 0.23}}
        result = self.engine.evaluate(
            ManagementQualityInput(documents=_docs(), actuals=actuals)
        )
        revenue = next(
            outcome
            for outcome in result.outcomes
            if outcome.commitment.commitment_type is CommitmentType.REVENUE_GUIDANCE
            and outcome.commitment.target_period == "FY25"
        )
        self.assertEqual(revenue.status, CommitmentStatus.DELIVERED)

    def test_missed_when_actual_below_promise(self) -> None:
        actuals = {"FY25": {"revenue_growth": 0.10}}
        result = self.engine.evaluate(
            ManagementQualityInput(documents=_docs(), actuals=actuals)
        )
        revenue = next(
            outcome
            for outcome in result.outcomes
            if outcome.commitment.commitment_type is CommitmentType.REVENUE_GUIDANCE
            and outcome.commitment.target_period == "FY25"
        )
        self.assertEqual(revenue.status, CommitmentStatus.MISSED)
        self.assertTrue(result.red_flags)

    def test_good_delivery_scores_higher_than_poor(self) -> None:
        good = self.engine.evaluate(
            ManagementQualityInput(
                documents=_docs(),
                actuals={
                    "FY25": {
                        "revenue_growth": 0.21,
                        "operating_margin": 0.23,
                        "capex": 505.0,
                        "net_debt": 280.0,
                        "capacity": 0.16,
                    },
                    "FY24": {"revenue_growth": 0.19},
                },
            )
        )
        poor = self.engine.evaluate(
            ManagementQualityInput(
                documents=_docs(),
                actuals={
                    "FY25": {
                        "revenue_growth": 0.08,
                        "operating_margin": 0.12,
                        "capex": 200.0,
                        "net_debt": 600.0,
                        "capacity": 0.03,
                    },
                    "FY24": {"revenue_growth": 0.09},
                },
            )
        )
        self.assertGreater(good.overall_score, poor.overall_score)
        self.assertGreater(good.guidance_accuracy, poor.guidance_accuracy)

    def test_grade_maps_to_score(self) -> None:
        self.assertEqual(self.engine._grade(92.0), ManagementGrade.A_PLUS)
        self.assertEqual(self.engine._grade(82.0), ManagementGrade.A)
        self.assertEqual(self.engine._grade(70.0), ManagementGrade.B)
        self.assertEqual(self.engine._grade(55.0), ManagementGrade.C)
        self.assertEqual(self.engine._grade(40.0), ManagementGrade.D)

    def test_no_documents_is_neutral_with_notes(self) -> None:
        result = self.engine.evaluate(ManagementQualityInput(documents=[]))
        self.assertTrue(result.notes)
        self.assertEqual(result.delivered, 0)
        self.assertEqual(result.missed, 0)

    def test_scores_bounded(self) -> None:
        result = self.engine.evaluate(
            ManagementQualityInput(
                documents=_docs(),
                actuals={"FY25": {"revenue_growth": 0.21, "operating_margin": 0.23}},
            )
        )
        for score in (
            result.overall_score,
            result.management_trust_score,
            result.execution_score,
            result.guidance_accuracy,
            result.transparency,
            result.capital_allocation,
            result.communication,
        ):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

    def test_ai_summary_present(self) -> None:
        result = self.engine.evaluate(
            ManagementQualityInput(
                documents=_docs(),
                actuals={"FY25": {"revenue_growth": 0.10}},
            )
        )
        self.assertIn("management score", result.ai_summary.lower())

    def test_type_validation(self) -> None:
        with self.assertRaises(TypeError):
            self.engine.evaluate(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
