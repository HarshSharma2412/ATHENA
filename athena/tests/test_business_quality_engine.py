import unittest

import pandas as pd

from athena.engines.business_quality_engine import (
    BusinessQualityEngine,
    BusinessQualityInput,
)
from athena.engines.ratio_engine import FinancialData, RatioResult
from athena.models.business_quality_models import (
    BusinessQualityResult,
    Grade,
    Rating,
)


def _years(n: int) -> list[str]:
    return [f"{2019 + i}" for i in range(n)]


def _statements(revenue_growth: float = 0.16, margin: float = 0.22) -> FinancialData:
    years = _years(6)
    revenue = [100.0 * ((1.0 + revenue_growth) ** i) for i in range(6)]
    net_income = [value * 0.14 for value in revenue]
    ebit = [value * margin for value in revenue]

    income = pd.DataFrame(
        {
            "TotalRevenue": revenue,
            "EBIT": ebit,
            "OperatingIncome": ebit,
            "NetIncome": net_income,
            "PretaxIncome": [value * (margin - 0.01) for value in revenue],
            "TaxProvision": [value * 0.05 for value in revenue],
        },
        index=years,
    ).T

    balance = pd.DataFrame(
        {
            "CurrentAssets": [value * 0.60 for value in revenue],
            "CurrentLiabilities": [value * 0.25 for value in revenue],
            "TotalDebt": [10.0] * 6,
            "CashAndCashEquivalents": [40.0] * 6,
            "StockholdersEquity": [value * 0.7 for value in revenue],
        },
        index=years,
    ).T

    cash = pd.DataFrame(
        {
            "CapitalExpenditure": [-value * 0.03 for value in revenue],
            "DepreciationAndAmortization": [value * 0.03 for value in revenue],
            "OperatingCashFlow": [value * 0.16 for value in revenue],
            "FreeCashFlow": [value * 0.13 for value in revenue],
        },
        index=years,
    ).T
    return FinancialData(income_statement=income, balance_sheet=balance, cash_flow=cash)


def _high_quality_ratios() -> RatioResult:
    return RatioResult(
        roe=0.25,
        roce=0.28,
        roa=0.15,
        gross_margin=0.45,
        operating_margin=0.22,
        net_margin=0.14,
        current_ratio=2.4,
        quick_ratio=1.8,
        cash_ratio=0.6,
        debt_equity=0.15,
        interest_coverage=15.0,
        revenue_cagr=0.16,
        pat_cagr=0.15,
        eps_cagr=0.14,
        fcf_cagr=0.15,
        owner_earnings=90.0,
        financial_health_score=85.0,
    )


def _weak_quality_ratios() -> RatioResult:
    return RatioResult(
        roe=0.04,
        roce=0.05,
        roa=0.02,
        gross_margin=0.12,
        operating_margin=0.04,
        net_margin=0.02,
        current_ratio=0.8,
        quick_ratio=0.4,
        cash_ratio=0.03,
        debt_equity=2.5,
        interest_coverage=1.1,
        revenue_cagr=-0.05,
        pat_cagr=-0.08,
        eps_cagr=-0.06,
        fcf_cagr=-0.10,
        financial_health_score=25.0,
    )


class BusinessQualityEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = BusinessQualityEngine()

    def test_returns_result_type(self) -> None:
        result = self.engine.evaluate(
            BusinessQualityInput(financial_data=_statements(), ratios=_high_quality_ratios())
        )
        self.assertIsInstance(result, BusinessQualityResult)

    def test_high_quality_scores_high(self) -> None:
        result = self.engine.evaluate(
            BusinessQualityInput(financial_data=_statements(), ratios=_high_quality_ratios())
        )
        self.assertGreaterEqual(result.overall_score, 75.0)
        self.assertIn(result.grade, {Grade.A_PLUS, Grade.A, Grade.B_PLUS, Grade.B})
        self.assertTrue(result.strengths)
        self.assertEqual(result.stars.count("\u2605"), 5)

    def test_weak_quality_scores_low(self) -> None:
        result = self.engine.evaluate(
            BusinessQualityInput(
                financial_data=_statements(revenue_growth=-0.03, margin=0.04),
                ratios=_weak_quality_ratios(),
            )
        )
        self.assertLessEqual(result.overall_score, 50.0)
        self.assertIn(result.grade, {Grade.C, Grade.D})
        self.assertTrue(result.risk_flags)
        self.assertIn("High debt", result.risk_flags)

    def test_scores_are_bounded(self) -> None:
        result = self.engine.evaluate(
            BusinessQualityInput(financial_data=_statements(), ratios=_high_quality_ratios())
        )
        for score in (
            result.overall_score,
            result.growth_score,
            result.profitability_score,
            result.financial_strength_score,
            result.cashflow_score,
            result.capital_allocation_score,
            result.consistency_score,
            result.competitive_strength_score,
        ):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

    def test_seven_dimensions_present(self) -> None:
        result = self.engine.evaluate(
            BusinessQualityInput(financial_data=_statements(), ratios=_high_quality_ratios())
        )
        names = {dimension.name for dimension in result.dimensions}
        self.assertEqual(
            names,
            {
                "Growth",
                "Profitability",
                "Financial Strength",
                "Cash Flow Quality",
                "Capital Allocation",
                "Consistency",
                "Competitive Strength",
            },
        )
        for dimension in result.dimensions:
            self.assertIsInstance(dimension.rating, Rating)

    def test_empty_statements_do_not_crash(self) -> None:
        result = self.engine.evaluate(
            BusinessQualityInput(
                financial_data=FinancialData(),
                ratios=RatioResult(),
            )
        )
        self.assertIsInstance(result, BusinessQualityResult)
        self.assertTrue(result.notes)

    def test_negative_growth_flagged(self) -> None:
        result = self.engine.evaluate(
            BusinessQualityInput(
                financial_data=_statements(revenue_growth=-0.04),
                ratios=_weak_quality_ratios(),
            )
        )
        self.assertIn("Declining revenue", result.risk_flags)

    def test_grade_thresholds(self) -> None:
        self.assertEqual(self.engine._grade(96.0), Grade.A_PLUS)
        self.assertEqual(self.engine._grade(92.0), Grade.A)
        self.assertEqual(self.engine._grade(85.0), Grade.B_PLUS)
        self.assertEqual(self.engine._grade(72.0), Grade.B)
        self.assertEqual(self.engine._grade(65.0), Grade.C)
        self.assertEqual(self.engine._grade(50.0), Grade.D)

    def test_stars_format(self) -> None:
        stars = self.engine._stars(85.0)
        self.assertEqual(len(stars), 5)
        self.assertEqual(stars.count("\u2605"), 5)
        low = self.engine._stars(10.0)
        self.assertEqual(low.count("\u2605"), 1)

    def test_type_validation(self) -> None:
        with self.assertRaises(TypeError):
            self.engine.evaluate(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
