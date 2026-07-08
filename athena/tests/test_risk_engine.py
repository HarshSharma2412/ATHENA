import unittest

import pandas as pd

from athena.engines.ratio_engine import FinancialData, RatioResult
from athena.engines.risk_engine import RiskEngine, RiskInput
from athena.models.reverse_dcf_models import (
    ExpectationLevel,
    GapAnalysis,
    HistoricalReality,
    ImpliedExpectations,
    ReverseDCFResult,
    ValuationRisk,
)
from athena.models.risk_models import CyclicalityLevel, RiskGrade, RiskResult


def _years(n: int) -> list[str]:
    return [f"{2018 + i}" for i in range(n)]


def _healthy_statements() -> FinancialData:
    years = _years(6)
    revenue = [100.0 * (1.12 ** i) for i in range(6)]
    income = pd.DataFrame(
        {
            "TotalRevenue": revenue,
            "EBIT": [value * 0.20 for value in revenue],
            "NetIncome": [value * 0.14 for value in revenue],
        },
        index=years,
    ).T
    balance = pd.DataFrame(
        {
            "CurrentAssets": [value * 0.6 for value in revenue],
            "CurrentLiabilities": [value * 0.25 for value in revenue],
            "TotalDebt": [20.0] * 6,
            "CashAndCashEquivalents": [40.0] * 6,
            "StockholdersEquity": [value * 0.9 for value in revenue],
        },
        index=years,
    ).T
    cash = pd.DataFrame(
        {
            "OperatingCashFlow": [value * 0.18 for value in revenue],
            "FreeCashFlow": [value * 0.13 for value in revenue],
            "CapitalExpenditure": [-value * 0.04 for value in revenue],
            "DepreciationAndAmortization": [value * 0.03 for value in revenue],
        },
        index=years,
    ).T
    return FinancialData(income_statement=income, balance_sheet=balance, cash_flow=cash)


def _risky_statements() -> FinancialData:
    years = _years(6)
    revenue = [100.0 * (0.94 ** i) for i in range(6)]  # declining
    debt = [50.0 * (1.2 ** i) for i in range(6)]  # rising fast
    income = pd.DataFrame(
        {
            "TotalRevenue": revenue,
            "EBIT": [value * 0.06 for value in revenue],
            "NetIncome": [value * 0.02 for value in revenue],
        },
        index=years,
    ).T
    balance = pd.DataFrame(
        {
            "CurrentAssets": [value * 0.3 for value in revenue],
            "CurrentLiabilities": [value * 0.5 for value in revenue],
            "TotalDebt": debt,
            "CashAndCashEquivalents": [5.0] * 6,
            "StockholdersEquity": [value * 0.3 for value in revenue],
        },
        index=years,
    ).T
    cash = pd.DataFrame(
        {
            "OperatingCashFlow": [value * 0.03 for value in revenue],
            "FreeCashFlow": [-value * 0.05 for value in revenue],  # negative FCF
            "CapitalExpenditure": [-value * 0.10 for value in revenue],
            "DepreciationAndAmortization": [value * 0.03 for value in revenue],
        },
        index=years,
    ).T
    return FinancialData(income_statement=income, balance_sheet=balance, cash_flow=cash)


def _healthy_ratios() -> RatioResult:
    return RatioResult(
        debt_equity=0.2,
        interest_coverage=12.0,
        current_ratio=2.4,
        quick_ratio=1.8,
        cash_ratio=1.0,
        operating_margin=0.20,
        net_margin=0.14,
        revenue_cagr=0.12,
    )


def _risky_ratios() -> RatioResult:
    return RatioResult(
        debt_equity=2.6,
        interest_coverage=1.0,
        current_ratio=0.8,
        quick_ratio=0.4,
        cash_ratio=0.1,
        operating_margin=0.06,
        net_margin=0.02,
        revenue_cagr=-0.06,
    )


def _reverse_dcf(risk: ValuationRisk, score: float) -> ReverseDCFResult:
    return ReverseDCFResult(
        required_growth=0.22,
        required_margin=0.20,
        required_roic=0.18,
        expectation_score=score,
        expectation_level=ExpectationLevel.AGGRESSIVE,
        valuation_risk=risk,
        historical_growth=0.11,
        historical_margin=0.18,
        difference=0.11,
        ai_summary="",
        implied=ImpliedExpectations(0.22, 0.20, 0.15, 0.18, 0.04, 0.11),
        historical=HistoricalReality(0.11, 0.18, 0.12, 0.14, 0.16, 0.18),
        gap=GapAnalysis(0.11, 0.02, 0.03),
    )


class RiskEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RiskEngine()

    def test_returns_result_type(self) -> None:
        result = self.engine.assess(RiskInput())
        self.assertIsInstance(result, RiskResult)

    def test_healthy_lower_than_risky(self) -> None:
        healthy = self.engine.assess(
            RiskInput(
                financial_data=_healthy_statements(),
                ratios=_healthy_ratios(),
                sector="FMCG",
            )
        )
        risky = self.engine.assess(
            RiskInput(
                financial_data=_risky_statements(),
                ratios=_risky_ratios(),
                sector="Metals & Mining",
            )
        )
        self.assertLess(healthy.overall_risk_score, risky.overall_risk_score)
        self.assertLess(healthy.financial_risk, risky.financial_risk)
        self.assertLess(healthy.leverage_risk, risky.leverage_risk)
        self.assertLess(healthy.cashflow_risk, risky.cashflow_risk)

    def test_scores_bounded(self) -> None:
        result = self.engine.assess(
            RiskInput(financial_data=_risky_statements(), ratios=_risky_ratios())
        )
        for score in (
            result.overall_risk_score,
            result.financial_risk,
            result.business_risk,
            result.valuation_risk,
            result.growth_risk,
            result.liquidity_risk,
            result.leverage_risk,
            result.cashflow_risk,
            result.cyclicality_risk,
            result.execution_risk,
            result.governance_risk,
        ):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

    def test_risk_flags_for_risky_company(self) -> None:
        result = self.engine.assess(
            RiskInput(financial_data=_risky_statements(), ratios=_risky_ratios())
        )
        joined = " ".join(result.risk_flags).lower()
        self.assertIn("negative free cash flow", joined)
        self.assertTrue(any("leverage" in flag.lower() for flag in result.risk_flags))

    def test_sector_drives_cyclicality(self) -> None:
        cyclical = self.engine.assess(
            RiskInput(financial_data=_healthy_statements(), ratios=_healthy_ratios(), sector="Steel")
        )
        defensive = self.engine.assess(
            RiskInput(financial_data=_healthy_statements(), ratios=_healthy_ratios(), sector="Pharma")
        )
        self.assertEqual(cyclical.cyclicality_level, CyclicalityLevel.HIGHLY_CYCLICAL)
        self.assertEqual(defensive.cyclicality_level, CyclicalityLevel.STABLE)
        self.assertGreater(cyclical.cyclicality_risk, defensive.cyclicality_risk)

    def test_valuation_risk_uses_reverse_dcf(self) -> None:
        low = self.engine.assess(
            RiskInput(ratios=_healthy_ratios(), reverse_dcf=_reverse_dcf(ValuationRisk.LOW, 20.0))
        )
        high = self.engine.assess(
            RiskInput(ratios=_healthy_ratios(), reverse_dcf=_reverse_dcf(ValuationRisk.HIGH, 90.0))
        )
        self.assertGreater(high.valuation_risk, low.valuation_risk)

    def test_top_risks_ranked_and_capped(self) -> None:
        result = self.engine.assess(
            RiskInput(financial_data=_risky_statements(), ratios=_risky_ratios())
        )
        self.assertLessEqual(len(result.top_risks), 10)
        self.assertGreater(len(result.top_risks), 0)
        severities = [risk.severity for risk in result.top_risks]
        self.assertEqual(severities, sorted(severities, reverse=True))

    def test_grade_bands(self) -> None:
        self.assertEqual(self.engine._grade(85.0), RiskGrade.VERY_HIGH)
        self.assertEqual(self.engine._grade(65.0), RiskGrade.HIGH)
        self.assertEqual(self.engine._grade(45.0), RiskGrade.MODERATE)
        self.assertEqual(self.engine._grade(25.0), RiskGrade.LOW)
        self.assertEqual(self.engine._grade(10.0), RiskGrade.VERY_LOW)

    def test_ai_summary_present(self) -> None:
        result = self.engine.assess(
            RiskInput(financial_data=_risky_statements(), ratios=_risky_ratios())
        )
        self.assertIn("overall risk", result.ai_summary.lower())

    def test_empty_input_is_neutralish(self) -> None:
        result = self.engine.assess(RiskInput())
        self.assertTrue(result.notes)
        self.assertGreaterEqual(result.overall_risk_score, 0.0)
        self.assertLessEqual(result.overall_risk_score, 100.0)

    def test_type_validation(self) -> None:
        with self.assertRaises(TypeError):
            self.engine.assess(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
