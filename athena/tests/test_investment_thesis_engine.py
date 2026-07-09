import unittest

import pandas as pd

from athena.engines.business_quality_engine import (
    BusinessQualityEngine,
    BusinessQualityInput,
)
from athena.engines.dcf_engine import DCFEngine, DCFInput
from athena.engines.investment_thesis_engine import (
    InvestmentThesisEngine,
    InvestmentThesisInput,
)
from athena.engines.ratio_engine import FinancialData, RatioResult
from athena.engines.reverse_dcf_engine import ReverseDCFEngine, ReverseDCFInput
from athena.engines.risk_engine import RiskEngine, RiskInput
from athena.models.investment_thesis_models import (
    InvestmentHorizon,
    InvestmentRating,
    InvestmentThesisResult,
)


def _years(n: int) -> list[str]:
    return [f"{2018 + i}" for i in range(n)]


def _statements(growth: float) -> FinancialData:
    years = _years(6)
    revenue = [100.0 * ((1.0 + growth) ** i) for i in range(6)]
    income = pd.DataFrame(
        {
            "TotalRevenue": revenue,
            "EBIT": [value * 0.20 for value in revenue],
            "OperatingIncome": [value * 0.20 for value in revenue],
            "NetIncome": [value * 0.14 for value in revenue],
            "PretaxIncome": [value * 0.19 for value in revenue],
            "TaxProvision": [value * 0.0475 for value in revenue],
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


def _ratios() -> RatioResult:
    return RatioResult(
        roe=0.20,
        roce=0.22,
        roa=0.12,
        operating_margin=0.20,
        net_margin=0.14,
        current_ratio=2.4,
        quick_ratio=1.8,
        cash_ratio=1.0,
        debt_equity=0.2,
        interest_coverage=12.0,
        asset_turnover=0.9,
        revenue_cagr=0.12,
    )


class InvestmentThesisEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = InvestmentThesisEngine()
        self.financial = _statements(0.12)
        self.ratios = _ratios()
        self.dcf = DCFEngine().value(
            DCFInput(
                income_statement=self.financial.income_statement,
                balance_sheet=self.financial.balance_sheet,
                cash_flow=self.financial.cash_flow,
                ratios=self.ratios,
                current_price=None,
                shares_outstanding=10.0,
                market_cap=None,
            )
        )
        self.business_quality = BusinessQualityEngine().evaluate(
            BusinessQualityInput(financial_data=self.financial, ratios=self.ratios, dcf=self.dcf)
        )

    def _reverse(self, price: float) -> object:
        return ReverseDCFEngine().analyze(
            ReverseDCFInput(
                current_price=price,
                shares_outstanding=10.0,
                cash=40.0,
                debt=20.0,
                financial_data=self.financial,
                ratios=self.ratios,
                dcf=self.dcf,
            )
        )

    def _risk(self, sector: str = "FMCG") -> object:
        return RiskEngine().assess(
            RiskInput(
                financial_data=self.financial,
                ratios=self.ratios,
                dcf=self.dcf,
                business_quality=self.business_quality,
                sector=sector,
            )
        )

    def _input(self, price: float, sector: str = "FMCG") -> InvestmentThesisInput:
        return InvestmentThesisInput(
            financial_data=self.financial,
            ratios=self.ratios,
            dcf=self.dcf,
            reverse_dcf=self._reverse(price),
            business_quality=self.business_quality,
            risk=self._risk(sector),
            ticker="TEST",
            current_price=price,
        )

    def test_returns_result_type(self) -> None:
        result = self.engine.build(InvestmentThesisInput())
        self.assertIsInstance(result, InvestmentThesisResult)

    def test_cheap_rates_higher_than_expensive(self) -> None:
        intrinsic = self.dcf.intrinsic_value
        assert intrinsic is not None
        cheap = self.engine.build(self._input(price=intrinsic * 0.5))
        expensive = self.engine.build(self._input(price=intrinsic * 2.5))
        self.assertGreater(cheap.overall_score, expensive.overall_score)
        order = list(InvestmentRating)
        self.assertLessEqual(
            order.index(cheap.investment_rating), order.index(expensive.investment_rating)
        )

    def test_cheap_quality_stock_is_a_buy(self) -> None:
        intrinsic = self.dcf.intrinsic_value
        assert intrinsic is not None
        result = self.engine.build(self._input(price=intrinsic * 0.5))
        self.assertIn(
            result.investment_rating,
            {
                InvestmentRating.STRONG_BUY,
                InvestmentRating.BUY,
                InvestmentRating.ACCUMULATE,
            },
        )
        self.assertGreater(result.margin_of_safety or 0.0, 0.0)

    def test_scores_bounded(self) -> None:
        intrinsic = self.dcf.intrinsic_value
        assert intrinsic is not None
        result = self.engine.build(self._input(price=intrinsic))
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 100.0)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 100.0)

    def test_report_sections_populated(self) -> None:
        intrinsic = self.dcf.intrinsic_value
        assert intrinsic is not None
        result = self.engine.build(self._input(price=intrinsic))
        self.assertTrue(result.summary)
        self.assertTrue(result.bull_case)
        self.assertTrue(result.bear_case)
        self.assertTrue(result.valuation_summary)
        self.assertTrue(result.business_summary)
        self.assertTrue(result.financial_summary)
        self.assertTrue(result.risk_summary)
        self.assertTrue(result.key_strengths)
        self.assertGreater(len(result.major_risks), 0)
        self.assertLessEqual(len(result.key_strengths), 5)
        self.assertLessEqual(len(result.key_weaknesses), 5)
        self.assertLessEqual(len(result.major_risks), 5)

    def test_expected_returns_present(self) -> None:
        intrinsic = self.dcf.intrinsic_value
        assert intrinsic is not None
        result = self.engine.build(self._input(price=intrinsic * 0.6))
        self.assertIsNotNone(result.expected_return_3y)
        self.assertIsNotNone(result.expected_return_5y)

    def test_high_quality_long_horizon(self) -> None:
        intrinsic = self.dcf.intrinsic_value
        assert intrinsic is not None
        result = self.engine.build(self._input(price=intrinsic))
        self.assertEqual(result.investment_horizon, InvestmentHorizon.LONG_TERM)

    def test_confidence_rises_with_more_inputs(self) -> None:
        intrinsic = self.dcf.intrinsic_value
        assert intrinsic is not None
        full = self.engine.build(self._input(price=intrinsic))
        sparse = self.engine.build(
            InvestmentThesisInput(ratios=self.ratios, current_price=intrinsic)
        )
        self.assertGreater(full.confidence_score, sparse.confidence_score)

    def test_empty_input_has_notes(self) -> None:
        result = self.engine.build(InvestmentThesisInput())
        self.assertTrue(result.notes)

    def test_type_validation(self) -> None:
        with self.assertRaises(TypeError):
            self.engine.build(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
