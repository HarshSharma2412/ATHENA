import os
import tempfile
import unittest

from athena.database.database import DatabaseManager, create_database
from athena.database.models import Company, PriceHistory, Ratios
from athena.database.repository import AthenaRepository


class DatabaseLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "athena_test.sqlite3")
        self.manager = create_database(self.db_path)

    def tearDown(self) -> None:
        self.manager.close()
        self.temp_dir.cleanup()

    def test_database_initializes_and_persists_entities(self) -> None:
        company = self.manager.save_company(
            ticker="TATAMOTORS",
            company_name="Tata Motors",
            sector="Automotive",
            market_cap=1000000000.0,
            currency="INR",
        )
        self.assertEqual(company.ticker, "TATAMOTORS")

        self.manager.save_financials(
            ticker="TATAMOTORS",
            statement_type="income",
            period="2024",
            data={"Revenue": 1000.0, "NetIncome": 100.0},
        )
        self.manager.save_price_history(
            ticker="TATAMOTORS",
            history=[{"date": "2024-01-01", "close": 10.0}],
        )
        self.manager.save_ratios(
            ticker="TATAMOTORS",
            ratios={"roe": 0.2, "current_ratio": 1.5},
        )

        loaded_company = self.manager.load_company("TATAMOTORS")
        self.assertIsNotNone(loaded_company)
        self.assertEqual(loaded_company.company_name, "Tata Motors")

        financials = self.manager.load_financials("TATAMOTORS")
        self.assertGreaterEqual(len(financials), 1)

        prices = self.manager.load_price_history("TATAMOTORS")
        self.assertEqual(len(prices), 1)

        ratios = self.manager.load_ratios("TATAMOTORS")
        self.assertGreaterEqual(len(ratios), 1)

    def test_repository_wrapper_works(self) -> None:
        repo = AthenaRepository(self.manager)
        company = repo.save_company(ticker="AAPL", company_name="Apple")
        self.assertIsInstance(company, Company)
        self.assertEqual(repo.load_company("AAPL").company_name, "Apple")

        prices = repo.save_price_history("AAPL", [{"date": "2024-01-01", "close": 100.0}])
        self.assertIsInstance(prices[0], PriceHistory)

        ratios = repo.save_ratios("AAPL", {"roe": 0.2})
        self.assertIsInstance(ratios[0], Ratios)


if __name__ == "__main__":
    unittest.main()
