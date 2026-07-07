import unittest

from athena.services.company_search_engine import CompanySearchEngine


class CompanySearchEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = CompanySearchEngine()

    def test_search_returns_partial_company_matches(self) -> None:
        results = self.engine.search("hc")
        tickers = [result["ticker"] for result in results]

        self.assertIn("HCLTECH", tickers)
        self.assertIn("HDFCBANK", tickers)
        self.assertIn("HCG", tickers)

    def test_search_supports_ticker_matches(self) -> None:
        results = self.engine.search("waa")

        self.assertEqual(results[0]["ticker"], "WAAREEENER")
        self.assertEqual(results[0]["company_name"], "Waaree Energies Ltd")

    def test_search_is_case_insensitive(self) -> None:
        lower_result = self.engine.search("hcltech")
        upper_result = self.engine.search("HCLTECH")

        self.assertEqual(lower_result[0], upper_result[0])

    def test_load_company_master_is_cached(self) -> None:
        first_load = self.engine.load_company_master()
        second_load = self.engine.load_company_master()

        self.assertIs(first_load, second_load)

    def test_search_returns_top_15_matches(self) -> None:
        results = self.engine.search("a")

        self.assertLessEqual(len(results), 15)


if __name__ == "__main__":
    unittest.main()
