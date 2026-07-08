import unittest

from athena.services.company_search_engine import CompanySearchEngine


class CompanySearchRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = CompanySearchEngine()

    def test_exact_symbol_ranks_first(self) -> None:
        results = self.engine.search("tcs")
        self.assertEqual(results[0]["ticker"], "TCS")

    def test_prefix_matches_surface_together(self) -> None:
        tickers = [result["ticker"] for result in self.engine.search("tata")]
        self.assertTrue(any(ticker.startswith("TATA") for ticker in tickers))
        self.assertIn("TATAPOWER", tickers)

    def test_contains_matches_banks(self) -> None:
        tickers = [result["ticker"] for result in self.engine.search("bank")]
        self.assertIn("HDFCBANK", tickers)
        self.assertIn("ICICIBANK", tickers)

    def test_search_by_symbol_ignores_name(self) -> None:
        results = self.engine.search_by_symbol("reliance")
        self.assertTrue(all("RELIANCE" in result["ticker"] for result in results))

    def test_results_expose_sector_for_dropdown(self) -> None:
        result = self.engine.search("hdfcbank")[0]
        self.assertEqual(result["ticker"], "HDFCBANK")
        self.assertTrue(result["company_name"])
        self.assertIn("sector", result)

    def test_blank_query_returns_empty(self) -> None:
        self.assertEqual(self.engine.search("   "), [])


if __name__ == "__main__":
    unittest.main()
