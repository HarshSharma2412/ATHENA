import unittest

from athena.services.company_search import CompanySearchService


class CompanySearchServiceTests(unittest.TestCase):

    def setUp(self) -> None:
        self.service = CompanySearchService()

    def test_search_company_returns_expected_metadata(self) -> None:
        result = self.service.search_company("TCS")

        self.assertEqual(result["company_name"], "Tata Consultancy Services Limited")
        self.assertEqual(result["ticker"], "TCS.NS")
        self.assertEqual(result["exchange"], "NSE")
        self.assertEqual(result["sector"], "Information Technology")

    def test_search_company_accepts_company_name(self) -> None:
        result = self.service.search_company("Tata Consultancy")

        self.assertEqual(result["ticker"], "TCS.NS")
        self.assertEqual(result["company_name"], "Tata Consultancy Services Limited")

    def test_search_company_handles_invalid_query(self) -> None:
        result = self.service.search_company("Unknown Company")

        self.assertIsNone(result["company_name"])
        self.assertIsNone(result["ticker"])
        self.assertIsNone(result["exchange"])
        self.assertIsNone(result["sector"])

    def test_resolve_ticker_returns_nse_ticker(self) -> None:
        ticker = self.service.resolve_ticker("KPIT")

        self.assertEqual(ticker, "KPITTECH.NS")

    def test_resolve_ticker_returns_none_for_invalid_query(self) -> None:
        ticker = self.service.resolve_ticker("NoSuchCompany")

        self.assertIsNone(ticker)

    def test_search_cache_works_for_repeated_queries(self) -> None:
        first_result = self.service.search_company("Reliance")
        second_result = self.service.search_company("Reliance")

        self.assertIs(first_result, second_result)

    def test_search_company_is_case_insensitive(self) -> None:
        result = self.service.search_company("happiest minds")

        self.assertEqual(result["ticker"], "HAPPSTMNDS.NS")

    def test_resolve_ticker_for_waaree(self) -> None:
        ticker = self.service.resolve_ticker("Waaree")

        self.assertEqual(ticker, "WAAREEENER.NS")


if __name__ == "__main__":
    unittest.main()
