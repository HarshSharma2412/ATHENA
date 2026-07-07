import yfinance as yf


class FinancialEngine:

    def get_company_info(self, ticker):

        ticker = ticker.upper().strip()

        if not ticker.endswith(".NS"):

            ticker += ".NS"

        stock = yf.Ticker(ticker)

        info = stock.info

        return {

            "Company": info.get("longName"),

            "Sector": info.get("sector"),

            "Industry": info.get("industry"),

            "Current Price": info.get("currentPrice"),

            "Market Cap": info.get("marketCap"),

            "PE": info.get("trailingPE"),

            "PB": info.get("priceToBook"),

            "Dividend Yield": info.get("dividendYield"),

            "ROE": info.get("returnOnEquity"),

            "Debt": info.get("debtToEquity")

        }