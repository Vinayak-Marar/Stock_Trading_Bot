import yfinance as yf

stocks_ticker = ["AAPL","MSFT","GOOG"]

data = yf.download(stocks_ticker, start="2020-01-01", end="2023-01-01", auto_adjust=True)

data = data["Close"]

data.to_csv("data.csv")