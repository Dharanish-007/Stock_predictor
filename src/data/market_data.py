import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def fetch(self, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical market data from Yahoo Finance
        """
        logger.info(f"Fetching data for {self.symbol} with period {period} and interval {interval}")
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
