import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from datetime import datetime, timedelta

from black_scholes import black_scholes_call


class DataFetcher:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def get_spot_history(self, days: int = 20) -> pd.DataFrame:
        """Get closing prices for the past N trading days."""
        hist = self.stock.history(period=f"{days}d")
        hist = hist.reset_index()[["Date", "Close"]]
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
        return hist

    def get_option_chain(self, expiry: str) -> pd.DataFrame:
        """Fetch calls for a given expiry."""
        try:
            chain = self.stock.option_chain(expiry).calls
            return chain[["strike", "lastPrice"]].dropna()
        except Exception:
            return pd.DataFrame(columns=["strike", "lastPrice"])

    def get_multi_day_data(self, expiries: list[str], days: int = 20) -> pd.DataFrame:
        """
        Build a multi-day dataset with option prices for several expiries.
        Returns a DataFrame with columns:
        [date, expiry, strike, lastPrice, spot]
        """
        hist = self.get_spot_history(days)
        all_records = []
        for _, row in hist.iterrows():
            date = row["Date"]
            spot = row["Close"]
            for expiry in expiries:
                chain = self.get_option_chain(expiry)
                if len(chain) == 0:
                    continue
                chain = chain.copy()
                chain["date"] = date
                chain["expiry"] = expiry
                chain["spot"] = spot
                all_records.append(chain)
        if not all_records:
            raise ValueError("No option data could be fetched.")
        return pd.concat(all_records, ignore_index=True)



class ImpliedVolCalculator:
    def __init__(self, r: float = 0.05):
        self.r = r

    @staticmethod
    def time_to_maturity(expiry: str, date: datetime) -> float:
        """Compute time to expiry (in years) from a given trade date."""
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        days = (expiry_date - date).days
        return max(days, 1) / 365.0

    def call_iv(self, S: float, K: float, T: float, market_price: float) -> float:
        """Invert BS formula to compute implied vol."""
        try:
            return brentq(
                lambda sigma: black_scholes_call(S, K, T, self.r, sigma) - market_price,
                1e-6, 5.0,
            )
        except ValueError:
            return np.nan

    def compute_for_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add implied volatility column to a DataFrame with spot, strike, date, expiry, and price."""
        ivs = []
        for _, row in df.iterrows():
            T = self.time_to_maturity(row["expiry"], row["date"])
            iv = self.call_iv(row["spot"], row["strike"], T, row["lastPrice"])
            ivs.append(iv)
        df = df.copy()
        df["T"] = [
            self.time_to_maturity(exp, d) for exp, d in zip(df["expiry"], df["date"])
        ]
        df["iv"] = ivs
        df["moneyness"] = df["strike"] / df["spot"]
        return df.dropna(subset=["iv"])


