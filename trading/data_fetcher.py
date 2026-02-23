import yfinance as yf
import pandas as pd
import datetime
import time
import pytz
from typing import Dict, List, Tuple, Optional
from pandas import DatetimeIndex
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DataFetcher:
    """Handles all data fetching operations from Yahoo Finance"""
    
    @staticmethod
    def download_stock_data(tickers: List[str], start_date: str, end_date: str, interval: str = "1d") -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Download stock data for multiple tickers
        
        Returns:
            Tuple of (stock_data_dict, successful_tickers)
        """
        stock_data = {}
        remaining_tickers = []
        
        for ticker in tickers:
            try:
                df = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
                time.sleep(1)
                if not df.empty and len(df.index) > 100:
                    stock_data[ticker] = df
                else:
                    remaining_tickers.append(ticker)
            except Exception as e:
                print(f"Error downloading data for {ticker}: {e}")
                remaining_tickers.append(ticker)
                continue
                
        return stock_data, list(set(tickers) - set(remaining_tickers))
    
    @staticmethod
    def get_current_stock_price(ticker: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current stock price and average fluctuation over 7 days
        """
        # Standardize ticker for crypto
        if ticker == "DOGEUSD":
            ticker = "DOGE-USD"
            
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period='1d')
            
            if not data.empty:
                current_price = data['Close'].iloc[-1]
            else:
                current_price = None
                
            # For average fluctuation over 7 days
            hist_7d = stock.history(period='7d')
            if not hist_7d.empty:
                hist_7d["fluc"] = hist_7d["High"] - hist_7d["Low"]
                average_fluc = hist_7d["fluc"].mean()
            else:
                average_fluc = None
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            current_price, average_fluc = None, None
            
        return current_price, average_fluc
    
    @staticmethod
    def get_dates(period: int) -> Tuple[datetime.datetime, datetime.datetime]:
        """Get start and end dates for data download"""
        utc = datetime.datetime.now(pytz.utc)
        tz = pytz.timezone("US/Eastern")
        ny = utc.astimezone(tz)
        ny_datetime_index = DatetimeIndex([ny])
        today = ny_datetime_index[0]
        seventh_date = today - datetime.timedelta(days=period)
        return today, seventh_date
    
    @staticmethod
    def download_intraday_data(ticker: str, period: int) -> pd.DataFrame:
        """Download intraday data for a ticker"""
        interval = "15m"
        end, start = DataFetcher.get_dates(period)
        data = yf.Ticker(ticker).history(interval="15m", start=start, end=end)
        return data
    
    @staticmethod
    def get_date_three_months_before(given_date) -> str:
        """Get date three months before given date"""
        if isinstance(given_date, str):
            given_date = datetime.datetime.strptime(given_date, "%Y-%m-%d")
        date_three_months_before = given_date - relativedelta(months=3)
        return date_three_months_before.strftime('%Y-%m-%d')

    @staticmethod
    def get_sp500_tickers() -> List[str]:
        """Fetch S&P 500 tickers from Wikipedia with fallback options"""
        import requests
        from bs4 import BeautifulSoup
        
        # Try primary method
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Try different table selectors
            table = soup.find("table", {"id": "constituents"})
            if not table:
                table = soup.find("table", {"class": "wikitable sortable"})
            
            if table:
                rows = table.find_all("tr")[1:]  # Skip header
                sp500 = []
                for row in rows:
                    cells = row.find_all('td')
                    if cells:
                        ticker = cells[0].text.strip()
                        # Clean up ticker (remove exchanges in parentheses, etc.)
                        ticker = ticker.replace('.', '-')  # Convert BRK.B to BRK-B for yfinance
                        sp500.append(ticker)
                
                if sp500:
                    print(f"Successfully fetched {len(sp500)} S&P 500 tickers from Wikipedia")
                    return sp500
        except Exception as e:
            print(f"Error fetching from Wikipedia primary method: {e}")
        
        # Fallback method 1: Hardcoded list of major tickers
        print("Using fallback ticker list...")
        fallback_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "LLY", "V", "JPM",
            "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "MRK", "ABBV", "CVX",
            "KO", "PEP", "WMT", "BAC", "TMO", "CSCO", "MCD", "NFLX", "AMD", "INTC",
            "DIS", "ABT", "CRM", "VZ", "CMCSA", "NKE", "PFE", "TXN", "NEE", "QCOM",
            "RTX", "HON", "UNP", "LOW", "SPGI", "UPS", "SBUX", "MS", "GS", "C",
            "BLK", "PLD", "DE", "CAT", "AXP", "AMGN", "GILD", "ADI", "LMT", "MDT",
            "TMUS", "AMAT", "BKNG", "LRCX", "ELV", "CI", "SYK", "VRTX", "ZTS", "REGN",
            "BSX", "ADP", "PANW", "ISRG", "MU", "KLAC", "MRNA", "ABNB", "UBER", "LYFT",
            "DASH", "SNOW", "PLTR", "COIN", "RBLX", "SHOP", "SQ", "PYPL", "ZM", "DOCU"
        ]
        print(f"Using fallback list with {len(fallback_tickers)} tickers")
        return fallback_tickers