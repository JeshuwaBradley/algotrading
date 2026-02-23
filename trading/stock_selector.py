import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from .indicators import TechnicalIndicators
from .data_fetcher import DataFetcher
import pandas_ta_classic as ta
from config import config


class StockSelector:
    """Selects stocks based on various technical criteria"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.data_fetcher = DataFetcher()
    
    def check_for_crosses(self, df: pd.DataFrame) -> List:
        """Check for price crossing above cloud"""
        crosses = []
        for i in range(1, len(df.index)):
            if (df["Close"].iloc[i] > df["span_a"].iloc[i] and 
                df["Close"].iloc[i-1] < df["span_b"].iloc[i-1] and 
                df["Close"].iloc[i] > df["span_b"].iloc[i]):
                crosses.append(df.index[i])
        return crosses
    
    def check_for_short_crosses(self, df: pd.DataFrame) -> List:
        """Check for price crossing below cloud"""
        crosses = []
        for i in range(1, len(df.index)):
            if (df["Close"].iloc[i] < df["span_a"].iloc[i] and 
                df["Close"].iloc[i-1] > df["span_b"].iloc[i-1] and 
                df["Close"].iloc[i] < df["span_b"].iloc[i]):
                crosses.append(df.index[i])
        return crosses
    
    def latest_price_cross_cloud(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
        """Get latest price cross of cloud"""
        df = df.iloc[:-26].copy()
        crosses = self.check_for_crosses(df)
        short_crosses = self.check_for_short_crosses(df)
        
        latest_long_cross = crosses[-1] if crosses else None
        latest_short_cross = short_crosses[-1] if short_crosses else None
        
        if latest_long_cross and latest_short_cross:
            if latest_long_cross > latest_short_cross:
                return "long", latest_long_cross
            else:
                return "short", latest_short_cross
        elif latest_long_cross:
            return "long", latest_long_cross
        elif latest_short_cross:
            return "short", latest_short_cross
        else:
            return None, None
    
    def run_all(self, tickers: List[str], tickers_data: Dict) -> Tuple[List[str], List[str]]:
        """Run Ichimoku analysis for all tickers"""
        latest_crosses = []
        latest_short_crosses = []
        
        for ticker in tickers:
            if ticker not in tickers_data or tickers_data[ticker] is None or tickers_data[ticker].empty:
                continue
            
            df = tickers_data[ticker].copy()
            
            # Ensure required Ichimoku columns are present
            if not all(col in df.columns for col in config.ICHIMOKU_COLS):
                df = self.indicators.ichimoku(df, ticker)
                if df is None:
                    continue
            
            # Get the latest cross and its type
            term, cross_date = self.latest_price_cross_cloud(df)
            
            if cross_date is not None:
                if term == "long":
                    latest_crosses.append((ticker, cross_date))
                elif term == "short":
                    latest_short_crosses.append((ticker, cross_date))
        
        # Sort lists by the latest cross date
        sorted_latest_crosses = [ticker for ticker, date in sorted(latest_crosses, key=lambda x: x[1], reverse=True)]
        sorted_latest_short_crosses = [ticker for ticker, date in sorted(latest_short_crosses, key=lambda x: x[1], reverse=True)]
        
        return sorted_latest_crosses, sorted_latest_short_crosses
    
    def improve_data(self, data: pd.DataFrame, period1: int, period2: int, date: str) -> Tuple[Optional[bool], Optional[pd.Timestamp]]:
        """Check for golden/death cross and return result"""
        data = self.indicators.calculate_moving_average(data, period1, period2)
        data.dropna(inplace=True)
        new_date = self.data_fetcher.get_date_three_months_before(date)
        stock_data = data.loc[new_date:]
        golden_crosses, death_crosses = self.indicators.check_golden_cross(stock_data)
        
        latest_golden_cross = golden_crosses[-1] if golden_crosses else None
        latest_death_cross = death_crosses[-1] if death_crosses else None
        
        if latest_golden_cross is None and latest_death_cross is not None:
            return False, latest_death_cross
        elif latest_golden_cross is not None and latest_death_cross is None:
            return True, latest_golden_cross
        elif latest_golden_cross is None and latest_death_cross is None:
            return None, None
        elif latest_golden_cross > latest_death_cross:
            return True, latest_golden_cross
        elif latest_death_cross > latest_golden_cross:
            return False, latest_death_cross
        else:
            return None, None
    
    def get_stocks_with_golden_crosses(self, lst: List[str], short: int, long: int, data: Dict, date: str) -> Tuple[List[str], List[str]]:
        """Filter stocks with golden crosses"""
        stocks_with_long_term_golden_crosses = []
        stocks_with_long_term_death_crosses = []
        
        for ticker in lst:
            stock_data = data[ticker].copy()
            result, latest_cross = self.improve_data(stock_data, short, long, date)
            if result is None:
                continue
            
            if result:
                stocks_with_long_term_golden_crosses.append((ticker, latest_cross))
            else:
                stocks_with_long_term_death_crosses.append((ticker, latest_cross))
        
        sorted_data = sorted(stocks_with_long_term_golden_crosses, key=lambda x: x[1], reverse=True)
        sorted_data_death = sorted(stocks_with_long_term_death_crosses, key=lambda x: x[1], reverse=True)
        
        sorted_data_lst = [tuple[0] for tuple in sorted_data]
        sorted_data_lst_death = [tuple[0] for tuple in sorted_data_death]
        
        return sorted_data_lst, sorted_data_lst_death
    
    def stock_trend_rsi(self, tickers_list: List[str], data: Dict, overbought: int = 70, oversold: int = 30) -> Tuple[List[str], List[str]]:
        """Filter stocks based on RSI trend"""
        upward_trend = []
        downward_trend = []
        
        for ticker in tickers_list:
            stock_data = data[ticker].iloc[:-26].copy()
            stock_data["RSI"] = ta.rsi(stock_data["Close"], window=7)
            
            try:
                if stock_data["RSI"].iloc[-1] > overbought or stock_data["RSI"].iloc[-1] < oversold:
                    downward_trend.append(ticker)
                else:
                    upward_trend.append(ticker)
            except:
                continue
        return upward_trend, downward_trend
    
    def stock_trend_macd(self, tickers_list: List[str], data: Dict) -> Tuple[List[str], List[str]]:
        """Filter stocks based on MACD trend"""
        upward_trend = []
        downward_trend = []
        
        for ticker in tickers_list:
            stock_data = data[ticker].iloc[:-26].copy()
            macd = ta.macd(close=stock_data['Close'], fast=5, slow=15, signal=3)
            stock_data = pd.concat([stock_data, macd], axis=1).reindex(stock_data.index)
            stock_data.rename(columns={
                'MACD_5_15_3': 'MACD',
                "MACDh_5_15_3": "Histogram",
                "MACDs_5_15_3": "Signal_Line"
            }, inplace=True)
            
            MACD = stock_data["MACD"].iloc[-1]
            Signal_Line = stock_data["Signal_Line"].iloc[-1]
            histogram = stock_data["Histogram"].iloc[-1]
            prev_MACD = stock_data["MACD"].iloc[-2]
            prev_Signal_Line = stock_data["Signal_Line"].iloc[-2]
            
            if MACD > Signal_Line and MACD > histogram and prev_MACD < MACD and prev_Signal_Line < Signal_Line:
                upward_trend.append(ticker)
            else:
                downward_trend.append(ticker)
        return upward_trend, downward_trend
    
    def most_volatile(self, tickers_list: List[str], data: Dict, limit: int = 300) -> List[str]:
        """Get most volatile stocks based on volume"""
        new_list = []
        for ticker in tickers_list:
            ticker_data = data[ticker]["Volume"].iloc[-1]
            new_list.append((ticker, ticker_data))
        new_list.sort(key=lambda a: a[1], reverse=True)
        new_list = list(x[0] for x in new_list)
        return new_list[:limit]
    
    def stock_selecting_code_1(self, ticker_list: List[str], date: str) -> Tuple[List[str], List]:
        """First stock selection strategy"""
        original_data, original_tickers = self.data_fetcher.download_stock_data(ticker_list, "2023-01-01", date)
        
        tickers = self.most_volatile(original_tickers, original_data)
        buy_signals, _ = self.run_all(tickers, original_data)
        
        long_term_lst, _ = self.get_stocks_with_golden_crosses(buy_signals, 50, 200, original_data, date)
        short_term_lst, _ = self.get_stocks_with_golden_crosses(long_term_lst, 20, 50, original_data, date)
        rsi_up, _ = self.stock_trend_rsi(short_term_lst, original_data)
        macd_up, _ = self.stock_trend_macd(rsi_up, original_data)
        
        return macd_up[:300], []
    
    def stock_selecting_code_2(self, ticker_list: List[str], date: str) -> Tuple[List[str], List]:
        """Second stock selection strategy"""
        try:
            original_data, original_tickers = self.data_fetcher.download_stock_data(ticker_list, "2023-01-01", date)
            
            buy_signals, _ = self.run_all(original_tickers, original_data)
            long_term_lst, _ = self.get_stocks_with_golden_crosses(buy_signals, 20, 50, original_data, date)
            short_term_lst, _ = self.get_stocks_with_golden_crosses(long_term_lst, 5, 20, original_data, date)
            rsi_up, _ = self.stock_trend_rsi(short_term_lst, original_data)
            macd_up, _ = self.stock_trend_macd(rsi_up, original_data)
            
            return macd_up[:300], []
        except Exception as e:
            print(f"Error in stock_selecting_code_2: {e}")
            return [], []