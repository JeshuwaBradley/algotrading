import yfinance as yf
import pandas as pd
import datetime
import time
import pytz
from pandas import DatetimeIndex
from dateutil.relativedelta import relativedelta

def download_stock_data(tickers, start_date, end_date, interval="1d"):
    stock_data = {}
    remaining_tickers = []
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(start=start_date, end=end_date, interval="1d")
            time.sleep(1)
            if not df.empty and len(df.index) > 100:
                stock_data[ticker] = df
            else:
                remaining_tickers.append(ticker)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
            remaining_tickers.append(ticker)
            continue

    successful_tickers = list(set(tickers) - set(remaining_tickers))
    return stock_data, successful_tickers

def get_dates(period):
    utc = datetime.datetime.now(pytz.utc)
    tz = pytz.timezone("US/Eastern")
    ny = utc.astimezone(tz)
    ny_datetime_index = DatetimeIndex([ny])
    today = ny_datetime_index[0]
    seventh_date = today - datetime.timedelta(days=period)
    return today, seventh_date

def download_data(ticker, period):
    period = period
    interval = "15m"
    end, start = get_dates(period)
    data = yf.Ticker(ticker).history(interval="15m", start=start, end=end)
    return data

def get_current_stock_price(ticker):
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

def get_date_three_months_before(given_date):
    if isinstance(given_date, str):
        given_date = datetime.datetime.strptime(given_date, "%Y-%m-%d")
    date_three_months_before = given_date - relativedelta(months=3)
    return date_three_months_before.strftime('%Y-%m-%d')

def get_data_to_date(date, data):
    new_data = {}
    for ticker in data:
        x = data[ticker].loc[:date].copy()
        new_data[ticker] = x
    return new_data