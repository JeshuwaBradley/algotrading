import pandas as pd
import requests
from bs4 import BeautifulSoup
from .indicators import ichimoku, calculate_moving_average, determine_trend_strength
from .data_fetcher import download_stock_data, get_date_three_months_before
import pandas_ta as ta

def check_for_crosses(df):
    crosses = []
    for i in range(1, len(df.index)):
        if (df["Close"].iloc[i] > df["span_a"].iloc[i] and 
            df["Close"].iloc[i-1] < df["span_b"].iloc[i-1] and 
            df["Close"].iloc[i] > df["span_b"].iloc[i]):
            crosses.append(df.index[i])
    return crosses

def check_for_short_crosses(df):
    crosses = []
    for i in range(1, len(df.index)):
        if (df["Close"].iloc[i] < df["span_a"].iloc[i] and 
            df["Close"].iloc[i-1] > df["span_b"].iloc[i-1] and 
            df["Close"].iloc[i] < df["span_b"].iloc[i]):
            crosses.append(df.index[i])
    return crosses

def latest_price_cross_cloud(df):
    df = df.iloc[:-26].copy()
    crosses = check_for_crosses(df)
    short_crosses = check_for_short_crosses(df)
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

def run_all(tickers, tickers_data):
    latest_crosses = []
    latest_short_crosses = []

    for ticker in tickers:
        if ticker not in tickers_data or tickers_data[ticker] is None or tickers_data[ticker].empty:
            continue

        df = tickers_data[ticker].copy()

        required_cols = ["span_a", "span_b", "Tenkan San 9", "Kijun San 26", "26 day lag"]
        if not all(col in df.columns for col in required_cols):
            df = ichimoku(df, ticker)
            if df is None:
                continue

        term, cross_date = latest_price_cross_cloud(df)

        if cross_date is not None:
            if term == "long":
                latest_crosses.append((ticker, cross_date))
            elif term == "short":
                latest_short_crosses.append((ticker, cross_date))

    sorted_latest_crosses = [ticker for ticker, date in sorted(latest_crosses, key=lambda x: x[1], reverse=True)]
    sorted_latest_short_crosses = [ticker for ticker, date in sorted(latest_short_crosses, key=lambda x: x[1], reverse=True)]

    return sorted_latest_crosses, sorted_latest_short_crosses

def check_for_cloud_crossover(data):
    crosses = []
    for i in range(1, len(data.index)):
        if (data["Close"].iloc[i] >= data["Tenkan San 9"].iloc[i] and 
            data["Close"].iloc[i-1] <= data["Tenkan San 9"].iloc[i-1]):
            crosses.append(data.index[i])
    return crosses

def check_buy_signal(tickers_list, data, condition_name):
    buy_signals = []
    sell_signals = []
    for ticker in tickers_list:
        if ticker not in data or data[ticker] is None or data[ticker].empty:
            continue
        df = data[ticker].copy()
        cols = ["span_a", "span_b", "Tenkan San 9", "Kijun San 26", "26 day lag"]
        if not all(col in df.columns for col in cols):
            df = ichimoku(df, ticker)

        last_3_days = df.iloc[:-26].iloc[-3:]
        conditions = {
            "condition1": (last_3_days["Close"] > last_3_days["span_a"]).fillna(False).astype(bool),
            "condition2": (df.iloc[-26:]["span_a"] > df.iloc[-26:]["span_b"]).fillna(False).astype(bool),
            "condition3": (df.iloc[:-26]["Tenkan San 9"] > df.iloc[:-26]["Kijun San 26"]).fillna(False).astype(bool),
            "condition4": (df.iloc[:-52]["26 day lag"] > df.iloc[:-52]["span_a"]).fillna(False).astype(bool),
        }

        valid_conditions = [condition for condition in condition_name if condition in conditions]
        buy_signal = all(conditions[name].iloc[-1] for name in valid_conditions)
        if buy_signal:
            crossover_dates = check_for_cloud_crossover(df)

            if len(crossover_dates) != 0:
                last_crossover_date = crossover_dates[-1]
                buy_signals.append((ticker, last_crossover_date))
            else:
                buy_signals.append((ticker, None))
        else:
            sell_signals.append(ticker)

    sorted_data = sorted(buy_signals, key=lambda x: x[1], reverse=True)
    sorted_data_lst = [tuple[0] for tuple in sorted_data]
    return sorted_data_lst, sell_signals

def check_golden_cross(data):
    golden_crosses = []
    death_crosses = []
    for i in range(1, len(data.index)):
        if (data["Short_Moving"].iloc[i] >= data["Long_Moving"].iloc[i] and 
            data["Short_Moving"].iloc[i-1] <= data["Long_Moving"].iloc[i-1]):
            golden_crosses.append(data.index[i])
        elif (data["Short_Moving"].iloc[i] <= data["Long_Moving"].iloc[i] and 
              data["Short_Moving"].iloc[i-1] >= data["Long_Moving"].iloc[i-1]):
            death_crosses.append(data.index[i])
    return golden_crosses, death_crosses

def improve_data(data, period1, period2, date):
    data = calculate_moving_average(data, period1, period2)
    data.dropna(inplace=True)
    new_date = get_date_three_months_before(date)
    stock_data = data.loc[new_date:]
    golden_crosses, death_crosses = check_golden_cross(stock_data)

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

def get_stocks_with_golden_crosses(lst, short, long, data, date):
    stocks_with_long_term_golden_crosses = []
    stocks_with_long_term_death_crosses = []

    for ticker in lst:
        stock_data = data[ticker].copy()
        result, latest_golden_cross = improve_data(stock_data, short, long, date)
        if result is None:
            continue

        if result:
            stocks_with_long_term_golden_crosses.append((ticker, latest_golden_cross))
        else:
            stocks_with_long_term_death_crosses.append((ticker, latest_golden_cross))

    sorted_data = sorted(stocks_with_long_term_golden_crosses, key=lambda x: x[1], reverse=True)
    sorted_data_death = sorted(stocks_with_long_term_death_crosses, key=lambda x: x[1], reverse=True)

    sorted_data_lst = [tuple[0] for tuple in sorted_data]
    sorted_data_lst_death = [tuple[0] for tuple in sorted_data_death]

    return sorted_data_lst, sorted_data_lst_death

def is_uptrend(data):
    if data['RSI'].iloc[-1] > 50 and data["RSI"].iloc[-1] < 70:
        recent_rsi = data['RSI'].iloc[-7:]
        x = np.arange(len(recent_rsi))
        slope, _, _, _, _ = linregress(x, recent_rsi)
        if slope > 0:
            return True
    return False

def stock_trend_rsi(tickers_list, data, overbought=70, oversold=30):
    import pandas_ta as ta
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

def is_macd_uptrend(data):
    recent_macd = data['MACD'].iloc[-7:]
    recent_signal = data['Signal_Line'].iloc[-7:]
    recent_histogram = data['Histogram'].iloc[-7:]

    if (recent_macd <= recent_signal).any() or (recent_histogram <= 0).any():
        return False

    x = np.arange(len(recent_macd))
    slope, _, _, _, _ = linregress(x, recent_macd)
    if slope > 0:
        return True

    return False

def stock_trend_macd(tickers_list, data, fast_ema=12, slow_ema=26, signal_ema=9):
    import pandas_ta as ta
    upward_trend = []
    downward_trend = []

    for ticker in tickers_list:
        stock_data = data[ticker].iloc[:-26].copy()
        macd = ta.macd(close=stock_data['Close'], fast=5, slow=15, signal=3)
        stock_data = pd.concat([stock_data, macd], axis=1).reindex(stock_data.index)
        stock_data.rename(columns={'MACD_5_15_3': 'MACD',
                                   "MACDh_5_15_3": "Histogram",
                                   "MACDs_5_15_3": "Signal_Line"}, inplace=True)

        MACD = stock_data["MACD"].iloc[-1]
        Signal_Line = stock_data["Signal_Line"].iloc[-1]
        histogram = stock_data["Histogram"].iloc[-1]
        prev_MACD = stock_data["MACD"].iloc[-2]
        prev_Signal_Line = stock_data["Signal_Line"].iloc[-2]
        if (MACD > Signal_Line and MACD > histogram and 
            prev_MACD < MACD and prev_Signal_Line < Signal_Line):
            upward_trend.append(ticker)
        else:
            downward_trend.append(ticker)
    return upward_trend, downward_trend

def stock_trend_adx(tickers_list, data):
    upward_strong_trend = []
    for ticker in tickers_list:
        stock_data = data[ticker].iloc[:-26].copy()
        stock_data = determine_trend_strength(stock_data)
        if (stock_data.iloc[-1]["Trend"] == "Upward" and 
            (stock_data.iloc[-1]["Strength"] == "Very Strong" or stock_data.iloc[-1]["Strength"] == "Strong")):
            upward_strong_trend.append(ticker)

    return upward_strong_trend

def most_volatile(tickers_list, data, limit=300):
    new_list = []
    for ticker in tickers_list:
        ticker_data = data[ticker]["Volume"].iloc[-1]
        new_list.append((ticker, ticker_data))
    new_list.sort(key=lambda a: a[1], reverse=True)
    new_list = [x[0] for x in new_list]
    return new_list[:limit]

def most_volatile_two(tickers_list, data):
    new_list = []
    for ticker in tickers_list:
        ticker_data = data[ticker]["Volume"].iloc[-1]
        new_list.append((ticker, ticker_data))
    new_list.sort(key=lambda a: a[1], reverse=True)
    new_list = [x[0] for x in new_list]
    return new_list

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    rows = table.find_all("tr")[1:]
    sp500 = []
    for row in rows:
        sp500.append(row.find_all('td')[0].text.strip())
    return sp500

def stock_selecting_code_1(ticker_list, date):
    from .data_fetcher import download_stock_data
    original_data, original_tickers = download_stock_data(ticker_list, "2023-01-01", date)

    tickers = most_volatile(original_tickers, original_data)
    buy_signals, sell_signals = run_all(tickers, original_data)

    long_term_lst, _ = get_stocks_with_golden_crosses(buy_signals, 50, 200, original_data, date)
    short_term_lst, _ = get_stocks_with_golden_crosses(long_term_lst, 20, 50, original_data, date)
    rsi_up, _ = stock_trend_rsi(short_term_lst, original_data)
    macd_up, _ = stock_trend_macd(rsi_up, original_data)
    macd_up = most_volatile_two(macd_up, original_data)

    return macd_up, []

def stock_selecting_code_2(ticker_list, date):
    try:
        original_data, original_tickers = download_stock_data(ticker_list, "2023-01-01", date)

        buy_signals, sell_signals = run_all(original_tickers, original_data)
        long_term_lst, _ = get_stocks_with_golden_crosses(buy_signals, 20, 50, original_data, date)
        short_term_lst, _ = get_stocks_with_golden_crosses(long_term_lst, 5, 20, original_data, date)
        rsi_up, _ = stock_trend_rsi(short_term_lst, original_data)
        macd_up, _ = stock_trend_macd(rsi_up, original_data)

        macd_up = most_volatile_two(macd_up, original_data)

        return macd_up, []
    except:
        return [], []

def stock_selecting_code():
    sp500 = get_sp500_tickers()
    from datetime import date
    initial_list, _ = stock_selecting_code_1(sp500, str(date.today()))
    return initial_list, []