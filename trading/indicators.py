import pandas as pd
import pandas_ta_classic as ta
import numpy as np
from scipy.stats import linregress

def ichimoku(data, ticker):
    original = data.copy()
    ichimoku = ta.ichimoku(high=data["High"], low=data["Low"], close=data["Close"])
    isa_9_series_1 = ichimoku[0]["ISA_9"]
    isa_9_series_2 = ichimoku[1]["ISA_9"]
    combined_isa_9 = pd.DataFrame(pd.concat([isa_9_series_1, isa_9_series_2]).sort_index().ffill())
    isb_9_series_1 = ichimoku[0]["ISB_26"]
    isb_9_series_2 = ichimoku[1]["ISB_26"]
    combined_isb_9 = pd.DataFrame(pd.concat([isb_9_series_1, isb_9_series_2]).sort_index().ffill())
    ichimoku[0].drop(["ISA_9", "ISB_26"], axis=1, inplace=True)
    ichimoku = pd.concat([ichimoku[0], combined_isa_9, combined_isb_9], axis=1)
    ichimoku = ichimoku.rename(columns={
        'ISA_9': 'span_a',
        'ISB_26': 'span_b',
        'ITS_9': "Tenkan San 9",
        'IKS_26': "Kijun San 26",
        'ICS_26': "26 day lag"
    })
    combined = pd.concat([original, ichimoku], axis=1)
    return combined

def calculate_moving_average(data, period1, period2):
    min_value = min(period1, period2)
    max_value = max(period1, period2)
    data[f"Short_Moving"] = ta.ema(data["Close"], min_value)
    data["Long_Moving"] = ta.ema(data["Close"], max_value)
    return data

def determine_trend_strength(df):
    df['Trend'] = ''
    df['Strength'] = ''

    for i in range(len(df.index)):
        adx = df.loc[df.index[i], 'ADX_14']
        di_plus = df.loc[df.index[i], 'DMP_14']
        di_minus = df.loc[df.index[i], 'DMN_14']

        if di_plus > di_minus:
            df.loc[df.index[i], 'Trend'] = 'Upward'
        elif di_minus > di_plus:
            df.loc[df.index[i], 'Trend'] = 'Downward'
        else:
            df.loc[df.index[i], 'Trend'] = 'Sideways'

        if adx < 20:
            df.loc[df.index[i], 'Strength'] = 'Weak'
        elif 20 <= adx < 40:
            df.loc[df.index[i], 'Strength'] = 'Moderate'
        elif 40 <= adx < 60:
            df.loc[df.index[i], 'Strength'] = 'Strong'
        else:
            df.loc[df.index[i], 'Strength'] = 'Very Strong'

    return df

def calculate_technical_indicators(data):
    # Short-term Moving Averages
    data['Short_Moving'] = ta.ema(data['Close'], 10)
    data['Long_Moving'] = ta.ema(data['Close'], 109)

    # MACD
    macd = ta.macd(close=data['Close'], fast=5, slow=15, signal=3)
    data = pd.concat([data, macd], axis=1).reindex(data.index)
    data.rename(columns={'MACD_5_15_3': 'MACD',
                         "MACDh_5_15_3": "Histogram",
                         "MACDs_5_15_3": "Signal_Line"}, inplace=True)

    # Bollinger Bands
    bb = ta.bbands(data['Close'], length=8, std=1.5)
    bb.rename(columns={ "BBL_8_2_2": "Lower Band",
                        "BBU_8_2_2": "Upper Band",
                        "BBM_8_2_2": "Middle Band",
                        "BBB_8_2_2": "Band Width"}, inplace=True)
    bb.drop(["BBP_8_2_2"], inplace=True, axis=1)
    data = pd.concat([data, bb], axis=1).reindex(data.index)

    # Stochastic Oscillator
    stoch_df = ta.stoch(data['High'], data['Low'], data['Close'], k=8, d=2, smooth_k=2)
    stoch_df.rename(columns={"STOCHk_8_2_2": "k", "STOCHd_8_2_2": "d"}, inplace=True)
    data = pd.concat([data, stoch_df], axis=1).reindex(data.index)

    # RSI
    data["rsi"] = ta.rsi(data["Close"], window=7)
    data['ATR'] = ta.atr(high=data['High'], low=data['Low'], close=data['Close'], length=7)
    data.dropna(inplace=True)
    return data