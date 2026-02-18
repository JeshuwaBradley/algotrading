import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from .indicators import calculate_technical_indicators
from .data_fetcher import download_data

def tomorrow(data):
    data["target"] = data["Close"].shift(-5)
    data["Tomorrow"] = (data["target"] > data["Close"]).astype(int)
    data.drop("target", axis=1, inplace=True)
    return data

def prepare_data(ticker, period):
    data = download_data(ticker, period)
    data = tomorrow(data)
    data = calculate_technical_indicators(data)
    data = moving_average_strategy(data)
    data = implement_so_strategy(data)
    data = bb_macd_strategy(data)
    data.dropna(inplace=True)
    return data

def moving_average_strategy(data):
    data["Moving_Strategy"] = 0
    buy_condition = (data["Short_Moving"] >= data["Long_Moving"]) & (data["Short_Moving"].shift(1) <= data["Long_Moving"].shift(1))
    sell_condition = (data["Short_Moving"] <= data["Long_Moving"]) & (data["Short_Moving"].shift(1) >= data["Long_Moving"].shift(1))
    data.loc[buy_condition, "Moving_Strategy"] = 1
    data.loc[sell_condition, "Moving_Strategy"] = -1
    return data

def implement_so_strategy(df, overbought_level=90, oversold_level=15):
    buy_condition = ((df["k"] < oversold_level) & (df["k"].shift(1) > oversold_level)) | (df["k"].shift(1) > df["d"].shift(1))
    sell_condition = ((df["k"] > overbought_level) & (df["k"].shift(1) < overbought_level)) | (df["k"].shift(1) < df["d"].shift(1))
    df["stoc_signal"] = 0
    df.loc[buy_condition, "stoc_signal"] = 1
    df.loc[sell_condition, "stoc_signal"] = -1
    return df

def bb_macd_strategy(data):
    data['bb_macd_strategy'] = 0
    data.loc[(data['Close'] > data['Upper Band']) & (data['MACD'] > data['Signal_Line']), 'bb_macd_strategy'] = 1
    data.loc[(data['Close'] < data['Lower Band']) & (data['MACD'] < data['Signal_Line']), 'bb_macd_strategy'] = -1
    return data

def most_frequent(lst):
    total = sum(lst)
    if total >= 1:
        total = 2
    elif total <= -1:
        total = 1
    else:
        total = 0
    return total

def final_vote(ticker, period, reset_index=False):
    data = prepare_data(ticker, period)
    columns = ["Moving_Strategy", "bb_macd_strategy", "stoc_signal"]
    data["Final"] = data.apply(lambda row: most_frequent(row[columns]), axis=1)
    data.drop(columns, axis=1, inplace=True)
    if reset_index:
        data.reset_index(inplace=True)
    data.dropna(inplace=True)
    return data

def combining_data(lst, period=59):
    dataframes = []
    for ticker in lst:
        dataframes.append(final_vote(ticker, period))
        time.sleep(1)
    result = pd.concat(dataframes)
    return result.sample(frac=1, random_state=42)

def train_RandomForest_model(lst):
    data = combining_data(lst)
    X = data.loc[:, data.columns != "Tomorrow"]
    y = data["Tomorrow"]
    X = X.drop([ "Lower Band", "Upper Band", "Middle Band", "Band Width", "k", "d", "rsi", "ATR", "Final"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    model = RandomForestClassifier(n_estimators=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Testing RandomForest model complete! Accuracy: {accuracy}")
    return model

def train_XGB_model(lst):
    data = combining_data(lst)
    X = data.loc[:, data.columns != "Final"]
    y = data["Final"]
    X = X.drop(["ATR", "Tomorrow"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)
    model = XGBClassifier(n_estimators=800)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Testing XGB model complete! Accuracy: {accuracy}")
    return model