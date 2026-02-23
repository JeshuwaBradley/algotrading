import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from typing import Tuple, List
from .indicators import TechnicalIndicators
from .data_fetcher import DataFetcher
from config import config


class ModelTrainer:
    """Trains and manages ML models for trading"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.data_fetcher = DataFetcher()
    
    def tomorrow(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for prediction"""
        data["target"] = data["Close"].shift(-5)
        data["Tomorrow"] = (data["target"] > data["Close"]).astype(int)
        data.drop("target", axis=1, inplace=True)
        return data
    
    def download_data(self, ticker: str, period: int) -> pd.DataFrame:
        """Download and prepare data for model training"""
        data = self.data_fetcher.download_intraday_data(ticker, period)
        data = self.tomorrow(data)
        data = self.indicators.calculate_technical_indicators(data)
        data.dropna(inplace=True)
        return data
    
    def moving_average_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate moving average strategy signals"""
        data["Moving_Strategy"] = 0
        buy_condition = (data["Short_Moving"] >= data["Long_Moving"]) & (data["Short_Moving"].shift(1) <= data["Long_Moving"].shift(1))
        sell_condition = (data["Short_Moving"] <= data["Long_Moving"]) & (data["Short_Moving"].shift(1) >= data["Long_Moving"].shift(1))
        data.loc[buy_condition, "Moving_Strategy"] = 1
        data.loc[sell_condition, "Moving_Strategy"] = -1
        return data
    
    def bb_macd_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate BB-MACD strategy signals"""
        data['bb_macd_strategy'] = 0
        data.loc[(data['Close'] > data['Upper Band']) & (data['MACD'] > data['Signal_Line']), 'bb_macd_strategy'] = 1
        data.loc[(data['Close'] < data['Lower Band']) & (data['MACD'] < data['Signal_Line']), 'bb_macd_strategy'] = -1
        return data
    
    def implement_so_strategy(self, df: pd.DataFrame, overbought_level: int = 90, oversold_level: int = 15) -> pd.DataFrame:
        """Generate stochastic oscillator strategy signals"""
        buy_condition = ((df["k"] < oversold_level) & (df["k"].shift(1) > oversold_level)) | (df["k"].shift(1) > df["d"].shift(1))
        sell_condition = ((df["k"] > overbought_level) & (df["k"].shift(1) < overbought_level)) | (df["k"].shift(1) < df["d"].shift(1))
        
        df["stoc_signal"] = 0
        df.loc[buy_condition, "stoc_signal"] = 1
        df.loc[sell_condition, "stoc_signal"] = -1
        return df
    
    def most_frequent(self, lst: List[int]) -> int:
        """Aggregate strategy signals"""
        total = sum(lst)
        if total >= 1:
            return 2  # buy
        elif total <= -1:
            return 1  # sell
        else:
            return 0  # hold
    
    def final_vote(self, ticker: str, period: int, reset_index: bool = False) -> pd.DataFrame:
        """Combine multiple strategies into final signal"""
        data = self.prepare_data(ticker, period)
        data["Final"] = data.apply(lambda row: self.most_frequent(row[config.STRATEGY_COLS]), axis=1)
        data.drop(config.STRATEGY_COLS, axis=1, inplace=True)
        if reset_index:
            data.reset_index(inplace=True)
        data.dropna(inplace=True)
        return data
    
    def prepare_data(self, ticker: str, period: int) -> pd.DataFrame:
        """Prepare data with all strategies"""
        data = self.download_data(ticker, period)
        data = self.moving_average_strategy(data)
        data = self.implement_so_strategy(data)
        data = self.bb_macd_strategy(data)
        return data
    
    def combining_data(self, lst: List[str], period: int = 59) -> pd.DataFrame:
        """Combine data from multiple tickers"""
        dataframes = []
        for ticker in lst:
            dataframes.append(self.final_vote(ticker, period))
            import time
            time.sleep(1)
        result = pd.concat(dataframes)
        return result.sample(frac=1, random_state=42)
    
    def train_random_forest_model(self, lst: List[str]) -> RandomForestClassifier:
        """Train Random Forest model"""
        data = self.combining_data(lst)
        X = data.loc[:, data.columns != "Tomorrow"]
        y = data["Tomorrow"]
        X = X.drop(["Lower Band", "Upper Band", "Middle Band", "Band Width", "k", "d", "rsi", "ATR", "Final"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        
        model = RandomForestClassifier(n_estimators=config.RANDOM_FOREST_ESTIMATORS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"Testing model complete! Accuracy: {accuracy}")
        return model
    
    def train_xgb_model(self, lst: List[str]) -> XGBClassifier:
        """Train XGBoost model"""
        data = self.combining_data(lst)
        X = data.loc[:, data.columns != "Final"]
        y = data["Final"]
        X = X.drop(["ATR", "Tomorrow"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)
        
        model = XGBClassifier(n_estimators=config.XGB_ESTIMATORS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"Testing model complete! Accuracy: {accuracy}")
        return model
        
    def save_models(self, buy_sell_model, up_down_model, suffix: str = ""):
        """Save both models"""
        from persistence import ModelPersistence
        
        if suffix:
            suffix = f"_{suffix}"
        
        ModelPersistence.save_xgb_model(buy_sell_model, f"buy_sell_model{suffix}")
        ModelPersistence.save_rf_model(up_down_model, f"up_down_model{suffix}")
        print(f"Models saved with suffix: {suffix}")

    def load_models(self, suffix: str = "") -> Tuple[Optional[XGBClassifier], Optional[RandomForestClassifier]]:
        """Load both models"""
        from persistence import ModelPersistence
        
        if suffix:
            suffix = f"_{suffix}"
        
        buy_sell_model = ModelPersistence.load_xgb_model(f"buy_sell_model{suffix}")
        up_down_model = ModelPersistence.load_rf_model(f"up_down_model{suffix}")
        
        return buy_sell_model, up_down_model