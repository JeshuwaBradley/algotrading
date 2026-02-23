
from dataclasses import dataclass
from typing import List, Optional
import os

@dataclass
class Config:
    # API 
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    BASE_URL: str = "https://paper-api.alpaca.markets/v2"
    
    # Trading Parameters
    INITIAL_CASH: float = 17495
    CASH_AT_RISK: float = 0.1
    TRAILING_DISTANCE: float = 0.5
    
    # Data Parameters
    START_DATE: str = "2023-01-01"
    DATA_PERIOD_DAYS: int = 59
    INTERVAL: str = "15m"
    
    # Strategy Parameters
    ICHIMOKU_COLS: List[str] = None
    STRATEGY_COLS: List[str] = None
    
    # Model Parameters
    RANDOM_FOREST_ESTIMATORS: int = 500
    XGB_ESTIMATORS: int = 800
    TEST_SIZE: float = 0.2
    
    # Persistence Parameters
    MODEL_DIR: str = "models"
    PORTFOLIO_STATE_FILE: str = "portfolio_state.json"
    LAST_UPDATE_FILE: str = "last_update.txt"
    MODEL_UPDATE_DAYS: int = 7  # Update models every 7 days
    
    def __post_init__(self):
        if self.ICHIMOKU_COLS is None:
            self.ICHIMOKU_COLS = ["span_a", "span_b", "Tenkan San 9", "Kijun San 26", "26 day lag"]
        if self.STRATEGY_COLS is None:
            self.STRATEGY_COLS = ["Moving_Strategy", "bb_macd_strategy", "stoc_signal"]
        
        # Create model directory if it doesn't exist
        os.makedirs(self.MODEL_DIR, exist_ok=True)

config = Config()