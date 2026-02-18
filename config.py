# Alpaca API Configuration
import os

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

# Trading Parameters
INITIAL_CASH = 17495
CASH_AT_RISK = 0.1
MAX_DAY_TRADES = 3
BAD_TRADES_LIMIT = 3

# Model Training Symbols
MODEL_TRAINING_SYMBOLS = ["MRNA", "KHC", "MRK", "NVDA", "AAPL", "TSM", "GE", "XOM", "GNRC", "EBAY"]

# Stock Selection List
STOCK_POOL = ['IREN', 'LITE', 'GLW', 'NCLH', 'ANET', 'GM']
ACTIVE_STOCKS = ['GLW', 'ANET', 'AKAM']

# Initial Cash Allocation
INITIAL_STOCK_CASH = {
    'GLW': {'cash': 85, 'on_hold': False, 'hold_set_date': None},
    'ANET': {'cash': 30, 'on_hold': False, 'hold_set_date': None},
    'AKAM': {'cash': 13, 'on_hold': False, 'hold_set_date': None}
}