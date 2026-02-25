import sys
import os
# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import datetime
import signal
from alpaca.trading.client import TradingClient
from config import config
from trading.data_fetcher import DataFetcher
from trading.stock_selector import StockSelector
from trading.models import ModelTrainer
from trading.portfolio import Portfolio
from trading.order import Order
from persistence import PortfolioPersistence, UpdateTracker, ModelPersistence
from utils.helpers import TradingUtils
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TradingBot:
    """Main trading bot class with persistence"""
    
    def __init__(self):
        self.trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
        self.data_fetcher = DataFetcher()
        self.stock_selector = StockSelector()
        self.model_trainer = ModelTrainer()
        self.trading_utils = TradingUtils()
        self.portfolio = None
        self.buy_sell_model = None
        self.up_down_model = None
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print("\nShutdown signal received. Saving state...")
        self.running = False
        try:
            self.save_state()
            print("State saved successfully")
        except Exception as e:
            print(f"Error saving state: {e}")
            # Try to save again with error handling
            try:
                if self.portfolio:
                    # Manual save as fallback
                    import json
                    from persistence import DateTimeEncoder
                    state = self.portfolio.get_state()
                    with open('portfolio_state.json', 'w') as f:
                        json.dump(state, f, cls=DateTimeEncoder, indent=2)
                    print("Fallback save completed")
            except:
                print("Could not save state")
        sys.exit(0)

    def save_state(self):
        """Save current state (models and portfolio)"""
        if self.portfolio:
            try:
                PortfolioPersistence.save_portfolio(self.portfolio)
                print("Portfolio state saved")
            except Exception as e:
                print(f"Error saving portfolio: {e}")
        
        if self.buy_sell_model and self.up_down_model:
            try:
                self.model_trainer.save_models(self.buy_sell_model, self.up_down_model)
                print("Models saved")
            except Exception as e:
                print(f"Error saving models: {e}")
    
    def load_state(self) -> bool:
        """Load previous state if exists"""
        # Try to load portfolio
        portfolio_state = PortfolioPersistence.load_portfolio()
        
        # Try to load models
        self.buy_sell_model, self.up_down_model = self.model_trainer.load_models()
        
        if portfolio_state and self.buy_sell_model and self.up_down_model:
            # Recreate portfolio from saved state - ADD trading_client here!
            self.portfolio = PortfolioPersistence.recreate_portfolio(
                portfolio_state,
                self.buy_order_function,
                self.sell_order_function,
                self.trading_client,  # <-- ADD THIS ARGUMENT
                self.buy_sell_model,
                self.up_down_model
            )
            print("Successfully loaded previous state")
            return True
        # ... rest of method
        elif portfolio_state:
            print("Loaded portfolio state but models not found")
        elif self.buy_sell_model and self.up_down_model:
            print("Loaded models but portfolio state not found")
        
        return False
    
    # In TradingBot class, update initialize_new_session method:

    def initialize_new_session(self):
        """Initialize a new trading session"""
        print("Starting new trading session...")
        
        # Select stocks
        selected_stocks = self.select_stocks()
        print(f"Selected stocks: {selected_stocks}")
        
        # Train or load models
        if UpdateTracker.should_update():
            print("Updating models...")
            self.buy_sell_model, self.up_down_model = self.train_models(selected_stocks)
            self.model_trainer.save_models(self.buy_sell_model, self.up_down_model)
            UpdateTracker.set_last_update()
        else:
            print("Using existing models...")
            self.buy_sell_model, self.up_down_model = self.model_trainer.load_models()
            if not self.buy_sell_model or not self.up_down_model:
                print("Models not found, training new ones...")
                self.buy_sell_model, self.up_down_model = self.train_models(selected_stocks)
                self.model_trainer.save_models(self.buy_sell_model, self.up_down_model)
        
        # Initialize portfolio - ADD trading_client here!
        self.portfolio = Portfolio(
            stocks=selected_stocks,
            initial_cash=config.INITIAL_CASH,
            cash_at_risk=config.CASH_AT_RISK,
            buy=self.buy_order_function,
            sell=self.sell_order_function,
            trading_client=self.trading_client,  # <-- ADD THIS LINE
            buy_sell_model=self.buy_sell_model,
            up_down_model=self.up_down_model
        )
    
    def select_stocks(self) -> list:
        """Select stocks for trading"""
        sp500 = DataFetcher.get_sp500_tickers()
        selected_stocks, _ = self.stock_selector.stock_selecting_code_1(sp500, str(datetime.date.today()))
        return selected_stocks[:3]
    
    def train_models(self, selected_stocks: list):
        """Train ML models"""
        buy_sell_model = self.model_trainer.train_xgb_model(selected_stocks)
        up_down_model = self.model_trainer.train_random_forest_model(selected_stocks)
        return buy_sell_model, up_down_model
    
    def get_previous_day_data(self, ticker: str):
        """Get predictions for a ticker"""
        buy_sell_drop = ["ATR", "Tomorrow", "Final"]
        up_down_drop = ["Lower Band", "Upper Band", "Middle Band", "Band Width", "k", "d", "rsi", "ATR", "Tomorrow", "Final"]
        
        data = self.model_trainer.final_vote(ticker, 10)
        previous_row = data.iloc[-1:]
        
        buy_sell_X = previous_row.drop(buy_sell_drop, axis=1)
        up_down_X = previous_row.drop(up_down_drop, axis=1)
        
        buy_sell_pred = self.buy_sell_model.predict(buy_sell_X)
        up_down_pred = self.up_down_model.predict(up_down_X)
        
        current_price, average_fluc = DataFetcher.get_current_stock_price(ticker)
        
        print(f"Predictions for {ticker}: Buy/Sell={buy_sell_pred[0]}, Up/Down={up_down_pred[0]}")
        
        return buy_sell_pred[0], up_down_pred[0], previous_row, current_price, average_fluc
    
    def trading_decision_maker(self, symbol: str, price: float, average_fluc: float, 
                              buy_sell_pred: int, up_down_pred: int, atr: float, cash: float) -> bool:
        """Make trading decisions based on predictions"""
        if buy_sell_pred == 2 and self.portfolio.cash_allocated_per_stock[symbol]["cash"] > float(price):
            quantity = TradingUtils.calculate_position_size(
                self.portfolio.cash_allocated_per_stock[symbol]["cash"], 
                self.portfolio.cash_at_risk, 
                atr, 
                price
            )
            return self.portfolio.execute_trade(symbol, quantity, price, average_fluc, cash)
        return False
    
    def make_trades(self):
        """Execute trades based on predictions"""
        tickers = self.portfolio.stocks
        cash_available = self.trading_client.get_account().cash
        
        self.portfolio.check_on_hold()
        self.portfolio.check_pending_buys()
        
        predictions = {}
        for ticker in tickers:
            buy_sell_pred, up_down_pred, stock_data, current_price, average_fluc = self.get_previous_day_data(ticker)
            
            if current_price is not None:
                predictions[ticker] = {
                    'buy_sell': buy_sell_pred,
                    'up_down': up_down_pred,
                    'price': current_price,
                    'average_fluc': average_fluc,
                    'atr': stock_data['ATR'].iloc[0] if 'ATR' in stock_data.columns else 0.5
                }
        
        # Execute trades
        for ticker, preds in predictions.items():
            self.trading_decision_maker(
                ticker, 
                preds['price'], 
                preds['average_fluc'], 
                preds['buy_sell'], 
                preds['up_down'], 
                preds['atr'], 
                cash_available
            )
        
        # Update orders
        old_prices = {}
        for order in self.portfolio.orders:
            if order.symbol in predictions:
                old_prices[order.symbol] = order.stop_price
        
        self.portfolio.update_orders(
            current_prices={t: p['price'] for t, p in predictions.items()},
            buy_sell_preds={t: p['buy_sell'] for t, p in predictions.items()},
            up_down_preds={t: p['up_down'] for t, p in predictions.items()},
            old_prices=old_prices
        )
    
    def buy_order_function(self, stock: str, quantity: int, limit_price: float):
        """Execute buy order"""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        rounded_price = round(limit_price, 2)
        limitOrder = LimitOrderRequest(
            symbol=stock,
            limit_price=rounded_price,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        
        try:
            buy_limit_order = self.trading_client.submit_order(order_data=limitOrder)
            print(f"Successfully placed limit buy order for {quantity} shares of {stock}")
        except Exception as e:
            print(f"Error placing order: {e}")
    
    def sell_order_function(self, stock: str, quantity: int):
        """Execute sell order"""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        marketOrder = MarketOrderRequest(
            symbol=stock,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        
        try:
            sell_market_order = self.trading_client.submit_order(order_data=marketOrder)
            print(f"Successfully placed market sell order for {quantity} shares of {stock}")
            return sell_market_order
        except Exception as e:
            print(f"Error placing order: {e}")
    
    def run(self):
        """Main trading loop"""
        # Try to load previous state
        if not self.load_state():
            self.initialize_new_session()
        
        # Trading loop
        bad_trades_count = self.portfolio.bad_trades if self.portfolio else 0
        
        while self.running and self.trading_utils.is_market_open(self.trading_client):
            if bad_trades_count >= 3:
                print("Too many bad trades, stopping trading")
                break
            
            self.make_trades()
            self.trading_utils.display_portfolio(self.trading_client, self.portfolio)
            
            # Save state periodically (every hour)
            if datetime.datetime.now().minute == 0:
                self.save_state()
            
            time.sleep(15 * 60)  # Wait 15 minutes
        
        print("Market closed")
        self.save_state()


def main():
    """Entry point"""
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()