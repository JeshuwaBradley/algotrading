#!/usr/bin/env python3
"""
Algorithmic Trading System with ML Model Persistence and Portfolio Database
Original strategy by JeshuwaBradley - Modified with persistence features
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import logging
import time
from pathlib import Path

# Import your existing modules
from indicators import calculate_indicators  # Adjust import based on your actual structure
from trading.strategy import TradingStrategy  # Adjust based on your actual structure
from scripts.data_fetcher import get_market_data  # Adjust based on your actual structure

# New persistence modules
from utils.model_storage import ModelStorage
from utils.portfolio_db import PortfolioDB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlgoTradingBot:
    """Main trading bot class with persistence"""
    
    def __init__(self, config_file='config.py'):
        """Initialize the trading bot with persistence layers"""
        logger.info("Initializing AlgoTradingBot with persistence...")
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Initialize persistence
        self.model_storage = ModelStorage(model_dir="saved_models")
        self.portfolio_db = PortfolioDB(db_path="portfolio.db")
        
        # Initialize trading components
        self.strategy = TradingStrategy(self.config)
        
        # Load or initialize ML model
        self.ml_model = self.load_or_train_model()
        
        # Load previous portfolio state if exists
        self.portfolio = self.load_or_initialize_portfolio()
        
        # Track today's trades for end-of-day summary
        self.today_trades = []
        self.today_start_value = self.portfolio['total_value']
        self.current_date = date.today()
        
        logger.info("Initialization complete")
    
    def load_config(self, config_file):
        """Load configuration from file"""
        try:
            # This assumes config.py exists with your settings
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_file)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config or raise
            raise
    
    def load_or_train_model(self):
        """Load existing ML model or train new one"""
        model_name = self.config.MODEL_NAME if hasattr(self.config, 'MODEL_NAME') else "trading_model"
        
        # Try to load existing model
        model = self.model_storage.load_latest_model(model_name)
        
        if model is None:
            logger.info("No existing model found. Training new model...")
            
            # ===== YOUR EXISTING MODEL TRAINING CODE HERE =====
            # Replace this with your actual training logic
            model = self.train_new_model()
            # ==================================================
            
            # Save the trained model with metadata
            metadata = {
                'training_date': str(datetime.now()),
                'training_data_range': 'last_30_days',  # Update with your actual range
                'features_used': self.config.FEATURES if hasattr(self.config, 'FEATURES') else ['sma', 'rsi', 'macd'],
                'model_type': type(model).__name__
            }
            
            saved_path = self.model_storage.save_model(model, model_name, metadata)
            logger.info(f"New model trained and saved to {saved_path}")
        else:
            logger.info("Loaded existing model from storage")
        
        return model
    
    def train_new_model(self):
        """
        YOUR EXISTING MODEL TRAINING LOGIC
        Replace this with whatever training code you currently have
        """
        logger.info("Executing model training...")
        
        # Example - replace with your actual training code
        # from sklearn.ensemble import RandomForestClassifier
        # model = RandomForestClassifier()
        # X_train, y_train = self.get_training_data()
        # model.fit(X_train, y_train)
        
        # Placeholder - return a dummy model
        # You MUST replace this with your actual model training
        model = None
        logger.warning("PLACEHOLDER: You need to replace train_new_model() with your actual training logic!")
        
        return model
    
    def load_or_initialize_portfolio(self):
        """Load previous portfolio state or initialize new one"""
        latest = self.portfolio_db.get_latest_portfolio()
        
        if latest:
            logger.info(f"Loaded previous portfolio state from {latest['timestamp']}")
            logger.info(f"Total Value: ${latest['total_value']:.2f}, Cash: ${latest['cash']:.2f}")
            
            portfolio = {
                'total_value': latest['total_value'],
                'cash': latest['cash'],
                'positions': latest['positions']
            }
        else:
            logger.info("No previous portfolio found. Initializing new portfolio...")
            portfolio = {
                'total_value': self.config.INITIAL_CAPITAL if hasattr(self.config, 'INITIAL_CAPITAL') else 10000.0,
                'cash': self.config.INITIAL_CAPITAL if hasattr(self.config, 'INITIAL_CAPITAL') else 10000.0,
                'positions': {}
            }
            
            # Save initial state
            self.portfolio_db.save_portfolio_snapshot(portfolio)
        
        return portfolio
    
    def update_portfolio(self, trades=None):
        """
        Update portfolio with current values and save to database
        trades: list of trade dicts executed in this update
        """
        # Calculate current portfolio value based on positions and current prices
        # ===== YOUR EXISTING PORTFOLIO VALUATION LOGIC HERE =====
        # This should update self.portfolio['total_value'] and self.portfolio['positions']
        # based on current market prices
        self.calculate_portfolio_value()
        # =======================================================
        
        # Calculate returns if we have previous snapshot
        previous = self.portfolio_db.get_latest_portfolio()
        returns = {}
        
        if previous:
            returns['daily'] = (self.portfolio['total_value'] - previous['total_value']) / previous['total_value']
            
            # Calculate cumulative return from first snapshot
            history = self.portfolio_db.get_portfolio_history()
            if not history.empty:
                first_value = history.iloc[0]['total_value']
                returns['cumulative'] = (self.portfolio['total_value'] - first_value) / first_value
        
        # Save snapshot
        snapshot_id = self.portfolio_db.save_portfolio_snapshot(self.portfolio, returns)
        logger.debug(f"Portfolio snapshot saved: ${self.portfolio['total_value']:.2f}")
        
        # Save individual trades if provided
        if trades:
            for trade in trades:
                trade['timestamp'] = datetime.now()
                self.portfolio_db.save_trade(trade, snapshot_id)
                self.today_trades.append(trade)
                logger.info(f"Trade recorded: {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
        
        # Check if we need to save end-of-day summary
        self.check_end_of_day()
        
        return snapshot_id
    
    def calculate_portfolio_value(self):
        """
        YOUR EXISTING PORTFOLIO VALUATION LOGIC
        This should update self.portfolio['total_value'] based on current positions and prices
        """
        # Example - replace with your actual valuation code
        # total = self.portfolio['cash']
        # for symbol, position in self.portfolio['positions'].items():
        #     current_price = self.get_current_price(symbol)
        #     total += position * current_price
        # self.portfolio['total_value'] = total
        pass
    
    def check_end_of_day(self):
        """Check if trading day is over and save daily summary"""
        now = datetime.now()
        today = date.today()
        
        # Check if date changed (new day)
        if today != self.current_date:
            # Save previous day's summary
            prev_date = self.current_date.isoformat()
            
            self.portfolio_db.save_daily_summary(
                date=prev_date,
                start_value=self.today_start_value,
                end_value=self.portfolio['total_value'],
                trades_count=len(self.today_trades),
                metadata={'notes': 'End of day snapshot'}
            )
            
            logger.info(f"End of day summary saved for {prev_date}")
            logger.info(f"Day return: {(self.portfolio['total_value'] - self.today_start_value) / self.today_start_value:.2%}")
            
            # Reset for new day
            self.current_date = today
            self.today_start_value = self.portfolio['total_value']
            self.today_trades = []
    
    def get_trading_signals(self, market_data):
        """
        Generate trading signals using ML model and indicators
        """
        # ===== YOUR EXISTING SIGNAL GENERATION LOGIC HERE =====
        # This is where you use your ML model to generate signals
        
        # Example structure:
        # features = self.prepare_features(market_data)
        # predictions = self.ml_model.predict(features)
        # signals = self.strategy.generate_signals(market_data, predictions)
        
        signals = []  # Replace with actual signal generation
        # =======================================================
        
        return signals
    
    def execute_trades(self, signals):
        """
        Execute trades based on signals and update portfolio
        """
        executed_trades = []
        
        for signal in signals:
            # ===== YOUR EXISTING TRADE EXECUTION LOGIC HERE =====
            # Execute the trade and create trade record
            
            # Example trade record:
            trade = {
                'symbol': signal['symbol'],
                'action': signal['action'],  # 'BUY' or 'SELL'
                'quantity': signal['quantity'],
                'price': signal['price']
            }
            
            # Update portfolio positions (your existing logic)
            # if trade['action'] == 'BUY':
            #     self.portfolio['cash'] -= trade['quantity'] * trade['price']
            #     self.portfolio['positions'][trade['symbol']] = 
            #         self.portfolio['positions'].get(trade['symbol'], 0) + trade['quantity']
            # else:  # SELL
            #     self.portfolio['cash'] += trade['quantity'] * trade['price']
            #     self.portfolio['positions'][trade['symbol']] -= trade['quantity']
            
            executed_trades.append(trade)
            # =======================================================
        
        return executed_trades
    
    def prepare_features(self, market_data):
        """
        Prepare features for ML model
        """
        # ===== YOUR EXISTING FEATURE ENGINEERING LOGIC HERE =====
        # Calculate indicators and prepare feature matrix
        
        # Example:
        # df = market_data.copy()
        # df = calculate_indicators(df)  # From your indicators.py
        # features = df[self.config.FEATURES].values
        
        features = None  # Replace with actual features
        # =======================================================
        
        return features
    
    def run_iteration(self):
        """
        Run one trading iteration (e.g., for current time period)
        """
        logger.info("Running trading iteration...")
        
        try:
            # Get market data
            market_data = get_market_data(self.config.SYMBOLS)
            
            # Generate trading signals
            signals = self.get_trading_signals(market_data)
            
            # Execute trades if any signals
            if signals:
                executed_trades = self.execute_trades(signals)
                
                # Update portfolio with trades
                self.update_portfolio(executed_trades)
                
                logger.info(f"Executed {len(executed_trades)} trades")
            else:
                # Still update portfolio value (for price changes)
                self.update_portfolio()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")
            return False
    
    def analyze_performance(self):
        """
        Analyze and display portfolio performance
        """
        logger.info("Analyzing portfolio performance...")
        
        # Get all history
        history = self.portfolio_db.get_portfolio_history()
        
        if history.empty:
            logger.warning("No historical data available")
            return
        
        # Calculate key metrics
        initial_value = history.iloc[0]['total_value']
        current_value = history.iloc[-1]['total_value']
        total_return = (current_value - initial_value) / initial_value
        
        # Calculate daily returns if available
        if 'returns_daily' in history.columns and not history['returns_daily'].isna().all():
            daily_returns = history['returns_daily'].dropna()
            if len(daily_returns) > 1:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Get trade count
        with sqlite3.connect(self.portfolio_db.db_path) as conn:
            trade_count = pd.read_sql_query("SELECT COUNT(*) as count FROM trades", conn).iloc[0]['count']
        
        logger.info("=" * 50)
        logger.info("PORTFOLIO PERFORMANCE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Initial Value: ${initial_value:.2f}")
        logger.info(f"Current Value: ${current_value:.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Total Trades: {trade_count}")
        logger.info(f"Trading Days: {len(history)}")
        logger.info("=" * 50)
    
    def run(self, continuous=True, interval_seconds=60):
        """
        Main run loop
        
        Args:
            continuous: If True, run continuously; if False, run once
            interval_seconds: Time between iterations when continuous
        """
        logger.info("Starting trading bot...")
        
        try:
            if continuous:
                logger.info(f"Running in continuous mode (interval: {interval_seconds}s)")
                while True:
                    self.run_iteration()
                    time.sleep(interval_seconds)
            else:
                logger.info("Running single iteration")
                self.run_iteration()
                
            # Final performance analysis
            self.analyze_performance()
            
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
            # Save final state
            self.update_portfolio()
            self.check_end_of_day()  # Force end-of-day save
            logger.info("Final portfolio state saved")
            self.analyze_performance()
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            # Try to save state on error
            try:
                self.update_portfolio()
                logger.info("Portfolio state saved before exit")
            except:
                pass
            raise

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Algorithmic Trading Bot')
    parser.add_argument('--config', type=str, default='config.py',
                        help='Configuration file path')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit')
    parser.add_argument('--interval', type=int, default=60,
                        help='Interval between iterations in seconds')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze existing portfolio data and exit')
    
    args = parser.parse_args()
    
    if args.analyze:
        # Just analyze existing data without running
        db = PortfolioDB()
        history = db.get_portfolio_history()
        if not history.empty:
            print("\nPortfolio History:")
            print(history[['timestamp', 'total_value', 'cash']].to_string())
            
            # Quick analysis
            initial = history.iloc[0]['total_value']
            final = history.iloc[-1]['total_value']
            print(f"\nTotal Return: {(final - initial) / initial:.2%}")
        else:
            print("No portfolio data found")
        return
    
    # Create and run bot
    bot = AlgoTradingBot(args.config)
    bot.run(continuous=not args.once, interval_seconds=args.interval)

if __name__ == "__main__":
    main()