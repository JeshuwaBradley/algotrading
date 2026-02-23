#!/usr/bin/env python3
"""
Algorithmic Trading System with ML Model Persistence and Portfolio Database
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import logging
import time
import sqlite3
from pathlib import Path

# Your actual imports based on the repository structure
from indicators import calculate_technical_indicators, ichimoku, calculate_moving_average, determine_trend_strength
from config import *
from scripts.data_fetcher import get_market_data
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
    
    def __init__(self):
        """Initialize the trading bot"""
        logger.info("Initializing AlgoTradingBot...")
        
        # Initialize persistence
        self.model_storage = ModelStorage(model_dir="saved_models")
        self.portfolio_db = PortfolioDB(db_path="portfolio.db")
        
        # Load or initialize ML model
        self.ml_model = self.load_or_train_model()
        
        # Load previous portfolio state if exists
        self.portfolio = self.load_or_initialize_portfolio()
        
        # Track today's trades for end-of-day summary
        self.today_trades = []
        self.today_start_value = self.portfolio['total_value']
        self.current_date = date.today()
        
        logger.info("Initialization complete")
    
    def load_or_train_model(self):
        """Load existing ML model or train new one"""
        model_name = "trading_model"
        
        # Try to load existing model
        model = self.model_storage.load_latest_model(model_name)
        
        if model is None:
            logger.info("No existing model found. Training new model...")
            
            # YOUR MODEL TRAINING CODE HERE
            # Replace with your actual model training logic
            model = self.train_new_model()
            
            # Save the trained model
            metadata = {
                'training_date': str(datetime.now()),
                'model_type': type(model).__name__ if model else 'unknown'
            }
            
            self.model_storage.save_model(model, model_name, metadata)
            logger.info("New model trained and saved")
        else:
            logger.info("Loaded existing model from storage")
        
        return model
    
    def train_new_model(self):
        """
        YOUR EXISTING MODEL TRAINING LOGIC
        Replace this with whatever training code you currently have
        """
        # Example placeholder - replace with your actual code
        logger.warning("Using placeholder model - REPLACE WITH YOUR ACTUAL MODEL")
        return None
    
    def load_or_initialize_portfolio(self):
        """Load previous portfolio state or initialize new one"""
        latest = self.portfolio_db.get_latest_portfolio()
        
        if latest:
            logger.info(f"Loaded previous portfolio from {latest['timestamp']}")
            portfolio = {
                'total_value': latest['total_value'],
                'cash': latest['cash'],
                'positions': latest['positions']
            }
        else:
            logger.info("Initializing new portfolio")
            portfolio = {
                'total_value': INITIAL_CAPITAL,
                'cash': INITIAL_CAPITAL,
                'positions': {}
            }
            self.portfolio_db.save_portfolio_snapshot(portfolio)
        
        return portfolio
    
    def prepare_features(self, market_data):
        """Prepare features for ML model using technical indicators"""
        # Calculate technical indicators
        df = calculate_technical_indicators(market_data)
        
        # Add moving averages
        df = calculate_moving_average(df, 20, 50)
        
        # Add trend strength
        df = determine_trend_strength(df)
        
        # Select features for model
        feature_columns = ['close', 'volume', 'rsi', 'macd', 'sma_20', 'sma_50', 'trend_strength']
        features = df[feature_columns].values
        
        return features
    
    def get_trading_signals(self, market_data):
        """Generate trading signals using ML model and indicators"""
        signals = []
        
        # Prepare features
        features = self.prepare_features(market_data)
        
        # YOUR SIGNAL GENERATION LOGIC HERE
        # Use self.ml_model to predict and generate signals
        
        return signals
    
    def execute_trades(self, signals):
        """Execute trades and update portfolio"""
        executed_trades = []
        
        for signal in signals:
            # YOUR TRADE EXECUTION LOGIC HERE
            # Update self.portfolio['cash'] and self.portfolio['positions']
            
            trade = {
                'symbol': signal.get('symbol', 'UNKNOWN'),
                'action': signal.get('action', 'BUY'),
                'quantity': signal.get('quantity', 0),
                'price': signal.get('price', 0)
            }
            executed_trades.append(trade)
        
        return executed_trades
    
    def calculate_portfolio_value(self, market_data):
        """Calculate current portfolio value based on positions and current prices"""
        total = self.portfolio['cash']
        
        for symbol, position in self.portfolio['positions'].items():
            if position > 0 and symbol in market_data:
                current_price = market_data[symbol]['close'].iloc[-1]
                total += position * current_price
        
        self.portfolio['total_value'] = total
        return total
    
    def update_portfolio(self, market_data, trades=None):
        """Update portfolio with current values and save to database"""
        # Update portfolio value based on current prices
        self.calculate_portfolio_value(market_data)
        
        # Calculate returns
        previous = self.portfolio_db.get_latest_portfolio()
        returns = {}
        
        if previous:
            returns['daily'] = (self.portfolio['total_value'] - previous['total_value']) / previous['total_value']
        
        # Save snapshot
        snapshot_id = self.portfolio_db.save_portfolio_snapshot(self.portfolio, returns)
        
        # Save trades if any
        if trades:
            for trade in trades:
                trade['timestamp'] = datetime.now()
                self.portfolio_db.save_trade(trade, snapshot_id)
                self.today_trades.append(trade)
                logger.info(f"Trade: {trade['action']} {trade['quantity']} {trade['symbol']}")
        
        # Check for end of day
        self.check_end_of_day()
        
        return snapshot_id
    
    def check_end_of_day(self):
        """Check if trading day is over and save daily summary"""
        today = date.today()
        
        if today != self.current_date:
            # Save previous day's summary
            self.portfolio_db.save_daily_summary(
                date=self.current_date.isoformat(),
                start_value=self.today_start_value,
                end_value=self.portfolio['total_value'],
                trades_count=len(self.today_trades)
            )
            
            # Reset for new day
            self.current_date = today
            self.today_start_value = self.portfolio['total_value']
            self.today_trades = []
            
            logger.info(f"End of day summary saved for {self.current_date}")
    
    def run_iteration(self):
        """Run one trading iteration"""
        try:
            # Get market data for all symbols
            market_data = {}
            for symbol in SYMBOLS:
                market_data[symbol] = get_market_data(symbol)
            
            # Generate signals
            signals = self.get_trading_signals(market_data)
            
            # Execute trades if any signals
            if signals:
                trades = self.execute_trades(signals)
                self.update_portfolio(market_data, trades)
                logger.info(f"Executed {len(trades)} trades")
            else:
                # Just update portfolio value
                self.update_portfolio(market_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")
            return False
    
    def analyze_performance(self):
        """Analyze and display portfolio performance"""
        history = self.portfolio_db.get_portfolio_history()
        
        if history.empty:
            logger.warning("No historical data available")
            return
        
        initial = history.iloc[0]['total_value']
        final = history.iloc[-1]['total_value']
        total_return = (final - initial) / initial
        
        logger.info("=" * 50)
        logger.info("PORTFOLIO PERFORMANCE")
        logger.info("=" * 50)
        logger.info(f"Initial Value: ${initial:.2f}")
        logger.info(f"Current Value: ${final:.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Trading Days: {len(history)}")
        logger.info("=" * 50)
    
    def run(self, continuous=True, interval_seconds=60):
        """Main run loop"""
        logger.info("Starting trading bot...")
        
        try:
            if continuous:
                while True:
                    self.run_iteration()
                    time.sleep(interval_seconds)
            else:
                self.run_iteration()
            
            self.analyze_performance()
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.analyze_performance()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Algorithmic Trading Bot')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=60, help='Interval between iterations')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing data')
    
    args = parser.parse_args()
    
    if args.analyze:
        db = PortfolioDB()
        history = db.get_portfolio_history()
        if not history.empty:
            print("\nPortfolio History:")
            print(history[['timestamp', 'total_value', 'cash']].tail(10).to_string())
            
            initial = history.iloc[0]['total_value']
            final = history.iloc[-1]['total_value']
            print(f"\nTotal Return: {(final - initial) / initial:.2%}")
        return
    
    # Create and run bot
    bot = AlgoTradingBot()
    bot.run(continuous=not args.once, interval_seconds=args.interval)

if __name__ == "__main__":
    main()