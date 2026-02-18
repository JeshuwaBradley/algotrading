#!/usr/bin/env python3
"""
Trading Bot Main Entry Point
Run on Google VM with: python main.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import datetime
import time
import pytz
import pandas as pd
from alpaca.trading.client import TradingClient

import config
from trading import (
    Order, Portfolio, get_current_stock_price, download_data,
    train_RandomForest_model, train_XGB_model, final_vote
)
from utils.helpers import buy, sell_market, is_market_open, return_time, calculate_position_size

def initialize_trading_client():
    """Initialize Alpaca trading client"""
    return TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)

def initialize_portfolio(trading_client):
    """Initialize portfolio with existing positions"""
    orders = []
    for position in trading_client.get_all_positions():
        print(f"Processing position: {position.symbol}")
        symbol = position.symbol
        trailing_distance = 0.5
        entry_price = float(position.avg_entry_price)
        quantity = int(position.qty_available)
        current_price, average_fluc = get_current_stock_price(symbol)
        
        if current_price is None:
            continue
            
        unrealized_pl = (current_price - entry_price) * quantity
        potential_stop_price = current_price - (current_price * trailing_distance / 100)
        
        if unrealized_pl > 100:
            old_stop_price = entry_price - average_fluc if average_fluc else entry_price * 0.95
            stop_price = max(old_stop_price, potential_stop_price)
        else:
            stop_price = entry_price - 150/quantity
            
        print(f"Symbol: {symbol}, quantity:{quantity}, entry_price:{entry_price}, stop_price: {round(stop_price, 2)}, current_price: {round(current_price, 2)} unrealized_pl: {round(unrealized_pl, 2)}")
        order = Order(symbol, quantity, entry_price, trailing_distance, stop_price)
        orders.append(order)
    
    portfolio = Portfolio(
        trading_client=trading_client,
        orders=orders,
        initial_cash=config.INITIAL_CASH,
        cash_at_risk=config.CASH_AT_RISK,
        buy_func=lambda stock, quantity, limit_price: buy(trading_client, stock, quantity, limit_price),
        sell_func=lambda stock, quantity: sell_market(trading_client, stock, quantity)
    )
    
    return portfolio

def get_previous_day_data(ticker, buy_sell_drop, up_down_drop, buy_sell_model, up_down_model):
    """Get previous day data and predictions for a ticker"""
    data = final_vote(ticker, 10)
    previous_row = data.iloc[-1:]

    buy_sell_X = previous_row.drop(buy_sell_drop, axis=1)
    up_down_X = previous_row.drop(up_down_drop, axis=1)

    buy_sell_pred = buy_sell_model.predict(buy_sell_X)
    up_down_pred = up_down_model.predict(up_down_X)

    current_price, average_fluc = get_current_stock_price(ticker)
    close_price = previous_row["Close"].iloc[0] if "Close" in previous_row.columns else current_price
    
    print(f"{buy_sell_pred}, {up_down_pred}, {ticker}")
    return buy_sell_pred, up_down_pred, previous_row, close_price, average_fluc

def trading_decision_maker(symbol, price, average_fluc, portfolio, buy_sell_pred, up_down_pred, atr, cash):
    """Make trading decision for a symbol"""
    if buy_sell_pred == 2 and portfolio.cash_allocated_per_stock[symbol]["cash"] > float(price):
        quantity = calculate_position_size(
            portfolio.cash_allocated_per_stock[symbol]["cash"], 
            portfolio.cash_at_risk, 
            atr, 
            price
        )
        return portfolio.execute_trade(symbol, quantity, price, average_fluc, cash)
    return False

def make_trades(portfolio):
    """Execute trading logic for all active stocks"""
    tickers = portfolio.stocks
    cash_available = portfolio.trading_client.get_account().cash

    buy_sell_model = portfolio.buy_sell_model
    up_down_model = portfolio.up_down_model

    portfolio.check_on_hold()
    portfolio.check_pending_buys()

    buy_sell_drop = ["ATR", "Tomorrow", "Final"]
    up_down_drop = ["Lower Band", "Upper Band", "Middle Band", "Band Width", "k", "d", "rsi", "ATR", "Tomorrow", "Final"]

    # Store predictions and data for each ticker
    ticker_data = {}
    old_prices = {}

    for symbol in tickers:
        try:
            buy_sell_pred, up_down_pred, stock_data, current_price, average_fluc = get_previous_day_data(
                ticker=symbol, 
                buy_sell_drop=buy_sell_drop, 
                up_down_drop=up_down_drop, 
                buy_sell_model=buy_sell_model, 
                up_down_model=up_down_model
            )
            
            ticker_data[symbol] = {
                'buy_sell_pred': buy_sell_pred,
                'up_down_pred': up_down_pred,
                'stock_data': stock_data,
                'current_price': current_price,
                'average_fluc': average_fluc,
                'close': stock_data["Close"].iloc[0] if "Close" in stock_data.columns else current_price,
                'atr': stock_data['ATR'] if 'ATR' in stock_data.columns else None
            }
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            ticker_data[symbol] = None

    # Execute trades
    for symbol in tickers:
        if ticker_data[symbol] is None:
            continue
            
        data = ticker_data[symbol]
        trading_decision_maker(
            symbol, 
            data['close'], 
            data['average_fluc'], 
            portfolio, 
            data['buy_sell_pred'], 
            data['up_down_pred'], 
            data['atr'], 
            cash_available
        )

    # Get old stop prices for orders
    for order in portfolio.orders:
        if order.symbol in tickers:
            old_prices[order.symbol] = order.stop_price

    # Update orders
    current_prices = {symbol: ticker_data[symbol]['close'] for symbol in tickers if ticker_data[symbol] is not None}
    buy_sell_preds = {symbol: ticker_data[symbol]['buy_sell_pred'] for symbol in tickers if ticker_data[symbol] is not None}
    up_down_preds = {symbol: ticker_data[symbol]['up_down_pred'] for symbol in tickers if ticker_data[symbol] is not None}

    portfolio.update_orders(current_prices, buy_sell_preds, up_down_preds, old_prices)

def print_portfolio_status(portfolio):
    """Print current portfolio status"""
    account = portfolio.trading_client.get_account()
    print(f"Total assets: {account.equity}, Remaining cash: {account.cash}, "
          f"Buying power: {account.buying_power}, daily change: {round(float(account.equity)-float(account.last_equity), 2)}, "
          f"day trade count: {account.daytrade_count}")
    
    for position in portfolio.trading_client.get_all_positions():
        order = next((x for x in portfolio.orders if x.symbol == position.symbol), None)
        current_price, _ = get_current_stock_price(position.symbol)
        if order:
            print(f"Symbol: {position.symbol}, Entry Price: {position.avg_entry_price}, "
                  f"Quantity: {position.qty_available}, Current Price: {current_price}, "
                  f"stop_price = {round(order.stop_price, 2)}, Unrealized_pl: {position.unrealized_pl}")
        else:
            print(f"Symbol: {position.symbol}, Entry Price: {position.avg_entry_price}, "
                  f"Quantity: {position.qty_available}, Current Price: {current_price}, "
                  f"Unrealized_pl: {position.unrealized_pl}")

def main():
    """Main execution function"""
    print("Initializing trading bot...")
    
    # Initialize trading client
    trading_client = initialize_trading_client()
    
    # Initialize portfolio
    portfolio = initialize_portfolio(trading_client)
    
    # Train models
    print("Training models...")
    buy_sell_model = train_XGB_model(config.MODEL_TRAINING_SYMBOLS)
    up_down_model = train_RandomForest_model(portfolio.stocks + portfolio.next_stocks)
    portfolio.set_models(buy_sell_model, up_down_model)
    
    # Trading loop
    print("Starting trading loop...")
    while is_market_open(trading_client):
        if portfolio.bad_trades >= config.BAD_TRADES_LIMIT:
            print("Too many bad trades, stopping trading")
            break
            
        make_trades(portfolio)
        print_portfolio_status(portfolio)
        
        # Wait 15 minutes before next iteration
        print(f"Waiting 15 minutes... Next check at {datetime.datetime.now() + datetime.timedelta(minutes=15)}")
        time.sleep(15 * 60)
    
    print("Market closed. Trading session ended.")
    
    # Print final status
    print("\n=== FINAL PORTFOLIO STATUS ===")
    print_portfolio_status(portfolio)
    print(f"Bad trades: {portfolio.bad_trades}")
    print(f"Day trade count: {portfolio.day_trade_count}")
    print(f"Cash allocated per stock: {portfolio.cash_allocated_per_stock}")
    print(f"Active stocks: {portfolio.stocks}")
    print(f"Next stocks: {portfolio.next_stocks}")

if __name__ == "__main__":
    main()