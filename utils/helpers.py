import datetime
import pytz
from typing import Tuple
from alpaca.trading.client import TradingClient


class TradingUtils:
    """Utility functions for trading operations"""
    
    @staticmethod
    def is_market_open(trading_client: TradingClient) -> bool:
        """Check if market is open"""
        try:
            return trading_client.get_clock().is_open
        except:
            # Fallback to time check if API fails
            import datetime
            import pytz
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))
            # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
            if now.weekday() >= 5:  # Weekend
                return False
            market_open = datetime.time(9, 30)
            market_close = datetime.time(16, 0)
            return market_open <= now.time() <= market_close
    
    @staticmethod
    def return_time() -> datetime.time:
        """Return current time in US/Eastern timezone"""
        desired_zone = pytz.timezone('US/Eastern')
        now_utc = datetime.datetime.now(pytz.utc)
        now = now_utc.astimezone(desired_zone)
        return now.time()
    
    @staticmethod
    def calculate_position_size(amount: float, cash_at_risk: float, atr: float, price: float) -> int:
        """Calculate position size based on risk management"""
        dollar_risk_per_share = price * cash_at_risk / atr
        max_quantity = int(amount / price)
        if hasattr(dollar_risk_per_share, 'iloc'):
            dollar_risk_per_share = dollar_risk_per_share.iloc[0]
        affordable_quantity = int((amount * cash_at_risk / dollar_risk_per_share))
        quantity = min(max_quantity, affordable_quantity)
        return round(max_quantity)
    
    @staticmethod
    def display_portfolio(trading_client: TradingClient, portfolio):
        """Display portfolio information"""
        account = trading_client.get_account()
        print(f"Total assets: {account.equity}, Remaining cash: {account.cash}, "
              f"Buying power: {account.buying_power}, daily change: {round(float(account.equity)-float(account.last_equity), 2)}, "
              f"day trade count: {account.daytrade_count}")
        
        for position in trading_client.get_all_positions():
            print(f"Symbol: {position.symbol}, Entry Price: {position.avg_entry_price}, "
                  f"Quantity: {position.qty_available}, Unrealized_pl: {position.unrealized_pl}")