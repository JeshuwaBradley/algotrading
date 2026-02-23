from typing import Dict, List, Optional, Callable, Any
from .order import Order
from config import config
import copy
import datetime


class Portfolio:
    """Manages portfolio positions and trading decisions"""
    
    def __init__(self, stocks: List[str], initial_cash: float, cash_at_risk: float, 
                 buy: Callable, sell: Callable, buy_sell_model=None, up_down_model=None):
        self.stocks = stocks
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.cash_at_risk = cash_at_risk
        self.buy = buy
        self.sell = sell
        self.buy_sell_model = buy_sell_model
        self.up_down_model = up_down_model
        self.orders: List[Order] = []
        self.bad_trades = 0
        self.day_trade_count = 0
        self.pending_buys = {}
        self.on_hold_until = {}
        self.cash_allocated_per_stock = {}
        self.next_stocks = []
        
        # Initialize cash allocation
        self._initialize_cash_allocation()
    
    def _initialize_cash_allocation(self):
        """Initialize cash allocation for each stock"""
        cash_per_stock = self.cash / len(self.stocks)
        for stock in self.stocks:
            self.cash_allocated_per_stock[stock] = {
                "cash": cash_per_stock,
                "used": False
            }
    
    def cash_per_stock(self) -> float:
        """Calculate cash per stock"""
        return self.cash / len(self.stocks)
    
    def execute_trade(self, symbol: str, quantity: int, price: float, 
                     average_fluc: float, cash: float) -> bool:
        """Execute a trade if conditions are met"""
        if self.cash_allocated_per_stock[symbol]["used"]:
            return False
        
        total_cost = quantity * price
        if total_cost <= float(cash) and not self.cash_allocated_per_stock[symbol]["used"]:
            self.buy(symbol, quantity, price)
            self.cash_allocated_per_stock[symbol]["used"] = True
            self.cash_allocated_per_stock[symbol]["cash"] -= total_cost
            self.cash -= total_cost
            return True
        return False
    
    def check_on_hold(self):
        """Check if any stocks are on hold"""
        current_time = datetime.datetime.now()
        expired_stocks = []
        for stock, hold_until in self.on_hold_until.items():
            if current_time > hold_until:
                expired_stocks.append(stock)
        
        for stock in expired_stocks:
            del self.on_hold_until[stock]
    
    def check_pending_buys(self):
        """Check pending buy orders"""
        current_time = datetime.datetime.now()
        expired_orders = []
        
        for symbol, order_info in self.pending_buys.items():
            if current_time > order_info['expiry']:
                expired_orders.append(symbol)
        
        for symbol in expired_orders:
            del self.pending_buys[symbol]
    
    def update_orders(self, current_prices: Dict[str, float], 
                     buy_sell_preds: Dict[str, int], 
                     up_down_preds: Dict[str, int],
                     old_prices: Dict[str, Optional[float]]):
        """Update stop loss orders"""
        new_orders = []
        
        for order in self.orders:
            symbol = order.symbol
            if symbol not in current_prices:
                new_orders.append(order)
                continue
            
            current_price = current_prices[symbol]
            buy_sell_pred = buy_sell_preds.get(symbol)
            up_down_pred = up_down_preds.get(symbol)
            old_price = old_prices.get(symbol)
            
            # Calculate new stop price
            if order.entry_price is not None:
                if current_price > order.entry_price + 0.5:
                    trailing_stop = current_price - (current_price * config.TRAILING_DISTANCE / 100)
                    if old_price is None or trailing_stop > old_price:
                        order.stop_price = trailing_stop
                    else:
                        order.stop_price = old_price
                elif current_price > order.entry_price:
                    order.stop_price = order.entry_price - 0.10
                else:
                    order.stop_price = order.entry_price - 0.10
            else:
                order.stop_price = current_price * 0.95
            
            # Check stop loss
            if current_price <= order.stop_price:
                self.sell(symbol, order.quantity)
                self.bad_trades += 1
                self.cash += order.quantity * current_price
                # Don't add to new_orders
            else:
                new_orders.append(order)
        
        self.orders = new_orders
    
    def get_state(self) -> Dict[str, Any]:
        """Get portfolio state for persistence"""
        return {
            'stocks': self.stocks,
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'cash_at_risk': self.cash_at_risk,
            'bad_trades': self.bad_trades,
            'day_trade_count': self.day_trade_count,
            'pending_buys': self.pending_buys,
            'on_hold_until': self.on_hold_until,
            'cash_allocated_per_stock': self.cash_allocated_per_stock,
            'next_stocks': self.next_stocks,
            'orders': [
                {
                    'symbol': order.symbol,
                    'quantity': order.quantity,
                    'entry_price': order.entry_price,
                    'trailing_distance': order.trailing_distance,
                    'stop_price': order.stop_price,
                    'old_stop_loss_price': order.old_stop_loss_price
                }
                for order in self.orders
            ]
        }