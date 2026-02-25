#from typing import Dict, List, Optional, Callable, Any
#from .order import Order
#from config import config
#import copy
#import datetime
#
#
#class Portfolio:
#    """Manages portfolio positions and trading decisions"""
#    
#    def __init__(self, stocks: List[str], initial_cash: float, cash_at_risk: float, 
#                 buy: Callable, sell: Callable, buy_sell_model=None, up_down_model=None):
#        self.stocks = stocks
#        self.initial_cash = initial_cash
#        self.cash = initial_cash
#        self.cash_at_risk = cash_at_risk
#        self.buy = buy
#        self.sell = sell
#        self.buy_sell_model = buy_sell_model
#        self.up_down_model = up_down_model
#        self.orders: List[Order] = []
#        self.bad_trades = 0
#        self.day_trade_count = 0
#        self.pending_buys = {}
#        self.on_hold_until = {}
#        self.cash_allocated_per_stock = {}
#        self.next_stocks = []
#        
#        # Initialize cash allocation
#        self._initialize_cash_allocation()
#    
#    def _initialize_cash_allocation(self):
#        """Initialize cash allocation for each stock"""
#        cash_per_stock = self.cash / len(self.stocks)
#        for stock in self.stocks:
#            self.cash_allocated_per_stock[stock] = {
#                "cash": cash_per_stock,
#                "used": False
#            }
#    
#    def cash_per_stock(self) -> float:
#        """Calculate cash per stock"""
#        return self.cash / len(self.stocks)
#    
#    def execute_trade(self, symbol: str, quantity: int, price: float, 
#                     average_fluc: float, cash: float) -> bool:
#        """Execute a trade if conditions are met"""
#        if self.cash_allocated_per_stock[symbol]["used"]:
#            return False
#        
#        total_cost = quantity * price
#        if total_cost <= float(cash) and not self.cash_allocated_per_stock[symbol]["used"]:
#            self.buy(symbol, quantity, price)
#            order = Order(
#                symbol=symbol,
#                quantity=quantity,
#                entry_price=price,
#                trailing_distance=config.TRAILING_DISTANCE,  # 0.5 from config
#                stop_price=price - (price * config.TRAILING_DISTANCE / 150)  # INITIAL STOP LOSS
#            )
#            self.orders.append(order)
#            self.cash_allocated_per_stock[symbol]["used"] = True
#            self.cash_allocated_per_stock[symbol]["cash"] -= total_cost
#            self.cash -= total_cost
#            return True
#        return False
#    
#    def check_on_hold(self):
#        """Check if any stocks are on hold"""
#        current_time = datetime.datetime.now()
#        expired_stocks = []
#        for stock, hold_until in self.on_hold_until.items():
#            if current_time > hold_until:
#                expired_stocks.append(stock)
#        
#        for stock in expired_stocks:
#            del self.on_hold_until[stock]
#    
#    def check_pending_buys(self):
#        """Check pending buy orders"""
#        current_time = datetime.datetime.now()
#        expired_orders = []
#        
#        for symbol, order_info in self.pending_buys.items():
#            if current_time > order_info['expiry']:
#                expired_orders.append(symbol)
#        
#        for symbol in expired_orders:
#            del self.pending_buys[symbol]
#    
#    def update_orders(self, current_prices: Dict[str, float], 
#                     buy_sell_preds: Dict[str, int], 
#                     up_down_preds: Dict[str, int],
#                     old_prices: Dict[str, Optional[float]]):
#        """Update stop loss orders"""
#        new_orders = []
#        
#        for order in self.orders:
#            symbol = order.symbol
#            if symbol not in current_prices:
#                new_orders.append(order)
#                continue
#            
#            current_price = current_prices[symbol]
#            buy_sell_pred = buy_sell_preds.get(symbol)
#            up_down_pred = up_down_preds.get(symbol)
#            old_price = old_prices.get(symbol)
#            
#            # Calculate new stop price
#            if order.entry_price is not None:
#                if current_price > order.entry_price + 0.5:
#                    trailing_stop = current_price - (current_price * config.TRAILING_DISTANCE / 100)
#                    if old_price is None or trailing_stop > old_price:
#                        order.stop_price = trailing_stop
#                    else:
#                        order.stop_price = old_price
#                elif current_price > order.entry_price:
#                    order.stop_price = order.entry_price - 0.10
#                else:
#                    order.stop_price = order.entry_price - 0.10
#            else:
#                order.stop_price = current_price * 0.95
#            
#            # Check stop loss
#            if current_price <= order.stop_price:
#                self.sell(symbol, order.quantity)
#                self.bad_trades += 1
#                self.cash += order.quantity * current_price
#                # Don't add to new_orders
#            else:
#                new_orders.append(order)
#        
#        self.orders = new_orders
#    
#    def get_state(self) -> Dict[str, Any]:
#        """Get portfolio state for persistence"""
#        return {
#            'stocks': self.stocks,
#            'initial_cash': self.initial_cash,
#            'cash': self.cash,
#            'cash_at_risk': self.cash_at_risk,
#            'bad_trades': self.bad_trades,
#            'day_trade_count': self.day_trade_count,
#            'pending_buys': self.pending_buys,
#            'on_hold_until': self.on_hold_until,
#            'cash_allocated_per_stock': self.cash_allocated_per_stock,
#            'next_stocks': self.next_stocks,
#            'orders': [
#                {
#                    'symbol': order.symbol,
#                    'quantity': order.quantity,
#                    'entry_price': order.entry_price,
#                    'trailing_distance': order.trailing_distance,
#                    'stop_price': order.stop_price,
#                    'old_stop_loss_price': order.old_stop_loss_price
#                }
#                for order in self.orders
#            ]
#        }


from typing import Dict, List, Optional, Callable, Any
from .order import Order
from config import config
import datetime
import time
import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide


class Portfolio:
    """Manages portfolio positions and trading decisions"""
    
    def __init__(self, stocks: List[str], initial_cash: float, cash_at_risk: float, 
                 buy: Callable, sell: Callable, trading_client: TradingClient = None,
                 buy_sell_model=None, up_down_model=None):
        self.stocks = stocks
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.cash_at_risk = cash_at_risk
        self.buy = buy
        self.sell = sell
        self.trading_client = trading_client
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
                "used": False,
                "on_hold": False,
                "hold_set_date": None
            }
    
    def cash_per_stock(self) -> float:
        """Calculate cash per stock"""
        return self.cash / len(self.stocks)
    
    def place_buy_and_track(self, symbol: str, quantity: int, price: float):
        """Place a buy order and track it"""
        try:
            buy_order = self.buy(stock=symbol, quantity=quantity, limit_price=price)
            
            # Get order ID properly
            order_id = buy_order.id if hasattr(buy_order, 'id') else str(buy_order)
            
            self.pending_buys[symbol] = {
                "order_id": order_id,
                "symbol": symbol,
                "qty": quantity,
                "price": price,
                "timestamp": datetime.datetime.now()  # This is the only timestamp field
            }
            
            print(f"[BUY PENDING] {symbol} – waiting to fill…")
            return buy_order
            
        except Exception as e:
            print(f"Error placing buy order for {symbol}: {e}")
            return None
    
    def check_pending_buys(self):
        """Check pending buy orders and handle fills/timeouts"""
        if not self.trading_client:
            print("No trading client available for pending buys")
            return
            
        to_delete = []
        
        for symbol, info in list(self.pending_buys.items()):  # Use list() to avoid modification during iteration
            order_id = info["order_id"]
            qty = info["qty"]
            price = info["price"]
            timestamp = info["timestamp"]
            
            try:
                # Handle if order_id is not a string (might be order object)
                if not isinstance(order_id, str):
                    if hasattr(order_id, 'id'):
                        order_id = order_id.id
                    else:
                        print(f"Invalid order_id for {symbol}")
                        continue
                
                alpaca_order = self.trading_client.get_order_by_id(order_id)
                
                # Case 1 — Filled
                if alpaca_order.status == "filled":
                    print(f'{symbol} filled')
                    filled_price = float(alpaca_order.filled_avg_price)
                    print(f"[BUY FILLED] {symbol} @ {filled_price}")
                    
                    # Calculate stop price (initial stop)
                    stop_price = filled_price - 150 / qty
                    
                    # Create new order with dynamic stop tracking
                    new_order = Order(
                        symbol=symbol,
                        quantity=qty,
                        entry_price=filled_price,
                        trailing_distance=config.TRAILING_DISTANCE,
                        stop_price=stop_price,
                        short=False
                    )
                    self.orders.append(new_order)
                    
                    # Mark the cash as used (it was already marked, but ensure it's correct)
                    if symbol in self.cash_allocated_per_stock:
                        self.cash_allocated_per_stock[symbol]["used"] = True
                    
                    to_delete.append(symbol)
                    print(f"Order added to portfolio. Total orders: {len(self.orders)}")
                    continue
                
                # Case 2 — Not filled after 15 min: cancel & re-place
                age_min = (datetime.datetime.now() - timestamp).seconds / 60
                if age_min >= 15:
                    print(f"[BUY TIMEOUT] {symbol}: cancelling & retrying…")
                    
                    self.trading_client.cancel_order_by_id(order_id)
                    
                    # re-submit order at SAME price
                    self.place_buy_and_track(symbol, qty, price)
                    
            except Exception as e:
                print(f"Error checking pending buy order for {symbol}: {e}")
                # If order not found, remove it from pending
                if "not found" in str(e).lower():
                    to_delete.append(symbol)
        
        # remove symbols whose orders filled
        for symbol in to_delete:
            if symbol in self.pending_buys:
                del self.pending_buys[symbol]
    
    def execute_trade(self, symbol: str, quantity: int, price: float, 
                     average_fluc: float, cash: float) -> bool:
        """Execute a trade if conditions are met"""
        if self.cash_allocated_per_stock[symbol]["used"]:
            return False
        
        total_cost = quantity * price
        # Check if stock is on hold and we have enough cash
        if (total_cost <= float(self.cash_allocated_per_stock[symbol]["cash"]) and 
            not self.cash_allocated_per_stock[symbol]["on_hold"] and
            not self.cash_allocated_per_stock[symbol]["used"]):
            
            # Place buy order and track it
            self.place_buy_and_track(symbol, quantity, price)
            
            # Mark as used but not fully executed yet
            self.cash_allocated_per_stock[symbol]["used"] = True
            self.cash_allocated_per_stock[symbol]["cash"] -= total_cost
            self.cash -= total_cost
            return True
        return False
    
    def check_on_hold(self):
        """Check if any stocks are on hold"""
        utc = datetime.datetime.now(pytz.utc)
        tz = pytz.timezone("US/Eastern")
        ny = utc.astimezone(tz)
        current_date = ny.date()
        
        for stock in self.cash_allocated_per_stock:
            hold_date = self.cash_allocated_per_stock[stock].get("hold_set_date")
            if hold_date is not None:
                # Convert to date if it's datetime
                if isinstance(hold_date, datetime.datetime):
                    hold_date = hold_date.date()
                
                time_diff = current_date - hold_date
                if time_diff.days >= 2:
                    self.cash_allocated_per_stock[stock]["on_hold"] = False
                    self.cash_allocated_per_stock[stock]["hold_set_date"] = None
    
    def no_progress(self, symbol: str, unrealized_pl: float) -> bool:
        """Check if a position has made no progress after 7 days"""
        if not self.trading_client:
            return False
            
        print('Checking for progress...')
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=5,
            symbols=[symbol],
            side=OrderSide.BUY
        )
        orders = self.trading_client.get_orders(filter=request)
        if not orders:
            print("No orders found")
            return False
        
        # Find the first filled order
        filled_order = None
        for order in orders:
            if order.filled_at is not None:
                filled_order = order
                break
                
        if not filled_order or not filled_order.filled_at:
            print("No filled order found")
            return False
        
        filled_date = filled_order.filled_at
        now = datetime.datetime.now(pytz.utc)
        days_held = (now - filled_date).days
        return days_held >= 7 and unrealized_pl < 100
    
    def change_trading_stock(self, changing_symbol: str) -> bool:
        """Replace a stock with one from next_stocks"""
        if not self.next_stocks:
            print("No stocks in next_stocks list")
            return False
            
        # Get data of stock to replace
        if changing_symbol not in self.cash_allocated_per_stock:
            print(f"Stock {changing_symbol} not found in portfolio")
            return False
            
        changing_symbol_data = self.cash_allocated_per_stock[changing_symbol].copy()
        
        # Find a new stock not already in portfolio
        for stock in self.next_stocks:
            if stock not in self.cash_allocated_per_stock:
                # Add new stock
                self.cash_allocated_per_stock[stock] = changing_symbol_data
                
                # Remove old stock
                del self.cash_allocated_per_stock[changing_symbol]
                
                # Update stocks list
                self.stocks.remove(changing_symbol)
                self.stocks.append(stock)
                
                # Remove from next_stocks
                self.next_stocks.remove(stock)
                
                print(f"Replaced {changing_symbol} with {stock}")
                return True
                
        print("All stocks in next_stocks are already being used")
        return False
    
    def update_orders(self, current_prices: Dict[str, float], 
                     buy_sell_preds: Dict[str, int], 
                     up_down_preds: Dict[str, int],
                     old_prices: Dict[str, Optional[float]]):
        """Update stop loss orders dynamically"""
        new_orders = []
        
        for order in self.orders:
            symbol = order.symbol
            if symbol not in current_prices:
                new_orders.append(order)
                continue
            
            current_price = current_prices[symbol]
            
            # Calculate unrealized P&L
            unrealized_pl = (current_price - order.entry_price) * order.quantity
            
            # Update dynamic stop
            order.update_dynamic_stop(current_price)
            
            # Check for no progress condition (7 days with < $100 profit)
            if self.no_progress(symbol, unrealized_pl):
                print(f"No progress for {symbol} after 7 days. Selling...")
                self._execute_sell(order, current_price)
                self.bad_trades += 1
                # Don't add to new_orders (position closed)
                continue
            
            # Check stop loss
            if current_price <= order.stop_price:
                print(f"Stop triggered for {symbol} at {current_price}")
                self._execute_sell(order, current_price)
                self.bad_trades += 1
                # Don't add to new_orders
            else:
                new_orders.append(order)
        
        self.orders = new_orders
    
    def _execute_sell(self, order: Order, current_price: float):
        """Execute a sell order and handle the proceeds"""
        try:
            # Place market sell order
            sell_order = self.sell(stock=order.symbol, quantity=order.quantity)
            
            # If we have trading client, wait for fill
            if self.trading_client and sell_order:
                # Handle if sell_order is order ID or object
                order_id = sell_order.id if hasattr(sell_order, 'id') else sell_order
                
                # Wait for fill (with timeout)
                max_wait = 30  # seconds
                start_time = time.time()
                
                while time.time() - start_time < max_wait:
                    order_status = self.trading_client.get_order_by_id(order_id)
                    if order_status.status == "filled":
                        filled_price = float(order_status.filled_avg_price)
                        filled_qty = float(order_status.filled_qty)
                        total_sale_value = filled_price * filled_qty
                        
                        print(f"Sold {filled_qty} {order.symbol} at ${filled_price:.2f} each")
                        print(f"Total Sale Value: ${total_sale_value:.2f}")
                        
                        # Add back to allocated cash
                        if order.symbol in self.cash_allocated_per_stock:
                            self.cash_allocated_per_stock[order.symbol]["cash"] += total_sale_value
                            self.cash_allocated_per_stock[order.symbol]["used"] = False
                            
                            # If sold at a loss, mark as bad trade
                            if filled_price < order.entry_price:
                                self.bad_trades += 1
                            
                            # Try to replace with next stock
                            self.change_trading_stock(order.symbol)
                        break
                    time.sleep(1)
            else:
                # Simple case - just add estimated value
                estimated_value = current_price * order.quantity
                if order.symbol in self.cash_allocated_per_stock:
                    self.cash_allocated_per_stock[order.symbol]["cash"] += estimated_value
                    self.cash_allocated_per_stock[order.symbol]["used"] = False
                    
                    if current_price < order.entry_price:
                        self.bad_trades += 1
                    
                    self.change_trading_stock(order.symbol)
                    
        except Exception as e:
            print(f"Error executing sell for {order.symbol}: {e}")
            # Fallback - just update cash
            estimated_value = current_price * order.quantity
            if order.symbol in self.cash_allocated_per_stock:
                self.cash_allocated_per_stock[order.symbol]["cash"] += estimated_value
                self.cash_allocated_per_stock[order.symbol]["used"] = False
    
    def get_state(self) -> Dict[str, Any]:
        """Get portfolio state for persistence"""
        return {
            'stocks': self.stocks,
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'cash_at_risk': self.cash_at_risk,
            'bad_trades': self.bad_trades,
            'day_trade_count': self.day_trade_count,
            'pending_buys': {
                symbol: {
                    'order_id': info['order_id'],
                    'symbol': info['symbol'],
                    'qty': info['qty'],
                    'price': info['price'],
                    'timestamp': info['timestamp'].isoformat() if isinstance(info['timestamp'], datetime.datetime) else info['timestamp']
                }
                for symbol, info in self.pending_buys.items()
            },
            'on_hold_until': {
                symbol: hold_until.isoformat() if isinstance(hold_until, (datetime.datetime, datetime.date)) else hold_until
                for symbol, hold_until in self.on_hold_until.items()
            },
            'cash_allocated_per_stock': self.cash_allocated_per_stock,
            'next_stocks': self.next_stocks,
            'orders': [
                {
                    'symbol': order.symbol,
                    'quantity': order.quantity,
                    'entry_price': order.entry_price,
                    'trailing_distance': order.trailing_distance,
                    'stop_price': order.stop_price,
                    'old_stop_loss_price': order.old_stop_loss_price,
                    'short': order.short,
                    'dynamic_stop_active': order.dynamic_stop_active,
                    'best_price': order.best_price
                }
                for order in self.orders
            ]
        }