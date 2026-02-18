import datetime
import time
import pytz
from pandas import DatetimeIndex
from .order import Order
import config

class Portfolio:
    def __init__(self, trading_client, orders, initial_cash, cash_at_risk, buy_func, sell_func):
        self.trading_client = trading_client
        self.cash = float(trading_client.get_account().cash)
        self.initialCash = initial_cash
        self.orders = orders
        self.cash_at_risk = cash_at_risk
        self.day_trade_count = trading_client.get_account().daytrade_count
        self.next_stocks = config.STOCK_POOL.copy()
        self.stocks = config.ACTIVE_STOCKS.copy()
        self.cash_allocated_per_stock = config.INITIAL_STOCK_CASH.copy()
        self.buy = buy_func
        self.sell_market = sell_func
        self.buy_sell_model = None
        self.up_down_model = None
        self.bad_trades = 0
        self.pending_buys = {}

    def set_models(self, buy_sell_model, up_down_model):
        self.buy_sell_model = buy_sell_model
        self.up_down_model = up_down_model

    def place_buy_and_track(self, symbol, quantity, price):
        try:
            buy_order = self.buy(stock=symbol, quantity=quantity, limit_price=price)
            self.pending_buys[symbol] = {
                "order_id": buy_order.id,
                "symbol": symbol,
                "qty": quantity,
                "price": price,
                "timestamp": datetime.datetime.now()
            }
            print(f"[BUY PENDING] {symbol} – waiting to fill…")
            return buy_order
        except Exception as e:
            print(f"Error placing buy order for {symbol}: {e}")
            return None

    def check_pending_buys(self):
        to_delete = []
        for symbol, info in self.pending_buys.items():
            order_id = info["order_id"]
            qty = info["qty"]
            price = info["price"]
            timestamp = info["timestamp"]

            try:
                alpaca_order = self.trading_client.get_order_by_id(order_id)

                if alpaca_order.status == "filled":
                    print(f'{symbol} filled')
                    filled_price = float(alpaca_order.filled_avg_price)
                    print(f"[BUY FILLED] {symbol} @ {filled_price}")

                    stop_price = filled_price - 150 / qty
                    new_order = Order(symbol, qty, filled_price, 0.5, stop_price)
                    self.orders.append(new_order)
                    to_delete.append(symbol)
                    continue

                age_min = (datetime.datetime.now() - timestamp).seconds / 60
                if age_min >= 15:
                    print(f"[BUY TIMEOUT] {symbol}: cancelling & retrying…")
                    self.trading_client.cancel_order_by_id(order_id)
                    new_order = self.place_buy_and_track(symbol, qty, price)
                    self.pending_buys[symbol]["timestamp"] = datetime.datetime.now()

            except Exception as e:
                print(f"Error checking pending buy order for {symbol}: {e}")

        for symbol in to_delete:
            del self.pending_buys[symbol]

    def no_progress(self, symbol, unrealized_pl):
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=5,
            symbols=[symbol],
            side=QueryOrderStatus.BUY
        )
        orders = self.trading_client.get_orders(filter=request)
        if not orders:
            return False

        filled_date = orders[0].filled_at
        if not filled_date:
            return False

        now = datetime.datetime.now(pytz.utc)
        days_held = (now - filled_date).days
        return days_held >= 7 and unrealized_pl < 100

    def change_trading_stock(self, changing_symbol):
        for stock in self.next_stocks:
            if stock not in self.cash_allocated_per_stock:
                changing_symbol_data = self.cash_allocated_per_stock[changing_symbol]
                del self.cash_allocated_per_stock[changing_symbol]
                self.cash_allocated_per_stock[stock] = changing_symbol_data
                self.stocks.remove(changing_symbol)
                self.stocks.append(stock)
                self.next_stocks.remove(stock)
                return True
        print("all stocks are being used")
        return False

    def update_orders(self, current_prices, buy_sell_preds, up_down_preds, old_prices):
        utc = datetime.datetime.now(pytz.utc)
        tz = pytz.timezone("US/Eastern")
        ny = utc.astimezone(tz)
        date = DatetimeIndex([ny])[0]

        for order in self.orders:
            try:
                symbol = order.symbol
                entry_price = order.entry_price
                quantity = int(order.quantity)
                current_price = current_prices[symbol]
                old_price = old_prices[symbol]
                unrealized_pl = round(float(current_price * quantity - entry_price * quantity))

                order.update_dynamic_stop(float(current_price))

                if current_price <= order.stop_price:
                    print(f"Stop triggered for {symbol} at {current_price}")

                    sell_order = self.sell_market(stock=symbol, quantity=quantity)

                    while True:
                        order_status = self.trading_client.get_order_by_id(sell_order.id)
                        if order_status.status == "filled":
                            break
                        time.sleep(1)

                    filled_price = float(order_status.filled_avg_price)
                    filled_qty = float(order_status.filled_qty)
                    total_sale_value = filled_price * filled_qty

                    print(f"Sold {filled_qty} {symbol} at ${filled_price:.2f} each")
                    print(f"Total Sale Value: ${total_sale_value:.2f}")

                    self.cash_allocated_per_stock[symbol]["cash"] += total_sale_value
                    if filled_price < entry_price:
                        self.bad_trades += 1

                    self.change_trading_stock(symbol)
                else:
                    continue

            except Exception as e:
                print(f"Error updating {order.symbol}: {e}")

    def check_on_hold(self):
        utc = datetime.datetime.now(pytz.utc)
        tz = pytz.timezone("US/Eastern")
        ny = utc.astimezone(tz)
        date = DatetimeIndex([ny])[0]
        for stock in self.cash_allocated_per_stock:
            d1 = self.cash_allocated_per_stock[stock]["hold_set_date"]
            if d1 is not None:
                time_diff = date - d1
                if time_diff.days >= 2:
                    self.cash_allocated_per_stock[stock]["on_hold"] = False
                    self.cash_allocated_per_stock[stock]["hold_set_date"] = None

    def execute_trade(self, symbol, quantity, price, average_fluc, cash):
        cost = quantity * price + 7
        if (float(self.cash_allocated_per_stock[symbol]["cash"]) > cost and 
            not self.cash_allocated_per_stock[symbol]["on_hold"] and 
            float(cash) >= cost and self.day_trade_count <= config.MAX_DAY_TRADES):

            stop_price = price - 150 / quantity
            order = Order(symbol, quantity, price, 0.5, stop_price, short=False)
            self.orders.append(order)
            self.place_buy_and_track(symbol, quantity, price)
            self.cash_allocated_per_stock[symbol]["cash"] -= cost + 7
            return True
        else:
            print('Trade skipped:', symbol)
            return False