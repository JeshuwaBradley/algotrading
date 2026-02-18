import datetime
import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def buy(trading_client, stock, quantity, limit_price):
    rounded_price = round(limit_price, 2)
    limitOrder = LimitOrderRequest(
        symbol=stock,
        limit_price=rounded_price,
        qty=quantity,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )
    try:
        buy_limit_order = trading_client.submit_order(order_data=limitOrder)
        print("Successfully placed limit buy order for", quantity, "shares of", stock)
        return buy_limit_order
    except Exception as e:
        print("Error placing order:", e)
        return None

def sell(trading_client, stock, quantity, limit_price):
    rounded_price = round(limit_price, 2)
    limitOrder = LimitOrderRequest(
        symbol=stock,
        limit_price=rounded_price,
        qty=quantity,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    try:
        sell_limit_order = trading_client.submit_order(order_data=limitOrder)
        print("Successfully placed limit sell order for", quantity, "shares of", stock)
        return sell_limit_order
    except Exception as e:
        print("Error placing order:", e)
        return None

def sell_market(trading_client, stock, quantity):
    marketOrder = MarketOrderRequest(
        symbol=stock,
        qty=quantity,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    try:
        sell_market_order = trading_client.submit_order(order_data=marketOrder)
        print("Successfully placed market sell order for", quantity, "shares of", stock)
        return sell_market_order
    except Exception as e:
        print("Error placing order:", e)
        return None

def is_market_open(trading_client):
    return trading_client.get_clock().is_open

def return_time():
    desired_zone = pytz.timezone('US/Eastern')
    now_utc = datetime.datetime.now(pytz.utc)
    now = now_utc.astimezone(desired_zone)
    return now.time()

def calculate_position_size(amount, cash_at_risk, atr, price):
    if atr is None or atr == 0:
        return max(1, int(amount / price))
    dollar_risk_per_share = price * cash_at_risk / atr
    max_quantity = int(amount / price)
    if isinstance(dollar_risk_per_share, pd.Series):
        dollar_risk_per_share = dollar_risk_per_share.iloc[0]
    affordable_quantity = int((amount * cash_at_risk / dollar_risk_per_share))
    quantity = min(max_quantity, affordable_quantity)
    return max(1, quantity)