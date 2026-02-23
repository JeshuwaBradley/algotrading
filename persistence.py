import json
import os
import pickle
import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import asdict, is_dataclass
from config import config
from trading.order import Order
from trading.portfolio import Portfolio
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, datetime.time):
            return obj.strftime('%H:%M:%S')
        return super().default(obj)


class DateTimeDecoder:
    """Custom JSON decoder for datetime objects"""
    @staticmethod
    def decode_datetime(obj):
        for key, value in obj.items():
            if isinstance(value, str):
                try:
                    # Try to parse as datetime
                    obj[key] = datetime.datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    try:
                        # Try to parse as date
                        obj[key] = datetime.date.fromisoformat(value)
                    except (ValueError, TypeError):
                        pass
        return obj


class ModelPersistence:
    """Handles saving and loading ML models"""
    
    @staticmethod
    def save_model(model, model_name: str):
        """Save a model to disk"""
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    
    @staticmethod
    def load_model(model_name: str):
        """Load a model from disk"""
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return model
        return None
    
    @staticmethod
    def save_xgb_model(model: XGBClassifier, model_name: str):
        """Save XGBoost model using joblib"""
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"XGBoost model saved to {model_path}")
    
    @staticmethod
    def load_xgb_model(model_name: str) -> Optional[XGBClassifier]:
        """Load XGBoost model using joblib"""
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"XGBoost model loaded from {model_path}")
            return model
        return None
    
    @staticmethod
    def save_rf_model(model: RandomForestClassifier, model_name: str):
        """Save Random Forest model using joblib"""
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"Random Forest model saved to {model_path}")
    
    @staticmethod
    def load_rf_model(model_name: str) -> Optional[RandomForestClassifier]:
        """Load Random Forest model using joblib"""
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Random Forest model loaded from {model_path}")
            return model
        return None


class PortfolioPersistence:
    """Handles saving and loading portfolio state"""
    
    @staticmethod
    def save_portfolio(portfolio: Portfolio, filepath: str = config.PORTFOLIO_STATE_FILE):
        """Save portfolio state to JSON"""
        portfolio_state = {
            'stocks': portfolio.stocks,
            'initial_cash': portfolio.initial_cash,
            'cash': portfolio.cash,
            'cash_at_risk': portfolio.cash_at_risk,
            'bad_trades': portfolio.bad_trades,
            'day_trade_count': portfolio.day_trade_count,
            'pending_buys': {
                symbol: {
                    'expiry': info['expiry'].isoformat() if isinstance(info['expiry'], datetime.datetime) else info['expiry']
                }
                for symbol, info in portfolio.pending_buys.items()
            },
            'on_hold_until': {
                symbol: hold_until.isoformat() if isinstance(hold_until, datetime.datetime) else hold_until
                for symbol, hold_until in portfolio.on_hold_until.items()
            },
            'cash_allocated_per_stock': portfolio.cash_allocated_per_stock,
            'next_stocks': portfolio.next_stocks,
            'orders': [
                {
                    'symbol': order.symbol,
                    'quantity': order.quantity,
                    'entry_price': order.entry_price,
                    'trailing_distance': order.trailing_distance,
                    'stop_price': order.stop_price,
                    'old_stop_loss_price': order.old_stop_loss_price
                }
                for order in portfolio.orders
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(portfolio_state, f, cls=DateTimeEncoder, indent=2)
        print(f"Portfolio state saved to {filepath}")
    
    @staticmethod
    def load_portfolio(filepath: str = config.PORTFOLIO_STATE_FILE) -> Optional[Dict[str, Any]]:
        """Load portfolio state from JSON"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                portfolio_state = json.load(f, object_hook=DateTimeDecoder.decode_datetime)
            print(f"Portfolio state loaded from {filepath}")
            return portfolio_state
        return None
    
    @staticmethod
    def recreate_portfolio(portfolio_state: Dict[str, Any], buy_func, sell_func, 
                          buy_sell_model=None, up_down_model=None) -> Portfolio:
        """Recreate portfolio object from saved state"""
        portfolio = Portfolio(
            stocks=portfolio_state['stocks'],
            initial_cash=portfolio_state['initial_cash'],
            cash_at_risk=portfolio_state['cash_at_risk'],
            buy=buy_func,
            sell=sell_func,
            buy_sell_model=buy_sell_model,
            up_down_model=up_down_model
        )
        
        # Restore state
        portfolio.cash = portfolio_state['cash']
        portfolio.bad_trades = portfolio_state['bad_trades']
        portfolio.day_trade_count = portfolio_state['day_trade_count']
        portfolio.pending_buys = portfolio_state['pending_buys']
        portfolio.on_hold_until = portfolio_state['on_hold_until']
        portfolio.cash_allocated_per_stock = portfolio_state['cash_allocated_per_stock']
        portfolio.next_stocks = portfolio_state['next_stocks']
        
        # Restore orders
        portfolio.orders = [
            Order(
                symbol=order_data['symbol'],
                quantity=order_data['quantity'],
                entry_price=order_data['entry_price'],
                trailing_distance=order_data['trailing_distance'],
                stop_price=order_data['stop_price'],
                old_stop_loss_price=order_data.get('old_stop_loss_price')
            )
            for order_data in portfolio_state['orders']
        ]
        
        return portfolio


class UpdateTracker:
    """Tracks when models were last updated"""
    
    @staticmethod
    def get_last_update() -> Optional[datetime.date]:
        """Get the date of the last model update"""
        if os.path.exists(config.LAST_UPDATE_FILE):
            with open(config.LAST_UPDATE_FILE, 'r') as f:
                date_str = f.read().strip()
                return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        return None
    
    @staticmethod
    def set_last_update(date: datetime.date = None):
        """Set the date of the last model update"""
        if date is None:
            date = datetime.date.today()
        with open(config.LAST_UPDATE_FILE, 'w') as f:
            f.write(date.strftime('%Y-%m-%d'))
    
    @staticmethod
    def should_update() -> bool:
        """Check if models should be updated"""
        last_update = UpdateTracker.get_last_update()
        if last_update is None:
            return True
        
        days_since_update = (datetime.date.today() - last_update).days
        return days_since_update >= config.MODEL_UPDATE_DAYS