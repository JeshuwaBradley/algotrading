from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    quantity: int
    entry_price: float
    trailing_distance: float
    stop_price: float
    old_stop_loss_price: Optional[float] = None