# from dataclasses import dataclass
# from typing import Optional
# 
# 
# @dataclass
# class Order:
#     """Represents a trading order"""
#     symbol: str
#     quantity: int
#     entry_price: float
#     trailing_distance: float
#     stop_price: float
#     old_stop_loss_price: Optional[float] = None


from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    """Represents a trading order with dynamic stop loss"""
    symbol: str
    quantity: int
    entry_price: float
    trailing_distance: float
    stop_price: float
    old_stop_loss_price: Optional[float] = None
    short: bool = False
    dynamic_stop_active: bool = False
    best_price: float = None  # Track best price reached since entry
    
    def __post_init__(self):
        """Initialize best_price after dataclass initialization"""
        if self.best_price is None:
            self.best_price = self.entry_price
    
    def update_dynamic_stop(self, current_price: float):
        """Update stop loss dynamically based on price movement"""
        # Calculate current unrealized P&L
        if not self.short:
            # For long positions
            unrealized_pl = (current_price - self.entry_price) * self.quantity

            # Update best price if current price is higher
            if current_price > self.best_price:
                self.best_price = current_price

            # Check if dynamic stop should activate
            if not self.dynamic_stop_active:
                if unrealized_pl >= 100:  # $100 profit threshold
                    self.dynamic_stop_active = True
                    # Set initial trailing stop
                    self.stop_price = current_price - (current_price * self.trailing_distance / 100)
                    print(f"Dynamic stop activated for {self.symbol} at stop ${self.stop_price:.2f}")
            else:
                # Dynamic stop is active - only move it UP
                # Calculate new potential stop based on best price
                potential_stop = self.best_price - (self.best_price * self.trailing_distance / 100)

                # Only update if new stop is HIGHER than current stop
                if potential_stop > self.stop_price:
                    self.old_stop_loss_price = self.stop_price
                    self.stop_price = potential_stop
                    print(f"Stop raised for {self.symbol} from ${self.old_stop_loss_price:.2f} to ${self.stop_price:.2f}")

        else:
            # For short positions (stop should move DOWN)
            unrealized_pl = (self.entry_price - current_price) * self.quantity

            # Update best price (lowest for short positions)
            if current_price < self.best_price:
                self.best_price = current_price

            # Check if dynamic stop should activate
            if not self.dynamic_stop_active:
                if unrealized_pl >= 100:  # $100 profit threshold
                    self.dynamic_stop_active = True
                    # Set initial trailing stop
                    self.stop_price = current_price + (current_price * self.trailing_distance / 100)
                    print(f"Dynamic stop activated for {self.symbol} (short) at stop ${self.stop_price:.2f}")
            else:
                # Dynamic stop is active - only move it DOWN for short positions
                # Calculate new potential stop based on best price (lowest)
                potential_stop = self.best_price + (self.best_price * self.trailing_distance / 100)

                # Only update if new stop is LOWER than current stop (for shorts, lower = better)
                if potential_stop < self.stop_price:
                    self.old_stop_loss_price = self.stop_price
                    self.stop_price = potential_stop
                    print(f"Stop lowered for {self.symbol} (short) from ${self.old_stop_loss_price:.2f} to ${self.stop_price:.2f}")