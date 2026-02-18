class Order:
    def __init__(self, symbol, quantity, entry_price, trailing_distance, stop_loss_price, old_stop_loss_price=None, short=False):
        self.symbol = symbol
        self.quantity = int(quantity)
        self.entry_price = entry_price
        self.trailing_distance = trailing_distance
        self.short = short
        self.stop_price = stop_loss_price
        self.old_stop_price = old_stop_loss_price
        self.dynamic_stop_active = False
        self.best_price = entry_price

    def update_dynamic_stop(self, current_price):
        if not self.short:
            # For long positions
            unrealized_pl = (current_price - self.entry_price) * self.quantity

            if current_price > self.best_price:
                self.best_price = current_price

            if not self.dynamic_stop_active:
                if unrealized_pl >= 100:
                    self.dynamic_stop_active = True
                    self.stop_price = current_price - (current_price * self.trailing_distance / 100)
                    print(f"Dynamic stop activated for {self.symbol} at stop ${self.stop_price:.2f}")
            else:
                potential_stop = self.best_price - (self.best_price * self.trailing_distance / 100)
                if potential_stop > self.stop_price:
                    self.stop_price = potential_stop
                    print(f"Stop raised for {self.symbol} to ${self.stop_price:.2f}")
        else:
            # For short positions
            unrealized_pl = (self.entry_price - current_price) * self.quantity

            if current_price < self.best_price:
                self.best_price = current_price

            if not self.dynamic_stop_active:
                if unrealized_pl >= 100:
                    self.dynamic_stop_active = True
                    self.stop_price = current_price + (current_price * self.trailing_distance / 100)
                    print(f"Dynamic stop activated for {self.symbol} (short) at stop ${self.stop_price:.2f}")
            else:
                potential_stop = self.best_price + (self.best_price * self.trailing_distance / 100)
                if potential_stop < self.stop_price:
                    self.stop_price = potential_stop
                    print(f"Stop lowered for {self.symbol} (short) to ${self.stop_price:.2f}")