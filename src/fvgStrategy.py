from backtesting import Backtest, Strategy
from backtesting.test import EURUSD  # Example dataset

class FvgStrategy(Strategy):
    # Strategy parameters
    gap_threshold = 0.002  # 0.2% minimum gap size
    stop_loss_pct = 0.01   # 1% stop loss
    take_profit_pct = 0.02 # 2% take profit

    def init(self):
        pass  # No pre-calculations needed

    def next(self):
        # Loop over the last few candles to check for FVG
        for i in range(2, len(self.data.Close)):  # Start at 2 to avoid out-of-range issues
            prev_high = self.data.High[i - 2]  # High of 1st candle
            prev_low = self.data.Low[i - 2]    # Low of 1st candle
            curr_high = self.data.High[i - 1]  # High of 2nd candle
            curr_low = self.data.Low[i - 1]    # Low of 2nd candle
            close = self.data.Close[i]        # Current close price

            # Calculate the gap size for validation
            upward_gap_size = curr_low - prev_high
            downward_gap_size = prev_low - curr_high

            # Check for upward FVG (bullish gap)
            if upward_gap_size > prev_high * self.gap_threshold:
                # Place buy order when price retraces into the gap
                if prev_high < close < curr_low:
                    sl = close * (1 - self.stop_loss_pct)  # Stop loss below entry
                    tp = close * (1 + self.take_profit_pct)  # Take profit above entry
                    self.buy(sl=sl, tp=tp)

            # Check for downward FVG (bearish gap)
            elif downward_gap_size > prev_low * self.gap_threshold:
                # Place sell order when price retraces into the gap
                if curr_high < close < prev_low:
                    sl = close * (1 + self.stop_loss_pct)  # Stop loss above entry
                    tp = close * (1 - self.take_profit_pct)  # Take profit below entry
                    self.sell(sl=sl, tp=tp)

        # Exit logic: Ensure no lingering positions
        if self.position.is_long and self.data.Close[-1] < self.data.Close[-2]:
            self.position.close()
        elif self.position.is_short and self.data.Close[-1] > self.data.Close[-2]:
            self.position.close()

# Backtest the strategy
bt = Backtest(EURUSD, FvgStrategy, cash=10_000, commission=.002, exclusive_orders=True)
output = bt.run()
print(output)
bt.plot()
