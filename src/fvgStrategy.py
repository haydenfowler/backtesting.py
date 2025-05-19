from backtesting import Backtest, Strategy
from data_loader import load_yahoo_finance_data

class FvgStrategy(Strategy):
    # Strategy parameters
    gap_threshold = 0.002  # 0.2% minimum gap size

    def init(self):
        pass  # No pre-calculations needed

    def next(self):
        # Only check for new setups if we don't have an open position
        if not self.position:
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
                        self.buy()

                # Check for downward FVG (bearish gap)
                elif downward_gap_size > prev_low * self.gap_threshold:
                    # Place sell order when price retraces into the gap
                    if curr_high < close < prev_low:
                        self.sell()

        # Exit logic based on price action
        # For long positions, exit if price closes below the previous candle
        elif self.position.is_long and self.data.Close[-1] < self.data.Close[-2]:
            self.position.close()
        # For short positions, exit if price closes above the previous candle
        elif self.position.is_short and self.data.Close[-1] > self.data.Close[-2]:
            self.position.close()

if __name__ == "__main__":
    # Load data using the data_loader module
    symbol = "SPY"  # S&P 500 ETF
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    data = load_yahoo_finance_data(symbol, start_date, end_date)
    
    # Backtest the strategy
    bt = Backtest(data, FvgStrategy, cash=10_000, commission=.002, exclusive_orders=True)
    output = bt.run()
    print(output)
    bt.plot()
