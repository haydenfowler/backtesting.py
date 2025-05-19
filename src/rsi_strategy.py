from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
from data_loader import load_yahoo_finance_data

class RSIStrategy(Strategy):
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    
    def init(self):
        close = self.data.Close
        # Calculate RSI
        self.rsi = self.I(RSI, close, self.rsi_period)
    
    def next(self):
        # Buy when RSI crosses below oversold threshold
        if self.rsi[-2] > self.rsi_oversold and self.rsi[-1] <= self.rsi_oversold:
            self.buy()
            
        # Sell when RSI crosses above overbought threshold
        elif self.rsi[-2] < self.rsi_overbought and self.rsi[-1] >= self.rsi_overbought:
            self.sell()

# Define RSI calculation
def RSI(array, n):
    """Relative Strength Index"""
    # The backtesting library uses a custom Array type, not pandas Series
    # Convert to numpy array
    prices = np.asarray(array)
    
    # Calculate price changes
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down if down != 0 else float('inf')
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)
    
    # Calculate RSI using Wilder's smoothing method
    for i in range(n, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up * (n-1) + upval) / n
        down = (down * (n-1) + downval) / n
        
        rs = up/down if down != 0 else float('inf')
        rsi[i] = 100. - 100./(1. + rs)
        
    # Return without trying to set an index
    return rsi

if __name__ == '__main__':
    # Load data for Tesla
    symbol = "TSLA"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    data = load_yahoo_finance_data(symbol, start_date, end_date)
    
    # Run backtest
    bt = Backtest(
        data,
        RSIStrategy,
        cash=100000,
        commission=.002,
        exclusive_orders=True,
    )
    
    output = bt.run()
    print(output)
    bt.plot(plot_volume=False)
    
    # Optimize RSI parameters
    stats = bt.optimize(
        rsi_period=range(10, 30, 2),
        rsi_overbought=[65, 70, 75, 80],
        rsi_oversold=[20, 25, 30, 35],
        maximize='Equity Final [$]'
    )
    
    print("\nOptimized parameters:")
    print(stats) 