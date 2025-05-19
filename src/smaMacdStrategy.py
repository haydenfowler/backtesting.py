from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
from data_loader import load_yahoo_finance_data

class SmaMacdStrategy(Strategy):
    # Define parameters
    sma_period = 10  # 10-period SMA (for 5-minute chart)
    macd_fast = 12   # MACD default parameters
    macd_slow = 26
    macd_signal = 9
    
    def init(self):
        # Calculate SMA
        self.sma = self.I(SMA, self.data.Close, self.sma_period)
        
        # Calculate MACD
        close = self.data.Close
        macd_line, signal_line, _ = self.I(talib.MACD, close, 
                                          fastperiod=self.macd_fast,
                                          slowperiod=self.macd_slow,
                                          signalperiod=self.macd_signal,
                                          )
        self.macd = macd_line
        self.signal = signal_line

    def next(self):
        # Check for entry conditions
        smaBullish = crossover(self.data.Close, self.sma)
        smaBearish = crossover(self.sma, self.data.Close)

        macd_bullish = crossover(self.macd, self.signal)
        macd_bearish = crossover(self.signal, self.macd)

        # Close positions on SMA crossovers
        if smaBullish and self.position.is_short:
            self.position.close()
        elif smaBearish and self.position.is_long:
            self.position.close()
        
        # Entry logic
        if smaBullish and macd_bullish:
            # enter position
            self.buy(size=0.3)
        elif smaBearish and macd_bearish:
            # enter position
            self.sell(size=0.3)

# Define SMA function since we're no longer importing from backtesting.test
def SMA(array, n):
    """Simple Moving Average"""
    return pd.Series(array).rolling(n).mean()

if __name__ == "__main__":
    # Load data using the data_loader module
    symbol = "AAPL"  # Apple Inc.
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    data = load_yahoo_finance_data(symbol, start_date, end_date)
    
    bt = Backtest(
        data,
        SmaMacdStrategy,
        cash=100000,
        commission=.002,
        exclusive_orders=True,
    )
    output = bt.run()
    bt.plot(
        plot_volume=False,
        plot_pl=True,
        plot_drawdown=False,
        plot_equity=True,
        plot_return=False,
    )
