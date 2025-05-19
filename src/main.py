from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
from data_loader import load_yahoo_finance_data

class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

# Define SMA function since we're no longer importing from backtesting.test
def SMA(array, n):
    """Simple Moving Average"""
    return pd.Series(array).rolling(n).mean()

# Load price data using the data_loader module
symbol = "AAPL"  # Apple Inc.
data = load_yahoo_finance_data(symbol, period="59d", interval="5m")

# Run backtest
bt = Backtest(
    data,
    SmaCross,
    cash=100000,
    commission=.002,
    exclusive_orders=True,
)
output = bt.run()
bt.plot(plot_volume=False)

# Optimize strategy parameters
stats = bt.optimize(n1=range(5, 30, 5),
                    n2=range(10, 70, 5),
                    maximize='Equity Final [$]',
                    constraint=lambda param: param.n1 < param.n2)

print(stats)

bt.plot(plot_volume=False)
