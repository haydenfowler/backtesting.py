from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, EURUSD

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

bt = Backtest(
    EURUSD,
    SmaCross,
    cash=100000,
    commission=.002,
    exclusive_orders=True,
)
output = bt.run()
bt.plot(plot_volume=False)

# n1 & n2 are class variables, so different combinations can be tested to optimise the strategy.
# In the following code, we specify a range of numbers to test for each, and specify a `constraint` lambda to help
# rule out any invalid scenarios (e.g. SMA1 should always be less than SMA2)
stats = bt.optimize(n1=range(5, 30, 5),
                    n2=range(10, 70, 5),
                    maximize='Equity Final [$]',
                    constraint=lambda param: param.n1 < param.n2)

print(stats)

bt.plot(plot_volume=False)
