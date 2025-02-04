
"""
Market Making Strategy Explanation:

This strategy follows a **market-making** approach where it places both **bid (buy) and ask (sell) orders** based on order book imbalance. The goal is to **capture the bid-ask spread** and profit from short-term price movements. The strategy adjusts its position dynamically based on market conditions.

### **Key Components:**
1. **Order Book Imbalance:**
   - The strategy calculates the imbalance between bid and ask volumes.
   - If there is a strong buy-side imbalance, it **increases buy orders**.
   - If there is a strong sell-side imbalance, it **increases sell orders**.

2. **Dynamic Positioning:**
   - It manages positions dynamically, ensuring that the position size does not exceed a maximum threshold (`max_position`).
   - The **threshold parameter** is used to determine when to place orders based on detected imbalance levels.

3. **Fee Management & Execution:**
   - The model accounts for **trading fees** in position sizing to ensure profitability.
   - Uses a **spread-based execution model**, ensuring that bid and ask orders are placed strategically.

4. **Backtesting Optimization:**
   - The `RandomizedSearchCV` function optimizes **`threshold` and `imbalance`** to maximize the **Sharpe Ratio**.
   - It evaluates **1,000 different configurations**, selecting the best-performing parameter set.

### **Trade Execution Logic:**
1. **Place Orders:**
   - If an order book imbalance is detected, the model places limit orders slightly away from the market price.
   - The bid price is adjusted downward (`new_bid = row[close] - tick_size`) to ensure favorable execution.
   - The ask price is adjusted upward (`new_ask = row[close] + tick_size`).

2. **Position Adjustment:**
   - If an order is filled, it updates the running position.
   - The model **balances risk exposure**, ensuring it does not overtrade or take excessive risks.

3. **Risk Control:**
   - The strategy **limits position sizes** to avoid overexposure.
   - Tracks daily **drawdowns and Sharpe Ratio** to evaluate performance.

### **Performance Metrics Evaluated:**
- **Sharpe Ratio:** Measures risk-adjusted returns.
- **CAGR (Compound Annual Growth Rate):** Shows long-term profitability.
- **Max Drawdown:** Monitors potential loss risk.
- **Equity Curve:** Plots cumulative P&L over time.

The strategy is optimized for **high-frequency trading** environments where fast execution and minimal latency are critical.
"""



import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from numba import njit


# Load historical order book data
df = pd.read_pickle('data_wp')

@njit
def predict_njit(start_equity, threshold, max_position, X, imbalance):
    tick_size = 0.5
    running_qty = 0
    static_equity = start_equity
    fee = 0
    equity = []
    running_qty_ = []
    order_qty = 100
    new_bid = np.nan
    new_ask = np.nan
    high = 1
    low = 2
    close = 3
    
    for row in X:
        if new_bid > row[low]:
            running_qty += order_qty
            static_equity += order_qty / new_bid
            fee += order_qty / new_bid * -0.00025    
        if new_ask < row[high]:
            running_qty -= order_qty
            static_equity -= order_qty / new_ask
            fee += order_qty / new_ask * -0.00025    
        
        equity.append(static_equity * row[close] - running_qty - fee * row[close])
        running_qty_.append(running_qty)
        running_qty_hedged = running_qty + equity[-1]
        
        ind = row[imbalance] - row[close]
        if ind > threshold and running_qty_hedged < max_position:
            new_bid = row[close] - tick_size
        else:
            new_bid = np.nan
        if ind < -threshold and running_qty_hedged > -max_position:
            new_ask = row[close] + tick_size
        else:
            new_ask = np.nan
    
    return equity, running_qty_

class Backtest:
    def __init__(self, equity=None, max_position=None, threshold=None, imbalance=None):
        self.equity = equity
        self.max_position = max_position
        self.threshold = threshold
        self.imbalance = imbalance
        
    def set_params(self, threshold, imbalance):
        self.threshold = threshold
        self.imbalance = imbalance
        return self
        
    def get_params(self, deep=True):
        return { 'equity': self.equity, 'max_position': self.max_position, 'threshold': self.threshold, 'imbalance': self.imbalance }
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        equity, running_qty = predict_njit(self.equity, self.threshold, self.max_position, X, self.imbalance)
        return equity, running_qty
    
    def score(self, X):
        equity, running_qty = self.predict(X)
        returns = pd.Series(equity).pct_change().fillna(0)
        return np.divide(returns.mean(), returns.std())

train = df[(df.index >= '2019-7-1') & (df.index < '2020-1-1')]
valid = df[(df.index >= '2020-1-1') & (df.index < '2021-1-8')]

param_dist = { 'threshold': stats.uniform(0, 100), 'imbalance': np.arange(4, 11) }
search = RandomizedSearchCV(Backtest(1, 10000),
                            cv=[(np.arange(len(train)), np.arange(len(train)))],
                            param_distributions=param_dist,
                            verbose=1,
                            n_iter=1000,
                            n_jobs=8)
search.fit(train.values)

print(search.best_params_)
print(search.best_estimator_.score(train.values))

equity, running_qty = search.best_estimator_.predict(train.values)
equity = pd.Series(equity, index=train.index)
running_qty = pd.Series(running_qty, index=train.index)

equity.plot()
train["close"].plot()
running_qty.plot()

returns = equity.resample('1d').last().pct_change()
bm_returns = train['close'].resample('1d').last().pct_change()
returns_ = returns
sr = np.divide(returns_.mean(), returns_.std()) * np.sqrt(252)

equity_1d = equity.resample('1d').last()
Roll_Max = equity_1d.cummax()
Daily_Drawdown = np.divide(equity_1d, Roll_Max) - 1.0
Max_Daily_Drawdown = Daily_Drawdown.cummin()

period = (equity.index[-1] - equity.index[0]).days

print(pd.Series({
    'Start date': equity.index[0].strftime('%Y-%m-%d'),
    'End date': equity.index[-1].strftime('%Y-%m-%d'),
    'Time period (days)': period,
    'Sharpe Ratio': sr,
    'CAGR': (equity[-1] / equity[0]) ** (365 / period) - 1,
    'Max Daily Drawdown': -Max_Daily_Drawdown.min(),
}))

equity.resample('1d').last().pct_change().hist(bins=20)
