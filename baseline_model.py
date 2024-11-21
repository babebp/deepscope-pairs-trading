import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.api import OLS
import numpy as np
import statsmodels.api as sm

"""Check Cointegration"""
# Step 1: Download Historical Data
xauusd = yf.download("GC=F", start="2019-01-01", end="2024-01-01")
xagusd = yf.download("SI=F", start="2019-01-01", end="2024-01-01")

# Prepare the data
data = pd.DataFrame({
    'Gold': xauusd['Adj Close']['GC=F'],
    'Silver': xagusd['Adj Close']['SI=F']
}).dropna()

# Step 2: Analyze Relationships
correlation = data.corr()
print(f"Correlation between Gold and Silver: {correlation.loc['Gold', 'Silver']}")

# Cointegration Test
score, p_value, _ = coint(data['Gold'], data['Silver'])
print(f"Cointegration p-value: {p_value}")


"""Calculate Hedge Ratio"""
xauusd = yf.download("GC=F", start="2022-01-01", end="2023-01-01")
xagusd = yf.download("SI=F", start="2022-01-01", end="2023-01-01")

# Prepare the data
data = pd.DataFrame({
    'Gold': xauusd['Adj Close']['GC=F'],
    'Silver': xagusd['Adj Close']['SI=F']
}).dropna()

# Step 3: Calculate the Spread
model = OLS(data['Gold'], data['Silver']).fit()
hedge_ratio = np.array(model.params)


"""Calculate Spread"""
xauusd = yf.download("GC=F", start="2023-01-01", end="2024-01-01")
xagusd = yf.download("SI=F", start="2023-01-01", end="2024-01-01")

# Prepare the data
data = pd.DataFrame({
    'Gold': xauusd['Adj Close']['GC=F'],
    'Silver': xagusd['Adj Close']['SI=F']
}).dropna()

data['Spread'] = data['Gold'] - (hedge_ratio * data['Silver'])

# Standardize the Spread (Z-Score)
WINDOW = 21
mean_spread = data['Spread'].rolling(center=False, window=WINDOW).mean()
std_spread = data['Spread'].rolling(center=False, window=WINDOW).std()
data['Z-Score'] = (data['Spread'] - mean_spread) / std_spread

backtest_df = pd.DataFrame({
    "gold": data['Gold'],
    "silver": data['Silver'],
    "zscore": data['Z-Score'],
    })

backtest_df['positions_Gold_Long'] = 0
backtest_df['positions_Silver_Long'] = 0
backtest_df['positions_Gold_Short'] = 0
backtest_df['positions_Silver_Short'] = 0

# Recall, we are trading the "synthetic pair" Gold/Silver and betting on its mean reversion

backtest_df.loc[backtest_df.zscore >= 1, ('positions_Gold_Short', 'positions_Silver_Short')] = [-1, 1] # Short spread
backtest_df.loc[backtest_df.zscore <= -1, ('positions_Gold_Long', 'positions_Silver_Long')] = [1, -1] # Buy spread
backtest_df.loc[backtest_df.zscore <= 0, ('positions_Gold_Short', 'positions_Silver_Short')] = 0 # Exit short spread
backtest_df.loc[backtest_df.zscore >= 0, ('positions_Gold_Long', 'positions_Silver_Long')] = 0 # Exit long spread

backtest_df.fillna(method='ffill', inplace=True) # ensure existing positions are carried forward unless there is an exit signal

positions_Long = backtest_df[['positions_Gold_Long', 'positions_Silver_Long']]
positions_Short = backtest_df[['positions_Gold_Short', 'positions_Silver_Short']]
positions = np.array(positions_Long) + np.array(positions_Short)
positions = pd.DataFrame(positions, index=positions_Long.index, columns=['gold','silver'])

dailyret = backtest_df[['gold', 'silver']].pct_change() 
pnl = (positions.shift() * dailyret).sum(axis=1)

pnl[1:].sum()*100

# PnL Calculation
# Shift the positions for the daily return calculation (positions taken on previous day)
daily_positions = positions.shift()  # shift positions one day back
daily_positions = daily_positions.fillna(0)  # Ensure there are no NaN values

# Calculate daily returns
dailyret = backtest_df[['gold', 'silver']].pct_change()

# Calculate PnL for the strategy
pnl = (daily_positions * dailyret).sum(axis=1)

# Total profit or loss percentage
total_pnl = pnl[1:].sum() * 100  # Don't include the first NaN value

total_pnl

plt.plot(pnl[1:].cumsum()*100)
plt.show()

threshold = 2
plt.plot(backtest_df.zscore, label="Z-Score")
plt.axhline(threshold, color='red', linestyle='--', label='Sell Threshold')
plt.axhline(-threshold, color='red', linestyle='--', label='Buy Threshold')
plt.axhline(0, color='black', linestyle='-', label='TP Level')
plt.title('Spread with Trading Bands')
plt.legend()
plt.show()

from tabulate import tabulate

# Calculate cumulative returns
cumulative_returns = (1 + pnl).cumprod()

# Metrics calculations
start_value = cumulative_returns.iloc[0]
end_value = cumulative_returns.iloc[-1]
num_years = len(cumulative_returns) / 252  # Assuming 252 trading days per year

# CAGR (Compound Annual Growth Rate)
CAGR = ((end_value / start_value) ** (1 / num_years)) - 1

# Maximum Drawdown
drawdown = cumulative_returns / cumulative_returns.cummax() - 1
max_drawdown = drawdown.min()

# Standard Deviation
std_dev = pnl.std()

# Create a metrics table
metrics = [
    ["CAGR", f"{CAGR:.2%}"],
    ["Max Drawdown", f"{max_drawdown:.2%}"],
    ["Standard Deviation", f"{std_dev:.2%}"]
]

# Print with tabulate
print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="grid"))