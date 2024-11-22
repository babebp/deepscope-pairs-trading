import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.api import OLS
import numpy as np
from tabulate import tabulate
import pickle

ROLLING_WINDOW = 21
THRESHOLD = 1.5
WINDOW = 30

class BacktestParisTrading:

    def __init__(self, pairs=["GC=F", "SI=F"]):
        self.nav = 10000
        self.fee = 0.001 # 0.1 %
        self.pairs = pairs
        self.hedge_ratio = self.__calculate_hedge_ratio()

    
    def get_correlation(self):
        data = self.__get_historical_data("2019-01-01", "2024-01-01")

        # Step 2: Analyze Relationships
        correlation = data.corr()
        print(f"Correlation between {self.pairs[0]} and {self.pairs[1]}: {correlation.loc['asset_1', 'asset_2']}")

        # Cointegration Test
        score, p_value, _ = coint(data['asset_1'], data['asset_2'])
        print(f"Cointegration p-value: {p_value}")


    def __calculate_hedge_ratio(self):
        data = self.__get_historical_data("2022-01-01", "2023-01-01")

        # Step 3: Calculate the Spread
        model = OLS(data['asset_1'], data['asset_2']).fit()
        hedge_ratio = np.array(model.params)

        return hedge_ratio
    

    def __calculate_spread(self, data: pd.DataFrame):
        data['Spread'] = data['asset_1'] - (self.hedge_ratio * data['asset_2'])

    
    def __calculate_z_score(self, data: pd.DataFrame):
        mean_spread = data['Spread'].rolling(center=False, window=WINDOW).mean()
        std_spread = data['Spread'].rolling(center=False, window=WINDOW).std()
        data['Z-Score'] = (data['Spread'] - mean_spread) / std_spread


    def __feature_engineering(self, data: pd.DataFrame):
        # Feature Engineering
        data['Asset_1_Return'] = data['asset_1'].pct_change()
        data['Asset_2_Return'] = data['asset_2'].pct_change()

        # Lag Features
        data['Z-Score_Lag'] = data['Z-Score'].shift(1)
        data['Spread_Lag'] = data['Spread'].shift(1)
        data['Spread_Std'] = data['Spread'].rolling(window=ROLLING_WINDOW).std()
        data['Spread_MA'] = data['Spread'].rolling(window=ROLLING_WINDOW).mean()
        data[f'Z-Score_Above_{THRESHOLD}'] = (data['Z-Score'] > THRESHOLD).astype(int)
        data[f'Z-Score_Below_-{THRESHOLD}'] = (data['Z-Score'] < -THRESHOLD).astype(int)

        data.dropna(inplace=True)


    def __prediction(self, data: pd.DataFrame, model_path: str, scaler_path: str):


        features = [
            'Asset_1_Return', 'Asset_2_Return', 'Z-Score_Lag', 'Spread_Lag', 'Spread_Std', 'Spread_MA', f'Z-Score_Above_{THRESHOLD}',
            f'Z-Score_Below_-{THRESHOLD}'
        ]

        test_data = data.copy()

        test_data = test_data[features]

        test_data.dropna(inplace=True)

        model = pickle.load(open(model_path, 'rb'))

        # Load the scaler from the pickle file
        with open(scaler_path, 'rb') as f:
            loaded_scaler = pickle.load(f)

        # Use the loaded scaler to transform new data
        X_new_scaled = loaded_scaler.transform(test_data)

        data['Prediction'] = model.predict(X_new_scaled)

        
    def backtest(self, model_path: str, scaler_path: str):
        data = self.__get_historical_data("2023-01-01", "2024-01-01")

        self.__calculate_spread(data)
        self.__calculate_z_score(data)
        self.__feature_engineering(data)
        self.__prediction(data, model_path, scaler_path)


        backtest_df = pd.DataFrame({
            "asset_1": data['asset_1'],
            "asset_2": data['asset_2'],
            "zscore": data['Z-Score'],
            "pred": data['Prediction']
            })

        backtest_df['positions_asset_1_Long'] = 0
        backtest_df['positions_asset_2_Long'] = 0
        backtest_df['positions_asset_1_Short'] = 0
        backtest_df['positions_asset_2_Short'] = 0


        # Recall, we are trading the "synthetic pair" asset_1/asset_2 and betting on its mean reversion

        # Entry Logic
        backtest_df.loc[(backtest_df.zscore >= 1) & (backtest_df.pred == 1), ('positions_asset_1_Short', 'positions_asset_2_Short')] = [-1, 1] # Short spread
        backtest_df.loc[(backtest_df.zscore <= -1) & (backtest_df.pred == 1), ('positions_asset_1_Long', 'positions_asset_2_Long')] = [1, -1] # Buy spread

        # TP Logic
        backtest_df.loc[(backtest_df.zscore <= 0), ('positions_asset_1_Short', 'positions_asset_2_Short')] = 0 # Exit short spread
        backtest_df.loc[(backtest_df.zscore >= 0), ('positions_asset_1_Long', 'positions_asset_2_Long')] = 0 # Exit long spread

        # SL Logic
        backtest_df.loc[(backtest_df.zscore >= 2), ('positions_asset_1_Short', 'positions_asset_2_Short')] = 0 # Exit short spread
        backtest_df.loc[(backtest_df.zscore <= -2), ('positions_asset_1_Long', 'positions_asset_2_Long')] = 0 # Exit long spread

        backtest_df.fillna(method='ffill', inplace=True) # ensure existing positions are carried forward unless there is an exit signal

        positions_Long = backtest_df[['positions_asset_1_Long', 'positions_asset_2_Long']]
        positions_Short = backtest_df[['positions_asset_1_Short', 'positions_asset_2_Short']]
        positions = np.array(positions_Long) + np.array(positions_Short)
        positions = pd.DataFrame(positions, index=positions_Long.index, columns=['asset_1','asset_2'])

        dailyret = backtest_df[['asset_1', 'asset_2']].pct_change() 
        pnl = (positions.shift() * dailyret).sum(axis=1)

        self.__calculate_metrics(pnl)



    def __calculate_metrics(self, pnl):

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


        
    def __get_historical_data(self, start, end):
        asset_1 = yf.download(self.pairs[0], start=start, end=end)
        asset_2 = yf.download(self.pairs[1], start=start, end=end)

        # Prepare the data
        data = pd.DataFrame({
            'asset_1': asset_1['Adj Close'][self.pairs[0]],
            'asset_2': asset_2['Adj Close'][self.pairs[1]]
        }).dropna()

        return data

