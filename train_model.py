from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from statsmodels.api import OLS
from utils import split_train_test_sequential
import yfinance as yf
import pickle
import numpy as np
import pandas as pd
from datetime import datetime



WINDOW = 21
ROLLING_WINDOW = 21
THRESHOLD = 1.5

class PairsTradingML:

    def __init__(self, start_date="2019-01-01", end_date="2023-01-01", pairs=["GC=F", "SI=F"]):
        self.start_date = start_date
        self.end_date = end_date
        self.pairs = pairs
        self.data = self.__get_historical_data()
        self.hedge_ratio = self.__calculate_hedge_ratio()

        self.__calculate_spread()
        self.__calculate_z_score()
        self.__feature_engineering()


    def __compute_revert_in_7_days(self, z_score_series):
        """
        Computes a Series indicating if the Z-Score reverts to 0 within the next 7 days.

        Parameters:
        z_score_series (pd.Series): The input Series for Z-Scores.

        Returns:
        pd.Series: A Series with 1 if the Z-Score reverts to 0 within the next 7 days, 0 otherwise.
        """
        return (
            z_score_series.rolling(window=14, center=False)
            .apply(lambda x: any(abs(x) <= 0.1), raw=True)
            .shift(-13)
            .fillna(0)
            .astype(int)
        )
    
    def __get_historical_data(self):
        asset_1 = yf.download(self.pairs[0], start=self.start_date, end=self.end_date)
        asset_2 = yf.download(self.pairs[1], start=self.start_date, end=self.end_date)

        # Prepare the data
        data = pd.DataFrame({
            'asset_1': asset_1['Adj Close']['GC=F'],
            'asset_2': asset_2['Adj Close']['SI=F']
        }).dropna()

        return data
    

    def __calculate_hedge_ratio(self):
        model = OLS(self.data['asset_1'], self.data['asset_2']).fit()
        hedge_ratio = np.array(model.params)
        return hedge_ratio
    

    def __calculate_spread(self):
        self.data['Spread'] = self.data['asset_1'] - (self.hedge_ratio * self.data['asset_2'])


    def __calculate_z_score(self):
        mean_spread = self.data['Spread'].rolling(center=False, window=WINDOW).mean()
        std_spread = self.data['Spread'].rolling(center=False, window=WINDOW).std()
        self.data['Z-Score'] = (self.data['Spread'] - mean_spread) / std_spread


    def __feature_engineering(self):
        self.data['Asset_1_Return'] = self.data['asset_1'].pct_change()
        self.data['Asset_2_Return'] = self.data['asset_2'].pct_change()

        # Lag Features
        self.data['Z-Score_Lag'] = self.data['Z-Score'].shift(1)
        self.data['Spread_Lag'] = self.data['Spread'].shift(1)
        self.data['Spread_Std'] = self.data['Spread'].rolling(window=ROLLING_WINDOW).std()
        self.data['Spread_MA'] = self.data['Spread'].rolling(window=ROLLING_WINDOW).mean()
        self.data[f'Z-Score_Above_{THRESHOLD}'] = (self.data['Z-Score'] > THRESHOLD).astype(int)
        self.data[f'Z-Score_Below_-{THRESHOLD}'] = (self.data['Z-Score'] < -THRESHOLD).astype(int)
        self.data['Target'] = self.__compute_revert_in_7_days(self.data['Z-Score'])

        self.data.dropna(inplace=True)

    
    def train_model(self):
        current_timestamp = int(datetime.now().timestamp())

        features = [
            'Asset_1_Return', 'Asset_2_Return', 'Z-Score_Lag', 'Spread_Lag', 'Spread_Std', 'Spread_MA', f'Z-Score_Above_{THRESHOLD}',
            f'Z-Score_Below_-{THRESHOLD}'
        ]

        X = self.data[features]
        y = self.data['Target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        with open(f'./scalers/scaler_{current_timestamp}.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_train_test_sequential(X_scaled, y)

        # Use GridSearchCV to tune hyperparameters for better performance
        param_grid = {
            'gamma': np.arange(0.1, 10, 0.05)
        }

        grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Print the best hyperparameters from GridSearchCV
        print("Best Hyperparameters:", grid_search.best_params_)

        # Train the best model found by GridSearchCV
        best_model = grid_search.best_estimator_

        # Evaluate the model using cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

        # Make predictions and evaluate on the test set
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.2f}")

        # save
        with open(f'./models/model_{current_timestamp}.pkl','wb') as f:
            pickle.dump(best_model,f)
