import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle
from statsmodels.api import OLS


def compute_revert_in_7_days(z_score_series):
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
        .shift(-6)
        .fillna(0)
        .astype(int)
    )



start_date = "2019-01-01"
end_date = "2023-01-01"

xauusd = yf.download("GC=F", start=start_date, end=end_date)
xagusd = yf.download("SI=F", start=start_date, end=end_date)

# Prepare the data
data = pd.DataFrame({
    'Gold': xauusd['Adj Close']['GC=F'],
    'Silver': xagusd['Adj Close']['SI=F']
}).dropna()

model = OLS(data['Gold'], data['Silver']).fit()
hedge_ratio = np.array(model.params)

data['Spread'] = data['Gold'] - (hedge_ratio * data['Silver'])

# Standardize the Spread (Z-Score)
WINDOW = 21
mean_spread = data['Spread'].rolling(center=False, window=WINDOW).mean()
std_spread = data['Spread'].rolling(center=False, window=WINDOW).std()
data['Z-Score'] = (data['Spread'] - mean_spread) / std_spread

# Feature Engineering
data['Gold_Return'] = data['Gold'].pct_change()
data['Silver_Return'] = data['Silver'].pct_change()

# Lag Features
rolling_window = 21
THRESHOLD = 1.5
data['Z-Score_Lag'] = data['Z-Score'].shift(1)
data['Spread_Lag'] = data['Spread'].shift(1)
data['Spread_Std'] = data['Spread'].rolling(window=rolling_window).std()
data['Spread_MA'] = data['Spread'].rolling(window=rolling_window).mean()
data[f'Z-Score_Above_{THRESHOLD}'] = (data['Z-Score'] > THRESHOLD).astype(int)
data[f'Z-Score_Below_-{THRESHOLD}'] = (data['Z-Score'] < -THRESHOLD).astype(int)

data.dropna(inplace=True)

features = [
    'Gold_Return', 'Silver_Return', 'Z-Score_Lag', 'Spread_Lag', 'Spread_Std', 'Spread_MA', f'Z-Score_Above_{THRESHOLD}',
    f'Z-Score_Below_-{THRESHOLD}'
]

data['Target'] = compute_revert_in_7_days(data['Z-Score'])

X = data[features]
y = data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Use GridSearchCV to tune hyperparameters for better performance
param_grid = {
    'gamma': np.arange(0.1, 10, 0.05)
}

grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters from GridSearchCV
print("Best Hyperparameters:", grid_search.best_params_)

# Train the best model found by GridSearchCV
best_xg_model = grid_search.best_estimator_

# Evaluate the model using cross-validation
cv_scores = cross_val_score(best_xg_model, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

# Make predictions and evaluate on the test set
y_pred = best_xg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# save
with open('model.pkl','wb') as f:
    pickle.dump(best_xg_model,f)