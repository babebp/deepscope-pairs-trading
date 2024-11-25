from backtest import BacktestParisTrading
from train_model import PairsTradingML


if __name__ == "__main__":
    # Model Training
    # model_training = PairsTradingML(start_date="2014-01-01", end_date="2019-01-01")
    # model_training.train_model()


    # Backtesting
    backtester = BacktestParisTrading()
    result = backtester.backtest(
        model_path="./models/model_1732533092.pkl",
        scaler_path="./scalers/scaler_1732533092.pkl",
        start_date="2019-01-01", 
        end_date="2024-01-01")
