from backtest import BacktestParisTrading
from train_model import PairsTradingML


if __name__ == "__main__":
    # Model Training
    model_training = PairsTradingML()
    model_training.train_model()


    # Backtesting
    backtester = BacktestParisTrading()
    result = backtester.backtest(
        model_path="./models/model_1732189631.pkl",
        scaler_path="./scalers/scaler_1732189631.pkl")
