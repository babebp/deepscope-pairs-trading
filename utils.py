import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test_sequential(X, y, train_size=0.8):
    """
    Splits the data sequentially (first 3/4 for training, last 1/4 for testing).
    
    Parameters:
    X (pd.DataFrame): Features DataFrame.
    y (pd.DataFrame): Target DataFrame.
    train_size (float): Proportion of data to use for training (default 0.75).
    
    Returns:
    X_train, X_test, y_train, y_test: Splitted DataFrames for training and testing.
    """
    # Calculate the split index based on train_size
    split_index = int(len(X) * train_size)
    
    # Split the data
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test
