import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(df, y, test_size=0.2, random_state=42):
    """
    Splits dataset into train and test sets.

    Parameters:
    df (pd.DataFrame): feature dataframe
    y (np.array): multi-label target
    """

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test