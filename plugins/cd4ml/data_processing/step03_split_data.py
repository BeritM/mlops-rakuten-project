# module for splitting data into training and test sets
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Splits a DataFrame into training and test sets using scikit-learn.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and label.
        target_column (str): Name of the label column.
        test_size (float): Proportion of test data. Defaults to 0.2.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)