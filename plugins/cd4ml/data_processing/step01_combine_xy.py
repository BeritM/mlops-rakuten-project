# the module contains function units for combining features and labels into a single DataFrame

import pandas as pd


def load_combined_data(csv_path: str) -> pd.DataFrame:
    """
    Loads feature data (X) and label data (y) from CSV files
    and combines them into a single DataFrame.

    Args:
        x_path (str): Path to the CSV file containing features.
        y_path (str): Path to the CSV file containing labels.

    Returns:
        pd.DataFrame: Combined DataFrame with features and target.
    """
    df = pd.read_csv(csv_path)
    return df