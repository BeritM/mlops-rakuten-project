"""
run_preprocessing.py

This script orchestrates the full data preprocessing pipeline for our machine learning project.
It performs the following steps:

1. Data Loading and Combining:
   - Loads feature and label data from two separate CSV files using the function `load_combined_data`.
   - Combines these datasets into a single DataFrame and optionally saves the combined raw data as a CSV.

2. Text Cleaning:
   - Applies a custom text cleaning function `clean_text` to the "description" column.
   - The cleaning function removes special characters (keeping letters, numbers, and spaces), converts text to lowercase,
     tokenizes, lemmatizes, and removes stopwords (from English, French, and custom-defined lists).

3. Data Splitting:
   - Splits the combined DataFrame into training, validation, and test sets using the function `split_dataset`.
   - The splitting is stratified based on the target column "prdtypecode" to preserve class distributions.

4. TF-IDF Vectorization:
   - Loads the cleaned text from the training, validation, and test sets.
   - Applies TF-IDF vectorization with custom parameters (ngram_range, max_df, min_df, max_features) via `apply_tfidf`.
   - The vectorizer is fitted on the training data and then used to transform the validation and test data.
   - Optionally, the TF-IDF matrices are saved as pickle files.

5. Saving Processed Data:
   - Saves the training, validation, and test sets (both features and targets) as CSV files in the processed data directory.

Usage:
    To run the preprocessing pipeline, execute this script from the command line:
    
        python run_preprocessing.py

All steps are executed in sequence, ensuring that the raw data is processed and saved for further modeling tasks.
"""

import sys
import os
import pandas as pd

# Append project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

#from cd4ml.data_processing import load_combined_data, clean_text, split_dataset, apply_tfidf
from step01_combine_xy import load_combined_data
from step02_text_cleaning import clean_text
from step03_split_data import split_dataset
from step04_tfidf_transform import apply_tfidf

def main():
    # Set directory paths for raw and processed data
    raw_dir = "/data/raw"
    #raw_dir = "../../../data/raw"
    proc_dir = "/data/processed"
    #proc_dir = "../../../data/processed"
    os.makedirs(proc_dir, exist_ok=True)
    
    # 1. Load combined data from separate CSV files and combine them
    df = load_combined_data(
        f"{raw_dir}/X_train_update.csv",
        f"{raw_dir}/Y_train_CVw08PX.csv",
        save_path=f"{raw_dir}/raw_x_y.csv"
    )

    # 2. Clean text using the 'description' column
    if "description" not in df.columns:
        raise KeyError("description not found.")
    df["cleaned_text"] = df["description"].astype(str).apply(clean_text)

    # 3. Train/Validate/Test Split
    if "prdtypecode" not in df.columns:
        raise KeyError("prdtypecode not found.")
    X_train, X_validate, X_test, y_train, y_validate, y_test = split_dataset(
        df, target_column="prdtypecode"
    )

    # 4. TF-IDF Transformation on text data using separate datasets
    tfidf_paths = {
        "train": f"{proc_dir}/X_train_tfidf.pkl",
        "validate": f"{proc_dir}/X_validate_tfidf.pkl",
        "test": f"{proc_dir}/X_test_tfidf.pkl",
        "vectorizer": f"{proc_dir}/tfidf_vectorizer.pkl"
    }
    X_train_tfidf, X_validate_tfidf, X_test_tfidf, vectorizer = apply_tfidf(
        X_train["cleaned_text"],
        X_validate["cleaned_text"],
        X_test["cleaned_text"],
        save_paths=tfidf_paths
    )

    # 5. Save outputs as CSV files
    for name, df_item in {
        "X_train": X_train,
        "X_validate": X_validate,
        "X_test": X_test,
        "y_train": y_train,
        "y_validate": y_validate,
        "y_test": y_test
    }.items():
        df_item.to_csv(f"{proc_dir}/{name}.csv", index=False)

    # Save y_train and y_validate as a pickle file
    y_train.to_pickle(f"{proc_dir}/y_train.pkl")
    y_validate.to_pickle(f"{proc_dir}/y_validate.pkl")
    y_test.to_pickle(f"{proc_dir}/y_test.pkl")


    # Summary output
    print("Preprocessing finished!")
    print(f"TF-IDF Shape (Train): {X_train_tfidf.shape}")
    print(f"TF-IDF Shape (Validate): {X_validate_tfidf.shape}")
    print(f"TF-IDF Shape (Test): {X_test_tfidf.shape}")
    print(f"Training Labels: {y_train.shape}")
    print(f"Validation Labels: {y_validate.shape}")
    print(f"Test Labels: {y_test.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
