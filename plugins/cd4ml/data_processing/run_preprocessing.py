import sys
import os

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../")))


from cd4ml.data_processing import (
    load_combined_data,
    clean_text,
    split_dataset,
    apply_tfidf
)


def main():
    # 1. Load combined data from separate CSV files and combine them
    df = load_combined_data(
        "../../../data/raw/X_train_update.csv",
        "../../../data/raw/Y_train_CVw08PX.csv",
        save_path="../../../data/raw/raw_x_y.csv"
    )

    # 2. Clean text
    if "description" not in df.columns:
        raise KeyError("description not found.")
    df["cleaned_text"] = df["description"].astype(str).apply(clean_text)

    # 3. Train/Validate/Test-Split
    if "prdtypecode" not in df.columns:
        raise KeyError("prdtypecode not found.")
    X_train, X_validate, X_test, y_train, y_validate, y_test = split_dataset(
        df, target_column="prdtypecode"
    )

    # 4. TF-IDF Transformation on training data
    X_train_tfidf, vectorizer = apply_tfidf(X_train["cleaned_text"])

    # 5. Save outputs
    output_dir = "../../../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_validate.to_csv(f"{output_dir}/X_validate.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_validate.to_csv(f"{output_dir}/y_validate.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Preprocessing finished!")
    print(f"TF-IDF Shape: {X_train_tfidf.shape}")
    print(f"Training Labels: {y_train.shape}")
    print(f"Validation Labels: {y_validate.shape}")
    print(f"Test Labels: {y_test.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
