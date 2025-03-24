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
    # 1. Load combined data
    df = load_combined_data("../../../data/raw/raw_x_y.csv")

    # 2. Clean text
    if "description" not in df.columns:
        raise KeyError("description not found.")
    df["cleaned_text"] = df["description"].astype(str).apply(clean_text)

    # 3. Train/Test-Split
    if "prdtypecode" not in df.columns:
        raise KeyError("prdtypecode not found.")
    X_train, X_test, y_train, y_test = split_dataset(
        df, target_column="prdtypecode")

    # 4. TF-IDF Transformation
    X_train_tfidf, vectorizer = apply_tfidf(X_train["cleaned_text"])

    # 5. Save outputs
    output_dir = "../../../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Preprocessing finished!")
    print(f"TF-IDF Shape: {X_train_tfidf.shape}")
    print(f"Labels: {y_train.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
