from model_training import load_train_data, calculate_class_weights, train_final_model
import joblib

def main():
    
    # Set paths for X and y data and output model folder

    input_dir = "../../../data/processed"
    output_dir = "../../../models"
    X_train_tfidf_path = f"{input_dir}/X_train_tfidf.pkl"
    y_train_path = f"{input_dir}/y_train.pkl"

    # 1. Load preprocessed data

    X_train_tfidf, y_train = load_train_data(X_train_tfidf_path, y_train_path)

    print("Data loading successful")

    # 2. Calculate adapted weights

    custom_class_weights = calculate_class_weights(y_train)

    print("Start model training")

    # 3. Train selected model

    model = train_final_model(custom_class_weights, X_train_tfidf, y_train)

    print("Training finished")

    # 4. Save trained model
    
    model_path = f"{output_dir}/sgd_text_model.pkl"
    joblib.dump(model, model_path)

    print(f"Model saved in {model_path}")


if __name__ == "__main__":
    main()