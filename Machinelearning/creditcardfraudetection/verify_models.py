import joblib
import numpy as np
import os

def verify_model_fit(model_path, scaler_path, feature_count):
    try:
        loaded_model = joblib.load(model_path)
        loaded_scaler = joblib.load(scaler_path)
        sample_data = np.random.randn(1, feature_count)
        sample_data_scaled = loaded_scaler.transform(sample_data)
        prediction = loaded_model.predict(sample_data_scaled)
        print(f"Prediction successful for model at {model_path}. Model is fitted.")
        print(f"Prediction: {prediction}")
    except Exception as e:
        print(f"Prediction failed for model at {model_path}: {e}. Model might not be fitted correctly.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    scaler_path = os.path.join(models_dir, "scaler.joblib")

    model_files = [
        "decision_tree_model (1).joblib",
        "logistic_regression_model (1).joblib",
        "random_forest_model (1).joblib"
    ]

    feature_count = 9  # Adjust based on your model's expected input features

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        print(f"Verifying model: {model_file}")
        verify_model_fit(model_path, scaler_path, feature_count)
        print("-" * 50)
