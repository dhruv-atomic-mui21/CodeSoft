import joblib
import numpy as np

# Load the model and scaler
loaded_model = joblib.load('Machinelearning/creditcardfraudetection/models/random_forest_model.joblib')
loaded_scaler = joblib.load('Machinelearning/creditcardfraudetection/models/scaler.joblib')

# Create a sample data point (replace with actual data structure and number of features)
# Assuming you know the number of features your model expects (it was 9 in the Colab notebook)
sample_data = np.random.randn(1, 9)  # Create a sample with 9 features

# Preprocess the sample data using the loaded scaler
sample_data_scaled = loaded_scaler.transform(sample_data)

# Make a prediction using the loaded model
try:
    prediction = loaded_model.predict(sample_data_scaled)
    print("Prediction successful. Model is fitted.")
    print(f"Prediction: {prediction}")
except Exception as e:
    print(f"Prediction failed: {e}. Model might not be fitted correctly.")
