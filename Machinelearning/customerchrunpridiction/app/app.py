import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load models
model_files = {
    "gradient_boosting": "../models/gradient_boosting_model.pkl",
    "logistic_regression": "../models/logistic_regression_model.pkl",
    "random_forest": "../models/random_forest_model.pkl",
    "tuned_gradient_boosting": "../models/tuned_gradient_boosting_model.pkl"
}

models = {}
for name, file in model_files.items():
    if os.path.exists(file):
        models[name] = joblib.load(file)
    else:
        print(f"Model file {file} not found.")

# Define categorical columns and numerical columns based on notebook
categorical_cols = [
    'Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts'
]
# Note: The original notebook one-hot encodes Geography and Gender, so these will be handled in preprocessing

# Full list of features after preprocessing (from notebook)
# We will define expected columns after one-hot encoding with drop_first=True
expected_columns = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary',
    'Geography_Germany', 'Geography_Spain',
    'Gender_Male'
]

def preprocess_input(data):
    """
    Preprocess input data dictionary to match model input.
    - Convert to DataFrame
    - One-hot encode Geography and Gender with drop_first=True
    - Ensure all expected columns are present
    """
    df = pd.DataFrame([data])

    # Drop columns not used (RowNumber, CustomerId, Surname) - not expected in input

    # One-hot encode Geography and Gender with drop_first=True
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

    # Add missing columns with 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to expected order
    df = df[expected_columns]

    return df

import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Load models
model_files = {
    "gradient_boosting": "../models/gradient_boosting_model.pkl",
    "logistic_regression": "../models/logistic_regression_model.pkl",
    "random_forest": "../models/random_forest_model.pkl",
    "tuned_gradient_boosting": "../models/tuned_gradient_boosting_model.pkl"
}

models = {}
for name, file in model_files.items():
    if os.path.exists(file):
        models[name] = joblib.load(file)
    else:
        print(f"Model file {file} not found.")

# Define categorical columns and numerical columns based on notebook
categorical_cols = [
    'Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts'
]
# Note: The original notebook one-hot encodes Geography and Gender, so these will be handled in preprocessing

# Full list of features after preprocessing (from notebook)
# We will define expected columns after one-hot encoding with drop_first=True
expected_columns = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary',
    'Geography_Germany', 'Geography_Spain',
    'Gender_Male'
]

def preprocess_input(data):
    """
    Preprocess input data dictionary to match model input.
    - Convert to DataFrame
    - One-hot encode Geography and Gender with drop_first=True
    - Ensure all expected columns are present
    """
    df = pd.DataFrame([data])

    # Drop columns not used (RowNumber, CustomerId, Surname) - not expected in input

    # One-hot encode Geography and Gender with drop_first=True
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

    # Add missing columns with 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to expected order
    df = df[expected_columns]

    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON with customer data.
    Returns predictions and probabilities from all loaded models.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        processed_data = preprocess_input(data)
        results = {}
        for model_name, model in models.items():
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0][1]
            results[model_name] = {
                'prediction': int(prediction),
                'probability': float(prediction_proba)
            }
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
