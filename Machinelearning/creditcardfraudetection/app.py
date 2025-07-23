from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
decision_tree_model = joblib.load(os.path.join(BASE_DIR, 'models', 'decision_tree_model (1).joblib'))
logistic_regression_model = joblib.load(os.path.join(BASE_DIR, 'models', 'logistic_regression_model (1).joblib'))
random_forest_model = joblib.load(os.path.join(BASE_DIR, 'models', 'random_forest_model (1).joblib'))

# Define expected features (common credit card fraud detection features)
FEATURES = [
    'Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
    'V28'
]

@app.route('/')
def index():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if data is None:
        return jsonify({'error': 'No input data provided'}), 400
    try:
        # Expecting a single key 'features' with comma-separated string
        features_str = data.get('features')
        if not features_str:
            return jsonify({'error': 'No features string provided'}), 400

        # Parse comma-separated string into list of floats
        input_values = [float(x.strip()) for x in features_str.split(',')]
        if len(input_values) != len(FEATURES):
            return jsonify({'error': f'Expected {len(FEATURES)} features but got {len(input_values)}'}), 400

        input_array = np.array(input_values).reshape(1, -1)

        # Predict with each model
        dt_pred = int(decision_tree_model.predict(input_array)[0])
        lr_pred = int(logistic_regression_model.predict(input_array)[0])
        rf_pred = int(random_forest_model.predict(input_array)[0])

        # Return predictions
        return jsonify({
            'decision_tree': dt_pred,
            'logistic_regression': lr_pred,
            'random_forest': rf_pred
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
