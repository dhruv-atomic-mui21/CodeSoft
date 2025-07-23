# Customer Churn Prediction

This project predicts customer churn for a subscription-based service using machine learning models. It includes a Flask web application for serving predictions, pre-trained models, and a Jupyter notebook for data exploration, model training, and evaluation.

## Project Structure

```
customer_churn_prediction/
│
├── app/
│   ├── app.py                  # Flask application
│   └── templates/              # Frontend HTML template
│       └── index.html
│
├── models/
│   ├── gradient_boosting_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── tuned_gradient_boosting_model.pkl
│
├── notebooks/
│   └── customer_churn_prediction.ipynb  # Data exploration and model training notebook
│
└── README.md                  # Project overview and instructions
```

## Setup Instructions

1. Clone the repository.

2. Create and activate a Python virtual environment (recommended):

```bash
python -m venv venv
source venv/Scripts/activate   # On Windows
# or
source venv/bin/activate       # On Linux/Mac
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

*(If `requirements.txt` is not present, install manually:)*

```bash
pip install flask pandas scikit-learn joblib
```

## Running the Flask Application

1. Navigate to the `app` directory:

```bash
cd app
```

2. Run the Flask app:

```bash
python app.py
```

3. Open your browser and go to `http://127.0.0.1:5000/` to access the web interface.

## Using the Jupyter Notebook

The notebook `customer_churn_prediction.ipynb` in the `notebooks` folder contains the full data science pipeline including:

- Dataset download and preprocessing
- Model training and evaluation
- Hyperparameter tuning
- Saving trained models

You can open and run the notebook using Jupyter:

```bash
jupyter notebook notebooks/customer_churn_prediction.ipynb
```

## Notes

- The Flask app loads pre-trained models from the `models` directory.
- The frontend template is located in the `app/templates` directory.
- Ensure the model files are present in the `models` folder for the app to work correctly.

## Contact

For any questions or issues, please contact [Your Name] at [your.email@example.com].
