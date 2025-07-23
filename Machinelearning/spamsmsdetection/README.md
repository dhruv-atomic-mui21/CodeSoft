# Spam SMS Detection

This project implements an AI model to classify SMS messages as spam or legitimate (ham) using machine learning techniques. The project includes a Python Flask web application with a frontend interface to input SMS messages and get spam detection results.

## Project Description

The goal is to build a spam SMS detection system using techniques like TF-IDF vectorization and classifiers such as Naive Bayes, Logistic Regression, and Support Vector Machines (SVM). The models are trained on the SMS Spam Collection Dataset from Kaggle.

## Features

- Preprocessing of SMS messages (lowercasing, punctuation removal)
- Feature extraction using TF-IDF vectorization
- Classification using pre-trained models: Naive Bayes, Logistic Regression, and SVM
- Flask web app with a user-friendly frontend to input SMS messages and view predictions
- Prediction results displayed as "Spam" or "Legitimate" (most accurate label)

## Project Structure

```
.
├── app.py                  # Flask application
├── models/                 # Directory containing saved models and TF-IDF vectorizer
│   ├── naive_bayes_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── svm_model.joblib
│   └── tfidf_vectorizer.joblib
├── templates/              # HTML templates for Flask frontend
│   ├── index.html
│   └── result.html
├── evaluate_models.py      # Script to evaluate model accuracy on test data
└── README.md               # Project documentation
```

## Setup Instructions

1. **Clone the repository** (if applicable) or download the project files.

2. **Install Python dependencies:**

```bash
pip install flask joblib scikit-learn pandas
```

3. **Ensure the `models/` directory contains the saved models and TF-IDF vectorizer files.**

4. **Run the Flask app:**

```bash
python app.py
```

5. **Open your web browser and navigate to:**

```
http://127.0.0.1:5000/
```

6. **Enter an SMS message in the input form and submit to see the spam detection result.**

7. **To evaluate model accuracy on the test dataset, run:**

```bash
python evaluate_models.py
```

## Notes

- The Logistic Regression model is used by default for prediction in the Flask app if available; otherwise, Naive Bayes or SVM models are used.
- The preprocessing in the Flask app matches the preprocessing used during model training.
- The evaluation script reports accuracy, classification report, and confusion matrix for the available models.
- This project is intended for educational purposes and can be extended with additional features or improved models.

## Dataset

The SMS Spam Collection Dataset is sourced from Kaggle: [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## License

This project is open source and free to use.

---
