import pandas as pd
import joblib
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_numeric'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message'] = df['message'].apply(preprocess_text)
    return df

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"--- {model_name} Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

def main():
    # Load dataset
    dataset_path = 'spam.csv'  # Ensure this file is in the project directory
    df = load_data(dataset_path)

    # Load vectorizer and models
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    nb_model = None
    lr_model = None
    svm_model = None

    try:
        nb_model = joblib.load('models/naive_bayes_model.joblib')
    except Exception as e:
        print(f"Warning: Could not load Naive Bayes model: {e}")

    try:
        lr_model = joblib.load('models/logistic_regression_model.joblib')
    except Exception as e:
        print(f"Warning: Could not load Logistic Regression model: {e}")

    try:
        svm_model = joblib.load('models/svm_model.joblib')
    except Exception as e:
        print(f"Warning: Could not load SVM model: {e}")

    # Prepare test data
    X = tfidf_vectorizer.transform(df['message'])
    y = df['label_numeric']

    # Split data (80% train, 20% test) - only test set used here for evaluation
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate models
    if nb_model is not None:
        evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    if lr_model is not None:
        evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    if svm_model is not None:
        evaluate_model(svm_model, X_test, y_test, "SVM")

if __name__ == "__main__":
    main()
