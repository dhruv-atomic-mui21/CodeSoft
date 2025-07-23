from flask import Flask, render_template, request
import joblib
import string

app = Flask(__name__)

# Load TF-IDF vectorizer
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

# Try loading models in order of accessibility
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

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    processed_message = preprocess_text(message)
    vectorized_message = tfidf_vectorizer.transform([processed_message])

    # Use the first accessible model for prediction
    if lr_model is not None:
        prediction = lr_model.predict(vectorized_message)[0]
    elif nb_model is not None:
        prediction = nb_model.predict(vectorized_message)[0]
    elif svm_model is not None:
        prediction = svm_model.predict(vectorized_message)[0]
    else:
        return render_template('result.html', message=message, label="Model not available")

    # Use the most accurate label names
    label = 'Spam' if prediction == 1 else 'Legitimate'

    return render_template('result.html', message=message, label=label)

if __name__ == '__main__':
    app.run(debug=True)
