import string
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask, render_template, request
from nltk.corpus import stopwords
import pandas as pd
import joblib 

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text):
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()
        return " ".join(word for word in text.split() if word not in self.stop_words)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        return X.apply(self.clean_text)


# Create Flask app
app = Flask(__name__)

# Load your pipeline (make sure it's saved with joblib)
pipeline = joblib.load('best_model_pipeline.joblib')

def detect(input_text):
    # Wrap the input_text as a list because pipeline expects iterable
    result = pipeline.predict([input_text])
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    detection_result = detect(input_text)
    return render_template('index.html', result=detection_result)

if __name__ == "__main__":
    app.run(debug=True)
