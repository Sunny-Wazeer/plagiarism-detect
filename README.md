## Plagiarism Detection System

A machine learning-based plagiarism detection web application built with Flask. This project processes input text and classifies whether it contains plagiarism using a trained text classification pipeline.

---

## Project Overview

This project implements a plagiarism detection pipeline that combines text preprocessing, feature extraction, and machine learning classification to identify plagiarized content in input text. The trained model is integrated into a Flask web app for easy interaction.

---

## Explanation of the Machine Learning Pipeline (From Colab Notebook)

# 1. Data Loading:
   The dataset was loaded into a Pandas DataFrame containing `source_text`, `plagiarized_text`, and `label` columns (label indicates plagiarism).

# 2. Text Preprocessing:
   A custom `TextCleaner` transformer class was created that:
   - Removes punctuation.
   - Converts text to lowercase.
   - Removes English stopwords using NLTK's stopword list.
   
   This step cleans and normalizes the input text to improve model accuracy.

# 3. Feature Extraction:
   The cleaned text is vectorized using `TfidfVectorizer`, converting the text into numerical features representing term importance.

# 4. Model Training:
   Multiple classification models were trained and evaluated on the dataset, including:
   - Support Vector Machine (SVM)
   - Random Forest Classifier
   - Multinomial Naive Bayes

   Accuracy and classification metrics were printed for each.

# 5. Model Selection and Saving:
   The best-performing model pipeline (in this case, Naive Bayes) was serialized and saved as `best_model_pipeline.pkl` using `pickle` for later use in the Flask app.

---

# Features

- Custom text preprocessing tailored for plagiarism detection.
- TF-IDF vectorization for effective text feature extraction.
- Multiple ML models trained and evaluated.
- Flask app for easy web-based input and detection.
- Model persistence and reuse with pickle.

# Dependencies

- Flask
- scikit-learn
- pandas
- nltk
- numpy

All listed in `requirements.txt` for easy installation.

---

## Notes

- The `TextCleaner` class is essential to preprocessing and must be included when loading the model.
- NLTK stopwords data must be downloaded prior to running (`nltk.download('stopwords')`).
- The dataset used for training is not included in the repo due to size/privacy.

---

# License

This project is open source and distributed under the MIT License.
