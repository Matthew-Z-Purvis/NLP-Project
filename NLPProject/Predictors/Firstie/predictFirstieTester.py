import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import joblib

#Structure adopted from predictMidTester.py

def evaluate_model(model_file):
    # Load the model, X_test, and y_test
    try:
        model, X_test, y_test = joblib.load(model_file)
        print(f"Model loaded from {model_file}")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Identify false positives and false negatives
    X_test_df = X_test.reset_index(drop=True)  # Reset index for easier indexing
    y_test_df = y_test.reset_index(drop=True)
    y_pred_df = pd.Series(y_pred)
    
    # False positives: Predicted 1 but actual 0
    false_positives = X_test_df[(y_test_df == 0) & (y_pred_df == 1)]
    print("\nFalse Positives:")
    for i, text in enumerate(false_positives[:2]):  # Print 2 instances
        print(f"Instance {i + 1}:")
        print(f"Text: {text}\n")
    
    # False negatives: Predicted 0 but actual 1
    false_negatives = X_test_df[(y_test_df == 1) & (y_pred_df == 0)]
    print("\nFalse Negatives:")
    for i, text in enumerate(false_negatives[:2]):  # Print 2 instances
        print(f"Instance {i + 1}:")
        print(f"Text: {text}\n")

# Load the saved model and evaluate it
evaluate_model(model_file="firstie_classifier.pkl")
