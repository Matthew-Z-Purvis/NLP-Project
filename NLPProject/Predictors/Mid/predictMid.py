import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

def extract_email_features(sender):
  
    # Check if the email is a string
    sender = str(sender)
    if not isinstance(sender, str):
        print("Invalid")
        return 0  # Return 0 if the email is not a valid string
    
    # Regular expression pattern to match the format 2XXXXX@usna.edu
    pattern = r"m2\w{5}@usna\.edu"
    
    # Return 1 if the email matches the pattern, else return 0
    return 1 if re.match(pattern, sender) else 0

def create_email_labels(email_df):
    email_df['label'] = email_df['Sender Email'].apply(extract_email_features)
    return email_df

import joblib

def train_classification_model(email_df, model_file="mid_email_classifier.pkl"):
    """
    Train a classifier to predict whether an email address follows the specified format
    and save the model to a file.
    """

    # Extract the email addresses as the features (X) and the labels (y)
    X = email_df['Body'].fillna('').astype(str)
    y = email_df['label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a classification pipeline with TF-IDF Vectorizer and Logistic Regression
    model = make_pipeline(
        TfidfVectorizer(analyzer='char', ngram_range=(1, 5), stop_words=None),  # Use character n-grams
        LogisticRegression(max_iter=200)
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model to a file
    joblib.dump((model, X_test, y_test), model_file)
    print(f"Model saved to {model_file}")
    return model



# Load the pickle file containing the email data
def load_pkl_file(pkl_file):
    try:
        email_df = pd.read_pickle(pkl_file)
        print(f"Loaded {pkl_file} successfully.")
        return email_df
    except Exception as e:
        print(f"Error loading the pickle file: {e}")
        return None


pkl_file = '../../cleaned_combined_emails.pkl'  

# Load the DataFrame from the pickle file
email_df = load_pkl_file(pkl_file)

if email_df is not None:
    # Label the emails based on the format
    email_df = create_email_labels(email_df)

    # Train the model and evaluate
    model = train_classification_model(email_df)
