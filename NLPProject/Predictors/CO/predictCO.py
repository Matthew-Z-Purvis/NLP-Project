import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline


#Code stucture just adapted from predictPlebe.py

def extract_email_features(row):
    sender = row['Sender Email']
   
    
    if not isinstance(sender, str):
        return 0
    
    pattern = r"nhall@usna.edu"
    if re.match(pattern, sender):
        #print(sender)
        return 1
    
    pattern = r"catina@usna.edu"
    if re.match(pattern, sender):
        #print(sender)
        return 1
    
    pattern = r"logazino@usna.edu"
    if re.match(pattern, sender):
        #print(sender)
        return 1
    
    pattern = r"zwatt@usna.edu"
    if re.match(pattern, sender):
        #print(sender)
        return 1
    
    pattern = r"gmcknigh@usna.edu"
    if re.match(pattern, sender):
        #print(sender)
        return 1
    
    pattern = r"mcompton@usna.edu"
    if re.match(pattern, sender):
        #print(sender)
        return 1

    return 0


def create_email_labels(email_df):
    email_df['label'] = email_df.apply(extract_email_features, axis=1)
    return email_df


def balance_dataset(email_df):
    """
    Adjust the dataset so that Action Required make up a quarter of the dataset.
    """
    # Separate plebes (label 1) and non-plebes (label 0)
    plebes = email_df[email_df['label'] == 1]
    non_plebes = email_df[email_df['label'] == 0]
    
    # Calculate the target number of plebes and non-plebes
    target_plebes = len(non_plebes) // 3  # 1/4 of total entries will be plebes
    
    # Randomly sample plebes and non-plebes
    sampled_plebes = plebes.sample(n=min(target_plebes, len(plebes)), random_state=42)
    sampled_non_plebes = non_plebes.sample(n=len(sampled_plebes) * 3, random_state=42)
    
    # Combine and shuffle the dataset
    balanced_df = pd.concat([sampled_plebes, sampled_non_plebes]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced dataset size: {len(balanced_df)} (CO: {len(sampled_plebes)}, Non-CO: {len(sampled_non_plebes)})")
    return balanced_df


import joblib

def train_classification_model(email_df, model_file="CO_classifier.pkl"):
    X = email_df['Body'].fillna('').astype(str)
    y = email_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(
        TfidfVectorizer(analyzer='char', ngram_range=(1, 5), stop_words=None),
        LogisticRegression(max_iter=200, class_weight='balanced')
    )
    model.fit(X_train, y_train)
    joblib.dump((model, X_test, y_test), model_file)
    print(f"Model saved to {model_file}")
    return model


def load_pkl_file(pkl_file):
    try:
        email_df = pd.read_pickle(pkl_file)
        print(f"Loaded {pkl_file} successfully.")
        return email_df
    except Exception as e:
        print(f"Error loading the pickle file: {e}")
        return None



pkl_file = '../../cleaned_combined_emails.pkl'  
email_df = load_pkl_file(pkl_file)

if email_df is not None:
    email_df = create_email_labels(email_df)
    balanced_email_df = balance_dataset(email_df)
    model = train_classification_model(balanced_email_df)
