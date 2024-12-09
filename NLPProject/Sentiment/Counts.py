import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

#Sentiment analysis from https://stackoverflow.com/questions/61608057/output-vader-sentiment-scores-in-columns-based-on-dataframe-rows-of-tweets and chatGPT

nltk.download("vader_lexicon")

def decode_body(body):
    
    if isinstance(body, bytes):
        return body.decode('utf-8', errors='ignore')  
    return str(body)  # Ensure it's a string

def analyze_sentiment_and_print_emails(input_file):
    try:
        df = pd.read_pickle(input_file)

        sia = SentimentIntensityAnalyzer()

        email_sentiments = []

        for index, row in df.iterrows():
            email_body = decode_body(row.get("Body", ""))
            sentiment_scores = sia.polarity_scores(email_body)

            email_sentiments.append((email_body, sentiment_scores["compound"]))

        email_sentiments.sort(key=lambda x: x[1], reverse=True)

        # Print the two most positive emails
        print("Two Most Positive Emails:")
        for email_body, score in email_sentiments[:2]:
            print(f"Score: {score}\nEmail: {email_body[:100]}\n")

        # Print the two most negative emails
        print("Two Most Negative Emails:")
        for email_body, score in email_sentiments[-2:]:
            print(f"Score: {score}\nEmail: {email_body[:100]}\n")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = "../combined_emails.pkl"  # Replace with your .pkl file path
    analyze_sentiment_and_print_emails(input_file)
