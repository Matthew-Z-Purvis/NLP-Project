import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import nltk

# Ensure NLTK resources are downloaded
nltk.download("vader_lexicon")

def decode_body(body):
   
    if isinstance(body, bytes):
        return body.decode('utf-8', errors='ignore')  # Decode bytes to string, ignoring errors
    return str(body)  # Ensure it's a string

def analyze_sender_sentiments(input_file):
   
    try:
        # Load the DataFrame from the .pkl file
        df = pd.read_pickle(input_file)

        # Initialize the sentiment analyzer
        sia = SentimentIntensityAnalyzer()

        # Ensure the Body column is properly decoded
        df['Body'] = df['Body'].apply(decode_body)

        # Create a column for sentiment classification
        def classify_sentiment(text):
            sentiment_scores = sia.polarity_scores(text)
            if sentiment_scores["compound"] >= 0.55:
                return "Positive"
            elif sentiment_scores["compound"] <= -0.55:
                return "Negative"
            else:
                return "Neutral"

        df['Sentiment'] = df['Body'].apply(classify_sentiment)

        positive_emails = df[df['Sentiment'] == 'Positive']
        negative_emails = df[df['Sentiment'] == 'Negative']

        positive_senders = Counter(positive_emails['Sender Name'])
        negative_senders = Counter(negative_emails['Sender Name'])

        # Print the top 3 senders for Positive emails
        print("Top 3 Senders of Positive Emails:")
        for sender, count in positive_senders.most_common(3):
            print(f"{sender}: {count}")

        # Print the top 3 senders for Negative emails
        print("\nTop 3 Senders of Negative Emails:")
        for sender, count in negative_senders.most_common(3):
            print(f"{sender}: {count}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = "combined_emails.pkl"  # Replace with your .pkl file path
    analyze_sender_sentiments(input_file)

# Output
# Top 3 Senders of Positive Emails:
# Midshipman Officer of the Watch (MOOW): 915
# The New York Times: 720
# BRIGADE Announcements for UNOFFICIAL EMAILS: 627

# Top 3 Senders of Negative Emails:
# The New York Times: 250
# The Washington Post: 231
# PAO: 145