import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import nltk

# Ensure NLTK resources are downloaded
nltk.download("vader_lexicon")

def decode_body(body):
    
    if isinstance(body, bytes):
        return body.decode('utf-8', errors='ignore')  # Decode bytes to string, ignoring errors
    return str(body)  # Ensure it's a string

def analyze_sender_ratios(input_file):
   
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

        # Calculate the total emails and sentiment counts for each sender
        sender_stats = defaultdict(lambda: {"total": 0, "positive": 0, "negative": 0})

        for _, row in df.iterrows():
            sender = row["Sender Name"]
            sentiment = row["Sentiment"]

            sender_stats[sender]["total"] += 1
            if sentiment == "Positive":
                sender_stats[sender]["positive"] += 1
            elif sentiment == "Negative":
                sender_stats[sender]["negative"] += 1

        # Exclude senders with fewer than 5 total emails
        filtered_stats = {
            sender: stats
            for sender, stats in sender_stats.items()
            if stats["total"] >= 5
        }

        # Calculate positive and negative ratios for each sender
        positive_ratios = {
            sender: (stats["positive"] / stats["total"], stats["total"])
            for sender, stats in filtered_stats.items()
        }
        negative_ratios = {
            sender: (stats["negative"] / stats["total"], stats["total"])
            for sender, stats in filtered_stats.items()
        }

        # Sort by ratio first and total emails in case of ties
        top_positive_senders = sorted(
            positive_ratios.items(),
            key=lambda x: (x[1][0], x[1][1]),  # Sort by ratio, then by total
            reverse=True
        )[:3]
        top_negative_senders = sorted(
            negative_ratios.items(),
            key=lambda x: (x[1][0], x[1][1]),  # Sort by ratio, then by total
            reverse=True
        )[:3]

        # Print the results
        print("Top 3 Senders by Positive Email Ratio (with Total Emails):")
        for sender, (ratio, total) in top_positive_senders:
            print(f"{sender}: Ratio={ratio:.2f}, Total Emails={total}")

        print("\nTop 3 Senders by Negative Email Ratio (with Total Emails):")
        for sender, (ratio, total) in top_negative_senders:
            print(f"{sender}: Ratio={ratio:.2f}, Total Emails={total}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = "../combined_emails.pkl"  
    analyze_sender_ratios(input_file)

# Top 3 Senders by Positive Email Ratio (with Total Emails):
# Papa Johns: Ratio=1.00, Total Emails=284
# Elijah Hiler: Ratio=1.00, Total Emails=221 Center of Academic Excellence
# ITSD Support: Ratio=1.00, Total Emails=153

# Top 3 Senders by Negative Email Ratio (with Total Emails):
# Mia Martinez (Google Sheets): Ratio=0.83, Total Emails=6
# Elizabeth Wrightson: Ratio=0.81, Total Emails=26 Chinfo person
# Isaac Thompson: Ratio=0.75, Total Emails=8 Remedial swim pwerson