import pandas as pd
import re

#chatGPT helped debug and implement error handling
#Run from parent directory

def clean_body_text(email_df):
   
    # Check if 'Body' column exists
    if 'Body' not in email_df.columns:
        raise ValueError("The input DataFrame does not contain a 'Body' column.")
    
    # Remove email addresses and numbers from each body text
    def clean_text(text):
        # Remove email addresses
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        return text
    
    email_df['Body'] = email_df['Body'].astype(str).apply(clean_text)
    return email_df

# Load the combined_emails.pkl file
input_file = 'combined_emails.pkl'
output_file = 'cleaned_combined_emails.pkl'

try:
    # Load the DataFrame
    email_df = pd.read_pickle(input_file)
    print(f"Loaded {input_file} successfully.")
    
    # Clean the 'Body' field by removing emails and numbers
    processed_email_df = clean_body_text(email_df)
    
    # Save the processed DataFrame to a new file
    processed_email_df.to_pickle(output_file)
    print(f"Processed data saved to {output_file}.")
except Exception as e:
    print(f"An error occurred: {e}")
