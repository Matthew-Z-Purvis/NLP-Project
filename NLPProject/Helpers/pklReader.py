import pandas as pd

#chatGPT helped debug and implement error handling
#Run from helpers

def load_and_print_top_senders(file_path, top_n=15):
 
    try:
        # Load the DataFrame from the .pkl file
        df = pd.read_pickle(file_path)
        
        # Check if the 'Sender Name' column exists
        if 'Sender Name' not in df.columns:
            raise ValueError("The dataset does not contain a 'Sender Name' column.")
        
        # Find the top N most frequently occurring Sender Names
        top_senders = df['Sender Name'].value_counts().head(top_n)
        
        print(f"Top {top_n} most frequently occurring Sender Names:")
        for sender, count in top_senders.items():
            print(f"{sender}: {count} occurrences")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while loading the .pkl file: {e}")

# Main function
def main():
    # Replace with the path to your .pkl file
    pkl_file = "../cleaned_combined_emails.pkl"
    df = pd.read_pickle(pkl_file)
        
    # Print the DataFrame content
    print(df)
    print(df.columns.tolist())
    print(df['Body'])
    load_and_print_top_senders(pkl_file)
    

if __name__ == "__main__":
    main()
