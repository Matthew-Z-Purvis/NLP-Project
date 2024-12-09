import pypff
import pandas as pd
import re
import os

#Run from Helper directoy
#How to parse from https://stackoverflow.com/questions/69905319/how-to-parse-read-outlook-pst-files-with-python and chatGPT
def extract_emails_from_pst(pst_file):

    # Open the PST file
    pst = pypff.file()
    pst.open(pst_file)

    # Prepare to store email data
    email_data = []

    # Function to parse the sender's email address from headers
    def parse_sender_email(headers):
        if headers:
            match = re.search(r"From:.*<(.+?)>", headers)
            if match:
                return match.group(1).strip()
        return "Unknown Email"

    # Function to parse recipients from headers
    def parse_recipients(headers):
        if headers:
            match = re.search(r"To: (.+)", headers)
            if match:
                return match.group(1).strip()
        return "Unknown"

    # Function to process folders recursively
    def process_folder(folder):
        folder_name = folder.name
        print(f"Processing folder: {folder_name}")
        for item in folder.sub_items:
            if isinstance(item, pypff.message):
                try:
                    # Extract email data
                    subject = item.subject or "No Subject"
                    sender_name = item.sender_name or "Unknown Sender"
                    headers = item.transport_headers
                    sender_email = parse_sender_email(headers)
                    date = item.delivery_time or "Unknown Date"
                    body = item.plain_text_body or item.html_body or "No Body"
                    receiver = parse_recipients(headers)
                    
                    email_data.append({
                        "Subject": subject,
                        "Sender Name": sender_name,
                        "Sender Email": sender_email,
                        "Receiver": receiver,
                        "Date": date,
                        "Body": body,
                        "Folder": folder_name
                    })
                except (OSError, UnicodeError, AttributeError) as e:
                    print(f"Error processing item: {e}")
            elif isinstance(item, pypff.folder):
                process_folder(item)

    # Process root folder
    root = pst.get_root_folder()
    process_folder(root)

    # Convert to DataFrame
    df = pd.DataFrame(email_data)
    return df

def save_dataframe_to_pkl(df, output_file):
   
    df.to_pickle(output_file)
    print(f"DataFrame saved to {output_file}")

def main():
    input_directory = "../pst_files"  # Replace with your directory path containing .pst files
    output_file = "combined_emails.pkl"  # Replace with your desired .pkl file path

    # List to hold all DataFrames
    all_emails = []

    # Loop through all .pst files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".pst"):
            pst_path = os.path.join(input_directory, filename)
            print(f"Extracting emails from {filename}...")
            try:
                email_df = extract_emails_from_pst(pst_path)
                all_emails.append(email_df)
            except Exception as e:
                print(f"Failed to extract emails from {filename}: {e}")

    # Combine all DataFrames
    if all_emails:
        combined_df = pd.concat(all_emails, ignore_index=True)
        save_dataframe_to_pkl(combined_df, output_file)
    else:
        print("No valid email data found.")

if __name__ == "__main__":
    main()
