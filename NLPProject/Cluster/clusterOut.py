import pandas as pd


#save_sampled_clusters from chatGPT
def save_sampled_clusters(input_file, output_file, sample_size=10):
    try:
        # Load the DataFrame from the .pkl file
        df = pd.read_pickle(input_file)
        
        # Check if the clustering column exists
        if "Cluster" not in df.columns:
            raise ValueError("The input file does not contain a 'Cluster' column.")

        with open(output_file, "w") as f:
            # Loop through each cluster and sample rows
            for cluster_id in sorted(df["Cluster"].unique()):
                cluster_data = df[df["Cluster"] == cluster_id]
                sampled_data = cluster_data.sample(n=min(sample_size, len(cluster_data)), random_state=42)

                f.write(f"Cluster {cluster_id}:\n")
                for _, row in sampled_data.iterrows():
                    f.write(f"{row['Body']}\n\n")  # Adjust column name if different
                f.write("\n" + "="*40 + "\n\n")

        print(f"Sampled clusters saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = "clustered_emails.pkl"  
    output_file = "clusterOutput.txt"
    save_sampled_clusters(input_file, output_file)
