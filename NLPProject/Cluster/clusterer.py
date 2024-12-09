import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Error handling and functionized BERT usage from chatGPT

def load_emails(file_path):
    try:
        df = pd.read_pickle(file_path)
        if 'Body' not in df.columns:
            raise ValueError("The dataset does not contain a 'Body' column.")
        return df['Body'].fillna('').astype(str)
    except Exception as e:
        print(f"Error loading emails: {e}")
        return None


def embed_texts(texts, model_name="all-MiniLM-L6-v2"):
    print("Loading BERT model...")
    model = SentenceTransformer(model_name)
    print("Embedding texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def cluster_texts(embeddings, num_clusters=5):
    print("Clustering embeddings...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans


def save_clustered_emails(file_path, emails, labels):
    clustered_emails = pd.DataFrame({
        'Body': emails,
        'Cluster': labels
    })
    clustered_emails.to_pickle(file_path)
    print(f"Clustered emails saved to {file_path}")




def main():
    pkl_file = "../combined_emails.pkl"
    
    emails = load_emails(pkl_file)
    if emails is None:
        return

    # Generate embeddings using BERT
    embeddings = embed_texts(emails.tolist())

    # Cluster the embeddings
    num_clusters = 10  # Adjusted based on how well num works
    labels, kmeans = cluster_texts(embeddings, num_clusters=num_clusters)

    output_file = "clustered_emails.pkl"
    save_clustered_emails(output_file, emails, labels)


if __name__ == "__main__":
    main()
