import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the clustered data
data = pd.read_csv('C:/Users/user/Desktop/pdfs/cluster_output.csv')

# Initialize the SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the text into SBERT embeddings
data['sbert_embeddings'] = list(sbert_model.encode(data['data'].tolist(), show_progress_bar=True))

# Function to extract representative sentences based on cluster centroids
def extract_representative_sentences(data, cluster_num, top_n=5):
    cluster_data = data[data['Cluster'] == cluster_num]
    embeddings = np.vstack(cluster_data['sbert_embeddings'])
    
    # Compute the centroid of the cluster
    centroid = np.mean(embeddings, axis=0)
    
    # Calculate cosine similarity between each sentence embedding and the centroid
    similarities = cosine_similarity([centroid], embeddings)[0]
    
    # Get the indices of the top N most similar sentences
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Retrieve the most representative sentences
    top_sentences = cluster_data.iloc[top_indices]['data'].tolist()
    return top_sentences

# Extract topics (representative sentences) for each cluster
cluster_topics = {}

for cluster_num in range(data['Cluster'].nunique()):
    representative_sentences = extract_representative_sentences(data, cluster_num)
    cluster_topics[cluster_num] = representative_sentences

# Display the topics for each cluster
for cluster, sentences in cluster_topics.items():
    print(f"Cluster {cluster}:")
    for i, sentence in enumerate(sentences):
        print(f"  {i+1}. {sentence}")
    print()