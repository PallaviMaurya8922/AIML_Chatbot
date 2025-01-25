import pandas as pd  # Import pandas for data manipulation
import matplotlib.pyplot as plt  # Import matplotlib for visualization
from sklearn.cluster import KMeans  # Import KMeans for clustering
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sentence_transformers import SentenceTransformer  # Import SentenceTransformer for embeddings

# Load the dataset containing the text data
data = pd.read_csv('C:/Users/user/Desktop/pdfs/merged_files.csv') 

# Load the pre-trained SentenceTransformer model for text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the 'data' column into sentence embeddings (vector representations)
embeddings = model.encode(data['data'].tolist(), show_progress_bar=True)

# Initialize the KMeans algorithm with 4 clusters, random seed, 10 initializations, and 300 max iterations
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)

# Fit the KMeans model to the generated embeddings
kmeans.fit(embeddings)

# Add the cluster labels to the DataFrame as a new column 'Cluster'
data['Cluster'] = kmeans.labels_

# Initialize PCA to reduce the embeddings to 2 dimensions for visualization
pca = PCA(n_components=2)

# Fit and transform the embeddings into 2D space using PCA
reduced_embeddings = pca.fit_transform(embeddings)

# Transform the cluster centroids into the same reduced 2D space
centroids_reduced = pca.transform(kmeans.cluster_centers_)

# Plot the reduced embeddings with cluster coloring
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans.labels_, cmap='viridis', s=5)

# Plot the centroids in red with 'X' markers
plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], marker='X', s=50, c='red', label='Centroids')

# Add a colorbar to indicate cluster membership
plt.colorbar(label='Cluster')

# Set the title and axis labels for the plot
plt.title('SBERT-Based K-Means Clustering with Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Display the legend for the centroids
plt.legend()

# Show the plot
plt.show()

# Save the updated DataFrame with cluster assignments to a new CSV file
data.to_csv('C:/Users/user/Desktop/pdfs/cluster_output.csv', index=False)
