import pandas as pd  # Import pandas for data manipulation
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.cluster import KMeans  # Import KMeans for clustering
from sentence_transformers import SentenceTransformer  # Import SBERT for embeddings

# Load the merged dataset containing text data
data = pd.read_csv('C:/Users/user/Desktop/pdfs/merged_files.csv')

# Initialize the SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Using SBERT model

# Encode the 'data' column into SBERT embeddings
X_sbert = sbert_model.encode(data['data'].tolist(), show_progress_bar=True)

# Initialize an empty list to store Within-Cluster Sum of Squares (WCSS) values
wcss = []

# Loop over a range of cluster numbers (1 to 10) to find the optimal number of clusters
for k in range(1, 11):
    # Initialize KMeans with k clusters, a random seed, 10 initializations, and 100 max iterations
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
    # Fit the KMeans model to the SBERT embeddings
    kmeans.fit(X_sbert)
    # Append the inertia (WCSS value) for the current k to the wcss list
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph to visualize the optimal number of clusters
plt.plot(range(1, 11), wcss, marker='o', color='b')  # Plot WCSS values for each k
plt.title('Elbow Method for Optimal Clusters')  # Title of the plot
plt.xlabel('Number of Clusters (k)')  # X-axis label
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')  # Y-axis label
plt.show()  # Display the plot
