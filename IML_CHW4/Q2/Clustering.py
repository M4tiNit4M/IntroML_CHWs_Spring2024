#!/usr/bin/env python
# coding: utf-8

# 
# <br>
# <font>
# <!-- <img src="https://cdn.freebiesupply.com/logos/large/2x/sharif-logo-png-transparent.png" alt="SUT logo" width=300 height=300 align=left class="saturate"> -->
# <div dir=ltr align=center>
# <img src="https://cdn.freebiesupply.com/logos/large/2x/sharif-logo-png-transparent.png" width=200 height=200>
# <br>
# <font color=0F5298 size=7>
# Machine Learning <br>
# <font color=2565AE size=5>
# Electrical Engineering Department <br>
# Spring 2024<br>
# <font color=3C99D size=5>
# Practical Assignment 4 <br>
# <font color=696880 size=4>
# <!-- <br> -->
# 
# 
# ____

# # Personal Data

# In[1]:


student_number = '99101943'
first_name = 'Matin'
last_name = 'Alinejad'


# # Introduction

# In this assignment, we will be performing clustering on Spotify songs.

# # Data Preprocessing

# In the next cell, import the libraries you'll need.

# In[27]:


# TODO: Write your code here
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import random
from sklearn.manifold import TSNE


# In the `spotify.csv` file, load the data. Exclude unrelated features and retain only the track name and the features you believe are relevant.

# In[7]:


# TODO: Write your code here

# Load the data
file_path = 'spotify.csv'
spotify_data = pd.read_csv(file_path)

# Retain only the relevant columns
relevant_columns = [
    'track_name', 'track_popularity', 'danceability', 'energy', 'loudness', 
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 
    'tempo', 'duration_ms'
]

spotify_data_relevant = spotify_data[relevant_columns]

# Display the first few rows of the dataset to confirm the changes
spotify_data_relevant.head()


# In this cell, you should implement a standard scalar function from scratch and applying it to your data. Explian importance behind using a standard scalar and the potential complications that could arise in clustering if it's not employed. (you can't use `sklearn.preprocessing.StandardScaler` but you are free to use `sklearn.preprocessing.LabelEncoder`)

# In[8]:


# TODO: Write your code here
# Extract the features for standardization
features = spotify_data_relevant.drop(columns=['track_name'])

# Implementing the standard scaler from scratch
class StandardScalerFromScratch:
    def fit_transform(self, data):
        self.means = data.mean(axis=0)
        self.stds = data.std(axis=0)
        standardized_data = (data - self.means) / self.stds
        return standardized_data

# Initialize the scaler
scaler = StandardScalerFromScratch()

# Standardize the features
standardized_features = scaler.fit_transform(features)

# Create a DataFrame with the standardized features
standardized_features_df = pd.DataFrame(standardized_features, columns=features.columns)

# Display the first few rows of the standardized features to confirm
standardized_features_df.head()


# # Dimensionality Reduction

# One method for dimensionality reduction is Principal Component Analysis (PCA). Use its implementation from the `sklearn` library to reduce the dimensions of your data. Then, by using an appropriate cut-off for the `_explained_variance_ratio_` in the PCA algorithm, determine the number of principal components to retain.

# In[16]:


# TODO: Write your code here

# Apply PCA
pca = PCA()
pca.fit(standardized_features_df)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate the cumulative explained variance
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Plot the cumulative explained variance to determine the cut-off
plt.figure(figsize=(10, 6))
plt.plot(cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Principal Components')
plt.grid(True)
plt.show()

# Find the number of components that explain at least 90% of the variance
num_components = (cumulative_explained_variance >= 0.90).argmax() + 1
print("number of principal components:", num_components)

# Apply PCA with the determined number of components
pca = PCA(n_components = num_components)
principal_components = pca.fit_transform(standardized_features_df)

# Create a DataFrame with the principal components
principal_components_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(num_components)])

# Display the first few rows of the DataFrame to confirm
principal_components_df.head()


# # Clustering

# Implement K-means for clustering from scratch.

# In[21]:


# TODO: Write your code here
import numpy as np

class KMeansFromScratch:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X):
        # Randomly initialize the centroids
        np.random.seed(42)
        random_indices = np.random.permutation(X.shape[0])
        self.centroids = X[random_indices[:self.n_clusters]]
        
        for _ in range(self.max_iter):
            # Assign clusters
            self.labels = self._assign_clusters(X)
            # Calculate new centroids
            new_centroids = self._calculate_centroids(X)
            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break
            self.centroids = new_centroids
    
    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def _calculate_centroids(self, X):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k] = X[self.labels == k].mean(axis=0)
        return centroids
    
    def predict(self, X):
        return self._assign_clusters(X)

# Transform principal components to numpy array
X_pca = principal_components_df.to_numpy()


# Using the function you've created to execute the K-means algorithm eight times on your data, with the number of clusters ranging from 2 to 9. For each run, display the genre of each cluster using the first two principal components in a plot.

# In[22]:


# TODO: Write your code here
import matplotlib.pyplot as plt

def plot_clusters(X, labels, centroids, n_clusters):
    plt.figure(figsize=(10, 6))
    for k in range(n_clusters):
        cluster = X[labels == k]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {k}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'K-means Clustering with {n_clusters} Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()

for n_clusters in range(2, 10):
    kmeans = KMeansFromScratch(n_clusters=n_clusters)
    kmeans.fit(X_pca)
    plot_clusters(X_pca, kmeans.labels, kmeans.centroids, n_clusters)


# The Silhouette score and the Within-Cluster Sum of Squares (WSS) score are two metrics used to assess the quality of your clustering. You can find more information about these two methods [here](https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb). Plot the Silhouette score and the WSS score for varying numbers of clusters, and use these plots to determine the optimal number of clusters (k).

# In[23]:


# TODO: Write your code here

def calculate_wss(X, labels, centroids):
    wss = 0
    for k in range(centroids.shape[0]):
        cluster = X[labels == k]
        wss += np.sum((cluster - centroids[k]) ** 2)
    return wss

silhouette_scores = []
wss_scores = []

for n_clusters in range(2, 10):
    kmeans = KMeansFromScratch(n_clusters=n_clusters)
    kmeans.fit(X_pca)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels))
    wss_scores.append(calculate_wss(X_pca, kmeans.labels, kmeans.centroids))

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(range(2, 10), silhouette_scores, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(2, 10), wss_scores, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WSS Score')
plt.title('WSS Score vs. Number of Clusters')
plt.grid(True)

plt.show()


# # Checking Output

# To see how good was our clustering we will use a sample check and t-SNE method.
# 
# first randomly select two song from every cluster and see how close these two songs are.

# In[26]:


# TODO: Write your code here

# Assuming kmeans is already fitted with the desired number of clusters
kmeans = KMeansFromScratch(n_clusters=5)  # Example with 5 clusters
kmeans.fit(principal_components_df.values)
labels = kmeans.predict(principal_components_df.values)

# Create a DataFrame with the original data and labels
clustered_data = principal_components_df.copy()
clustered_data['label'] = labels
clustered_data['track_name'] = spotify_data_relevant['track_name']

# Randomly select two songs from each cluster
for cluster in range(kmeans.n_clusters):
    cluster_songs = clustered_data[clustered_data['label'] == cluster]
    sample_songs = cluster_songs.sample(n=2, random_state=42)
    print(f"Cluster {cluster}:")
    print(sample_songs[['track_name', 'PC1', 'PC2']])
    print("\n")


# Using t-SNE reduce dimension of data pointe to 2D and plot it to check how good datapoints are clustered (implementing this part is optional and have extra points)

# In[28]:


# TODO: Write your code here

# Apply t-SNE to reduce to 2D
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(principal_components_df.values)

# Create a DataFrame with the t-SNE results and labels
tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
tsne_df['label'] = labels

# Plot the t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df['label'], cmap='viridis', marker='.')
plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar()
plt.show()

