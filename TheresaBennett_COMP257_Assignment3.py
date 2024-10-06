from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Olivetti faces dataset
data = fetch_olivetti_faces()
X = data.images  
y = data.target  

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Print out the size of each set to verify the split
print(f"Training set size: {X_train.shape[0]} images")
print(f"Validation set size: {X_val.shape[0]} images")
print(f"Test set size: {X_test.shape[0]} images")

# Initialize the classifier
knn = KNeighborsClassifier(n_neighbors=5)  # Using 5 neighbors for classification

# Use Stratified K-Fold Cross-Validation to evaluate the model on the training data
kf = StratifiedKFold(n_splits=5)

# Perform cross-validation 
scores = cross_val_score(knn, X_train.reshape(len(X_train), -1), y_train, cv=kf)
print(f"Cross-validated scores (5 folds): {scores}")
print(f"Average cross-validated accuracy: {scores.mean():.4f}")

# Fit the KNN model to the training data
knn.fit(X_train.reshape(len(X_train), -1), y_train)

# Evaluate the model
val_score = knn.score(X_val.reshape(len(X_val), -1), y_val)
print(f"Validation set accuracy: {val_score:.4f}")

# Apply PCA 
pca = PCA(n_components=0.95) 
X_reduced = pca.fit_transform(X.reshape(len(X), -1))  

# Determine the optimal number of clusters 
silhouette_scores = []
cluster_range = range(2, 20)  

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_reduced)
    
    # Calculate the average silhouette score
    silhouette_avg = silhouette_score(X_reduced, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the average silhouette score is {silhouette_avg:.4f}")

# Plot the silhouette scores 
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for K-Means Clustering')
plt.grid(True)
plt.show()

# Choose the number of clusters 
optimal_n_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_n_clusters}")

# Fit K-Means 
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
X_reduced_kmeans = kmeans.fit_transform(X_reduced)

# Re-train the classifier 
knn.fit(X_reduced_kmeans, y)

# Evaluate  classifier 
scores_reduced = cross_val_score(knn, X_reduced_kmeans, y, cv=kf)
print(f"Cross-validated scores on reduced dataset: {scores_reduced}")
print(f"Mean accuracy on reduced dataset: {scores_reduced.mean():.4f}")
